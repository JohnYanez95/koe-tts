"""
GAN Controller for VITS training.

Monitors discriminator/generator health and applies auto-mitigations:
- Alarm A: D overpowering (d_loss too low + g_loss_adv rising)
- Alarm B: G collapse (low RMS, high silence)
- Alarm C: Instability (grad norm spikes, NaN/Inf)

Mitigation ladder:
1. D update throttling (update D every N steps)
2. D LR scaling (temporary reduction)
3. ADV weight ramping (0 → target over K steps after disc start)

STABILITY INVARIANTS (enforced by controller + train loop):
1. No optimizer step runs with non-finite loss (check_loss_finite in train loop)
2. No run continues past N consecutive non-finite steps (nan_inf_max_consecutive)
3. Mitigation cannot return to baseline without sustained stability (_can_decay_level)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AlarmState(Enum):
    """Current alarm state."""
    HEALTHY = "healthy"
    D_DOMINANT = "d_dominant"
    G_COLLAPSE = "g_collapse"
    UNSTABLE = "unstable"
    EMERGENCY = "emergency"  # Requires immediate stop


@dataclass
class GANControllerConfig:
    """Configuration for GAN controller."""

    # Window sizes for rolling metrics
    window_size: int = 200

    # Alarm A: D overpowering thresholds
    d_loss_low_threshold: float = 0.15
    g_loss_adv_high_threshold: float = 8.0
    d_dominant_window: int = 200  # Steps to confirm D dominance

    # Alarm B: G collapse thresholds
    rms_low_threshold: float = 0.002
    silence_high_threshold: float = 40.0  # percent

    # Alarm C: Instability thresholds
    grad_norm_spike_factor: float = 2.0  # p99 * factor = spike threshold
    consecutive_spikes_for_unstable: int = 3  # Require N consecutive spikes to trigger UNSTABLE
    # Spike density: catch "stormy" regimes where spikes have recovery steps between
    spike_density_window: int = 20  # Look back M steps
    spike_density_threshold: int = 3  # N spikes in window triggers UNSTABLE

    # EMA-based early warning: catch gradual escalation before hard ceiling
    ema_elevated_limit_g: float = 500.0  # Trigger if EMA above this...
    ema_elevated_limit_d: float = 500.0  # ...for K steps
    ema_elevated_steps_threshold: int = 50  # K steps of elevated EMA → UNSTABLE

    # Clip coefficient trigger: catch "sawing off the mast" regime
    # Triggers UNSTABLE if clipping too hard for too long
    clip_coef_hard_threshold: float = 0.05  # "Hard clipping" if below this
    clip_coef_hard_steps: int = 30  # Consecutive hard-clip steps → UNSTABLE
    clip_coef_median_threshold: float = 0.1  # Median threshold
    clip_coef_median_window: int = 50  # Window for median calculation

    # D score velocity tracking (replaces threshold-based triggers)
    # Philosophy: D improving (d_real rising) is GOOD, but too fast destabilizes G.
    # We scale LR based on velocity to allow smooth ascent, not sudden jumps.
    d_real_velocity_window: int = 20  # Steps to calculate velocity over
    d_real_velocity_warning: float = 0.01  # Velocity above this → reduce LR slightly
    d_real_velocity_critical: float = 0.02  # Velocity above this → reduce LR more
    d_real_velocity_emergency: float = 0.05  # Velocity above this → strong LR reduction

    # LR scaling based on d_real velocity (multiplicative)
    d_real_velocity_lr_scales: tuple[float, ...] = (0.7, 0.5, 0.25)  # warning, critical, emergency

    # Minimum LR scale (floor to prevent complete stall)
    d_real_velocity_lr_min: float = 0.1

    # D confusion detection: both d_real and d_fake near 0.5 = D can't distinguish
    # This is BAD during training (D collapsed), GOOD at convergence (G perfect)
    d_confusion_threshold: float = 0.15  # |prob - 0.5| < this = confused
    d_confusion_steps: int = 50  # Steps of confusion before alarm
    d_confusion_mel_floor: float = 1.0  # Only alarm if mel_loss above this (not converged)

    # D-freeze can be configured to start earlier than L3 (e.g. L2 for 55k cascade)
    d_freeze_start_level: int = 3

    # After leaving a frozen-D regime, keep LR scaled down briefly to reduce oscillation
    d_unfreeze_warmup_steps: int = 50

    # EMERGENCY: Absolute grad ceiling (hard stop, no recovery)
    # Separate limits for G and D (D can spike differently)
    absolute_grad_limit_g: float = 3000.0  # Generator hard ceiling (was 5000)
    absolute_grad_limit_d: float = 3000.0  # Discriminator hard ceiling (was 5000)
    nan_inf_max_consecutive: int = 3  # Emergency after N consecutive NaN/Inf steps

    # D freeze probe: periodically check if D can still discriminate
    d_freeze_probe_interval: int = 100  # Probe every N steps
    d_freeze_probe_duration: int = 10  # Probe window duration
    d_freeze_probe_forward_only: bool = True  # If True, probes don't update D weights (safer)

    # Mitigation: D throttling
    d_throttle_every: int = 2  # Update D every N steps when throttling
    d_throttle_duration: int = 500  # Steps to maintain throttle

    # Mitigation: D LR scaling
    d_lr_scale_factor: float = 0.5
    d_lr_scale_duration: int = 500

    # Mitigation: ADV weight ramp
    adv_ramp_steps: int = 3000  # Steps to ramp adv_weight 0→1 after disc start

    # Mitigation: Mel weight boost for G collapse
    mel_weight_boost: float = 1.1  # Multiply by this
    mel_boost_duration: int = 500

    # Mitigation: Grad clip tightening (escalating)
    # Escalation ladder: 1.0 → 0.5 → 0.25 → 0.1
    grad_clip_scales: tuple[float, ...] = (0.5, 0.25, 0.1)
    grad_clip_tighten_duration: int = 500  # Per-level duration

    # Mitigation: LR reduction (escalation level 2+)
    lr_scale_factor: float = 0.5  # Reduce LR to 50%
    lr_scale_duration: int = 1000

    # Mitigation: D freeze (escalation level 3+)
    d_freeze_duration: int = 500  # Skip D updates entirely

    # Escalation: fail closed after max attempts
    max_escalation_level: int = 4  # After this, trigger emergency
    escalation_memory_steps: int = 2000  # Remember escalation for this long

    # OBSERVATION MODE: Disable escalation but keep event logging
    # When False: detect spikes/NaN, emit events, but don't escalate or apply mitigations
    # Only hard ceiling (absolute_grad_limit) can still trigger emergency if enabled
    escalation_enabled: bool = True
    hard_ceiling_enabled: bool = True  # If False, even absolute_grad_limit won't emergency

    # P2: Conditional decay requirements
    # Mitigations expire on schedule, but escalation level only decays when stable
    min_dwell_steps: int = 500  # Min steps at a level before decay allowed
    # Exponential stability requirements by level (L1 recovers fast, L3 slow)
    stability_required_steps_l1: int = 200   # Level 1: fast recovery
    stability_required_steps_l2: int = 500   # Level 2: medium recovery
    stability_required_steps_l3: int = 1000  # Level 3: slow recovery
    soft_grad_limit_g: float = 2000.0  # Don't decay level if G grad above this
    soft_grad_limit_d: float = 2000.0  # Don't decay level if D grad above this
    grad_ema_alpha: float = 0.1  # EMA smoothing for grad tracking (0.1 = slow)


@dataclass
class GANControllerState:
    """Current controller state (for logging/checkpointing)."""
    alarm_state: AlarmState = AlarmState.HEALTHY

    # Active mitigations
    d_throttle_active: bool = False
    d_throttle_until: int = 0
    d_throttle_every: int = 1

    d_lr_scale_active: bool = False
    d_lr_scale_until: int = 0
    d_lr_scale: float = 1.0

    mel_boost_active: bool = False
    mel_boost_until: int = 0
    mel_boost: float = 1.0

    grad_clip_tighten_active: bool = False
    grad_clip_tighten_until: int = 0
    grad_clip_scale: float = 1.0

    # P1: LR reduction state
    lr_scale_active: bool = False
    lr_scale_until: int = 0
    lr_scale: float = 1.0  # For both G and D

    # P1: D freeze state
    d_freeze_active: bool = False
    d_freeze_until: int = 0
    d_freeze_probe_active: bool = False  # Currently in probe window
    d_freeze_probe_until: int = 0  # Probe window end
    d_freeze_last_probe: int = 0  # Last probe start step

    # P1: Escalation tracking
    escalation_level: int = 0  # 0=none, 1=clip, 2=clip+lr, 3=clip+lr+d_freeze, 4=emergency
    escalation_last_step: int = 0  # Last step we escalated
    escalation_count: int = 0  # Total escalations this run

    # P2: Conditional decay tracking
    level_entry_step: int = 0  # Step when we entered current escalation level
    stable_steps_at_level: int = 0  # Consecutive stable steps since level entry
    ema_grad_g: float = 0.0  # EMA of G grad norm (for soft limit check)
    ema_grad_d: float = 0.0  # EMA of D grad norm (for soft limit check)

    # D-score velocity tracking
    last_d_real_score: Optional[float] = None
    last_d_fake_score: Optional[float] = None
    d_real_velocity: float = 0.0  # Current velocity (change per step)
    d_real_velocity_lr_scale: float = 1.0  # LR scale from velocity

    # D confusion tracking (both near 0.5 = can't distinguish)
    d_confusion_steps: int = 0  # Consecutive steps where both are confused
    d_confusion_active: bool = False  # Currently in confusion state

    # Post-freeze warmup (keep LR scaled down briefly after unfreeze)
    d_unfreeze_warmup_until: int = 0

    # Counters
    d_steps_skipped: int = 0
    alarms_triggered: int = 0

    # Emergency state
    emergency_stop: bool = False
    emergency_reason: str = ""
    last_alarm_reason: str = ""  # Reason for most recent alarm trigger
    consecutive_nan_inf: int = 0

    # Spike tracking (for consecutive spike requirement)
    consecutive_spikes: int = 0

    # EMA elevated tracking (for early warning)
    ema_elevated_steps_g: int = 0  # Consecutive steps with ema_grad_g above limit
    ema_elevated_steps_d: int = 0  # Consecutive steps with ema_grad_d above limit

    # Clip coefficient tracking (for "sawing off the mast" detection)
    hard_clip_steps_g: int = 0  # Consecutive steps with g_clip_coef < hard threshold
    hard_clip_steps_d: int = 0  # Consecutive steps with d_clip_coef < hard threshold


class GANController:
    """
    GAN training controller with automatic health monitoring and mitigations.

    Usage:
        controller = GANController(config, disc_start_step=10000)

        for step in range(max_steps):
            # Check if D should be updated this step
            if controller.should_update_d(step):
                # ... D update ...

            # Get current scales/weights
            lr_d_scale = controller.get_d_lr_scale(step)
            adv_weight_scale = controller.get_adv_weight_scale(step)
            mel_weight_scale = controller.get_mel_weight_scale(step)

            # ... G update ...

            # Record metrics (call every step)
            controller.record_step(
                step=step,
                d_loss=d_loss,
                g_loss_adv=g_loss_adv,
                grad_norm_g=grad_norm_g,
                grad_norm_d=grad_norm_d,
            )
    """

    def __init__(
        self,
        config: Optional[GANControllerConfig] = None,
        disc_start_step: int = 0,
    ):
        self.config = config or GANControllerConfig()
        self.disc_start_step = disc_start_step
        self.state = GANControllerState()

        # Rolling windows for metrics
        self._d_loss_window: deque[float] = deque(maxlen=self.config.window_size)
        self._g_loss_adv_window: deque[float] = deque(maxlen=self.config.window_size)
        self._grad_norm_g_window: deque[float] = deque(maxlen=self.config.window_size)
        self._grad_norm_d_window: deque[float] = deque(maxlen=self.config.window_size)

        # Spike density tracking: 1 = spike, 0 = no spike for last M steps
        self._spike_history: deque[int] = deque(maxlen=self.config.spike_density_window)

        # Clip coefficient history for median calculation
        self._clip_coef_g_history: deque[float] = deque(maxlen=self.config.clip_coef_median_window)
        self._clip_coef_d_history: deque[float] = deque(maxlen=self.config.clip_coef_median_window)

        # D-real score history for velocity calculation
        self._d_real_history: deque[float] = deque(maxlen=self.config.d_real_velocity_window)

        # Track consecutive low d_loss steps
        self._d_low_count = 0

        # Track step within D throttle cycle
        self._d_throttle_counter = 0

    def record_step(
        self,
        step: int,
        d_loss: float,
        g_loss_adv: float,
        grad_norm_g: Optional[float],
        grad_norm_d: Optional[float],
        pred_rms: Optional[float] = None,
        pred_silence_pct: Optional[float] = None,
        g_step_skipped: bool = False,
        d_step_skipped: bool = False,
        g_clip_coef: Optional[float] = None,
        d_clip_coef: Optional[float] = None,
        d_real_score: Optional[float] = None,
        d_fake_score: Optional[float] = None,
        mel_loss: Optional[float] = None,
    ) -> dict:
        """
        Record metrics for this step and check alarms.

        Args:
            grad_norm_g: Generator grad norm (None = not computed or invalid)
            grad_norm_d: Discriminator grad norm (None = not computed or invalid)
            g_step_skipped: Explicit signal that G optimizer step was skipped (inf/nan)
            d_step_skipped: Explicit signal that D optimizer step was skipped (inf/nan)
            g_clip_coef: Generator clip coefficient (1.0 = no clip, <1 = clipped, 0 = skipped)
            d_clip_coef: Discriminator clip coefficient
            mel_loss: Current mel reconstruction loss (for convergence check)

        Returns dict with controller decisions for logging.
        """
        # Skip if GAN not active yet
        if step < self.disc_start_step:
            return self._get_decision_dict(step, skipped_reason="pre_disc_start")

        # Check for NaN/Inf in loss values
        nan_inf = any(
            math.isnan(v) or math.isinf(v)
            for v in [d_loss, g_loss_adv]
        )

        # Use explicit skip signals from training loop (preferred over inferring from None)
        # This correctly distinguishes "step skipped due to inf/nan" from "grad not computed"
        if g_step_skipped:
            nan_inf = True
        elif grad_norm_g is not None and (math.isnan(grad_norm_g) or math.isinf(grad_norm_g)):
            nan_inf = True

        if d_step_skipped:
            nan_inf = True
        elif grad_norm_d is not None and (math.isnan(grad_norm_d) or math.isinf(grad_norm_d)):
            nan_inf = True

        if nan_inf:
            self.state.consecutive_nan_inf += 1
            # Check for emergency (too many consecutive NaN/Inf)
            # NOTE: This ALWAYS triggers emergency regardless of escalation_enabled
            # because consecutive NaN/Inf corrupts weights - observation mode is for
            # spikes/clipping, not for letting infinities take hold
            if self.state.consecutive_nan_inf >= self.config.nan_inf_max_consecutive:
                self._trigger_emergency(step, f"nan_inf_consecutive_{self.state.consecutive_nan_inf}")
                return self._get_decision_dict(step, nan_inf_detected=True)
            self._trigger_alarm(AlarmState.UNSTABLE, step, "nan_inf_detected")
            return self._get_decision_dict(step, nan_inf_detected=True)
        else:
            # Reset consecutive counter on valid step
            self.state.consecutive_nan_inf = 0

        # P0: Absolute grad ceiling check (hard stop, no recovery possible)
        # Separate limits for G and D since they can spike differently
        # Can be disabled via hard_ceiling_enabled for observation-only mode
        if self.config.hard_ceiling_enabled:
            if grad_norm_g is not None and grad_norm_g > self.config.absolute_grad_limit_g:
                self._trigger_emergency(step, f"grad_g_exceeded_{grad_norm_g:.0f}_limit_{self.config.absolute_grad_limit_g:.0f}")
                return self._get_decision_dict(step)
            if grad_norm_d is not None and grad_norm_d > self.config.absolute_grad_limit_d:
                self._trigger_emergency(step, f"grad_d_exceeded_{grad_norm_d:.0f}_limit_{self.config.absolute_grad_limit_d:.0f}")
                return self._get_decision_dict(step)


        # Track d_real velocity and adjust LR scaling accordingly.
        # This doesn't trigger alarms - D improving is good, we just slow it if too fast.
        self._check_d_real_velocity(
            d_real_score=d_real_score,
            d_fake_score=d_fake_score,
            step=step,
        )

        # Check for D confusion (both d_real and d_fake near 0.5)
        self._check_d_confusion(mel_loss=mel_loss, step=step)

        # Check alarms BEFORE updating windows (so current value doesn't affect thresholds)
        # Alarm C: Instability (grad spikes) - check first
        if self._check_instability(grad_norm_g, grad_norm_d, step):
            # Add to windows after check (for future threshold calculation)
            self._d_loss_window.append(d_loss)
            self._g_loss_adv_window.append(g_loss_adv)
            if grad_norm_g is not None:
                self._grad_norm_g_window.append(grad_norm_g)
            if grad_norm_d is not None:
                self._grad_norm_d_window.append(grad_norm_d)
            return self._get_decision_dict(step)

        # Update windows (only add valid values to grad norm windows)
        self._d_loss_window.append(d_loss)
        self._g_loss_adv_window.append(g_loss_adv)
        if grad_norm_g is not None:
            self._grad_norm_g_window.append(grad_norm_g)
        if grad_norm_d is not None:
            self._grad_norm_d_window.append(grad_norm_d)

        # Check alarms (in priority order)

        # Alarm A: D overpowering
        if self._check_d_dominant(d_loss, g_loss_adv, step):
            return self._get_decision_dict(step)

        # Alarm B: G collapse (if RMS provided)
        if pred_rms is not None and pred_silence_pct is not None:
            if self._check_g_collapse(pred_rms, pred_silence_pct, step):
                return self._get_decision_dict(step)

        # P2: Update EMA grad tracking (always, for accurate decay decisions)
        if grad_norm_g is not None:
            if self.state.ema_grad_g == 0.0:
                self.state.ema_grad_g = grad_norm_g  # Initialize on first valid
            else:
                alpha = self.config.grad_ema_alpha
                self.state.ema_grad_g = alpha * grad_norm_g + (1 - alpha) * self.state.ema_grad_g
        if grad_norm_d is not None:
            if self.state.ema_grad_d == 0.0:
                self.state.ema_grad_d = grad_norm_d
            else:
                alpha = self.config.grad_ema_alpha
                self.state.ema_grad_d = alpha * grad_norm_d + (1 - alpha) * self.state.ema_grad_d

        # EMA-based early warning: detect gradual escalation before hard ceiling
        # This catches "living dangerously" regimes where EMA climbs but no single spike
        g_ema_elevated = self.state.ema_grad_g > self.config.ema_elevated_limit_g
        d_ema_elevated = self.state.ema_grad_d > self.config.ema_elevated_limit_d

        if g_ema_elevated:
            self.state.ema_elevated_steps_g += 1
            if self.state.ema_elevated_steps_g >= self.config.ema_elevated_steps_threshold:
                self._trigger_alarm(AlarmState.UNSTABLE, step, f"ema_g_elevated_{self.state.ema_grad_g:.0f}")
                self.state.ema_elevated_steps_g = 0  # Reset after triggering
                return self._get_decision_dict(step)
        else:
            self.state.ema_elevated_steps_g = 0

        if d_ema_elevated:
            self.state.ema_elevated_steps_d += 1
            if self.state.ema_elevated_steps_d >= self.config.ema_elevated_steps_threshold:
                self._trigger_alarm(AlarmState.UNSTABLE, step, f"ema_d_elevated_{self.state.ema_grad_d:.0f}")
                self.state.ema_elevated_steps_d = 0  # Reset after triggering
                return self._get_decision_dict(step)
        else:
            self.state.ema_elevated_steps_d = 0

        # Clip coefficient trigger: catch "sawing off the mast" regime
        # Two conditions (OR): consecutive hard-clip OR median below threshold
        #
        # IMPORTANT: Skip this check when grad_clip_scale < 1.0 (i.e., when we've
        # intentionally tightened clipping as a mitigation). Low clip_coef is
        # *expected* when clipping is tightened — triggering on it creates a
        # self-reinforcing escalation loop. See postmortem:
        # docs/postmortems/2026-01-30_multi_vits_gan_20260129_200158_clip_coef_feedback_loop.md
        clip_check_enabled = self.state.grad_clip_scale >= 1.0

        if g_clip_coef is not None and not g_step_skipped:
            self._clip_coef_g_history.append(g_clip_coef)

            # Condition 1: Consecutive hard-clipping (g_clip_coef < 0.05 for 30 steps)
            # Only check when we haven't intentionally tightened clipping
            if clip_check_enabled:
                if g_clip_coef < self.config.clip_coef_hard_threshold:
                    self.state.hard_clip_steps_g += 1
                    if self.state.hard_clip_steps_g >= self.config.clip_coef_hard_steps:
                        self._trigger_alarm(AlarmState.UNSTABLE, step, f"hard_clip_g_{self.state.hard_clip_steps_g}_steps")
                        self.state.hard_clip_steps_g = 0
                        return self._get_decision_dict(step)
                else:
                    self.state.hard_clip_steps_g = 0
            else:
                # Reset counter when clip check disabled (don't accumulate during mitigation)
                self.state.hard_clip_steps_g = 0

            # Condition 2: Median clip coef too low (median < 0.1 over 50 steps)
            # Also skip when clipping is intentionally tightened
            if clip_check_enabled and len(self._clip_coef_g_history) >= self.config.clip_coef_median_window:
                median_clip = sorted(self._clip_coef_g_history)[len(self._clip_coef_g_history) // 2]
                if median_clip < self.config.clip_coef_median_threshold:
                    self._trigger_alarm(AlarmState.UNSTABLE, step, f"median_clip_g_{median_clip:.3f}")
                    self._clip_coef_g_history.clear()
                    return self._get_decision_dict(step)

        if d_clip_coef is not None and not d_step_skipped:
            self._clip_coef_d_history.append(d_clip_coef)

            if clip_check_enabled:
                if d_clip_coef < self.config.clip_coef_hard_threshold:
                    self.state.hard_clip_steps_d += 1
                    if self.state.hard_clip_steps_d >= self.config.clip_coef_hard_steps:
                        self._trigger_alarm(AlarmState.UNSTABLE, step, f"hard_clip_d_{self.state.hard_clip_steps_d}_steps")
                        self.state.hard_clip_steps_d = 0
                        return self._get_decision_dict(step)
                else:
                    self.state.hard_clip_steps_d = 0
            else:
                self.state.hard_clip_steps_d = 0

            if clip_check_enabled and len(self._clip_coef_d_history) >= self.config.clip_coef_median_window:
                median_clip = sorted(self._clip_coef_d_history)[len(self._clip_coef_d_history) // 2]
                if median_clip < self.config.clip_coef_median_threshold:
                    self._trigger_alarm(AlarmState.UNSTABLE, step, f"median_clip_d_{median_clip:.3f}")
                    self._clip_coef_d_history.clear()
                    return self._get_decision_dict(step)

        # P2: Track stability for conditional decay
        # Accumulate if grads present and below soft limits (regardless of alarm state)
        # This fixes the circular dependency: can prove stability while UNSTABLE
        # Reset only happens on new alarm triggers (in _trigger_alarm)
        grads_present = grad_norm_g is not None and grad_norm_d is not None
        grads_below_soft = (
            grads_present
            and grad_norm_g <= self.config.soft_grad_limit_g
            and grad_norm_d <= self.config.soft_grad_limit_d
        )
        if grads_below_soft:
            self.state.stable_steps_at_level += 1
        else:
            # Reset on elevated grads (actual instability signal)
            self.state.stable_steps_at_level = 0

        # Decay mitigations if active and expired
        self._decay_mitigations(step)

        # Clear alarm if healthy for a while
        if len(self._d_loss_window) >= self.config.window_size // 2:
            if self.state.alarm_state != AlarmState.HEALTHY:
                if not self._any_mitigation_active():
                    self.state.alarm_state = AlarmState.HEALTHY
                    self.state.last_alarm_reason = ""  # Clear reason on recovery

        return self._get_decision_dict(step)

    def _check_d_dominant(self, d_loss: float, g_loss_adv: float, step: int) -> bool:
        """Check Alarm A: D overpowering."""
        if d_loss < self.config.d_loss_low_threshold:
            self._d_low_count += 1
        else:
            self._d_low_count = max(0, self._d_low_count - 1)

        # Need sustained low d_loss AND high g_loss_adv
        if (self._d_low_count >= self.config.d_dominant_window and
            g_loss_adv > self.config.g_loss_adv_high_threshold):
            self._trigger_alarm(AlarmState.D_DOMINANT, step, "d_dominant")
            return True

        return False

    def _check_g_collapse(self, pred_rms: float, pred_silence_pct: float, step: int) -> bool:
        """Check Alarm B: G collapse."""
        if (pred_rms < self.config.rms_low_threshold or
            pred_silence_pct > self.config.silence_high_threshold):
            self._trigger_alarm(AlarmState.G_COLLAPSE, step, "g_collapse")
            return True
        return False

    def _check_instability(
        self,
        grad_norm_g: Optional[float],
        grad_norm_d: Optional[float],
        step: int,
    ) -> bool:
        """Check Alarm C: Instability (grad norm spikes).

        Uses two detection methods:
        1. Consecutive spikes: N spikes in a row (original)
        2. Spike density: N spikes within M steps (catches bursty patterns)
        """
        # Need enough valid samples in windows
        if len(self._grad_norm_g_window) < 50 or len(self._grad_norm_d_window) < 50:
            self._spike_history.append(0)  # Track even during warmup
            return False  # Not enough data

        # Skip check if current values are None (already handled as invalid)
        if grad_norm_g is None and grad_norm_d is None:
            self._spike_history.append(0)
            return False

        # Compute p99 thresholds from valid values only
        g_sorted = sorted(self._grad_norm_g_window)
        d_sorted = sorted(self._grad_norm_d_window)
        p99_idx_g = int(len(g_sorted) * 0.99)
        p99_idx_d = int(len(d_sorted) * 0.99)

        g_p99 = g_sorted[min(p99_idx_g, len(g_sorted) - 1)]
        d_p99 = d_sorted[min(p99_idx_d, len(d_sorted) - 1)]

        g_threshold = g_p99 * self.config.grad_norm_spike_factor
        d_threshold = d_p99 * self.config.grad_norm_spike_factor

        # Check for spikes (only if we have valid current values)
        g_spike = grad_norm_g is not None and grad_norm_g > g_threshold
        d_spike = grad_norm_d is not None and grad_norm_d > d_threshold
        is_spike = g_spike or d_spike

        # Track spike in rolling window for density check
        self._spike_history.append(1 if is_spike else 0)

        if is_spike:
            self.state.consecutive_spikes += 1

            # Method 1: Consecutive spikes (original trigger)
            if self.state.consecutive_spikes >= self.config.consecutive_spikes_for_unstable:
                self._trigger_alarm(AlarmState.UNSTABLE, step, "grad_spike_consecutive")
                self.state.consecutive_spikes = 0  # Reset after triggering
                return True

            # Method 2: Spike density (catches bursty patterns with recovery steps)
            spike_count = sum(self._spike_history)
            if spike_count >= self.config.spike_density_threshold:
                self._trigger_alarm(AlarmState.UNSTABLE, step, f"grad_spike_density_{spike_count}_in_{len(self._spike_history)}")
                self._spike_history.clear()  # Reset window after triggering
                self.state.consecutive_spikes = 0
                return True
        else:
            # Reset consecutive spike counter on non-spike step
            self.state.consecutive_spikes = 0

        return False

    def _trigger_emergency(self, step: int, reason: str) -> None:
        """Trigger emergency stop - training must halt immediately."""
        self.state.alarm_state = AlarmState.EMERGENCY
        self.state.emergency_stop = True
        self.state.emergency_reason = reason
        self.state.alarms_triggered += 1

    def _trigger_unstable_to_level(self, step: int, target_level: int, reason: str) -> None:
        """Trigger UNSTABLE and jump escalation directly to target_level.

        This addresses cascades where one-step escalation can't catch up.
        """
        self.state.alarm_state = AlarmState.UNSTABLE
        self.state.alarms_triggered += 1
        self.state.last_alarm_reason = reason

        # Reset stability counter on any alarm
        self.state.stable_steps_at_level = 0

        old_level = self.state.escalation_level
        new_level = max(old_level, int(target_level))

        # Refresh escalation memory even if level unchanged
        self.state.escalation_last_step = step

        if new_level != old_level:
            self.state.escalation_level = min(new_level, self.config.max_escalation_level)
            self.state.escalation_count += 1
            self.state.level_entry_step = step
            self.state.stable_steps_at_level = 0

            if self.state.escalation_level >= self.config.max_escalation_level:
                self._trigger_emergency(step, f"max_escalation_level_{self.state.escalation_level}")
                return

            self._apply_mitigations_for_level(step)

    def _check_d_real_velocity(
        self,
        d_real_score: Optional[float],
        d_fake_score: Optional[float],
        step: int,
    ) -> None:
        """
        Track d_real velocity and adjust LR scaling accordingly.

        Philosophy: D improving (d_real rising toward positive) is GOOD - it means
        D is learning to distinguish real from fake. However, if D improves too
        fast, G can't keep up and gradients explode.

        Instead of freezing D or treating positive d_real as "bad", we scale LR
        based on velocity to allow smooth ascent.
        """
        cfg = self.config
        st = self.state

        # Record scores for logging
        if d_fake_score is not None and math.isfinite(d_fake_score):
            st.last_d_fake_score = float(d_fake_score)

        # Skip velocity calculation during adv_ramp - D still learning basics
        adv_ramp_end = self.disc_start_step + cfg.adv_ramp_steps
        if step < adv_ramp_end:
            if d_real_score is not None and math.isfinite(d_real_score):
                st.last_d_real_score = float(d_real_score)
                self._d_real_history.append(d_real_score)
            return

        # Missing / NaN: skip velocity update
        if d_real_score is None or not math.isfinite(d_real_score):
            return

        # Record current score
        st.last_d_real_score = float(d_real_score)
        self._d_real_history.append(d_real_score)

        # Need enough history to calculate velocity
        if len(self._d_real_history) < cfg.d_real_velocity_window:
            st.d_real_velocity = 0.0
            st.d_real_velocity_lr_scale = 1.0
            return

        # Calculate velocity: (newest - oldest) / window_size
        oldest = self._d_real_history[0]
        newest = self._d_real_history[-1]
        velocity = (newest - oldest) / len(self._d_real_history)
        st.d_real_velocity = velocity

        # Determine LR scale based on velocity (only penalize fast RISING)
        # Falling or stable d_real is fine
        if velocity <= 0:
            st.d_real_velocity_lr_scale = 1.0
        elif velocity < cfg.d_real_velocity_warning:
            st.d_real_velocity_lr_scale = 1.0
        elif velocity < cfg.d_real_velocity_critical:
            st.d_real_velocity_lr_scale = cfg.d_real_velocity_lr_scales[0]
        elif velocity < cfg.d_real_velocity_emergency:
            st.d_real_velocity_lr_scale = cfg.d_real_velocity_lr_scales[1]
        else:
            st.d_real_velocity_lr_scale = cfg.d_real_velocity_lr_scales[2]

        # Apply floor
        st.d_real_velocity_lr_scale = max(
            st.d_real_velocity_lr_scale,
            cfg.d_real_velocity_lr_min
        )

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid function for normalizing scores to (0, 1)."""
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            # Numerically stable for negative x
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)

    def get_d_real_normalized(self) -> Optional[float]:
        """
        Get d_real as normalized probability in (0, 1).

        Interpretation:
        - 0.5 = D uncertain (d_real ≈ 0)
        - >0.5 = D thinks it's real (d_real > 0, GOOD)
        - <0.5 = D thinks it's fake (d_real < 0)

        Higher is better - means D correctly identifies real samples.
        """
        if self.state.last_d_real_score is None:
            return None
        return self.sigmoid(self.state.last_d_real_score)

    def get_d_fake_normalized(self) -> Optional[float]:
        """
        Get d_fake as normalized probability in (0, 1).

        Interpretation:
        - 0.5 = D uncertain (d_fake ≈ 0)
        - >0.5 = D thinks fake is real (d_fake > 0, BAD - G fooling D too much)
        - <0.5 = D thinks fake is fake (d_fake < 0, GOOD)

        Lower is better - means D correctly identifies fake samples.
        """
        if self.state.last_d_fake_score is None:
            return None
        return self.sigmoid(self.state.last_d_fake_score)

    def _check_d_confusion(self, mel_loss: Optional[float], step: int) -> None:
        """
        Check if D is confused (both d_real and d_fake near 0.5).

        This indicates D can't distinguish real from fake, which is:
        - BAD during training: D has collapsed, no gradient signal for G
        - GOOD at convergence: G produces perfect samples

        We use mel_loss to distinguish: high mel = bad D, low mel = good G.
        """
        cfg = self.config
        st = self.state

        d_real_prob = self.get_d_real_normalized()
        d_fake_prob = self.get_d_fake_normalized()

        if d_real_prob is None or d_fake_prob is None:
            return

        # Check if both are near 0.5 (confused)
        real_confused = abs(d_real_prob - 0.5) < cfg.d_confusion_threshold
        fake_confused = abs(d_fake_prob - 0.5) < cfg.d_confusion_threshold

        if real_confused and fake_confused:
            st.d_confusion_steps += 1
        else:
            st.d_confusion_steps = 0
            st.d_confusion_active = False

        # Only trigger alarm if confused for long enough AND mel_loss is high
        # (meaning G hasn't converged, so this is D collapse, not good G)
        if st.d_confusion_steps >= cfg.d_confusion_steps:
            if mel_loss is not None and mel_loss > cfg.d_confusion_mel_floor:
                st.d_confusion_active = True
                # Boost D learning: we want D to learn faster to escape confusion
                # This is the opposite of velocity slowdown - D needs MORE training
                # For now, just flag it. Could add D LR boost later.
            else:
                # mel_loss is low, G is probably good - this is convergence, not collapse
                st.d_confusion_active = False

    def _trigger_alarm(self, alarm: AlarmState, step: int, reason: str) -> None:
        """Trigger alarm and apply mitigations."""
        self.state.alarm_state = alarm
        self.state.alarms_triggered += 1
        self.state.last_alarm_reason = reason

        # P2: Reset stability counter on any alarm
        self.state.stable_steps_at_level = 0

        if alarm == AlarmState.D_DOMINANT:
            # Mitigation 1: D throttling
            self.state.d_throttle_active = True
            self.state.d_throttle_until = step + self.config.d_throttle_duration
            self.state.d_throttle_every = self.config.d_throttle_every
            self._d_throttle_counter = 0

            # Mitigation 2: D LR scaling (if throttle alone doesn't help)
            # For now, just throttle. Can escalate later.

        elif alarm == AlarmState.G_COLLAPSE:
            # Boost mel weight temporarily
            self.state.mel_boost_active = True
            self.state.mel_boost_until = step + self.config.mel_boost_duration
            self.state.mel_boost = self.config.mel_weight_boost

        elif alarm == AlarmState.UNSTABLE:
            # OBSERVATION MODE: If escalation disabled, just track but don't escalate
            if not self.config.escalation_enabled:
                # Still count for observability but don't change level
                self.state.escalation_count += 1
                return

            # P1: Escalating mitigation ladder
            # Check if we should escalate (recent instability = escalate further)
            recent_escalation = (step - self.state.escalation_last_step) < self.config.escalation_memory_steps
            old_level = self.state.escalation_level
            if old_level == 0:
                # Fresh instability from healthy state - start at level 1
                self.state.escalation_level = 1
            elif recent_escalation:
                # Recent instability while already escalated - escalate further
                self.state.escalation_level = min(
                    self.state.escalation_level + 1,
                    self.config.max_escalation_level
                )
            # else: Already at L1+ but not recent - stay at current level
            # (just refresh escalation memory, don't reset to L1)

            self.state.escalation_last_step = step
            self.state.escalation_count += 1

            # P2: Track level entry for dwell time calculation
            if self.state.escalation_level != old_level:
                self.state.level_entry_step = step

            # Check for max escalation (fail closed)
            if self.state.escalation_level >= self.config.max_escalation_level:
                self._trigger_emergency(step, f"max_escalation_level_{self.state.escalation_level}")
                return

            # Apply mitigations based on new escalation level
            self._apply_mitigations_for_level(step)

    def _get_stability_threshold(self) -> int:
        """Get stability threshold based on current escalation level.

        Exponential decay: L1 recovers fast, L3 recovers slow.
        This prevents oscillation while allowing genuine recovery.
        """
        level = self.state.escalation_level
        if level >= 3:
            return self.config.stability_required_steps_l3
        elif level == 2:
            return self.config.stability_required_steps_l2
        else:
            return self.config.stability_required_steps_l1

    def _apply_mitigations_for_level(self, step: int) -> None:
        """Apply mitigations based on current escalation level.

        Mitigations are tied directly to escalation level, not timers.
        When level changes (up or down), mitigations adjust accordingly.
        This prevents the gap where mitigations expire but level hasn't decayed.
        """
        level = self.state.escalation_level

        # Level 0: No mitigations
        if level == 0:
            self.state.grad_clip_tighten_active = False
            self.state.grad_clip_scale = 1.0
            self.state.lr_scale_active = False
            self.state.lr_scale = 1.0
            self.state.d_freeze_active = False
            # Clear clip_coef history when returning to normal to avoid
            # stale values from the tightened regime triggering alarms
            self._clip_coef_g_history.clear()
            self._clip_coef_d_history.clear()
            return

        # Level 1+: Tighten grad clipping
        self.state.grad_clip_tighten_active = True
        clip_idx = min(level - 1, len(self.config.grad_clip_scales) - 1)
        self.state.grad_clip_scale = self.config.grad_clip_scales[clip_idx]

        # Level 2+: Also reduce LR
        if level >= 2:
            self.state.lr_scale_active = True
            self.state.lr_scale = self.config.lr_scale_factor
        else:
            self.state.lr_scale_active = False
            self.state.lr_scale = 1.0

        # Freeze D at configurable escalation level (default: 3, can be 2 for 55k cascade)
        was_frozen = self.state.d_freeze_active
        should_freeze = level >= self.config.d_freeze_start_level
        if should_freeze:
            self.state.d_freeze_active = True
            self.state.d_freeze_last_probe = step  # Reset probe timer
            self.state.d_unfreeze_warmup_until = 0  # Clear warmup while frozen
        else:
            self.state.d_freeze_active = False
            self.state.d_freeze_probe_active = False
            # If we were previously frozen, keep LR scaled down briefly to reduce oscillation
            if was_frozen and self.config.d_unfreeze_warmup_steps > 0:
                self.state.d_unfreeze_warmup_until = step + self.config.d_unfreeze_warmup_steps

    def _can_decay_level(self, step: int) -> bool:
        """Check if we can decay from current escalation level.

        P2: Level decay requires sustained stability, not just time passing.
        Uses exponential thresholds: L1 recovers fast, L3 recovers slow.
        """
        cfg = self.config

        # Enforce minimum dwell time at current level
        if step - self.state.level_entry_step < cfg.min_dwell_steps:
            return False

        # Require sustained stability (level-dependent threshold)
        stability_threshold = self._get_stability_threshold()
        if self.state.stable_steps_at_level < stability_threshold:
            return False

        # Soft grad ceiling check (don't decay if grads still elevated)
        if self.state.ema_grad_g > cfg.soft_grad_limit_g:
            return False
        if self.state.ema_grad_d > cfg.soft_grad_limit_d:
            return False

        # Note: We no longer gate de-escalation on d_real threshold.
        # D improving (d_real rising) is good - we use velocity-based LR scaling
        # instead of blocking de-escalation.

        return True

    def _decay_mitigations(self, step: int) -> None:
        """Decay mitigations based on stability and escalation level.

        Non-escalation mitigations (D_DOMINANT, G_COLLAPSE) expire on timer.
        Escalation mitigations (grad_clip, lr_scale, d_freeze) are tied to
        escalation level - they only change when the level changes.

        This prevents the gap where mitigations expire but level hasn't decayed,
        which caused clip-coef triggers to re-fire immediately.
        """
        # Non-escalation mitigations: timer-based expiration
        if self.state.d_throttle_active and step >= self.state.d_throttle_until:
            self.state.d_throttle_active = False
            self.state.d_throttle_every = 1
            self._d_low_count = 0  # Reset counter

        if self.state.d_lr_scale_active and step >= self.state.d_lr_scale_until:
            self.state.d_lr_scale_active = False
            self.state.d_lr_scale = 1.0

        if self.state.mel_boost_active and step >= self.state.mel_boost_until:
            self.state.mel_boost_active = False
            self.state.mel_boost = 1.0

        # Escalation mitigations: level-based, no timer expiration
        # Mitigations persist as long as we're at the escalation level
        # They only change when the level decays (below)

        # Check if we can decay escalation level
        if self.state.escalation_level == 0:
            return  # Already at baseline

        # Check stability requirements for decay
        if not self._can_decay_level(step):
            return  # Stability requirements not met

        # Decay exactly one level
        old_level = self.state.escalation_level
        self.state.escalation_level -= 1
        self.state.level_entry_step = step  # Reset dwell timer for new level
        self.state.stable_steps_at_level = 0  # Reset stability counter

        # Apply mitigations for the new (lower) level
        self._apply_mitigations_for_level(step)

    def _any_mitigation_active(self) -> bool:
        """Check if any mitigation is currently active."""
        return any([
            self.state.d_throttle_active,
            self.state.d_lr_scale_active,
            self.state.mel_boost_active,
            self.state.grad_clip_tighten_active,
            self.state.lr_scale_active,
            self.state.d_freeze_active,
        ])

    def should_update_d(self, step: int) -> bool:
        """
        Check if discriminator should be updated this step.

        Handles:
        - Pre disc_start_step: always False
        - D freeze (P1 escalation level 3): False, except during probe windows
        - D throttling: True every N steps
        """
        if step < self.disc_start_step:
            return False

        # P1: D freeze with periodic probe
        if self.state.d_freeze_active:
            # Check if we should start a probe window
            steps_since_probe = step - self.state.d_freeze_last_probe
            if not self.state.d_freeze_probe_active and steps_since_probe >= self.config.d_freeze_probe_interval:
                # Start probe window
                self.state.d_freeze_probe_active = True
                self.state.d_freeze_probe_until = step + self.config.d_freeze_probe_duration
                self.state.d_freeze_last_probe = step

            # During probe window
            if self.state.d_freeze_probe_active:
                if step >= self.state.d_freeze_probe_until:
                    # Probe window ended - back to freeze
                    self.state.d_freeze_probe_active = False
                elif not self.config.d_freeze_probe_forward_only:
                    # Probe with D updates enabled (legacy behavior, risky)
                    return True
                # else: forward-only probe - D forward pass runs but no weight updates
                # Fall through to skip D update, but probe_active remains True for logging

            # Not in probe window - skip D
            self.state.d_steps_skipped += 1
            return False

        if self.state.d_throttle_active:
            self._d_throttle_counter += 1
            if self._d_throttle_counter >= self.state.d_throttle_every:
                self._d_throttle_counter = 0
                return True
            self.state.d_steps_skipped += 1
            return False

        return True

    def get_d_lr_scale(self, step: int) -> float:
        """Get current D learning rate scale factor."""
        if step < self.disc_start_step:
            return 1.0
        return self.state.d_lr_scale if self.state.d_lr_scale_active else 1.0

    def get_adv_weight_scale(self, step: int) -> float:
        """
        Get current adversarial weight scale.

        Ramps from 0 → 1 over adv_ramp_steps after disc_start_step.
        """
        if step < self.disc_start_step:
            return 0.0

        steps_since_start = step - self.disc_start_step
        if steps_since_start >= self.config.adv_ramp_steps:
            return 1.0

        return steps_since_start / self.config.adv_ramp_steps

    def get_mel_weight_scale(self, step: int) -> float:
        """Get current mel weight scale factor."""
        return self.state.mel_boost if self.state.mel_boost_active else 1.0

    def get_grad_clip_scale(self, step: int) -> float:
        """Get current grad clip scale factor."""
        return self.state.grad_clip_scale if self.state.grad_clip_tighten_active else 1.0

    def get_lr_scale(self, step: int) -> float:
        """Get current learning rate scale factor (applies to both G and D)."""
        scale = self.state.lr_scale if self.state.lr_scale_active else 1.0
        # Post-freeze warmup: keep LR reduced briefly after D unfreezes
        if step < self.state.d_unfreeze_warmup_until:
            scale = min(scale, self.config.lr_scale_factor)
        # Apply d_real velocity scaling (slow down when D improving too fast)
        scale = scale * self.state.d_real_velocity_lr_scale
        return scale

    def _get_decision_dict(
        self,
        step: int,
        skipped_reason: Optional[str] = None,
        nan_inf_detected: bool = False,
    ) -> dict:
        """Get dict of current decisions for logging."""
        return {
            "controller_alarm": self.state.alarm_state.value,
            "d_throttle_active": self.state.d_throttle_active,
            "d_throttle_every": self.state.d_throttle_every if self.state.d_throttle_active else 1,
            "d_lr_scale": self.get_d_lr_scale(step),
            "adv_weight_scale": self.get_adv_weight_scale(step),
            "mel_weight_scale": self.get_mel_weight_scale(step),
            "grad_clip_scale": self.get_grad_clip_scale(step),
            "d_steps_skipped_total": self.state.d_steps_skipped,
            "alarms_triggered_total": self.state.alarms_triggered,
            "nan_inf_detected": nan_inf_detected,
            "skipped_reason": skipped_reason,
            # P1: Escalation state
            "escalation_level": self.state.escalation_level,
            "lr_scale": self.get_lr_scale(step),
            "d_freeze_active": self.state.d_freeze_active,
            "d_freeze_probe_active": self.state.d_freeze_probe_active,
            # Emergency state
            "emergency_stop": self.state.emergency_stop,
            "emergency_reason": self.state.emergency_reason if self.state.emergency_stop else None,
            # P2: Conditional decay state
            "stable_steps_at_level": self.state.stable_steps_at_level,
            "stability_threshold": self._get_stability_threshold(),
            "ema_grad_g": round(self.state.ema_grad_g, 1),
            "ema_grad_d": round(self.state.ema_grad_d, 1),
            # EMA elevated tracking
            "ema_elevated_steps_g": self.state.ema_elevated_steps_g,
            "ema_elevated_steps_d": self.state.ema_elevated_steps_d,
            # Spike density
            "spike_density": sum(self._spike_history) if self._spike_history else 0,
            # Clip coefficient tracking
            "hard_clip_steps_g": self.state.hard_clip_steps_g,
            "hard_clip_steps_d": self.state.hard_clip_steps_d,
            # D-real velocity tracking
            "d_real_velocity": round(self.state.d_real_velocity, 5),
            "d_real_velocity_lr_scale": round(self.state.d_real_velocity_lr_scale, 3),
            # Normalized D scores (0-1 probability, intuitive interpretation)
            "d_real_prob": round(self.get_d_real_normalized(), 3) if self.get_d_real_normalized() is not None else None,
            "d_fake_prob": round(self.get_d_fake_normalized(), 3) if self.get_d_fake_normalized() is not None else None,
            # D confusion tracking (both near 0.5 = can't distinguish)
            "d_confusion_steps": self.state.d_confusion_steps,
            "d_confusion_active": self.state.d_confusion_active,
        }

    def requires_emergency_stop(self) -> bool:
        """Check if training must stop immediately."""
        return self.state.emergency_stop

    def get_state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "alarm_state": self.state.alarm_state.value,
            "d_throttle_active": self.state.d_throttle_active,
            "d_throttle_until": self.state.d_throttle_until,
            "d_throttle_every": self.state.d_throttle_every,
            "d_lr_scale_active": self.state.d_lr_scale_active,
            "d_lr_scale_until": self.state.d_lr_scale_until,
            "d_lr_scale": self.state.d_lr_scale,
            "mel_boost_active": self.state.mel_boost_active,
            "mel_boost_until": self.state.mel_boost_until,
            "mel_boost": self.state.mel_boost,
            "grad_clip_tighten_active": self.state.grad_clip_tighten_active,
            "grad_clip_tighten_until": self.state.grad_clip_tighten_until,
            "grad_clip_scale": self.state.grad_clip_scale,
            "d_steps_skipped": self.state.d_steps_skipped,
            "alarms_triggered": self.state.alarms_triggered,
            "d_low_count": self._d_low_count,
            "consecutive_nan_inf": self.state.consecutive_nan_inf,
            "consecutive_spikes": self.state.consecutive_spikes,
            # P1: Escalation state
            "lr_scale_active": self.state.lr_scale_active,
            "lr_scale_until": self.state.lr_scale_until,
            "lr_scale": self.state.lr_scale,
            "d_freeze_active": self.state.d_freeze_active,
            "d_freeze_until": self.state.d_freeze_until,
            "d_freeze_probe_active": self.state.d_freeze_probe_active,
            "d_freeze_probe_until": self.state.d_freeze_probe_until,
            "d_freeze_last_probe": self.state.d_freeze_last_probe,
            "escalation_level": self.state.escalation_level,
            "escalation_last_step": self.state.escalation_last_step,
            "escalation_count": self.state.escalation_count,
            # P2: Conditional decay state
            "level_entry_step": self.state.level_entry_step,
            "stable_steps_at_level": self.state.stable_steps_at_level,
            "ema_grad_g": self.state.ema_grad_g,
            "ema_grad_d": self.state.ema_grad_d,
            # EMA elevated tracking
            "ema_elevated_steps_g": self.state.ema_elevated_steps_g,
            "ema_elevated_steps_d": self.state.ema_elevated_steps_d,
            # Clip coefficient tracking
            "hard_clip_steps_g": self.state.hard_clip_steps_g,
            "hard_clip_steps_d": self.state.hard_clip_steps_d,
            # Alarm reason (for diagnostics)
            "last_alarm_reason": self.state.last_alarm_reason,
            # D-real velocity tracking
            "last_d_real_score": self.state.last_d_real_score,
            "last_d_fake_score": self.state.last_d_fake_score,
            "d_real_velocity": self.state.d_real_velocity,
            "d_real_velocity_lr_scale": self.state.d_real_velocity_lr_scale,
            "d_real_history": list(self._d_real_history),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.state.alarm_state = AlarmState(state_dict.get("alarm_state", "healthy"))
        self.state.d_throttle_active = state_dict.get("d_throttle_active", False)
        self.state.d_throttle_until = state_dict.get("d_throttle_until", 0)
        self.state.d_throttle_every = state_dict.get("d_throttle_every", 1)
        self.state.d_lr_scale_active = state_dict.get("d_lr_scale_active", False)
        self.state.d_lr_scale_until = state_dict.get("d_lr_scale_until", 0)
        self.state.d_lr_scale = state_dict.get("d_lr_scale", 1.0)
        self.state.mel_boost_active = state_dict.get("mel_boost_active", False)
        self.state.mel_boost_until = state_dict.get("mel_boost_until", 0)
        self.state.mel_boost = state_dict.get("mel_boost", 1.0)
        self.state.grad_clip_tighten_active = state_dict.get("grad_clip_tighten_active", False)
        self.state.grad_clip_tighten_until = state_dict.get("grad_clip_tighten_until", 0)
        self.state.grad_clip_scale = state_dict.get("grad_clip_scale", 1.0)
        self.state.d_steps_skipped = state_dict.get("d_steps_skipped", 0)
        self.state.alarms_triggered = state_dict.get("alarms_triggered", 0)
        self._d_low_count = state_dict.get("d_low_count", 0)
        self.state.consecutive_nan_inf = state_dict.get("consecutive_nan_inf", 0)
        self.state.consecutive_spikes = state_dict.get("consecutive_spikes", 0)
        # P1: Escalation state
        self.state.lr_scale_active = state_dict.get("lr_scale_active", False)
        self.state.lr_scale_until = state_dict.get("lr_scale_until", 0)
        self.state.lr_scale = state_dict.get("lr_scale", 1.0)
        self.state.d_freeze_active = state_dict.get("d_freeze_active", False)
        self.state.d_freeze_until = state_dict.get("d_freeze_until", 0)
        self.state.d_freeze_probe_active = state_dict.get("d_freeze_probe_active", False)
        self.state.d_freeze_probe_until = state_dict.get("d_freeze_probe_until", 0)
        self.state.d_freeze_last_probe = state_dict.get("d_freeze_last_probe", 0)
        self.state.escalation_level = state_dict.get("escalation_level", 0)
        self.state.escalation_last_step = state_dict.get("escalation_last_step", 0)
        self.state.escalation_count = state_dict.get("escalation_count", 0)
        # P2: Conditional decay state
        self.state.level_entry_step = state_dict.get("level_entry_step", 0)
        self.state.stable_steps_at_level = state_dict.get("stable_steps_at_level", 0)
        self.state.ema_grad_g = state_dict.get("ema_grad_g", 0.0)
        self.state.ema_grad_d = state_dict.get("ema_grad_d", 0.0)
        # EMA elevated tracking
        self.state.ema_elevated_steps_g = state_dict.get("ema_elevated_steps_g", 0)
        self.state.ema_elevated_steps_d = state_dict.get("ema_elevated_steps_d", 0)
        # Clip coefficient tracking
        self.state.hard_clip_steps_g = state_dict.get("hard_clip_steps_g", 0)
        self.state.hard_clip_steps_d = state_dict.get("hard_clip_steps_d", 0)
        # Alarm reason
        self.state.last_alarm_reason = state_dict.get("last_alarm_reason", "")
        # D-real velocity tracking
        self.state.last_d_real_score = state_dict.get("last_d_real_score")
        self.state.last_d_fake_score = state_dict.get("last_d_fake_score")
        self.state.d_real_velocity = state_dict.get("d_real_velocity", 0.0)
        self.state.d_real_velocity_lr_scale = state_dict.get("d_real_velocity_lr_scale", 1.0)
        # Restore d_real history deque
        d_real_hist = state_dict.get("d_real_history", [])
        self._d_real_history.clear()
        for val in d_real_hist[-self.config.d_real_velocity_window:]:
            self._d_real_history.append(val)
        # Don't restore emergency state - fresh start after resume
