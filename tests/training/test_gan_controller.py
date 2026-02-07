"""Tests for GAN controller stability mechanisms."""

from modules.training.common.gan_controller import (
    GANController,
    GANControllerConfig,
    AlarmState,
)


class TestConditionalDecay:
    """P2: Conditional decay tests."""

    def test_expired_mitigations_clear_without_stability(self):
        """Expired mitigations clear even when stability conditions are not met.

        This ensures mitigations don't get "stuck on" forever.
        """
        config = GANControllerConfig(
            min_dwell_steps=500,
            stability_required_steps=1000,
            grad_clip_tighten_duration=100,
        )
        controller = GANController(config, disc_start_step=0)

        # Manually set up state: clip active, will expire at step 100
        controller.state.grad_clip_tighten_active = True
        controller.state.grad_clip_tighten_until = 100
        controller.state.grad_clip_scale = 0.5
        controller.state.escalation_level = 1

        # Stability NOT met: 0 stable steps, no dwell time
        controller.state.stable_steps_at_level = 0
        controller.state.level_entry_step = 0

        # Call decay at step 101 (past expiry)
        controller._decay_mitigations(101)

        # Mitigation should clear (step >= until)
        assert controller.state.grad_clip_tighten_active is False
        assert controller.state.grad_clip_scale == 1.0

        # But level should NOT decay (stability not met)
        assert controller.state.escalation_level == 1

    def test_level_requires_dwell_and_stability_to_decay(self):
        """Level does not decay without both dwell time and stability."""
        config = GANControllerConfig(
            min_dwell_steps=500,
            stability_required_steps=1000,
            soft_grad_limit_g=2000.0,
            soft_grad_limit_d=2000.0,
        )
        controller = GANController(config, disc_start_step=0)

        # Set up: level 2, mitigations cleared, step 2000, but only 999 stable steps
        controller.state.escalation_level = 2
        controller.state.grad_clip_tighten_active = False
        controller.state.lr_scale_active = False
        controller.state.level_entry_step = 1000  # 1000 steps ago = meets dwell
        controller.state.stable_steps_at_level = 999  # Just shy of requirement
        controller.state.ema_grad_g = 500.0  # Below soft limit
        controller.state.ema_grad_d = 500.0

        # Decay should NOT happen (stability requirement not met)
        controller._decay_mitigations(2000)
        assert controller.state.escalation_level == 2

        # Now meet stability requirement
        controller.state.stable_steps_at_level = 1000

        # Decay should happen
        controller._decay_mitigations(2001)
        assert controller.state.escalation_level == 1

        # Level entry step should be updated
        assert controller.state.level_entry_step == 2001

        # Stable steps should be reset for new level
        assert controller.state.stable_steps_at_level == 0

    def test_level_blocked_by_soft_grad_limit(self):
        """Level does not decay if grads are above soft limit."""
        config = GANControllerConfig(
            min_dwell_steps=100,
            stability_required_steps=100,
            soft_grad_limit_g=2000.0,
            soft_grad_limit_d=2000.0,
        )
        controller = GANController(config, disc_start_step=0)

        # Set up: all conditions met EXCEPT grad is elevated
        controller.state.escalation_level = 1
        controller.state.grad_clip_tighten_active = False
        controller.state.level_entry_step = 0
        controller.state.stable_steps_at_level = 200  # Exceeds requirement
        controller.state.ema_grad_g = 2500.0  # Above soft limit!
        controller.state.ema_grad_d = 500.0

        # Decay should NOT happen (G grad too high)
        controller._decay_mitigations(200)
        assert controller.state.escalation_level == 1

        # Lower the grad
        controller.state.ema_grad_g = 1500.0

        # Now decay should happen
        controller._decay_mitigations(201)
        assert controller.state.escalation_level == 0

    def test_alarm_resets_stability_counter(self):
        """Triggering an alarm resets the stability counter."""
        config = GANControllerConfig()
        controller = GANController(config, disc_start_step=0)

        # Accumulate some stability
        controller.state.stable_steps_at_level = 500
        controller.state.escalation_level = 1

        # Trigger an alarm
        controller._trigger_alarm(AlarmState.UNSTABLE, step=1000, reason="test")

        # Stability counter should be reset
        assert controller.state.stable_steps_at_level == 0

    def test_stability_only_increments_when_healthy(self):
        """Stability counter only increments on genuinely stable steps."""
        config = GANControllerConfig(
            soft_grad_limit_g=2000.0,
            soft_grad_limit_d=2000.0,
            grad_norm_spike_factor=2.0,  # default
        )
        controller = GANController(config, disc_start_step=0)

        # Prime the windows with data around 500 so 500-600 won't spike
        # (p99 * 2.0 threshold means we need values that make threshold > 600)
        for _ in range(60):
            controller._grad_norm_g_window.append(500.0)
            controller._grad_norm_d_window.append(500.0)
            controller._d_loss_window.append(0.5)
            controller._g_loss_adv_window.append(2.0)

        # Stable step: HEALTHY alarm state, grads below soft limit and not spiking
        controller.state.alarm_state = AlarmState.HEALTHY
        result = controller.record_step(
            step=100,
            d_loss=0.5,
            g_loss_adv=2.0,
            grad_norm_g=500.0,  # Below soft limit, not a spike
            grad_norm_d=500.0,
        )
        assert controller.state.stable_steps_at_level == 1

        # Another stable step
        result = controller.record_step(
            step=101,
            d_loss=0.5,
            g_loss_adv=2.0,
            grad_norm_g=550.0,  # Still not a spike (< 500 * 2.0 = 1000)
            grad_norm_d=550.0,
        )
        assert controller.state.stable_steps_at_level == 2

        # Non-stable: missing grad (None)
        result = controller.record_step(
            step=102,
            d_loss=0.5,
            g_loss_adv=2.0,
            grad_norm_g=None,  # Missing signal
            grad_norm_d=500.0,
        )
        # Counter should reset (don't accumulate on missing signal)
        assert controller.state.stable_steps_at_level == 0

    def test_ema_grad_tracking(self):
        """EMA grad values are tracked correctly."""
        config = GANControllerConfig(
            grad_ema_alpha=0.1,
            grad_norm_spike_factor=3.0,  # Higher factor so our values don't spike
        )
        controller = GANController(config, disc_start_step=0)

        # Prime windows with values around 1000 so 1000-2000 won't spike
        # (p99 * 3.0 threshold = 3000, so 2000 won't trigger)
        for _ in range(60):
            controller._grad_norm_g_window.append(1000.0)
            controller._grad_norm_d_window.append(500.0)
            controller._d_loss_window.append(0.5)
            controller._g_loss_adv_window.append(2.0)

        # First value initializes EMA
        controller.record_step(
            step=100, d_loss=0.5, g_loss_adv=2.0,
            grad_norm_g=1000.0, grad_norm_d=500.0,
        )
        assert controller.state.ema_grad_g == 1000.0  # First value
        assert controller.state.ema_grad_d == 500.0

        # Second value applies EMA: 0.1 * new + 0.9 * old
        controller.record_step(
            step=101, d_loss=0.5, g_loss_adv=2.0,
            grad_norm_g=1500.0, grad_norm_d=800.0,  # Within range, won't spike
        )
        expected_g = 0.1 * 1500.0 + 0.9 * 1000.0  # 1050
        expected_d = 0.1 * 800.0 + 0.9 * 500.0    # 530
        assert abs(controller.state.ema_grad_g - expected_g) < 0.01
        assert abs(controller.state.ema_grad_d - expected_d) < 0.01


class TestLevelEntryTracking:
    """Test level_entry_step is set correctly on level changes."""

    def test_level_entry_set_on_escalation(self):
        """level_entry_step is set when escalating."""
        config = GANControllerConfig()
        controller = GANController(config, disc_start_step=0)

        # Trigger escalation
        controller._trigger_alarm(AlarmState.UNSTABLE, step=1000, reason="test")

        assert controller.state.escalation_level == 1
        assert controller.state.level_entry_step == 1000

    def test_level_entry_set_on_decay(self):
        """level_entry_step is set when decaying a level."""
        config = GANControllerConfig(
            min_dwell_steps=100,
            stability_required_steps=100,
            soft_grad_limit_g=5000.0,
            soft_grad_limit_d=5000.0,
        )
        controller = GANController(config, disc_start_step=0)

        # Set up for decay
        controller.state.escalation_level = 2
        controller.state.grad_clip_tighten_active = False
        controller.state.lr_scale_active = False
        controller.state.level_entry_step = 0
        controller.state.stable_steps_at_level = 200
        controller.state.ema_grad_g = 100.0
        controller.state.ema_grad_d = 100.0

        # Decay
        controller._decay_mitigations(200)

        assert controller.state.escalation_level == 1
        assert controller.state.level_entry_step == 200  # Updated to decay step


class TestDecayOneAtATime:
    """Test that recovery is one level at a time."""

    def test_decay_only_one_level_per_call(self):
        """Decay happens one level at a time, not snapping back to baseline."""
        config = GANControllerConfig(
            min_dwell_steps=10,
            stability_required_steps=10,
            soft_grad_limit_g=5000.0,
            soft_grad_limit_d=5000.0,
        )
        controller = GANController(config, disc_start_step=0)

        # Start at level 3, all mitigations cleared
        controller.state.escalation_level = 3
        controller.state.grad_clip_tighten_active = False
        controller.state.lr_scale_active = False
        controller.state.d_freeze_active = False
        controller.state.level_entry_step = 0
        controller.state.stable_steps_at_level = 100
        controller.state.ema_grad_g = 100.0
        controller.state.ema_grad_d = 100.0

        # First decay: 3 -> 2
        controller._decay_mitigations(100)
        assert controller.state.escalation_level == 2
        assert controller.state.stable_steps_at_level == 0  # Reset

        # Need to rebuild stability
        controller.state.stable_steps_at_level = 100

        # Second decay: 2 -> 1
        controller._decay_mitigations(200)
        assert controller.state.escalation_level == 1

        # Need to rebuild stability again
        controller.state.stable_steps_at_level = 100

        # Third decay: 1 -> 0
        controller._decay_mitigations(300)
        assert controller.state.escalation_level == 0
