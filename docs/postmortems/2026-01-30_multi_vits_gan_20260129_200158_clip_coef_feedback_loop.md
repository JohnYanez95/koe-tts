# Postmortem: Clip Coefficient Feedback Loop Emergency

**Run ID:** `multi_vits_gan_20260129_200158`
**Date:** 2026-01-30
**Incident Type:** `trigger_miscalibration` / `controller_design_flaw`
**Outcome:** Emergency stop at step 25,114 with healthy gradients

---

## Summary

Training resumed from a healthy checkpoint (step 24,000) and progressed normally until step 24,747 when a NaN/Inf event triggered escalation. The controller applied mitigations (tighter grad clipping), but its own mitigation caused the `clip_coef` to drop below the alarm threshold, triggering further escalation in a feedback loop. Training reached emergency stop at step 25,114 despite having **perfectly healthy gradient metrics**.

---

## Timeline

| Step | Event | Metrics |
|------|-------|---------|
| 24,000 | Resume from healthy checkpoint | All metrics normal |
| 24,233 | Last confirmed healthy | stable_steps = 668 |
| 24,747 | **NaN/Inf detected** | g_grad_norm = 100 → null |
| 24,748 | Escalate L0 → L1 | grad_clip_scale = 0.5, g_step_skipped |
| 24,749-24,751 | Gradient explosion | g_grad_norm: 158 → 211 → **257** |
| 24,800+ | Gradients recover | g_grad_norm ≈ 20-40 |
| ~24,965 | Escalate L1 → L2 | grad_clip_scale = 0.25, lr_scale = 0.5 |
| 25,060 | Metrics healthy but stuck | g_grad_norm = 21, g_clip_coef = 0.012 |
| 25,065 | Escalate L2 → L3 | d_freeze_active = true |
| 25,085+ | D frozen, clip_coef perpetually low | d_grad_norm = 0, g_clip_coef ≈ 0.005 |
| **25,114** | **EMERGENCY STOP** | g_grad_norm = 13 (healthy!), ema_grad_g = 18 (healthy!) |

---

## Root Cause Analysis

### The Feedback Loop

```
1. NaN/Inf → escalate L1 → grad_clip_scale = 0.5
2. Tighter clipping → clip_coef drops (expected behavior)
3. clip_coef < 0.05 → "hard clipping" alarm fires
4. Alarm → escalate L2 → grad_clip_scale = 0.25
5. Even tighter clipping → clip_coef drops more
6. clip_coef still < 0.05 → alarm keeps firing
7. Escalate L3, L4 → EMERGENCY
```

### Why This Happens

The clip coefficient is calculated as:

```python
clip_coef = min(1.0, clip_value / grad_norm)
```

When `grad_clip_scale` is reduced:

```
effective_clip = base_clip × grad_clip_scale

At L0: effective_clip = 1.0 × 1.0 = 1.0  → clip_coef ≈ 1.0/20 = 0.05
At L1: effective_clip = 1.0 × 0.5 = 0.5  → clip_coef ≈ 0.5/20 = 0.025
At L2: effective_clip = 1.0 × 0.25 = 0.25 → clip_coef ≈ 0.25/20 = 0.0125
At L3: effective_clip = 1.0 × 0.1 = 0.1  → clip_coef ≈ 0.1/20 = 0.005
```

**The controller's own mitigation (tighter clipping) triggers the "hard clipping" alarm.**

### The Soft Limit Red Herring

The `soft_grad_limit` of 2000 is irrelevant here. It's only used for decay decisions, not alarm triggers. The gradients were 13-21 (perfectly normal), but the system still reached emergency because of the clip_coef alarm.

---

## Evidence

### Healthy Metrics at Emergency Stop

```json
{
  "step": 25114,
  "g_grad_norm": 13.35,
  "d_grad_norm": 0.0,
  "ema_grad_g": 18.2,
  "ema_grad_d": 0.1,
  "g_clip_coef": 0.0075,
  "ctrl_controller_alarm": "unstable",
  "ctrl_escalation_level": 3,
  "ctrl_d_freeze_active": true
}
```

Key observations:
- `g_grad_norm = 13.35` — well below normal (20-40)
- `ema_grad_g = 18.2` — well below soft limit (2000)
- `g_clip_coef = 0.0075` — low because `grad_clip_scale = 0.1`

### Escalation with Healthy Gradients

At step 25,065 when L3 activated:

```json
{
  "step": 25065,
  "g_grad_norm": 16.95,
  "d_grad_norm": 8.91,
  "ema_grad_g": 17.2,
  "ema_grad_d": 9.0,
  "ctrl_escalation_level": 3,
  "ctrl_alarms_triggered_total": 4
}
```

**Gradients were completely normal.** The escalation was driven entirely by the clip_coef feedback loop.

---

## Fix Applied (2026-01-30)

**Implemented Option 1:** Disable hard clip detection when `grad_clip_scale < 1.0`.

```python
# In gan_controller.py _check_instability():
clip_check_enabled = self.state.grad_clip_scale >= 1.0

if clip_check_enabled:
    if g_clip_coef < self.config.clip_coef_hard_threshold:
        self.state.hard_clip_steps_g += 1
        # ... trigger logic
else:
    # Reset counter — low clip_coef is expected when scale < 1.0
    self.state.hard_clip_steps_g = 0
```

The same fix applies to both G and D clip coefficient checks, and to both consecutive and median triggers.

---

## Lessons Learned

1. **Mitigations can cause alarms:** The controller's own actions (tightening clip) triggered further alarms (hard clipping), creating a self-reinforcing loop.

2. **Soft limits ≠ alarm thresholds:** The `soft_grad_limit` (2000) is for decay decisions only. Perfectly healthy gradients (13-21) can still trigger emergency via clip_coef.

3. **D freeze masks D metrics:** When D is frozen, `d_grad_norm = 0` by design. The controller correctly ignores this, but it can look alarming in logs.

4. **Test escalation paths:** Need tests that verify escalation doesn't self-reinforce. A single NaN shouldn't cascade to emergency when gradients recover.

---

## Action Items

- [x] Fix clip_coef feedback loop (Option 1) — **Done 2026-01-30**
- [x] Update GAN_CONTROLLER.md with feedback loop documentation — **Done 2026-01-30**
- [ ] Add test: "single NaN + recovery should not reach emergency"
- [ ] Add test: "escalation with healthy gradients should eventually decay"
- [ ] Validate fix with training run through 27k-30k danger band

---

## Related

- [GAN Controller Reference](../training/GAN_CONTROLLER.md)
- [GAN Stability Log](./gan_stability_log.md)
- [Previous: De-escalation Bug](./2026-01-29_multi_vits_gan_calibrated_deescalation_bug.md)
