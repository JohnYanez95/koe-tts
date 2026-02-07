# Postmortem: Controller De-escalation Bug

**Date**: 2026-01-29
**Run ID**: `multi_vits_gan_20260129_calibrated`
**Type**: `trigger_miscalibration` / `controller_bug`
**Outcome**: Training stopped at step 26272, despite healthy losses

## Summary

Training was stopped by an emergency controller action due to a **circular dependency bug** in the de-escalation logic. A single transient NaN/Inf event triggered UNSTABLE state, and the controller could never prove stability to de-escalate because stability tracking required being HEALTHY first.

## Timeline

| Step | Event |
|------|-------|
| 18000 | Resumed from checkpoint, healthy |
| 22065 | Single G NaN/Inf → alarm #1 → recovered after 500 steps |
| 25929 | Single D NaN/Inf → alarm #2 → escalation L1 |
| 26190 | Escalation L2 (LR halved) |
| 26223 | Escalation L3 (D-freeze, last_known_good saved) |
| 26272 | **Emergency stop** (L4) after 50 steps at L3 without stability |

## Root Cause Analysis

### Immediate Trigger
At step 25929, a single discriminator step was skipped due to NaN/Inf:
```
d_step_skipped: True
ctrl_nan_inf_detected: True
```

### Underlying Bug
The controller's stability tracking had a **circular dependency**:

```python
# Old code (buggy)
if self.state.alarm_state == AlarmState.HEALTHY and grads_below_soft:
    self.state.stable_steps_at_level += 1
else:
    self.state.stable_steps_at_level = 0  # Reset every step!
```

**Problem**: Stability counter only accumulated when `alarm_state == HEALTHY`, but the alarm couldn't return to HEALTHY without proving stability first. Result: `stable_steps_at_level` was stuck at 0 forever.

### Evidence Training Was Actually Healthy

During the entire L1 "unstable" period (25929-26272), losses were completely normal:

| Metric | Range | Assessment |
|--------|-------|------------|
| D_loss | 14.8-15.9 | Perfect equilibrium |
| Adv | 10.8-13.0 | Normal |
| Mel | 1.1-1.6 | Normal |
| KL | 1.0-2.0 | **No spike** (vs 7.2 in real instability) |
| gn_G | 12-50 | Slightly elevated but fine |
| gn_D | 8-25 | Normal |

## Resolution

Fixed the de-escalation logic:

1. **Removed HEALTHY requirement**: Stability now accumulates when grads are below soft limits, regardless of alarm state
2. **Level-based thresholds**: Exponential decay - L1 recovers fast (200 steps), L3 recovers slow (1000 steps)
3. **Added logging**: `stability_threshold` now visible in metrics for debugging

```python
# New code (fixed)
if grads_below_soft:
    self.state.stable_steps_at_level += 1
else:
    self.state.stable_steps_at_level = 0  # Only reset on actual elevated grads
```

## Config Changes

Added to `vits_gan.yaml`:
```yaml
stability_required_steps_l1: 200   # Fast recovery from L1
stability_required_steps_l2: 500   # Medium recovery from L2
stability_required_steps_l3: 1000  # Slow recovery from L3
```

## Files Modified

- `modules/training/common/gan_controller.py` - De-escalation logic fix
- `modules/training/pipelines/train_vits.py` - Config mapping
- `configs/training/vits_gan.yaml` - New thresholds

## Lessons Learned

1. **Single NaN/Inf events are often transient** - The training loop correctly skipped the step and recovered, but the controller didn't recognize this
2. **Circular dependencies in state machines are subtle** - The HEALTHY→stable→HEALTHY loop looked correct but was impossible to traverse
3. **Monitor actual loss metrics, not just controller state** - The losses showed training was healthy throughout

## Follow-up Actions

- [x] Fix de-escalation logic
- [x] Add level-based stability thresholds
- [x] Add `stability_threshold` to logging
- [ ] Resume training from `last_known_good` checkpoint to validate fix
- [ ] Consider adding "transient NaN/Inf" handling (single skip doesn't trigger alarm)

## Related

- CLAUDE.md GAN stability section updated
- `gan_stability_log.md` entry added
