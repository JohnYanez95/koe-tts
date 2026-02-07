# Postmortem: multi_vits_gan_20260201_101706

**Date:** 2026-02-01
**Run ID:** multi_vits_gan_20260201_101706
**Type:** trigger_miscalibration
**Outcome:** D starved during D-freeze, oscillated between L1-L3 for 10k steps

## Summary

Fresh restart from VITS core checkpoint (step 14000) with forward-only probes disabled, giving D weight updates during probes. However, probe frequency (10 steps every 100 = 10% of training time) was insufficient for D to learn. The run spent most of its time oscillating between L1/L2/L3 as d_real kept drifting toward critical/inversion thresholds.

## Timeline

| Step | Event | Details |
|------|-------|---------|
| 14000 | Start | Fresh D from core checkpoint, healthy |
| 14277 | Instability | median_clip_d trigger → L1 (expected cold start) |
| 14787 | Recovery | De-escalated to L0 |
| 15004 | d_real_critical | → L2, D-freeze begins |
| 15054 | d_real_inversion | → L3 |
| 15054-21527 | D-freeze | L3 for 6400+ steps |
| 21527-25000 | Oscillation | 11 level changes, L1↔L2↔L3 |
| 25000 | Killed | Manual stop for parameter tuning |

**Total escalation events:** 13 level changes in 11,000 steps
**d_real_critical triggers:** 8
**d_real_inversion triggers:** 2

## Root Cause

D-freeze probes were configured with:
- `d_freeze_probe_interval: 100` (probe every 100 steps)
- `d_freeze_probe_duration: 10` (probe lasts 10 steps)

This gives D only **10% of training time** to learn during D-freeze.

With D frozen 90% of the time:
1. D can't learn to distinguish real vs fake
2. d_real score drifts toward 0 (critical/inversion zone)
3. Controller escalates when d_real crosses thresholds
4. Brief de-escalation allows D some learning
5. D can't recover fast enough, d_real drifts back
6. Re-escalation, cycle repeats

### The D Starvation Problem

D-freeze protects G from a runaway discriminator but starves D of learning signal:

```
Normal training:  G and D alternate updates, balanced competition
D-freeze mode:    G trains continuously, D gets 10% updates
                  D falls behind → d_real drifts → more D-freeze
```

The 10% probe window wasn't enough for D to catch up.

## What Worked

1. **d_real detection** — correctly identified D falling behind
2. **Escalation levels** — system responded appropriately to d_real drift
3. **Level preservation** — fixed bug from previous run, levels didn't reset incorrectly
4. **Fresh start** — avoided carrying over corrupted state from inverted checkpoints

## What Didn't Work

1. **Probe frequency too low** — 10% training time insufficient for D recovery
2. **Oscillation pattern** — system correctly detected problems but couldn't resolve them
3. **No probe duration scaling** — probe parameters didn't adapt to how far behind D was

## Metrics

```
Escalation pattern (steps 14000-25000):
  L0: ~800 steps (7%)
  L1: ~1500 steps (14%)
  L2: ~2500 steps (23%)
  L3: ~6200 steps (56%)

Probe configuration (10% training time):
  Interval: 100 steps
  Duration: 10 steps
  D training ratio: 10/100 = 10%
```

## Resolution

Increased probe frequency and duration to give D ~40% training time:

```yaml
# Before
d_freeze_probe_interval: 100   # Probe every 100 steps
d_freeze_probe_duration: 10    # Probe lasts 10 steps

# After
d_freeze_probe_interval: 50    # Probe every 50 steps
d_freeze_probe_duration: 20    # Probe lasts 20 steps
# D training ratio: 20/50 = 40%
```

This 4x increase in D training time should allow D to:
- Learn enough during probes to push d_real back to healthy range
- Recover from escalation within reasonable time
- Avoid the oscillation pattern

## Lessons Learned

1. **D-freeze is a tradeoff.** It protects G but starves D. Must balance protection vs learning.

2. **Probe parameters need calibration.** 10% was too aggressive; 40% is a better starting point.

3. **Oscillation = under-resourced recovery.** When the system bounces between states, it means the recovery mechanism (probes) isn't strong enough.

4. **Fresh D is fragile.** Cold start from core checkpoint means D has no learned features. It needs more training time to stabilize, not less.

## Follow-up Actions

- [x] Increase probe interval to 50 steps
- [x] Increase probe duration to 20 steps
- [ ] Monitor next run for improved d_real stability
- [ ] Consider adaptive probe duration based on escalation level or d_real distance from healthy
- [ ] Document probe parameter tuning in GAN_CONTROLLER.md

## Related

- Previous incident: [soft_limit_too_tight](./2026-01-31_multi_vits_gan_20260131_123942_soft_limit_too_tight.md)
- Controller docs: [GAN_CONTROLLER.md](../training/GAN_CONTROLLER.md)
