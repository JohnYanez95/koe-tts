# Postmortem: multi_vits_gan_20260131_123942

**Date:** 2026-01-31
**Run ID:** multi_vits_gan_20260131_123942
**Type:** trigger_miscalibration
**Outcome:** Training stable but de-escalation blocked by overly tight soft_grad_limit

## Summary

After fixing the config mapping bug, the soft_grad_limit_g gate (200) was correctly enforced but proved too tight for normal G gradient variance. The stable step counter reset constantly due to G gradient spikes exceeding 200, preventing any progress toward de-escalation from L3.

## Timeline

| Step | Event | Details |
|------|-------|---------|
| 51140 | Resume | From last_known_good, L3 with D-freeze |
| 51140-54800 | Training | Stable losses, mel improving |
| 54800 | Observation | stable=1/1000 despite hours of training |
| 57757 | Example reset | g_grad=355.6 > 200 → stable counter reset |
| 57816 | Probe spike | g_grad=712.8 during D-freeze probe |
| 57850 | Stopped | Raised soft_grad_limit_g to 400 |

## Root Cause

The stable step counting logic requires raw g_grad to be below soft_grad_limit_g **every single step**:

```python
# gan_controller.py line 504
grads_below_soft = (
    grad_norm_g <= self.config.soft_grad_limit_g
    and grad_norm_d <= self.config.soft_grad_limit_d
)
if grads_below_soft:
    self.state.stable_steps_at_level += 1
else:
    # Reset on elevated grads
    self.state.stable_steps_at_level = 0
```

With soft_grad_limit_g=200:
- Normal G gradient range during D-freeze: 50-400
- Frequent spikes to 300-500 are normal variance
- One spike above 200 → counter resets to 0
- Counter never accumulates past ~10 steps

### Probe Interference

D-freeze probes (10 steps every 100) unfreeze D briefly. During probes:
- D fights back against G
- G gradients spike (500-700+ observed)
- Counter resets every probe cycle

## What Worked

1. **Config mapping fix** — soft_grad_limit_g=200 was correctly read from YAML
2. **D-freeze** — G losses stable and improving during freeze
3. **d_real healthy** — stayed at -0.3 to -0.5, no inversion risk
4. **Detection system** — no false alarms triggered

## What Didn't Work

1. **soft_grad_limit_g=200 too tight** — normal G variance exceeds this
2. **Raw g_grad check** — using instantaneous value instead of EMA causes noise sensitivity
3. **Probe spikes reset counter** — expected behavior but blocks stability accumulation

## Metrics

```
G gradient during D-freeze:
  p10:  ~60
  p50:  ~150
  p90:  ~350
  spikes: 500-700+

Stable counter behavior:
  Max accumulated: ~10 steps
  Resets per 100 steps: ~15-20
```

## Resolution

Raised soft_grad_limit_g from 200 to 400:

```yaml
controller:
  soft_grad_limit_g: 400    # Was 200
  soft_grad_limit_d: 200    # Unchanged (D is frozen, grads are 0)
```

This allows:
- Normal G variance (50-350) to pass
- Only true spikes (400+) to reset counter
- Probes may still cause resets, but fewer

## Alternative Approaches Considered

1. **Use EMA_G instead of raw g_grad** — smoother signal, less noise sensitivity
   - Pro: More robust to single-step spikes
   - Con: Requires code change, EMA is already tracked separately

2. **Increase threshold further (500+)** — match EMA elevated limit
   - Pro: Very permissive
   - Con: May allow genuinely unstable regimes to de-escalate

3. **Reduce probe frequency** — fewer D-freeze probes = fewer resets
   - Pro: Addresses probe interference
   - Con: Less visibility into D health

## Lessons Learned

1. **Calibrate thresholds from data.** The 200 value was a guess. Should have analyzed g_grad distribution during D-freeze first.

2. **Raw metrics are noisy.** Instantaneous gradient norms have high variance. EMA-based gates are more robust.

3. **Probes are double-edged.** They provide useful diagnostic data but interfere with stability accumulation.

## Follow-up Actions

- [x] Raise soft_grad_limit_g to 400
- [ ] Monitor if 400 allows reasonable de-escalation progress
- [ ] Consider using EMA_G for stable counting instead of raw g_grad
- [ ] Document typical g_grad ranges in GAN_CONTROLLER.md

## Related

- Previous incident: [config_mapping_gap](./2026-01-31_multi_vits_gan_20260131_115157_config_mapping_gap.md)
- Controller docs: [GAN_CONTROLLER.md](../training/GAN_CONTROLLER.md)
