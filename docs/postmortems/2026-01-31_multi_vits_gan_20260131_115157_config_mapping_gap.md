# Postmortem: multi_vits_gan_20260131_115157

**Date:** 2026-01-31
**Run ID:** multi_vits_gan_20260131_115157
**Type:** config_mapping_gap
**Outcome:** Partial success — d_real detection worked, but config mapping bug allowed premature de-escalation

## Summary

This run tested the new d_real leading indicator system implemented in commit eb4dcb0. The d_real inversion detection **worked correctly**, catching an inversion at step 51140 and triggering L3 escalation with D-freeze. However, a config mapping bug caused the soft_grad_limit gate to use code defaults (2000) instead of YAML values (200), allowing premature de-escalation that led to the inversion.

## Timeline

| Step | Event | Details |
|------|-------|---------|
| 50000 | Resume | From step_050000.pt, L3 with D-freeze active |
| 50620 | L3→L2 | De-escalated after 1000 stable steps |
| 51120 | L2→L1 | **Premature** — EMA_G=318 but soft_limit=2000 (should be 200) |
| 51140 | L1→L3 | **d_real_inversion triggered** — d_real crossed 0.0 |
| 51140 | Checkpoint | last_known_good saved automatically |
| 52240 | L3→L2 | De-escalated after recovery |
| 52800 | L2→L1 | De-escalated again |
| 52929 | Stopped | Manual stop for batch size reduction |

## Root Cause

**Config mapping incomplete in train_vits.py.** The YAML config values were not being passed to `GANControllerConfig`:

```python
# train_vits.py lines 1472-1499 — MISSING mappings:
# - soft_grad_limit_g (YAML: 200, code default: 2000)
# - soft_grad_limit_d (YAML: 200, code default: 2000)
# - All d_real threshold configs
# - d_freeze_start_level
# - d_unfreeze_warmup_steps
```

This is documented in CLAUDE.md as a known gotcha:
> **Config gotcha:** All controller fields in `vits_gan.yaml` must be explicitly mapped in `train_vits.py`'s `GANControllerConfig()` constructor, or they'll use code defaults.

### Impact

At step 51120:
- EMA_G = 318.1 (elevated)
- soft_grad_limit_g = 2000 (code default, should be 200)
- 318 < 2000 → de-escalation allowed
- 20 steps later → inversion

## What Worked

### 1. d_real Inversion Detection ✓
The new leading indicator caught the inversion immediately:
```
51138   d_real=-0.083   L1
51139   d_real=-0.162   L1
51140   d_real=+0.054   L1 → L3 (d_real_inversion)
```

### 2. Automatic Checkpoint on D-Freeze ✓
`last_known_good` checkpoint saved at step 51140 when D-freeze activated.

### 3. D-Freeze Stabilization ✓
After L3 escalation, D-freeze prevented further deterioration. Model recovered:
```
51141   d_real=+0.022   L3F
51145   d_real=+0.064   L3F
...
51300   d_real=-0.31    L3F (recovered)
```

### 4. Escalation Level Jumps ✓
L1→L3 jump worked correctly (not one level at a time).

### 5. Existing Controls from eb4dcb0

The run utilized these controls from the previous commit:
- **Escalation levels (L1-L3)** with graduated mitigations
- **D-freeze at L3** preventing discriminator updates
- **D-freeze probes** (10 steps every 100) to test discrimination
- **Stability counters** with level-specific thresholds (L1=200, L2=500, L3=1000)
- **EMA gradient tracking** for trend detection
- **Spike density detection** (5 in 20 steps)

## What Didn't Work

### 1. Soft Grad Limit Gate ✗
Gate used code default (2000) instead of YAML value (200). Did not block de-escalation when EMA_G was elevated.

### 2. d_real De-escalation Threshold ✗
`d_real_deescalation_threshold` (-0.15) was not mapped, so the d_real gate for de-escalation wasn't active.

## Fix Applied

Added 17 missing config mappings to `train_vits.py`:

```python
# Soft gates for de-escalation
soft_grad_limit_g=controller_cfg.get("soft_grad_limit_g", 2000.0),
soft_grad_limit_d=controller_cfg.get("soft_grad_limit_d", 2000.0),
# D-real leading indicator thresholds
d_real_warning_threshold=controller_cfg.get("d_real_warning_threshold", -0.10),
d_real_critical_threshold=controller_cfg.get("d_real_critical_threshold", -0.05),
d_real_emergency_threshold=controller_cfg.get("d_real_emergency_threshold", 0.0),
# ... (12 more fields)
```

## Metrics

- **Steps completed:** 2929 (50000 → 52929)
- **Duration:** ~45 minutes
- **Throughput:** 0.92 sec/step (65 steps/min)
- **GPU memory:** 98.6% (24.2/24.6 GB) — constrained
- **Inversions detected:** 1 (step 51140)
- **Inversions recovered:** 1

## Lessons Learned

1. **Config mapping is error-prone.** Every new GANControllerConfig field needs explicit mapping in train_vits.py. Consider automating this or using a different pattern.

2. **Detection worked, prevention didn't.** The d_real system successfully detected the inversion, but the soft_grad_limit gate that should have prevented the premature de-escalation wasn't configured correctly.

3. **D-freeze is effective.** Once triggered, D-freeze allowed the model to recover from an inverted state.

## Follow-up Actions

- [x] Fix config mapping in train_vits.py
- [x] Add print statements to verify config values at startup
- [ ] Restart training with batch_size=6 (GPU memory constrained)
- [ ] Verify soft_grad_limit=200 blocks premature de-escalation
- [ ] Consider adding config validation to catch mapping gaps

## Related

- Previous incident: [2026-01-31_multi_vits_gan_20260130_222741_grad_cascade.md](./2026-01-31_multi_vits_gan_20260130_222741_grad_cascade.md)
- Controller docs: [GAN_CONTROLLER.md](../training/GAN_CONTROLLER.md)
