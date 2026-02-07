# Postmortem: KL Loss Explosion at Step 31049

**Date**: 2026-01-30
**Run ID**: `multi_vits_gan_20260130_095936`
**Incident Type**: `numeric_instability`
**Severity**: Critical (emergency stop)

## Summary

Training resumed from step 30000 (checkpoint from run `20260129_212937`) hit a catastrophic KL loss explosion at step 31049. KL loss jumped from ~1.5 to 2,153,485 (1.4 million × increase) in a single step, causing NaN propagation and emergency stop.

## Timeline

| Step | Event | Details |
|------|-------|---------|
| 30000 | Resume | Controller state restored, healthy |
| 30325 | Checkpoint | `early` checkpoint saved |
| 30546 | L0→L1 | `median_clip_g_0.020` |
| 31037 | L1→L2 | `nan_inf_detected` (first NaN) |
| 31048 | L2→L3 | `nan_inf_detected`, saved `last_known_good` |
| 31049 | KL explosion | `g_loss_kl: 2,153,485`, all losses → NaN |
| 31050 | L3→L4 | `nan_inf_detected` → emergency stop |

Total training time: 958 seconds (~16 minutes)

## Metrics Before Failure

Step 31048 (last healthy):
```
g_grad_norm: 20.68
d_grad_norm: 15.41
g_loss_kl: 1.67
g_loss_mel: 1.15
loss_g: 65.45
loss_d: 15.39
ema_grad_g: 52.5
ema_grad_d: 21.5
```

Step 31049 (explosion):
```
g_loss_kl: 2,153,485  ← EXPLOSION
g_loss_mel: NaN
g_loss_adv: NaN
g_loss_fm: NaN
loss_g: NaN
loss_d: NaN
g_grad_norm: null (skipped)
```

## Root Cause Analysis

### Immediate Cause
Single-step KL loss explosion from ~1.5 to 2.1M. This indicates the VAE posterior distribution became degenerate - either:
1. Posterior variance collapsed to near-zero (log-var → -∞)
2. Posterior mean exploded
3. Prior/posterior mismatch became extreme

### Contributing Factors

1. **Model already unstable**: First NaN detected at step 31037, 12 steps before the catastrophic explosion
2. **Rapid escalation**: L1→L2→L3→L4 in just 13 steps (31037→31050)
3. **Possible bad batch**: Metrics looked healthy at 31048, suggesting a specific batch triggered the collapse

### Why Mitigations Didn't Help

The level-based mitigations (D freeze, LR reduction, clip scaling) are designed for **gradual** instability, not **instantaneous** catastrophic failure. When KL explodes 1.4M× in one step, no mitigation can recover.

## Data Analysis

Comparing to previous run's 35k danger band:
- Previous failure (run 20260129_212937): Step 35342, gradual escalation over ~300 steps
- This failure: Step 31049, rapid escalation over ~13 steps

This suggests either:
1. Different failure mode (VAE collapse vs gradient instability)
2. Bad batch specific to this shuffle order
3. Accumulated numerical precision issues

## Recommendations

### Short-term
1. **KL clamping**: Add `torch.clamp(kl_loss, max=100)` to prevent single-step explosions
2. **KL monitoring**: Track KL loss EMA separately, escalate on sudden spikes (>10× EMA)
3. **Batch-level validation**: Log batch statistics when NaN first detected

### Medium-term
1. **KL annealing**: Gradually increase KL weight during unstable periods
2. **Posterior regularization**: Add minimum variance constraint to VAE encoder
3. **Checkpoint on first NaN**: Save checkpoint immediately when first NaN detected (not just at L3)

### Investigation Needed
1. Examine the specific batch at step 31049 - what audio samples were in it?
2. Check if batch contains unusual characteristics (very short/long, silence, noise)
3. Profile VAE encoder outputs at steps 31047-31049 to trace collapse origin

## Affected Files

- Checkpoint: `runs/multi_vits_gan_20260130_095936/checkpoints/step_031050_emergency.pt`
- Last known good: `runs/multi_vits_gan_20260130_095936/checkpoints/step_031048_last_known_good.pt`
- Metrics: `runs/multi_vits_gan_20260130_095936/train/sessions/20260130_095936/metrics.jsonl`
- Events: `runs/multi_vits_gan_20260130_095936/train/sessions/20260130_095936/events.jsonl`

## Resolution Status

- [x] Emergency stop worked correctly
- [x] Last known good checkpoint saved
- [x] Console output improved to show state changes with step numbers
- [ ] KL clamping not yet implemented
- [ ] Root cause (specific batch) not yet identified
