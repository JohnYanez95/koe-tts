# Postmortem: Gradient Cascade at Step 55053

**Date**: 2026-01-31
**Run ID**: `multi_vits_gan_20260130_222741`
**Incident Type**: `numeric_instability`
**Severity**: Critical (emergency stop)

## Summary

Training hit emergency stop at step 55,053 due to G gradient exceeding the hard ceiling (5230 > 5000). This followed a rapid gradient cascade that escalated from normal operation to catastrophic failure in just 9 steps. The model had been operating in a persistently elevated state (EMA_G 400-1800) for thousands of steps before the final cascade.

## Timeline

### High-Level Progression

| Step Range | State | EMA_G Range | Notes |
|------------|-------|-------------|-------|
| 30000-44000 | Mostly healthy | 40-150 | Occasional L1 excursions |
| 44000-52000 | Deteriorating | 150-500 | More frequent L1/L2 oscillation |
| 52000-55000 | Chronic instability | 400-1800 | Never recovered to healthy |
| 55038-55053 | Cascade | 439→1807 | 10 steps from de-escalation to emergency |

### Final Cascade (Step-by-Step)

| Step | GN_G | GN_D | EMA_G | D_real | D_fake | Level | Event |
|------|------|------|-------|--------|--------|-------|-------|
| 55038 | - | - | - | - | - | L2→L1 | De-escalation |
| 55044 | 303 | 47.6 | 439 | -0.32 | -0.41 | L1 | Stable |
| 55045 | 435 | 34.8 | 439 | -0.34 | -0.41 | L1 | Rising |
| 55046 | **1988** | 23.5 | 593 | -0.26 | -0.34 | L1 | First major spike |
| 55047 | 112 | 54.9 | 545 | -0.32 | -0.38 | L1 | Brief recovery |
| 55048 | **2174** | 271.2 | 708 | **-0.11** | -0.28 | L1→L2 | NaN detected, D drifting |
| 55049 | **NaN** | 328.1 | 708 | **-0.03** | -0.22 | L2 | G step skipped |
| 55050 | **4433** | 278.4 | 1081 | **+0.01** | -0.22 | L2 | D_real goes positive |
| 55051 | 3238 | 194.4 | 1296 | **+0.16** | -0.22 | L2 | D confused |
| 55052 | 3544 | 82.0 | 1521 | +0.005 | -0.29 | L2 | Sustained explosion |
| 55053 | 4376 | 88.8 | 1807 | +0.02 | -0.23 | L2 | Last logged step |
| 55054 | **5230** | - | - | - | - | EMERGENCY | Hard ceiling hit |

### Events Timeline

```
55038: L2 → L1 (de-escalation after stability)
55048: L1 → L2 (nan_inf_detected)
55053: unstable → emergency (nan_inf_detected)
55054: grad_g_exceeded_5230_limit_5000
```

## Root Cause Analysis

### Primary Cause: Discriminator Score Drift

The most telling signal is `d_real_score` transitioning from negative to positive:

```
Step 55044: d_real = -0.32 (healthy: D thinks real is real)
Step 55048: d_real = -0.11 (drifting)
Step 55049: d_real = -0.03 (near zero)
Step 55050: d_real = +0.01 (INVERTED: D thinks real is fake)
Step 55051: d_real = +0.16 (D completely confused)
```

When D starts misclassifying real samples as fake, it provides wrong gradients to G. G receives signals saying "your fake outputs are more realistic than real data," causing gradient explosion.

### Secondary Cause: Insufficient Recovery Time

The de-escalation at step 55038 came just 10 steps before the cascade:
- L2 → L1 at 55038
- L1 → L2 at 55048 (only 10 steps later)

The `stability_required_steps_l1 = 200` threshold was met, but the underlying EMA_G (439) was still elevated. The model de-escalated based on surface stability while chronic instability persisted.

### Contributing Factors

1. **Chronic elevated EMA_G**: The model operated with EMA_G of 400-1800 for 3000+ steps (52k-55k), never truly recovering to healthy levels (< 100).

2. **L2 mitigations insufficient**: At L2, mitigations are:
   - `grad_clip_scale = 0.25`
   - `lr_scale = 0.5`
   - `d_freeze = false` (only activates at L3)

   These weren't aggressive enough to stop the cascade.

3. **Fast cascade outpaced escalation**: The gradient went 303 → 5230 in 10 steps. The controller escalated to L2 but the cascade was already in progress. It never reached L3 (d_freeze) which might have helped.

4. **D gradient correlation**: Note that D gradients also spiked (271, 328, 278, 194, 82, 88) during the cascade. Both networks were destabilizing together.

## Metrics at Failure

From emergency stop output:
```
G loss: N/A (already diverged)
D loss: N/A
Mel loss: 1.1981
KL loss: 1.8760
G grad norm: 5230.1
D grad norm: 43.9
Escalation level: L2
Consecutive NaN/Inf: 0
```

## What Worked

1. **Level-based mitigations**: Got us through 31k (KL explosion) and 35k (previous danger band)
2. **EMA tracking**: Correctly identified elevated state
3. **Emergency stop**: Caught the failure before complete divergence
4. **Checkpoints**: 50k checkpoint saved, only lost 5k steps

## What Didn't Work

1. **De-escalation too aggressive**: 200 stable steps at L1 wasn't enough when EMA_G was still 439
2. **L2 mitigations too weak**: No d_freeze meant D could continue destabilizing G
3. **No D-score monitoring**: The d_real_score drift was a leading indicator that wasn't being used for escalation

## Recommendations

### Immediate (Next Run)

1. **Don't de-escalate with elevated EMA**: Add check `ema_grad_g < 200` before allowing de-escalation
2. **Enable d_freeze at L2**: Move discriminator freeze from L3 to L2

### Short-term

1. **d_real drift detection** (replaces gap-based detection—see Distribution Analysis):
   - Warning: `d_real > -0.10` for 3 consecutive steps
   - Critical: `d_real > -0.05` for 2 consecutive steps
   - Emergency: `d_real > 0.00` for 2 consecutive steps (D inversion)
2. **Longer stability windows**: Increase `stability_required_steps_l1` from 200 to 500

### Medium-term

1. **Investigate root cause of chronic instability**: Why did EMA_G stay elevated for 3000+ steps?
2. **Consider lower hard ceiling**: 5000 might be too high if recoverable range is < 500
3. **Gradient history tracking**: Log last N gradients to detect acceleration patterns

## Checkpoints

| Checkpoint | Step | Status |
|------------|------|--------|
| step_040000.pt | 40000 | Clean |
| step_050000.pt | 50000 | Clean (recommended resume) |
| step_055000.pt | 55000 | Unstable (EMA_G elevated) |
| step_055053_emergency.pt | 55053 | Post-cascade |

**Recommended resume point**: `step_050000.pt`

## Distribution Analysis

See [analysis/multi_vits_gan_20260130_222741/](../../analysis/multi_vits_gan_20260130_222741/) for detailed visualizations:

- `d_real_distributions.png` - Key signal: d_real approaching zero
- `threshold_analysis.png` - Threshold selection with false positive rates
- `cascade_signature.png` - The 55k cascade step-by-step
- `ema_distributions.png` - EMA_G chronic elevation patterns
- `grad_distributions.png` - Gradient norm distributions by level
- `gan_controller_distributions.ipynb` - Full analysis notebook

### Key Observations from Distributions

1. **d_real is the leading indicator** (see cascade_signature.png)
   - d_real drifts toward 0 several steps before G gradient explodes
   - G gradient is a lagging indicator - by the time it spikes, it's too late
   - EMA_G is too slow to catch rapid cascades

2. **The gap (d_real - d_fake) is NOT useful**
   - Gap actually *grows* during the cascade (from ~0.09 to ~0.23)
   - This is because d_real approaches zero while d_fake stays around -0.22
   - The original recommendation for gap < 0.1 detection would be noisy

3. **Chronic instability visible in EMA_G** (see ema_distributions.png)
   - 50-55k band has P50 EMA_G of ~500 vs ~100 for 30-40k
   - The model never recovered to healthy baseline after 44k

### Recommended d_real Thresholds

Based on distribution analysis of 25,053 training steps:

| Alarm Level | Threshold | Hysteresis | False Positives (L0) | Action |
|-------------|-----------|------------|---------------------|--------|
| Warning | d_real > -0.10 | 3 steps | ~7 runs | Log, increase monitoring |
| Critical | d_real > -0.05 | 2 steps | ~3 runs | Escalate, prep D-freeze |
| Emergency | d_real > 0.00 | 2 steps | ~1 run | Immediate D-freeze or stop |

### False Positive Impact Analysis

**What false positives cost:**
- Each false escalation triggers mitigations (LR reduction, grad clip tightening)
- At L2: `lr_scale = 0.5`, `grad_clip_scale = 0.25`
- Training at 50% LR converges ~2× slower during mitigation window
- If stability threshold is 200 steps at reduced LR, each false positive costs ~100 steps of effective training

**Quantified impact for d_real > -0.10 with 3-step hysteresis:**
- ~47 total runs of 3+ steps across 25k training steps
- ~7 runs during healthy L0 operation (false positives)
- Estimated training slowdown: 7 × 100 = ~700 effective steps lost
- This is ~2.8% overhead on a 25k step run

**Why this is acceptable:**
- The 55k cascade cost 5,000 steps (had to resume from 50k checkpoint)
- 7 false escalations × 100 steps = 700 steps of slowdown
- Break-even: If d_real monitoring prevents even 1 cascade per 7 runs, it's net positive
- Given the cascade frequency (3 emergencies in ~55k steps), monitoring pays for itself

**Risk of cascade into higher levels:**
- False positives at Warning level are contained (no permanent state change)
- Escalation to L2 requires sustained instability, not single-step triggers
- The 3-step hysteresis prevents noisy single-step events from cascading
- Even if a false positive reaches L2, it will de-escalate after stability window
- True cascades show sustained drift (5-10 steps), distinct from noise

## Appendix: Full Event Log (52k-55k)

```
52012: healthy, L0 (brief recovery)
52062: L0 → L1 (hard_clip_g_50_steps)
52265: L1 → L2 (ema_g_elevated_782)
52765: L2 → L1
52870: L1 → L2 (ema_g_elevated_1155)
53837: L2 → L1
53847: L1 → L2 (nan_inf_detected)
54353: L2 → L1
54459: L1 → L2 (ema_g_elevated_1621)
55038: L2 → L1
55048: L1 → L2 (nan_inf_detected)
55053: EMERGENCY (grad_g_exceeded_5230)
```

Pattern: Constant L1↔L2 oscillation with increasing EMA thresholds (782 → 1155 → 1621), indicating progressive deterioration that never truly recovered.
