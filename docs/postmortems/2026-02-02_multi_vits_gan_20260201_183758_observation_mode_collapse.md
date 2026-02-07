# Postmortem: Segment Cropping Misalignment + D Dominance Collapse

**Run ID:** `multi_vits_gan_20260201_183758` (and follow-up `multi_vits_gan_20260202_092438`)
**Date:** 2026-02-01 to 2026-02-02
**Incident Type:** `training_collapse`, `data_pipeline_bug`
**Severity:** Training failure at step 48445 (initial), step 79875 (follow-up)

## Summary

Initial investigation focused on "observation mode" experiment where GAN controller mitigations were disabled. Training failed at step 48445 due to apparent D dominance. Follow-up run with asymmetric D clipping (`grad_clip_d: 0.5`) extended training to step 79875 before also failing.

**Root cause:** A fundamental data pipeline bug where `segment_seconds: 3.0` randomly cropped audio but used phonemes from the full utterance, creating impossible learning objectives. The model was trained on misaligned phoneme/audio pairs from the start, causing progressive gradient explosion that manifested as apparent GAN instability.

## Hypothesis Under Test

**Question:** Are the escalation-based mitigations (gradient clipping, LR scaling, D freeze) actually necessary, or do they slow learning by dampening G's natural response to D improvements?

**Rationale:** Original VITS paper didn't use aggressive intervention systems. Our controller machinery might be over-engineered based on early instabilities that were actually data/config issues.

## Configuration

```yaml
controller:
  escalation_enabled: false      # No L0→L1→L2→L3 escalation
  hard_ceiling_enabled: false    # No emergency stop on high gradients
  absolute_grad_limit_g: 10000   # Very high ceiling (effectively disabled)
  absolute_grad_limit_d: 10000
  grad_clip_scales: [1.0, 1.0, 1.0]  # No clip tightening
  lr_scale_factor: 1.0           # No LR reduction

optim:
  grad_clip_g: 1.0               # Base clipping only
  grad_clip_d: 1.0               # Base clipping only (same as G)
```

## Timeline

| Step | Event | Metrics |
|------|-------|---------|
| 27075 | Resume from checkpoint | mel=1.09, D=15, healthy |
| 30000-36000 | Turbulent but productive | mel=1.0-1.3, gradients 1k-6k, D separation occurring |
| 36400 | Massive G gradient spike | GN_G=6,669, mel held at 1.12 |
| 37134 | D achieves sharp separation | d_fake briefly > d_real, FM spike |
| 38000 | G counterattack | mel recovered 1.6→1.25, D confused |
| 39000 | Continued battle | GN_G=26,497, mel=1.25 |
| 45000 | Checkpoint saved | mel=1.14, D=15.0, stable |
| 47800 | Last healthy state | mel=1.11, D=16.1, KL=1.4 |
| 48000 | **D wins decisively** | mel=1.90, D=4.9, KL=1.1 |
| 48200 | G struggling | mel=1.77, D=6.9 |
| 48400 | G collapse begins | mel=2.60, D=6.0, KL=1.6 |
| 48443 | KL explosion | KL=24,351, posterior destabilizing |
| 48445 | NaN cascade | KL=4.5M, training stops |

## Root Cause Analysis

### What Happened

1. **D achieved dominance** (D loss dropped 16→5) around step 48000
2. **G couldn't adapt** - gradients were high but ineffective against confident D
3. **Mel spiked** (1.1→2.6) as G output quality degraded
4. **KL exploded** as posterior encoder couldn't match diverging G output
5. **NaN cascade** ended training

### Why D Won

With both G and D using the same `grad_clip: 1.0`:
- D's task is simpler (binary classification)
- D converges faster than G (reconstruction + fooling D)
- Once D becomes confident, G gradients become less informative
- D's low gradients (70-90) indicated confidence, not learning
- G's high gradients (10k-75k) indicated desperation, not progress

### Key Insight (Initial Analysis)

The collapse wasn't caused by:
- ❌ d_real going "near perfect" (was only 0.85)
- ❌ High gradients directly (G had 75k spikes and recovered)
- ❌ KL initially (was normal at 1.1 when D won)

It WAS caused by:
- ✅ D loss dropping below ~8 (D too confident)
- ✅ Asymmetric learning rates (same clip for unequal tasks)
- ✅ No mechanism to slow D when it's winning

### Deeper Root Cause (Follow-up Analysis)

After the follow-up run (`multi_vits_gan_20260202_092438`) also failed at step 79875 despite asymmetric clipping, further investigation revealed a **fundamental data pipeline bug**:

**The segment cropping implementation was causing phoneme/audio misalignment.**

With `segment_seconds: 3.0`, the dataset randomly crops 3-second audio segments, but uses phonemes from the **full utterance**:

```python
# Audio: randomly cropped to 3 seconds
waveform = waveform[start : start + self.segment_samples]

# Phonemes: from FULL utterance (not the cropped segment!)
phoneme_str = item.get("phonemes_span") or item.get("phonemes", "")

# Duration: uniform distribution of ALL phonemes over cropped frames
durations = compute_uniform_durations(n_tokens, n_frames)  # MISMATCH
```

**Example of the mismatch:**
- Full utterance: 10 seconds, 100 phonemes
- Cropped audio: random 3-second slice (e.g., seconds 4-7)
- Phonemes used: all 100 (for full 10 seconds)
- Model asked to learn: "these 100 phonemes correspond to this random 3-second clip"

This is an **impossible learning objective**. The model was being trained on contradictory data from the start.

**Why it appeared to work early (steps 27k-32k):**
- Adversarial loss was still ramping up (`disc_start_step: 10000`, `adv_ramp_steps: 5000`)
- Model was mostly learning reconstruction, where misalignment hurts less
- Once GAN training fully engaged, the misalignment became catastrophic

**Gradient explosion timeline:**

| Step | g_grad_norm | Status |
|------|-------------|--------|
| 27k | 17 | Healthy (baseline) |
| 32k | 48 | Healthy |
| 35k | **888** | 20x jump - instability begins |
| 37k | 1,310 | Growing |
| 39k | 4,171 | Growing fast |
| 41k | **24,692** | Exploded |
| 45k+ | 10k-100k | Chaotic |

The checkpoint at step 45k was already "poisoned" with confused weights from training on misaligned data.

**Why `grad_clip_g: 1.0` appeared too aggressive:**
- With healthy gradients (~30), clip_coef ≈ 0.03-0.05 (normal)
- With exploded gradients (~50,000), clip_coef ≈ 0.00002 (G effectively frozen)
- The clip threshold was correct; the gradients were pathologically large

**The fix:** Disable segment cropping until forced alignment is implemented.

```yaml
data:
  segment_seconds: null  # Train on full utterances
```

Original VITS trains on full utterances with bucket-based length sampling. Random segment cropping requires phoneme-level timestamps to slice both audio AND phonemes together.

## Observations

### Positive Findings

1. **High gradients are often productive** - G survived 6k, 13k, 26k, even 75k spikes and recovered
2. **Mel is resilient** - spiked to 1.6, recovered to 1.25 through pure adaptation
3. **Adversarial oscillation is normal** - the "turbulence" was G and D pushing each other
4. **Original VITS approach mostly works** - 20k+ steps of healthy training with minimal intervention

### Negative Findings

1. **D can win decisively** without external limits
2. **G collapse is sudden** - healthy→dead in ~500 steps
3. **KL explosion is a symptom**, not the cause - posterior fails when G fails
4. **Equal treatment isn't fair** - D's simpler task needs handicapping

## Resolution

Resumed from step 45000 with asymmetric clipping:

```yaml
optim:
  grad_clip_g: 1.0    # G can respond fully
  grad_clip_d: 0.5    # D learns slower, giving G room
```

This preserves G's ability to adapt quickly while preventing D from running away.

## Lessons Learned

1. **Observation mode was valuable** - revealed the true failure mode (D dominance, not gradient explosion)
2. **Less intervention is often better** - but not zero intervention
3. **Asymmetry is key** - G and D need different treatment
4. **Watch D loss, not just gradients** - D loss < 8 is the danger signal
5. **KL is a lagging indicator** - by the time KL spikes, it's too late
6. **Data pipeline bugs can masquerade as model instability** - the gradient explosion wasn't a GAN problem, it was a data alignment problem
7. **Validate assumptions against literature** - original VITS uses full utterances, not random crops; our "standard approach" comment was incorrect
8. **Constant clipping is a red flag** - healthy training has g_clip_coef ~0.03-0.07, not 0.00002

## Recommendations

### Immediate (Data Pipeline)
- **Disable segment cropping** - set `segment_seconds: null` in VITS configs
- **Train on full utterances** - matches original VITS approach
- **Implement length-based bucket sampling** - group similar-duration utterances for efficient batching

### Immediate (GAN Training)
- Use `grad_clip_d: 0.5` (half of G) as default
- Monitor D loss, alert if sustained < 8
- Start fresh from pre-35k checkpoint or from core stage

### Future Investigation
- **Labeling app for forced alignment** - required for proper segment training
- Test D throttling (skip D updates) instead of/in addition to clip reduction
- Consider adaptive D learning rate based on D loss

## Metrics for Success (Next Run)

- [ ] g_grad_norm stays in healthy range (20-40)
- [ ] g_clip_coef stays in healthy range (0.03-0.07)
- [ ] Survive the 48k region where previous run collapsed
- [ ] D loss stays above 8 during separation
- [ ] Mel trends down over 10k steps
- [ ] No KL explosion
- [ ] val_loss improves from baseline (was stuck at 3.2216)

## Related

- Original VITS paper: No aggressive intervention, trains on full utterances
- Original VITS implementation: [jaywalnut310/vits](https://github.com/jaywalnut310/vits) - uses `DistributedBucketSampler`, no random segment cropping
- Follow-up run: `multi_vits_gan_20260202_092438` - failed at step 79875, led to discovery of data alignment bug
- Previous postmortem: `2026-01-30_clip_coef_feedback_loop.md` (over-intervention)
- GAN stability log: `gan_stability_log.md`

## Config Changes Made

```yaml
# configs/training/vits_gan.yaml and vits_core.yaml
data:
  segment_seconds: null  # Was 3.0 - disabled to fix alignment
  # TODO: Implement length-based bucket sampling
```
