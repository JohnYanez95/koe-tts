# GAN Stability Log

Rolling summary of stability incidents, trends, and mitigation effectiveness.

---

## Incident Summary (chronological)

| Date | Run | Steps | Trigger | Outcome | Lesson |
|------|-----|-------|---------|---------|--------|
| 2026-01-26 | [20260126_041738](./2026-01-26_multi_vits_gan_20260126_041738_nan_collapse.md) | ~10k | NaN collapse | Emergency | P0/P1 escalation ladder shipped |
| 2026-01-27 | [resume27k](./2026-01-27_multi_vits_gan_resume27k_bsod_lost.md) | 27.4k–28.2k | BSOD (infra) | Lost | Checkpoint-on-resume shipped |
| 2026-01-29 | [20260128_210613](./2026-01-29_multi_vits_gan_20260128_210613_grad_ceiling_emergency.md) | 28k–30.8k | Grad ceiling (5272) | Emergency | Spike density + EMA triggers shipped |
| 2026-01-29 | [CUDA crash (KVM)](./2026-01-29_cuda_unknown_error_kvm.md) | 15k–19.6k | GPU inaccessible | Infra crash | KVM switch + outdated driver (576.28 → 591.86) |
| 2026-01-29 | [Trigger miscalibration](./2026-01-29_poisoned_checkpoint_resumes.md) | 15k–30k | Escalation (clip-coef) | 6/6 failed → **RESOLVED** | Config wasn't being read from YAML; fixed in train_vits.py |
| 2026-01-30 | [20260129_200158](./2026-01-30_multi_vits_gan_20260129_200158_clip_coef_feedback_loop.md) | 24k–25.1k | **Clip-coef feedback loop** | Emergency (healthy grads) | Controller mitigation triggers its own alarm → **FIXED** |
| 2026-01-30 | [20260129_212937](./2026-01-30_multi_vits_gan_20260129_212937_mitigation_expiry_gap.md) | 34.7k–35.3k | **Mitigation expiry gap** | Emergency (mel improving) | Mitigations expired before level decay → **FIXED** |
| 2026-01-30 | [20260130_095936](./2026-01-30_multi_vits_gan_20260130_095936_kl_explosion.md) | 31k | **KL explosion** | Emergency (sudden) | KL 1.5→2.1M in one step; bad batch suspected |
| 2026-01-31 | [20260130_222741](./2026-01-31_multi_vits_gan_20260130_222741_grad_cascade.md) | 55k | **Gradient cascade** | Emergency (D drift) | D score inversion → G explosion; de-escalation too aggressive |
| 2026-02-02 | [20260201_183758](./2026-02-02_multi_vits_gan_20260201_183758_observation_mode_collapse.md) | 48k, 79k | **Segment cropping misalignment** | Emergency (NaN) | Phoneme/audio misalignment from random crops → **ROOT CAUSE OF GRADIENT EXPLOSION** |

---

## Critical Finding: Segment Cropping Bug

**All gradient explosion incidents after step 35k were likely caused by a data pipeline bug, not GAN instability.**

The `segment_seconds: 3.0` config randomly cropped audio but used phonemes from the full utterance, creating impossible learning objectives. This explains:
- Why g_grad_norm grew from ~30 (healthy) to 10k-100k (pathological)
- Why g_clip_coef dropped from ~0.05 (healthy) to ~0.00002 (frozen)
- Why interventions only delayed failure rather than preventing it
- Why val_loss was stuck at 3.2216 (model couldn't learn correct alignments)

**Fix:** `segment_seconds: null` in vits_core.yaml and vits_gan.yaml. Train on full utterances until forced alignment is implemented.

---

## Danger Bands Identified

| Step Range | Symptom | Status | Confidence | Lead Indicator |
|------------|---------|--------|------------|----------------|
| 10k–12k | Post-disc_start instability | Stabilized with adv_ramp | High | g_grad spikes |
| 27k–30k | KL spikes, grad oscillation | Partially mitigated (lr_d, adv_weight) | Med | KL > 3, g_grad volatility |
| 30k+ | D confusion → G explosion | New detection shipped, untested | Low | D scores drift toward 0, ema_grad_g rising |
| 31k | KL explosion (VAE collapse) | Not mitigated | Med | KL loss sudden spike (>10× in one step) |
| 52k–55k | Chronic instability → cascade | **CONFIRMED** | High | d_real_score > 0 (D inversion), EMA_G elevated |

---

## Mitigation Effectiveness

### Hyperparameter tuning

| Change | When | Effect |
|--------|------|--------|
| lr_d: 1e-4 → 5e-5 | 2026-01-28 | Helped survive 27k–30k |
| adv_weight: 1.0 → 0.5 | 2026-01-28 | Reduced G gradient pressure |
| adv_ramp_steps: 3000 → 5000 | 2026-01-28 | Smoother disc activation |

### Controller triggers

| Trigger | Shipped | Fires on | Tested? |
|---------|---------|----------|---------|
| NaN/Inf consecutive (3) | 2026-01-26 | NaN collapse | Yes |
| Consecutive spikes (3) | 2026-01-28 | Sustained instability | Yes (false negatives on bursty) |
| Spike density (3 in 20) | 2026-01-29 | Bursty instability | Yes (worked on poisoned resumes) |
| EMA elevated (>500 for 50) | 2026-01-29 | Gradual escalation | Yes (worked on poisoned resumes) |
| Clip-coef hard (<0.05 for 30) | 2026-01-29 | Sustained clipping | ✅ Fixed (was causing feedback loop) |
| Hard ceiling (5000→3000) | 2026-01-29 | Last resort | Yes |

### ✅ Clip-Coef Feedback Loop — FIXED (2026-01-30)

The clip-coef hard trigger had a **design flaw**: when the controller tightens `grad_clip_scale` as a mitigation, the `clip_coef` naturally drops. This caused the clip-coef alarm to fire, triggering further escalation, in a self-reinforcing loop.

**Fix:** Clip-coef checks are now **disabled** when `grad_clip_scale < 1.0`. Low clip_coef is expected when clipping is intentionally tightened.

See: [Full postmortem](./2026-01-30_multi_vits_gan_20260129_200158_clip_coef_feedback_loop.md)

---

## Open Questions

1. ~~**How far back does the instability extend?**~~
   - **Answered:** It doesn't. The checkpoints were fine; config wasn't being read.

2. ~~**Are triggers too aggressive, or is model genuinely unstable?**~~
   - **Answered:** Config bug — YAML values weren't being read.

3. ~~**What is normal clip_coef during healthy GAN training?**~~
   - **Answered:** g_clip_coef ~ 0.03-0.07, d_clip_coef ~ 0.06-0.10

4. ~~**What are appropriate clip-coef thresholds?**~~
   - **Answered:** The 0.05 threshold was in the middle of normal.
   - Recommendation: `clip_coef_hard_threshold: 0.01-0.02`

5. **Will training survive the 27k-30k danger band with corrected config?**
   - Need longer test to validate

6. ~~**Why did training die at 35k with healthy metrics?**~~
   - **Answered:** Mitigation expiry gap — mitigations expired (timer-based) but escalation level persisted. Clip-coef triggers re-fired immediately.
   - **Fix:** Mitigations now tied to escalation level, not timers.

7. **Should de-escalation require EMA_G to be healthy?**
   - Run 20260130_222741 de-escalated at step 55038 with EMA_G=439, then cascaded 10 steps later.
   - Recommendation: Don't de-escalate unless `ema_grad_g < 200`.

8. **Should D-score inversion trigger emergency?**
   - When `d_real_score > 0`, D is misclassifying real samples as fake.
   - This was a leading indicator in the 55k cascade (d_real went -0.32 → +0.16).

---

## Metrics to Track

For each run, log:
- Hard ceiling triggers (should decrease)
- Spike density triggers (should increase if catching early)
- EMA elevated triggers (should catch gradual escalation)
- AMP skip frequency (g_step_skipped rate)
- Clip coefficient distribution (how often g_clip_coef << 1)
- **First trigger step vs emergency step** (time-to-react margin)
  - Goal: early triggers fire 50+ steps before hard ceiling would
  - Track: `(emergency_step - first_unstable_step)` for each incident

---

## Next Steps

### Completed
1. ~~**Stress test:** Resume from 30000, run to 35000 with new detection~~ — **Failed** (config bug)
2. ~~**Test step_028000_resume_start.pt**~~ — **Failed** (config bug)
3. ~~**Adjust hyperparams:** adv_weight → 0.25, lr_g → 1e-4~~ — **Failed** (config bug)
4. ~~**Fresh start from VITS core**~~ — **Failed** (config bug)
5. ~~**Fix config bug in train_vits.py**~~ — **Done** (YAML values now read)
6. ~~**Establish baseline**~~ — **Done** (normal g_clip_coef is 0.03-0.07)
7. ~~**Calibrate thresholds**~~ — **Done** (clip_coef: 0.01/50, spike_density: 5, ema: 100)
8. ~~**Run longer test**~~ — **Interrupted** (GPU crash at 19.6k, zero alarms)
9. ~~**Resume from step 18k** and complete test through 27k-30k danger band~~ — Passed, reached 55k
10. ~~**Disable segment cropping**~~ — **Done** (`segment_seconds: null` in vits_core.yaml, vits_gan.yaml)

### Priority: Data Pipeline
11. **Implement length-based bucket sampling** — Required for efficient full-utterance training
12. **Labeling app for forced alignment** — Required for proper segment training (backlog)

### Priority: Training
13. **Fresh training run from scratch** — All existing checkpoints are poisoned by misaligned data
14. **Validate healthy gradient norms** — Expect g_grad_norm ~20-40, g_clip_coef ~0.03-0.07

### Lower Priority (GAN Controller)
15. **Update NVIDIA driver** (576.28 → 591.86 WHQL)
16. **Add D-score inversion detection**: Trigger alarm when `d_real_score > 0`
17. **Block de-escalation with elevated EMA**: Require `ema_grad_g < 200` to de-escalate

---

## Infrastructure Issues

| Date | Issue | Impact | Fix |
|------|-------|--------|-----|
| 2026-01-27 | BSOD (driver crash) | Lost run 27.4k-28.2k | Checkpoint-on-resume |
| 2026-01-29 | CUDA crash (KVM switch) | Lost 1.6k steps (18k-19.6k) | Update driver, avoid KVM during training |

**Driver status:** 576.28 (outdated) — recommend updating to 591.86 WHQL
