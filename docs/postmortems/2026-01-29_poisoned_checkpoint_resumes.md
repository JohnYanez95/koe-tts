# Postmortem: Clip Coefficient Trigger Too Aggressive

*Originally titled "Poisoned Checkpoint Resumes" — diagnosis updated after root cause identified*

## Incident Summary

| Field | Value |
|-------|-------|
| Incident Type | `trigger_miscalibration` |
| **Resolution** | **Config values weren't being read from YAML** |
| Date | 2026-01-29 |
| Affected Run | `multi_vits_gan_20260128_210613` |
| Checkpoint Range | 28,294 → 30,000 |
| Outcome | **All resume attempts failed** — 3/3 hit emergency stop |
| Detection | Controller escalation ladder (new triggers worked) |

---

## Timeline of Attempts

| Checkpoint | Run ID | Steps Survived | Hyperparams | Emergency Step |
|------------|--------|----------------|-------------|----------------|
| `step_030000.pt` | `multi_vits_gan_20260129_020503` | 59 | default | 30059 |
| `step_029000.pt` | `multi_vits_gan_20260129_resume29k` | 51 | default | 29051 |
| `step_028294_early.pt` | `multi_vits_gan_20260129_resume28k` | 68 | default | 28362 |
| `step_028000_resume_start.pt` | `multi_vits_gan_20260129_resume28000` | 72 | default | 28072 |
| `step_028000_resume_start.pt` | `multi_vits_gan_20260129_resume28000_lowlr` | 51 | adv=0.25, lr_g=1e-4 | 28051 |
| `vits_core/final.pt` | `multi_vits_gan_20260129_fresh` | **105** | adv=0.25, lr_g=1e-4 | 15105 |

**Pattern:** All six show immediate hard-clipping, rapid escalation through all 4 controller levels, and emergency stop.

**Critical finding:** The fresh start from VITS core (step 15000) — a known-good checkpoint before any GAN instability — also died in 105 steps. This proves the **triggers are too aggressive**, not the checkpoints.

---

## Evidence

### Clip Coefficients at Failure

**30k resume (worst):**
| Step | g_clip_coef | g_grad_norm | hard_clip_steps_g |
|------|-------------|-------------|-------------------|
| 30055 | 0.0024 | 42.5 | 25 |
| 30056 | 0.0012 | 84.2 | 26 |
| 30057 | 0.0013 | 76.3 | 27 |
| 30058 | 0.0013 | 77.5 | 28 |
| 30059 | 0.0015 | 68.6 | 29 |

**29k resume:**
| Step | g_clip_coef | g_grad_norm | hard_clip_steps_g |
|------|-------------|-------------|-------------------|
| 29049 | 0.0153 | 16.3 | 19 |
| 29050 | 0.0088 | 28.3 | 20 |
| 29051 | 0.0058 | 17.2 | 21 |

**28k resume:**
| Step | g_clip_coef | ctrl_escalation_level | hard_clip_steps_g |
|------|-------------|----------------------|-------------------|
| 28362 | 0.0105 | 3 | 29 |

### Escalation Ladder (typical pattern)

| Step Offset | Event | Level |
|-------------|-------|-------|
| +0 | Resume (healthy) | 0 |
| +15–30 | UNSTABLE | 1 → clip tightening |
| +30–45 | UNSTABLE | 2 → LR reduction |
| +45–55 | UNSTABLE | 3 → D freeze, save last_known_good |
| +50–70 | UNSTABLE | 4 → EMERGENCY |

---

## Root Cause

**~~Poisoned checkpoints~~** → **Triggers too aggressive**

Initial diagnosis was that checkpoints were "poisoned" — model weights in a bad basin. However, when a fresh start from VITS core (step 15000, known-good) also died in 105 steps, this diagnosis was proven wrong.

### Actual Root Cause

The **clip coefficient trigger** (`g_clip_coef < 0.05 for 30 steps`) is interpreting normal GAN training behavior as catastrophic instability.

- Normal early GAN training has high gradients as D and G find equilibrium
- Gradient clipping is **expected behavior**, not a sign of failure
- The original run (which reached 30k+) would have had similar clip coefficients but didn't have this trigger
- A clip_coef of 0.05 means grad_norm is 20x the clip value — aggressive but not catastrophic

### Why It Looked Like Poisoned Checkpoints

- All GAN-stage checkpoints (28k-30k) failed immediately
- Gradients were being 99% clipped from step 1
- Easy to conclude "weights are in a bad basin"

But the VITS core checkpoint (pre-GAN, step 15000) also failed immediately — proving the issue is the detection system, not the weights.

---

## Key Insight

**Controller state ≠ model health.** A checkpoint can restore with "healthy" alarm state but have poisoned model weights. Sustained `g_clip_coef < 0.01` is a strong signal that a checkpoint is unsafe.

---

## What Worked

1. **Clip-coef trigger detected the problem** — `hard_clip_steps_g` was 25–29/30 when emergency fired
2. **Escalation ladder worked** — Escalated through levels with appropriate mitigations
3. **last_known_good checkpoints saved** at level 3
4. **Emergency stop prevented worse damage**

---

## Time-to-React Analysis

| Run | First UNSTABLE | Emergency | Margin |
|-----|---------------|-----------|--------|
| 30k resume | 30029 | 30059 | 30 steps |
| 29k resume | ~29020 | 29051 | ~31 steps |
| 28k resume | ~28310 | 28362 | ~52 steps |

The escalation ladder consumed all margin before mitigations could take effect — suggesting either:
- Triggers are too aggressive for normal GAN training
- The model is genuinely poisoned and no mitigation would help

---

## Open Questions

1. **Did the original run have similar clip coefficients?** We don't have clip_coef data from before the BSOD — if the original run also had g_clip_coef ~0.01 and survived, triggers may be too sensitive.

2. ~~**Is step_028000 also poisoned?**~~ **Yes.** Died in 72 steps. Instability predates 28k.

---

## Options

| Option | Risk | Reward | Status |
|--------|------|--------|--------|
| Resume from step_028000 | May also fail | Rules out "all checkpoints poisoned" | ❌ Failed (72 steps) |
| Lower adv_weight + lr_g | Slower GAN progress | More stable training | ❌ Failed (51 steps, worse) |
| Fresh start from VITS core | Loses all GAN progress | Clean slate | ❌ Failed (105 steps) |
| **Disable clip-coef trigger** | May miss real instability | Tests hypothesis | **Next** |
| Loosen clip-coef thresholds | Balance safety vs progress | Less aggressive detection | Alternative |

---

## Resolution

**Root cause:** The YAML config values for `consecutive_spikes_for_unstable`, `clip_coef_*` etc. were **not being read** by `train_vits.py`. All "disabled" settings were ignored — the code defaults (aggressive values) were always used.

**Fix:** Added missing config fields to the `GANControllerConfig()` constructor in `train_vits.py`.

**Validation:** After fix, fresh start from VITS core (step 15000) ran 5000 steps to step 20000 with:
- `ctrl_alarms_triggered_total: 0`
- `ctrl_escalation_level: 0`
- `g_clip_coef: 0.03-0.07` (normal range)

**Baseline established:** Normal GAN training has `g_clip_coef ~ 0.03-0.07`. The original 0.05 threshold was right in the middle of normal operation.

---

## Action Items

| Priority | Action | Status |
|----------|--------|--------|
| P0 | ~~Test step_028000_resume_start.pt~~ | ❌ Failed in 72 steps |
| P0 | ~~Adjust hyperparams: adv_weight → 0.25, lr_g → 1e-4~~ | ❌ Failed in 51 steps (worse) |
| P0 | ~~Fresh start from VITS core (step 15000)~~ | ❌ Failed in 105 steps |
| P0 | ~~Loosen/disable clip coefficient trigger~~ | ✅ Fixed — config wasn't being read |
| P0 | ~~Establish baseline clip_coef from healthy training~~ | ✅ Normal is 0.03-0.07 |
| P1 | Recalibrate trigger thresholds based on baseline | **Next** |

---

## Lessons Learned

1. **Three consecutive poisoned checkpoints** — The instability window is wider than expected
2. **Clip-coef is extremely sensitive** — All runs show ~99% clipping immediately
3. **Need baseline data** — Without clip_coef from a healthy run, hard to know if triggers are calibrated correctly
4. **Escalation ladder is fast** — 4 levels in ~60 steps may be too aggressive for recovery
5. **KL spikes at 28.8k were the real warning** — Model was already degrading before checkpoints were saved
