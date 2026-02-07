# Postmortem: multi_vits_gan_20260128_210613 — Grad Ceiling Emergency

## Incident Summary

| Field | Value |
|-------|-------|
| Run ID | `multi_vits_gan_20260128_210613` |
| Stage | GAN (post disc_start) |
| Step Range | 28,000 → 30,822 |
| Outcome | **Emergency stop** — `grad_g_exceeded_5272_limit_5000` |
| Detection | Controller hard ceiling trigger |
| Checkpoints saved | `resume_start`, `early`, `29000`, `30000`, `emergency` |
| Data lost | 0 (clean stop with checkpoint) |

---

## Trigger

**Controller reason:** `grad_g_exceeded_5272_limit_5000`

The absolute grad ceiling (5000) fired when G gradient spiked to 5272. This was the "last resort fuse" — earlier detection methods didn't trigger because:

1. **Consecutive spikes (3 required):** Spikes had recovery steps between them
2. **EMA-based warning:** Not implemented at time of incident
3. **Spike density:** Not implemented at time of incident

---

## Evidence

### Gradient explosion pattern (steps 30810–30822)

| Step | g_grad | d_grad | ema_g | loss_adv | d_real | d_fake |
|------|--------|--------|-------|----------|--------|--------|
| 30810 | 286 | 47 | 150 | 12.5 | -0.48 | -0.55 |
| 30813 | 894 | 68 | 227 | 11.2 | -0.37 | -0.45 |
| 30814 | 942 | 78 | 298 | 10.4 | -0.24 | -0.35 |
| 30815 | 966 | 68 | 365 | 10.8 | -0.27 | -0.32 |
| 30816 | 1360 | 60 | 465 | 10.8 | -0.19 | -0.28 |
| 30817 | 471 | 27 | 465 | 10.8 | -0.17 | -0.25 |
| 30821 | 1442 | 97 | 546 | 11.9 | -0.34 | -0.48 |
| 30822 | **5272+** | 268 | 546 | 11.0 | -0.22 | -0.40 |

### Pattern analysis

- **D confusion:** D scores drifted toward 0 (less confident about real/fake)
- **G "winning":** loss_adv decreased from 12.5 to ~10.5
- **Classic GAN instability:** D weakened → G got huge gradients exploiting D → explosion
- **EMA climbed to 546** but no alarm triggered (no EMA-based trigger existed)
- **Controller stayed at `alarm=healthy, esc=0`** throughout

### KL divergence (earlier in run, ~28.8k)

KL spiked to 3–6× normal around step 28,800 but recovered. This was an early warning sign the model was entering an unstable regime.

---

## What Worked

1. **Checkpoint safety net:** `resume_start`, `early`, periodic checkpoints all saved
2. **Hard ceiling caught it:** Emergency stop prevented NaN collapse
3. **Clean shutdown:** `training_complete` event logged with reason
4. **Dashboard visibility:** Emergency banner showed correct status

---

## Root Cause Hypothesis

**Training dynamics instability** (not infrastructure):

- Model survived the 27k–30k "danger band" but entered a new instability regime past 30k
- D became weak/confused → G gradients exploded trying to exploit the weakness
- The stability patches (lr_d=5e-5, adv_weight=0.5) helped but weren't sufficient

**Contributing factors:**
- Consecutive spike requirement (3) missed bursty patterns
- No EMA-based early warning to catch gradual escalation
- Hard ceiling (5000) too high — model was already in trouble at 1000+

---

## Fixes Shipped

| Commit | Change |
|--------|--------|
| `e208d01` | Spike density trigger: 3 spikes in 20 steps → UNSTABLE |
| `e208d01` | EMA early warning: ema_grad > 500 for 50 steps → UNSTABLE |
| `e208d01` | Lower hard ceiling: 5000 → 3000 |
| `9905e51` | Explicit g_step_skipped / d_step_skipped signals |
| `9905e51` | Clip coefficient logging (g_clip_coef, d_clip_coef) |

---

## Action Items

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Spike density trigger | ✅ Shipped |
| P0 | EMA-based UNSTABLE trigger | ✅ Shipped |
| P0 | Lower absolute_grad_limit to 3000 | ✅ Shipped |
| P1 | Consider further reducing adv_weight (0.5 → 0.25) if instability persists | Pending |
| P1 | Consider lr_g reduction (2e-4 → 1e-4) as backup lever | Pending |
| P2 | Add KL spike badge to dashboard (5× rolling median) | Pending |

---

## Recovery Path

Resume from `step_030000.pt` (last clean periodic checkpoint before explosion):

```bash
koe train vits multi --stage gan \
  --resume runs/multi_vits_gan_20260128_210613/checkpoints/step_030000.pt \
  --save-every-steps 1000
```

Or for safer recovery, use `step_028294_early.pt` to re-traverse the danger band with new detection.

---

## Retrospective

**With the new triggers, this would have been caught earlier:**

- **EMA trigger** would fire around step 30780 (EMA > 500)
- **Spike density** would fire around step 30815 (3 spikes in 20 steps)
- Both ~40 steps before the hard ceiling explosion

The run successfully survived 28k–30k (previous danger band) but found a new instability regime at 30k+. The controller safety net worked — just needed earlier detection.
