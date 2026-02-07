# Staged Trim + Pause Optimization Pipeline

## Context

The current pause detection optimizer (`optimize_heuristic()`) filters heuristic candidates to `[trim_start, trim_end]` before scoring — evaluating in a curated window the user already cleaned up. But at inference time, the heuristic runs on full, untrimmed audio. This creates an input distribution divergence: parameters tuned on clean regions don't generalize to raw audio with leading/trailing noise, silence, or artifacts.

The user's `trim_start_ms` and `trim_end_ms` labels are valuable ground truth, but for a *different* problem than pause detection. Trim detection ("where does speech begin and end?") and pause detection ("where are internal silences?") have different signal characteristics, different parameter spaces, and different output cardinalities. They must be optimized independently and staged as a cascaded pipeline — Stage 2 (pause) trains on Stage 1 (trim) *predictions*, not on user GT trims and not on raw untrimmed audio. Pause GT is only meaningful within the content window, so evaluation must apply the same windowing to candidates and GT.

### Why two stages, not one

| | Trim detection | Pause detection |
|---|---|---|
| **Question** | Where does speech begin and end? | Where are internal silences? |
| **Output cardinality** | Always 2 (start, end) | Variable N |
| **Signal regime** | Boundary energy transition (onset/offset) | Interior energy dips |
| **Failure mode** | Breath, mic pop, room noise at edges | Missed/phantom pauses in speech |
| **Parameter sensitivity** | Onset vs offset may need asymmetric thresholds | Uniform threshold across interior |
| **User labeling** | `trim_start_ms`, `trim_end_ms` | `breaks[].ms`, `breaks[].use_break` |

Mixing them in one optimizer conflates two different signal regimes and creates the trim-as-evaluation-window circular dependency.

### Why pause must train on predicted trims, not GT trims or raw audio

This is standard cascaded system training:

- **Raw audio (no trim)**: Pause optimizer gets penalized for orphan breakpoints in leading/trailing junk that the user never reviewed. Published pause labels are implicitly conditioned on the user's trim — scoring outside that window is meaningless.
- **User GT trims**: Creates the original distribution mismatch — at inference, Stage 1 predictions (not GT) define the window.
- **Predicted trims from frozen Stage 1**: Matches inference exactly. The pause optimizer learns to work with whatever trim quality Stage 1 actually achieves.

Training order:
1. Optimize trim params → freeze best trim config T*
2. Compute predicted trims ŝ, ê = T*(x) **once** as a fixed artifact
3. Optimize pause params on candidates within [ŝ-m, ê+m] (with slack band)

No circular dependency. Stable. Matches inference distribution.

## Implementation Order

1. `detect_trim_region()` + `TrimDetectionConfig` + `run_trim_heuristic()` (with clamping)
2. `optimize_trim()` (simple, fast — 2-point regression, capped losses, fallback penalty)
3. Cache trim predictions as JSONL artifact
4. Modify pause optimizer to accept pre-computed `trim_predictions` dict + slack band + filter GT to same window
5. Provenance fields in pause `.meta.json`
6. UI seeding with auto-trim proposals (nice-to-have, defer)

## Changes

### 1. Add trim detection function

**File:** `modules/data_engineering/common/audio.py`

New dataclass and function alongside the existing pause detection infrastructure:

- **`TrimDetectionConfig`** — separate parameter space:
  - `window_ms`, `hop_ms` (reuse RMS infra)
  - `onset_margin_db` — sensitivity for speech onset
  - `offset_margin_db` — sensitivity for speech offset (asymmetric — tails are harder, breath/room noise)
  - `floor_db` — absolute floor
  - `percentile` — for adaptive threshold (included in optimizer bounds — matters across datasets)
  - `min_content_ms` — minimum content duration (prevents degenerate trims)
  - `pad_start_ms` — outward padding at onset
  - `pad_end_ms` — outward padding at offset

- **`detect_trim_region(waveform, sr, config)`** → `(trim_start_ms, trim_end_ms, debug_info)`
  - Reuses `compute_rms_db()` for energy computation
  - Reuses `compute_adaptive_threshold()` for noise floor estimation
  - Scans forward from start to find first frame above threshold → `trim_start`
  - Scans backward from end to find last frame above threshold → `trim_end`
  - Applies asymmetric padding outward (`pad_start_ms`, `pad_end_ms`)
  - **Clamp + validity check** (prevents garbage trims):
    ```python
    s = max(0, min(s, duration_ms))
    e = max(0, min(e, duration_ms))
    if e - s < config.min_content_ms:
        s, e = 0, duration_ms  # fallback to full audio
    ```
  - `debug_info["fallback_to_full_audio"]` = True/False — tracks whether clamping triggered
  - `debug_info["fallback_reason"]` = `"min_content"` | `"inverted"` | `None` — free diagnostic for 2am debugging
  - Always returns exactly 2 positions (constrained output)

- **Confidence scores** in `debug_info`:
  - Logistic on energy transition delta at each edge:
    ```python
    # Compare mean RMS in small windows inside vs outside the boundary
    delta_db = mean_rms_db(inside_window) - mean_rms_db(outside_window)
    confidence = 1.0 / (1.0 + math.exp(-k * (delta_db - b)))
    ```
  - Sharp onset (speech jumps from silence) → high confidence
  - Gradual onset (breath, room noise ramp) → low confidence
  - Monotonic, bounded [0, 1], stable across audio types
  - Enables active learning: focus labeling on low-confidence trims

### 2. Add trim optimizer

**File:** `modules/labeler/heuristic.py`

- **`_compute_trim_loss_utterance(gt_start, gt_end, pred_start, pred_end, tau_ms)`**
  - Capped error on all terms (robust to outliers, well-behaved with DE):
    - `L_start = min(|gt_start - pred_start| / τ, 1.0)`
    - `L_end = min(|gt_end - pred_end| / τ, 1.0)`
    - `L_dur = min(|gt_duration - pred_duration| / (gt_duration + ε), 1.0)`
  - Loss = `α_start * L_start + α_end * L_end + λ_dur * L_dur`
  - No Hungarian matching — always 1:1 (start↔start, end↔end)

- **`_evaluate_trim_config(config, labels, preloaded_audio, ...)`**
  - Filters to labels where both `trim_start_ms` and `trim_end_ms` are non-null
  - For each utterance: `detect_trim_region(waveform, sr, config)` → compare with GT
  - **Fallback penalty** — prevents DE from "winning" by producing invalid trims that hide errors:
    ```python
    fallback_rate = n_fallback / n_evaluated
    total_loss = mean_trim_loss + lambda_fallback * fallback_rate
    ```
    Even `lambda_fallback=0.1` is enough to discourage configs that trigger frequent clamping
  - Returns mean loss + aggregated metrics (MAE start, MAE end, per-edge confidence, fallback_rate)
  - **Stratified metrics by trim confidence** (reported, not used in loss):
    ```python
    if min(conf_start, conf_end) < 0.3:
        bucket = "low_conf"
    else:
        bucket = "high_conf"
    # aggregate MAE / recall per bucket
    ```
    Tells you whether Stage 1 quality is limiting Stage 2

- **`optimize_trim(dataset, n_folds, max_iter, seed, name)`**
  - Same DE + k-fold CV pattern as `optimize_heuristic()`
  - Optimizes 7 params with bounds:

    | Parameter | Bounds | Default | Rationale |
    |-----------|--------|---------|-----------|
    | `onset_margin_db` | [2.0, 20.0] | 8.0 | Same range as pause margin — onset sensitivity |
    | `offset_margin_db` | [2.0, 25.0] | 10.0 | Wider upper bound — tails are noisier, may need more aggressive threshold |
    | `floor_db` | [-80.0, -35.0] | -60.0 | Same as pause — absolute safety floor |
    | `percentile` | [5, 30] | 10 | Clamped to avoid bizarre thresholds. Too low (< 5) → threshold dominated by digital silence. Too high (> 30) → threshold eats into speech |
    | `min_content_ms` | [150, 1200] | 500 | Floor at 150ms (shortest plausible utterance). Ceiling at 1200ms — higher risks frequent fallback-to-full-audio on short clips, masking trim failures. Fallback penalty discourages this further |
    | `pad_start_ms` | [0, 150] | 30 | Don't need huge onset padding — speech starts are usually clean |
    | `pad_end_ms` | [0, 250] | 50 | Wider range — tails have more variability (breath, room decay) |

  - Preloads audio (same pattern)
  - Evaluates baseline `TrimDetectionConfig()` for comparison
  - Saves winner as trim heuristic run with method prefix `trim_v1_*`

### 3. Add trim heuristic runner

**File:** `modules/labeler/heuristic.py`

- **`run_trim_heuristic(dataset, ...)`**
  - Parallel to `run_heuristic()` but outputs trim predictions
  - **Applies clamp + validity check** before writing (no garbage trims in cache)
  - JSONL record: `{utterance_id, trim_start_ms, trim_end_ms, confidence_start, confidence_end, fallback_to_full_audio, fallback_reason, duration_ms, method, params_hash}`
  - Writes `.meta.json` sidecar
  - Output path: `runs/labeling/heuristics/{dataset}_trim_{params_hash}.jsonl`

### 4. Replace user-trim filtering with predicted-trim filtering in pause optimizer

**File:** `modules/labeler/heuristic.py` — `_evaluate_config()` and `optimize_heuristic()`

**Critical: trim predictions are computed once and passed as a fixed artifact, not recomputed per DE evaluation.**

**Stage 2 uses its own safety constant, decoupled from Stage 1's `min_content_ms`:**

```python
# Stage-2 guardrail — independent of trim config hyperparameters
MIN_TRIM_WINDOW_MS_FOR_PAUSE = 200
```

`optimize_heuristic()` updated to:
- Accept optional `trim_predictions: dict[str, tuple[int, int]] | None` (pre-computed, frozen)
- Accept `trim_margin_ms: int = 200` (slack band around predicted trims)
- If `trim_predictions` is None: full audio fallback (for bootstrapping before any trim model exists — will change learned params vs old user-GT-filtered behavior, intentional)

```python
# In optimize_heuristic(), BEFORE the DE loop:
# Trim predictions are a fixed dataset artifact — computed once, never inside objective
trim_predictions = load_trim_cache(dataset, trim_params_hash)  # or None

# In _evaluate_config(), per utterance:
pred_trim = trim_predictions.get(uid) if trim_predictions else None
if pred_trim:
    s, e = pred_trim
    # Clamp (defensive — should already be clean from run_trim_heuristic)
    s = max(0, min(s, duration_ms))
    e = max(0, min(e, duration_ms))

    if e - s < MIN_TRIM_WINDOW_MS_FOR_PAUSE:
        # Trim invalid for pause evaluation → treat as missing
        # Use full audio, do NOT filter GT
        candidates = breakpoints
        window_mode = "full_audio"
        window_start_ms, window_end_ms = 0, duration_ms
    else:
        m = trim_margin_ms
        window_start = max(0, s - m)
        window_end = e + m
        candidates = [bp for bp in breakpoints if window_start <= bp <= window_end]
        gt_accepted = [g for g in gt_accepted if window_start <= g <= window_end]
        gt_rejected = [r for r in gt_rejected if window_start <= r <= window_end]
        window_mode = "pred+slack"
        window_start_ms, window_end_ms = window_start, window_end
else:
    # No trim predictions available → full audio, do NOT filter GT
    candidates = breakpoints
    window_mode = "full_audio"
    window_start_ms, window_end_ms = 0, duration_ms

# Per-utterance debug (aggregated in metrics, not in label output)
# window_mode, window_start_ms, window_end_ms — explains why 0 matches happened
```

**Invariant: if trim is invalid or missing, Stage 2 uses full audio and does NOT window GT.** This must be preserved across refactors — if someone later filters GT based on an invalid window, it silently zeros out data.

**Slack band (m = trim_margin_ms)**: Stage 1 isn't perfect. Without slack, small trim boundary errors cause Stage 2 to miss GT pauses near edges and get unfairly penalized. The slack band (default 200ms) absorbs boundary drift while still constraining the evaluation to the content region. Both candidates AND gt are filtered to the same `[ŝ-m, ê+m]` window — coordinate frames stay aligned.

**Training vs inference slack**: Training uses slack (200ms) to absorb Stage 1 boundary drift. Inference uses no slack by default (filter to exact `[ŝ, ê]`). Optional small inference slack (e.g., 50ms) can be added later if it helps robustness — but start without it.

**Note on asymmetric slack**: v1 uses symmetric `trim_margin_ms`. Future upgrade path: split to `trim_margin_start_ms` / `trim_margin_end_ms` if recordings show systematically different junk profiles at head vs tail.

### 5. Provenance: frozen trim run ID in pause metadata

Pause heuristic `.meta.json` gains fields linking it to its Stage 1:
- `trained_on_trim_params_hash` — which trim config was frozen
- `trim_margin_ms` — slack band used
- `trim_method` — e.g. `trim_v1_adaptive`

This makes runs fully reproducible: you can always trace which Stage 1 a Stage 2 was conditioned on.

### 6. Wire into the labeling app data layer

**File:** `modules/labeler/app/data.py`

- **`load_auto_trim(dataset, trim_params_hash)`** — loads trim predictions from cache
  - Returns `dict[utterance_id → (trim_start_ms, trim_end_ms)]`
  - Priority: trim JSONL cache → fallback to None (user sets manually)

- Update `get_batch()` to seed initial `trim_start_ms`/`trim_end_ms` from auto-trim when no saved labels exist
  - User can still override — auto-trim is a proposal, not a constraint

### 7. Coordinate frame enforcement

All ms values are **absolute** (from file start). This is already the convention but should be explicitly enforced:

- `trim_start_ms`, `trim_end_ms`: absolute
- `breaks[].ms`, `breaks[].ms_proposed`: absolute
- `breakpoints_ms[]` from heuristic: absolute
- Segment `start_ms`, `end_ms`: absolute (relative to parent audio start)

No coordinate transforms needed between stages — predicted trim boundaries are in the same frame as pause breakpoints and user labels.

### 8. Update architecture diagram

**File:** `labeler-architecture.md`

- Replace the current single-optimizer flow with the staged pipeline
- Document the cascaded training order and slack band
- Update the feedback loop to show two separate optimization cycles

## Staged Pipeline (target architecture)

```
Full Audio (waveform)
    │
    ▼
┌───────────────────────────────┐
│  Stage 1: Trim Optimizer       │
│  detect_trim_region()          │
│  Always 2 points: ŝ, ê        │
│  Own params, own loss          │
│  GT: user trim_start/trim_end  │
│  Clamp: ê-ŝ ≥ min_content_ms  │
│  Fallback penalty in loss      │
└──────────┬────────────────────┘
           │ predicted [ŝ, ê]  (computed once, frozen)
           ▼
┌───────────────────────────────┐
│  Stage 2: Pause Optimizer      │
│  detect_silence_regions()      │
│  N breakpoints within [ŝ-m,ê+m]│
│  + GT filtered to same window  │
│  Own params, own loss          │
│  Own safety: MIN_TRIM_WINDOW=200│
│  Invalid trim → full audio,    │
│  do NOT window GT              │
│  Confidence-stratified metrics │
│  Trained on Stage 1 output     │
└──────────┬────────────────────┘
           │ breakpoints_ms[]
           ▼
      Labeling UI
   (user refines both)
```

Training order (no circular dependency):
1. `optimize_trim(dataset)` → freeze best `TrimDetectionConfig` T*
2. `run_trim_heuristic(dataset, **T*)` → cache predicted trims as JSONL artifact (with clamp)
3. `optimize_heuristic(dataset, trim_predictions=cached, trim_margin_ms=200)` → pause optimizer sees Stage 1 output distribution + slack

Inference (no slack by default):
1. `detect_trim_region(wav, T*)` → [ŝ, ê] (clamped)
2. `detect_silence_regions(wav, P*)` → breakpoints, filtered to [ŝ, ê]

Training uses slack to avoid penalizing Stage 2 for small Stage 1 drift; inference can optionally add small slack later if empirical evaluation shows it improves robustness.

## Files Modified

| File | Change |
|------|--------|
| `modules/data_engineering/common/audio.py` | Add `TrimDetectionConfig`, `detect_trim_region()` with asymmetric padding, clamp/validity check, logistic confidence scores, `fallback` flag in debug_info |
| `modules/labeler/heuristic.py` | Add `optimize_trim()`, `run_trim_heuristic()`, `_compute_trim_loss_utterance()`, `_evaluate_trim_config()` with fallback penalty + confidence-stratified metrics. Add `MIN_TRIM_WINDOW_MS_FOR_PAUSE` constant. Update `_evaluate_config()` to accept `trim_predictions` + `trim_margin_ms` with slack-band filtering on both candidates and GT, decoupled safety clamp, explicit no-filter-GT on invalid trim. Update `optimize_heuristic()` to pass pre-computed trim predictions. Add trim provenance to pause `.meta.json` |
| `modules/labeler/app/data.py` | Add `load_auto_trim()`, update `get_batch()` to seed trim proposals |
| `labeler-architecture.md` | Update diagrams for staged pipeline, cascaded training, slack band, coordinate frame docs |

## Verification

1. **Clamp**: Verify `detect_trim_region()` and `run_trim_heuristic()` never produce inverted or sub-`min_content_ms` intervals — falls back to full audio, `debug_info["fallback_to_full_audio"]` = True, `fallback_reason` populated
2. **Fallback penalty**: Verify `_evaluate_trim_config()` includes `lambda_fallback * fallback_rate` in returned loss. Configs that trigger frequent clamping should score worse
3. **Trim optimizer**: Run `optimize_trim()` on published labels with trim data — verify convergence, reasonable MAE on both edges, `percentile` stays in [5, 30], `min_content_ms` doesn't drift to upper bound
4. **Trim runner**: Run `run_trim_heuristic()` — verify JSONL has `trim_start_ms`, `trim_end_ms`, `confidence_start`, `confidence_end`, `fallback_to_full_audio`, `fallback_reason` per utterance, all clamped
5. **Stage-2 safety decoupled**: Verify `MIN_TRIM_WINDOW_MS_FOR_PAUSE` is a constant (200), not read from trim config
6. **Invalid trim invariant**: Verify that when `e - s < MIN_TRIM_WINDOW_MS_FOR_PAUSE`, Stage 2 uses full audio AND does not filter GT
7. **Cascaded pause**: Run `optimize_heuristic(trim_predictions=cached)` — verify candidates AND gt are filtered to `[ŝ-m, ê+m]`, not user trims
8. **Slack sanity**: Verify that with `trim_margin_ms=200`, no GT pauses near edges are lost (compare matched count vs old approach)
9. **Confidence stratification**: Check that `_evaluate_trim_config()` reports separate MAE/recall for low_conf vs high_conf buckets
10. **Provenance**: Verify pause `.meta.json` contains `trained_on_trim_params_hash`, `trim_margin_ms`, `trim_method`
11. **No-trim fallback**: `optimize_heuristic()` with no `trim_predictions` uses full audio and does not filter GT — intentional behavior change
12. **Lint**: `make lint` passes
13. **Tests**: `make test` — no regressions
