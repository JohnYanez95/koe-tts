# Segmentation Labeler (TBD)
**Purpose:** Create sub-utterances by inserting breakpoints aligned to pauses, enabling shorter training sequences (lower VRAM) and increasing effective sample count without duplicating audio storage.

## Context & Motivation
We validated that VITS multi-speaker training can proceed by **random fixed-length cropping** (`segment_samples`) to bound memory. This avoids discarding data, but it does not create explicit, human-auditable segments aligned to text.

**Pause-based segmentation is the next step**: split utterances into smaller training-ready segments at meaningful pause points (canonical `pau` tokens), while preserving:
- full provenance to the original utterance
- deterministic splits (no train/val leakage)
- reproducibility via artifacts under `data/assets/<dataset>/`

## Goals
1. **Reduce VRAM** by ensuring training samples have bounded duration (e.g., <= 6–10s).
2. **Increase sample count** by turning one utterance into N segments.
3. **Preserve data lineage**: every segment maps back to the source utterance + source dataset + source snapshot.
4. **Be storage efficient**: no audio duplication — segments reference spans of the original audio file.
5. **Enable labeling workflows**: make segmentation reviewable/editable.

## Non-Goals (for v0)
- Perfect forced alignment / phone-level timestamps (MFA etc.)
- Full phoneme correctness labeling (separate track)
- Real-time collaborative labeling

---

## Existing (Current) Solution: Random Crop Training
### What it does
Training crops every waveform to a fixed number of samples (e.g., 3 seconds @ 22.05 kHz) each time an item is loaded, so the model sees different windows across epochs.

### Why it works
VITS/HiFiGAN are commonly trained on fixed-length random segments. This bounds padding and keeps GPU memory stable even with long utterances.

### What it does *not* solve
- Does not produce stable, inspectable segments
- Does not align text/phonemes to segment boundaries
- Harder to use for duration predictor training and human review

---

## Lakehouse Integration

### Existing tables
- Bronze: `lake/bronze/<dataset>/utterances`
- Silver: `lake/silver/<dataset>/utterances` (QC, split, phonemes)
- Gold: `lake/gold/<dataset>/utterances` + `manifests/`

### New tables (proposed)
**Silver (labels):**
- `lake/silver/<dataset>/segment_breaks`  
  Stores human/auto breakpoints and optional snapped boundaries.

**Gold (training-ready):**
- `lake/gold/<dataset>/segments`  
  Materialized segments with audio/text/phoneme spans + inherited split.

### New assets (proposed)
- `data/assets/<dataset>/segmentation/`
  - `segment_inventory.json` (counts, durations, hash)
  - optional exports (batch JSON, audit logs)

---

## Data Model

### 1) Silver: `segment_breaks`
**Primary key:** `(utterance_id, break_idx)`  
**One row per breakpoint** for auditing and analytics.

Required fields:
- `utterance_id: string` (FK to silver utterances)
- `break_idx: int` (0..k-1)
- `break_time_ms: int` (timestamp in ms from utterance start)

Optional alignment fields:
- `break_phoneme_idx: int | null` (token boundary index in canonical phoneme list)
- `break_char_idx: int | null` (unicode index in `text_norm`)

Label provenance:
- `break_kind: string` (pause|breath|noise|uncertain)
- `snap_method: string` (manual|energy|pau_token|forced_align)
- `confidence: float | null`
- `notes: string | null`

Standard label tracking:
- `label_status: string` (unlabeled|labeled|reviewed|rejected)
- `label_batch_id: string | null`
- `labeled_at: timestamp | null`
- `labeled_by: string | null`
- `label_version: string` (e.g., "seg_v0")

Invariants:
- Breaks per utterance strictly increasing by `break_time_ms`
- 0 < break_time_ms < utterance_duration_ms
- No duplicate `(utterance_id, break_idx)`
- If `break_phoneme_idx` set, must be within token bounds

---

### 2) Gold: `segments`
**Primary key:** `segment_id`  
Deterministic:
- `segment_id = sha1("{utterance_id}|{start_ms}|{end_ms}")[:16]`

Required fields:
- `segment_id: string`
- `utterance_id: string` (parent)
- `segment_index: int` (0..n-1)
- `dataset: string`
- `speaker_id: string`
- `split: string` (**inherit from parent silver.split**)
- `audio_relpath: string` (same as parent)
- `start_ms: int`
- `end_ms: int`
- `duration_sec: float`

Recommended fields:
- `start_sample: int` (derived from SR)
- `num_samples: int`
- `text: string` (text span for this segment)
- `phonemes: string` (phoneme span for this segment)
- `phonemes_method: string`
- `segmentation_version: string`
- `generated_at: timestamp`

Provenance (required for multi):
- `source_dataset: string`
- `source_snapshot_id: string`
- `source_utterance_id: string`
- `source_utterance_key: string | null`

Invariants:
- `split` equals parent utterance split (no recompute)
- 0 <= start_ms < end_ms <= utterance_duration_ms
- Segments do not overlap within an utterance (v0)
- Duration bounds: `min_segment_sec` <= duration <= `max_segment_sec`

---

## Segmentation Strategy

### v0: Automatic (pause-based)
**Primary signal:** canonical phonemes contain pause markers:
- internal `pau` should be preserved
- boundary `sil` already stripped by phoneme contract

Heuristic:
1. Tokenize phonemes
2. Identify candidate cut points at `pau`
3. Create segments such that:
   - duration in [min_segment_sec, max_segment_sec]
   - avoid tiny leading/trailing fragments
   - keep punctuation-driven pauses inside segments when required

Write:
- `segment_breaks` with `snap_method = "pau_token"`
- `segment_inventory.json` artifact with hash for change detection

### v1: Human correction (labeler UI)
UI features:
- listen + waveform view
- click time to add/remove break
- snap to nearest `pau` or energy dip
- optionally select phoneme boundary index
- set `break_kind`, add notes

---

## CLI (proposed)

### Auto segmentation
- `koe segment auto <dataset> [--max-sec 8] [--min-sec 0.6] [--strategy pau]`
  - reads `lake/silver/<dataset>/utterances`
  - writes `lake/silver/<dataset>/segment_breaks`
  - writes `data/assets/<dataset>/segmentation/segment_inventory.json`

### Materialize gold segments
- `koe segment build <dataset> [--version seg_v0]`
  - reads silver utterances + segment_breaks
  - writes `lake/gold/<dataset>/segments`
  - writes manifest `lake/gold/<dataset>/manifests/<segments_snapshot>.jsonl`

### Cache creation (segments)
- `koe cache create <dataset> --from segments`
  - manifest includes `start_sample/num_samples` (preferred) or `start_ms/end_ms`
  - audio remains symlinked to ingest source (no duplication)

---

## Training Integration
Two modes:
1. **utterance mode** (current): manifest points to full utterances; training uses random crop
2. **segment mode** (new): manifest points to original audio + span; training may still crop *within* segment if desired

Loader (`TTSDataset`) should:
- load audio (symlink OK)
- resample to mel SR (already enforced)
- slice by span (prefer sample indices)
- compute mel on sliced audio

Sampling:
- multi-speaker: continue `SpeakerBalancedBatchSampler`
- optional future: length-bucketing on segment duration

---

## Validation Checklist
1. No split leakage: segment.split == parent split
2. Segment ID determinism: same inputs => same segment_ids
3. Duration bounds: no segment outside [min,max] unless flagged
4. Slicing correctness: sliced duration matches expected within tolerance
5. Non-empty content: text/phonemes not empty (not BOS/EOS only)
6. Inventory: pause markers obey contract (no boundary sil)

---

## Recommended Path
1. Finish current multi-speaker validation run using random crop (proves conditioning works)
2. Implement `koe segment auto` to generate pause-based breakpoints (Silver)
3. Implement `koe segment build` to materialize segments (Gold)
4. Use segments for:
   - duration predictor training (requires aligned text spans)
   - targeted GAN fine-tuning on longer-form prosody (optional)
