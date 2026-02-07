# Gold Utterances Table

> Training-ready utterances with deterministic splits and manifest exports.

## Location

```
lake/gold/{dataset}/utterances      # Delta table
lake/gold/{dataset}/manifests/      # JSONL exports
```

## Primary Key

`utterance_id` (inherited from silver)

## Partitioning

```python
partitionBy=["dataset", "split"]
```

## Schema

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `utterance_id` | string | No | Stable deterministic ID |
| `utterance_key` | string | No | Human-readable key |
| `dataset` | string | No | Dataset name |
| `speaker_id` | string | No | Speaker ID for conditioning |
| `audio_relpath` | string | No | Relative audio path |
| `duration_sec` | float | No | Duration in seconds |
| `sample_rate` | int | No | Sample rate in Hz |
| `text` | string | No | Canonical text: `coalesce(text_norm, text_norm_raw, text_raw)` |
| `phonemes` | string | Yes | Canonical phonemes: `coalesce(phonemes, phonemes_raw)` |
| `phonemes_source` | string | No | Effective source: `coalesce(phonemes_method, phonemes_source)` |
| `split` | string | No | `train`, `val`, or `test` |
| `duration_bucket` | string | No | Length bucket: `xs`, `s`, `m`, `l`, `xl`, `xxl` |
| `sample_weight` | float | No | Training weight (default 1.0) |
| `gold_version` | string | No | Gold pipeline version |
| `silver_version` | long | Yes | Silver Delta version |
| `created_at` | timestamp | No | When gold row was created |

## Column Applications

### Training Inputs
- **text**: The canonical text to train on. Never null.
- **phonemes**: Phoneme representation. May be null if not yet generated.
- **audio_relpath**: Resolve to `DATA_ROOT/data/{audio_relpath}` at training time.

### Batching & Sampling
- **duration_sec**: Used for dynamic batching, curriculum learning.
- **duration_bucket**: Pre-computed for length-balanced batching.
- **sample_weight**: Adjust importance of certain samples (e.g., upweight rare speakers).

### Splits
- **split**: Deterministic assignment based on `hash(utterance_id + seed)`.

### Provenance
- **gold_version**: Track which gold pipeline version produced this.
- **silver_version**: Points back to source silver for reproducibility.

## Duration Buckets

| Bucket | Duration Range |
|--------|----------------|
| `xs` | < 2s |
| `s` | 2-4s |
| `m` | 4-6s |
| `l` | 6-8s |
| `xl` | 8-10s |
| `xxl` | > 10s |

## Split Assignment

Splits are assigned **deterministically** based on:

```python
hash_input = f"{utterance_id}_{seed}"
hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
p = (hash_val % 10000) / 10000.0

if p < val_pct:
    return "val"
elif p < val_pct + test_pct:
    return "test"
else:
    return "train"
```

**Key property**: Same `seed` + same `utterance_id` = same split, regardless of filtering.

## CLI Defaults

```bash
koe gold jsut  # Uses these defaults:
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 42 | Random seed for splits |
| `--train-pct` | 0.90 | Training fraction |
| `--val-pct` | 0.10 | Validation fraction |
| `--test-pct` | 0.00 | Test fraction |
| `--min-duration` | 0.5 | Minimum duration filter |
| `--max-duration` | 20.0 | Maximum duration filter |

## Manifest JSONL Format

Exported to: `lake/gold/{dataset}/manifests/{snapshot_id}.jsonl`

```json
{
  "snapshot_id": "jsut-20260125-055016-fd76d53a",
  "utterance_id": "abc123...",
  "utterance_key": "jsut_basic5000_BASIC5000_0001",
  "split": "train",
  "audio_relpath": "ingest/jsut/extracted/...",
  "audio_abspath": "${DATA_ROOT}/data/ingest/jsut/extracted/...",
  "text": "...",
  "phonemes": null,
  "speaker_id": "spk00",
  "duration_sec": 3.45,
  "sample_rate": 48000,
  "duration_bucket": "s"
}
```

## Invariants

1. `text` is never null
2. `split` is one of: `train`, `val`, `test`
3. Split percentages match config (within 1% tolerance due to hashing)
4. Same seed + same filters = identical output (deterministic)
5. All rows have `is_trainable=True` from silver (filtered out otherwise)

## Validation Checks

```python
# Run by gold pipeline
assert df.filter(col("text").isNull()).count() == 0
assert df.filter(col("audio_relpath").isNull()).count() == 0
assert df.filter(~col("split").isin(["train", "val", "test"])).count() == 0

# Split percentages
total = df.count()
train_pct = df.filter(col("split") == "train").count() / total
assert abs(train_pct - expected_train_pct) < 0.02
```
