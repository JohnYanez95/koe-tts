# Silver Utterances Table

> Curated utterances with QC gating, normalization, and labeling hooks.

## Location

```
lake/silver/{dataset}/utterances
```

## Primary Key

`utterance_id` (inherited from bronze)

## Partitioning

```python
partitionBy=["dataset"]
```

## Schema

### Bronze Passthrough (23 columns)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `utterance_id` | string | No | Stable deterministic ID |
| `utterance_key` | string | No | Human-readable key |
| `dataset` | string | No | Dataset name |
| `speaker_id` | string | No | Normalized speaker ID |
| `speaker_name` | string | Yes | Optional friendly name |
| `subset` | string | No | Corpus subset |
| `corpus_utt_id` | string | No | Original corpus utterance ID |
| `audio_relpath` | string | No | Relative audio path |
| `audio_format` | string | No | Audio format |
| `sample_rate` | int | No | Sample rate in Hz |
| `channels` | int | No | Number of channels |
| `duration_sec` | float | No | Duration in seconds |
| `text_raw` | string | No | Original transcript |
| `text_norm_raw` | string | Yes | Corpus-provided normalization |
| `phonemes_source` | string | No | Source of raw phonemes |
| `phonemes_raw` | string | Yes | Corpus phonemes |
| `ingest_version` | string | No | Ingest version |
| `source_version` | string | Yes | Corpus version |
| `source_url` | string | Yes | Download URL |
| `source_archive_checksum` | string | Yes | Archive checksum |
| `audio_checksum` | string | Yes | Audio file checksum |
| `ingested_at` | timestamp | No | Ingest timestamp |
| `meta` | map<string,string> | Yes | Overflow metadata |

### Silver Enrichment (17 columns)

#### QC / Filtering

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `is_trainable` | boolean | No | `true` | Master gate. Gold only uses rows where `is_trainable=True`. |
| `exclude_reason` | string | Yes | `null` | Why excluded: `bad_audio`, `too_short`, `too_long`, `clipping`, etc. |
| `qc_version` | string | Yes | `null` | QC ruleset version tag |
| `qc_checked_at` | timestamp | Yes | `null` | When QC was last run |

#### Normalized Text

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `text_norm` | string | Yes | `null` | Canonical normalized text (NFKC, punctuation policy, etc.) |
| `text_norm_method` | string | Yes | `null` | How produced: `none`, `corpus`, `rule_v1`, `jaconv_v2` |

#### Canonical Phonemes

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `phonemes` | string | Yes | `null` | Curated phoneme string for training |
| `phonemes_method` | string | Yes | `null` | How produced: `ground_truth`, `g2p_pyopenjtalk`, `manual` |
| `phonemes_checked` | boolean | No | `false` | Whether human/validator verified phonemes |

#### Split Assignment

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `split` | string | Yes | `null` | `train`, `val`, `test` (usually assigned in gold) |

#### Labeling Workflow

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `label_status` | string | No | `unlabeled` | Status: `unlabeled`, `in_progress`, `labeled`, `rejected` |
| `label_batch_id` | string | Yes | `null` | Which batch this utterance belongs to |
| `labeled_at` | timestamp | Yes | `null` | When label was committed |
| `labeled_by` | string | Yes | `null` | Human/device identifier |

#### Lineage

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `bronze_version` | string | Yes | `null` | Bronze Delta version used |
| `silver_version` | string | Yes | pipeline tag | Silver pipeline version |
| `processed_at` | timestamp | No | `now()` | When row was processed |

## Column Applications

### QC Gating
- **is_trainable**: The master switch. Set to `false` to exclude from gold.
- **exclude_reason**: Enables debugging why rows were excluded. Powers dashboards.
- **qc_version**: Allows re-running QC with new rules and tracking what changed.

### Text Normalization
- **text_norm**: Your canonical text after cleanup. Gold uses: `coalesce(text_norm, text_norm_raw, text_raw)`.
- **text_norm_method**: Enables ablation studies (compare models trained on different normalizations).

### Phonemes
- **phonemes**: Your curated phoneme representation for training.
- **phonemes_method**: Track provenance (G2P vs ground truth vs manual).
- **phonemes_checked**: High-value flag for "verified" phonemes.

### Labeling
- **label_status**: Drives the labeling UI workflow.
- **label_batch_id**: Groups utterances for batch labeling sessions.
- **labeled_by**: Audit trail for who labeled what.

## Invariants

1. Record count matches bronze (no silent drops)
2. If `is_trainable=False`, then `exclude_reason` should be set
3. `label_status` is one of: `unlabeled`, `in_progress`, `labeled`, `rejected`
4. `phonemes_source` (bronze) and `phonemes_method` (silver) are independent

## Stub Defaults

For initial silver (no QC/phonemization yet):

```python
SILVER_STUB_DEFAULTS = {
    "is_trainable": True,
    "exclude_reason": None,
    "qc_version": None,
    "qc_checked_at": None,
    "text_norm": None,
    "text_norm_method": None,
    "phonemes": None,
    "phonemes_method": None,
    "phonemes_checked": False,
    "split": None,
    "label_status": "unlabeled",
    "label_batch_id": None,
    "labeled_at": None,
    "labeled_by": None,
}
```
