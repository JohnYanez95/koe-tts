# Lakehouse Contract

> Single source of truth for the koe-tts data lakehouse architecture.

## Philosophy

This project uses a **medallion architecture** (Bronze → Silver → Gold) for TTS training data:

| Layer | Purpose | Mutability |
|-------|---------|------------|
| **Bronze** | Raw ingested data with full provenance | Append-only, never edit content |
| **Silver** | Curated data with QC, normalization, labeling | Updateable (QC flags, phonemes, labels) |
| **Gold** | Training-ready snapshots with deterministic splits | Immutable snapshots, versioned by snapshot_id |

### What belongs where

| Operation | Bronze | Silver | Gold |
|-----------|--------|--------|------|
| Store raw transcripts | Yes | Passthrough | No |
| Edit/normalize text | No | Yes (`text_norm`) | No |
| Generate phonemes | No | Yes (`phonemes`) | No |
| QC gating decisions | No | Yes (`is_trainable`, `exclude_reason`) | No |
| Human labeling | No | Yes (`label_*` columns) | No |
| Train/val/test splits | No | Optional | Yes (deterministic) |
| Export manifests | No | No | Yes |

---

## Path Contract

All paths are relative to `DATA_ROOT` (configured via env or `~/.koe/config.yaml`).

```
DATA_ROOT/
├── data/                    # Raw files (audio, archives)
│   └── ingest/
│       └── {dataset}/
│           ├── archives/    # Original downloaded archives
│           └── extracted/   # Extracted audio files
│
└── lake/                    # Delta Lake tables
    ├── bronze/
    │   └── {dataset}/
    │       └── utterances/  # Delta table
    ├── silver/
    │   └── {dataset}/
    │       └── utterances/  # Delta table
    └── gold/
        └── {dataset}/
            ├── utterances/  # Delta table
            └── manifests/   # JSONL exports
                └── {snapshot_id}.jsonl
```

### Audio path resolution

```
audio_abspath = DATA_ROOT / "data" / audio_relpath
```

The `audio_relpath` column is stable across mounts. Absolute paths are computed at export time.

---

## Naming Conventions

### Table names
```
lake/{layer}/{dataset}/{table}
```

Examples:
- `lake/bronze/jsut/utterances`
- `lake/silver/jvs/utterances`
- `lake/gold/common_voice/utterances`

### Partitioning

All utterance tables are partitioned by:
```
partitionBy=["dataset"]
```

Gold additionally partitions by:
```
partitionBy=["dataset", "split"]
```

---

## Primary Keys & Identifiers

| Column | Format | Purpose |
|--------|--------|---------|
| `utterance_id` | `sha1({dataset}\|{speaker_id}\|{subset}\|{corpus_utt_id})[:16]` | Stable deterministic PK for joins |
| `utterance_key` | `{dataset}_{subset}_{corpus_utt_id}` | Human-readable, for debugging |

### Invariants

1. `utterance_id` is **deterministic** - same inputs always produce same ID
2. `utterance_id` is **immutable** - never changes once assigned in bronze
3. All layers join on `utterance_id`

---

## Versioning & Lineage

Each layer tracks its provenance:

| Column | Layer | Purpose |
|--------|-------|---------|
| `ingest_version` | Bronze | Ingest pipeline version tag |
| `source_version` | Bronze | Corpus version (e.g., "v1.1") |
| `bronze_version` | Silver | Points to bronze Delta version |
| `silver_version` | Silver/Gold | Silver pipeline version tag |
| `gold_version` | Gold | Gold pipeline version tag |
| `snapshot_id` | Gold | Unique snapshot identifier |

### Snapshot ID format
```
{dataset}-{YYYYMMDD-HHMMSS}-{config_hash[:8]}
```

Example: `jsut-20260125-055016-fd76d53a`

---

## Schema Evolution

### Rules

1. **Adding columns**: Always allowed with nullable defaults
2. **Removing columns**: Deprecate first, remove in next major version
3. **Changing types**: Requires table rebuild (delete + recreate)
4. **Renaming columns**: Not allowed (add new, deprecate old)

### Backfill process

When schema changes require backfill:
1. Update schema definition in `modules/data_engineering/common/schemas/`
2. Bump pipeline version (`SILVER_VERSION`, etc.)
3. Delete existing table
4. Rerun pipeline

---

## Validation Invariants

### Bronze
- No duplicate `utterance_id`
- Required columns non-null: `utterance_id`, `utterance_key`, `dataset`, `speaker_id`, `subset`, `audio_relpath`, `text_raw`, `phonemes_source`
- `audio_relpath` points to existing file

### Silver
- Record count matches bronze (no drops without `exclude_reason`)
- `is_trainable=False` implies `exclude_reason` is set
- `label_status` in `{unlabeled, in_progress, labeled, rejected}`

### Gold
- `text` column is never null (coalesced from `text_norm` → `text_norm_raw` → `text_raw`)
- `split` in `{train, val, test}`
- Split percentages match config (within tolerance)
- Deterministic: same `seed` + `utterance_id` = same split assignment

---

## Phonemes Contract

### Canonical Inventory

The canonical phoneme inventory is derived from **JVS corpus** phonemes (OpenJTalk HTS format).
JVS phonemes == pyopenjtalk output (verified 100% match on 12,734 utterances).

**Inventory size:** 41 phonemes

| Category | Phonemes |
|----------|----------|
| Vowels | `a`, `i`, `u`, `e`, `o` |
| Devoiced vowels | `I`, `U` |
| Syllabic nasal | `N` |
| Geminate (glottal) | `cl` |
| Silence/pause | `sil`, `pau` |
| Basic consonants | `k`, `s`, `t`, `n`, `h`, `m`, `y`, `r`, `w`, `g`, `z`, `d`, `b`, `p`, `f`, `v`, `j` |
| Palatalized/affricate | `ky`, `sh`, `ch`, `ny`, `hy`, `my`, `ry`, `gy`, `by`, `py`, `dy`, `ty`, `ts` |

### Normalization Rules

| Rule | Description |
|------|-------------|
| Tokenize by whitespace | `"sil a i u sil"` → `["sil", "a", "i", "u", "sil"]` |
| Strip boundary `sil` | Remove leading/trailing `sil` markers |
| Keep internal `pau` | Preserve pause markers between words (meaningful prosody) |

Example:
```
Input:  "sil a pau i u sil"
Output: "a pau i u"
```

### Phoneme Sources

| Dataset | `phonemes_source` | `phonemes_method` | Notes |
|---------|-------------------|-------------------|-------|
| JVS | `lab_files` | `openjtalk_hts_trim_sil_v1` | HTS label files from corpus (100% match with pyopenjtalk) |
| JSUT | `none` | `pyopenjtalk_g2p_v0` | Generated via `koe silver jsut --phonemize` |

### Phoneme Column Semantics

| Column | Meaning |
|--------|---------|
| `phonemes_source` | Where the raw phonemes came from (`lab_files`, `none`) |
| `phonemes_raw` | Raw phonemes before normalization (Bronze only) |
| `phonemes` | Normalized phonemes (boundary `sil` stripped) |
| `phonemes_method` | Normalization method identifier |
| `phonemes_checked` | `True` if corpus-provided, `False` if auto-generated |

### Inventory Artifacts

Phoneme inventories are persisted for reproducibility:

```
data/assets/{dataset}/phoneme_inventory.json
```

Schema:
```json
{
  "dataset": "jvs",
  "layer": "silver",
  "source_table": "lake/silver/jvs/utterances",
  "phonemes_method": "openjtalk_hts_trim_sil_v1",
  "inventory_version": "v1",
  "created_at": "2026-01-25T08:00:00+00:00",
  "normalize_rules": {"strip_boundary_sil": true, "keep_internal_pau": true},
  "inventory": ["I", "N", "U", "a", ...],
  "counts": {"a": 123456, "pau": 78910, ...},
  "num_utterances_total": 14997,
  "num_utterances_with_phonemes": 12734,
  "coverage_all": 0.849,
  "coverage_trainable": 0.91,
  "inventory_hash": "sha1:..."
}
```

The `inventory_hash` enables detecting when the phoneme set changes.

### Invariants

1. **All phonemes in canonical inventory**: Every token in `phonemes` column is in `CANONICAL_INVENTORY`
2. **No boundary `sil`**: Normalized phonemes never start or end with `sil`
3. **JVS == pyopenjtalk**: JVS corpus phonemes match pyopenjtalk output exactly (after normalization)
4. **Deterministic generation**: Same `text_raw` always produces same `phonemes` output

---

## CLI Reference

```bash
# Ingest (download + extract)
koe ingest {dataset} [--force]

# Bronze (raw → Delta)
koe bronze {dataset} [--force]

# Silver (QC + enrichment)
koe silver {dataset} [--force]

# Gold (splits + manifest)
koe gold {dataset} \
  [--seed 42] \
  [--train-pct 0.90] \
  [--val-pct 0.10] \
  [--test-pct 0.00] \
  [--min-duration 0.5] \
  [--max-duration 20.0] \
  [--snapshot-id TEXT] \
  [--no-write-delta] \
  [--manifest-out PATH]

# Full pipeline
koe build {dataset} [--skip-ingest] [--skip-bronze] [--skip-silver]
```

---

## Table Specs

See detailed column-level specs:

- [bronze_utterances.md](../../modules/data_engineering/common/schemas/bronze_utterances.md)
- [silver_utterances.md](../../modules/data_engineering/common/schemas/silver_utterances.md)
- [gold_utterances.md](../../modules/data_engineering/common/schemas/gold_utterances.md)

---

## Schema Hash

Each table has a deterministic schema hash for change detection:

```python
schema_hash = sha256(json(schema))[:12]
```

The hash is:
- Computed from the PySpark StructType JSON representation
- Stored in the catalog table (`lake/_catalog/tables`)
- Used to detect schema drift between runs

When `live_schema_hash != catalog.schema_hash`, you have schema drift.

---

## Catalog

The catalog provides metastore-like discoverability without external infrastructure.

**Location:** `lake/_catalog/tables` (Delta table)

**Usage:**
```bash
koe catalog list                         # List all tables
koe catalog list --layer bronze          # Filter by layer
koe catalog describe bronze.jsut.utterances
koe catalog refresh bronze.jsut.utterances
```

Tables are auto-registered when pipelines run via `write_table()`.

---

## Reprocessing Semantics

This lakehouse uses **overwrite-latest** semantics:

| Layer | Semantics | Notes |
|-------|-----------|-------|
| Bronze | Overwrite | Re-ingest replaces table |
| Silver | Overwrite | Re-process replaces table |
| Gold | Overwrite + Snapshots | Table is overwritten, but JSONL manifests are immutable per `snapshot_id` |

**Key implications:**
- Gold manifests (`lake/gold/{dataset}/manifests/{snapshot_id}.jsonl`) are immutable
- Same `snapshot_id` = same exact training data (reproducible)
- Delta table shows "latest" view; use manifests for reproducibility
- Delta time-travel available for recent history

**Future consideration:** Move to append-only with `processed_at` partitioning if we need full history.

---

## Data Quality Checks

### Bronze

| Check | Expression |
|-------|------------|
| No duplicate IDs | `df.select("utterance_id").distinct().count() == df.count()` |
| Required non-null | `utterance_id`, `utterance_key`, `dataset`, `speaker_id`, `subset`, `corpus_utt_id`, `audio_relpath`, `text_raw`, `phonemes_source`, `ingested_at` |
| Duration valid | `duration_sec > 0` |
| Audio exists | `audio_relpath` resolves to existing file (optional check) |

### Silver

| Check | Expression |
|-------|------------|
| Count matches bronze | `silver.count() == bronze.count()` |
| Exclusion reason | `is_trainable=False` ⇒ `exclude_reason IS NOT NULL` |
| Phonemes checked | `phonemes_checked=True` ⇒ `phonemes IS NOT NULL` |
| Label status enum | `label_status IN ('unlabeled', 'in_progress', 'labeled', 'rejected')` |
| Split enum or null | `split IN ('train', 'val', 'test') OR split IS NULL` |

### Gold

| Check | Expression |
|-------|------------|
| Text never null | `df.filter(col("text").isNull()).count() == 0` |
| Audio path never null | `df.filter(col("audio_relpath").isNull()).count() == 0` |
| Split enum | `split IN ('train', 'val', 'test')` |
| Split distribution | Within 2% of expected percentages |
| Snapshot ID set | `snapshot_id IS NOT NULL` (in manifest) |

---

## Browsing Data

Since we don't have a metastore, here's how to inspect tables:

```python
from modules.data_engineering.common.spark import get_spark
from modules.data_engineering.common.paths import paths

spark = get_spark()

# Read a Delta table
df = spark.read.format("delta").load(str(paths.silver / "jsut" / "utterances"))

# Show schema
df.printSchema()

# Count by subset/split
df.groupBy("subset").count().show()
df.groupBy("split").count().show()

# Sample rows
df.select("utterance_key", "text_raw", "duration_sec").show(5, truncate=False)

# Check for nulls in key columns
from pyspark.sql import functions as F
df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()
```

Or use the catalog CLI:
```bash
koe catalog list
koe catalog describe silver.jsut.utterances
```
