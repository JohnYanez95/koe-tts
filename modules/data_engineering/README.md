# Data Engineering Module

ETL pipelines for TTS training data using Delta Lake medallion architecture.

## Architecture

```
Bronze (raw)  →  Silver (curated)  →  Gold (training-ready)
```

See [Lakehouse Contract](../../docs/lakehouse/CONTRACT.md) for full architecture documentation.

## Quick Start

```bash
# Full pipeline for a dataset
koe build jsut

# Or step by step
koe ingest jsut    # Download + extract
koe bronze jsut    # Raw → Delta
koe silver jsut    # QC + enrichment
koe gold jsut      # Splits + manifest
```

## Directory Structure

```
modules/data_engineering/
├── common/
│   ├── io.py           # Delta read/write utilities
│   ├── paths.py        # Path resolution (DATA_ROOT, lake paths)
│   ├── spark.py        # SparkSession management
│   └── schemas/
│       ├── __init__.py
│       ├── bronze.py           # Bronze StructType
│       ├── silver.py           # Silver StructType + defaults
│       ├── gold.py             # Gold StructType
│       ├── bronze_utterances.md  # Column spec
│       ├── silver_utterances.md  # Column spec
│       └── gold_utterances.md    # Column spec
│
├── ingest/
│   ├── jsut/           # JSUT download + extraction
│   ├── jvs/            # JVS (TODO)
│   └── common_voice/   # Common Voice (TODO)
│
├── bronze/
│   ├── cli.py          # koe bronze CLI
│   └── jsut.py         # JSUT bronze pipeline
│
├── silver/
│   ├── cli.py          # koe silver CLI
│   └── jsut.py         # JSUT silver pipeline
│
├── gold/
│   ├── cli.py          # koe gold CLI
│   └── jsut.py         # JSUT gold pipeline
│
└── pipelines/
    └── build_dataset.py  # Full pipeline orchestration
```

## CLI Reference

### Ingest

```bash
koe ingest <dataset> [--force]
```

Downloads and extracts raw corpus files.

### Bronze

```bash
koe bronze <dataset> [--force]
```

Converts raw files to Delta table with full provenance.

### Silver

```bash
koe silver <dataset> [--force]
```

Adds QC columns, normalization, phonemes, labeling hooks.

### Gold

```bash
koe gold <dataset> \
  [--seed 42] \
  [--train-pct 0.90] \
  [--val-pct 0.10] \
  [--test-pct 0.00] \
  [--min-duration 0.5] \
  [--max-duration 20.0] \
  [--snapshot-id TEXT] \
  [--write-delta/--no-write-delta] \
  [--manifest-out PATH]
```

Creates training-ready dataset with deterministic splits.

### Build (Full Pipeline)

```bash
koe build <dataset> \
  [--skip-ingest] \
  [--skip-bronze] \
  [--skip-silver]
```

## Table Specs

| Layer | Columns | Spec |
|-------|---------|------|
| Bronze | 23 | [bronze_utterances.md](common/schemas/bronze_utterances.md) |
| Silver | 40 | [silver_utterances.md](common/schemas/silver_utterances.md) |
| Gold | 15 | [gold_utterances.md](common/schemas/gold_utterances.md) |

## Adding a New Dataset

1. **Ingest**: Create `ingest/{dataset}/download.py` with extraction logic
2. **Bronze**: Create `bronze/{dataset}.py` implementing `build_bronze_{dataset}()`
3. **Silver**: Create `silver/{dataset}.py` implementing `build_silver_{dataset}()`
4. **Gold**: Create `gold/{dataset}.py` implementing `build_gold_{dataset}()`
5. **CLI**: Register in each layer's `cli.py`

Use JSUT as the reference implementation.

## Development

```bash
# Run specific pipeline
python -m modules.data_engineering.bronze.cli jsut

# Check schema
python -c "from modules.data_engineering.common.schemas import SILVER_UTTERANCES_SCHEMA; print(len(SILVER_UTTERANCES_SCHEMA.fields))"

# Inspect Delta table
python -c "
from modules.data_engineering.common.spark import get_spark
from modules.data_engineering.common.paths import paths
spark = get_spark()
df = spark.read.format('delta').load(str(paths.silver / 'jsut' / 'utterances'))
df.printSchema()
"
```
