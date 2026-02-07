# 声 koe-tts

Japanese Text-to-Speech training pipeline with Delta Lake.

## Overview

A modular TTS training system built on:
- **Delta Lake**: Versioned, ACID-compliant data lakehouse
- **PySpark**: Scalable data processing
- **PyTorch Lightning**: Training infrastructure
- **Piper/VITS**: Phoneme-authoritative synthesis

## Project Structure

```
koe-tts/
├── modules/                  # All Python code (importable packages)
│   ├── platform/             # Shared Spark/Delta infrastructure
│   │   ├── paths.py          # Centralized path management
│   │   ├── spark.py          # Spark session factory
│   │   └── delta.py          # Delta read/write operations
│   ├── data_engineering/     # ETL pipelines
│   │   ├── ingest/           # Download and stage raw data
│   │   ├── bronze/           # Raw data ingestion
│   │   ├── silver/           # Cleaned, normalized data
│   │   └── gold/             # ML-ready features
│   ├── training/             # Model training code
│   └── labeler/              # Labeling application
├── configs/                  # Configuration files
│   └── paths.yaml            # DATA_ROOT configuration
├── scripts/                  # CLI entrypoints
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
├── docker/                   # Docker configurations
├── legacy/                   # Frozen v1 codebase
├── Makefile                  # Common commands
└── pyproject.toml            # Package configuration
```

## Data Architecture

The lakehouse lives at `$DATA_ROOT` (configurable):

```
$DATA_ROOT/
├── data/
│   ├── raw/                  # Downloaded corpus archives
│   └── staging/              # Unpacked files before bronze
├── lake/
│   ├── bronze/               # Raw data, append-only
│   ├── silver/               # Cleaned, typed, deduplicated
│   └── gold/                 # ML-ready features
├── runs/
│   └── train/{run_name}/     # Training outputs
│       ├── checkpoints/
│       ├── logs/
│       ├── samples/
│       └── config.yaml
└── models/                   # Curated model registry
```

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/JohnYanez95/koe-tts.git
cd koe-tts
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 2. Configure data root
export KOE_DATA_ROOT=/path/to/your/data
# Or edit configs/paths.yaml

# 3. Verify setup
make test
```

## Usage

```python
from modules.platform import paths, read_table, write_table, get_spark

# Access paths
paths.bronze / "jsut"
paths.train_run("experiment_v1").checkpoints

# Read/write Delta tables
df = read_table("silver", "utterances")
write_table(df, "gold", "training_set", mode="overwrite")

# Spark session
spark = get_spark()
```

## Development

```bash
# Run tests
make test

# Lint and format
make lint
make format

# Run specific pipeline
python -m modules.data_engineering.bronze.jsut

# Monitor training dashboard
koe monitor --latest
```

## Documentation

- [Postmortems](./docs/postmortems/) — Incident reports for training failures

## Legacy Code

The `legacy/` directory contains the frozen v1/v2 codebase:
- XTTS-v2 fine-tuning
- Piper Plus training
- Original preprocessing scripts

See `legacy/README.md` for documentation.

## License

MIT
