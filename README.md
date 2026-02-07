# 声 koe-tts

From-scratch Japanese Text-to-Speech engine built on VITS and Delta Lake.

## Overview

A complete TTS training system featuring:
- **VITS synthesis**: End-to-end TTS with variational inference and adversarial training
- **4-stage training**: baseline (mel) → duration → vits_core → vits_gan
- **Delta Lake pipeline**: Medallion architecture (Bronze → Silver → Gold)
- **Training dashboard**: Live metrics, GAN stability monitoring, control plane
- **Segmentation labeler**: Pause boundary refinement with waveform visualization

Supported corpora: JSUT (single speaker), JVS (100 speakers), Common Voice.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/JohnYanez95/koe-tts.git
cd koe-tts
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For training with CUDA
make train-install-cu124  # or cu121

# Configure data paths (optional - defaults to repo root)
export KOE_DATA_ROOT=/path/to/data
export KOE_LOCAL_ROOT=/path/to/fast/ssd

# Verify setup
make test
```

## Usage

### Data Pipeline

```bash
koe build jsut                    # Full pipeline: ingest → bronze → silver → gold

# Or step by step:
koe ingest jsut                   # Download + extract corpus
koe bronze jsut                   # Raw → Delta table
koe silver jsut                   # QC + phonemes
koe gold jsut                     # Splits + JSONL manifest
```

### Training

```bash
koe cache create jsut             # Create training cache from gold
koe train smoke-test jsut         # Verify pipeline works

koe train baseline jsut           # Stage 1: mel prediction
koe train duration jsut           # Stage 2: duration prediction
koe train vits jsut --stage core  # Stage 3: VITS reconstruction
koe train vits jsut --stage gan   # Stage 4: VITS with adversarial training
```

### Monitoring & Synthesis

```bash
koe monitor --latest              # Launch training dashboard
koe synth jsut -r <run_id> --text "こんにちは"
```

### Segmentation Labeling

```bash
koe label serve jsut              # Launch labeling UI on localhost:8081
```

## Project Structure

```
koe-tts/
├── modules/
│   ├── data_engineering/         # ETL with medallion architecture
│   │   ├── common/               # Paths, Spark, schemas, audio DSP
│   │   ├── ingest/               # Download + extract (jsut/, jvs/)
│   │   ├── bronze/               # Raw → Delta
│   │   ├── silver/               # QC, phonemes, segments
│   │   └── gold/                 # Training manifests (JSONL)
│   ├── training/
│   │   ├── models/               # VITS, baseline, duration models
│   │   ├── dataloading/          # Dataset, collator, cache management
│   │   ├── pipelines/            # train_*, eval_*, synthesize
│   │   └── common/               # Checkpoints, GAN controller, metrics
│   ├── dashboard/                # FastAPI + React training monitor
│   └── labeler/                  # Segmentation labeling app
├── configs/
│   ├── training/                 # baseline, duration, vits_core, vits_gan
│   ├── datasets/                 # jsut, jvs, multi
│   └── lakehouse/                # paths, bronze, silver, gold
├── scripts/                      # CLI (Typer-based)
├── docs/
│   ├── postmortems/              # Training incident reports
│   └── training/                 # GAN controller reference
└── tests/
```

## Data Flow

```
ingest → bronze (raw Delta) → silver (QC + phonemes) → gold (splits + JSONL) → cache → training
```

Storage paths derive from `$KOE_DATA_ROOT` (archive) and `$KOE_LOCAL_ROOT` (fast SSD for active training).

## Configuration

- **Training**: `configs/training/{baseline,duration,vits_core,vits_gan}.yaml`
- **Datasets**: `configs/datasets/{jsut,jvs,multi}.yaml`
- **Paths**: `configs/lakehouse/paths.yaml`

## Documentation

- [GAN Controller](./docs/training/GAN_CONTROLLER.md) — Stability monitoring and mitigations
- [Postmortems](./docs/postmortems/) — Training incident reports and learnings

## License

MIT
