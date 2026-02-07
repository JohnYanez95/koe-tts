# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

koe-tts (声) is a Japanese Text-to-Speech training pipeline built on Delta Lake and PyTorch Lightning. It implements a medallion lakehouse architecture (Bronze → Silver → Gold) for data processing and supports VITS-based synthesis with multi-speaker conditioning.

Supported corpora: JSUT (single speaker), JVS (100 speakers), Common Voice.

## Commands

**Always use the virtual environment:**
```bash
source .venv/bin/activate
```

**Installation:**
```bash
pip install -e ".[dev]"                    # Development (no torch)
make train-install-cu121                   # Training with CUDA 12.1
make train-install-cu124                   # Training with CUDA 12.4
```

**Testing & Linting:**
```bash
make test                                  # Run pytest
make lint                                  # ruff check
make format                                # ruff format + fix
make verify-train                          # Check torch/CUDA setup
```

**Data Pipeline (full workflow):**
```bash
koe build jsut                             # Full pipeline: ingest → bronze → silver → gold

# Or step by step:
koe ingest jsut                            # Download + extract
koe bronze jsut                            # Raw → Delta table
koe silver jsut                            # QC + normalization + phonemes
koe gold jsut                              # Splits + JSONL manifest
```

**Segmentation (Tier 1 - unlabeled segments):**
```bash
koe segment auto jsut                      # Detect pause breakpoints
koe segment build jsut                     # Build segment manifests
koe segment list jsut                      # Show inventory
```

**Training:**
```bash
koe cache create jsut                      # Create training cache from gold
koe train smoke-test jsut                  # Verify pipeline

koe train baseline jsut                    # Stage 1: mel prediction
koe train duration jsut                    # Stage 2: duration prediction
koe train vits jsut --stage core           # Stage 3: VITS reconstruction
koe train vits jsut --stage gan            # Stage 3: VITS with GAN

koe train eval vits jsut --run-id <id>     # Evaluate checkpoint
```

**Synthesis:**
```bash
koe synth jsut -r <run_id> --text "こんにちは"
koe eval-multispeaker <run_id>             # Multi-speaker grid
koe probe-speaker <run_id>                 # Speaker embedding sanity check
```

**Dashboard:**
```bash
koe monitor --latest                       # Monitor most recent run
koe monitor <run_id>                       # Monitor specific run
```

## Architecture

### Module Structure

```
modules/
├── data_engineering/          # ETL with medallion architecture
│   ├── common/                # Shared: paths, spark, schemas, audio DSP
│   ├── ingest/                # Download + extract (jsut/, jvs/)
│   ├── bronze/                # Raw → Delta (schema enforcement)
│   ├── silver/                # QC, normalization, phonemes, splits, segments
│   ├── gold/                  # Training manifests (JSONL), segments
│   └── catalog/               # Table discovery
│
├── training/
│   ├── audio/                 # Mel extraction, vocoder utilities
│   ├── dataloading/           # TTSDataset, TTSCollator, cache management
│   ├── models/                # VITS, baseline, duration models
│   ├── pipelines/             # train_*, eval_*, synthesize entrypoints
│   └── common/                # Checkpoints, logging, control plane
│
└── dashboard/
    ├── backend.py             # FastAPI + SSE streaming
    └── frontend/              # React + TypeScript (Vite)
```

### Data Flow

1. **Ingest**: Download corpus → `data/ingest/{dataset}/`
2. **Bronze**: Raw files → `lake/bronze/{dataset}/` (Delta table)
3. **Silver**: QC + phonemes → `lake/silver/{dataset}/` (Delta table)
4. **Gold**: Training splits → `lake/gold/{dataset}/manifests/*.jsonl`
5. **Cache**: Local copy → `data/cache/{dataset}/{snapshot}/`
6. **Training**: Cache → model checkpoints in `runs/{run_id}/`

### Key Patterns

**Path Management**: All paths derive from `DATA_ROOT` (env `KOE_DATA_ROOT` or `configs/lakehouse/paths.yaml`):
```python
from modules.data_engineering.common.paths import paths
paths.gold / "jsut" / "manifests"
paths.cache / "jsut" / "latest"
```

**Spark + Delta**: Tables use PySpark with Delta Lake for ACID guarantees:
```python
from modules.data_engineering.common.spark import get_spark
from modules.data_engineering.common.io import write_table
spark = get_spark()
write_table(df, layer="silver", table_name="jsut/utterances", mode="overwrite")
```

**CLI Structure**: Typer-based with sub-apps (`train_app`, `cache_app`, `segment_app`) in `scripts/cli.py`.

**Training Stages**: Progressive complexity (baseline → duration → VITS core → VITS GAN).

**Segment Training (Tier 1)**: Segments are always unlabeled (`segment_label_status="unlabeled"`). They reference parent audio via `start_ms`/`end_ms` (no audio duplication). Dataset slices audio after resampling; collator returns `phonemes=None` for unlabeled batches.

### Dashboard Control Plane

The training dashboard supports safe command injection via nonce-based idempotency:
- Backend: `modules/dashboard/backend.py` with SSE streaming
- Frontend: `modules/dashboard/frontend/` (React + Vite)
- Control integration: `modules/training/common/control.py`

## Configuration

**Training configs**: `configs/training/{baseline,duration,vits_core,vits_gan}.yaml`

**Dataset configs**: `configs/datasets/{jsut,jvs,multi}.yaml`

**Lakehouse paths**: `configs/lakehouse/paths.yaml`

## Japanese Text Processing

Uses pyopenjtalk for G2P with accent. Phonemes are stored in Silver layer and promoted to Gold manifests:
```python
from modules.data_engineering.common.phonemes import tokenize, CANONICAL_INVENTORY
```

## GAN Training Stability

The GAN controller (`modules/training/common/gan_controller.py`) monitors training health and applies mitigations. See `docs/training/GAN_CONTROLLER.md` for full reference.

**Key baseline metrics (from healthy training):**
- Normal `g_clip_coef`: 0.03–0.07
- Normal `d_clip_coef`: 0.06–0.10
- Normal grad norms: G=20–40, D=10–20

**Known danger bands:**
- 10k–12k: Post-disc_start instability (mitigated by adv_ramp)
- 27k–30k: KL spikes + grad volatility (KL > 3 is warning sign)

**Config gotcha:** All controller fields in `vits_gan.yaml` must be explicitly mapped in `train_vits.py`'s `GANControllerConfig()` constructor, or they'll use code defaults.

## Postmortems

Training incidents are documented in `docs/postmortems/`:
- Per-incident reports: `YYYY-MM-DD_<run_id>_<slug>.md`
- Rolling trends: `gan_stability_log.md`
- Incident types: `numeric_instability`, `infra_crash`, `trigger_miscalibration`

## Infrastructure (WSL2)

- GPU driver is on **Windows** — update via GeForce Experience or nvidia.com
- After driver update: `wsl --shutdown` then reopen
- KVM switches can crash GPU — avoid switching during training
- Keep drivers current (check with `nvidia-smi`)
