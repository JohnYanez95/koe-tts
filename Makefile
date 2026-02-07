# =============================================================================
# koe-tts Makefile
# =============================================================================

.PHONY: help install dev-install train-install train-install-cu121 train-install-cu124 torch-cu121 torch-cu124 test lint format clean spark-shell smoke-test load-batch-test train-baseline train-baseline-quick dashboard-typecheck dashboard-build

# Default target
help:
	@echo "koe-tts development commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install             Install base package (data engineering, no torch)"
	@echo "  make dev-install         Install with dev dependencies"
	@echo "  make train-install       Install CUDA 12.1 torch + training deps (default)"
	@echo "  make train-install-cu121 Install with CUDA 12.1 (Ubuntu 22.04)"
	@echo "  make train-install-cu124 Install with CUDA 12.4 (newer drivers)"
	@echo "  make verify-train        Verify training environment (torch, torchaudio)"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run test suite"
	@echo "  make lint           Run linter (ruff check)"
	@echo "  make format         Format code (ruff format)"
	@echo "  make clean          Remove build artifacts"
	@echo "  make spark-shell    Start interactive PySpark shell"
	@echo "  make dashboard-build Build dashboard UI (for koe monitor)"
	@echo ""
	@echo "Training:"
	@echo "  make smoke-test       Run training pipeline smoke test"
	@echo "  make load-batch-test  Verify dataloading pipeline (Stage 1)"
	@echo ""
	@echo "Data Engineering:"
	@echo "  make bronze-jsut    Ingest JSUT to bronze layer"
	@echo "  make bronze-jvs     Ingest JVS to bronze layer"
	@echo ""
	@echo "Configuration:"
	@echo "  DATA_ROOT is set via:"
	@echo "    1. KOE_DATA_ROOT env var"
	@echo "    2. configs/lakehouse/paths.yaml"
	@echo "    3. Default: repo root"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install -e .
	pip install --upgrade pip

dev-install:
	pip install -e ".[dev]"

# PyTorch with CUDA - run before train-install
torch-cu121:
	pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

torch-cu124:
	pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Training install - includes CUDA torch + lightning
# Use train-install-cu121 or train-install-cu124 depending on your CUDA version
train-install: train-install-cu121

train-install-cu121: torch-cu121
	pip install -e ".[train]"
	@echo "=== PyTorch CUDA verification ==="
	python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"

train-install-cu124: torch-cu124
	pip install -e ".[train]"
	@echo "=== PyTorch CUDA verification ==="
	python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"

# Verify training environment is correctly configured
verify-train:
	@echo "=== Training Environment Verification ==="
	@python -c "import sys; print('Python:', sys.version.split()[0])"
	@python -c "import torch; print('torch:', torch.__version__); print('  cuda available:', torch.cuda.is_available()); print('  cuda version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" || (echo "torch: NOT INSTALLED - run: make train-install" && exit 1)
	@python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)" || (echo "torchaudio: NOT INSTALLED - run: make train-install" && exit 1)
	@python -c "import pytorch_lightning; print('pytorch_lightning:', pytorch_lightning.__version__)" 2>/dev/null || echo "pytorch_lightning: not installed (optional)"
	@echo "=== Ready for training ==="

# Quick smoke test of training pipeline
smoke-test:
	python scripts/cli.py train smoke-test jsut --max-utterances 50

# Stage 1 verification: load one batch from cache
load-batch-test:
	python -m modules.training.pipelines.load_batch_test --dataset jsut

# Stage 2: train baseline model (quick test)
train-baseline-quick:
	python scripts/cli.py train baseline jsut --max-steps 500 --batch-size 8

# Stage 2: train baseline model (full run)
train-baseline:
	python scripts/cli.py train baseline jsut --config configs/training/baseline.yaml

# Stage 3: evaluate baseline model (requires RUN_ID env var)
# Usage: RUN_ID=jsut_baseline_mel_20260125_044418 make eval-baseline
eval-baseline:
	python scripts/cli.py train eval baseline jsut --run-id $(RUN_ID) --write-audio

# =============================================================================
# Testing & Linting
# =============================================================================

test:
	pytest tests/ -v

test-spark:
	pytest tests/ -v -m spark

lint:
	ruff check modules/ scripts/ tests/

format:
	ruff format modules/ scripts/ tests/
	ruff check --fix modules/ scripts/ tests/

typecheck:
	mypy modules/

# Dashboard frontend typecheck and build
dashboard-typecheck:
	cd modules/dashboard/frontend && npm run build

dashboard-build:
	cd modules/dashboard/frontend && npm ci && npm run build
	@echo "Dashboard UI built. Run: koe monitor --latest"

# =============================================================================
# Data Engineering
# =============================================================================

bronze-jsut:
	python -m modules.data_engineering.bronze.jsut

bronze-jvs:
	python -m modules.data_engineering.bronze.jvs

silver-utterances:
	python -m modules.data_engineering.silver.utterances

# =============================================================================
# Development Tools
# =============================================================================

spark-shell:
	@echo "Starting PySpark shell with Delta Lake..."
	pyspark \
		--packages io.delta:delta-spark_2.12:3.0.0 \
		--conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
		--conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"

jupyter:
	jupyter lab --notebook-dir=.

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-spark:
	rm -rf spark-warehouse/
	rm -rf metastore_db/
	rm -f derby.log

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
