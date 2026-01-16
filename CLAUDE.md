# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

koe-tts (声) is a Japanese Text-to-Speech training pipeline for fine-tuning open-source TTS models (MeloTTS, XTTS-v2) on Japanese speech data.

## Commands

```bash
# Install (editable with dev deps)
pip install -e ".[dev]"

# Verify setup - checks Python, deps, CUDA, directories, data
python scripts/verify_setup.py

# Download JSUT corpus (~2.5GB)
bash scripts/download_jsut.sh

# Run tests
pytest

# Training (not yet implemented)
python src/training/train.py --config configs/default.yaml
```

## Architecture

**Preprocessing pipeline** (`src/preprocessing/`):
- `audio.py` - Audio loading, resampling, normalization, silence trimming (uses librosa/soundfile)
- `japanese_text.py` - Text normalization (jaconv), phoneme conversion (pyopenjtalk), sentence splitting

**Configuration** (`configs/default.yaml`):
- Audio params: 22050 Hz, 80 mel bins, 256 hop length
- Training params: batch size 8, mixed precision, gradient checkpointing for 24GB VRAM
- Model options: melotts, xtts, vits

**Data flow**:
1. Raw audio in `data/raw/` (e.g., JSUT corpus)
2. Preprocessed to `data/processed/` (normalized, trimmed, resampled to 22kHz)
3. Checkpoints to `models/checkpoints/`

## Japanese Text Processing

Uses pyopenjtalk for grapheme-to-phoneme conversion with accent information. Text is normalized via jaconv (fullwidth ↔ halfwidth conversion).

```python
from src.preprocessing.japanese_text import text_to_phonemes
phonemes = text_to_phonemes("こんにちは")  # Returns phoneme string
```

## Hardware

Minimum: RTX 3090 (24GB VRAM). Config defaults are tuned for this with gradient accumulation (effective batch 32).
