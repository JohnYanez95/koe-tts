# 声 koe-tts

Japanese Text-to-Speech model training and inference pipeline.

## Overview

Fine-tune open-source TTS models (MeloTTS, XTTS-v2) on Japanese speech data to build custom voice synthesis systems.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Verify setup
python scripts/verify_setup.py
```

## Data

See [`data/DATASETS.md`](data/DATASETS.md) for dataset sources and download instructions.

```bash
# Download JSUT corpus (~2.5GB)
bash scripts/download_jsut.sh
```

## Project Structure

```
koe-tts/
├── configs/           # Training configurations
├── data/              # Datasets (not tracked in git)
│   ├── raw/           # Downloaded datasets
│   ├── processed/     # Preprocessed audio
│   └── DATASETS.md    # Dataset documentation
├── models/            # Checkpoints and exports (not tracked)
├── scripts/           # Utility scripts
├── src/
│   ├── preprocessing/ # Audio and text processing
│   ├── training/      # Training loops
│   └── inference/     # Model inference
└── notebooks/         # Experiments
```

## Usage

### Preprocessing

```python
from src.preprocessing.japanese_text import text_to_phonemes
from src.preprocessing.audio import process_audio_file

# Convert Japanese text to phonemes
phonemes = text_to_phonemes("こんにちは")

# Process audio file
process_audio_file("input.wav", "output.wav", target_sr=22050)
```

### Training

```bash
# (Training scripts coming soon)
python src/training/train.py --config configs/default.yaml
```

## Hardware Requirements

- **Minimum**: RTX 3090 (24GB VRAM) for fine-tuning
- **Recommended**: 2-4 GPUs for faster iteration
- **RAM**: 32GB+ (64GB recommended)
- **Storage**: 100GB+ for datasets and checkpoints

## License

Apache 2.0
