#!/usr/bin/env bash
# Train a TTS model
# Usage: ./scripts/train.sh vits --dataset jsut --run-name jsut-vits-v1
#        ./scripts/train.sh vits --config configs/training/vits.yaml

set -e
python -m modules.training.pipelines.train "$@"
