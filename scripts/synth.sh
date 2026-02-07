#!/usr/bin/env bash
# Synthesize audio from a trained model
# Usage: ./scripts/synth.sh --checkpoint models/checkpoints/jsut-vits/best.ckpt --text "こんにちは"
#        ./scripts/synth.sh --model-name tts-ja-vits --alias prod --text-file sentences.txt

set -e
python -m modules.training.pipelines.synthesize "$@"
