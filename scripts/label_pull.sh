#!/usr/bin/env bash
# Pull a batch of utterances for labeling
# Usage: ./scripts/label_pull.sh --dataset jsut --batch-size 100
#        ./scripts/label_pull.sh --query "phoneme_confidence < 0.8" --batch-size 50

set -e
python -m modules.labeler.pipelines.pull_batch "$@"
