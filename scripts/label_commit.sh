#!/usr/bin/env bash
# Commit labeled data back to the lake
# Usage: ./scripts/label_commit.sh --batch-id batch_20240115_001

set -e
python -m modules.labeler.pipelines.write_labels "$@"
