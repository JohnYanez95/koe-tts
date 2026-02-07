#!/usr/bin/env bash
# Create local cache snapshot from gold manifest
# Usage: ./scripts/train_cache.sh jsut --snapshot-id 20240115_v1
#        ./scripts/train_cache.sh jsut --latest

set -e
python -m modules.training.dataloading.cli cache "$@"
