#!/usr/bin/env bash
# Build gold layer (train/val manifests)
# Usage: ./scripts/de_gold.sh jsut
#        ./scripts/de_gold.sh jsut --version 20240115

set -e
python -m modules.data_engineering.gold.cli "$@"
