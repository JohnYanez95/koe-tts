#!/usr/bin/env bash
# Process dataset to bronze layer
# Usage: ./scripts/de_bronze.sh jsut

set -e
python -m modules.data_engineering.bronze.cli "$@"
