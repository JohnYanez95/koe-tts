#!/usr/bin/env bash
# Process dataset to silver layer (validation, cleaning)
# Usage: ./scripts/de_silver.sh jsut

set -e
python -m modules.data_engineering.silver.cli "$@"
