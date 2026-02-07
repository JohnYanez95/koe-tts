#!/usr/bin/env bash
# Run full data engineering pipeline (ingest -> bronze -> silver -> gold)
# Usage: ./scripts/de_build_dataset.sh jsut
#        ./scripts/de_build_dataset.sh jsut --skip-ingest

set -e
python -m modules.data_engineering.pipelines.cli build "$@"
