#!/usr/bin/env bash
# Ingest a dataset into the bronze layer
# Usage: ./scripts/de_ingest.sh jsut
#        ./scripts/de_ingest.sh jvs --version v1.0

set -e
python -m modules.data_engineering.ingest.cli "$@"
