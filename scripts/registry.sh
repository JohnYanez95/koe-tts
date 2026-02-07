#!/usr/bin/env bash
# MLflow model registry operations
# Usage: ./scripts/registry.sh list-models
#        ./scripts/registry.sh list-versions tts-ja-vits
#        ./scripts/registry.sh promote tts-ja-vits --from-alias best

set -e
python -m modules.registry.cli "$@"
