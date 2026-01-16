#!/bin/bash
# Download JSUT corpus - the recommended starting dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/raw/jsut"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading JSUT v1.1 corpus ==="
echo "Target: $DATA_DIR"

if [ -d "jsut_ver1.1" ]; then
    echo "JSUT already exists, skipping download."
else
    wget -q --show-progress http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    echo "Extracting..."
    unzip -q jsut_ver1.1.zip
    rm jsut_ver1.1.zip
fi

echo ""
echo "=== JSUT Download Complete ==="
echo "Contents:"
ls -la jsut_ver1.1/

echo ""
echo "Subsets available:"
ls jsut_ver1.1/

echo ""
echo "Sample counts:"
for dir in jsut_ver1.1/*/; do
    count=$(find "$dir" -name "*.wav" 2>/dev/null | wc -l)
    echo "  $(basename $dir): $count wav files"
done
