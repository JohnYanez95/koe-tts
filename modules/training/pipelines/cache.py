"""
Cache pipeline: materialize local cache from gold manifest.

Usage:
    python -m modules.training.pipelines.cache --gold-version jsut_v1 --mode symlink
"""

import argparse
from pathlib import Path

from modules.training.dataloading.cache_snapshot import cache_snapshot


def main():
    parser = argparse.ArgumentParser(description="Cache gold data locally")
    parser.add_argument("--gold-version", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--cache-path", type=Path)

    args = parser.parse_args()

    result = cache_snapshot(
        gold_version=args.gold_version,
        mode=args.mode,
        cache_path=args.cache_path,
    )

    print(f"Cache result: {result}")


if __name__ == "__main__":
    main()
