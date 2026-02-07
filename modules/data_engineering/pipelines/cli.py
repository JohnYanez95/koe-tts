"""
Main CLI for data engineering pipelines.

Usage:
    python -m modules.data_engineering.pipelines.cli build-dataset --dataset jsut
    python -m modules.data_engineering.pipelines.cli refresh-gold --gold-version v2
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Data engineering pipelines",
        prog="python -m modules.data_engineering.pipelines.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-dataset command
    build_parser = subparsers.add_parser(
        "build-dataset",
        help="Build dataset end-to-end (ingest → bronze → silver → gold)",
    )
    build_parser.add_argument(
        "--dataset",
        choices=["jsut", "jvs", "common_voice", "all"],
        default="jsut",
    )
    build_parser.add_argument("--version", default="v1")
    build_parser.add_argument("--gold-version")
    build_parser.add_argument("--skip-ingest", action="store_true")
    build_parser.add_argument("--skip-bronze", action="store_true")
    build_parser.add_argument("--skip-silver", action="store_true")
    build_parser.add_argument("--skip-gold", action="store_true")

    # refresh-gold command
    refresh_parser = subparsers.add_parser(
        "refresh-gold",
        help="Refresh gold layer only (new splits/sampling rules)",
    )
    refresh_parser.add_argument("--gold-version", required=True)
    refresh_parser.add_argument("--silver-version", type=int)

    args = parser.parse_args()

    if args.command == "build-dataset":
        from .build_dataset import build_dataset
        build_dataset(
            dataset=args.dataset,
            version=args.version,
            gold_version=args.gold_version,
            skip_ingest=args.skip_ingest,
            skip_bronze=args.skip_bronze,
            skip_silver=args.skip_silver,
            skip_gold=args.skip_gold,
        )
    elif args.command == "refresh-gold":
        from .refresh_gold import refresh_gold
        refresh_gold(
            gold_version=args.gold_version,
            silver_version=args.silver_version,
        )


if __name__ == "__main__":
    main()
