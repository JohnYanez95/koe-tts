"""
CLI for data ingestion.

Usage:
    python -m modules.data_engineering.ingest.cli jsut
    python -m modules.data_engineering.ingest.cli jsut --force
    python -m modules.data_engineering.ingest.cli jvs
"""

import argparse
import sys


def main(
    dataset: str = None,
    version: str = None,
    force: bool = False,
) -> int:
    """
    Main entry point for ingest CLI.

    Can be called programmatically or via command line.
    """
    # If called without args, parse from command line
    if dataset is None:
        args = parse_args()
        dataset = args.dataset
        version = args.version
        force = args.force

    if dataset == "jsut":
        return ingest_jsut_cli(force=force)
    elif dataset == "jvs":
        return ingest_jvs_cli(force=force)
    elif dataset == "common_voice":
        return ingest_cv_cli(version=version, force=force)
    else:
        print(f"Unknown dataset: {dataset}")
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data ingestion CLI",
        prog="python -m modules.data_engineering.ingest.cli",
    )
    parser.add_argument(
        "dataset",
        choices=["jsut", "jvs", "common_voice"],
        help="Dataset to ingest",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Dataset version (for datasets with multiple versions)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-ingest even if already complete",
    )
    return parser.parse_args()


def ingest_jsut_cli(force: bool = False) -> int:
    """Run JSUT ingest pipeline."""
    try:
        from .jsut import ingest_jsut
        result = ingest_jsut(force=force)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JSUT ingest: {e}")
        raise


def ingest_jvs_cli(force: bool = False) -> int:
    """Run JVS ingest pipeline."""
    try:
        from .jvs import ingest_jvs
        result = ingest_jvs(force=force)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JVS ingest: {e}")
        raise


def ingest_cv_cli(version: str = None, force: bool = False) -> int:
    """Run Common Voice ingest pipeline."""
    print("Common Voice ingestion not yet implemented")
    print("TODO: Implement modules.data_engineering.ingest.common_voice")
    return 1


if __name__ == "__main__":
    sys.exit(main())
