"""
CLI for bronze layer operations.

Usage:
    python -m modules.data_engineering.bronze.cli jsut
    python -m modules.data_engineering.bronze.cli jsut --force
"""

import argparse
import sys


def main(dataset: str = None, force: bool = False) -> int:
    """
    Main entry point for bronze CLI.

    Can be called programmatically or via command line.
    """
    # If called without args, parse from command line
    if dataset is None:
        args = parse_args()
        dataset = args.dataset
        force = args.force

    if dataset == "jsut":
        return bronze_jsut_cli(force=force)
    elif dataset == "jvs":
        return bronze_jvs_cli(force=force)
    elif dataset == "common_voice":
        return bronze_cv_cli(force=force)
    else:
        print(f"Unknown dataset: {dataset}")
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bronze layer CLI",
        prog="python -m modules.data_engineering.bronze.cli",
    )
    parser.add_argument(
        "dataset",
        choices=["jsut", "jvs", "common_voice"],
        help="Dataset to process to bronze",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild even if table exists",
    )
    return parser.parse_args()


def bronze_jsut_cli(force: bool = False) -> int:
    """Run JSUT bronze pipeline."""
    try:
        from .jsut import build_bronze_jsut
        result = build_bronze_jsut(force=force)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JSUT bronze: {e}")
        raise


def bronze_jvs_cli(force: bool = False) -> int:
    """Run JVS bronze pipeline."""
    try:
        from .jvs import build_bronze_jvs
        result = build_bronze_jvs(force=force)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JVS bronze: {e}")
        raise


def bronze_cv_cli(force: bool = False) -> int:
    """Run Common Voice bronze pipeline."""
    print("Common Voice bronze not yet implemented")
    print("TODO: Implement modules.data_engineering.bronze.common_voice")
    return 1


if __name__ == "__main__":
    sys.exit(main())
