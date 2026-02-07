"""
CLI for gold layer operations.

Usage:
    python -m modules.data_engineering.gold.cli jsut
    python -m modules.data_engineering.gold.cli jsut --val-pct 0.1
    python -m modules.data_engineering.gold.cli jsut --snapshot-id my-snapshot
"""

import argparse
import sys


def main(dataset: str = None, version: str = None, **kwargs) -> int:
    """
    Main entry point for gold CLI.

    Can be called programmatically or via command line.

    Args:
        dataset: Dataset name (jsut, jvs, common_voice)
        version: Alias for snapshot_id (for CLI compatibility)
        **kwargs: Additional arguments passed to build functions
            - snapshot_id: Snapshot ID (auto-generated if not provided)
            - min_duration: Minimum duration filter
            - max_duration: Maximum duration filter
            - val_pct: Validation set fraction
            - test_pct: Test set fraction
            - seed: Random seed for splits
            - export_jsonl: Whether to export JSONL manifest
            - write_delta: Whether to write Delta table
            - manifest_out: Override manifest output path
    """
    # If called without args, parse from command line
    if dataset is None:
        args = parse_args()
        dataset = args.dataset
        kwargs = {
            "snapshot_id": args.snapshot_id,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "val_pct": args.val_pct,
            "test_pct": args.test_pct,
            "seed": args.seed,
            "export_jsonl": not args.no_jsonl,
            "write_delta": not args.no_write_delta,
            "manifest_out": args.manifest_out,
        }
    else:
        # Called programmatically - map version to snapshot_id if provided
        if version is not None and "snapshot_id" not in kwargs:
            kwargs["snapshot_id"] = version

    if dataset == "jsut":
        return gold_jsut_cli(**kwargs)
    elif dataset == "jvs":
        return gold_jvs_cli(**kwargs)
    elif dataset == "multi":
        return gold_multi_cli(**kwargs)
    elif dataset == "common_voice":
        return gold_cv_cli(**kwargs)
    else:
        print(f"Unknown dataset: {dataset}")
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gold layer CLI",
        prog="python -m modules.data_engineering.gold.cli",
    )
    parser.add_argument(
        "dataset",
        choices=["jsut", "jvs", "multi", "common_voice"],
        help="Dataset to process to gold",
    )
    parser.add_argument(
        "--snapshot-id",
        help="Snapshot ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Maximum duration in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.10,
        help="Validation set fraction (default: 0.10)",
    )
    parser.add_argument(
        "--test-pct",
        type=float,
        default=0.0,
        help="Test set fraction (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip JSONL manifest export",
    )
    parser.add_argument(
        "--no-write-delta",
        action="store_true",
        help="Skip writing Delta table (manifest only)",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="Override manifest output path",
    )
    return parser.parse_args()


def gold_jsut_cli(**kwargs) -> int:
    """Run JSUT gold pipeline."""
    try:
        from .jsut import build_gold_jsut
        result = build_gold_jsut(**kwargs)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JSUT gold: {e}")
        raise


def gold_jvs_cli(**kwargs) -> int:
    """Run JVS gold pipeline."""
    try:
        from .jvs import build_gold_jvs
        result = build_gold_jvs(**kwargs)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JVS gold: {e}")
        raise


def gold_multi_cli(**kwargs) -> int:
    """Run multi-speaker combined gold pipeline."""
    try:
        from .multi import build_gold_multi
        result = build_gold_multi(**kwargs)
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during multi gold: {e}")
        raise


def gold_cv_cli(**kwargs) -> int:
    """Run Common Voice gold pipeline."""
    print("Common Voice gold not yet implemented")
    print("TODO: Implement modules.data_engineering.gold.common_voice")
    return 1


if __name__ == "__main__":
    sys.exit(main())
