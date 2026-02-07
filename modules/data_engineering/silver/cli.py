"""
CLI for silver layer operations.

Usage:
    python -m modules.data_engineering.silver.cli jsut
    python -m modules.data_engineering.silver.cli jsut --force
    python -m modules.data_engineering.silver.cli jvs --val-pct 0.15 --seed 123
"""

import argparse
import sys

from .common import DEFAULT_SEED, DEFAULT_TEST_PCT, DEFAULT_VAL_PCT


def main(
    dataset: str = None,
    force: bool = False,
    val_pct: float = None,
    test_pct: float = None,
    seed: int = None,
    phonemize: bool = False,
) -> int:
    """
    Main entry point for silver CLI.

    Can be called programmatically or via command line.
    """
    # If called without args, parse from command line
    if dataset is None:
        args = parse_args()
        dataset = args.dataset
        force = args.force
        val_pct = args.val_pct
        test_pct = args.test_pct
        seed = args.seed
        phonemize = args.phonemize

    # Use defaults if not specified
    if val_pct is None:
        val_pct = DEFAULT_VAL_PCT
    if test_pct is None:
        test_pct = DEFAULT_TEST_PCT
    if seed is None:
        seed = DEFAULT_SEED

    if dataset == "jsut":
        return silver_jsut_cli(force=force, val_pct=val_pct, test_pct=test_pct, seed=seed, phonemize=phonemize)
    elif dataset == "jvs":
        return silver_jvs_cli(force=force, val_pct=val_pct, test_pct=test_pct, seed=seed)
    elif dataset == "common_voice":
        return silver_cv_cli(force=force)
    else:
        print(f"Unknown dataset: {dataset}")
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Silver layer CLI",
        prog="python -m modules.data_engineering.silver.cli",
    )
    parser.add_argument(
        "dataset",
        choices=["jsut", "jvs", "common_voice"],
        help="Dataset to process to silver",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild even if table exists",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=DEFAULT_VAL_PCT,
        help=f"Validation set fraction (default: {DEFAULT_VAL_PCT})",
    )
    parser.add_argument(
        "--test-pct",
        type=float,
        default=DEFAULT_TEST_PCT,
        help=f"Test set fraction (default: {DEFAULT_TEST_PCT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for split assignment (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--phonemize",
        action="store_true",
        help="Generate phonemes with pyopenjtalk (JSUT only)",
    )
    return parser.parse_args()


def silver_jsut_cli(
    force: bool = False,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
    phonemize: bool = False,
) -> int:
    """Run JSUT silver pipeline."""
    try:
        from .jsut import build_silver_jsut
        result = build_silver_jsut(
            force=force,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            phonemize=phonemize,
        )
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JSUT silver: {e}")
        raise


def silver_jvs_cli(
    force: bool = False,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
) -> int:
    """Run JVS silver pipeline."""
    try:
        from .jvs import build_silver_jvs
        result = build_silver_jvs(
            force=force,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
        )
        return 0 if result["status"] == "success" else 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during JVS silver: {e}")
        raise


def silver_cv_cli(force: bool = False) -> int:
    """Run Common Voice silver pipeline."""
    print("Common Voice silver not yet implemented")
    print("TODO: Implement modules.data_engineering.silver.common_voice")
    return 1


if __name__ == "__main__":
    sys.exit(main())
