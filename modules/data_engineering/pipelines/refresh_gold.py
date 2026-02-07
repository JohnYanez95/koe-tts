"""
Refresh gold layer with new rules.

Re-runs only the gold step (splits, sampling) without re-processing bronze/silver.

Usage:
    python -m modules.data_engineering.pipelines.refresh_gold --gold-version v2
"""

import argparse


def refresh_gold(
    gold_version: str,
    silver_version: int | None = None,
) -> dict:
    """
    Refresh gold layer from existing silver data.

    Args:
        gold_version: New gold version tag
        silver_version: Silver table version to use (latest if None)

    Returns:
        Dict with results
    """
    print(f"Refreshing gold layer: version={gold_version}")

    if silver_version:
        print(f"  Using silver version: {silver_version}")
    else:
        print("  Using latest silver version")

    # TODO: Implement gold refresh
    # 1. Read silver.utterances_clean at version
    # 2. Apply split rules from configs/lakehouse/gold.yaml
    # 3. Compute sampling weights
    # 4. Write gold.train_manifest with gold_version tag
    # 5. Generate eval sets

    print("  Gold refresh not yet implemented")

    return {
        "gold_version": gold_version,
        "silver_version": silver_version or "latest",
        "status": "not_implemented",
    }


def main():
    parser = argparse.ArgumentParser(description="Refresh gold layer")
    parser.add_argument("--gold-version", required=True, help="Gold version tag")
    parser.add_argument("--silver-version", type=int, help="Silver version to use")

    args = parser.parse_args()

    results = refresh_gold(
        gold_version=args.gold_version,
        silver_version=args.silver_version,
    )

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
