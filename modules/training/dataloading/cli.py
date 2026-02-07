"""
CLI for training data loading.

Usage:
    koe cache create jsut                    # Create cache from latest gold manifest
    koe cache create jsut -s snapshot_id     # Create cache from specific snapshot
    koe cache list                           # List all caches
    koe cache list jsut                      # List caches for a dataset
"""

from pathlib import Path
from typing import Optional

from .cache_snapshot import cache_snapshot, list_caches


def cache_create(
    dataset: str,
    snapshot_id: Optional[str] = None,
    gold_version: Optional[str] = None,  # Alias for snapshot_id
    mode: str = "symlink",
    cache_root: Optional[Path] = None,
) -> dict:
    """
    Create a local cache snapshot from gold manifest.

    Args:
        dataset: Dataset name (jsut, jvs, etc.)
        snapshot_id: Specific snapshot to cache (None = latest)
        gold_version: Alias for snapshot_id (for backwards compat)
        mode: "symlink" (fast) or "copy" (portable)
        cache_root: Override cache directory

    Returns:
        Dict with cache info
    """
    # Handle alias
    if gold_version and not snapshot_id:
        snapshot_id = gold_version

    return cache_snapshot(
        dataset=dataset,
        snapshot_id=snapshot_id,
        mode=mode,
        cache_root=cache_root,
    )


def cache_list(dataset: Optional[str] = None) -> None:
    """
    List available cache snapshots.

    Args:
        dataset: Filter by dataset (None = all)
    """
    caches = list_caches(dataset=dataset)

    if not caches:
        print("No caches found")
        if dataset:
            print(f"  Run: koe cache create {dataset}")
        return

    print(f"Found {len(caches)} cache(s):\n")
    for cache in caches:
        print(f"  {cache['dataset']}/{cache['snapshot_id']}")
        print(f"    Created: {cache.get('created_at', 'unknown')}")
        print(f"    Utterances: {cache.get('num_utterances', 'unknown')}")
        print(f"    Mode: {cache.get('mode', 'unknown')}")
        splits = cache.get("splits", [])
        if splits:
            print(f"    Splits: {', '.join(splits)}")
        print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Training data loading CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Cache create command
    create_parser = subparsers.add_parser("create", help="Create cache snapshot")
    create_parser.add_argument("dataset", help="Dataset name")
    create_parser.add_argument("--snapshot-id", "-s", help="Specific snapshot ID")
    create_parser.add_argument("--gold-version", help="Alias for --snapshot-id")
    create_parser.add_argument(
        "--mode", choices=["symlink", "copy"], default="symlink",
        help="Cache mode (default: symlink)"
    )
    create_parser.add_argument("--cache-root", type=Path, help="Override cache directory")

    # Cache list command
    list_parser = subparsers.add_parser("list", help="List cache snapshots")
    list_parser.add_argument("dataset", nargs="?", help="Filter by dataset")

    args = parser.parse_args()

    if args.command == "create":
        result = cache_create(
            dataset=args.dataset,
            snapshot_id=args.snapshot_id,
            gold_version=args.gold_version,
            mode=args.mode,
            cache_root=args.cache_root,
        )
        print(f"\nResult: {result['status']}")

    elif args.command == "list":
        cache_list(dataset=args.dataset)


if __name__ == "__main__":
    main()
