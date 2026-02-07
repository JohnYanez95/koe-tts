"""
Materialize local cache from gold manifest.

Creates a local snapshot of training data for fast I/O during training.
The cache contains symlinks (or copies) of audio files plus a manifest.

Usage:
    koe cache create jsut --snapshot-id jsut-20260125-094843-fd76d53a
    koe cache create jsut  # Uses latest manifest

Cache structure:
    data/cache/{dataset}/{snapshot_id}/
    ├── manifest.jsonl           # Training manifest (audio paths rewritten)
    ├── audio/                   # Audio files (symlinks or copies)
    │   └── {utterance_id}.wav
    └── metadata.json            # Cache metadata

The manifest has audio_path rewritten to point to local cache files,
enabling the training loop to load directly without path resolution.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from modules.data_engineering.common.paths import paths


def find_latest_manifest(dataset: str) -> Optional[Path]:
    """
    Find the most recent gold manifest for a dataset.

    Returns:
        Path to latest manifest, or None if none found
    """
    manifest_dir = paths.gold / dataset / "manifests"
    if not manifest_dir.exists():
        return None

    manifests = sorted(manifest_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        return None

    return manifests[-1]


def find_manifest_by_snapshot(dataset: str, snapshot_id: str) -> Optional[Path]:
    """
    Find a specific gold manifest by snapshot ID.

    Returns:
        Path to manifest, or None if not found
    """
    manifest_path = paths.gold / dataset / "manifests" / f"{snapshot_id}.jsonl"
    if manifest_path.exists():
        return manifest_path
    return None


def cache_snapshot(
    dataset: str,
    snapshot_id: Optional[str] = None,
    mode: Literal["symlink", "copy"] = "symlink",
    cache_root: Optional[Path] = None,
    splits: Optional[list[str]] = None,
) -> dict:
    """
    Materialize a local cache snapshot from gold manifest.

    Args:
        dataset: Dataset name (jsut, jvs, etc.)
        snapshot_id: Specific snapshot to cache (None = latest)
        mode: "symlink" (fast, same filesystem) or "copy" (portable)
        cache_root: Override cache directory (default: paths.cache)
        splits: Which splits to cache (default: ["train", "val"])

    Returns:
        Dict with cache info and statistics
    """
    if splits is None:
        splits = ["train", "val"]

    print(f"Creating cache snapshot for {dataset}")
    print(f"  Mode: {mode}")
    print(f"  Splits: {', '.join(splits)}")

    # Find the manifest
    if snapshot_id:
        manifest_path = find_manifest_by_snapshot(dataset, snapshot_id)
        if not manifest_path:
            raise FileNotFoundError(
                f"Manifest not found: {snapshot_id}\n"
                f"Check: {paths.gold / dataset / 'manifests'}"
            )
    else:
        manifest_path = find_latest_manifest(dataset)
        if not manifest_path:
            raise FileNotFoundError(
                f"No manifests found for {dataset}\n"
                f"Run gold first: koe gold {dataset}"
            )
        snapshot_id = manifest_path.stem  # Extract from filename

    print(f"  Snapshot: {snapshot_id}")
    print(f"  Manifest: {manifest_path}")

    # Set up cache directory
    if cache_root is None:
        cache_root = paths.cache

    cache_dir = cache_root / dataset / snapshot_id
    audio_dir = cache_dir / "audio"
    cache_manifest_path = cache_dir / "manifest.jsonl"
    metadata_path = cache_dir / "metadata.json"

    # Check if cache already exists
    if cache_dir.exists():
        print(f"  Cache already exists: {cache_dir}")
        if metadata_path.exists():
            with open(metadata_path) as f:
                existing = json.load(f)
            print(f"    Created: {existing.get('created_at', 'unknown')}")
            print(f"    Utterances: {existing.get('num_utterances', 'unknown')}")
            return {
                "status": "exists",
                "cache_dir": str(cache_dir),
                "metadata": existing,
            }

    # Create cache directory structure
    print(f"  Creating cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    # Read manifest and process utterances
    print("  Processing utterances...")
    cached_rows = []
    speaker_ids_seen = set()
    stats = {
        "total": 0,
        "cached": 0,
        "skipped": 0,
        "missing_audio": 0,
        "symlink_errors": 0,
        "by_split": {},
    }

    with open(manifest_path) as f:
        for line in f:
            row = json.loads(line)
            stats["total"] += 1

            # Filter by split
            split = row.get("split", "train")
            if split not in splits:
                stats["skipped"] += 1
                continue

            # Track split counts
            stats["by_split"][split] = stats["by_split"].get(split, 0) + 1

            # Resolve source audio path
            audio_relpath = row.get("audio_relpath")
            audio_abspath = row.get("audio_abspath")

            if audio_abspath and Path(audio_abspath).exists():
                source_path = Path(audio_abspath)
            elif audio_relpath:
                source_path = paths.data_root / "data" / audio_relpath
                if not source_path.exists():
                    stats["missing_audio"] += 1
                    continue
            else:
                stats["missing_audio"] += 1
                continue

            # Determine cache filename (use utterance_id for uniqueness)
            utterance_id = row["utterance_id"]
            ext = source_path.suffix or ".wav"
            cache_filename = f"{utterance_id}{ext}"
            cache_audio_path = audio_dir / cache_filename

            # Create symlink or copy
            try:
                if mode == "symlink":
                    if cache_audio_path.exists():
                        cache_audio_path.unlink()
                    # Use relative symlink for portability (NAS, mount changes)
                    try:
                        rel_target = os.path.relpath(source_path, start=cache_audio_path.parent)
                        cache_audio_path.symlink_to(rel_target)
                    except ValueError:
                        # Fallback to absolute if on different drives (Windows)
                        cache_audio_path.symlink_to(source_path)
                else:  # copy
                    if not cache_audio_path.exists():
                        shutil.copy2(source_path, cache_audio_path)

                stats["cached"] += 1

                # Update row with cache-local path
                cached_row = row.copy()
                cached_row["audio_path"] = str(cache_audio_path)
                cached_row["audio_relpath_original"] = audio_relpath
                cached_rows.append(cached_row)

                # Track speaker for multi-speaker metadata
                if "speaker_id" in row:
                    speaker_ids_seen.add(row["speaker_id"])

            except OSError as e:
                stats["symlink_errors"] += 1
                print(f"    Warning: Failed to {mode} {source_path}: {e}")

    # Write cache manifest
    print(f"  Writing manifest: {cache_manifest_path}")
    with open(cache_manifest_path, "w") as f:
        for row in cached_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write metadata
    speaker_list = sorted(speaker_ids_seen) if speaker_ids_seen else []
    metadata = {
        "dataset": dataset,
        "snapshot_id": snapshot_id,
        "source_manifest": str(manifest_path),
        "mode": mode,
        "splits": splits,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_utterances": stats["cached"],
        "num_speakers": len(speaker_list),
        "speaker_list": speaker_list,
        "stats": stats,
    }

    print(f"  Writing metadata: {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Update latest symlink
    latest_link = cache_root / dataset / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(cache_dir.name)

    # Print summary
    print("\n" + "=" * 50)
    print("Cache Snapshot Complete")
    print("=" * 50)
    print(f"  Dataset: {dataset}")
    print(f"  Snapshot: {snapshot_id}")
    print(f"  Mode: {mode}")
    print(f"  Utterances: {stats['cached']}")
    for split, count in sorted(stats["by_split"].items()):
        print(f"    {split}: {count}")
    if len(speaker_list) > 0:
        print(f"  Speakers: {len(speaker_list)}")
        if len(speaker_list) <= 5:
            print(f"    {speaker_list}")
        else:
            print(f"    {speaker_list[:3]}...{speaker_list[-2:]}")
    if stats["missing_audio"] > 0:
        print(f"  Missing audio: {stats['missing_audio']}")
    if stats["symlink_errors"] > 0:
        print(f"  Symlink errors: {stats['symlink_errors']}")
    print(f"  Cache: {cache_dir}")
    print("=" * 50)

    return {
        "status": "created",
        "cache_dir": str(cache_dir),
        "manifest_path": str(cache_manifest_path),
        "metadata": metadata,
    }


def list_caches(dataset: Optional[str] = None) -> list[dict]:
    """
    List available cache snapshots.

    Args:
        dataset: Filter by dataset (None = all datasets)

    Returns:
        List of cache metadata dicts
    """
    cache_root = paths.cache
    caches = []

    if not cache_root.exists():
        return caches

    # Determine which datasets to scan
    if dataset:
        datasets = [dataset] if (cache_root / dataset).exists() else []
    else:
        datasets = [d.name for d in cache_root.iterdir() if d.is_dir()]

    for ds in datasets:
        ds_cache = cache_root / ds
        for snapshot_dir in ds_cache.iterdir():
            if snapshot_dir.name == "latest":
                continue
            if not snapshot_dir.is_dir():
                continue

            metadata_path = snapshot_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                caches.append(metadata)

    # Sort by creation time (newest first)
    caches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return caches


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cache gold snapshot locally")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create cache snapshot")
    create_parser.add_argument("dataset", help="Dataset name")
    create_parser.add_argument("--snapshot-id", "-s", help="Specific snapshot ID")
    create_parser.add_argument(
        "--mode", choices=["symlink", "copy"], default="symlink",
        help="Cache mode (default: symlink)"
    )
    create_parser.add_argument("--cache-root", type=Path, help="Override cache directory")

    # List command
    list_parser = subparsers.add_parser("list", help="List cache snapshots")
    list_parser.add_argument("dataset", nargs="?", help="Filter by dataset")

    args = parser.parse_args()

    if args.command == "create":
        result = cache_snapshot(
            dataset=args.dataset,
            snapshot_id=args.snapshot_id,
            mode=args.mode,
            cache_root=args.cache_root,
        )
        print(f"\nResult: {result['status']}")

    elif args.command == "list":
        caches = list_caches(dataset=args.dataset)
        if not caches:
            print("No caches found")
        else:
            print(f"Found {len(caches)} cache(s):")
            for cache in caches:
                print(f"\n  {cache['dataset']}/{cache['snapshot_id']}")
                print(f"    Created: {cache.get('created_at', 'unknown')}")
                print(f"    Utterances: {cache.get('num_utterances', 'unknown')}")
                print(f"    Mode: {cache.get('mode', 'unknown')}")


if __name__ == "__main__":
    main()
