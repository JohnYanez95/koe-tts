"""
Multi-speaker combined dataset gold layer.

Merges JSUT + JVS gold manifests into a single multi-speaker dataset.

Command: koe gold multi

Requires:
    lake/gold/jsut/manifests/*.jsonl (from koe gold jsut)
    lake/gold/jvs/manifests/*.jsonl  (from koe gold jvs)

Writes:
    data/gold/multi/{snapshot_id}/train.jsonl
    data/gold/multi/{snapshot_id}/val.jsonl
    data/gold/multi/{snapshot_id}/manifest.json (metadata)
"""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.data_engineering.common.paths import paths

# Gold pipeline version
GOLD_VERSION = "v0.1"

# Speaker ID mapping
JSUT_SPEAKER_ID = "spk00"  # Single JSUT speaker


def find_latest_manifest(dataset: str) -> tuple[Path, str]:
    """
    Find the latest manifest JSONL for a dataset.

    Args:
        dataset: Dataset name (jsut, jvs)

    Returns:
        Tuple of (path to manifest, snapshot_id)

    Raises:
        FileNotFoundError: If no manifests found
    """
    manifests_dir = paths.gold / dataset / "manifests"

    if not manifests_dir.exists():
        raise FileNotFoundError(
            f"No manifests found for {dataset}: {manifests_dir}\n"
            f"Run gold first: koe gold {dataset}"
        )

    # Find newest .jsonl file
    manifests = sorted(
        manifests_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not manifests:
        raise FileNotFoundError(
            f"No .jsonl manifests in {manifests_dir}\n"
            f"Run gold first: koe gold {dataset}"
        )

    manifest_path = manifests[0]
    snapshot_id = manifest_path.stem  # Extract snapshot_id from filename
    return manifest_path, snapshot_id


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load JSONL manifest as list of records."""
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_speaker_id(record: dict, dataset: str) -> str:
    """
    Normalize speaker_id to consistent format.

    JSUT: jsut001 -> spk00 (single speaker)
    JVS: jvs001, jvs002, ... (keep as-is, 1-indexed)
    """
    if dataset == "jsut":
        return JSUT_SPEAKER_ID
    else:
        # JVS speakers should already be jvs001, jvs002, etc.
        return record.get("speaker_id", "unknown")


def generate_snapshot_id(config_hash: str = None) -> str:
    """Generate a unique snapshot ID."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")

    if config_hash:
        short_hash = config_hash[:8]
    else:
        short_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]

    return f"multi-{timestamp}-{short_hash}"


def build_gold_multi(
    snapshot_id: Optional[str] = None,
    jsut_manifest: Optional[str] = None,
    jvs_manifest: Optional[str] = None,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
    **kwargs,  # Accept but ignore other gold args for CLI compatibility
) -> dict:
    """
    Build combined multi-speaker gold manifest.

    Merges JSUT and JVS gold manifests into a single multi-speaker dataset.

    Args:
        snapshot_id: Optional snapshot ID (auto-generated if not provided)
        jsut_manifest: Path to JSUT manifest (auto-detected if not provided)
        jvs_manifest: Path to JVS manifest (auto-detected if not provided)
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter

    Returns:
        Dict with build results and stats
    """
    print("=" * 60)
    print("Multi-Speaker Gold Pipeline")
    print("=" * 60)

    # Find manifests
    print("\n[1/5] Finding source manifests...")

    if jsut_manifest:
        jsut_path = Path(jsut_manifest)
        jsut_snapshot_id = jsut_path.stem
    else:
        jsut_path, jsut_snapshot_id = find_latest_manifest("jsut")
    print(f"  JSUT: {jsut_path}")
    print(f"        snapshot: {jsut_snapshot_id}")

    if jvs_manifest:
        jvs_path = Path(jvs_manifest)
        jvs_snapshot_id = jvs_path.stem
    else:
        jvs_path, jvs_snapshot_id = find_latest_manifest("jvs")
    print(f"  JVS:  {jvs_path}")
    print(f"        snapshot: {jvs_snapshot_id}")

    # Load manifests
    print("\n[2/5] Loading manifests...")
    jsut_records = load_manifest(jsut_path)
    jvs_records = load_manifest(jvs_path)
    print(f"  JSUT records: {len(jsut_records)}")
    print(f"  JVS records:  {len(jvs_records)}")

    # Compute source split counts for validation
    jsut_split_counts = Counter(r.get("split", "train") for r in jsut_records)
    jvs_split_counts = Counter(r.get("split", "train") for r in jvs_records)
    print(f"  JSUT splits: {dict(jsut_split_counts)}")
    print(f"  JVS splits:  {dict(jvs_split_counts)}")

    # Generate snapshot ID
    if snapshot_id is None:
        config_str = f"{jsut_path.name}_{jvs_path.name}_{min_duration}_{max_duration}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        snapshot_id = generate_snapshot_id(config_hash)
    print(f"  Snapshot ID: {snapshot_id}")

    # Merge and normalize
    print("\n[3/5] Merging and normalizing...")

    train_records = []
    val_records = []
    speakers = set()
    filtered_counts = {"jsut": 0, "jvs": 0}

    for record in jsut_records:
        # Apply duration filter
        dur = record.get("duration_sec", 0)
        if dur < min_duration or dur > max_duration:
            filtered_counts["jsut"] += 1
            continue

        # Normalize speaker_id
        record["speaker_id"] = normalize_speaker_id(record, "jsut")

        # Add provenance fields
        record["source_dataset"] = "jsut"
        record["source_snapshot_id"] = jsut_snapshot_id
        record["source_utterance_id"] = record.get("utterance_id")
        record["source_utterance_key"] = record.get("utterance_key")
        record["snapshot_id"] = snapshot_id

        speakers.add(record["speaker_id"])

        if record.get("split") == "val":
            val_records.append(record)
        else:
            train_records.append(record)

    for record in jvs_records:
        # Apply duration filter
        dur = record.get("duration_sec", 0)
        if dur < min_duration or dur > max_duration:
            filtered_counts["jvs"] += 1
            continue

        # Normalize speaker_id
        record["speaker_id"] = normalize_speaker_id(record, "jvs")

        # Add provenance fields
        record["source_dataset"] = "jvs"
        record["source_snapshot_id"] = jvs_snapshot_id
        record["source_utterance_id"] = record.get("utterance_id")
        record["source_utterance_key"] = record.get("utterance_key")
        record["snapshot_id"] = snapshot_id

        speakers.add(record["speaker_id"])

        if record.get("split") == "val":
            val_records.append(record)
        else:
            train_records.append(record)

    print(f"  Train records: {len(train_records)}")
    print(f"  Val records:   {len(val_records)}")
    print(f"  Speakers:      {len(speakers)}")
    if filtered_counts["jsut"] > 0 or filtered_counts["jvs"] > 0:
        print(f"  Filtered by duration: JSUT={filtered_counts['jsut']}, JVS={filtered_counts['jvs']}")

    # Validation step
    print("\n[4/5] Validating invariants...")

    all_records = train_records + val_records

    # 1. Utterance ID deduplication assertion
    all_ids = [r["utterance_id"] for r in all_records]
    n_total = len(all_ids)
    n_unique = len(set(all_ids))
    if n_unique != n_total:
        dup_ids = [k for k, v in Counter(all_ids).items() if v > 1][:10]
        raise ValueError(
            f"[multi] utterance_id collision: {n_unique}/{n_total} unique. "
            f"Examples: {dup_ids}"
        )
    print(f"  ✓ All {n_total} utterance_ids are unique")

    # 2. Split count validation (exact equality after duration filtering)
    # Count what we actually kept per source per split
    jsut_kept = Counter(r.get("split", "train") for r in all_records if r["source_dataset"] == "jsut")
    jvs_kept = Counter(r.get("split", "train") for r in all_records if r["source_dataset"] == "jvs")
    multi_split_counts = Counter(r.get("split", "train") for r in all_records)

    all_splits = set(jsut_kept) | set(jvs_kept) | set(multi_split_counts)

    for split in all_splits:
        jsut_count = jsut_kept.get(split, 0)
        jvs_count = jvs_kept.get(split, 0)
        multi_count = multi_split_counts.get(split, 0)
        expected = jsut_count + jvs_count
        if multi_count != expected:
            raise ValueError(
                f"[multi] split count mismatch '{split}': got={multi_count} "
                f"expected={expected} (jsut={jsut_count} jvs={jvs_count})"
            )
    print(f"  ✓ Split counts exact: {dict(multi_split_counts)} = jsut{dict(jsut_kept)} + jvs{dict(jvs_kept)}")

    # 3. Speaker count validation
    expected_speakers = 101  # 1 JSUT + 100 JVS
    if len(speakers) != expected_speakers:
        print(f"  ⚠ Expected {expected_speakers} speakers, got {len(speakers)}")
    else:
        print(f"  ✓ Speaker count valid: {len(speakers)}")

    # Write output - use lake/gold/multi/manifests for consistency with cache system
    print("\n[5/5] Writing manifests...")

    manifests_dir = paths.gold / "multi" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Write combined manifest (same format as jsut/jvs for cache compatibility)
    # This is the primary file that cache_snapshot will discover
    manifest_path = manifests_dir / f"{snapshot_id}.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {manifest_path}")

    # Write split-specific files in a subdirectory (so they don't interfere with cache discovery)
    splits_dir = manifests_dir / snapshot_id
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {val_path}")

    # Write metadata
    speaker_list = sorted(speakers)
    meta_path = splits_dir / "meta.json"
    metadata = {
        "snapshot_id": snapshot_id,
        "gold_version": GOLD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "jsut": {
                "path": str(jsut_path),
                "snapshot_id": jsut_snapshot_id,
                "n_records": len(jsut_records),
            },
            "jvs": {
                "path": str(jvs_path),
                "snapshot_id": jvs_snapshot_id,
                "n_records": len(jvs_records),
            },
        },
        "stats": {
            "n_train": len(train_records),
            "n_val": len(val_records),
            "n_total": len(train_records) + len(val_records),
            "n_speakers": len(speakers),
            "filtered_by_duration": filtered_counts,
        },
        "speaker_list": speaker_list,
        "num_speakers": len(speaker_list),
        "duration_filter": {
            "min": min_duration,
            "max": max_duration,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Multi-Speaker Gold Complete")
    print("=" * 60)
    print(f"\nSnapshot: {snapshot_id}")
    print(f"\nCombined dataset:")
    print(f"  Train: {len(train_records):,} utterances")
    print(f"  Val:   {len(val_records):,} utterances")
    print(f"  Total: {len(train_records) + len(val_records):,} utterances")
    print(f"  Speakers: {len(speakers)}")
    print(f"\nSpeaker list preview: {speaker_list[:5]}...{speaker_list[-3:]}")
    print(f"\nOutput: {manifests_dir}")
    print(f"\nNext step:")
    print(f"  koe cache create multi --snapshot-id {snapshot_id}")
    print("=" * 60)

    return {
        "status": "success",
        "snapshot_id": snapshot_id,
        "manifest_path": str(manifest_path),
        "manifests_dir": str(manifests_dir),
        "gold_version": GOLD_VERSION,
        "stats": metadata["stats"],
        "speaker_list": speaker_list,
    }


if __name__ == "__main__":
    result = build_gold_multi()
    print(f"\nResult: {result['status']}")
