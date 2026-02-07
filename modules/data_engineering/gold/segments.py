"""
Gold segments manifest generation.

Builds segment manifests from silver breakpoints + gold parent utterances.
Segments reference parent audio via start_ms/end_ms (no audio duplication).

Tier 1 scope: All segments are unlabeled (segment_label_status="unlabeled",
tokens=null). Not usable for VITS core/duration/baseline until Tier 2 labeling.

Usage:
    from modules.data_engineering.gold.segments import build_gold_segments

    result = build_gold_segments(
        dataset="jsut",
        config=SegmentConfig(),
    )

Command: koe segment build jsut
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.spark import get_spark
from modules.data_engineering.silver.segments import read_segment_breaks

# Pipeline version for tracking
GOLD_SEGMENTS_VERSION = "v1.0"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SegmentConfig:
    """Configuration for segment generation."""
    min_segment_ms: int = 800
    max_segment_ms: int = 6000
    target_segment_ms: int = 3000
    min_lead_ms: int = 250  # no breakpoint before this
    allow_over_max_tail_ms: int = 400  # last segment can exceed max by this


@dataclass
class SegmentInfo:
    """Information about a generated segment."""
    start_ms: int
    end_ms: int
    cut_reason: str  # "breakpoint" or "hard_max"
    cut_breakpoint_ms: Optional[int]  # the chosen breakpoint (None if hard_max)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


# =============================================================================
# Segmentation Algorithm
# =============================================================================

def compute_segments_from_breakpoints(
    breakpoints: list[int],
    duration_ms: int,
    config: SegmentConfig,
) -> list[SegmentInfo]:
    """
    Compute segments from breakpoints using greedy accumulation.

    Algorithm:
    1. Filter breakpoints: remove any < min_lead_ms from start
    2. Greedy accumulation:
       - Start at 0
       - Find best breakpoint in [min_segment_ms, max_segment_ms] range
       - Prefer breakpoint closest to target_segment_ms
       - If breakpoint found: cut_reason="breakpoint", cut_breakpoint_ms=bp
       - If no breakpoint: cut_reason="hard_max", cut at max_segment_ms
    3. Handle tail:
       - If remainder < min_segment_ms: merge into previous
       - Allow previous to exceed max by up to allow_over_max_tail_ms
       - If would exceed too much: redistribute by shifting last cut earlier

    Args:
        breakpoints: Sorted list of breakpoint positions in ms
        duration_ms: Total audio duration in ms
        config: Segment configuration

    Returns:
        List of SegmentInfo objects
    """
    if duration_ms <= 0:
        return []

    # If duration is less than min, return single segment
    if duration_ms < config.min_segment_ms:
        return []  # Too short, will be filtered out

    # If duration is less than max, return single segment
    if duration_ms <= config.max_segment_ms:
        return [SegmentInfo(
            start_ms=0,
            end_ms=duration_ms,
            cut_reason="natural_end",
            cut_breakpoint_ms=None,
        )]

    # Filter breakpoints by min_lead
    valid_breakpoints = [
        bp for bp in breakpoints
        if bp >= config.min_lead_ms and bp <= duration_ms - config.min_lead_ms
    ]

    segments = []
    current_start = 0

    while current_start < duration_ms:
        remaining = duration_ms - current_start

        # If remaining fits in one segment, finish
        if remaining <= config.max_segment_ms:
            segments.append(SegmentInfo(
                start_ms=current_start,
                end_ms=duration_ms,
                cut_reason="natural_end",
                cut_breakpoint_ms=None,
            ))
            break

        # Find breakpoints in valid range for this segment
        min_end = current_start + config.min_segment_ms
        max_end = current_start + config.max_segment_ms
        target_end = current_start + config.target_segment_ms

        candidates = [
            bp for bp in valid_breakpoints
            if min_end <= bp <= max_end
        ]

        if candidates:
            # Choose breakpoint closest to target
            best_bp = min(candidates, key=lambda bp: abs(bp - target_end))
            segments.append(SegmentInfo(
                start_ms=current_start,
                end_ms=best_bp,
                cut_reason="breakpoint",
                cut_breakpoint_ms=best_bp,
            ))
            current_start = best_bp
        else:
            # No breakpoint found - hard cut at max
            segments.append(SegmentInfo(
                start_ms=current_start,
                end_ms=max_end,
                cut_reason="hard_max",
                cut_breakpoint_ms=None,
            ))
            current_start = max_end

    # Handle short tail by merging into previous
    if len(segments) >= 2:
        last = segments[-1]
        if last.duration_ms < config.min_segment_ms:
            # Check if we can merge with previous
            prev = segments[-2]
            merged_duration = prev.duration_ms + last.duration_ms
            max_allowed = config.max_segment_ms + config.allow_over_max_tail_ms

            if merged_duration <= max_allowed:
                # Merge: extend previous to end
                segments[-2] = SegmentInfo(
                    start_ms=prev.start_ms,
                    end_ms=last.end_ms,
                    cut_reason=prev.cut_reason,
                    cut_breakpoint_ms=prev.cut_breakpoint_ms,
                )
                segments.pop()

    return segments


# =============================================================================
# ID Generation
# =============================================================================

def generate_segment_id(utterance_id: str, start_ms: int, end_ms: int) -> str:
    """
    Generate unique segment ID from parent + bounds.

    Format: sha1(utterance_id|start_ms|end_ms)[:16]
    """
    key = f"{utterance_id}|{start_ms}|{end_ms}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def generate_snapshot_id(dataset: str, config_hash: str) -> str:
    """Generate snapshot ID for segments manifest."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{dataset}-seg-{timestamp}-{config_hash[:8]}"


def config_to_hash(config: SegmentConfig, pause_params_hash: str) -> str:
    """Generate deterministic hash of segment config."""
    params = asdict(config)
    params["pause_params_hash"] = pause_params_hash
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha1(params_json.encode()).hexdigest()[:12]


# =============================================================================
# Tier 1 Validation
# =============================================================================

def validate_tier1_segment(seg: dict) -> None:
    """
    Validate that segment meets Tier 1 contract.

    Tier 1: All segments are unlabeled.

    Raises:
        AssertionError: If segment violates Tier 1 contract
    """
    assert seg["segment_label_status"] == "unlabeled", \
        f"Tier 1 requires unlabeled, got: {seg['segment_label_status']}"
    assert seg["text_span"] is None, \
        f"Tier 1 requires text_span=None, got: {seg['text_span']}"
    assert seg["phonemes_span"] is None, \
        f"Tier 1 requires phonemes_span=None, got: {seg['phonemes_span']}"
    assert seg["tokens"] is None, \
        f"Tier 1 requires tokens=None, got: {seg['tokens']}"


# =============================================================================
# Stats
# =============================================================================

@dataclass
class GoldSegmentStats:
    """Statistics for gold segment generation."""
    source_utterances: int = 0
    generated_segments: int = 0
    dropped_too_short: int = 0
    dropped_invalid_bounds: int = 0
    cut_breakpoint: int = 0
    cut_hard_max: int = 0
    cut_natural_end: int = 0

    # Distribution
    duration_p50: Optional[int] = None
    duration_p90: Optional[int] = None
    segments_per_utt_p50: Optional[float] = None
    segments_per_utt_p90: Optional[float] = None


def compute_gold_stats(
    segments: list[dict],
    segments_per_utterance: dict[str, int],
) -> GoldSegmentStats:
    """Compute statistics from generated segments."""
    stats = GoldSegmentStats()
    stats.generated_segments = len(segments)
    stats.source_utterances = len(segments_per_utterance)

    # Cut reason counts
    for seg in segments:
        reason = seg.get("cut_reason", "")
        if reason == "breakpoint":
            stats.cut_breakpoint += 1
        elif reason == "hard_max":
            stats.cut_hard_max += 1
        elif reason == "natural_end":
            stats.cut_natural_end += 1

    # Duration distribution
    durations = sorted([seg["duration_ms"] for seg in segments])
    if durations:
        stats.duration_p50 = durations[len(durations) // 2]
        stats.duration_p90 = durations[int(len(durations) * 0.9)]

    # Segments per utterance distribution
    counts = sorted(segments_per_utterance.values())
    if counts:
        stats.segments_per_utt_p50 = counts[len(counts) // 2]
        stats.segments_per_utt_p90 = counts[int(len(counts) * 0.9)]

    return stats


def print_gold_report(dataset: str, stats: GoldSegmentStats) -> None:
    """Print gold segment report."""
    print()
    print(f"Gold Segments Summary - {dataset}")
    print("=" * 40)
    print(f"Source utterances:       {stats.source_utterances:,}")
    print(f"Generated segments:      {stats.generated_segments:,}")
    print(f"Dropped (too short):     {stats.dropped_too_short}")
    print(f"Dropped (invalid bounds):{stats.dropped_invalid_bounds}")
    print()

    if stats.duration_p50:
        print(f"Segment duration (ms):   p50={stats.duration_p50:,}  p90={stats.duration_p90:,}")
    if stats.segments_per_utt_p50:
        print(f"Segments/utterance:      p50={stats.segments_per_utt_p50}      p90={stats.segments_per_utt_p90}")
    print()

    total_cuts = stats.cut_breakpoint + stats.cut_hard_max + stats.cut_natural_end
    if total_cuts > 0:
        print("Cut reasons:")
        if stats.cut_breakpoint:
            pct = stats.cut_breakpoint / total_cuts * 100
            print(f"  breakpoint:            {stats.cut_breakpoint:,} ({pct:.1f}%)")
        if stats.cut_hard_max:
            pct = stats.cut_hard_max / total_cuts * 100
            print(f"  hard_max:              {stats.cut_hard_max:,} ({pct:.1f}%)")
        if stats.cut_natural_end:
            pct = stats.cut_natural_end / total_cuts * 100
            print(f"  natural_end:           {stats.cut_natural_end:,} ({pct:.1f}%)")


# =============================================================================
# Main Pipeline
# =============================================================================

def read_gold_manifest_records(
    dataset: str,
    snapshot_id: Optional[str] = None,
) -> tuple[list[dict], str]:
    """
    Read gold manifest records.

    Returns:
        Tuple of (records list, snapshot_id)
    """
    manifests_dir = paths.gold / dataset / "manifests"

    if not manifests_dir.exists():
        raise FileNotFoundError(
            f"Gold manifests not found: {manifests_dir}\n"
            f"Run: koe gold {dataset}"
        )

    if snapshot_id:
        manifest_path = manifests_dir / f"{snapshot_id}.jsonl"
    else:
        manifests = sorted(manifests_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not manifests:
            raise FileNotFoundError(f"No manifests in {manifests_dir}")
        manifest_path = manifests[0]

    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))

    actual_snapshot = manifest_path.stem
    return records, actual_snapshot


def build_gold_segments(
    dataset: str,
    config: Optional[SegmentConfig] = None,
    source_snapshot: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Build gold segments manifest from silver breakpoints + gold utterances.

    Args:
        dataset: Dataset name
        config: Segment configuration
        source_snapshot: Source gold manifest snapshot ID
        dry_run: Print stats only, don't write

    Returns:
        Dict with status and stats
    """
    if config is None:
        config = SegmentConfig()

    print("=" * 60)
    print(f"Gold Segments Build - {dataset}")
    print("=" * 60)

    print(f"\nConfig:")
    print(f"  min_segment_ms: {config.min_segment_ms}")
    print(f"  max_segment_ms: {config.max_segment_ms}")
    print(f"  target_segment_ms: {config.target_segment_ms}")
    print(f"  min_lead_ms: {config.min_lead_ms}")

    # Read silver segment breaks
    print(f"\n[1/5] Reading silver segment breaks...")
    spark = get_spark()
    breaks_df = read_segment_breaks(dataset, spark)
    breaks_rows = breaks_df.collect()
    print(f"  Found {len(breaks_rows):,} utterances with breakpoints")

    # Build lookup: utterance_id -> breakpoints
    breakpoints_lookup = {}
    pause_params_hash = None
    for row in breaks_rows:
        breakpoints_lookup[row["utterance_id"]] = {
            "breakpoints_ms": list(row["breakpoints_ms"]),
            "duration_ms": row["duration_ms"],
        }
        if pause_params_hash is None:
            pause_params_hash = row["params_hash"]

    if pause_params_hash is None:
        pause_params_hash = "unknown"

    # Read gold manifest
    print(f"\n[2/5] Reading gold manifest...")
    parent_records, parent_snapshot = read_gold_manifest_records(dataset, source_snapshot)
    print(f"  Found {len(parent_records):,} parent utterances")
    print(f"  Source snapshot: {parent_snapshot}")

    # Generate snapshot ID
    segment_config_hash = config_to_hash(config, pause_params_hash)
    snapshot_id = generate_snapshot_id(dataset, segment_config_hash)
    print(f"  Segments snapshot: {snapshot_id}")

    # Generate segments
    print(f"\n[3/5] Generating segments...")
    segments = []
    segments_per_utterance = {}
    dropped_too_short = 0
    dropped_invalid_bounds = 0
    now = datetime.now(timezone.utc)

    for parent in parent_records:
        utterance_id = parent["utterance_id"]
        bp_info = breakpoints_lookup.get(utterance_id)

        if bp_info is None:
            # No breakpoints detected for this utterance
            # Create single segment covering full utterance
            duration_ms = int(parent.get("duration_sec", 0) * 1000)
            seg_infos = [SegmentInfo(
                start_ms=0,
                end_ms=duration_ms,
                cut_reason="natural_end",
                cut_breakpoint_ms=None,
            )] if duration_ms >= config.min_segment_ms else []
        else:
            duration_ms = bp_info["duration_ms"]
            breakpoints = bp_info["breakpoints_ms"]
            seg_infos = compute_segments_from_breakpoints(breakpoints, duration_ms, config)

        segment_count = 0
        for seg_idx, seg_info in enumerate(seg_infos):
            # Validate bounds
            if seg_info.start_ms < 0 or seg_info.end_ms > duration_ms:
                dropped_invalid_bounds += 1
                continue

            # Validate minimum duration
            if seg_info.duration_ms < config.min_segment_ms:
                dropped_too_short += 1
                continue

            # Generate segment record
            segment_id = generate_segment_id(utterance_id, seg_info.start_ms, seg_info.end_ms)

            segment = {
                # Identity
                "snapshot_id": snapshot_id,
                "segment_id": segment_id,
                "segment_index": seg_idx,

                # Parent reference
                "utterance_id": utterance_id,
                "speaker_id": parent.get("speaker_id", "unknown"),
                "split": parent.get("split"),

                # Audio (reference, not copy)
                "audio_relpath": parent.get("audio_relpath"),
                "audio_abspath": parent.get("audio_abspath"),
                "start_ms": seg_info.start_ms,
                "end_ms": seg_info.end_ms,
                "duration_ms": seg_info.duration_ms,
                "duration_sec": round(seg_info.duration_ms / 1000, 3),

                # Tier 1: Always unlabeled
                "segment_label_status": "unlabeled",
                "text_span": None,
                "phonemes_span": None,
                "tokens": None,

                # Parent text (for reference, not for training)
                "parent_text": parent.get("text"),
                "parent_phonemes": parent.get("phonemes"),

                # Segmentation metadata
                "cut_reason": seg_info.cut_reason,
                "cut_breakpoint_ms": seg_info.cut_breakpoint_ms,

                # Provenance
                "source_dataset": dataset,
                "source_snapshot_id": parent_snapshot,
                "source_utterance_id": utterance_id,
                "source_utterance_key": parent.get("utterance_key"),

                # Config tracking
                "pause_method": "pau_v1_adaptive",
                "pause_params_hash": pause_params_hash,
                "segment_method": "seg_v1_greedy",
                "segment_params_hash": segment_config_hash,

                # Metadata
                "created_at": now.isoformat(),
            }

            # Validate Tier 1 contract
            validate_tier1_segment(segment)

            segments.append(segment)
            segment_count += 1

        segments_per_utterance[utterance_id] = segment_count

    print(f"  Generated {len(segments):,} segments")
    print(f"  Dropped (too short): {dropped_too_short}")
    print(f"  Dropped (invalid bounds): {dropped_invalid_bounds}")

    # Validate uniqueness
    print(f"\n[4/5] Validating uniqueness...")
    segment_ids = [s["segment_id"] for s in segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("Segment ID collision detected!")
    print(f"  All {len(segment_ids):,} segment IDs unique")

    # Compute stats
    stats = compute_gold_stats(segments, segments_per_utterance)
    stats.dropped_too_short = dropped_too_short
    stats.dropped_invalid_bounds = dropped_invalid_bounds
    print_gold_report(dataset, stats)

    # Write manifest
    output_dir = paths.gold / dataset / "segments"
    manifest_path = output_dir / f"{snapshot_id}.jsonl"

    if dry_run:
        print(f"\n[5/5] Dry run - skipping write")
        print(f"  Would write to: {manifest_path}")
    else:
        print(f"\n[5/5] Writing manifest...")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(segments):,} segments to {manifest_path}")

        # Write split-specific manifests
        by_split = {}
        for seg in segments:
            split = seg.get("split", "train")
            if split not in by_split:
                by_split[split] = []
            by_split[split].append(seg)

        for split, split_segs in by_split.items():
            split_path = output_dir / f"{snapshot_id}.{split}.jsonl"
            with open(split_path, "w", encoding="utf-8") as f:
                for seg in split_segs:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            print(f"  Wrote {len(split_segs):,} {split} segments to {split_path.name}")

    print("\n" + "=" * 60)
    print("Gold Segments Complete")
    print("=" * 60)

    return {
        "status": "success",
        "snapshot_id": snapshot_id,
        "manifest_path": str(manifest_path) if not dry_run else None,
        "n_segments": len(segments),
        "stats": asdict(stats),
    }


def list_gold_segments(dataset: str) -> dict:
    """
    List gold segment manifests for a dataset.

    Returns:
        Dict with manifest inventory
    """
    segments_dir = paths.gold / dataset / "segments"

    if not segments_dir.exists():
        return {
            "status": "not_found",
            "dataset": dataset,
            "path": str(segments_dir),
        }

    manifests = sorted(segments_dir.glob("*.jsonl"))

    # Filter to main manifests (not split-specific)
    main_manifests = [
        m for m in manifests
        if not any(m.stem.endswith(f".{split}") for split in ["train", "val", "test"])
    ]

    inventory = []
    for manifest_path in main_manifests:
        # Count segments
        n_segments = 0
        with open(manifest_path) as f:
            for _ in f:
                n_segments += 1

        inventory.append({
            "snapshot_id": manifest_path.stem,
            "path": str(manifest_path),
            "n_segments": n_segments,
            "modified": datetime.fromtimestamp(manifest_path.stat().st_mtime).isoformat(),
        })

    return {
        "status": "found",
        "dataset": dataset,
        "path": str(segments_dir),
        "manifests": inventory,
    }
