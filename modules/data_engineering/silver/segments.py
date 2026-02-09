"""
Silver segment breaks detection.

Processes audio to detect pause-based breakpoints for sub-utterance segmentation.
Stores full silence regions + derived breakpoints (midpoints) in Delta table.

Usage:
    from modules.data_engineering.silver.segments import build_segment_breaks

    result = build_segment_breaks(
        dataset="jsut",
        config=PauseDetectionConfig(),
        limit=100,  # For testing
    )

Command: koe segment auto jsut
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchaudio
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from modules.data_engineering.common.audio import (
    PauseDetectionConfig,
    SilenceRegion,
    detect_silence_regions,
    regions_to_breakpoints,
)
from modules.data_engineering.common.io import write_table
from modules.data_engineering.common.paths import paths
from modules.forge.query.spark import get_spark

# Pipeline version for tracking
SEGMENT_BREAKS_VERSION = "v1.0"


# =============================================================================
# Schema
# =============================================================================

SEGMENT_BREAKS_SCHEMA = StructType([
    StructField("dataset", StringType(), nullable=False),
    StructField("utterance_id", StringType(), nullable=False),
    StructField("speaker_id", StringType(), nullable=False),
    StructField("split", StringType(), nullable=True),
    StructField("duration_ms", IntegerType(), nullable=False),

    # Full silence regions
    StructField("silence_regions_ms", ArrayType(
        StructType([
            StructField("start_ms", IntegerType()),
            StructField("end_ms", IntegerType()),
        ])
    ), nullable=False),
    StructField("n_regions", IntegerType(), nullable=False),

    # Derived breakpoints (midpoints, filtered by margin)
    StructField("breakpoints_ms", ArrayType(IntegerType()), nullable=False),
    StructField("n_breakpoints", IntegerType(), nullable=False),

    # Debug/reproducibility info
    StructField("rms_db_p10", FloatType(), nullable=False),
    StructField("threshold_db_used", FloatType(), nullable=False),
    StructField("thr_formula", StringType(), nullable=False),
    StructField("silence_pct", FloatType(), nullable=False),

    # Config tracking
    StructField("method", StringType(), nullable=False),
    StructField("params_json", StringType(), nullable=False),
    StructField("params_hash", StringType(), nullable=False),

    StructField("pipeline_version", StringType(), nullable=False),
    StructField("created_at", TimestampType(), nullable=False),
])


# =============================================================================
# Audio Processing
# =============================================================================

def load_audio_for_segmentation(
    audio_path: str,
    target_sr: int = 22050,
) -> tuple[torch.Tensor, int]:
    """
    Load audio and resample for segmentation.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (waveform [T], sample_rate)
    """
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        sr = target_sr

    return waveform, sr


def process_utterance(
    audio_path: str,
    config: PauseDetectionConfig,
    min_lead_ms: int = 250,
) -> tuple[list[SilenceRegion], list[int], dict]:
    """
    Process a single utterance for pause detection.

    Args:
        audio_path: Path to audio file
        config: Pause detection configuration
        min_lead_ms: Minimum distance from audio boundaries for breakpoints

    Returns:
        Tuple of (regions, breakpoints, debug_info)
    """
    waveform, sr = load_audio_for_segmentation(audio_path)
    duration_ms = int(len(waveform) * 1000 / sr)

    regions, debug_info = detect_silence_regions(waveform, sr, config)

    breakpoints = regions_to_breakpoints(
        regions,
        duration_ms,
        min_lead_ms=min_lead_ms,
        min_tail_ms=min_lead_ms,
    )

    debug_info["duration_ms"] = duration_ms

    return regions, breakpoints, debug_info


# =============================================================================
# Config Hashing
# =============================================================================

def config_to_method_name(config: PauseDetectionConfig) -> str:
    """Generate method name from config."""
    if config.adaptive:
        return "pau_v1_adaptive"
    else:
        return "pau_v1_manual"


def config_to_hash(config: PauseDetectionConfig) -> str:
    """Generate deterministic hash of config for reproducibility."""
    params = asdict(config)
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha1(params_json.encode()).hexdigest()[:12]


# =============================================================================
# Report Generation
# =============================================================================

@dataclass
class SegmentBreaksStats:
    """Statistics for segment breaks processing."""
    total_processed: int = 0
    with_pause: int = 0
    with_multi_pause: int = 0
    failed: int = 0

    # Distribution percentiles
    pause_duration_p50: float | None = None
    pause_duration_p90: float | None = None
    breakpoints_per_utt_p50: float | None = None
    breakpoints_per_utt_p90: float | None = None
    threshold_p50: float | None = None
    threshold_p90: float | None = None
    rms_p10_p50: float | None = None
    rms_p10_p90: float | None = None

    # Silence ratio (explains breakpoint counts)
    silence_pct_p50: float | None = None
    silence_pct_p90: float | None = None
    silence_pct_high_count: int = 0  # utterances with >20% silence


def compute_stats_from_results(results: list[dict]) -> SegmentBreaksStats:
    """Compute statistics from processing results."""
    stats = SegmentBreaksStats()
    stats.total_processed = len(results)

    pause_durations = []
    breakpoint_counts = []
    thresholds = []
    rms_p10s = []
    silence_pcts = []

    for r in results:
        n_regions = r["n_regions"]
        n_bp = r["n_breakpoints"]

        if n_regions > 0:
            stats.with_pause += 1
            for region in r["silence_regions_ms"]:
                dur = region["end_ms"] - region["start_ms"]
                pause_durations.append(dur)
        if n_bp >= 2:
            stats.with_multi_pause += 1

        breakpoint_counts.append(n_bp)
        thresholds.append(r["threshold_db_used"])
        rms_p10s.append(r["rms_db_p10"])

        # Silence ratio
        silence_pct = r.get("silence_pct", 0.0)
        silence_pcts.append(silence_pct)
        if silence_pct > 20.0:
            stats.silence_pct_high_count += 1

    # Compute percentiles
    if pause_durations:
        pause_durations.sort()
        stats.pause_duration_p50 = pause_durations[len(pause_durations) // 2]
        stats.pause_duration_p90 = pause_durations[int(len(pause_durations) * 0.9)]

    if breakpoint_counts:
        breakpoint_counts.sort()
        stats.breakpoints_per_utt_p50 = breakpoint_counts[len(breakpoint_counts) // 2]
        stats.breakpoints_per_utt_p90 = breakpoint_counts[int(len(breakpoint_counts) * 0.9)]

    if thresholds:
        thresholds.sort()
        stats.threshold_p50 = round(thresholds[len(thresholds) // 2], 1)
        stats.threshold_p90 = round(thresholds[int(len(thresholds) * 0.9)], 1)

    if rms_p10s:
        rms_p10s.sort()
        stats.rms_p10_p50 = round(rms_p10s[len(rms_p10s) // 2], 1)
        stats.rms_p10_p90 = round(rms_p10s[int(len(rms_p10s) * 0.9)], 1)

    if silence_pcts:
        silence_pcts.sort()
        stats.silence_pct_p50 = round(silence_pcts[len(silence_pcts) // 2], 1)
        stats.silence_pct_p90 = round(silence_pcts[int(len(silence_pcts) * 0.9)], 1)

    return stats


def print_distribution_report(dataset: str, stats: SegmentBreaksStats) -> None:
    """Print distribution report to console."""
    total = stats.total_processed
    with_pause = stats.with_pause
    with_multi = stats.with_multi_pause

    print()
    print(f"Segment Breaks Summary - {dataset}")
    print("=" * 40)
    print(f"Utterances processed:    {total:,}")
    if total > 0:
        print(f"With >=1 pause:          {with_pause:,} ({with_pause/total*100:.1f}%)")
        print(f"With >=2 pauses:         {with_multi:,} ({with_multi/total*100:.1f}%)")
    print()

    if stats.pause_duration_p50 is not None:
        print(f"Pause duration (ms):     p50={int(stats.pause_duration_p50)}  p90={int(stats.pause_duration_p90)}")
    if stats.breakpoints_per_utt_p50 is not None:
        print(f"Breakpoints/utterance:   p50={int(stats.breakpoints_per_utt_p50)}    p90={int(stats.breakpoints_per_utt_p90)}")
    if stats.threshold_p50 is not None:
        print(f"Threshold (dB):          p50={stats.threshold_p50}  p90={stats.threshold_p90}  (adaptive)")
    if stats.rms_p10_p50 is not None:
        print(f"RMS p10 (dB):            p50={stats.rms_p10_p50}  p90={stats.rms_p10_p90}")
    if stats.silence_pct_p50 is not None:
        high_pct = stats.silence_pct_high_count / total * 100 if total > 0 else 0
        print(f"Silence ratio (%):       p50={stats.silence_pct_p50}  p90={stats.silence_pct_p90}  (>20%: {high_pct:.1f}% of utterances)")


# =============================================================================
# Main Pipeline
# =============================================================================

def read_gold_manifest(dataset: str, snapshot_id: str | None = None) -> list[dict]:
    """
    Read gold manifest for a dataset.

    Args:
        dataset: Dataset name
        snapshot_id: Specific snapshot ID (uses latest if None)

    Returns:
        List of manifest records
    """
    manifests_dir = paths.gold / dataset / "manifests"

    if not manifests_dir.exists():
        raise FileNotFoundError(
            f"Gold manifests not found: {manifests_dir}\n"
            f"Run gold first: koe gold {dataset}"
        )

    if snapshot_id:
        manifest_path = manifests_dir / f"{snapshot_id}.jsonl"
    else:
        # Find latest manifest
        manifests = sorted(manifests_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not manifests:
            raise FileNotFoundError(f"No manifests found in {manifests_dir}")
        manifest_path = manifests[0]
        print(f"Using latest manifest: {manifest_path.name}")

    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))

    return records


def build_segment_breaks(
    dataset: str,
    config: PauseDetectionConfig | None = None,
    source_snapshot: str | None = None,
    min_lead_ms: int = 250,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Build segment breaks table from gold manifest.

    Processes audio files to detect pauses and stores results in
    lake/silver/{dataset}/segment_breaks Delta table.

    Args:
        dataset: Dataset name (jsut, jvs, etc.)
        config: Pause detection config (uses defaults if None)
        source_snapshot: Source gold manifest snapshot ID (latest if None)
        min_lead_ms: Minimum distance from audio boundaries for breakpoints
        limit: Process only N utterances (for testing)
        dry_run: Print stats only, don't write table
        force: Overwrite existing table

    Returns:
        Dict with status, stats, and output path
    """
    if config is None:
        config = PauseDetectionConfig()

    print("=" * 60)
    print(f"Segment Breaks Detection - {dataset}")
    print("=" * 60)

    # Config info
    method = config_to_method_name(config)
    params_hash = config_to_hash(config)
    params_json = json.dumps(asdict(config), sort_keys=True)

    print("\nConfig:")
    print(f"  method: {method}")
    print(f"  params_hash: {params_hash}")
    print(f"  adaptive: {config.adaptive}")
    print(f"  floor_db: {config.floor_db}")
    print(f"  margin_db: {config.margin_db}")
    print(f"  min_pause_ms: {config.min_pause_ms}")

    # Read source manifest
    print("\n[1/4] Reading gold manifest...")
    manifest = read_gold_manifest(dataset, source_snapshot)
    print(f"  Found {len(manifest):,} utterances")

    if limit:
        manifest = manifest[:limit]
        print(f"  Limited to {limit} utterances")

    # Process utterances
    print("\n[2/4] Processing audio files...")
    results = []
    failed = 0
    now = datetime.now(UTC)

    for i, record in enumerate(manifest):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(manifest)}...")

        audio_path = record.get("audio_abspath") or record.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            failed += 1
            continue

        try:
            regions, breakpoints, debug_info = process_utterance(
                audio_path, config, min_lead_ms
            )

            result = {
                "dataset": dataset,
                "utterance_id": record["utterance_id"],
                "speaker_id": record.get("speaker_id", "unknown"),
                "split": record.get("split"),
                "duration_ms": debug_info["duration_ms"],
                "silence_regions_ms": [r.to_dict() for r in regions],
                "n_regions": len(regions),
                "breakpoints_ms": breakpoints,
                "n_breakpoints": len(breakpoints),
                "rms_db_p10": debug_info.get("rms_db_p10", 0.0),
                "threshold_db_used": debug_info.get("threshold_db_used", config.floor_db),
                "thr_formula": debug_info.get("thr_formula", ""),
                "silence_pct": debug_info.get("silence_pct", 0.0),
                "method": method,
                "params_json": params_json,
                "params_hash": params_hash,
                "pipeline_version": SEGMENT_BREAKS_VERSION,
                "created_at": now,
            }
            results.append(result)

        except Exception as e:
            print(f"  Warning: Failed to process {record['utterance_id']}: {e}")
            failed += 1

    print(f"  Processed: {len(results):,}")
    if failed > 0:
        print(f"  Failed: {failed}")

    # Compute stats
    print("\n[3/4] Computing statistics...")
    stats = compute_stats_from_results(results)
    stats.failed = failed
    print_distribution_report(dataset, stats)

    # Write table
    output_path = paths.silver / dataset / "segment_breaks"

    if dry_run:
        print("\n[4/4] Dry run - skipping write")
        print(f"  Would write to: {output_path}")
    else:
        print("\n[4/4] Writing Delta table...")
        print(f"  Output: {output_path}")

        # Check existing
        if output_path.exists() and not force:
            print("  Table exists. Use --force to overwrite.")
            return {
                "status": "skipped",
                "reason": "exists",
                "output_path": str(output_path),
            }

        spark = get_spark()

        # Convert to Spark DataFrame
        df = spark.createDataFrame(results, schema=SEGMENT_BREAKS_SCHEMA)

        write_table(
            df,
            layer="silver",
            table_name=f"{dataset}/segment_breaks",
            mode="overwrite",
            partition_by=["dataset", "split"],
        )
        print(f"  Wrote {len(results):,} records")

    print("\n" + "=" * 60)
    print("Segment Breaks Complete")
    print("=" * 60)

    return {
        "status": "success",
        "dataset": dataset,
        "output_path": str(output_path),
        "n_processed": len(results),
        "n_failed": failed,
        "method": method,
        "params_hash": params_hash,
        "stats": asdict(stats),
    }


def read_segment_breaks(
    dataset: str,
    spark: SparkSession | None = None,
) -> DataFrame:
    """
    Read segment breaks table for a dataset.

    Args:
        dataset: Dataset name
        spark: SparkSession (creates one if None)

    Returns:
        Spark DataFrame with segment breaks
    """
    if spark is None:
        spark = get_spark()

    table_path = paths.silver / dataset / "segment_breaks"

    if not table_path.exists():
        raise FileNotFoundError(
            f"Segment breaks not found: {table_path}\n"
            f"Run: koe segment auto {dataset}"
        )

    return spark.read.format("delta").load(str(table_path))


def list_segment_breaks(dataset: str) -> dict:
    """
    List segment breaks inventory and stats.

    Args:
        dataset: Dataset name

    Returns:
        Dict with inventory information
    """
    table_path = paths.silver / dataset / "segment_breaks"

    if not table_path.exists():
        return {
            "status": "not_found",
            "dataset": dataset,
            "path": str(table_path),
        }

    spark = get_spark()
    df = spark.read.format("delta").load(str(table_path))

    total = df.count()
    with_pause = df.filter(F.col("n_regions") > 0).count()
    with_breakpoints = df.filter(F.col("n_breakpoints") > 0).count()

    # Get method info
    method_row = df.select("method", "params_hash").first()
    method = method_row["method"] if method_row else "unknown"
    params_hash = method_row["params_hash"] if method_row else "unknown"

    # Split distribution
    split_counts = df.groupBy("split").count().collect()
    split_dist = {row["split"]: row["count"] for row in split_counts}

    return {
        "status": "found",
        "dataset": dataset,
        "path": str(table_path),
        "total_utterances": total,
        "with_pause": with_pause,
        "with_breakpoints": with_breakpoints,
        "method": method,
        "params_hash": params_hash,
        "split_distribution": split_dist,
    }
