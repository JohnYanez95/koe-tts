"""
JSUT corpus gold layer processing.

Gold = "what exactly is the dataset I will train on, with stable splits
and a stable snapshot id?"

Command: koe gold jsut

Reads:
    lake/silver/jsut/utterances (Delta table)

Writes:
    lake/gold/jsut/utterances (Delta table)
    lake/gold/jsut/manifests/{snapshot_id}.jsonl (training manifest)
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from modules.data_engineering.common.io import write_table
from modules.data_engineering.common.paths import paths
from modules.forge.query.spark import get_spark

# Gold pipeline version
GOLD_VERSION = "v0.1-stub"

# Default filter bounds
DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_DURATION = 20.0

# Default split ratios
DEFAULT_VAL_PCT = 0.10
DEFAULT_TEST_PCT = 0.0  # No test set for now
DEFAULT_SEED = 42


def read_silver_jsut(spark: SparkSession) -> DataFrame:
    """
    Read JSUT silver table.

    Returns:
        Silver DataFrame

    Raises:
        FileNotFoundError: If silver table doesn't exist
    """
    silver_path = paths.silver / "jsut" / "utterances"

    if not silver_path.exists():
        raise FileNotFoundError(
            f"JSUT silver table not found: {silver_path}\n"
            "Run silver first: koe silver jsut"
        )

    return spark.read.format("delta").load(str(silver_path))


def filter_trainable(
    df: DataFrame,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> tuple[DataFrame, dict]:
    """
    Filter to trainable rows only.

    Args:
        df: Silver DataFrame
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Tuple of (filtered DataFrame, filter stats)
    """
    total_count = df.count()

    # Apply filters: is_trainable + duration bounds
    filtered = df.filter(
        (F.col("is_trainable") == True) &
        (F.col("duration_sec") >= min_duration) &
        (F.col("duration_sec") <= max_duration)
    )

    filtered_count = filtered.count()

    # Compute exclusion stats
    excluded_trainable = df.filter(F.col("is_trainable") == False).count()
    excluded_short = df.filter(F.col("duration_sec") < min_duration).count()
    excluded_long = df.filter(F.col("duration_sec") > max_duration).count()

    stats = {
        "total_silver": total_count,
        "after_filter": filtered_count,
        "excluded_total": total_count - filtered_count,
        "excluded_not_trainable": excluded_trainable,
        "excluded_too_short": excluded_short,
        "excluded_too_long": excluded_long,
    }

    return filtered, stats


def use_silver_split_or_fallback(
    df: DataFrame,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
) -> tuple[DataFrame, str]:
    """
    Use silver's persisted split column, or fall back to recomputation.

    Silver v1.1+ computes splits once and persists them. Gold should consume
    these splits rather than recomputing. This function handles backwards
    compatibility with older silver tables that don't have splits.

    Args:
        df: DataFrame with utterance_id and split columns
        val_pct: Fallback validation fraction (only used if split is null)
        test_pct: Fallback test fraction (only used if split is null)
        seed: Fallback seed (only used if split is null)

    Returns:
        Tuple of (DataFrame with split column, mode string)
        mode is "silver" if using persisted splits, "fallback" if recomputed
    """
    # Check if silver has computed splits
    null_split_count = df.filter(F.col("split").isNull()).count()
    total_count = df.count()

    if null_split_count == 0:
        # All rows have splits from silver - use them
        print("  Using splits from silver layer (persisted)")
        return df, "silver"
    elif null_split_count == total_count:
        # No splits in silver - compute as fallback
        print(f"  WARNING: Silver has no splits, computing fallback (val={val_pct}, test={test_pct}, seed={seed})")

        @F.udf(StringType())
        def compute_split(utterance_id: str) -> str:
            hash_input = f"{utterance_id}_{seed}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            p = (hash_val % 10000) / 10000.0
            if p < val_pct:
                return "val"
            elif p < val_pct + test_pct:
                return "test"
            else:
                return "train"

        return df.withColumn("split", compute_split(F.col("utterance_id"))), "fallback"
    else:
        # Partial splits - this is an error state
        raise ValueError(
            f"Silver has partial splits ({null_split_count}/{total_count} null). "
            "Re-run silver pipeline to fix: koe silver jsut --force"
        )


def compute_duration_bucket(df: DataFrame) -> DataFrame:
    """
    Assign duration buckets for length-balanced batching.

    Buckets: xs (<2s), s (2-4s), m (4-6s), l (6-8s), xl (8-10s), xxl (>10s)
    """
    return df.withColumn(
        "duration_bucket",
        F.when(F.col("duration_sec") < 2.0, "xs")
        .when(F.col("duration_sec") < 4.0, "s")
        .when(F.col("duration_sec") < 6.0, "m")
        .when(F.col("duration_sec") < 8.0, "l")
        .when(F.col("duration_sec") < 10.0, "xl")
        .otherwise("xxl")
    )


def build_gold_df(
    spark: SparkSession,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
) -> tuple[DataFrame, dict]:
    """
    Build gold DataFrame from silver.

    Args:
        spark: SparkSession
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        val_pct: Validation set fraction
        test_pct: Test set fraction
        seed: Random seed for splits

    Returns:
        Tuple of (gold DataFrame, build stats)
    """
    print("Reading silver table...")
    silver_df = read_silver_jsut(spark)
    silver_count = silver_df.count()
    print(f"  Silver records: {silver_count}")

    # Filter to trainable
    print(f"Filtering (duration {min_duration}-{max_duration}s, is_trainable=true)...")
    filtered_df, filter_stats = filter_trainable(silver_df, min_duration, max_duration)
    print(f"  After filter: {filter_stats['after_filter']} records")
    print(f"  Excluded: {filter_stats['excluded_total']}")

    # Use silver splits (or fall back to recomputation for old silver tables)
    print("Resolving splits...")
    split_df, split_mode = use_silver_split_or_fallback(filtered_df, val_pct, test_pct, seed)

    # Compute duration buckets
    print("Computing duration buckets...")
    bucket_df = compute_duration_bucket(split_df)

    # Build canonical training columns
    print("Building training columns...")
    gold_df = bucket_df.select(
        # Keys
        F.col("utterance_id"),
        F.col("utterance_key"),

        # Source
        F.col("dataset"),
        F.col("speaker_id"),

        # Audio
        F.col("audio_relpath"),
        F.col("duration_sec"),
        F.col("sample_rate"),

        # Text - coalesce: prefer canonical text_norm, fall back to raw
        F.coalesce(
            F.col("text_norm"),       # Silver canonical (may be null for stub)
            F.col("text_norm_raw"),   # Bronze corpus-provided
            F.col("text_raw")         # Original
        ).alias("text"),

        # Phonemes - prefer canonical, fall back to raw
        F.coalesce(
            F.col("phonemes"),        # Silver canonical (may be null for stub)
            F.col("phonemes_raw")     # Bronze corpus-provided
        ).alias("phonemes"),
        F.coalesce(
            F.col("phonemes_method"), # Silver method (may be null)
            F.col("phonemes_source")  # Bronze source
        ).alias("phonemes_source"),

        # Splits
        F.col("split"),
        F.col("duration_bucket"),

        # Metadata for training
        F.lit(1.0).alias("sample_weight"),  # Default weight
        F.lit(GOLD_VERSION).alias("gold_version"),
        F.lit(None).cast("long").alias("silver_version"),  # TODO: get Delta version
        F.current_timestamp().alias("created_at"),
    )

    # Compute split distribution
    split_counts = gold_df.groupBy("split").count().collect()
    split_dist = {row["split"]: row["count"] for row in split_counts}

    stats = {
        **filter_stats,
        "split_distribution": split_dist,
        "gold_version": GOLD_VERSION,
    }

    return gold_df, stats


def generate_snapshot_id(dataset: str, config_hash: str = None) -> str:
    """
    Generate a unique snapshot ID.

    Format: {dataset}-{yyyymmdd-HHMMSS}-{short_hash}
    """
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%d-%H%M%S")

    if config_hash:
        short_hash = config_hash[:8]
    else:
        # Hash the timestamp for uniqueness
        short_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]

    return f"{dataset}-{timestamp}-{short_hash}"


def _write_manifest_to_path(
    df: DataFrame,
    snapshot_id: str,
    manifest_path: Path,
) -> None:
    """
    Write manifest JSONL to a specific path.

    Args:
        df: Gold DataFrame
        snapshot_id: Snapshot identifier
        manifest_path: Full path to manifest file
    """
    data_root = paths.data_root
    rows = df.collect()

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "snapshot_id": snapshot_id,
                "utterance_id": row["utterance_id"],
                "utterance_key": row["utterance_key"],
                "split": row["split"],
                "audio_relpath": row["audio_relpath"],
                "audio_abspath": str(data_root / "data" / row["audio_relpath"]),
                "text": row["text"],
                "phonemes": row["phonemes"],
                "speaker_id": row["speaker_id"],
                "duration_sec": float(row["duration_sec"]),
                "sample_rate": int(row["sample_rate"]),
                "duration_bucket": row["duration_bucket"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(rows)} records")


def export_manifest_jsonl(
    df: DataFrame,
    snapshot_id: str,
    output_dir: Path,
) -> Path:
    """
    Export gold DataFrame as JSONL manifest for training.

    Args:
        df: Gold DataFrame
        snapshot_id: Snapshot identifier
        output_dir: Directory to write manifest

    Returns:
        Path to manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{snapshot_id}.jsonl"

    print(f"Exporting manifest: {manifest_path}")
    _write_manifest_to_path(df, snapshot_id, manifest_path)
    return manifest_path


def validate_gold_df(df: DataFrame) -> dict:
    """
    Validate gold DataFrame.

    Returns:
        Dict with validation results

    Raises:
        ValueError: If validation fails
    """
    print("\nValidating gold DataFrame...")
    stats = {}

    total_count = df.count()
    stats["total_count"] = total_count
    print(f"  Total records: {total_count}")

    # Check for null text (critical for training)
    null_text = df.filter(F.col("text").isNull()).count()
    if null_text > 0:
        raise ValueError(f"Found {null_text} records with null text!")
    print("  Null text: 0 ✓")

    # Check for null audio_relpath
    null_audio = df.filter(F.col("audio_relpath").isNull()).count()
    if null_audio > 0:
        raise ValueError(f"Found {null_audio} records with null audio_relpath!")
    print("  Null audio_relpath: 0 ✓")

    # Split distribution
    split_counts = df.groupBy("split").count().orderBy("split").collect()
    stats["split_distribution"] = {row["split"]: row["count"] for row in split_counts}
    print("  Split distribution:")
    for row in split_counts:
        pct = row["count"] / total_count * 100
        print(f"    {row['split']}: {row['count']} ({pct:.1f}%)")

    # Duration bucket distribution
    bucket_counts = df.groupBy("duration_bucket").count().orderBy("duration_bucket").collect()
    stats["bucket_distribution"] = {row["duration_bucket"]: row["count"] for row in bucket_counts}
    print("  Duration buckets:")
    for row in bucket_counts:
        print(f"    {row['duration_bucket']}: {row['count']}")

    # Phonemes status (how many have phonemes)
    has_phonemes = df.filter(F.col("phonemes").isNotNull()).count()
    stats["has_phonemes"] = has_phonemes
    stats["missing_phonemes"] = total_count - has_phonemes
    print(f"  Has phonemes: {has_phonemes} ({has_phonemes/total_count*100:.1f}%)")
    print(f"  Missing phonemes: {total_count - has_phonemes}")

    print("  Validation PASSED")
    return stats


def build_gold_jsut(
    snapshot_id: str | None = None,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
    export_jsonl: bool = True,
    write_delta: bool = True,
    manifest_out: str | None = None,
) -> dict:
    """
    Build JSUT gold table and manifest.

    This is the main entry point for `koe gold jsut`.

    Args:
        snapshot_id: Optional snapshot ID (auto-generated if not provided)
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        val_pct: Validation set fraction
        test_pct: Test set fraction
        seed: Random seed for splits
        export_jsonl: Whether to export JSONL manifest
        write_delta: Whether to write Delta table
        manifest_out: Override manifest output path

    Returns:
        Dict with build results and stats
    """
    print("=" * 60)
    print("JSUT Gold Pipeline")
    print("=" * 60)

    # Compute train_pct from val/test
    train_pct = 1.0 - val_pct - test_pct

    # Print config upfront
    print("\nConfig:")
    print(f"  seed: {seed}")
    print(f"  splits: train={train_pct:.0%} / val={val_pct:.0%} / test={test_pct:.0%}")
    print(f"  duration: {min_duration}s - {max_duration}s")

    spark = get_spark()

    # Generate snapshot ID
    if snapshot_id is None:
        config_str = f"{min_duration}_{max_duration}_{val_pct}_{test_pct}_{seed}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        snapshot_id = generate_snapshot_id("jsut", config_hash)
    print(f"  snapshot_id: {snapshot_id}")

    # Build gold DataFrame
    print("\n[1/4] Building gold DataFrame...")
    gold_df, build_stats = build_gold_df(
        spark,
        min_duration=min_duration,
        max_duration=max_duration,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
    )

    # Validate
    print("\n[2/4] Validating...")
    val_stats = validate_gold_df(gold_df)

    # Write Delta table
    output_path = paths.gold / "jsut" / "utterances"
    if write_delta:
        print("\n[3/4] Writing Delta table...")
        print(f"  Output: {output_path}")

        write_table(
            gold_df,
            layer="gold",
            table_name="jsut/utterances",
            mode="overwrite",
            partition_by=["dataset", "split"],
        )
    else:
        print("\n[3/4] Skipping Delta table write (--no-write-delta)")

    # Export JSONL manifest
    manifest_path = None
    if export_jsonl:
        print("\n[4/4] Exporting JSONL manifest...")
        if manifest_out:
            # Use provided manifest path directly
            manifest_path = Path(manifest_out)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Exporting manifest: {manifest_path}")
            _write_manifest_to_path(gold_df, snapshot_id, manifest_path)
        else:
            manifests_dir = paths.gold / "jsut" / "manifests"
            manifest_path = export_manifest_jsonl(gold_df, snapshot_id, manifests_dir)

    # Build summary
    total = val_stats['total_count']
    split_dist = val_stats['split_distribution']

    print("\n" + "=" * 60)
    print("JSUT Gold Complete")
    print("=" * 60)
    print(f"\nSnapshot: {snapshot_id}")
    print(f"Seed: {seed}")
    print(f"\nDuration filter: {min_duration}s - {max_duration}s")
    print(f"  Silver records: {build_stats['total_silver']}")
    print(f"  After filter:   {build_stats['after_filter']} (dropped {build_stats['excluded_total']})")
    print("\nSplit breakdown:")
    for split_name in ["train", "val", "test"]:
        if split_name in split_dist:
            count = split_dist[split_name]
            pct = count / total * 100
            print(f"  {split_name:5}: {count:6} ({pct:5.1f}%)")
    print(f"  {'total':5}: {total:6}")
    print("\nOutputs:")
    if write_delta:
        print(f"  Delta table: {output_path}")
    if manifest_path:
        print(f"  Manifest:    {manifest_path}")
    print("=" * 60)

    return {
        "status": "success",
        "snapshot_id": snapshot_id,
        "output_path": str(output_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "gold_version": GOLD_VERSION,
        "build_stats": build_stats,
        "validation_stats": val_stats,
    }


if __name__ == "__main__":
    result = build_gold_jsut()
    print(f"\nResult: {result['status']}")
