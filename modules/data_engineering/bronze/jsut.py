"""
JSUT corpus bronze layer ingestion.

Reads from ingest outputs and writes to bronze Delta table.

Command: koe bronze jsut

Reads:
    data/ingest/jsut/raw/MANIFEST.json
    data/ingest/jsut/extracted/**/transcript_utf8.txt
    data/ingest/jsut/extracted/audio_checksums.parquet

Writes:
    lake/bronze/jsut/utterances (Delta table)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

from modules.data_engineering.common.ids import make_utterance_id, make_utterance_key
from modules.data_engineering.common.io import write_table
from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.schemas import BRONZE_UTTERANCES_SCHEMA
from modules.data_engineering.common.spark import get_spark

# JSUT constants
JSUT_SPEAKER_ID = "spk00"
JSUT_SPEAKER_NAME = "jsut"
JSUT_SUBSETS = [
    "basic5000",
    "counters128",
    "loanword128",
    "onomatopee300",
    "precedent130",
    "repeat500",
    "travel1000",
    "utparaphrase512",
    "voiceactress100",
]


def get_manifest() -> dict:
    """
    Load JSUT ingest manifest.

    Returns:
        Manifest dict with provenance info

    Raises:
        FileNotFoundError: If manifest not found (run ingest first)
    """
    manifest_path = paths.ingest_raw("jsut") / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"JSUT manifest not found: {manifest_path}\n"
            "Run ingest first: koe ingest jsut"
        )

    with open(manifest_path) as f:
        return json.load(f)


def read_transcripts() -> pd.DataFrame:
    """
    Read all JSUT transcripts into a pandas DataFrame.

    Returns:
        DataFrame with columns: subset, corpus_utt_id, text_raw, audio_relpath
    """
    extracted_dir = paths.ingest_extracted("jsut")
    jsut_root = extracted_dir / "jsut_ver1.1"

    if not jsut_root.exists():
        raise FileNotFoundError(
            f"JSUT root not found: {jsut_root}\n"
            "Run ingest first: koe ingest jsut"
        )

    records = []

    for subset in JSUT_SUBSETS:
        subset_dir = jsut_root / subset
        transcript_path = subset_dir / "transcript_utf8.txt"

        if not transcript_path.exists():
            print(f"  Warning: No transcript for subset {subset}")
            continue

        # Parse transcript file
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                # Format: BASIC5000_0001:text
                corpus_utt_id, text_raw = line.split(":", 1)
                corpus_utt_id = corpus_utt_id.strip()
                text_raw = text_raw.strip()

                if not text_raw:
                    continue

                # Build expected audio_relpath
                # Format: ingest/jsut/extracted/jsut_ver1.1/<subset>/wav/<corpus_utt_id>.wav
                audio_relpath = f"ingest/jsut/extracted/jsut_ver1.1/{subset}/wav/{corpus_utt_id}.wav"

                records.append({
                    "subset": subset,
                    "corpus_utt_id": corpus_utt_id,
                    "text_raw": text_raw,
                    "audio_relpath": audio_relpath,
                })

    if not records:
        raise ValueError("No transcripts found in JSUT extraction")

    return pd.DataFrame(records)


def read_audio_checksums() -> pd.DataFrame:
    """
    Read audio checksums parquet from ingest.

    Returns:
        DataFrame with audio metadata
    """
    checksums_path = paths.ingest_extracted("jsut") / "audio_checksums.parquet"

    if not checksums_path.exists():
        raise FileNotFoundError(
            f"Audio checksums not found: {checksums_path}\n"
            "Run ingest first: koe ingest jsut"
        )

    return pd.read_parquet(checksums_path)


def build_bronze_df(spark: SparkSession) -> DataFrame:
    """
    Build bronze DataFrame by joining transcripts with audio metadata.

    Args:
        spark: SparkSession

    Returns:
        Spark DataFrame with locked bronze schema
    """
    # Load manifest for provenance
    manifest = get_manifest()
    archive_info = manifest["archives"][0]

    print("Reading transcripts...")
    transcripts_pdf = read_transcripts()
    print(f"  Found {len(transcripts_pdf)} transcripts")

    print("Reading audio checksums...")
    audio_pdf = read_audio_checksums()
    print(f"  Found {len(audio_pdf)} audio files")

    # Join on audio_relpath
    print("Joining transcripts with audio metadata...")
    merged_pdf = transcripts_pdf.merge(
        audio_pdf,
        on="audio_relpath",
        how="left",
        indicator=True,
    )

    # Check for join failures
    missing_audio = merged_pdf[merged_pdf["_merge"] == "left_only"]
    if len(missing_audio) > 0:
        print(f"  ERROR: {len(missing_audio)} transcripts have no matching audio!")
        print("  First 5 missing:")
        for _, row in missing_audio.head().iterrows():
            print(f"    {row['corpus_utt_id']}: {row['audio_relpath']}")
        raise ValueError(f"Join failed: {len(missing_audio)} transcripts missing audio")

    merged_pdf = merged_pdf.drop(columns=["_merge"])
    print(f"  Joined: {len(merged_pdf)} records")

    # Generate IDs
    print("Generating utterance IDs...")
    merged_pdf["utterance_id"] = merged_pdf.apply(
        lambda r: make_utterance_id(
            dataset="jsut",
            speaker_id=JSUT_SPEAKER_ID,
            subset=r["subset"],
            corpus_utt_id=r["corpus_utt_id"],
        ),
        axis=1,
    )
    merged_pdf["utterance_key"] = merged_pdf.apply(
        lambda r: make_utterance_key(
            dataset="jsut",
            subset=r["subset"],
            corpus_utt_id=r["corpus_utt_id"],
        ),
        axis=1,
    )

    # Add fixed columns
    merged_pdf["dataset"] = "jsut"
    merged_pdf["speaker_id"] = JSUT_SPEAKER_ID
    merged_pdf["speaker_name"] = JSUT_SPEAKER_NAME
    # Use empty string for nullable string columns (Spark can't infer type from all-None)
    # We'll convert to null in Spark after
    merged_pdf["text_norm_raw"] = ""
    merged_pdf["phonemes_source"] = "none"
    merged_pdf["phonemes_raw"] = ""

    # Provenance from manifest
    merged_pdf["ingest_version"] = manifest["ingest_version"]
    merged_pdf["source_version"] = manifest["source_version"]
    merged_pdf["source_url"] = archive_info["source_url"]
    merged_pdf["source_archive_checksum"] = archive_info["source_archive_checksum"]

    # Meta overflow (corpus-specific stuff)
    merged_pdf["meta"] = merged_pdf.apply(
        lambda r: {
            "subset_dir": r["subset"],
            "transcript_file": f"{r['subset']}/transcript_utf8.txt",
            "archive": archive_info["filename"],
        },
        axis=1,
    )

    # Add ingested_at timestamp to pandas before Spark conversion
    merged_pdf["ingested_at"] = datetime.now(timezone.utc)

    # Create Spark DataFrame
    print("Creating Spark DataFrame...")
    spark_df = spark.createDataFrame(merged_pdf)

    # Convert empty strings to null for nullable columns
    spark_df = spark_df.withColumn(
        "text_norm_raw",
        F.when(F.col("text_norm_raw") == "", None).otherwise(F.col("text_norm_raw"))
    )
    spark_df = spark_df.withColumn(
        "phonemes_raw",
        F.when(F.col("phonemes_raw") == "", None).otherwise(F.col("phonemes_raw"))
    )

    # Cast meta to MapType
    spark_df = spark_df.withColumn(
        "meta",
        F.from_json(F.to_json(F.col("meta")), MapType(StringType(), StringType()))
    )

    # Select columns in schema order
    schema_columns = [f.name for f in BRONZE_UTTERANCES_SCHEMA.fields]
    spark_df = spark_df.select(schema_columns)

    return spark_df


def validate_bronze_df(df: DataFrame) -> dict:
    """
    Validate bronze DataFrame and compute stats.

    Args:
        df: Bronze DataFrame

    Returns:
        Dict with validation results and stats

    Raises:
        ValueError: If validation fails
    """
    print("\nValidating bronze DataFrame...")
    stats = {}

    # Total count
    total_count = df.count()
    stats["total_count"] = total_count
    print(f"  Total records: {total_count}")

    # Count per subset
    subset_counts = (
        df.groupBy("subset")
        .count()
        .orderBy("subset")
        .collect()
    )
    stats["subset_counts"] = {row["subset"]: row["count"] for row in subset_counts}
    print("  Per subset:")
    for row in subset_counts:
        print(f"    {row['subset']}: {row['count']}")

    # Check for duplicates on utterance_id
    dup_count = total_count - df.select("utterance_id").distinct().count()
    stats["duplicate_count"] = dup_count
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate utterance_ids!")
    print(f"  Duplicates: {dup_count}")

    # Check required columns for nulls
    required_cols = [
        "utterance_id", "utterance_key", "dataset", "speaker_id",
        "subset", "corpus_utt_id", "audio_relpath", "audio_format",
        "sample_rate", "channels", "duration_sec", "text_raw",
        "phonemes_source", "ingest_version", "ingested_at",
    ]
    null_counts = {}
    for col in required_cols:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            null_counts[col] = null_count

    stats["null_counts"] = null_counts
    if null_counts:
        print(f"  ERROR: Null values in required columns: {null_counts}")
        raise ValueError(f"Null values in required columns: {null_counts}")
    print(f"  Required column nulls: 0")

    # Duration stats
    duration_stats = df.agg(
        F.min("duration_sec").alias("min"),
        F.max("duration_sec").alias("max"),
        F.avg("duration_sec").alias("avg"),
        F.sum("duration_sec").alias("total"),
    ).collect()[0]

    stats["duration"] = {
        "min_sec": float(duration_stats["min"]),
        "max_sec": float(duration_stats["max"]),
        "avg_sec": float(duration_stats["avg"]),
        "total_hours": float(duration_stats["total"]) / 3600,
    }
    print(f"  Duration: min={duration_stats['min']:.2f}s, "
          f"max={duration_stats['max']:.2f}s, "
          f"avg={duration_stats['avg']:.2f}s, "
          f"total={duration_stats['total']/3600:.2f}h")

    # Sample rate check
    sample_rates = df.select("sample_rate").distinct().collect()
    stats["sample_rates"] = [row["sample_rate"] for row in sample_rates]
    print(f"  Sample rates: {stats['sample_rates']}")

    print("  Validation PASSED")
    return stats


def build_bronze_jsut(force: bool = False) -> dict:
    """
    Build JSUT bronze table.

    This is the main entry point for `koe bronze jsut`.

    Args:
        force: If True, overwrite existing table

    Returns:
        Dict with build results and stats
    """
    print("=" * 60)
    print("JSUT Bronze Pipeline")
    print("=" * 60)

    spark = get_spark()

    # Build DataFrame
    print("\n[1/3] Building bronze DataFrame...")
    bronze_df = build_bronze_df(spark)

    # Validate
    print("\n[2/3] Validating...")
    stats = validate_bronze_df(bronze_df)

    # Write to Delta
    print("\n[3/3] Writing Delta table...")
    output_path = paths.bronze / "jsut" / "utterances"
    print(f"  Output: {output_path}")

    write_table(
        bronze_df,
        layer="bronze",
        table_name="jsut/utterances",
        mode="overwrite",
        partition_by=["dataset"],
    )

    print("\n" + "=" * 60)
    print("JSUT bronze complete!")
    print(f"  Records: {stats['total_count']}")
    print(f"  Duration: {stats['duration']['total_hours']:.2f} hours")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return {
        "status": "success",
        "output_path": str(output_path),
        "stats": stats,
    }


if __name__ == "__main__":
    result = build_bronze_jsut()
    print(f"\nResult: {result['status']}")
