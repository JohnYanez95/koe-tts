"""
JVS corpus bronze layer ingestion.

Reads from ingest outputs and writes to bronze Delta table.

JVS structure:
- 100 speakers (jvs001-jvs100)
- Per speaker: parallel100, nonpara30, whisper10, falset10
- .lab files contain phoneme alignments (HTS format)
- Transcripts stored per subset

Command: koe bronze jvs

Reads:
    data/ingest/jvs/raw/MANIFEST.json
    data/ingest/jvs/extracted/jvs_ver1/{speaker}/{subset}/wav24kHz16bit/*.wav
    data/ingest/jvs/extracted/jvs_ver1/{speaker}/{subset}/lab/mon/*.lab
    data/ingest/jvs/extracted/audio_checksums.parquet

Writes:
    lake/bronze/jvs/utterances (Delta table)
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

from modules.data_engineering.common.ids import make_utterance_id, make_utterance_key
from modules.data_engineering.common.io import write_table
from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.schemas import BRONZE_UTTERANCES_SCHEMA
from modules.forge.query.spark import get_spark

# JVS constants
JVS_SUBSETS = ["parallel100", "nonpara30", "whisper10", "falset10"]
JVS_SPEAKER_COUNT = 100


def get_manifest() -> dict:
    """
    Load JVS ingest manifest.

    Returns:
        Manifest dict with provenance info

    Raises:
        FileNotFoundError: If manifest not found (run ingest first)
    """
    manifest_path = paths.ingest_raw("jvs") / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"JVS manifest not found: {manifest_path}\n"
            "Run ingest first: koe ingest jvs"
        )

    with open(manifest_path) as f:
        return json.load(f)


def get_jvs_root() -> Path:
    """Get the JVS extracted root directory."""
    extracted_dir = paths.ingest_extracted("jvs")
    jvs_root = extracted_dir / "jvs_ver1"

    if not jvs_root.exists():
        raise FileNotFoundError(
            f"JVS root not found: {jvs_root}\n"
            "Run ingest first: koe ingest jvs"
        )

    return jvs_root


def parse_lab_file(lab_path: Path) -> str | None:
    """
    Parse HTS label file to extract phoneme string.

    HTS format:
        start_time end_time phoneme

    Example:
        0 50000 pau
        50000 250000 k
        250000 450000 o
        ...

    Args:
        lab_path: Path to .lab file

    Returns:
        Space-separated phoneme string, or None if parsing fails
    """
    try:
        phonemes = []
        with open(lab_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    # Format: start end phoneme
                    phoneme = parts[2]
                    phonemes.append(phoneme)
        return " ".join(phonemes) if phonemes else None
    except Exception:
        return None


def read_transcript_file(transcript_path: Path) -> dict[str, str]:
    """
    Read a transcript file in JSUT format.

    Format: UTTERANCE_ID:text

    Args:
        transcript_path: Path to transcript file

    Returns:
        Dict mapping corpus_utt_id to text
    """
    transcripts = {}
    if not transcript_path.exists():
        return transcripts

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            corpus_utt_id, text = line.split(":", 1)
            transcripts[corpus_utt_id.strip()] = text.strip()

    return transcripts


def discover_utterances(jvs_root: Path) -> pd.DataFrame:
    """
    Discover all utterances by walking the directory structure.

    Returns:
        DataFrame with columns: speaker_id, subset, corpus_utt_id, audio_relpath, lab_relpath
    """
    records = []
    data_root = paths.data_root

    for speaker_idx in range(1, JVS_SPEAKER_COUNT + 1):
        speaker_id = f"jvs{speaker_idx:03d}"
        speaker_dir = jvs_root / speaker_id

        if not speaker_dir.exists():
            continue

        for subset in JVS_SUBSETS:
            subset_dir = speaker_dir / subset
            wav_dir = subset_dir / "wav24kHz16bit"
            lab_dir = subset_dir / "lab" / "mon"

            if not wav_dir.exists():
                continue

            for wav_file in sorted(wav_dir.glob("*.wav")):
                corpus_utt_id = wav_file.stem

                # Compute relative paths
                audio_relpath = str(wav_file.relative_to(data_root / "data"))

                # Look for corresponding .lab file
                lab_path = lab_dir / f"{corpus_utt_id}.lab"
                lab_relpath = None
                if lab_path.exists():
                    lab_relpath = str(lab_path.relative_to(data_root / "data"))

                records.append({
                    "speaker_id": speaker_id,
                    "subset": subset,
                    "corpus_utt_id": corpus_utt_id,
                    "audio_relpath": audio_relpath,
                    "lab_relpath": lab_relpath,
                })

    return pd.DataFrame(records)


def load_transcripts(jvs_root: Path) -> dict[str, str]:
    """
    Load all transcripts from JVS.

    JVS uses shared transcript files:
    - parallel100/whisper10/falset10: Use VOICEACTRESS100 texts
    - nonpara30: Speaker-specific texts in nonpara30/transcripts_utf8.txt

    Args:
        jvs_root: Path to jvs_ver1 directory

    Returns:
        Dict mapping corpus_utt_id to text
    """
    all_transcripts = {}

    # Load VOICEACTRESS100 texts (shared across parallel100, whisper10, falset10)
    # These are the same texts used in JSUT voiceactress100
    # Try to find them in a common location or first speaker's parallel100
    for speaker_idx in range(1, JVS_SPEAKER_COUNT + 1):
        speaker_id = f"jvs{speaker_idx:03d}"
        transcript_path = jvs_root / speaker_id / "parallel100" / "transcripts_utf8.txt"
        if transcript_path.exists():
            all_transcripts.update(read_transcript_file(transcript_path))
            break  # VOICEACTRESS100 texts are same for all speakers

    # Load nonpara30 transcripts (speaker-specific)
    for speaker_idx in range(1, JVS_SPEAKER_COUNT + 1):
        speaker_id = f"jvs{speaker_idx:03d}"
        transcript_path = jvs_root / speaker_id / "nonpara30" / "transcripts_utf8.txt"
        if transcript_path.exists():
            speaker_transcripts = read_transcript_file(transcript_path)
            # Prefix with speaker_id to avoid collisions
            for utt_id, text in speaker_transcripts.items():
                all_transcripts[f"{speaker_id}_{utt_id}"] = text

    return all_transcripts


def read_audio_checksums() -> pd.DataFrame:
    """
    Read audio checksums parquet from ingest.

    Returns:
        DataFrame with audio metadata
    """
    checksums_path = paths.ingest_extracted("jvs") / "audio_checksums.parquet"

    if not checksums_path.exists():
        raise FileNotFoundError(
            f"Audio checksums not found: {checksums_path}\n"
            "Run ingest first: koe ingest jvs"
        )

    return pd.read_parquet(checksums_path)


def build_bronze_df(spark: SparkSession) -> DataFrame:
    """
    Build bronze DataFrame by joining discovered files with audio metadata.

    Args:
        spark: SparkSession

    Returns:
        Spark DataFrame with locked bronze schema
    """
    # Load manifest for provenance
    manifest = get_manifest()
    archive_info = manifest["archives"][0]

    jvs_root = get_jvs_root()

    # Discover all utterances from directory structure
    print("Discovering utterances from directory structure...")
    utterances_pdf = discover_utterances(jvs_root)
    print(f"  Found {len(utterances_pdf)} audio files")

    # Load transcripts
    print("Loading transcripts...")
    transcripts = load_transcripts(jvs_root)
    print(f"  Found {len(transcripts)} transcript entries")

    # Load audio checksums
    print("Reading audio checksums...")
    audio_pdf = read_audio_checksums()
    print(f"  Found {len(audio_pdf)} audio checksums")

    # Join with audio metadata
    print("Joining with audio metadata...")
    merged_pdf = utterances_pdf.merge(
        audio_pdf,
        on="audio_relpath",
        how="left",
        indicator=True,
    )

    # Check for join failures
    missing_audio = merged_pdf[merged_pdf["_merge"] == "left_only"]
    if len(missing_audio) > 0:
        print(f"  Warning: {len(missing_audio)} utterances missing audio metadata")

    merged_pdf = merged_pdf[merged_pdf["_merge"] == "both"].drop(columns=["_merge"])
    print(f"  Joined: {len(merged_pdf)} records")

    # Add transcripts
    print("Adding transcripts...")
    def get_transcript(row):
        # Try direct lookup first
        text = transcripts.get(row["corpus_utt_id"])
        if text:
            return text

        # Try speaker-prefixed lookup for nonpara30
        if row["subset"] == "nonpara30":
            prefixed_id = f"{row['speaker_id']}_{row['corpus_utt_id']}"
            text = transcripts.get(prefixed_id)
            if text:
                return text

        # For whisper10/falset10, try to map to VOICEACTRESS100 ID
        # These use same texts but may have different ID format
        return None

    merged_pdf["text_raw"] = merged_pdf.apply(get_transcript, axis=1)

    # Count transcripts found
    has_transcript = merged_pdf["text_raw"].notna().sum()
    print(f"  Transcripts found: {has_transcript} / {len(merged_pdf)}")

    # Parse lab files for phonemes
    print("Parsing lab files for phonemes...")
    def get_phonemes(row):
        if row["lab_relpath"] is None:
            return None
        lab_path = paths.data_root / "data" / row["lab_relpath"]
        return parse_lab_file(lab_path)

    merged_pdf["phonemes_raw"] = merged_pdf.apply(get_phonemes, axis=1)

    has_phonemes = merged_pdf["phonemes_raw"].notna().sum()
    print(f"  Phonemes found: {has_phonemes} / {len(merged_pdf)}")

    # Generate IDs
    print("Generating utterance IDs...")
    merged_pdf["utterance_id"] = merged_pdf.apply(
        lambda r: make_utterance_id(
            dataset="jvs",
            speaker_id=r["speaker_id"],
            subset=r["subset"],
            corpus_utt_id=r["corpus_utt_id"],
        ),
        axis=1,
    )
    merged_pdf["utterance_key"] = merged_pdf.apply(
        lambda r: make_utterance_key(
            dataset="jvs",
            subset=f"{r['speaker_id']}_{r['subset']}",
            corpus_utt_id=r["corpus_utt_id"],
        ),
        axis=1,
    )

    # Add fixed columns
    merged_pdf["dataset"] = "jvs"
    # Normalize speaker_id to spkNN format
    merged_pdf["speaker_name"] = merged_pdf["speaker_id"]  # Keep original as name
    merged_pdf["speaker_id"] = merged_pdf["speaker_id"].apply(
        lambda x: f"spk{int(x[3:]):02d}" if x.startswith("jvs") else x
    )

    # Empty string for nullable columns (will convert to null in Spark)
    merged_pdf["text_norm_raw"] = ""

    # Phonemes source - JVS has automatically generated alignments
    merged_pdf["phonemes_source"] = merged_pdf["phonemes_raw"].apply(
        lambda x: "corpus_provided" if x else "none"
    )
    # Fill None phonemes with empty string for Spark type inference
    merged_pdf["phonemes_raw"] = merged_pdf["phonemes_raw"].fillna("")

    # Fill None text_raw with empty string (will filter out in silver)
    merged_pdf["text_raw"] = merged_pdf["text_raw"].fillna("")

    # Provenance from manifest
    merged_pdf["ingest_version"] = manifest["ingest_version"]
    merged_pdf["source_version"] = manifest["source_version"]
    merged_pdf["source_url"] = archive_info["source_url"]
    merged_pdf["source_archive_checksum"] = archive_info["source_archive_checksum"]

    # Meta overflow (corpus-specific stuff)
    merged_pdf["meta"] = merged_pdf.apply(
        lambda r: {
            "speaker_name": r["speaker_name"],
            "subset": r["subset"],
            "lab_relpath": r.get("lab_relpath") or "",
            "archive": archive_info["filename"],
        },
        axis=1,
    )

    # Rename speaker_name column to match schema expectation
    # (speaker_name in meta is the original jvsXXX, speaker_name in schema is optional friendly name)

    # Add ingested_at timestamp
    merged_pdf["ingested_at"] = datetime.now(UTC)

    # Create Spark DataFrame
    print("Creating Spark DataFrame...")
    spark_df = spark.createDataFrame(merged_pdf)

    # Convert empty strings to null for nullable columns
    spark_df = spark_df.withColumn(
        "text_raw",
        F.when(F.col("text_raw") == "", None).otherwise(F.col("text_raw"))
    )
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

    # Count per speaker (sample)
    speaker_count = df.select("speaker_id").distinct().count()
    stats["speaker_count"] = speaker_count
    print(f"  Unique speakers: {speaker_count}")

    # Check for duplicates on utterance_id
    dup_count = total_count - df.select("utterance_id").distinct().count()
    stats["duplicate_count"] = dup_count
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate utterance_ids!")
    print(f"  Duplicates: {dup_count}")

    # Check null counts for key columns
    # Note: JVS may have missing transcripts for some subsets
    null_text = df.filter(F.col("text_raw").isNull()).count()
    stats["null_text_raw"] = null_text
    print(f"  Missing transcripts (text_raw is null): {null_text}")

    # Phonemes availability
    has_phonemes = df.filter(F.col("phonemes_raw").isNotNull()).count()
    stats["has_phonemes"] = has_phonemes
    print(f"  Has phonemes: {has_phonemes}")

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


def build_bronze_jvs(force: bool = False) -> dict:
    """
    Build JVS bronze table.

    This is the main entry point for `koe bronze jvs`.

    Args:
        force: If True, overwrite existing table

    Returns:
        Dict with build results and stats
    """
    print("=" * 60)
    print("JVS Bronze Pipeline")
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
    output_path = paths.bronze / "jvs" / "utterances"
    print(f"  Output: {output_path}")

    write_table(
        bronze_df,
        layer="bronze",
        table_name="jvs/utterances",
        mode="overwrite",
        partition_by=["dataset"],
    )

    print("\n" + "=" * 60)
    print("JVS bronze complete!")
    print(f"  Records: {stats['total_count']}")
    print(f"  Speakers: {stats['speaker_count']}")
    print(f"  Duration: {stats['duration']['total_hours']:.2f} hours")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return {
        "status": "success",
        "output_path": str(output_path),
        "stats": stats,
    }


if __name__ == "__main__":
    result = build_bronze_jvs()
    print(f"\nResult: {result['status']}")
