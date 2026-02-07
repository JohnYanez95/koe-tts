"""
JVS corpus silver layer processing.

Silver v1.0 adds:
- Quality control (is_trainable, exclude_reason)
- Text normalization (passthrough for now)
- Phoneme promotion (JVS lab files → canonical)
- Deterministic split assignment

Note: JVS has corpus-provided phonemes from HTS label files in bronze.
These are promoted to canonical phonemes in Silver.

Command: koe silver jvs

Reads:
    lake/bronze/jvs/utterances (Delta table)

Writes:
    lake/silver/jvs/utterances (Delta table)
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from modules.data_engineering.common.io import write_table
from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.phonemes import export_inventory_json
from modules.data_engineering.common.schemas import SILVER_UTTERANCES_SCHEMA
from modules.data_engineering.common.spark import get_spark

from .common import (
    DEFAULT_SEED,
    DEFAULT_TEST_PCT,
    DEFAULT_VAL_PCT,
    apply_qc_rules,
    assign_splits,
    compute_silver_stats,
    normalize_text,
    promote_corpus_phonemes,
)

# Silver pipeline version - bump when logic changes
SILVER_VERSION = "v1.1"

# JVS phoneme method identifier
# - openjtalk_hts: format from OpenJTalk HTS label files
# - trim_sil: boundary 'sil' markers stripped, internal 'pau' preserved
JVS_PHONEME_METHOD = "openjtalk_hts_trim_sil_v1"


def read_bronze_jvs(spark: SparkSession) -> DataFrame:
    """
    Read JVS bronze table.

    Returns:
        Bronze DataFrame

    Raises:
        FileNotFoundError: If bronze table doesn't exist
    """
    bronze_path = paths.bronze / "jvs" / "utterances"

    if not bronze_path.exists():
        raise FileNotFoundError(
            f"JVS bronze table not found: {bronze_path}\n"
            "Run bronze first: koe bronze jvs"
        )

    return spark.read.format("delta").load(str(bronze_path))


def add_silver_defaults(df: DataFrame) -> DataFrame:
    """
    Add silver-only columns with initial default values.

    These will be overwritten by the processing functions,
    but we need them to exist for schema conformance.
    """
    # QC columns (will be set by apply_qc_rules)
    df = df.withColumn("is_trainable", F.lit(True))
    df = df.withColumn("exclude_reason", F.lit(None).cast("string"))
    df = df.withColumn("qc_version", F.lit(None).cast("string"))
    df = df.withColumn("qc_checked_at", F.lit(None).cast("timestamp"))

    # Text normalization (will be set by normalize_text)
    df = df.withColumn("text_norm", F.lit(None).cast("string"))
    df = df.withColumn("text_norm_method", F.lit(None).cast("string"))

    # Phoneme columns (will be set by promote_corpus_phonemes)
    df = df.withColumn("phonemes", F.lit(None).cast("string"))
    df = df.withColumn("phonemes_method", F.lit(None).cast("string"))
    df = df.withColumn("phonemes_checked", F.lit(False))

    # Split (will be set by assign_splits)
    df = df.withColumn("split", F.lit(None).cast("string"))

    # Labeler columns (not used yet, but need defaults)
    df = df.withColumn("label_status", F.lit("unlabeled"))
    df = df.withColumn("label_batch_id", F.lit(None).cast("string"))
    df = df.withColumn("labeled_at", F.lit(None).cast("timestamp"))
    df = df.withColumn("labeled_by", F.lit(None).cast("string"))

    # Lineage
    df = df.withColumn("bronze_version", F.lit(None).cast("string"))  # TODO: get Delta version
    df = df.withColumn("silver_version", F.lit(SILVER_VERSION))
    df = df.withColumn("processed_at", F.current_timestamp())

    return df


def build_silver_df(
    spark: SparkSession,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
) -> DataFrame:
    """
    Build silver DataFrame from bronze with full processing.

    Args:
        spark: SparkSession
        val_pct: Validation set fraction
        test_pct: Test set fraction
        seed: Random seed for splits

    Returns:
        Silver DataFrame with all columns populated
    """
    print("Reading bronze table...")
    bronze_df = read_bronze_jvs(spark)
    bronze_count = bronze_df.count()
    print(f"  Bronze records: {bronze_count}")

    print("Adding silver column defaults...")
    df = add_silver_defaults(bronze_df)

    # Apply QC rules
    # Note: JVS may have missing transcripts for whisper10/falset10 subsets
    print("Applying QC rules...")
    df = apply_qc_rules(df, require_text=True, qc_version=SILVER_VERSION)

    # Normalize text (passthrough for now)
    print("Normalizing text...")
    df = normalize_text(df, method="passthrough")

    # Promote JVS corpus phonemes to canonical
    print("Promoting corpus phonemes...")
    df = promote_corpus_phonemes(df, phonemes_method=JVS_PHONEME_METHOD)

    # Assign deterministic splits
    print(f"Assigning splits (val={val_pct:.0%}, test={test_pct:.0%}, seed={seed})...")
    df = assign_splits(df, val_pct=val_pct, test_pct=test_pct, seed=seed)

    # Select columns in schema order
    schema_columns = [f.name for f in SILVER_UTTERANCES_SCHEMA.fields]
    df = df.select(schema_columns)

    return df


def validate_silver_df(df: DataFrame, expected_count: int) -> dict:
    """
    Validate silver DataFrame.

    Args:
        df: Silver DataFrame
        expected_count: Expected record count (from bronze)

    Returns:
        Dict with validation results

    Raises:
        ValueError: If validation fails
    """
    print("\nValidating silver DataFrame...")
    stats = compute_silver_stats(df)

    actual_count = stats["total_count"]
    print(f"  Record count: {actual_count}")

    if actual_count != expected_count:
        raise ValueError(f"Record count mismatch: expected {expected_count}, got {actual_count}")
    print(f"  Count matches bronze: ✓")

    # Check required columns are non-null
    required_cols = [
        "utterance_id", "utterance_key", "dataset", "speaker_id",
        "subset", "audio_relpath", "phonemes_source",
    ]
    null_counts = {}
    for col in required_cols:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            null_counts[col] = null_count

    if null_counts:
        raise ValueError(f"Null values in required columns: {null_counts}")
    print(f"  Required columns non-null: ✓")

    # Print QC summary
    print(f"\n  QC Summary:")
    print(f"    Trainable: {stats['trainable_count']} ({stats['trainable_count']/actual_count*100:.1f}%)")
    print(f"    Excluded:  {stats['not_trainable_count']}")
    if stats['exclude_reasons']:
        print(f"    Exclusion reasons:")
        for reason, count in sorted(stats['exclude_reasons'].items()):
            print(f"      {reason}: {count}")

    # Print split distribution
    print(f"\n  Split Distribution:")
    for split_name in ["train", "val", "test"]:
        if split_name in stats['split_distribution']:
            count = stats['split_distribution'][split_name]
            pct = count / actual_count * 100
            print(f"    {split_name}: {count} ({pct:.1f}%)")

    # Phoneme stats
    print(f"\n  Phonemes:")
    print(f"    Has phonemes: {stats['has_phonemes']} ({stats['has_phonemes']/actual_count*100:.1f}%)")
    print(f"    Missing: {stats['missing_phonemes']}")

    # Speaker count
    speaker_count = df.select("speaker_id").distinct().count()
    stats["speaker_count"] = speaker_count
    print(f"\n  Speakers: {speaker_count}")

    print("\n  Validation PASSED")
    return stats


def build_silver_jvs(
    force: bool = False,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Build JVS silver table (v1.0).

    This is the main entry point for `koe silver jvs`.

    Args:
        force: If True, overwrite existing table
        val_pct: Validation set fraction
        test_pct: Test set fraction
        seed: Random seed for splits

    Returns:
        Dict with build results and stats
    """
    print("=" * 60)
    print(f"JVS Silver Pipeline ({SILVER_VERSION})")
    print("=" * 60)

    # Print config
    train_pct = 1.0 - val_pct - test_pct
    print(f"\nConfig:")
    print(f"  splits: train={train_pct:.0%} / val={val_pct:.0%} / test={test_pct:.0%}")
    print(f"  seed: {seed}")

    spark = get_spark()

    # Read bronze to get expected count
    bronze_df = read_bronze_jvs(spark)
    expected_count = bronze_df.count()

    # Build silver DataFrame
    print("\n[1/4] Building silver DataFrame...")
    silver_df = build_silver_df(
        spark,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
    )

    # Validate
    print("\n[2/4] Validating...")
    stats = validate_silver_df(silver_df, expected_count)

    # Write to Delta
    print("\n[3/4] Writing Delta table...")
    output_path = paths.silver / "jvs" / "utterances"
    print(f"  Output: {output_path}")

    write_table(
        silver_df,
        layer="silver",
        table_name="jvs/utterances",
        mode="overwrite",
        partition_by=["dataset"],
        pipeline_version=SILVER_VERSION,
    )

    # Export phoneme inventory
    print("\n[4/4] Exporting phoneme inventory...")
    inventory_path = paths.data_root / "data" / "assets" / "jvs" / "phoneme_inventory.json"
    export_inventory_json(
        silver_df,
        output_path=inventory_path,
        dataset="jvs",
        layer="silver",
        source_table=str(output_path),
        phonemes_method=JVS_PHONEME_METHOD,
    )
    print(f"  Inventory: {inventory_path}")

    print("\n" + "=" * 60)
    print("JVS Silver Complete")
    print("=" * 60)
    print(f"\n  Records: {stats['total_count']}")
    print(f"  Speakers: {stats['speaker_count']}")
    print(f"  Trainable: {stats['trainable_count']}")
    print(f"  Has phonemes: {stats['has_phonemes']}")
    print(f"  Silver version: {SILVER_VERSION}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return {
        "status": "success",
        "output_path": str(output_path),
        "silver_version": SILVER_VERSION,
        "stats": stats,
    }


if __name__ == "__main__":
    result = build_silver_jvs()
    print(f"\nResult: {result['status']}")
