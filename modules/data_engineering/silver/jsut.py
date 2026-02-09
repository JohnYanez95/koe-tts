"""
JSUT corpus silver layer processing.

Silver v1.1 adds:
- Quality control (is_trainable, exclude_reason)
- Text normalization (passthrough for now)
- Deterministic split assignment
- Phoneme generation via pyopenjtalk (optional, --phonemize flag)

Command: koe silver jsut
         koe silver jsut --phonemize

Reads:
    lake/bronze/jsut/utterances (Delta table)

Writes:
    lake/silver/jsut/utterances (Delta table)
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from modules.data_engineering.common.io import write_table
from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.phonemes import export_inventory_json
from modules.data_engineering.common.schemas import SILVER_UTTERANCES_SCHEMA
from modules.forge.query.spark import get_spark

from .common import (
    DEFAULT_SEED,
    DEFAULT_TEST_PCT,
    DEFAULT_VAL_PCT,
    apply_qc_rules,
    assign_splits,
    compute_silver_stats,
    normalize_text,
)

# Silver pipeline version - bump when logic changes
SILVER_VERSION = "v1.1"

# Phoneme method identifier for pyopenjtalk
JSUT_PHONEME_METHOD = "pyopenjtalk_g2p_v0"


def read_bronze_jsut(spark: SparkSession) -> DataFrame:
    """
    Read JSUT bronze table.

    Returns:
        Bronze DataFrame

    Raises:
        FileNotFoundError: If bronze table doesn't exist
    """
    bronze_path = paths.bronze / "jsut" / "utterances"

    if not bronze_path.exists():
        raise FileNotFoundError(
            f"JSUT bronze table not found: {bronze_path}\n"
            "Run bronze first: koe bronze jsut"
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

    # Phoneme columns (may be set by generate_phonemes)
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
    df = df.withColumn("bronze_version", F.lit(None).cast("string"))
    df = df.withColumn("silver_version", F.lit(SILVER_VERSION))
    df = df.withColumn("processed_at", F.current_timestamp())

    return df


def generate_phonemes_batch(df: DataFrame) -> DataFrame:
    """
    Generate phonemes for all rows using pyopenjtalk.

    For ~7.6k rows, we process on the driver to avoid serialization overhead.

    Args:
        df: DataFrame with text_raw column

    Returns:
        DataFrame with phonemes, phonemes_method, phonemes_checked populated
    """
    from modules.data_engineering.common.phonemes import (
        generate_phonemes,
        normalize_phonemes,
    )

    # Collect to driver for processing (JSUT is small enough)
    print("  Collecting to driver for phoneme generation...")
    rows = df.select("utterance_id", "text_raw").collect()
    print(f"  Processing {len(rows)} utterances...")

    # Generate phonemes
    phoneme_map = {}
    for row in rows:
        uid = row["utterance_id"]
        text = row["text_raw"]
        if text:
            raw = generate_phonemes(text)
            normalized = normalize_phonemes(raw) if raw else None
            phoneme_map[uid] = normalized
        else:
            phoneme_map[uid] = None

    # Count successes
    success_count = sum(1 for v in phoneme_map.values() if v is not None)
    print(f"  Generated phonemes: {success_count}/{len(rows)}")

    # Create UDF to lookup phonemes
    phoneme_broadcast = df.sparkSession.sparkContext.broadcast(phoneme_map)

    @F.udf(StringType())
    def lookup_phonemes(uid: str) -> str:
        return phoneme_broadcast.value.get(uid)

    # Apply to DataFrame
    df = df.withColumn("phonemes", lookup_phonemes(F.col("utterance_id")))
    df = df.withColumn(
        "phonemes_method",
        F.when(F.col("phonemes").isNotNull(), F.lit(JSUT_PHONEME_METHOD))
        .otherwise(F.lit(None).cast("string"))
    )
    df = df.withColumn(
        "phonemes_checked",
        F.lit(False)  # Auto-generated, not human-verified
    )

    return df


def build_silver_df(
    spark: SparkSession,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
    phonemize: bool = False,
) -> DataFrame:
    """
    Build silver DataFrame from bronze with full processing.

    Args:
        spark: SparkSession
        val_pct: Validation set fraction
        test_pct: Test set fraction
        seed: Random seed for splits
        phonemize: Whether to generate phonemes with pyopenjtalk

    Returns:
        Silver DataFrame with all columns populated
    """
    print("Reading bronze table...")
    bronze_df = read_bronze_jsut(spark)
    bronze_count = bronze_df.count()
    print(f"  Bronze records: {bronze_count}")

    print("Adding silver column defaults...")
    df = add_silver_defaults(bronze_df)

    # Apply QC rules
    print("Applying QC rules...")
    df = apply_qc_rules(df, require_text=True, qc_version=SILVER_VERSION)

    # Normalize text (passthrough for now)
    print("Normalizing text...")
    df = normalize_text(df, method="passthrough")

    # Generate phonemes if requested
    if phonemize:
        print("Generating phonemes with pyopenjtalk...")
        df = generate_phonemes_batch(df)
    else:
        print("Phonemes: skipped (use --phonemize to generate)")

    # Assign deterministic splits
    print(f"Assigning splits (val={val_pct:.0%}, test={test_pct:.0%}, seed={seed})...")
    df = assign_splits(df, val_pct=val_pct, test_pct=test_pct, seed=seed)

    # Select columns in schema order
    schema_columns = [f.name for f in SILVER_UTTERANCES_SCHEMA.fields]
    df = df.select(schema_columns)

    return df


def validate_silver_df(df: DataFrame, expected_count: int, phonemize: bool = False) -> dict:
    """
    Validate silver DataFrame.

    Args:
        df: Silver DataFrame
        expected_count: Expected record count (from bronze)
        phonemize: Whether phonemes were generated

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
    print("  Count matches bronze: ✓")

    # Check required columns are non-null
    required_cols = [
        "utterance_id", "utterance_key", "dataset", "speaker_id",
        "subset", "audio_relpath", "text_raw", "phonemes_source",
    ]
    null_counts = {}
    for col in required_cols:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            null_counts[col] = null_count

    if null_counts:
        raise ValueError(f"Null values in required columns: {null_counts}")
    print("  Required columns non-null: ✓")

    # Print QC summary
    print("\n  QC Summary:")
    print(f"    Trainable: {stats['trainable_count']} ({stats['trainable_count']/actual_count*100:.1f}%)")
    print(f"    Excluded:  {stats['not_trainable_count']}")
    if stats['exclude_reasons']:
        print("    Exclusion reasons:")
        for reason, count in sorted(stats['exclude_reasons'].items()):
            print(f"      {reason}: {count}")

    # Print split distribution
    print("\n  Split Distribution:")
    for split_name in ["train", "val", "test"]:
        if split_name in stats['split_distribution']:
            count = stats['split_distribution'][split_name]
            pct = count / actual_count * 100
            print(f"    {split_name}: {count} ({pct:.1f}%)")

    # Phoneme stats
    print("\n  Phonemes:")
    if phonemize:
        print(f"    Has phonemes: {stats['has_phonemes']} ({stats['has_phonemes']/actual_count*100:.1f}%)")
        print(f"    Missing: {stats['missing_phonemes']}")
    else:
        print(f"    Has phonemes: {stats['has_phonemes']} (skipped, use --phonemize)")

    print("\n  Validation PASSED")
    return stats


def build_silver_jsut(
    force: bool = False,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
    phonemize: bool = False,
) -> dict:
    """
    Build JSUT silver table.

    This is the main entry point for `koe silver jsut`.

    Args:
        force: If True, overwrite existing table
        val_pct: Validation set fraction
        test_pct: Test set fraction
        seed: Random seed for splits
        phonemize: Whether to generate phonemes with pyopenjtalk

    Returns:
        Dict with build results and stats
    """
    print("=" * 60)
    print(f"JSUT Silver Pipeline ({SILVER_VERSION})")
    print("=" * 60)

    # Print config
    train_pct = 1.0 - val_pct - test_pct
    print("\nConfig:")
    print(f"  splits: train={train_pct:.0%} / val={val_pct:.0%} / test={test_pct:.0%}")
    print(f"  seed: {seed}")
    print(f"  phonemize: {phonemize}")

    spark = get_spark()

    # Read bronze to get expected count
    bronze_df = read_bronze_jsut(spark)
    expected_count = bronze_df.count()

    # Determine step count (4 if phonemize, else 3)
    total_steps = 4 if phonemize else 3

    # Build silver DataFrame
    print(f"\n[1/{total_steps}] Building silver DataFrame...")
    silver_df = build_silver_df(
        spark,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
        phonemize=phonemize,
    )

    # Validate
    print(f"\n[2/{total_steps}] Validating...")
    stats = validate_silver_df(silver_df, expected_count, phonemize=phonemize)

    # Write to Delta
    print(f"\n[3/{total_steps}] Writing Delta table...")
    output_path = paths.silver / "jsut" / "utterances"
    print(f"  Output: {output_path}")

    write_table(
        silver_df,
        layer="silver",
        table_name="jsut/utterances",
        mode="overwrite",
        partition_by=["dataset"],
        pipeline_version=SILVER_VERSION,
    )

    # Export phoneme inventory (only if phonemize was used)
    if phonemize:
        print(f"\n[4/{total_steps}] Exporting phoneme inventory...")
        inventory_path = paths.data_root / "data" / "assets" / "jsut" / "phoneme_inventory.json"
        export_inventory_json(
            silver_df,
            output_path=inventory_path,
            dataset="jsut",
            layer="silver",
            source_table=str(output_path),
            phonemes_method=JSUT_PHONEME_METHOD,
        )
        print(f"  Inventory: {inventory_path}")

    print("\n" + "=" * 60)
    print("JSUT Silver Complete")
    print("=" * 60)
    print(f"\n  Records: {stats['total_count']}")
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
    result = build_silver_jsut()
    print(f"\nResult: {result['status']}")
