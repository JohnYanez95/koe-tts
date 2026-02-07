"""
Gold layer schema definitions.

Gold tables are "manifests + views" - the final training-ready data.
They are small, deterministic, and versioned.

Tables:
- train_manifest: Main training set with splits
- eval_sets: Fixed evaluation sets (edge cases, regressions)
- sampling_frames: Batches for labeler (stratified by rarity/confidence)
"""

from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Main training manifest
GOLD_TRAIN_MANIFEST_SCHEMA = StructType([
    # === Keys (from silver) ===
    StructField("utterance_id", StringType(), nullable=False),
    StructField("utterance_key", StringType(), nullable=False),

    # === Source identification ===
    StructField("dataset", StringType(), nullable=False),
    StructField("speaker_id", StringType(), nullable=False),

    # === Training data ===
    StructField("audio_relpath", StringType(), nullable=False),
    StructField("duration_sec", FloatType(), nullable=False),
    StructField("text", StringType(), nullable=False),
    StructField("phonemes", StringType(), nullable=False),
    StructField("n_phonemes", IntegerType(), nullable=False),

    # === Split assignment ===
    StructField("split", StringType(), nullable=False),         # train, val, test
    StructField("duration_bucket", StringType(), nullable=False),  # short, medium, long (for batching)

    # === Sampling (optional) ===
    StructField("sample_weight", FloatType(), nullable=True),   # For weighted sampling

    # === Versioning ===
    StructField("gold_version", StringType(), nullable=False),  # e.g., "v1", "20240115"
    StructField("silver_version", LongType(), nullable=True),   # Delta version of source silver
    StructField("created_at", TimestampType(), nullable=False),
])

# Fixed evaluation sets (for regression testing, edge cases)
GOLD_EVAL_SETS_SCHEMA = StructType([
    # === Eval set identification ===
    StructField("eval_set_id", StringType(), nullable=False),   # e.g., "edge_cases_v1"
    StructField("utterance_id", StringType(), nullable=False),

    # === Content ===
    StructField("text", StringType(), nullable=False),
    StructField("phonemes", StringType(), nullable=False),
    StructField("category", StringType(), nullable=False),      # long_sentence, rare_phoneme, etc.

    # === Reference data ===
    StructField("expected_duration_sec", FloatType(), nullable=True),
    StructField("reference_audio_relpath", StringType(), nullable=True),
    StructField("notes", StringType(), nullable=True),
])

# Sampling frames for labeler (stratified batches)
GOLD_SAMPLING_FRAMES_SCHEMA = StructType([
    # === Frame identification ===
    StructField("frame_id", StringType(), nullable=False),      # e.g., "low_confidence_batch_001"
    StructField("utterance_id", StringType(), nullable=False),

    # === Stratification info ===
    StructField("stratum", StringType(), nullable=False),       # phoneme_rarity, confidence_bin, etc.
    StructField("priority", IntegerType(), nullable=False),     # Lower = higher priority

    # === For labeler display ===
    StructField("audio_relpath", StringType(), nullable=False),
    StructField("text", StringType(), nullable=False),
    StructField("current_phonemes", StringType(), nullable=True),
    StructField("phoneme_confidence", FloatType(), nullable=True),

    # === Status ===
    StructField("labeled", StringType(), nullable=False),       # pending, in_progress, done
    StructField("created_at", TimestampType(), nullable=False),
])
