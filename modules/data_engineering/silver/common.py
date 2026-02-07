"""
Shared Silver layer processing utilities.

These functions provide consistent QC, normalization, and split logic
across all datasets in the Silver layer.

The key insight: splits are computed ONCE in Silver and persisted.
Gold consumes these splits rather than recomputing them.

Usage:
    from modules.data_engineering.silver.common import (
        apply_qc_rules,
        assign_splits,
        normalize_text,
    )
"""

import hashlib
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# =============================================================================
# Default Split Configuration
# =============================================================================
# These defaults can be overridden at the CLI level, but provide sensible
# starting points for all datasets.

DEFAULT_TRAIN_PCT = 0.90
DEFAULT_VAL_PCT = 0.10
DEFAULT_TEST_PCT = 0.00  # No test set by default (add later when needed)
DEFAULT_SEED = 42

# =============================================================================
# QC Configuration
# =============================================================================
# Minimum and maximum duration for trainable utterances
QC_MIN_DURATION_SEC = 0.1   # Very short utterances are often noise/artifacts
QC_MAX_DURATION_SEC = 30.0  # Very long utterances may have issues


def compute_split(
    utterance_id: str,
    seed: int,
    val_pct: float,
    test_pct: float,
) -> str:
    """
    Compute deterministic split assignment for a single utterance.

    Uses hash of (utterance_id + seed) for reproducibility.
    Same inputs always produce same split.

    Args:
        utterance_id: Unique utterance identifier
        seed: Random seed for hash mixing
        val_pct: Fraction for validation set (0.0-1.0)
        test_pct: Fraction for test set (0.0-1.0)

    Returns:
        Split name: "train", "val", or "test"
    """
    # Mix seed into hash for reproducibility
    hash_input = f"{utterance_id}_{seed}"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    p = (hash_val % 10000) / 10000.0

    if p < val_pct:
        return "val"
    elif p < val_pct + test_pct:
        return "test"
    else:
        return "train"


def assign_splits(
    df: DataFrame,
    val_pct: float = DEFAULT_VAL_PCT,
    test_pct: float = DEFAULT_TEST_PCT,
    seed: int = DEFAULT_SEED,
) -> DataFrame:
    """
    Assign deterministic train/val/test splits based on hash of utterance_id.

    This ensures reruns produce the same splits. Splits are computed once
    in Silver and persisted, so Gold and training consume consistent splits.

    Args:
        df: DataFrame with utterance_id column
        val_pct: Fraction for validation set (default 0.10)
        test_pct: Fraction for test set (default 0.00)
        seed: Random seed for hash mixing (default 42)

    Returns:
        DataFrame with split column set
    """
    # Create UDF from the pure function
    @F.udf(StringType())
    def split_udf(utterance_id: str) -> str:
        return compute_split(utterance_id, seed, val_pct, test_pct)

    return df.withColumn("split", split_udf(F.col("utterance_id")))


def apply_qc_rules(
    df: DataFrame,
    min_duration: float = QC_MIN_DURATION_SEC,
    max_duration: float = QC_MAX_DURATION_SEC,
    require_text: bool = True,
    qc_version: str = "v1.0",
) -> DataFrame:
    """
    Apply quality control rules to determine trainability.

    Sets is_trainable and exclude_reason columns based on:
    - Duration bounds (too short or too long)
    - Text availability (if require_text=True)

    Args:
        df: DataFrame with duration_sec and text_raw columns
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        require_text: Whether to require non-null text_raw
        qc_version: Version string for tracking QC rule changes

    Returns:
        DataFrame with is_trainable, exclude_reason, qc_version, qc_checked_at set
    """
    # Build exclusion conditions and reasons
    # Priority order: first matching reason wins
    conditions = []

    # Duration too short
    conditions.append((
        F.col("duration_sec") < min_duration,
        F.lit(f"duration_below_{min_duration}s")
    ))

    # Duration too long
    conditions.append((
        F.col("duration_sec") > max_duration,
        F.lit(f"duration_above_{max_duration}s")
    ))

    # Missing transcript
    if require_text:
        conditions.append((
            F.col("text_raw").isNull(),
            F.lit("missing_transcript")
        ))

    # Build chained WHEN expression for exclude_reason
    exclude_expr = F.lit(None).cast("string")
    for condition, reason in reversed(conditions):
        exclude_expr = F.when(condition, reason).otherwise(exclude_expr)

    # is_trainable = no exclusion reason
    df = df.withColumn("exclude_reason", exclude_expr)
    df = df.withColumn("is_trainable", F.col("exclude_reason").isNull())

    # Add QC metadata
    df = df.withColumn("qc_version", F.lit(qc_version))
    df = df.withColumn("qc_checked_at", F.current_timestamp())

    return df


def normalize_text(
    df: DataFrame,
    method: str = "passthrough",
) -> DataFrame:
    """
    Normalize text for training.

    Currently implements a passthrough (identity) normalization.
    Future: Add proper Japanese text normalization (numbers, symbols, etc.)

    Args:
        df: DataFrame with text_raw column
        method: Normalization method (currently only "passthrough")

    Returns:
        DataFrame with text_norm and text_norm_method set
    """
    if method == "passthrough":
        # Identity: copy text_raw to text_norm
        df = df.withColumn("text_norm", F.col("text_raw"))
        df = df.withColumn("text_norm_method", F.lit("passthrough_v1"))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df


def promote_corpus_phonemes(
    df: DataFrame,
    phonemes_method: str,
    normalize: bool = True,
) -> DataFrame:
    """
    Promote corpus-provided phonemes (phonemes_raw) to canonical phonemes.

    For corpora like JVS that provide their own phoneme alignments,
    this normalizes phonemes_raw → phonemes and marks them as checked.

    Normalization: strips boundary 'sil' markers, keeps internal 'pau'.

    Args:
        df: DataFrame with phonemes_raw and phonemes_source columns
        phonemes_method: Method string to record (e.g., "openjtalk_hts_trim_sil_v1")
        normalize: Whether to apply normalization (default True)

    Returns:
        DataFrame with phonemes, phonemes_method, phonemes_checked set
    """
    from modules.data_engineering.common.phonemes import normalize_phonemes

    # Create normalization UDF
    from pyspark.sql.types import StringType

    @F.udf(StringType())
    def normalize_udf(phonemes: str) -> str:
        if normalize:
            return normalize_phonemes(phonemes)
        return phonemes

    # Apply normalization and promote
    df = df.withColumn(
        "phonemes",
        F.when(
            F.col("phonemes_raw").isNotNull(),
            normalize_udf(F.col("phonemes_raw"))
        ).otherwise(F.col("phonemes"))  # Keep existing if any
    )

    df = df.withColumn(
        "phonemes_method",
        F.when(
            F.col("phonemes_raw").isNotNull(),
            F.lit(phonemes_method)
        ).otherwise(F.col("phonemes_method"))
    )

    df = df.withColumn(
        "phonemes_checked",
        F.when(
            F.col("phonemes_raw").isNotNull(),
            F.lit(True)
        ).otherwise(F.col("phonemes_checked"))
    )

    return df


def compute_silver_stats(df: DataFrame) -> dict:
    """
    Compute summary statistics for a Silver DataFrame.

    Returns:
        Dict with counts and distributions
    """
    total = df.count()

    # Trainability stats
    trainable = df.filter(F.col("is_trainable") == True).count()
    not_trainable = total - trainable

    # Exclusion reasons
    exclude_reasons = (
        df.filter(F.col("exclude_reason").isNotNull())
        .groupBy("exclude_reason")
        .count()
        .collect()
    )
    exclude_dist = {row["exclude_reason"]: row["count"] for row in exclude_reasons}

    # Split distribution
    split_counts = df.groupBy("split").count().collect()
    split_dist = {
        str(row["split"]): row["count"]
        for row in split_counts
    }

    # Phonemes stats
    has_phonemes = df.filter(F.col("phonemes").isNotNull()).count()

    return {
        "total_count": total,
        "trainable_count": trainable,
        "not_trainable_count": not_trainable,
        "exclude_reasons": exclude_dist,
        "split_distribution": split_dist,
        "has_phonemes": has_phonemes,
        "missing_phonemes": total - has_phonemes,
    }
