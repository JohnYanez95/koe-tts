"""
Silver layer: QC, normalization, splits, and phoneme processing.

Silver v1.0 provides:
- Quality control (is_trainable, exclude_reason, qc_version)
- Text normalization (passthrough v1, future: proper Japanese normalization)
- Deterministic split assignment (train/val/test)
- Phoneme promotion (JVS) or generation (JSUT - future)
- Labeling workflow hooks (label_status, label_batch_id, etc.)

Key insight: Splits are computed ONCE in Silver and persisted.
Gold consumes these splits rather than recomputing them.

Tables:
- jsut/utterances: JSUT corpus (v1.0 - no phonemes yet)
- jvs/utterances: JVS corpus (v1.0 - with corpus phonemes)
- common_voice/utterances: Common Voice (TODO)

Usage:
    from modules.data_engineering.silver import build_silver_jsut, build_silver_jvs
    build_silver_jsut()
    build_silver_jvs()

Or via CLI:
    koe silver jsut
    koe silver jvs --val-pct 0.15 --seed 123
"""

from .common import (
    DEFAULT_SEED,
    DEFAULT_TEST_PCT,
    DEFAULT_TRAIN_PCT,
    DEFAULT_VAL_PCT,
    apply_qc_rules,
    assign_splits,
    compute_silver_stats,
    compute_split,
    normalize_text,
    promote_corpus_phonemes,
)
from .jsut import build_silver_jsut
from .jvs import build_silver_jvs

__all__ = [
    # Shared utilities
    "DEFAULT_SEED",
    "DEFAULT_TEST_PCT",
    "DEFAULT_TRAIN_PCT",
    "DEFAULT_VAL_PCT",
    "apply_qc_rules",
    "assign_splits",
    "compute_silver_stats",
    "compute_split",
    "normalize_text",
    "promote_corpus_phonemes",
    # JSUT
    "build_silver_jsut",
    # JVS
    "build_silver_jvs",
]
