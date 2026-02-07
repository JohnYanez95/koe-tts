"""
Schema definitions for lakehouse tables.

Schemas are defined as PySpark StructType for type safety
and documentation.

Usage:
    from modules.data_engineering.common.schemas import (
        BRONZE_UTTERANCES_SCHEMA,
        SILVER_UTTERANCES_SCHEMA,
        GOLD_TRAIN_MANIFEST_SCHEMA,
        PHONEME_SOURCES,
    )
"""

from .bronze import BRONZE_UTTERANCES_SCHEMA, PHONEME_SOURCES
from .gold import (
    GOLD_EVAL_SETS_SCHEMA,
    GOLD_SAMPLING_FRAMES_SCHEMA,
    GOLD_TRAIN_MANIFEST_SCHEMA,
)
from .silver import SILVER_STUB_DEFAULTS, SILVER_UTTERANCES_SCHEMA

__all__ = [
    # Bronze
    "BRONZE_UTTERANCES_SCHEMA",
    "PHONEME_SOURCES",
    # Silver
    "SILVER_UTTERANCES_SCHEMA",
    "SILVER_STUB_DEFAULTS",
    # Gold
    "GOLD_TRAIN_MANIFEST_SCHEMA",
    "GOLD_EVAL_SETS_SCHEMA",
    "GOLD_SAMPLING_FRAMES_SCHEMA",
]
