"""
Bronze layer: raw data ingestion with minimal transformation.

Tables:
- jsut/utterances: JSUT corpus utterances
- jvs/utterances: JVS corpus utterances
- common_voice/utterances: Common Voice utterances (TODO)

Usage:
    from modules.data_engineering.bronze import build_bronze_jsut, build_bronze_jvs
    build_bronze_jsut()
    build_bronze_jvs()

Or via CLI:
    koe bronze jsut
    koe bronze jvs
"""

from .jsut import build_bronze_jsut, build_bronze_df, validate_bronze_df
from .jvs import build_bronze_jvs

__all__ = [
    # JSUT
    "build_bronze_jsut",
    "build_bronze_df",
    "validate_bronze_df",
    # JVS
    "build_bronze_jvs",
]
