"""
Common utilities for data engineering.

Provides shared helpers for paths, Spark, Delta I/O, and ID generation.
"""

from modules.forge.query.spark import get_spark, stop_spark

from .ids import make_audio_hash, make_speaker_id, make_utterance_id, parse_utterance_key
from .io import (
    Layer,
    WriteMode,
    get_table_history,
    get_table_version,
    optimize_table,
    read_table,
    table_exists,
    vacuum_table,
    write_table,
)
from .paths import LakePaths, TrainRunPaths, get_paths, paths

__all__ = [
    # Paths
    "paths",
    "get_paths",
    "LakePaths",
    "TrainRunPaths",
    # Spark
    "get_spark",
    "stop_spark",
    # I/O
    "read_table",
    "write_table",
    "table_exists",
    "get_table_history",
    "get_table_version",
    "vacuum_table",
    "optimize_table",
    "Layer",
    "WriteMode",
    # IDs
    "make_utterance_id",
    "make_speaker_id",
    "make_audio_hash",
    "parse_utterance_key",
]
