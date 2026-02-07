"""
Data Engineering module: ETL pipelines for the koe-tts lakehouse.

Structure:
    ingest/     - Download, extract, and create manifests for raw data
    bronze/     - Raw → Delta table (one row per utterance)
    silver/     - QC, normalization, phoneme harmonization
    gold/       - Splits, sampling, training manifests
    common/     - Shared utilities (paths, spark, io, ids, schemas)
    pipelines/  - Orchestrated end-to-end workflows

Quick Start:
    # Build JSUT dataset end-to-end
    python -m modules.data_engineering.pipelines.cli build-dataset --dataset jsut

    # Or step by step:
    python -m modules.data_engineering.ingest.cli jsut --extract
    python -m modules.data_engineering.bronze.cli build --dataset jsut
    python -m modules.data_engineering.silver.cli build
    python -m modules.data_engineering.gold.cli build --version v1
"""

from .common import (
    Layer,
    LakePaths,
    TrainRunPaths,
    WriteMode,
    get_paths,
    get_spark,
    get_table_history,
    get_table_version,
    make_audio_hash,
    make_speaker_id,
    make_utterance_id,
    optimize_table,
    parse_utterance_key,
    paths,
    read_table,
    stop_spark,
    table_exists,
    vacuum_table,
    write_table,
)
from .pipelines import build_dataset, refresh_gold

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
    # Pipelines
    "build_dataset",
    "refresh_gold",
]
