"""
Delta Lake I/O operations for the koe-tts lakehouse.

Provides utilities for reading/writing Delta tables with consistent
conventions across all modules.

Usage:
    from modules.data_engineering.common.io import read_table, write_table, table_exists

    # Read from silver layer
    df = read_table("silver", "utterances_clean")

    # Write to bronze layer
    write_table(df, "bronze", "utterances", mode="overwrite")

    # Upsert with merge
    write_table(new_df, "bronze", "utterances", mode="merge", merge_key="utterance_id")
"""

from pathlib import Path
from typing import Literal

from delta import DeltaTable
from pyspark.sql import DataFrame

from modules.forge.query.spark import get_spark

from .paths import paths

Layer = Literal["bronze", "silver", "gold"]
WriteMode = Literal["overwrite", "append", "merge"]


def _get_table_path(layer: Layer, table_name: str) -> Path:
    """Get the path to a Delta table."""
    if layer == "bronze":
        return paths.bronze_table(table_name)
    elif layer == "silver":
        return paths.silver_table(table_name)
    elif layer == "gold":
        return paths.gold_table(table_name)
    else:
        raise ValueError(f"Unknown layer: {layer}")


def table_exists(layer: Layer, table_name: str) -> bool:
    """Check if a Delta table exists."""
    table_path = _get_table_path(layer, table_name)
    return (table_path / "_delta_log").exists()


def read_table(
    layer: Layer,
    table_name: str,
    version: int | None = None,
    timestamp: str | None = None,
) -> DataFrame:
    """
    Read a Delta table from the lakehouse.

    Args:
        layer: Lake layer (bronze, silver, gold)
        table_name: Name of the table
        version: Optional version number for time travel
        timestamp: Optional timestamp for time travel (e.g., "2026-01-24")

    Returns:
        Spark DataFrame

    Raises:
        FileNotFoundError: If table doesn't exist
    """
    spark = get_spark()
    table_path = _get_table_path(layer, table_name)

    if not table_exists(layer, table_name):
        raise FileNotFoundError(f"Table not found: {layer}/{table_name}")

    reader = spark.read.format("delta")

    if version is not None:
        reader = reader.option("versionAsOf", version)
    elif timestamp is not None:
        reader = reader.option("timestampAsOf", timestamp)

    return reader.load(str(table_path))


def write_table(
    df: DataFrame,
    layer: Layer,
    table_name: str,
    mode: WriteMode = "overwrite",
    partition_by: list[str] | None = None,
    merge_key: str | list[str] | None = None,
    register_catalog: bool = True,
    pipeline_version: str | None = None,
) -> None:
    """
    Write a DataFrame to a Delta table.

    Args:
        df: Spark DataFrame to write
        layer: Lake layer (bronze, silver, gold)
        table_name: Name of the table
        mode: Write mode (overwrite, append, merge)
        partition_by: Columns to partition by
        merge_key: Column(s) for merge mode (upsert)
        register_catalog: Auto-register in catalog (default True)
        pipeline_version: Optional version tag for catalog

    Raises:
        ValueError: If merge mode used without merge_key
    """
    table_path = _get_table_path(layer, table_name)

    if mode == "merge":
        if merge_key is None:
            raise ValueError("merge_key required for merge mode")
        _merge_table(df, table_path, merge_key)
    else:
        writer = df.write.format("delta").mode(mode)

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        writer.save(str(table_path))

    # Auto-register in catalog
    if register_catalog:
        _register_in_catalog(
            layer=layer,
            table_name=table_name,
            table_path=table_path,
            schema=df.schema,
            record_count=df.count(),
            pipeline_version=pipeline_version,
        )


def _register_in_catalog(
    layer: str,
    table_name: str,
    table_path: Path,
    schema,
    record_count: int,
    pipeline_version: str | None = None,
) -> None:
    """Register table in catalog (best-effort, silent fail)."""
    try:
        from .catalog import register_table

        # Parse dataset from table_name (e.g., "jsut/utterances" -> "jsut")
        parts = table_name.split("/")
        dataset = parts[0] if len(parts) > 1 else "default"
        tbl_name = parts[-1]

        register_table(
            layer=layer,
            dataset=dataset,
            table_name=tbl_name,
            delta_path=table_path,
            schema=schema,
            record_count=record_count,
            pipeline_version=pipeline_version,
        )
    except Exception:
        # Silent fail - catalog registration is best-effort
        pass


def _merge_table(
    df: DataFrame,
    table_path: Path,
    merge_key: str | list[str],
) -> None:
    """Perform a merge (upsert) operation."""
    spark = get_spark()

    if isinstance(merge_key, str):
        merge_key = [merge_key]

    # Build merge condition
    condition = " AND ".join([f"target.{k} = source.{k}" for k in merge_key])

    if (table_path / "_delta_log").exists():
        # Table exists - merge
        delta_table = DeltaTable.forPath(spark, str(table_path))
        (
            delta_table.alias("target")
            .merge(df.alias("source"), condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        # Table doesn't exist - create it
        df.write.format("delta").save(str(table_path))


def get_table_history(layer: Layer, table_name: str, limit: int = 10) -> DataFrame:
    """Get the history of a Delta table."""
    spark = get_spark()
    table_path = _get_table_path(layer, table_name)

    if not table_exists(layer, table_name):
        raise FileNotFoundError(f"Table not found: {layer}/{table_name}")

    delta_table = DeltaTable.forPath(spark, str(table_path))
    return delta_table.history(limit)


def get_table_version(layer: Layer, table_name: str) -> int:
    """Get the current version of a Delta table."""
    spark = get_spark()
    table_path = _get_table_path(layer, table_name)

    if not table_exists(layer, table_name):
        raise FileNotFoundError(f"Table not found: {layer}/{table_name}")

    delta_table = DeltaTable.forPath(spark, str(table_path))
    return delta_table.history(1).collect()[0]["version"]


def vacuum_table(
    layer: Layer,
    table_name: str,
    retention_hours: int = 168,  # 7 days
) -> None:
    """
    Vacuum a Delta table to remove old files.

    Args:
        layer: Lake layer
        table_name: Name of the table
        retention_hours: Files older than this will be deleted
    """
    spark = get_spark()
    table_path = _get_table_path(layer, table_name)

    if not table_exists(layer, table_name):
        raise FileNotFoundError(f"Table not found: {layer}/{table_name}")

    delta_table = DeltaTable.forPath(spark, str(table_path))
    delta_table.vacuum(retention_hours)


def optimize_table(
    layer: Layer,
    table_name: str,
    z_order_by: list[str] | None = None,
) -> None:
    """
    Optimize a Delta table (compaction + optional Z-ordering).

    Args:
        layer: Lake layer
        table_name: Name of the table
        z_order_by: Columns to Z-order by for query optimization
    """
    spark = get_spark()
    table_path = _get_table_path(layer, table_name)

    if not table_exists(layer, table_name):
        raise FileNotFoundError(f"Table not found: {layer}/{table_name}")

    delta_table = DeltaTable.forPath(spark, str(table_path))

    if z_order_by:
        delta_table.optimize().executeZOrderBy(*z_order_by)
    else:
        delta_table.optimize().executeCompaction()
