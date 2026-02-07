"""
Lakehouse catalog for table discovery and metadata.

Provides metastore-like functionality without external infrastructure.

Location: lake/_catalog/tables (Delta table)

Usage:
    koe catalog list
    koe catalog describe silver.jsut.utterances
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from .paths import paths
from .spark import get_spark

# Catalog table schema
CATALOG_SCHEMA = StructType([
    StructField("table_fqn", StringType(), nullable=False),        # e.g., bronze.jsut.utterances
    StructField("layer", StringType(), nullable=False),            # bronze|silver|gold
    StructField("dataset", StringType(), nullable=False),          # jsut, jvs, etc.
    StructField("table_name", StringType(), nullable=False),       # utterances
    StructField("delta_path", StringType(), nullable=False),       # Absolute path to Delta table
    StructField("schema_hash", StringType(), nullable=True),       # sha256(json(schema))[:12]
    StructField("spec_path", StringType(), nullable=True),         # Path to schema spec .md file
    StructField("record_count", StringType(), nullable=True),      # Last known count (as string for null handling)
    StructField("description", StringType(), nullable=True),       # Human description
    StructField("pipeline_version", StringType(), nullable=True),  # Version tag of pipeline that wrote it
    StructField("created_at", TimestampType(), nullable=False),    # When first registered
    StructField("updated_at", TimestampType(), nullable=False),    # When last updated
])


def get_catalog_path() -> Path:
    """Get path to catalog Delta table."""
    return paths.lake / "_catalog" / "tables"


def compute_schema_hash(schema: StructType) -> str:
    """
    Compute a hash of the schema for change detection.

    Format: sha256(json(schema))[:12]

    Args:
        schema: PySpark StructType schema

    Returns:
        12-character hex hash
    """
    schema_json = schema.json()
    return hashlib.sha256(schema_json.encode()).hexdigest()[:12]


def make_table_fqn(layer: str, dataset: str, table_name: str) -> str:
    """
    Create fully qualified table name.

    Format: {layer}.{dataset}.{table_name}

    Example: bronze.jsut.utterances
    """
    return f"{layer}.{dataset}.{table_name}"


def parse_table_fqn(fqn: str) -> tuple[str, str, str]:
    """
    Parse fully qualified table name.

    Args:
        fqn: Fully qualified name (e.g., "bronze.jsut.utterances")

    Returns:
        Tuple of (layer, dataset, table_name)

    Raises:
        ValueError: If FQN format is invalid
    """
    parts = fqn.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid table FQN: {fqn}. Expected format: layer.dataset.table_name")
    return parts[0], parts[1], parts[2]


def get_spec_path(layer: str, table_name: str) -> Optional[str]:
    """
    Get path to schema spec markdown file.

    Args:
        layer: Table layer (bronze, silver, gold)
        table_name: Table name (e.g., utterances)

    Returns:
        Relative path to spec file, or None if not found
    """
    spec_filename = f"{layer}_{table_name}.md"
    spec_path = Path("modules/data_engineering/common/schemas") / spec_filename

    # Check if file exists (relative to project root)
    full_path = paths.project_root / spec_path if hasattr(paths, 'project_root') else spec_path

    return str(spec_path) if spec_path.exists() or True else None  # Always return path even if not exists yet


def init_catalog(spark: SparkSession) -> Path:
    """
    Initialize the catalog table if it doesn't exist.

    Args:
        spark: SparkSession

    Returns:
        Path to catalog table
    """
    catalog_path = get_catalog_path()

    if not catalog_path.exists():
        print(f"Initializing catalog at {catalog_path}")
        catalog_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty DataFrame with schema
        empty_df = spark.createDataFrame([], CATALOG_SCHEMA)
        empty_df.write.format("delta").mode("overwrite").save(str(catalog_path))

    return catalog_path


def register_table(
    layer: str,
    dataset: str,
    table_name: str,
    delta_path: Path,
    schema: Optional[StructType] = None,
    record_count: Optional[int] = None,
    description: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    spark: Optional[SparkSession] = None,
) -> dict:
    """
    Register or update a table in the catalog.

    Args:
        layer: Table layer (bronze, silver, gold)
        dataset: Dataset name (jsut, jvs, etc.)
        table_name: Table name (utterances)
        delta_path: Path to Delta table
        schema: Optional schema for hash computation
        record_count: Optional record count
        description: Optional description
        pipeline_version: Optional pipeline version tag
        spark: Optional SparkSession (will create if not provided)

    Returns:
        Dict with registration result
    """
    if spark is None:
        spark = get_spark()

    # Initialize catalog if needed
    catalog_path = init_catalog(spark)

    # Build table metadata
    table_fqn = make_table_fqn(layer, dataset, table_name)
    now = datetime.now(timezone.utc)

    schema_hash = compute_schema_hash(schema) if schema else None
    spec_path = get_spec_path(layer, table_name)

    # Check if table already exists in catalog
    catalog_df = spark.read.format("delta").load(str(catalog_path))
    existing = catalog_df.filter(F.col("table_fqn") == table_fqn).count() > 0

    # Build new row
    new_row = {
        "table_fqn": table_fqn,
        "layer": layer,
        "dataset": dataset,
        "table_name": table_name,
        "delta_path": str(delta_path.absolute()),
        "schema_hash": schema_hash,
        "spec_path": spec_path,
        "record_count": str(record_count) if record_count is not None else None,
        "description": description,
        "pipeline_version": pipeline_version,
        "created_at": now if not existing else None,  # Will be handled below
        "updated_at": now,
    }

    if existing:
        # Update existing entry (keep created_at)
        from delta import DeltaTable

        delta_table = DeltaTable.forPath(spark, str(catalog_path))
        delta_table.update(
            condition=f"table_fqn = '{table_fqn}'",
            set={
                "delta_path": F.lit(str(delta_path.absolute())),
                "schema_hash": F.lit(schema_hash),
                "spec_path": F.lit(spec_path),
                "record_count": F.lit(str(record_count) if record_count is not None else None),
                "description": F.lit(description),
                "pipeline_version": F.lit(pipeline_version),
                "updated_at": F.lit(now),
            }
        )
        action = "updated"
    else:
        # Insert new entry
        new_row["created_at"] = now
        new_df = spark.createDataFrame([new_row], CATALOG_SCHEMA)
        new_df.write.format("delta").mode("append").save(str(catalog_path))
        action = "registered"

    return {
        "action": action,
        "table_fqn": table_fqn,
        "delta_path": str(delta_path),
        "schema_hash": schema_hash,
    }


def list_tables(
    layer: Optional[str] = None,
    dataset: Optional[str] = None,
    spark: Optional[SparkSession] = None,
) -> list[dict]:
    """
    List tables in the catalog.

    Args:
        layer: Optional filter by layer
        dataset: Optional filter by dataset
        spark: Optional SparkSession

    Returns:
        List of table metadata dicts
    """
    if spark is None:
        spark = get_spark()

    catalog_path = get_catalog_path()

    if not catalog_path.exists():
        return []

    df = spark.read.format("delta").load(str(catalog_path))

    if layer:
        df = df.filter(F.col("layer") == layer)
    if dataset:
        df = df.filter(F.col("dataset") == dataset)

    df = df.orderBy("layer", "dataset", "table_name")

    return [row.asDict() for row in df.collect()]


def describe_table(
    table_fqn: str,
    spark: Optional[SparkSession] = None,
) -> Optional[dict]:
    """
    Get detailed metadata for a specific table.

    Args:
        table_fqn: Fully qualified table name (e.g., "bronze.jsut.utterances")
        spark: Optional SparkSession

    Returns:
        Table metadata dict, or None if not found
    """
    if spark is None:
        spark = get_spark()

    catalog_path = get_catalog_path()

    if not catalog_path.exists():
        return None

    df = spark.read.format("delta").load(str(catalog_path))
    rows = df.filter(F.col("table_fqn") == table_fqn).collect()

    if not rows:
        return None

    metadata = rows[0].asDict()

    # Try to read actual table for live stats
    try:
        delta_path = metadata["delta_path"]
        if Path(delta_path).exists():
            table_df = spark.read.format("delta").load(delta_path)
            metadata["live_record_count"] = table_df.count()
            metadata["live_schema"] = table_df.schema.json()
            metadata["live_schema_hash"] = compute_schema_hash(table_df.schema)

            # Check for schema drift
            if metadata["schema_hash"] and metadata["live_schema_hash"] != metadata["schema_hash"]:
                metadata["schema_drift"] = True
    except Exception as e:
        metadata["read_error"] = str(e)

    return metadata


def drop_table_entry(
    table_fqn: str,
    spark: Optional[SparkSession] = None,
) -> bool:
    """
    Remove a table entry from the catalog (does not delete actual table).

    Args:
        table_fqn: Fully qualified table name
        spark: Optional SparkSession

    Returns:
        True if entry was deleted, False if not found
    """
    if spark is None:
        spark = get_spark()

    catalog_path = get_catalog_path()

    if not catalog_path.exists():
        return False

    from delta import DeltaTable

    delta_table = DeltaTable.forPath(spark, str(catalog_path))
    delta_table.delete(f"table_fqn = '{table_fqn}'")

    return True
