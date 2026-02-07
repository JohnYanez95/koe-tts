"""Unity Catalog client wrapper for koe-tts (async-safe).

CRITICAL: The `unitycatalog-client` SDK is async (aiohttp). All API calls return coroutines
that must be awaited. We use an event-loop-safe sync facade that works in both regular Python
and Jupyter/notebook environments.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from typing import TYPE_CHECKING

from unitycatalog.client import ApiClient, Configuration
from unitycatalog.client.api import CatalogsApi, SchemasApi, TablesApi
from unitycatalog.client.models import (
    ColumnInfo,
    CreateCatalog,
    CreateSchema,
    CreateTable,
    DataSourceFormat,
    TableType,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType

# Configuration
# Python REST client uses /api/2.1/unity-catalog suffix
# Spark/DuckDB use base URL (UC_SERVER_URL env var) - configured elsewhere
UC_REST_API = os.getenv(
    "UC_API_URL", "http://localhost:8080/api/2.1/unity-catalog"
)
UC_TOKEN = os.getenv("UC_TOKEN")  # None if auth disabled
CATALOG_NAME = "koe_tts"
UC_ENABLED = os.getenv("UC_ENABLED", "false").lower() == "true"

# Spark type -> UC type mapping
SPARK_TO_UC_TYPES = {
    "string": "STRING",
    "int": "INT",
    "integer": "INT",
    "long": "LONG",
    "bigint": "LONG",
    "float": "FLOAT",
    "double": "DOUBLE",
    "boolean": "BOOLEAN",
    "timestamp": "TIMESTAMP",
    "date": "DATE",
    "binary": "BINARY",
}


# --- Event-loop-safe async runner ---


def _run_sync(coro):
    """
    Run async coroutine synchronously, safe for any context.

    - No running loop: use asyncio.run()
    - Running loop (Jupyter, etc.): spawn thread, run there, block for result
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)

    # Already in an event loop (Jupyter, async CLI, etc.)
    # Run in a dedicated thread to avoid "cannot call asyncio.run() from running loop"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# --- Async internals ---


def _client() -> ApiClient:
    """
    Get configured UC API client.

    NOTE: No @lru_cache - async clients can bind to event loops, causing
    "attached to a different loop" errors when cached across contexts.
    UC calls are metadata-sized; overhead is fine.
    """
    config = Configuration(host=UC_REST_API)  # Python client uses /api/... URL
    client = ApiClient(configuration=config)
    if UC_TOKEN:
        client.default_headers["Authorization"] = f"Bearer {UC_TOKEN}"
    return client


def _is_not_found(exc: Exception) -> bool:
    """Check if exception is a 404 Not Found."""
    # Prefer structured status code when available
    status = getattr(exc, "status", None)
    if status == 404:
        return True

    # Fallback: only match clear 404 patterns (avoid generic "not found")
    msg = str(exc).lower()
    return ("404 not found" in msg) or ("status: 404" in msg) or ("http 404" in msg)


async def _ensure_catalog_exists_async() -> None:
    """Create catalog if it doesn't exist (async)."""
    api = CatalogsApi(_client())
    try:
        await api.get_catalog(CATALOG_NAME)
    except Exception as e:
        if _is_not_found(e):
            await api.create_catalog(create_catalog=CreateCatalog(name=CATALOG_NAME))
        else:
            raise  # Re-raise auth errors, connectivity issues, etc.


async def _ensure_schema_exists_async(schema_name: str) -> None:
    """Create schema if it doesn't exist (async)."""
    api = SchemasApi(_client())
    try:
        await api.get_schema(f"{CATALOG_NAME}.{schema_name}")
    except Exception as e:
        if _is_not_found(e):
            await api.create_schema(
                create_schema=CreateSchema(
                    catalog_name=CATALOG_NAME,
                    name=schema_name,
                )
            )
        else:
            raise


async def _register_table_async(
    schema_name: str,
    table_name: str,
    storage_location: str,
    columns: list[ColumnInfo],
) -> str:
    """Register table with UC (async)."""
    api = TablesApi(_client())
    full_name = f"{CATALOG_NAME}.{schema_name}.{table_name}"

    # Check if already exists
    try:
        await api.get_table(full_name)
        return full_name  # Already registered
    except Exception as e:
        if not _is_not_found(e):
            raise  # Re-raise non-404 errors

    # Create
    await api.create_table(
        create_table=CreateTable(
            catalog_name=CATALOG_NAME,
            schema_name=schema_name,
            name=table_name,
            table_type=TableType.EXTERNAL,
            data_source_format=DataSourceFormat.DELTA,
            storage_location=storage_location,
            columns=columns,
        )
    )
    return full_name


async def _list_tables_async(schema_name: str | None) -> list[dict]:
    """List tables from UC (async)."""
    tables_api = TablesApi(_client())
    results = []

    if schema_name:
        tables = await tables_api.list_tables(
            catalog_name=CATALOG_NAME,
            schema_name=schema_name,
        )
        results.extend([t.to_dict() for t in tables.tables or []])
    else:
        schemas_api = SchemasApi(_client())
        schemas = await schemas_api.list_schemas(catalog_name=CATALOG_NAME)
        for schema in schemas.schemas or []:
            tables = await tables_api.list_tables(
                catalog_name=CATALOG_NAME,
                schema_name=schema.name,
            )
            results.extend([t.to_dict() for t in tables.tables or []])

    return results


# --- Sync public API ---


def spark_schema_to_uc_columns(schema: StructType) -> list[ColumnInfo]:
    """Convert Spark schema to UC column definitions."""
    columns = []
    for i, field in enumerate(schema.fields):
        simple_type = field.dataType.simpleString()
        base_type = simple_type.split("<")[0].split("(")[0].lower()
        uc_type = SPARK_TO_UC_TYPES.get(base_type, "STRING")

        columns.append(
            ColumnInfo(
                name=field.name,
                type_name=uc_type,
                type_text=simple_type,
                type_json=field.dataType.json(),
                position=i,
                nullable=field.nullable,
            )
        )
    return columns


def ensure_catalog_exists() -> None:
    """Create catalog if it doesn't exist."""
    if UC_ENABLED:
        _run_sync(_ensure_catalog_exists_async())


def ensure_schema_exists(schema_name: str) -> None:
    """Create schema if it doesn't exist."""
    if UC_ENABLED:
        _run_sync(_ensure_schema_exists_async(schema_name))


def register_table(
    layer: str,
    dataset: str,
    table_name: str,
    storage_location: str,
    spark: SparkSession,
) -> str:
    """
    Register a Delta table with Unity Catalog (idempotent).

    Args:
        layer: bronze, silver, or gold
        dataset: Dataset name (e.g., jsut)
        table_name: Table name (e.g., utterances)
        storage_location: file:///lake/... URI
        spark: SparkSession for schema inference

    Returns:
        Full table name (catalog.schema.table)
    """
    if not UC_ENABLED:
        return f"{layer}.{dataset}.{table_name}"

    schema_name = f"{layer}_{dataset}"

    # Ensure catalog and schema exist
    ensure_catalog_exists()
    ensure_schema_exists(schema_name)

    # Infer schema from Delta table
    local_path = storage_location.replace("file://", "")
    df = spark.read.format("delta").load(local_path)
    columns = spark_schema_to_uc_columns(df.schema)

    # Register (async)
    return _run_sync(_register_table_async(schema_name, table_name, storage_location, columns))


def list_tables(layer: str | None = None, dataset: str | None = None) -> list[dict]:
    """List tables from Unity Catalog."""
    if not UC_ENABLED:
        return []

    schema_name = f"{layer}_{dataset}" if layer and dataset else None
    return _run_sync(_list_tables_async(schema_name))
