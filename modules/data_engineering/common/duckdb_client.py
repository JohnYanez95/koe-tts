"""DuckDB client with Unity Catalog integration.

Provides query access to Delta tables via UC attach or direct delta_scan fallback.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from .paths import paths

# Configuration
# IMPORTANT: DuckDB UC extension expects base server URL, NOT /api/2.1/unity-catalog
# The extension builds its own REST paths internally
UC_SERVER_URL = os.getenv("UC_SERVER_URL", "http://localhost:8080")
UC_TOKEN = os.getenv("UC_TOKEN", "not-used")  # Required param even if auth disabled
UC_ENABLED = os.getenv("UC_ENABLED", "false").lower() == "true"
CATALOG_NAME = "koe_tts"


def _try_install_uc_extension(conn: duckdb.DuckDBPyConnection) -> str | None:
    """
    Try to install UC extension. Returns extension name if successful, None otherwise.

    Per UC OSS docs, try uc_catalog from core_nightly first.
    """
    # Try uc_catalog from core_nightly (documented path)
    try:
        conn.execute("INSTALL uc_catalog FROM core_nightly;")
        conn.execute("LOAD uc_catalog;")
        return "uc_catalog"
    except Exception:
        pass

    # Fallback to unity_catalog if/when it appears in stable channels
    try:
        conn.execute("INSTALL unity_catalog;")
        conn.execute("LOAD unity_catalog;")
        return "unity_catalog"
    except Exception:
        pass

    return None


def _setup_uc_catalog(conn: duckdb.DuckDBPyConnection, ext_type: str) -> bool:
    """Set up UC secret and attach catalog. Returns True if successful."""
    try:
        # Create named UC secret
        # IMPORTANT: ENDPOINT is base server URL, not /api/... path
        if ext_type == "uc_catalog":
            conn.execute(f"""
                CREATE SECRET uc (
                    TYPE UC,
                    TOKEN '{UC_TOKEN}',
                    ENDPOINT '{UC_SERVER_URL}',
                    AWS_REGION 'us-east-2'
                );
            """)
            conn.execute(f"ATTACH '{CATALOG_NAME}' AS {CATALOG_NAME} (TYPE UC_CATALOG);")
        else:
            conn.execute(f"""
                CREATE SECRET uc (
                    TYPE unity_catalog,
                    TOKEN '{UC_TOKEN}',
                    ENDPOINT '{UC_SERVER_URL}',
                    AWS_REGION 'us-east-2'
                );
            """)
            conn.execute(f"ATTACH '{CATALOG_NAME}' AS {CATALOG_NAME} (TYPE unity_catalog);")
        return True
    except Exception:
        return False


@lru_cache
def get_connection(use_uc: bool | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get DuckDB connection, optionally with UC catalog attached.

    Args:
        use_uc: Force UC on/off. None = use UC_ENABLED env var.

    Returns:
        DuckDB connection with delta extension loaded.
        If UC enabled and available, catalog is attached.
    """
    conn = duckdb.connect()

    # Always install delta for direct scans
    conn.execute("INSTALL delta;")
    conn.execute("LOAD delta;")

    # Try UC if enabled
    should_use_uc = use_uc if use_uc is not None else UC_ENABLED
    if should_use_uc:
        ext_type = _try_install_uc_extension(conn)
        if ext_type:
            _setup_uc_catalog(conn, ext_type)

    return conn


def query(sql: str) -> pd.DataFrame:
    """Execute SQL query and return DataFrame."""
    conn = get_connection()
    return conn.execute(sql).fetchdf()


def query_table(
    layer: str,
    dataset: str,
    table: str,
    columns: str = "*",
    where: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Query a specific table.

    Tries UC catalog first if enabled, falls back to delta_scan.

    Args:
        layer: bronze, silver, or gold
        dataset: Dataset name (e.g., jsut)
        table: Table name (e.g., utterances)
        columns: Column selection (default "*")
        where: Optional WHERE clause
        limit: Optional LIMIT

    Returns:
        pandas DataFrame with results
    """
    conn = get_connection()

    # Build SQL
    if UC_ENABLED:
        # Try UC catalog path
        schema_name = f"{layer}_{dataset}"
        full_name = f"{CATALOG_NAME}.{schema_name}.{table}"
        sql = f"SELECT {columns} FROM {full_name}"
    else:
        # Direct delta_scan fallback
        table_path = paths.lake / layer / dataset / table
        sql = f"SELECT {columns} FROM delta_scan('{table_path}')"

    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"

    return conn.execute(sql).fetchdf()


def scan_delta(path: str | Path, columns: str = "*", limit: int | None = None) -> pd.DataFrame:
    """
    Direct delta_scan for tables not in UC.

    Args:
        path: Path to Delta table
        columns: Column selection
        limit: Optional LIMIT

    Returns:
        pandas DataFrame
    """
    conn = get_connection()
    sql = f"SELECT {columns} FROM delta_scan('{path}')"
    if limit:
        sql += f" LIMIT {limit}"
    return conn.execute(sql).fetchdf()


def list_tables() -> pd.DataFrame:
    """List all tables (UC if enabled, otherwise empty)."""
    conn = get_connection()
    if UC_ENABLED:
        return conn.execute("SHOW ALL TABLES;").fetchdf()
    return conn.execute("SELECT 'UC not enabled' AS message;").fetchdf()


def table_info(layer: str, dataset: str, table: str) -> pd.DataFrame:
    """Get schema info for a table."""
    conn = get_connection()

    if UC_ENABLED:
        schema_name = f"{layer}_{dataset}"
        full_name = f"{CATALOG_NAME}.{schema_name}.{table}"
        return conn.execute(f"DESCRIBE {full_name};").fetchdf()
    else:
        table_path = paths.lake / layer / dataset / table
        return conn.execute(f"DESCRIBE SELECT * FROM delta_scan('{table_path}');").fetchdf()
