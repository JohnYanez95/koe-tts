"""DuckDB client for querying Delta tables.

Uses delta_scan() for direct Delta table access. UC catalog integration
is handled via Spark; DuckDB queries the underlying Delta files directly.

All user-supplied identifiers (layer, dataset, table, columns) and filter
values are validated before touching SQL.  String interpolation only occurs
for values that have passed through ``validate_ident`` / ``safe_sql_string``.

NOTE: DuckDB's UC extension is not yet in stable channels. When it reaches
stable, we can add native UC ATTACH support here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from .filters import (
    build_where,
    parse_columns,
    safe_sql_string,
    schema_map_from_columns,
    validate_ident,
)
from .paths import paths

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

ALLOWED_LAYERS: frozenset[str] = frozenset({"bronze", "silver", "gold"})
MAX_LIMIT: int = 500_000


# ── Connection management ───────────────────────────────────────────────

@lru_cache
def get_connection() -> duckdb.DuckDBPyConnection:
    """Get DuckDB connection with Delta extension loaded."""
    conn = duckdb.connect()
    conn.execute("INSTALL delta;")
    conn.execute("LOAD delta;")
    return conn


# ── Path + identifier safety ────────────────────────────────────────────

def validate_layer(layer: str) -> str:
    """Validate and normalise a medallion layer name."""
    layer = layer.strip().lower()
    if layer not in ALLOWED_LAYERS:
        raise ValueError(
            f"Invalid layer {layer!r}; must be one of {sorted(ALLOWED_LAYERS)}"
        )
    return layer


def safe_table_path(layer: str, dataset: str, table: str) -> Path:
    """Build and validate a Delta table path under ``paths.lake``.

    All three components are validated as safe identifiers (``[a-z][a-z0-9_]*``).
    The resolved path is checked to remain under the lake root to prevent
    traversal attacks.
    """
    layer = validate_layer(layer)
    dataset = validate_ident(dataset, "dataset")
    table = validate_ident(table, "table")

    p = (paths.lake / layer / dataset / table).resolve(strict=False)
    root = paths.lake.resolve(strict=False)

    if not p.is_relative_to(root):
        raise ValueError("Path escapes lake root")
    return p


def _delta_scan_literal(table_path: Path) -> str:
    """Build a safe ``delta_scan('...')`` expression.

    DuckDB table functions don't support bind parameters, so the path
    must be interpolated as a string literal.  We escape single quotes
    as defense-in-depth (``safe_table_path`` already guarantees the path
    components contain only ``[a-z0-9_]``).
    """
    return f"delta_scan({safe_sql_string(str(table_path))})"


# ── Schema introspection ────────────────────────────────────────────────

def _describe_table(table_path: Path) -> list[str]:
    """Return column names for a Delta table via DESCRIBE."""
    conn = get_connection()
    scan = _delta_scan_literal(table_path)
    df = conn.execute(f"DESCRIBE SELECT * FROM {scan}").fetchdf()
    return df["column_name"].tolist()


# ── Core query functions ────────────────────────────────────────────────

def query(sql: str) -> pd.DataFrame:
    """Execute raw SQL and return DataFrame.

    .. warning::
       This accepts arbitrary SQL.  Use ``query_table`` for safe
       parameterised access.  ``query sql`` in the CLI exposes this
       intentionally for power users who want full SQL control.
    """
    return get_connection().execute(sql).fetchdf()


def query_table(
    layer: str,
    dataset: str,
    table: str,
    columns: str = "*",
    filters: Sequence[str] | None = None,
    any_filters: Sequence[str] | None = None,
    limit: int | None = None,
    introspect: bool = True,
) -> pd.DataFrame:
    """Query a Delta table via delta_scan with safe parameterised filters.

    Every user-supplied value is either:
    - Validated as a safe identifier and quoted (layer, dataset, table, columns)
    - Bound as a ``?`` parameter (filter values, limit)

    Args:
        layer: bronze, silver, or gold
        dataset: Dataset name (e.g., jsut)
        table: Table name (e.g., utterances)
        columns: Column selection (default "*")
        filters: AND filters (e.g., ["split='train'", "duration_sec>=1.0"])
        any_filters: OR filters combined with AND filters
        limit: Optional LIMIT
        introspect: If True, validate columns + filters against table schema

    Returns:
        pandas DataFrame with results

    Raises:
        FilterParseError: If filter syntax is invalid or column unknown
        ValueError: If layer/dataset/table are invalid identifiers
    """
    table_path = safe_table_path(layer, dataset, table)
    conn = get_connection()
    scan = _delta_scan_literal(table_path)

    # ── Schema introspection (optional but recommended) ─────────────
    schema_map = None
    if introspect:
        col_names = _describe_table(table_path)
        schema_map = schema_map_from_columns(col_names)
        logger.debug("Schema for %s: %s", table_path, sorted(schema_map))

    # ── Safe column list ────────────────────────────────────────────
    cols_sql = parse_columns(columns, schema_map=schema_map)

    # ── Safe WHERE clause ───────────────────────────────────────────
    where_sql, params = build_where(
        filters=filters,
        any_filters=any_filters,
        schema_map=schema_map,
    )

    # ── Assemble query ──────────────────────────────────────────────
    sql = f"SELECT {cols_sql} FROM {scan}"
    if where_sql:
        sql += f" {where_sql}"

    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit > MAX_LIMIT:
            raise ValueError(f"limit {limit} exceeds max {MAX_LIMIT}")
        sql += " LIMIT ?"
        params.append(limit)

    logger.debug("SQL: %s", sql)
    logger.debug("Params: %s", params)

    return conn.execute(sql, params).fetchdf()


def scan_delta(
    path: str | Path,
    columns: str = "*",
    limit: int | None = None,
) -> pd.DataFrame:
    """Direct delta_scan for any Delta table path.

    .. warning::
       This accepts an arbitrary path.  The path is escaped via
       ``safe_sql_string`` but not validated against the lake root.
       Prefer ``query_table`` for catalog-managed tables.
    """
    scan = f"delta_scan({safe_sql_string(str(path))})"

    # columns: without a schema map we still validate identifiers
    cols_sql = parse_columns(columns) if columns.strip() != "*" else "*"

    sql = f"SELECT {cols_sql} FROM {scan}"

    params: list = []
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        sql += " LIMIT ?"
        params.append(limit)

    return get_connection().execute(sql, params).fetchdf()


def list_tables() -> list[dict]:
    """List Delta tables by scanning the lake directory structure."""
    tables = []
    lake = paths.lake

    if not lake.exists():
        return tables

    for layer in sorted(ALLOWED_LAYERS):
        layer_path = lake / layer
        if not layer_path.exists():
            continue
        for dataset_dir in layer_path.iterdir():
            if not dataset_dir.is_dir():
                continue
            for table_dir in dataset_dir.iterdir():
                if (table_dir / "_delta_log").exists():
                    tables.append({
                        "layer": layer,
                        "dataset": dataset_dir.name,
                        "table": table_dir.name,
                        "path": str(table_dir),
                    })
    return tables


def table_info(layer: str, dataset: str, table: str) -> pd.DataFrame:
    """Get schema info for a table (validated path)."""
    table_path = safe_table_path(layer, dataset, table)
    scan = _delta_scan_literal(table_path)
    return get_connection().execute(f"DESCRIBE SELECT * FROM {scan};").fetchdf()
