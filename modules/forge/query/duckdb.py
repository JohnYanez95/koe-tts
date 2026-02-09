"""DuckDB client for querying Delta tables.

Uses ``delta_scan()`` for direct Delta table access.  UC catalog integration
is handled via Spark; DuckDB queries the underlying Delta files directly.

All user-supplied identifiers (layer, dataset, table, columns) and filter
values are validated before touching SQL.  String interpolation only occurs
for values that have passed through ``validate_ident`` / ``safe_sql_string``.

``duckdb`` and ``pandas`` are lazy-imported — this module can be imported
without either installed.  ``ImportError`` is raised at connection time
with an actionable message.

Public API
----------
- ``DuckDBClient`` — parameterized DuckDB client
- ``create_duckdb_client(lake_root)`` — local-filesystem factory
- ``create_s3_duckdb_client(lake_root, ...)`` — S3-backed factory
- ``validate_layer(layer)`` — medallion layer validation
- ``ALLOWED_LAYERS`` / ``MAX_LIMIT`` — constants
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modules.forge.sql.filters import (
    build_where,
    parse_columns,
    safe_sql_string,
    schema_map_from_columns,
    validate_ident,
)

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

ALLOWED_LAYERS: frozenset[str] = frozenset({"bronze", "silver", "gold"})
MAX_LIMIT: int = 500_000


# ── Layer validation ─────────────────────────────────────────────────


def validate_layer(layer: str) -> str:
    """Validate and normalise a medallion layer name."""
    layer = layer.strip().lower()
    if layer not in ALLOWED_LAYERS:
        raise ValueError(
            f"Invalid layer {layer!r}; must be one of {sorted(ALLOWED_LAYERS)}"
        )
    return layer


# ── DuckDB client ────────────────────────────────────────────────────


class DuckDBClient:
    """DuckDB client with parameterized lake root.

    All queries are rooted at ``lake_root``.  The connection is created
    lazily on first use and cached for the lifetime of the instance.

    Parameters
    ----------
    lake_root
        Root directory of the Delta lake (local path or S3 URI).
    s3_config
        Optional dict with S3 credentials to configure at connection time.
        Keys: ``endpoint``, ``access_key``, ``secret_key``, ``region``.
    """

    def __init__(
        self,
        lake_root: Path | str,
        s3_config: dict[str, str] | None = None,
    ) -> None:
        self.lake_root = Path(lake_root) if not str(lake_root).startswith("s3") else lake_root
        self._s3_config = s3_config
        self._conn: duckdb.DuckDBPyConnection | None = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create a DuckDB connection with the Delta extension.

        Raises ``ImportError`` if duckdb is not installed.
        """
        if self._conn is not None:
            return self._conn

        try:
            import duckdb as _duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for DuckDB operations. "
                "Install with: pip install duckdb"
            ) from None

        conn = _duckdb.connect()
        conn.execute("INSTALL delta;")
        conn.execute("LOAD delta;")

        # Configure S3 credentials if provided
        if self._s3_config:
            if endpoint := self._s3_config.get("endpoint"):
                conn.execute(f"SET s3_endpoint='{endpoint.replace('http://', '').replace('https://', '')}';")
                if endpoint.startswith("http://"):
                    conn.execute("SET s3_use_ssl=false;")
            if access_key := self._s3_config.get("access_key"):
                conn.execute(f"SET s3_access_key_id='{access_key}';")
            if secret_key := self._s3_config.get("secret_key"):
                conn.execute(f"SET s3_secret_access_key='{secret_key}';")
            if region := self._s3_config.get("region"):
                conn.execute(f"SET s3_region='{region}';")
            conn.execute("SET s3_url_style='path';")

        self._conn = conn
        return conn

    # ── Path safety ──────────────────────────────────────────────

    def _safe_table_path(self, layer: str, dataset: str, table: str) -> Path:
        """Build and validate a Delta table path under ``lake_root``.

        All three components are validated as safe identifiers
        (``[a-z][a-z0-9_]*``).  The resolved path is checked to remain
        under the lake root to prevent traversal attacks.
        """
        layer = validate_layer(layer)
        dataset = validate_ident(dataset, "dataset")
        table = validate_ident(table, "table")

        lake_root = Path(self.lake_root)
        p = (lake_root / layer / dataset / table).resolve(strict=False)
        root = lake_root.resolve(strict=False)

        if not p.is_relative_to(root):
            raise ValueError("Path escapes lake root")
        return p

    def _delta_scan_literal(self, table_path: Path) -> str:
        """Build a safe ``delta_scan('...')`` expression.

        DuckDB table functions don't support bind parameters, so the path
        must be interpolated as a string literal.  We escape single quotes
        as defense-in-depth (``_safe_table_path`` already guarantees the
        path components contain only ``[a-z0-9_]``).
        """
        return f"delta_scan({safe_sql_string(str(table_path))})"

    def _describe_table(self, table_path: Path) -> list[str]:
        """Return column names for a Delta table via DESCRIBE."""
        conn = self.get_connection()
        scan = self._delta_scan_literal(table_path)
        df = conn.execute(f"DESCRIBE SELECT * FROM {scan}").fetchdf()
        return df["column_name"].tolist()

    # ── Core query methods ───────────────────────────────────────

    def query_raw(self, sql: str) -> pd.DataFrame:
        """Execute raw SQL and return DataFrame.

        .. warning::
           This accepts arbitrary SQL with **no validation or
           parameterization**.  Prefer ``query_table`` for safe,
           structured access.  This method exists for power users
           who need full SQL control.
        """
        return self.get_connection().execute(sql).fetchdf()

    def query_table(
        self,
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
        - Validated as a safe identifier and quoted (layer, dataset, table,
          columns)
        - Bound as a ``?`` parameter (filter values, limit)

        Parameters
        ----------
        layer
            bronze, silver, or gold
        dataset
            Dataset name (e.g., jsut)
        table
            Table name (e.g., utterances)
        columns
            Column selection (default ``"*"``)
        filters
            AND filters (e.g., ``["split='train'", "duration_sec>=1.0"]``)
        any_filters
            OR filters combined with AND filters
        limit
            Optional LIMIT
        introspect
            If True, validate columns + filters against table schema

        Raises
        ------
        FilterParseError
            If filter syntax is invalid or column unknown
        ValueError
            If layer/dataset/table are invalid identifiers
        """
        table_path = self._safe_table_path(layer, dataset, table)
        conn = self.get_connection()
        scan = self._delta_scan_literal(table_path)

        # ── Schema introspection (optional but recommended) ──────
        schema_map = None
        if introspect:
            col_names = self._describe_table(table_path)
            schema_map = schema_map_from_columns(col_names)
            logger.debug("Schema for %s: %s", table_path, sorted(schema_map))

        # ── Safe column list ─────────────────────────────────────
        cols_sql = parse_columns(columns, schema_map=schema_map)

        # ── Safe WHERE clause ────────────────────────────────────
        where_sql, params = build_where(
            filters=filters,
            any_filters=any_filters,
            schema_map=schema_map,
        )

        # ── Assemble query ───────────────────────────────────────
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
        self,
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

        cols_sql = parse_columns(columns) if columns.strip() != "*" else "*"

        sql = f"SELECT {cols_sql} FROM {scan}"

        params: list[Any] = []
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            params.append(limit)

        return self.get_connection().execute(sql, params).fetchdf()

    def list_tables(self) -> list[dict]:
        """List Delta tables by scanning the lake directory structure."""
        tables: list[dict] = []
        lake = Path(self.lake_root)

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

    def table_info(
        self, layer: str, dataset: str, table: str,
    ) -> pd.DataFrame:
        """Get schema info for a table (validated path)."""
        table_path = self._safe_table_path(layer, dataset, table)
        scan = self._delta_scan_literal(table_path)
        return self.get_connection().execute(
            f"DESCRIBE SELECT * FROM {scan};"
        ).fetchdf()


# ── Factory functions ────────────────────────────────────────────────


def create_duckdb_client(lake_root: Path | str) -> DuckDBClient:
    """Create a DuckDB client for a local-filesystem lake.

    Parameters
    ----------
    lake_root
        Root directory of the Delta lake.
    """
    return DuckDBClient(lake_root=lake_root)


def create_s3_duckdb_client(
    lake_root: str,
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str | None = None,
) -> DuckDBClient:
    """Create a DuckDB client for an S3-backed lake.

    Validates ``lake_root`` starts with ``s3://``.  Raises ``ValueError``
    if ``s3a://`` is passed (that's for Spark — use ``FORGE_LAKE_ROOT_S3A``).

    Parameters
    ----------
    lake_root
        S3 URI for the lake root (must start with ``s3://``).
    endpoint
        S3/MinIO endpoint (e.g. ``http://localhost:9000``).
    access_key
        S3 access key.
    secret_key
        S3 secret key.
    region
        S3 region (e.g. ``us-east-1``).
    """
    if lake_root.startswith("s3a://"):
        raise ValueError(
            f"lake_root uses wrong scheme: {lake_root!r}. "
            "DuckDB requires s3:// (not s3a://). "
            "s3a:// is for Spark — use FORGE_LAKE_ROOT_S3A for Spark."
        )
    if not lake_root.startswith("s3://"):
        raise ValueError(
            f"lake_root must start with s3://, got: {lake_root!r}"
        )

    s3_config: dict[str, str] = {}
    if endpoint:
        s3_config["endpoint"] = endpoint
    if access_key:
        s3_config["access_key"] = access_key
    if secret_key:
        s3_config["secret_key"] = secret_key
    if region:
        s3_config["region"] = region

    return DuckDBClient(lake_root=lake_root, s3_config=s3_config or None)
