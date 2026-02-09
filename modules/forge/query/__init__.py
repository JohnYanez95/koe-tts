"""Query engine helpers for Spark and DuckDB.

Spark is imported directly::

    from modules.forge.query.spark import get_spark

DuckDB names are re-exported here via lazy ``__getattr__`` to avoid
requiring ``duckdb`` at package import time.
"""

__all__ = [
    "DuckDBClient",
    "create_duckdb_client",
    "create_s3_duckdb_client",
]


def __getattr__(name: str):  # noqa: ANN001
    if name in __all__:
        from modules.forge.query import duckdb as _duckdb

        return getattr(_duckdb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
