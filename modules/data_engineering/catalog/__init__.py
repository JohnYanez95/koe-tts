"""
Lakehouse catalog for table discovery and metadata.

The catalog tracks all Delta tables in the lakehouse, providing
metastore-like functionality without external infrastructure.

Location: lake/_catalog/tables

Usage:
    koe catalog list
    koe catalog describe bronze.jsut.utterances

Programmatic:
    from modules.data_engineering.catalog import list_tables, describe_table
    tables = list_tables(layer="bronze")
    meta = describe_table("bronze.jsut.utterances")
"""

from modules.data_engineering.common.catalog import (
    CATALOG_SCHEMA,
    compute_schema_hash,
    describe_table,
    drop_table_entry,
    get_catalog_path,
    init_catalog,
    list_tables,
    make_table_fqn,
    parse_table_fqn,
    register_table,
)

__all__ = [
    "CATALOG_SCHEMA",
    "compute_schema_hash",
    "describe_table",
    "drop_table_entry",
    "get_catalog_path",
    "init_catalog",
    "list_tables",
    "make_table_fqn",
    "parse_table_fqn",
    "register_table",
]
