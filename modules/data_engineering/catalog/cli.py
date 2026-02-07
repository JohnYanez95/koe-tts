"""
CLI for catalog operations.

Uses delta-rs (deltalake) for fast reads without Spark startup.
Spark is only used for writes (which happen during pipeline runs).

Usage:
    koe catalog list
    koe catalog list --layer bronze
    koe catalog list --dataset jsut
    koe catalog describe bronze.jsut.utterances
    koe catalog refresh bronze.jsut.utterances
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def get_catalog_path() -> Path:
    """Get path to catalog Delta table."""
    from modules.data_engineering.common.paths import paths
    return paths.lake / "_catalog" / "tables"


def read_catalog_fast() -> Optional[pd.DataFrame]:
    """
    Read catalog using delta-rs (no Spark startup).

    Returns:
        pandas DataFrame or None if catalog doesn't exist
    """
    catalog_path = get_catalog_path()

    if not catalog_path.exists():
        return None

    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(catalog_path))
        return dt.to_pandas()
    except Exception as e:
        # Fallback: catalog may be empty or corrupted
        return None


def main(args=None) -> int:
    """Main entry point for catalog CLI."""
    if args is None:
        args = parse_args()

    if args.command == "list":
        return catalog_list(layer=args.layer, dataset=args.dataset)
    elif args.command == "describe":
        return catalog_describe(args.table_fqn)
    elif args.command == "refresh":
        return catalog_refresh(args.table_fqn)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Lakehouse catalog CLI",
        prog="koe catalog",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List tables in catalog")
    list_parser.add_argument("--layer", choices=["bronze", "silver", "gold"], help="Filter by layer")
    list_parser.add_argument("--dataset", help="Filter by dataset")

    # describe command
    describe_parser = subparsers.add_parser("describe", help="Describe a table")
    describe_parser.add_argument("table_fqn", help="Fully qualified table name (e.g., bronze.jsut.utterances)")

    # refresh command
    refresh_parser = subparsers.add_parser("refresh", help="Refresh table metadata in catalog")
    refresh_parser.add_argument("table_fqn", help="Fully qualified table name")

    return parser.parse_args()


def catalog_list(layer: str = None, dataset: str = None) -> int:
    """List tables in the catalog (fast, no Spark)."""
    pdf = read_catalog_fast()

    if pdf is None or len(pdf) == 0:
        print("No tables registered in catalog.")
        print("\nTables are registered automatically when pipelines run.")
        print("Or use: koe catalog refresh <table_fqn>")
        return 0

    # Apply filters
    if layer:
        pdf = pdf[pdf["layer"] == layer]
    if dataset:
        pdf = pdf[pdf["dataset"] == dataset]

    if len(pdf) == 0:
        print(f"No tables found matching filters (layer={layer}, dataset={dataset})")
        return 0

    # Sort
    pdf = pdf.sort_values(["layer", "dataset", "table_name"])

    # Print header
    print(f"\n{'Table FQN':<35} {'Records':>10} {'Schema Hash':>14} {'Updated':<20}")
    print("-" * 85)

    for _, row in pdf.iterrows():
        fqn = row["table_fqn"]
        count = row.get("record_count") or "-"
        schema_hash = row.get("schema_hash") or "-"
        updated = row.get("updated_at")
        if updated:
            if isinstance(updated, pd.Timestamp):
                updated_str = updated.strftime("%Y-%m-%d %H:%M")
            elif isinstance(updated, datetime):
                updated_str = updated.strftime("%Y-%m-%d %H:%M")
            else:
                updated_str = str(updated)[:16]
        else:
            updated_str = "-"

        print(f"{fqn:<35} {count:>10} {schema_hash:>14} {updated_str:<20}")

    print(f"\nTotal: {len(pdf)} tables")
    return 0


def catalog_describe(table_fqn: str) -> int:
    """Describe a specific table (fast read, optional live stats with Spark)."""
    pdf = read_catalog_fast()

    if pdf is None or len(pdf) == 0:
        print(f"Catalog is empty. Table not found: {table_fqn}")
        return 1

    # Find the table
    matches = pdf[pdf["table_fqn"] == table_fqn]

    if len(matches) == 0:
        print(f"Table not found in catalog: {table_fqn}")
        print("\nTo register, run the pipeline or use: koe catalog refresh <table_fqn>")
        return 1

    row = matches.iloc[0]

    print(f"\n{'=' * 60}")
    print(f"Table: {table_fqn}")
    print(f"{'=' * 60}")

    print(f"\n  Layer:            {row['layer']}")
    print(f"  Dataset:          {row['dataset']}")
    print(f"  Table Name:       {row['table_name']}")
    print(f"  Delta Path:       {row['delta_path']}")

    if row.get("spec_path"):
        print(f"  Spec Path:        {row['spec_path']}")

    print(f"\n  Schema Hash:      {row.get('schema_hash') or '-'}")
    print(f"  Pipeline Version: {row.get('pipeline_version') or '-'}")

    # Catalog stats
    print(f"\n  Catalog Stats:")
    print(f"    Record Count:   {row.get('record_count') or '-'}")
    print(f"    Created:        {row.get('created_at')}")
    print(f"    Updated:        {row.get('updated_at')}")

    # Try to get live stats using delta-rs (still no Spark)
    delta_path = row.get("delta_path")
    if delta_path and Path(delta_path).exists():
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(delta_path)

            # Get metadata
            metadata = dt.metadata()
            print(f"\n  Live Stats (delta-rs):")
            print(f"    Name:           {metadata.name or '-'}")
            print(f"    Description:    {metadata.description or '-'}")
            print(f"    Partitions:     {metadata.partition_columns or []}")

            # Count files (proxy for size) - API varies by version
            try:
                files = dt.file_uris()
                print(f"    Data Files:     {len(files)}")
            except AttributeError:
                # Older API or different method
                pass

            # Schema from delta-rs
            schema = dt.schema()
            print(f"    Columns:        {len(schema.fields)}")

        except Exception as e:
            print(f"\n  Live Stats: Error reading table ({e})")

    if row.get("description"):
        print(f"\n  Description:")
        print(f"    {row['description']}")

    print()
    return 0


def catalog_refresh(table_fqn: str) -> int:
    """Refresh table metadata from live table (requires Spark for schema hash)."""
    from modules.data_engineering.common.paths import paths

    # Parse FQN
    parts = table_fqn.split(".")
    if len(parts) != 3:
        print(f"Invalid table FQN: {table_fqn}")
        print("Expected format: layer.dataset.table_name (e.g., bronze.jsut.utterances)")
        return 1

    layer, dataset, table_name = parts

    # Determine delta path
    layer_path = getattr(paths, layer, None)
    if layer_path is None:
        print(f"Unknown layer: {layer}")
        return 1

    delta_path = layer_path / dataset / table_name

    if not delta_path.exists():
        print(f"Delta table not found: {delta_path}")
        return 1

    print(f"Refreshing catalog entry for {table_fqn}...")
    print("  (This requires Spark for schema hash computation)")

    # Use Spark for the actual registration (needed for schema hash)
    from modules.data_engineering.common.catalog import register_table
    from modules.data_engineering.common.spark import get_spark

    spark = get_spark()
    df = spark.read.format("delta").load(str(delta_path))

    result = register_table(
        layer=layer,
        dataset=dataset,
        table_name=table_name,
        delta_path=delta_path,
        schema=df.schema,
        record_count=df.count(),
        spark=spark,
    )

    print(f"  Action: {result['action']}")
    print(f"  Schema Hash: {result['schema_hash']}")
    print(f"Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
