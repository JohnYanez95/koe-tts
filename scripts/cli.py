"""
Unified CLI for koe-tts.

Usage:
    koe datasets                # List available datasets
    koe ingest jsut
    koe bronze jsut
    koe silver jsut
    koe gold jsut
    koe build jsut              # Full pipeline: ingest -> bronze -> silver -> gold

    koe cache create jsut       # Create training cache snapshot
    koe train vits jsut         # Train a model

    koe synth jsut -r <run_id> --text "こんにちは"
    koe eval-multispeaker <run_id>
    koe probe-speaker <run_id>

    koe registry list-models
    koe registry promote tts-ja-vits

    koe label pull --batch-size 100
    koe label commit --batch-id batch_001
"""

import sys
from pathlib import Path
from typing import Optional

import typer

from modules.data_engineering.common.paths import paths

__version__ = "0.2.0"

# =============================================================================
# Dataset Discovery
# =============================================================================

def available_datasets() -> list[str]:
    """
    List available datasets from configs/datasets/*.yaml.

    This is the single source of truth for dataset names.
    """
    root = Path(__file__).resolve().parents[1]
    ds_dir = root / "configs" / "datasets"
    if not ds_dir.exists():
        # Fallback: known datasets
        return ["jsut", "jvs", "common_voice"]
    return sorted(p.stem for p in ds_dir.glob("*.yaml"))


DATASET_HELP = "Dataset name. Run `koe datasets` to list available datasets."


def _handle_training_result(result: dict) -> None:
    """
    Handle training result, exiting appropriately for different outcomes.

    - emergency_stop / thermal_shutdown: controlled exit, checkpoint saved
    - best_val_loss == inf: training failed, exit with error
    - otherwise: success
    """
    status = result.get("status")

    # Controlled shutdown - checkpoint saved, exit cleanly
    if status in ("emergency_stop", "thermal_shutdown"):
        typer.echo(f"Training terminated: {status}")
        if reason := result.get("reason"):
            typer.echo(f"Reason: {reason}")
        if ckpt := result.get("checkpoint"):
            typer.echo(f"Checkpoint: {ckpt}")
        return

    # Normal completion - check if training made progress
    if result.get("best_val_loss", float("inf")) == float("inf"):
        raise typer.Exit(1)


def version_callback(value: bool):
    if value:
        print(f"koe-tts {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="koe",
    help="koe-tts: Japanese TTS training pipeline",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True,
        help="Show version and exit"
    ),
):
    """koe-tts: Japanese TTS training pipeline."""
    pass


# =============================================================================
# Dataset Discovery Command
# =============================================================================


@app.command()
def datasets():
    """
    List available datasets.

    Datasets are defined in configs/datasets/*.yaml and determine
    which corpora can be processed through the pipeline.

    Examples:
        koe datasets
    """
    ds = available_datasets()
    if not ds:
        typer.echo("No datasets found. Check configs/datasets/ directory.")
        raise typer.Exit(code=1)
    typer.echo("Available datasets:")
    for name in ds:
        typer.echo(f"  {name}")


# =============================================================================
# Bootstrap (Environment Setup)
# =============================================================================


@app.command()
def bootstrap(
    project: str = typer.Option("koe-tts", "--project", "-p", help="Project name in Vault"),
    show: bool = typer.Option(False, "--show", "-s", help="Show secrets even on TTY"),
):
    """
    Output environment variables for shell eval.

    Intended use:
        eval "$(koe bootstrap)"
        source <(koe bootstrap)

    Security notes:
    - If stdout is a TTY, secrets are masked unless --show is set
    - When piped (eval/source), real secrets are emitted
    - All values are shell-escaped to prevent injection

    Examples:
        eval "$(koe bootstrap)"              # Standard usage
        koe bootstrap --show                 # Debug: show secrets on TTY
        koe bootstrap --project other-proj   # Use different Vault path
    """
    import os
    import shlex
    import subprocess

    def export(name: str, value: str) -> str:
        """Generate safe export statement with proper shell escaping."""
        return f"export {name}={shlex.quote(value)}"

    vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
    exports: list[str] = []

    # If running interactively (TTY), mask secrets unless --show is set
    # When piped to eval/source, stdout is not a TTY so secrets are revealed
    reveal_secrets = show or (not sys.stdout.isatty())

    # Check Vault availability
    vault_available = False
    try:
        result = subprocess.run(
            ["vault", "status", "-address", vault_addr],
            capture_output=True,
            timeout=5,
        )
        vault_available = (result.returncode == 0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if vault_available:
        # Use --project to determine Vault path
        minio_path = f"secret/{project}/minio"

        try:
            r_user = subprocess.run(
                ["vault", "kv", "get", "-address", vault_addr, "-field=user", minio_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r_user.returncode == 0:
                exports.append(export("MINIO_ROOT_USER", r_user.stdout.strip()))

            r_pass = subprocess.run(
                ["vault", "kv", "get", "-address", vault_addr, "-field=password", minio_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r_pass.returncode == 0:
                pw = r_pass.stdout.strip()
                if reveal_secrets:
                    exports.append(export("MINIO_ROOT_PASSWORD", pw))
                else:
                    exports.append("export MINIO_ROOT_PASSWORD='***'")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("# Warning: Could not read MinIO secrets from Vault", file=sys.stderr)
    else:
        print("# Warning: Vault not available, using env defaults", file=sys.stderr)

    # Standard infrastructure config (all safely escaped)
    exports.extend([
        export("VAULT_ADDR", vault_addr),
        export("METASTORE_URI", "thrift://localhost:9083"),
        export("MINIO_ENDPOINT", "http://localhost:9000"),
        export("FORGE_LAKE_ROOT_S3A", "s3a://forge/lake"),
        export("FORGE_LAKE_ROOT_S3", "s3://forge/lake"),
    ])

    # Output to stdout (warnings already went to stderr)
    for line in exports:
        print(line)


# =============================================================================
# Data Engineering Commands
# =============================================================================


@app.command()
def ingest(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-extraction even if exists"),
):
    """
    Download and extract a dataset into data/ingest/<dataset>.

    This is the first step in the data pipeline. It downloads the raw
    corpus assets and extracts them to local storage.

    Outputs:
        data/ingest/<dataset>/          Audio files and transcripts
        data/assets/<dataset>/          Licenses, READMEs, inventory

    Examples:
        koe ingest jsut
        koe ingest jvs
        koe ingest jsut --force          # Re-download even if exists
    """
    from modules.data_engineering.ingest import cli as ingest_cli

    ingest_cli.main(dataset, force=force)


@app.command()
def bronze(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if table exists"),
):
    """
    Process dataset to bronze layer (raw -> Delta table).

    Bronze is a 1:1 copy of raw data into a queryable Delta table.
    No transformations, just schema enforcement and metadata.

    Requires:
        data/ingest/<dataset>/          From `koe ingest`

    Outputs:
        lake/bronze/<dataset>/          Delta table

    Examples:
        koe bronze jsut
        koe bronze jvs --force           # Rebuild from scratch
    """
    from modules.data_engineering.bronze import cli as bronze_cli

    bronze_cli.main(dataset, force=force)


@app.command()
def silver(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if table exists"),
    val_pct: float = typer.Option(0.10, "--val-pct", help="Validation set fraction"),
    test_pct: float = typer.Option(0.00, "--test-pct", help="Test set fraction"),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic splits"),
    phonemize: bool = typer.Option(False, "--phonemize", help="Generate phonemes with pyopenjtalk"),
):
    """
    Process dataset to silver layer (QC, normalization, splits, phonemes).

    Silver applies quality control, creates train/val/test splits,
    and optionally generates phoneme sequences for training.

    Requires:
        lake/bronze/<dataset>/          From `koe bronze`

    Outputs:
        lake/silver/<dataset>/          Delta table with splits + phonemes

    Key columns added:
        split           train/val/test assignment
        is_trainable    QC pass/fail flag
        phonemes        Phoneme sequence (if --phonemize)

    Examples:
        koe silver jsut
        koe silver jsut --phonemize      # Generate phonemes
        koe silver jsut --val-pct 0.05   # Smaller val set
        koe silver jvs --seed 123        # Different split seed
    """
    from modules.data_engineering.silver import cli as silver_cli

    # Validate split percentages
    train_pct = 1.0 - val_pct - test_pct
    if train_pct < 0:
        print(f"Error: val-pct + test-pct must be <= 1.0 (got {val_pct + test_pct:.6f})")
        raise typer.Exit(1)

    silver_cli.main(
        dataset,
        force=force,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
        phonemize=phonemize,
    )


@app.command()
def gold(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic splits"),
    train_pct: float = typer.Option(0.90, "--train-pct", help="Training set fraction"),
    val_pct: float = typer.Option(0.10, "--val-pct", help="Validation set fraction"),
    test_pct: float = typer.Option(0.00, "--test-pct", help="Test set fraction"),
    snapshot_id: str = typer.Option(None, "--snapshot-id", help="Snapshot ID (auto-generated if not provided)"),
    min_duration: float = typer.Option(0.5, "--min-duration", help="Minimum duration in seconds"),
    max_duration: float = typer.Option(20.0, "--max-duration", help="Maximum duration in seconds"),
    write_delta: bool = typer.Option(True, "--write-delta/--no-write-delta", help="Write Delta table"),
    manifest_out: str = typer.Option(None, "--manifest-out", help="Override manifest output path"),
):
    """
    Build gold snapshot and training manifest from silver.

    Gold is the final data layer before training. It filters by duration,
    exports train/val JSONL manifests, and creates a versioned snapshot.

    Requires:
        lake/silver/<dataset>/          From `koe silver`

    Outputs:
        lake/gold/<dataset>/            Delta table (versioned)
        data/gold/<dataset>/            JSONL manifests (train.jsonl, val.jsonl)

    Examples:
        koe gold jsut
        koe gold jsut --val-pct 0.05 --seed 123
        koe gold jvs --min-duration 1.0 --max-duration 12.0
        koe gold jsut --snapshot-id jsut-experiment-1
    """
    from modules.data_engineering.gold import cli as gold_cli

    # Validate split percentages sum to 1.0
    total_pct = train_pct + val_pct + test_pct
    if abs(total_pct - 1.0) > 1e-6:
        print(f"Error: Split percentages must sum to 1.0 (got {total_pct:.6f})")
        print(f"  train-pct: {train_pct}")
        print(f"  val-pct: {val_pct}")
        print(f"  test-pct: {test_pct}")
        raise typer.Exit(1)

    # Validate each percentage is in [0, 1]
    for name, pct in [("train-pct", train_pct), ("val-pct", val_pct), ("test-pct", test_pct)]:
        if not 0.0 <= pct <= 1.0:
            print(f"Error: {name} must be between 0 and 1 (got {pct})")
            raise typer.Exit(1)

    gold_cli.main(
        dataset,
        snapshot_id=snapshot_id,
        min_duration=min_duration,
        max_duration=max_duration,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
        export_jsonl=True,
        write_delta=write_delta,
        manifest_out=manifest_out,
    )


@app.command()
def build(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip ingest (assumes raw exists)"),
    skip_bronze: bool = typer.Option(False, "--skip-bronze", help="Skip bronze (assumes Delta exists)"),
    skip_silver: bool = typer.Option(False, "--skip-silver", help="Skip silver (assumes Delta exists)"),
):
    """
    Run full data engineering pipeline (ingest -> bronze -> silver -> gold).

    This is the recommended way to prepare a dataset for training.
    It runs all four medallion layers in sequence.

    Layers:
        ingest  Download/extract raw corpus to data/ingest/
        bronze  Raw-to-table normalization (Delta)
        silver  QC + splits + phonemes (Delta)
        gold    Training-ready manifest + snapshot (Delta + JSONL)

    Examples:
        koe build jsut                   # Full pipeline from scratch
        koe build jsut --skip-ingest     # Re-process existing raw data
        koe build jvs --skip-ingest --skip-bronze --skip-silver  # Rebuild gold only
    """
    from modules.data_engineering.pipelines.build_dataset import build_dataset

    build_dataset(
        dataset=dataset,
        skip_ingest=skip_ingest,
        skip_bronze=skip_bronze,
        skip_silver=skip_silver,
    )


# =============================================================================
# Catalog Commands
# =============================================================================

catalog_app = typer.Typer(help="Lakehouse catalog (table discovery)")
app.add_typer(catalog_app, name="catalog")


@catalog_app.command("list")
def catalog_list(
    layer: str = typer.Option(None, "--layer", "-l", help="Filter by layer (bronze, silver, gold)"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Filter by dataset"),
):
    """List tables in the catalog."""
    from modules.data_engineering.catalog.cli import catalog_list as do_list

    do_list(layer=layer, dataset=dataset)


@catalog_app.command("describe")
def catalog_describe(
    table_fqn: str = typer.Argument(..., help="Table name (e.g., bronze.jsut.utterances)"),
):
    """Describe a table in detail."""
    from modules.data_engineering.catalog.cli import catalog_describe as do_describe

    do_describe(table_fqn)


@catalog_app.command("refresh")
def catalog_refresh(
    table_fqn: str = typer.Argument(..., help="Table name (e.g., bronze.jsut.utterances)"),
):
    """Refresh table metadata from live Delta table."""
    from modules.data_engineering.catalog.cli import catalog_refresh as do_refresh

    do_refresh(table_fqn)


@catalog_app.command("hms-register")
def catalog_hms_register(
    layer: str = typer.Argument(..., help="Layer: bronze, silver, or gold"),
    dataset: str = typer.Argument(..., help="Dataset name (e.g., koe)"),
    table: str = typer.Argument(..., help="Table name (e.g., utterances)"),
    location: str = typer.Option(None, "--location", "-l", help="Override S3 location"),
):
    """
    Register a single table with Hive Metastore.

    Creates the database (schema) if it doesn't exist, then registers
    the Delta table as an external table pointing to S3.

    Examples:
        koe catalog hms-register gold koe utterances
        koe catalog hms-register silver koe utterances --location s3a://forge/lake/silver/koe/utterances
    """
    import os
    from modules.data_engineering.common.spark import get_spark

    spark = get_spark()
    lake_root = os.getenv("FORGE_LAKE_ROOT_S3A", "s3a://forge/lake")

    # Schema name follows medallion prefix convention: gold_koe, silver_koe
    schema_name = f"{layer}_{dataset}"

    # Default location if not specified
    if location is None:
        location = f"{lake_root}/{layer}/{dataset}/{table}"

    print(f"Registering {schema_name}.{table}")
    print(f"  Location: {location}")

    # Create database (schema) if not exists
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema_name}")

    # Register external Delta table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table}
        USING DELTA
        LOCATION '{location}'
    """)

    # Verify
    tables = spark.sql(f"SHOW TABLES IN {schema_name}").collect()
    print(f"  Tables in {schema_name}: {[t.tableName for t in tables]}")
    print("Done.")


@catalog_app.command("hms-refresh")
def catalog_hms_refresh(
    dataset: str = typer.Option(None, "--dataset", "-d", help="Filter to specific dataset"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be registered"),
):
    """
    Scan lake and register all Delta tables with Hive Metastore.

    Walks the S3 bucket structure and registers each Delta table found.
    Creates databases (schemas) as needed: gold_koe, silver_koe, bronze_koe.

    Examples:
        koe catalog hms-refresh                    # Register all tables
        koe catalog hms-refresh --dataset koe      # Only koe dataset
        koe catalog hms-refresh --dry-run          # Preview only
    """
    import os
    from modules.data_engineering.common.spark import get_spark
    from modules.data_engineering.common.paths import paths

    lake_root = os.getenv("FORGE_LAKE_ROOT_S3A", "s3a://forge/lake")

    # For now, scan local filesystem to discover tables
    # TODO: Use boto3/mc to scan S3 directly when running against real MinIO
    layers = ["bronze", "silver", "gold"]
    tables_found = []

    for layer in layers:
        layer_path = paths.lake / layer
        if not layer_path.exists():
            continue

        for dataset_dir in layer_path.iterdir():
            if not dataset_dir.is_dir():
                continue

            if dataset and dataset_dir.name != dataset:
                continue

            for table_dir in dataset_dir.iterdir():
                # Check if it's a Delta table (has _delta_log)
                if (table_dir / "_delta_log").exists():
                    tables_found.append({
                        "layer": layer,
                        "dataset": dataset_dir.name,
                        "table": table_dir.name,
                        "location": f"{lake_root}/{layer}/{dataset_dir.name}/{table_dir.name}",
                    })

    if not tables_found:
        print("No Delta tables found in lake.")
        return

    print(f"Found {len(tables_found)} Delta tables:\n")

    if dry_run:
        for t in tables_found:
            schema_name = f"{t['layer']}_{t['dataset']}"
            print(f"  Would register: {schema_name}.{t['table']}")
            print(f"    Location: {t['location']}")
        print("\n(Dry run - no changes made)")
        return

    # Register with HMS
    spark = get_spark()

    for t in tables_found:
        schema_name = f"{t['layer']}_{t['dataset']}"
        table_name = t["table"]
        location = t["location"]

        print(f"  Registering {schema_name}.{table_name}...")

        # Create database if not exists
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema_name}")

        # Register external Delta table
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.{table_name}
            USING DELTA
            LOCATION '{location}'
        """)

    print(f"\nRegistered {len(tables_found)} tables with Hive Metastore.")
    print("\nVerify with: koe catalog hms-list")


@catalog_app.command("hms-list")
def catalog_hms_list():
    """
    List all databases and tables registered in Hive Metastore.

    Shows the current state of HMS - what Spark can resolve by name.

    Examples:
        koe catalog hms-list
    """
    from modules.data_engineering.common.spark import get_spark

    spark = get_spark()

    # List databases
    databases = spark.sql("SHOW DATABASES").collect()

    print("Hive Metastore Contents:\n")

    for db in databases:
        db_name = db.namespace
        if db_name == "default":
            continue  # Skip default database

        tables = spark.sql(f"SHOW TABLES IN {db_name}").collect()
        if not tables:
            continue

        print(f"  {db_name}/")
        for t in tables:
            print(f"    └── {t.tableName}")

    print()


@catalog_app.command("sync-duckdb")
def catalog_sync_duckdb(
    output: str = typer.Option(".forge/catalog.duckdb", "--output", "-o", help="Output DuckDB file"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Filter to specific dataset"),
):
    """
    Create DuckDB views from lake structure.

    Generates a DuckDB catalog file with views pointing to S3 paths.
    Credentials are set at connection time, not stored in the file.

    Examples:
        koe catalog sync-duckdb
        koe catalog sync-duckdb --output my-catalog.duckdb
        koe catalog sync-duckdb --dataset koe
    """
    import os
    from pathlib import Path

    import duckdb

    from modules.data_engineering.common.paths import paths

    lake_root = os.getenv("FORGE_LAKE_ROOT_S3", "s3://forge/lake")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to start fresh
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))

    # Load extensions (stored in the file)
    conn.execute("INSTALL delta; LOAD delta;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    # Scan local filesystem for tables
    layers = ["bronze", "silver", "gold"]
    views_created = 0

    for layer in layers:
        layer_path = paths.lake / layer
        if not layer_path.exists():
            continue

        for dataset_dir in layer_path.iterdir():
            if not dataset_dir.is_dir():
                continue

            if dataset and dataset_dir.name != dataset:
                continue

            schema_name = f"{layer}_{dataset_dir.name}"

            # Create schema
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

            for table_dir in dataset_dir.iterdir():
                if (table_dir / "_delta_log").exists():
                    table_name = table_dir.name
                    s3_path = f"{lake_root}/{layer}/{dataset_dir.name}/{table_name}"

                    # Create view
                    conn.execute(f"""
                        CREATE OR REPLACE VIEW {schema_name}.{table_name} AS
                        SELECT * FROM delta_scan('{s3_path}')
                    """)

                    print(f"  Created view: {schema_name}.{table_name}")
                    views_created += 1

    conn.close()

    print(f"\nCreated {views_created} views in {output_path}")
    print("\nUsage:")
    print(f"  duckdb {output_path}")
    print("  > SELECT * FROM gold_koe.utterances LIMIT 10;")


# =============================================================================
# Query Commands (DuckDB + Unity Catalog)
# =============================================================================

query_app = typer.Typer(help="Query Delta tables via DuckDB")
app.add_typer(query_app, name="query")


@query_app.command("sql")
def query_sql(
    sql: str = typer.Argument(..., help="SQL query to execute"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    limit: int = typer.Option(None, "--limit", "-n", help="Limit rows (applied if not in query)"),
):
    """
    Run SQL query against Delta tables.

    Uses DuckDB with delta_scan() for direct access.

    Examples:
        koe query sql "SELECT COUNT(*) FROM delta_scan('/lake/silver/jsut/utterances')"
        koe query sql "SELECT * FROM delta_scan('/lake/gold/jsut/utterances') WHERE split='train' LIMIT 10"
    """
    from modules.data_engineering.common.duckdb_client import query

    result = query(sql)

    if limit and len(result) > limit:
        result = result.head(limit)

    if format == "json":
        print(result.to_json(orient="records", indent=2))
    elif format == "csv":
        print(result.to_csv(index=False))
    else:
        print(result.to_string())


@query_app.command("tables")
def query_tables(
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """
    List all Delta tables in the lake.

    Scans the lake directory structure to find tables.
    """
    import pandas as pd

    from modules.data_engineering.common.duckdb_client import list_tables

    tables = list_tables()

    if not tables:
        print("No tables found in lake.")
        return

    if format == "json":
        import json
        print(json.dumps(tables, indent=2))
    else:
        df = pd.DataFrame(tables)
        print(df.to_string(index=False))


@query_app.command("table")
def query_table(
    layer: str = typer.Argument(..., help="Layer: bronze, silver, or gold"),
    dataset: str = typer.Argument(..., help="Dataset name"),
    table: str = typer.Argument(..., help="Table name"),
    columns: str = typer.Option("*", "--columns", "-c", help="Column selection"),
    where: str = typer.Option(None, "--where", "-w", help="WHERE clause"),
    limit: int = typer.Option(10, "--limit", "-n", help="Row limit"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
):
    """
    Query a specific table by layer/dataset/table.

    Examples:
        koe query table silver jsut utterances
        koe query table silver jsut utterances --limit 5
        koe query table silver jsut utterances --columns "id,text,duration_sec"
        koe query table gold jsut utterances --where "split='train'" --limit 100
    """
    from modules.data_engineering.common.duckdb_client import query_table as qt

    result = qt(layer, dataset, table, columns=columns, where=where, limit=limit)

    if format == "json":
        print(result.to_json(orient="records", indent=2))
    elif format == "csv":
        print(result.to_csv(index=False))
    else:
        print(result.to_string())


@query_app.command("scan")
def query_scan(
    path: str = typer.Argument(..., help="Path to Delta table"),
    columns: str = typer.Option("*", "--columns", "-c", help="Column selection"),
    limit: int = typer.Option(10, "--limit", "-n", help="Row limit"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
):
    """
    Direct delta_scan on a path (bypasses catalog).

    Examples:
        koe query scan /lake/silver/jsut/utterances
        koe query scan ./lake/bronze/jvs/utterances --limit 5
    """
    from modules.data_engineering.common.duckdb_client import scan_delta

    result = scan_delta(path, columns=columns, limit=limit)

    if format == "json":
        print(result.to_json(orient="records", indent=2))
    elif format == "csv":
        print(result.to_csv(index=False))
    else:
        print(result.to_string())


# =============================================================================
# Segmentation Commands
# =============================================================================

segment_app = typer.Typer(help="Audio segmentation")
app.add_typer(segment_app, name="segment")


# Dataset-specific pause detection defaults
# These are tuned based on corpus characteristics:
# - JSUT: Studio read speech, minimal pauses, needs lower min_pause_ms
# - JVS: 100 speakers, more variation, longer natural pauses
DATASET_PAUSE_DEFAULTS: dict[str, dict] = {
    "jsut": {"min_pause_ms": 50},
    "jvs": {"min_pause_ms": 200},
}

# Global defaults (used when dataset not in DATASET_PAUSE_DEFAULTS)
PAUSE_DEFAULTS = {
    "floor_db": -60.0,
    "margin_db": 8.0,
    "min_pause_ms": 150,
    "merge_gap_ms": 80,
}


def _resolve_pause_config(
    dataset: str,
    floor_db: float | None,
    margin_db: float | None,
    min_pause_ms: int | None,
    merge_gap_ms: int | None,
) -> tuple[dict, dict[str, str]]:
    """
    Resolve effective pause detection config with dataset defaults.

    Returns:
        Tuple of (effective_config, sources) where sources indicates
        where each value came from ("cli", "dataset", "default")
    """
    dataset_defaults = DATASET_PAUSE_DEFAULTS.get(dataset, {})

    def resolve(name: str, cli_value, default_type):
        if cli_value is not None:
            return cli_value, "cli"
        if name in dataset_defaults:
            return default_type(dataset_defaults[name]), "dataset"
        return default_type(PAUSE_DEFAULTS[name]), "default"

    effective = {}
    sources = {}

    effective["floor_db"], sources["floor_db"] = resolve("floor_db", floor_db, float)
    effective["margin_db"], sources["margin_db"] = resolve("margin_db", margin_db, float)
    effective["min_pause_ms"], sources["min_pause_ms"] = resolve("min_pause_ms", min_pause_ms, int)
    effective["merge_gap_ms"], sources["merge_gap_ms"] = resolve("merge_gap_ms", merge_gap_ms, int)

    return effective, sources


@segment_app.command("auto")
def segment_auto(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    # Adaptive threshold (None = use dataset/global default)
    floor_db: float = typer.Option(None, "--floor-db", help="Absolute floor (dB)"),
    margin_db: float = typer.Option(None, "--margin-db", help="Margin below p10 (dB)"),
    # Manual override
    threshold_db: float = typer.Option(None, "--threshold-db", help="Manual threshold (disables adaptive)"),
    # Region filtering (None = use dataset/global default)
    min_pause_ms: int = typer.Option(None, "--min-pause-ms", help="Min pause duration (ms)"),
    merge_gap_ms: int = typer.Option(None, "--merge-gap-ms", help="Merge regions closer than this (ms)"),
    # Iteration
    limit: int = typer.Option(None, "--limit", "-n", help="Process only N utterances"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print stats only, don't write"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
    snapshot: str = typer.Option(None, "--snapshot", "-s", help="Source gold manifest snapshot ID"),
):
    """
    Detect pause breakpoints in audio.

    Uses adaptive RMS thresholding to find silence regions, then extracts
    breakpoint midpoints for segmentation.

    Dataset-specific defaults are applied automatically:
        - jsut: min_pause_ms=50 (studio read speech)
        - jvs: min_pause_ms=200 (multi-speaker variation)

    CLI overrides always take precedence.

    Outputs:
        lake/silver/{dataset}/segment_breaks    Delta table with breakpoints

    Examples:
        koe segment auto jsut                    # Uses jsut defaults
        koe segment auto jvs                     # Uses jvs defaults
        koe segment auto jsut --min-pause-ms 100 # Override default
        koe segment auto jsut --dry-run         # Preview stats only
    """
    from modules.data_engineering.common.audio import PauseDetectionConfig
    from modules.data_engineering.silver.segments import build_segment_breaks

    # Resolve effective config with dataset defaults
    effective, sources = _resolve_pause_config(
        dataset, floor_db, margin_db, min_pause_ms, merge_gap_ms
    )

    # Print resolved config
    def fmt_source(name: str) -> str:
        src = sources[name]
        if src == "cli":
            return ""
        elif src == "dataset":
            return f" ({dataset} default)"
        else:
            return " (default)"

    print("\nUsing pause config (resolved):")
    print(f"  min_pause_ms={effective['min_pause_ms']}{fmt_source('min_pause_ms')}")
    print(f"  margin_db={effective['margin_db']}{fmt_source('margin_db')}")
    print(f"  floor_db={effective['floor_db']}{fmt_source('floor_db')}")
    print(f"  merge_gap_ms={effective['merge_gap_ms']}{fmt_source('merge_gap_ms')}")

    # Build config
    config = PauseDetectionConfig(
        floor_db=effective["floor_db"],
        margin_db=effective["margin_db"],
        min_pause_ms=effective["min_pause_ms"],
        merge_gap_ms=effective["merge_gap_ms"],
    )

    # Manual threshold overrides adaptive
    if threshold_db is not None:
        config.adaptive = False
        config.silence_threshold_db = threshold_db

    result = build_segment_breaks(
        dataset=dataset,
        config=config,
        source_snapshot=snapshot,
        limit=limit,
        dry_run=dry_run,
        force=force,
    )

    if result["status"] != "success":
        raise typer.Exit(1)


@segment_app.command("build")
def segment_build(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    source_snapshot: str = typer.Option(None, "--source", "-s", help="Source gold manifest snapshot"),
    min_ms: int = typer.Option(800, "--min-ms", help="Min segment duration (ms)"),
    max_ms: int = typer.Option(6000, "--max-ms", help="Max segment duration (ms)"),
    target_ms: int = typer.Option(3000, "--target-ms", help="Target segment duration (ms)"),
    min_lead_ms: int = typer.Option(250, "--min-lead-ms", help="Min lead before first breakpoint (ms)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print stats only, don't write"),
):
    """
    Build segments manifest from breakpoints.

    Uses greedy algorithm to split utterances at pause breakpoints,
    targeting a specific segment duration.

    Requires:
        lake/silver/{dataset}/segment_breaks    From `koe segment auto`

    Outputs:
        lake/gold/{dataset}/segments/           JSONL segment manifests

    Note: Tier 1 segments are always unlabeled (no text/phonemes).
    Use for audio-only training (HiFi-GAN) only.

    Examples:
        koe segment build jsut                  # Build with defaults
        koe segment build jsut --dry-run       # Preview stats
        koe segment build jsut --max-ms 4000   # Shorter segments
    """
    from modules.data_engineering.gold.segments import SegmentConfig, build_gold_segments

    config = SegmentConfig(
        min_segment_ms=min_ms,
        max_segment_ms=max_ms,
        target_segment_ms=target_ms,
        min_lead_ms=min_lead_ms,
    )

    result = build_gold_segments(
        dataset=dataset,
        config=config,
        source_snapshot=source_snapshot,
        dry_run=dry_run,
    )

    if result["status"] != "success":
        raise typer.Exit(1)


@segment_app.command("list")
def segment_list(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
):
    """
    Show segment inventory and distribution stats.

    Displays:
        - Silver segment breaks table status
        - Gold segment manifests inventory

    Examples:
        koe segment list jsut
    """
    from modules.data_engineering.silver.segments import list_segment_breaks
    from modules.data_engineering.gold.segments import list_gold_segments

    print(f"\n{'='*60}")
    print(f"Segment Inventory - {dataset}")
    print(f"{'='*60}")

    # Silver breaks
    print("\n[Silver] Segment Breaks:")
    breaks_info = list_segment_breaks(dataset)
    if breaks_info["status"] == "not_found":
        print(f"  Not found: {breaks_info['path']}")
        print(f"  Run: koe segment auto {dataset}")
    else:
        print(f"  Path: {breaks_info['path']}")
        print(f"  Total utterances: {breaks_info['total_utterances']:,}")
        print(f"  With pauses: {breaks_info['with_pause']:,}")
        print(f"  With breakpoints: {breaks_info['with_breakpoints']:,}")
        print(f"  Method: {breaks_info['method']} ({breaks_info['params_hash']})")
        if breaks_info.get("split_distribution"):
            print("  Splits:", breaks_info["split_distribution"])

    # Gold segments
    print("\n[Gold] Segment Manifests:")
    gold_info = list_gold_segments(dataset)
    if gold_info["status"] == "not_found":
        print(f"  Not found: {gold_info['path']}")
        print(f"  Run: koe segment build {dataset}")
    else:
        print(f"  Path: {gold_info['path']}")
        manifests = gold_info.get("manifests", [])
        if not manifests:
            print("  No manifests found")
        else:
            for m in manifests:
                print(f"  - {m['snapshot_id']}: {m['n_segments']:,} segments")

    print()


# =============================================================================
# Training Commands
# =============================================================================

train_app = typer.Typer(help="Training commands")
app.add_typer(train_app, name="train")


@train_app.command("run")
def train_run(
    model: str = typer.Argument("vits", help="Model type (vits)"),
    config: str = typer.Option(None, "--config", "-c", help="Config file path"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    run_name: str = typer.Option(None, "--run-name", "-n", help="Run name"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
):
    """Train a TTS model."""
    from modules.training.pipelines.train import main as train_main

    train_main(
        model=model,
        config=config,
        dataset=dataset,
        run_name=run_name,
        resume=resume,
    )


@train_app.command("smoke-test")
def train_smoke_test(
    cache: str = typer.Argument(..., help=DATASET_HELP),
    snapshot: str = typer.Option(None, "--snapshot", "-s", help="Specific snapshot ID"),
    max_utterances: int = typer.Option(100, "--max-utterances", "-n", help="Max utterances to use"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device (cpu, cuda)"),
):
    """
    Run training pipeline smoke test.

    Verifies data loading, collation, and batch creation without
    actually training. Use this to catch data issues early.

    Examples:
        koe train smoke-test jsut
        koe train smoke-test jsut --device cuda
        koe train smoke-test jsut --max-utterances 50 --batch-size 8
    """
    from modules.training.pipelines.smoke_test import SmokeTestConfig, run_smoke_test

    config = SmokeTestConfig(
        max_utterances=max_utterances,
        batch_size=batch_size,
        device=device,
    )

    result = run_smoke_test(
        dataset=cache,
        snapshot_id=snapshot,
        config=config,
    )

    if result["status"] != "passed":
        raise typer.Exit(1)


@train_app.command("baseline")
def train_baseline(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    config: str = typer.Option(
        "configs/training/baseline.yaml",
        "--config", "-c",
        help="Config file path"
    ),
    snapshot: str = typer.Option(None, "--snapshot", "-s", help="Cache snapshot ID"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max_steps"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Override batch_size"),
    lr: float = typer.Option(None, "--lr", help="Override learning rate"),
    no_amp: bool = typer.Option(False, "--no-amp", help="Disable AMP"),
):
    """
    Train baseline mel prediction model.

    Simple encoder-decoder that predicts mel spectrograms from phonemes.
    Use this to validate the training pipeline before VITS.

    Examples:
        koe train baseline jsut
        koe train baseline jsut --max-steps 5000
    """
    from pathlib import Path
    from modules.training.pipelines.train_baseline import train, load_config

    # Load config
    config_path = Path(config)
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        cfg = {
            "run": {"name": "baseline_mel", "seed": 42, "device": "cuda", "amp": True},
            "data": {"batch_size": 16, "num_workers": 4, "max_val_batches": 50},
            "mel": {"sample_rate": 22050, "n_mels": 80, "hop_length": 256},
            "model": {"d_model": 256, "n_conv_layers_enc": 6, "n_conv_layers_dec": 4},
            "optim": {"lr": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0},
            "train": {"max_steps": 20000, "log_every_steps": 50, "val_every_steps": 1000, "save_every_steps": 1000},
        }

    # Apply overrides
    if max_steps:
        cfg["train"]["max_steps"] = max_steps
    if batch_size:
        cfg["data"]["batch_size"] = batch_size
    if lr:
        cfg["optim"]["lr"] = lr
    if no_amp:
        cfg["run"]["amp"] = False

    # Train
    result = train(
        config=cfg,
        dataset=dataset,
        snapshot_id=snapshot,
        resume_path=Path(resume) if resume else None,
        output_dir=Path(output_dir) if output_dir else None,
    )

    _handle_training_result(result)


@train_app.command("duration")
def train_duration(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    config: str = typer.Option(
        "configs/training/duration.yaml",
        "--config", "-c",
        help="Config file path"
    ),
    snapshot: str = typer.Option(None, "--snapshot", "-s", help="Cache snapshot ID"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max_steps"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Override batch_size"),
    lr: float = typer.Option(None, "--lr", help="Override learning rate"),
    no_amp: bool = typer.Option(False, "--no-amp", help="Disable AMP"),
):
    """
    Train duration prediction model.

    Baseline model extended with duration prediction head.
    Intermediate step before VITS.

    Examples:
        koe train duration jsut
        koe train duration jsut --max-steps 5000
    """
    from pathlib import Path
    from modules.training.pipelines.train_duration import train, load_config

    # Load config
    config_path = Path(config)
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        cfg = {
            "run": {"name": "duration", "seed": 42, "device": "cuda", "amp": True},
            "data": {"batch_size": 16, "num_workers": 4, "max_val_batches": 50},
            "mel": {"sample_rate": 22050, "n_mels": 80, "hop_length": 256},
            "model": {
                "d_model": 256, "n_conv_layers_enc": 6, "n_conv_layers_dec": 4,
                "dur_hidden": 256, "dur_kernel": 3, "dur_layers": 2, "dur_loss_weight": 1.0,
            },
            "optim": {"lr": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0},
            "train": {"max_steps": 20000, "log_every_steps": 50, "val_every_steps": 1000, "save_every_steps": 1000},
        }

    # Apply overrides
    if max_steps:
        cfg["train"]["max_steps"] = max_steps
    if batch_size:
        cfg["data"]["batch_size"] = batch_size
    if lr:
        cfg["optim"]["lr"] = lr
    if no_amp:
        cfg["run"]["amp"] = False

    # Train
    result = train(
        config=cfg,
        dataset=dataset,
        snapshot_id=snapshot,
        resume_path=Path(resume) if resume else None,
        output_dir=Path(output_dir) if output_dir else None,
    )

    _handle_training_result(result)


@train_app.command("vits")
def train_vits(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    stage: str = typer.Option("core", "--stage", "-S", help="Training stage: core or gan"),
    config: str = typer.Option(
        None,
        "--config", "-c",
        help="Config file path (default: vits_core.yaml or vits_gan.yaml based on stage)"
    ),
    snapshot: str = typer.Option(None, "--snapshot", "-s", help="Cache snapshot ID"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max_steps"),
    save_every_steps: int = typer.Option(None, "--save-every-steps", help="Override save_every_steps (checkpoint cadence)"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Override batch_size"),
    lr: float = typer.Option(None, "--lr", help="Override learning rate"),
    no_amp: bool = typer.Option(False, "--no-amp", help="Disable AMP"),
):
    """
    Train VITS model.

    Stages:
        core    Reconstruction-only training (mel + KL + duration losses).
                Stable, fast iteration. Use this first.
        gan     Enables MPD/MSD discriminators after disc_start_step.
                Adds adversarial + feature matching losses for audio realism.
                Use after core is stable.

    Requires:
        data/cache/<dataset>/           From `koe cache create`

    Outputs:
        runs/<run_id>/                  Checkpoints, logs, config

    Examples:
        koe train vits jsut --stage core --max-steps 10000
        koe train vits jsut --stage gan --resume runs/.../checkpoints/best.pt
        koe train vits jvs --stage core --batch-size 8 --lr 1e-4
    """
    from pathlib import Path
    from modules.training.pipelines.train_vits import train, train_gan, load_config

    # Select config based on stage if not specified
    if config is None:
        if stage == "gan":
            config_path = Path("configs/training/vits_gan.yaml")
        else:
            config_path = Path("configs/training/vits_core.yaml")
    else:
        config_path = Path(config)

    if config_path.exists():
        cfg = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        cfg = {
            "run": {"name": f"vits_{stage}", "seed": 42, "device": "cuda", "amp": True},
            "data": {"batch_size": 16, "num_workers": 4, "max_val_batches": 50},
            "mel": {"sample_rate": 22050, "n_mels": 80, "hop_length": 256},
            "model": {"latent_dim": 192},
            "loss": {"kl_weight": 1.0, "dur_weight": 1.0},
            "optim": {"lr": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0},
            "train": {"max_steps": 100000, "log_every_steps": 100, "val_every_steps": 2000, "save_every_steps": 5000},
        }

    # Apply overrides
    if max_steps:
        cfg["train"]["max_steps"] = max_steps
    if save_every_steps:
        old_val = cfg["train"].get("save_every_steps", 5000)
        cfg["train"]["save_every_steps"] = save_every_steps
        print(f"⚠️  Save cadence overridden by CLI: {save_every_steps} (config was {old_val})")
    if batch_size:
        cfg["data"]["batch_size"] = batch_size
    if lr:
        if stage == "gan":
            cfg["optim"]["lr_g"] = lr
        else:
            cfg["optim"]["lr"] = lr
    if no_amp:
        cfg["run"]["amp"] = False

    # Select training function based on stage
    if stage == "gan":
        result = train_gan(
            config=cfg,
            dataset=dataset,
            snapshot_id=snapshot,
            resume_path=Path(resume) if resume else None,
            output_dir=Path(output_dir) if output_dir else None,
        )
    else:
        result = train(
            config=cfg,
            dataset=dataset,
            snapshot_id=snapshot,
            resume_path=Path(resume) if resume else None,
            output_dir=Path(output_dir) if output_dir else None,
        )

    _handle_training_result(result)


@train_app.command("eval")
def train_eval(
    model_type: str = typer.Argument("baseline", help="Model type (baseline, duration, vits)"),
    dataset: str = typer.Argument(..., help="Dataset name"),
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID (directory name)"),
    ckpt: str = typer.Option("best.pt", "--ckpt", "-c", help="Checkpoint name"),
    mode: str = typer.Option("teacher", "--mode", "-m", help="Eval mode: teacher, free, or inference"),
    n_samples: int = typer.Option(20, "--n-samples", "-n", help="Number of samples to evaluate"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for sample selection"),
    write_audio: bool = typer.Option(False, "--write-audio", "-a", help="Write audio (pred + target for A/B)"),
    no_target_audio: bool = typer.Option(False, "--no-target-audio", help="Don't write target audio (pred only)"),
    no_mels: bool = typer.Option(False, "--no-mels", help="Don't write mel files"),
    duration_scale: float = typer.Option(1.0, "--duration-scale", help="Duration scale (inference/free mode)"),
    noise_scale: float = typer.Option(0.667, "--noise-scale", help="Noise scale (vits inference mode)"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
):
    """Evaluate a trained model on val subset."""
    # Find run directory (shared logic)
    run_dir = paths.runs / run_id
    if not run_dir.exists():
        candidates = list(paths.runs.glob(f"{dataset}*{run_id}*"))
        if candidates:
            run_dir = candidates[0]
        else:
            candidates = list(paths.runs.glob(f"*{run_id}*"))
            if candidates:
                run_dir = candidates[0]
            else:
                print(f"Run not found: {run_id}")
                available = [p.name for p in paths.runs.glob("*") if p.is_dir()]
                if available:
                    print(f"Available runs: {available[:10]}")
                raise typer.Exit(1)

    # Determine target audio behavior
    write_target_audio = None
    if no_target_audio:
        write_target_audio = False

    if model_type == "baseline":
        from modules.training.pipelines.eval_baseline import evaluate

        result = evaluate(
            run_dir=run_dir,
            checkpoint_name=ckpt,
            n_samples=n_samples,
            seed=seed,
            write_mels=not no_mels,
            write_audio=write_audio,
            write_target_audio=write_target_audio,
            device=device,
        )

        if result["n_samples"] == 0:
            raise typer.Exit(1)

    elif model_type == "duration":
        from modules.training.pipelines.eval_duration import evaluate

        if mode not in ("teacher", "free"):
            print(f"Invalid mode: {mode}. Use 'teacher' or 'free'.")
            raise typer.Exit(1)

        result = evaluate(
            run_dir=run_dir,
            checkpoint_name=ckpt,
            mode=mode,
            n_samples=n_samples,
            seed=seed,
            write_mels=not no_mels,
            write_audio=write_audio,
            write_target_audio=write_target_audio,
            duration_scale=duration_scale,
            device=device,
        )

        if result["n_samples"] == 0:
            raise typer.Exit(1)

    elif model_type == "vits":
        from modules.training.pipelines.eval_vits import evaluate

        if mode not in ("teacher", "inference"):
            print(f"Invalid mode for vits: {mode}. Use 'teacher' or 'inference'.")
            raise typer.Exit(1)

        result = evaluate(
            run_dir=run_dir,
            checkpoint_name=ckpt,
            mode=mode,
            n_samples=n_samples,
            seed=seed,
            write_mels=not no_mels,
            write_audio=write_audio,
            write_target_audio=write_target_audio,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
            device=device,
        )

        if result["n_samples"] == 0:
            raise typer.Exit(1)

    else:
        print(f"Unknown model type: {model_type}")
        print("Available: baseline, duration, vits")
        raise typer.Exit(1)


@train_app.command("compare")
def train_compare(
    run_a: str = typer.Option(..., "--a", "-a", help="Run A (baseline)"),
    run_b: str = typer.Option(..., "--b", "-b", help="Run B (new)"),
    eval_tag: str = typer.Option(None, "--eval-tag", "-e", help="Specific eval tag to compare"),
    gate: bool = typer.Option(False, "--gate", "-g", help="Apply regression gates (exit 1 on failure)"),
    mel_l1_max: float = typer.Option(5.0, "--mel-l1-max", help="Max mel_l1 increase % for gate"),
    mel_l2_max: float = typer.Option(5.0, "--mel-l2-max", help="Max mel_l2 increase % for gate"),
    snr_min: float = typer.Option(10.0, "--snr-min", help="Max snr decrease % for gate"),
    silence_max: float = typer.Option(3.0, "--silence-max", help="Max silence% increase for gate"),
):
    """Compare eval metrics between two runs."""
    from modules.training.eval import compare_and_print, CompareThresholds

    thresholds = CompareThresholds(
        mel_l1_max_increase=mel_l1_max,
        mel_l2_max_increase=mel_l2_max,
        snr_min_decrease=snr_min,
        silence_max_increase=silence_max,
    )

    result = compare_and_print(
        run_a_id=run_a,
        run_b_id=run_b,
        eval_tag=eval_tag,
        apply_gates=gate,
        thresholds=thresholds,
    )

    if not result.passed:
        raise typer.Exit(1)


# =============================================================================
# Cache Commands
# =============================================================================

cache_app = typer.Typer(help="Training cache management")
app.add_typer(cache_app, name="cache")


@cache_app.command("create")
def cache_create(
    dataset: str = typer.Argument(..., help=DATASET_HELP),
    snapshot_id: str = typer.Option(None, "--snapshot-id", "-s", help="Gold snapshot ID to cache"),
    gold_version: str = typer.Option(None, "--gold-version", help="Alias for --snapshot-id (legacy)"),
):
    """
    Create a local training cache from a gold manifest.

    Copies audio files and metadata to SSD-local storage for fast
    training I/O. This avoids NAS/mount latency during training.

    Requires:
        data/gold/<dataset>/            From `koe gold`

    Outputs:
        data/cache/<dataset>/<snapshot>/    Cached audio + manifest

    Examples:
        koe cache create jsut
        koe cache create jsut --snapshot-id jsut-20260125-abc123
        koe cache list jsut              # See available snapshots
    """
    from modules.training.dataloading.cli import cache_create as create_cache

    create_cache(dataset=dataset, snapshot_id=snapshot_id, gold_version=gold_version)


@cache_app.command("list")
def cache_list(
    dataset: str = typer.Argument(None, help="Dataset name (optional, shows all if omitted)"),
):
    """
    List available cache snapshots.

    Shows cached datasets ready for training. Each snapshot includes
    file count, total duration, and creation date.

    Example output:
        jsut  jsut-20260125-abc123  7670 files  10.9h  created=2026-01-25
              └── latest (symlink)

    Examples:
        koe cache list                   # List all datasets
        koe cache list jsut              # List snapshots for jsut only
    """
    from modules.training.dataloading.cli import cache_list as list_caches

    list_caches(dataset=dataset)


# =============================================================================
# Registry Commands
# =============================================================================

registry_app = typer.Typer(help="MLflow model registry")
app.add_typer(registry_app, name="registry")


@registry_app.command("list-models")
def registry_list_models():
    """List all registered models."""
    from modules.registry.cli import cmd_list_models

    class Args:
        pass

    cmd_list_models(Args())


@registry_app.command("list-versions")
def registry_list_versions(
    model_name: str = typer.Argument(..., help="Model name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max versions to show"),
):
    """List versions of a model."""
    from modules.registry.cli import cmd_list_versions

    class Args:
        pass

    args = Args()
    args.model_name = model_name
    args.limit = limit
    cmd_list_versions(args)


@registry_app.command("promote")
def registry_promote(
    model_name: str = typer.Argument(..., help="Model name"),
    from_alias: str = typer.Option("best", "--from-alias", help="Source alias"),
):
    """Promote a model version to production."""
    from modules.registry.cli import cmd_promote

    class Args:
        pass

    args = Args()
    args.model_name = model_name
    args.from_alias = from_alias
    cmd_promote(args)


# =============================================================================
# Labeler Commands
# =============================================================================

label_app = typer.Typer(help="Labeling workflow")
app.add_typer(label_app, name="label")


@label_app.command("pull")
def label_pull(
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    batch_size: int = typer.Option(100, "--batch-size", "-n", help="Batch size"),
    query: str = typer.Option(None, "--query", "-q", help="Filter query"),
):
    """Pull a batch of utterances for labeling."""
    from modules.labeler.pipelines.pull_batch import pull_batch

    result = pull_batch(dataset=dataset, batch_size=batch_size, query=query)
    if result.get("status") != "success":
        raise typer.Exit(1)


@label_app.command("commit")
def label_commit(
    batch_id: str = typer.Argument(..., help="Batch ID"),
):
    """Commit labeled data back to the lake."""
    from modules.labeler.pipelines.write_labels import write_labels

    write_labels(batch_id=batch_id)


@label_app.command("serve")
def label_serve(
    dataset: str = typer.Argument(..., help="Dataset name"),
    port: int = typer.Option(8081, "--port", "-p", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Server host"),
):
    """Launch the segmentation labeling UI."""
    from modules.labeler.app.backend import run_server

    run_server(host=host, port=port, dataset=dataset)


@label_app.command("heuristic")
def label_heuristic(
    dataset: str = typer.Argument(..., help="Dataset name"),
    limit: int = typer.Option(0, "--limit", "-n", help="Max utterances (0 = all)"),
    min_pause_ms: int = typer.Option(50, "--min-pause-ms", help="Min pause duration (ms)"),
    margin_db: float = typer.Option(8.0, "--margin-db", help="Margin below p10 (dB)"),
    floor_db: float = typer.Option(-60.0, "--floor-db", help="Absolute floor (dB)"),
    merge_gap_ms: int = typer.Option(80, "--merge-gap-ms", help="Merge regions closer than this (ms)"),
    pad_ms: int = typer.Option(30, "--pad-ms", help="Pad regions (ms)"),
    threshold_db: float = typer.Option(None, "--threshold-db", help="Manual threshold (disables adaptive)"),
    name: str = typer.Option(None, "--name", help="Human-readable name for this run"),
):
    """
    Run RMS energy heuristic and write JSONL cache for labeling.

    This bypasses Spark/Delta for fast iteration. Results are cached as
    JSONL in runs/labeling/heuristics/ and automatically picked up by
    the labeling UI.

    Examples:
        koe label heuristic jsut                    # Run on all utterances
        koe label heuristic jsut -n 100             # Quick test on 100
        koe label heuristic jsut --min-pause-ms 80  # Tweak sensitivity
        koe label heuristic jsut --name "baseline"  # Name the run
    """
    from modules.labeler.heuristic import run_heuristic

    result = run_heuristic(
        dataset=dataset,
        limit=limit,
        min_pause_ms=min_pause_ms,
        margin_db=margin_db,
        floor_db=floor_db,
        merge_gap_ms=merge_gap_ms,
        pad_ms=pad_ms,
        threshold_db=threshold_db,
        name=name,
    )

    if result.get("status") != "ok":
        raise typer.Exit(1)


@label_app.command("eval")
def label_eval(
    dataset: str = typer.Argument(..., help="Dataset name"),
):
    """
    Evaluate saved labels against heuristic proposals.

    Reads all labeled sessions for the dataset and reports:
    - MAE of delta_ms (mean absolute error of user corrections)
    - Within-tolerance accuracy (50ms, 100ms)
    - Rejection rate (use_break=false percentage)
    - Drag distance stats

    Use this to measure heuristic quality and decide when to retune.

    Examples:
        koe label eval jsut
    """
    from modules.labeler.heuristic import eval_labels

    result = eval_labels(dataset)
    if result.get("status") == "error":
        print(f"Error: {result.get('message', 'unknown')}")
        raise typer.Exit(1)


@label_app.command("optimize")
def label_optimize(
    dataset: str = typer.Argument(..., help="Dataset name"),
    tau_ms: float = typer.Option(120.0, help="Matching tolerance (ms)"),
    max_iter: int = typer.Option(50, help="DE max iterations"),
    n_folds: int = typer.Option(3, help="K-fold CV splits"),
    name: str = typer.Option(None, help="Name for the optimized run"),
):
    """
    Optimize heuristic parameters against published labels.

    Runs differential evolution over pause detection parameters,
    using Hungarian-matched loss with k-fold CV.

    The optimized config is saved as a new heuristic run,
    immediately selectable in the labeling UI.

    Examples:
        koe label optimize jsut
        koe label optimize jsut --tau-ms 150 --max-iter 100
    """
    from modules.labeler.heuristic import optimize_heuristic

    result = optimize_heuristic(
        dataset,
        tau_ms=tau_ms,
        max_iter=max_iter,
        n_folds=n_folds,
        name=name,
    )
    if result.get("status") != "ok":
        print(f"Optimization failed: {result}")
        raise typer.Exit(1)


# =============================================================================
# Synthesis Commands
# =============================================================================


@app.command()
def synth(
    dataset: str = typer.Argument(..., help="Dataset name (jsut, jvs) - used for run discovery"),
    text: str = typer.Option(None, "--text", "-t", help="Text to synthesize"),
    text_file: str = typer.Option(None, "--text-file", "-f", help="File with text lines"),
    run_id: str = typer.Option(None, "--run-id", "-r", help="Run ID (directory name or partial match)"),
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Checkpoint name (default: best.pt)"),
    output: str = typer.Option(None, "--output", "-o", help="Output path (default: synth_000.wav)"),
    duration_scale: float = typer.Option(1.0, "--duration-scale", help="Duration multiplier (1.0 = normal)"),
    noise_scale: float = typer.Option(0.667, "--noise-scale", help="Noise scale for sampling"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    no_json: bool = typer.Option(False, "--no-json", help="Don't write metadata JSON"),
    speaker: str = typer.Option(None, "--speaker", "-s", help="Speaker index (0) or ID (jvs001, spk00)"),
    target_rms: float = typer.Option(None, "--target-rms", help="Target RMS for loudness normalization (e.g., 0.05)"),
):
    """Synthesize audio from a trained VITS model.

    Examples:
        koe synth jsut --run-id jsut_vits_gan_20260125 --text "こんにちは"
        koe synth jsut -r <run_id> --text "..." -o output.wav
        koe synth jsut -r <run_id> --text-file prompts.txt
        koe synth multi -r <run_id> --text "..." --speaker 5      # By index
        koe synth multi -r <run_id> --text "..." --speaker jvs001 # By ID
        koe synth jsut -r <run_id> --text "..." --target-rms 0.05 # Louder output
    """
    from modules.training.pipelines.synthesize import synthesize as do_synth

    # If no run_id specified, try to find latest run for dataset
    if not run_id and not checkpoint:
        candidates = sorted(
            [p for p in paths.runs.glob(f"{dataset}_vits*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            run_id = candidates[0].name
            print(f"Auto-selected latest run: {run_id}")
        else:
            print(f"No VITS runs found for dataset: {dataset}")
            print(f"Available runs: {[p.name for p in runs_dir.glob('*') if p.is_dir()][:10]}")
            raise typer.Exit(1)

    # Parse speaker: try int first, else keep as string
    speaker_arg = None
    if speaker is not None:
        try:
            speaker_arg = int(speaker)
        except ValueError:
            speaker_arg = speaker  # Keep as string ID

    result = do_synth(
        text=text,
        text_file=text_file,
        run_id=run_id,
        checkpoint=checkpoint,
        output=output,
        duration_scale=duration_scale,
        noise_scale=noise_scale,
        seed=seed,
        device=device,
        write_json=not no_json,
        speaker=speaker_arg,
        target_rms=target_rms,
    )

    if result["status"] != "success":
        raise typer.Exit(1)


@app.command("synth-compare")
def synth_compare(
    run_a: str = typer.Option(..., "--a", "-a", help="Run A (baseline)"),
    run_b: str = typer.Option(..., "--b", "-b", help="Run B (new)"),
    text_file: str = typer.Option(..., "--text-file", "-f", help="Prompts file (one per line)"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    checkpoint_a: str = typer.Option(None, "--ckpt-a", help="Checkpoint for A"),
    checkpoint_b: str = typer.Option(None, "--ckpt-b", help="Checkpoint for B"),
    duration_scale: float = typer.Option(1.0, "--duration-scale", help="Duration multiplier"),
    noise_scale: float = typer.Option(0.667, "--noise-scale", help="Noise scale"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (same for both)"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
):
    """A/B synthesis comparison between two runs.

    Generates identical prompts with identical seeds for fair comparison.
    Outputs HTML page for interactive listening.

    Examples:
        koe synth-compare --a jsut_vits_core_123 --b jsut_vits_gan_456 -f prompts.txt
        koe synth-compare -a run1 -b run2 -f prompts.txt -o compare_out/
    """
    from modules.training.pipelines.synthesize import synth_compare as do_compare

    result = do_compare(
        run_a=run_a,
        run_b=run_b,
        text_file=text_file,
        output_dir=output_dir,
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        duration_scale=duration_scale,
        noise_scale=noise_scale,
        seed=seed,
        device=device,
    )

    if result["status"] != "success":
        raise typer.Exit(1)


@app.command("eval-multispeaker")
def eval_multispeaker(
    run_id: str = typer.Argument(..., help="Run ID to evaluate"),
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Checkpoint name (default: best.pt)"),
    speakers: str = typer.Option(None, "--speakers", help="Comma-separated speaker IDs (auto-selects from checkpoint if omitted)"),
    prompts_file: str = typer.Option(None, "--prompts-file", "-f", help="File with prompts (one per line)"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    duration_scale: float = typer.Option(1.0, "--duration-scale", help="Duration multiplier"),
    noise_scale: float = typer.Option(0.667, "--noise-scale", help="Noise scale"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
):
    """
    Multi-speaker evaluation grid.

    Synthesizes each prompt with each speaker for A/B/C/... comparison.
    Generates HTML grid with audio players for human evaluation.

    Outputs:
        runs/<run>/eval/multispeaker_<seed>/
            index.html              Interactive listening grid
            manifest.json           Results + separation metrics
            <speaker>/*.wav         Audio files per speaker

    Metrics reported:
        mean_inter_speaker_distance     Identity separation proxy
        per_speaker_consistency         Stability across prompts
        silence_pct                     Decoder health indicator

    Examples:
        koe eval-multispeaker <run_id>
        koe eval-multispeaker <run_id> --speakers spk00,jvs001,jvs020
        koe eval-multispeaker <run_id> --prompts-file my_prompts.txt
    """
    from modules.training.pipelines.synthesize import eval_multispeaker as do_eval

    speaker_list = None
    if speakers:
        speaker_list = [s.strip() for s in speakers.split(",")]

    result = do_eval(
        run_id=run_id,
        checkpoint=checkpoint,
        speakers=speaker_list,
        prompts_file=prompts_file,
        output_dir=output_dir,
        duration_scale=duration_scale,
        noise_scale=noise_scale,
        seed=seed,
        device=device,
    )

    if result["status"] != "success":
        raise typer.Exit(1)


@app.command("probe-speaker")
def probe_speaker(
    run_id: str = typer.Argument(..., help="Run ID to probe"),
    probe: str = typer.Option("both", "--probe", "-p", help="Probe type: determinism, difference, both"),
    speaker_a: str = typer.Option("0", "--speaker-a", "-a", help="First speaker ID or index"),
    speaker_b: str = typer.Option("1", "--speaker-b", "-b", help="Second speaker ID or index"),
    text: str = typer.Option("本日は晴天なり。", "--text", "-t", help="Test text"),
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Checkpoint name"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    mel_threshold: float = typer.Option(0.1, "--mel-threshold", help="Mel L1 threshold for difference probe"),
):
    """
    Speaker embedding sanity probes.

    Run these after training to verify multi-speaker conditioning works.
    Both probes must PASS for reliable multi-speaker synthesis.

    Probes:
        determinism     Same speaker + same seed → identical audio
                        Tests reproducibility. MUST pass.
        difference      Different speaker + same seed → different audio
                        Tests that conditioning actually changes output.

    Examples:
        koe probe-speaker <run_id>
        koe probe-speaker <run_id> --speaker-a spk00 --speaker-b jvs001
        koe probe-speaker <run_id> --probe determinism --speaker-a 0
        koe probe-speaker <run_id> --mel-threshold 0.05
    """
    from modules.training.pipelines.synthesize import (
        probe_speaker_determinism,
        probe_speaker_difference,
    )

    results = []

    if probe in ("determinism", "both"):
        print("=" * 50)
        print("PROBE A: Speaker Determinism")
        print("=" * 50)
        result_a = probe_speaker_determinism(
            run_id=run_id,
            speaker=speaker_a,
            text=text,
            checkpoint=checkpoint,
            seed=seed,
            device=device,
        )
        print(f"  Speaker: {result_a['speaker']}")
        print(f"  Max abs diff: {result_a['max_abs_diff']:.2e}")
        print(f"  Status: {result_a['status']}")
        results.append(result_a)
        print()

    if probe in ("difference", "both"):
        print("=" * 50)
        print("PROBE B: Speaker Difference")
        print("=" * 50)
        result_b = probe_speaker_difference(
            run_id=run_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            text=text,
            checkpoint=checkpoint,
            seed=seed,
            device=device,
            mel_l1_threshold=mel_threshold,
        )
        print(f"  Speaker A: {result_b['speaker_a']}")
        print(f"  Speaker B: {result_b['speaker_b']}")
        print(f"  Mel L1: {result_b['mel_l1']:.4f} (threshold: {result_b['mel_l1_threshold']})")
        print(f"  Waveform correlation: {result_b['waveform_correlation']:.4f}")
        print(f"  Status: {result_b['status']}")
        results.append(result_b)
        print()

    # Summary
    all_pass = all(r["status"] == "PASS" for r in results)
    print("=" * 50)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 50)

    if not all_pass:
        raise typer.Exit(1)


# =============================================================================
# Monitor Dashboard
# =============================================================================

@app.command()
def monitor(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID to monitor. Use --list to see available runs.",
    ),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Server host"),
    list_runs: bool = typer.Option(False, "--list", "-l", help="List available runs"),
    latest: bool = typer.Option(False, "--latest", help="Monitor most recent run"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Filter by dataset"),
):
    """
    Start the training monitor dashboard.

    Examples:
        koe monitor --list                    # List all runs
        koe monitor multi_vits_gan_20260125   # Monitor specific run
        koe monitor --latest                  # Monitor most recent run
        koe monitor --latest --dataset multi  # Most recent run for dataset
    """
    runs_dir = paths.runs

    if list_runs:
        # List mode
        from modules.dashboard.backend import discover_runs

        runs = discover_runs(runs_dir)
        if not runs:
            print(f"No training runs found in {runs_dir}")
            raise typer.Exit(1)

        print(f"{'Run ID':<50} {'Dataset':<10} {'Stage':<8} {'Step':>8} {'Status':<10}")
        print("-" * 90)
        for run in runs[:20]:  # Show top 20
            print(f"{run.run_id:<50} {run.dataset:<10} {run.stage:<8} {run.step:>8} {run.status:<10}")
        return

    if latest:
        # Find most recent run
        from modules.dashboard.backend import discover_runs

        runs = discover_runs(runs_dir)
        if dataset:
            runs = [r for r in runs if r.dataset == dataset]
        if not runs:
            print("No runs found" + (f" for dataset '{dataset}'" if dataset else ""))
            raise typer.Exit(1)

        run_id = runs[0].run_id
        print(f"Monitoring latest run: {run_id}")

    if not run_id:
        print("Error: Please specify a run_id or use --list / --latest")
        raise typer.Exit(1)

    # Verify run exists
    run_path = runs_dir / run_id
    if not run_path.exists():
        print(f"Error: Run not found: {run_id}")
        print("Use 'koe monitor --list' to see available runs")
        raise typer.Exit(1)

    # Start server
    print("Starting KOE Monitor Dashboard")
    print(f"  Run: {run_id}")
    print(f"  URL: http://{host}:{port}")
    print(f"  API: http://{host}:{port}/api/runs/{run_id}/meta")
    print()
    print("Press Ctrl+C to stop")
    print()

    from modules.dashboard.backend import run_server
    run_server(host=host, port=port, runs_dir=runs_dir)


# =============================================================================
# Runs Management (archive/clone)
# =============================================================================

runs_app = typer.Typer(help="Manage training runs (archive/clone between local and G:)")
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list(
    archived: bool = typer.Option(False, "--archived", "-a", help="List archived runs (G:)"),
    local: bool = typer.Option(False, "--local", "-l", help="List local runs (WSL)"),
):
    """List training runs in local or archive storage."""
    from modules.data_engineering.common.paths import paths, list_local_runs, list_archived_runs

    if not archived and not local:
        # Show both
        archived = True
        local = True

    if local:
        local_runs = list_local_runs()
        print(f"Local runs ({paths.runs}):")
        if local_runs:
            for r in local_runs:
                print(f"  {r}")
        else:
            print("  (none)")
        print()

    if archived:
        archived_runs = list_archived_runs()
        print(f"Archived runs ({paths.runs_archive}):")
        if archived_runs:
            for r in archived_runs:
                print(f"  {r}")
        else:
            print("  (none)")


@runs_app.command("archive")
def runs_archive(
    run_id: str = typer.Argument(..., help="Run ID to archive"),
    keep_local: bool = typer.Option(False, "--keep", "-k", help="Keep local copy after archiving"),
):
    """Archive a run from local to G: drive."""
    from modules.data_engineering.common.paths import paths, archive_run

    if paths.runs == paths.runs_archive:
        print("Local and archive are same location, nothing to do")
        raise typer.Exit(0)

    local_run = paths.runs / run_id
    if not local_run.exists():
        print(f"Error: Run not found locally: {run_id}")
        print(f"Local runs dir: {paths.runs}")
        raise typer.Exit(1)

    archived = archive_run(run_id, delete_local=not keep_local)
    print(f"Archived to: {archived}")


@runs_app.command("clone")
def runs_clone(
    run_id: str = typer.Argument(..., help="Run ID to clone from archive"),
    checkpoint: str = typer.Option("best.pt", "--ckpt", "-c", help="Checkpoint to clone"),
):
    """Clone a checkpoint from G: archive to local for training."""
    from modules.data_engineering.common.paths import paths, clone_checkpoint_to_local

    if paths.runs == paths.runs_archive:
        print("Local and archive are same location, nothing to do")
        raise typer.Exit(0)

    # Find checkpoint in archive
    archive_run = paths.runs_archive / run_id
    if not archive_run.exists():
        print(f"Error: Run not found in archive: {run_id}")
        print(f"Archive dir: {paths.runs_archive}")
        raise typer.Exit(1)

    ckpt_path = archive_run / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        # Try direct path
        ckpt_path = archive_run / checkpoint
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        available = list((archive_run / "checkpoints").glob("*.pt")) if (archive_run / "checkpoints").exists() else []
        if available:
            print(f"Available: {[p.name for p in available[:10]]}")
        raise typer.Exit(1)

    local_ckpt = clone_checkpoint_to_local(ckpt_path, run_id)
    print(f"Cloned to: {local_ckpt}")


@runs_app.command("archive-all")
def runs_archive_all(
    keep_local: bool = typer.Option(False, "--keep", "-k", help="Keep local copies"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be archived"),
):
    """Archive all local runs to G: drive."""
    from modules.data_engineering.common.paths import paths, list_local_runs, archive_run

    if paths.runs == paths.runs_archive:
        print("Local and archive are same location, nothing to do")
        raise typer.Exit(0)

    local_runs = list_local_runs()
    if not local_runs:
        print("No local runs to archive")
        raise typer.Exit(0)

    print(f"Runs to archive: {len(local_runs)}")
    for r in local_runs:
        if dry_run:
            print(f"  Would archive: {r}")
        else:
            print(f"  Archiving: {r}...")
            archive_run(r, delete_local=not keep_local)

    if dry_run:
        print("\n(Dry run - no changes made)")


if __name__ == "__main__":
    app()
