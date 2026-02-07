"""
Spark session management for the koe-tts lakehouse.

Provides a configured SparkSession with Delta Lake support.
Optionally integrates with Unity Catalog when UC_ENABLED=true.

Usage:
    from modules.data_engineering.common.spark import get_spark

    spark = get_spark()
    df = spark.read.format("delta").load(str(paths.bronze / "utterances"))

    # With UC enabled:
    df = spark.table("koe_tts.silver_jsut.utterances")
"""

import os

from pyspark.sql import SparkSession

from .paths import paths

# Unity Catalog configuration
UC_SERVER_URL = os.getenv("UC_SERVER_URL", "http://localhost:8080")
UC_TOKEN = os.getenv("UC_TOKEN", "")
UC_ENABLED = os.getenv("UC_ENABLED", "false").lower() == "true"
CATALOG_NAME = "koe_tts"

# Global spark session (lazy initialized)
_spark: SparkSession | None = None


def get_spark(
    app_name: str = "koe-tts",
    warehouse_dir: str | None = None,
    local_mode: bool = True,
    extra_configs: dict | None = None,
    use_uc: bool | None = None,
) -> SparkSession:
    """
    Get or create a configured SparkSession with Delta Lake support.

    Args:
        app_name: Spark application name
        warehouse_dir: Spark warehouse directory. Defaults to lake path.
        local_mode: If True, use local[*] master. If False, expects
                    SPARK_MASTER env var or cluster config.
        extra_configs: Additional Spark configs to set
        use_uc: Force UC on/off. None = use UC_ENABLED env var.

    Returns:
        Configured SparkSession with Delta Lake extensions.
        If UC enabled, includes koe_tts catalog via Unity Catalog.
    """
    global _spark

    if _spark is not None:
        return _spark

    if warehouse_dir is None:
        warehouse_dir = str(paths.lake)

    # Delta Lake version must match pyspark version
    # PySpark 4.x requires delta-spark 4.x with Scala 2.13
    DELTA_VERSION = "4.0.0"
    UC_SPARK_VERSION = "0.3.1"
    SCALA_VERSION = "2.13"

    # Determine if UC should be used
    should_use_uc = use_uc if use_uc is not None else UC_ENABLED

    # Build jars list
    jars = [f"io.delta:delta-spark_{SCALA_VERSION}:{DELTA_VERSION}"]
    if should_use_uc:
        jars.append(f"io.unitycatalog:unitycatalog-spark_{SCALA_VERSION}:{UC_SPARK_VERSION}")

    # Use the same Python for driver and worker
    import sys
    python_path = sys.executable

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.pyspark.python", python_path)
        .config("spark.pyspark.driver.python", python_path)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        # Delta Lake jars (+ UC if enabled)
        .config("spark.jars.packages", ",".join(jars))
        # Delta Lake configs
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        # Performance configs for local dev
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        # Delta defaults
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", "true")
    )

    # Add Unity Catalog configuration if enabled
    if should_use_uc:
        builder = (
            builder
            .config(f"spark.sql.catalog.{CATALOG_NAME}", "io.unitycatalog.spark.UCSingleCatalog")
            .config(f"spark.sql.catalog.{CATALOG_NAME}.uri", UC_SERVER_URL)
        )
        if UC_TOKEN:
            builder = builder.config(f"spark.sql.catalog.{CATALOG_NAME}.token", UC_TOKEN)

    if local_mode:
        builder = builder.master("local[*]")

    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)

    _spark = builder.getOrCreate()

    # Set log level to reduce noise
    _spark.sparkContext.setLogLevel("WARN")

    return _spark


def stop_spark() -> None:
    """Stop the global Spark session."""
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
