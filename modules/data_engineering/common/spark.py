"""
Spark session management for the koe-tts lakehouse.

Provides a configured SparkSession with Delta Lake support.

Usage:
    from modules.data_engineering.common.spark import get_spark

    spark = get_spark()
    df = spark.read.format("delta").load(str(paths.bronze / "utterances"))
"""

from typing import Optional

from pyspark.sql import SparkSession

from .paths import paths

# Global spark session (lazy initialized)
_spark: Optional[SparkSession] = None


def get_spark(
    app_name: str = "koe-tts",
    warehouse_dir: Optional[str] = None,
    local_mode: bool = True,
    extra_configs: Optional[dict] = None,
) -> SparkSession:
    """
    Get or create a configured SparkSession with Delta Lake support.

    Args:
        app_name: Spark application name
        warehouse_dir: Spark warehouse directory. Defaults to lake path.
        local_mode: If True, use local[*] master. If False, expects
                    SPARK_MASTER env var or cluster config.
        extra_configs: Additional Spark configs to set

    Returns:
        Configured SparkSession with Delta Lake extensions
    """
    global _spark

    if _spark is not None:
        return _spark

    if warehouse_dir is None:
        warehouse_dir = str(paths.lake)

    # Delta Lake version must match pyspark version
    # PySpark 4.x requires delta-spark 4.x with Scala 2.13
    DELTA_VERSION = "4.0.0"
    SCALA_VERSION = "2.13"

    # Use the same Python for driver and worker
    import sys
    python_path = sys.executable

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.pyspark.python", python_path)
        .config("spark.pyspark.driver.python", python_path)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        # Delta Lake jars (downloaded automatically on first run)
        .config(
            "spark.jars.packages",
            f"io.delta:delta-spark_{SCALA_VERSION}:{DELTA_VERSION}"
        )
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
