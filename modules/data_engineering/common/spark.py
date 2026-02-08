"""
Spark session management for the koe-tts lakehouse.

Connects to Hive Metastore for table resolution and MinIO for S3 storage.

Usage:
    from modules.data_engineering.common.spark import get_spark

    spark = get_spark()

    # Query by table name (resolved via Hive Metastore)
    df = spark.table("gold_koe.utterances")

    # Or direct path access
    df = spark.read.format("delta").load("s3a://forge/lake/gold/koe/utterances")
"""

import os
import sys

from pyspark.sql import SparkSession


# =============================================================================
# Configuration from environment
# =============================================================================

# Hive Metastore - table name resolution
METASTORE_URI = os.getenv("METASTORE_URI", "thrift://localhost:9083")

# MinIO/S3 - object storage
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "")

# Lake root (S3 path)
LAKE_ROOT = os.getenv("FORGE_LAKE_ROOT_S3A", "s3a://forge/lake")


# =============================================================================
# Package versions - all in one place
# =============================================================================

# Delta + Spark version must match
# PySpark 4.x requires delta-spark 4.x with Scala 2.13
PACKAGES = [
    "io.delta:delta-spark_2.13:4.0.0",
    "org.apache.hadoop:hadoop-aws:3.3.4",  # S3A filesystem support
]


# =============================================================================
# Spark session
# =============================================================================

_spark: SparkSession | None = None


def get_spark(app_name: str = "koe-tts") -> SparkSession:
    """
    Get or create a SparkSession connected to Hive Metastore + MinIO.

    One session per process, cached after first call.
    The app_name only matters on first call (for Spark UI label).

    Configuration:
    - Hive Metastore: table name → S3 location resolution
    - Delta Lake: table format
    - S3A/MinIO: object storage

    Returns:
        Configured SparkSession
    """
    global _spark

    if _spark is not None:
        return _spark

    builder = (
        SparkSession.builder
        .appName(app_name)

        # --------------------------------------------------------------------
        # Python executable - ensure driver and workers use same Python
        # --------------------------------------------------------------------
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)

        # --------------------------------------------------------------------
        # Hive Metastore - "phone book" for table names
        # spark.table("gold_koe.utterances") asks HMS: "where is this table?"
        # HMS replies: "s3a://forge/lake/gold/koe/utterances"
        # --------------------------------------------------------------------
        .config("spark.sql.catalogImplementation", "hive")
        .config("hive.metastore.uris", METASTORE_URI)

        # --------------------------------------------------------------------
        # Delta Lake - table format
        # MUST use spark_catalog (Delta's requirement)
        # --------------------------------------------------------------------
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )

        # --------------------------------------------------------------------
        # S3A / MinIO - object storage
        # Spark reads/writes to s3a://forge/lake/...
        # --------------------------------------------------------------------
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")  # MinIO requires path-style
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")  # localhost, no TLS

        # --------------------------------------------------------------------
        # Packages - ALL in one string (spark.jars.packages is non-additive)
        # --------------------------------------------------------------------
        .config("spark.jars.packages", ",".join(PACKAGES))

        # --------------------------------------------------------------------
        # Performance - tuned for local dev
        # --------------------------------------------------------------------
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")

        # --------------------------------------------------------------------
        # Delta defaults
        # --------------------------------------------------------------------
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")

        # --------------------------------------------------------------------
        # Local mode
        # --------------------------------------------------------------------
        .master("local[*]")
    )

    _spark = builder.getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")

    return _spark


def stop_spark() -> None:
    """Stop the global Spark session."""
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
