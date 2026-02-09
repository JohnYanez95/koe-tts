"""Spark session management for the forge lakehouse.

Connects to Hive Metastore for table resolution and MinIO for S3 storage.
PySpark is lazy-imported — this module can be imported without PySpark
installed, but ``get_spark()`` will raise ``ImportError`` at call time.

Usage::

    from modules.forge.query.spark import get_spark

    spark = get_spark()
    df = spark.table("gold_koe.utterances")

Environment Variables
---------------------
- ``METASTORE_URI``          — Hive Metastore (default: thrift://localhost:9083)
- ``MINIO_ENDPOINT``         — MinIO endpoint (default: http://localhost:9000)
- ``MINIO_ROOT_USER``        — MinIO access key
- ``MINIO_ROOT_PASSWORD``    — MinIO secret key
- ``FORGE_LAKE_ROOT_S3A``    — Lake root, must start with ``s3a://``
                               (default: s3a://forge/lake)
- ``FORGE_S3_REGION``        — AWS/MinIO region (default: us-east-1)
- ``OPENLINEAGE_URL``        — OpenLineage collector (optional, no default)

Public API
----------
- ``get_spark(app_name)`` — cached Spark session factory
- ``stop_spark()`` — shutdown the global session
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────

METASTORE_URI: str = os.getenv("METASTORE_URI", "thrift://localhost:9083")
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY: str = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_SECRET_KEY: str = os.getenv("MINIO_ROOT_PASSWORD", "")
LAKE_ROOT: str = os.getenv("FORGE_LAKE_ROOT_S3A", "s3a://forge/lake")
S3_REGION: str = os.getenv("FORGE_S3_REGION", "us-east-1")
OPENLINEAGE_URL: str | None = os.getenv("OPENLINEAGE_URL")

# ── Package versions — all in one place ──────────────────────────────

PACKAGES = [
    "io.delta:delta-spark_2.13:4.0.0",
    "org.apache.hadoop:hadoop-aws:3.3.4",
]

# ── Spark session ────────────────────────────────────────────────────

_spark: SparkSession | None = None


def _validate_lake_root(lake_root: str) -> str:
    """Validate that lake root uses the s3a:// scheme.

    Raises ``ValueError`` with an actionable message if the scheme is
    wrong (e.g. ``s3://`` or a bare filesystem path).
    """
    if lake_root.startswith("s3a://"):
        return lake_root
    if lake_root.startswith("s3://"):
        raise ValueError(
            f"FORGE_LAKE_ROOT_S3A uses wrong scheme: {lake_root!r}. "
            "Spark requires s3a:// (not s3://). "
            "Set FORGE_LAKE_ROOT_S3A=s3a://forge/lake"
        )
    raise ValueError(
        f"FORGE_LAKE_ROOT_S3A must start with s3a://, got: {lake_root!r}. "
        "Set FORGE_LAKE_ROOT_S3A=s3a://forge/lake"
    )


def get_spark(app_name: str = "forge") -> SparkSession:
    """Get or create a SparkSession connected to Hive Metastore + MinIO.

    One session per process, cached after first call.
    The ``app_name`` only matters on first call (for Spark UI label).

    Configuration:
    - Hive Metastore: table name → S3 location resolution
    - Delta Lake: table format
    - S3A/MinIO: object storage
    - OpenLineage: optional lineage collector

    Raises:
        ImportError: If PySpark is not installed.
        ValueError: If ``FORGE_LAKE_ROOT_S3A`` has wrong URI scheme.
    """
    global _spark  # noqa: PLW0603

    if _spark is not None:
        return _spark

    try:
        from pyspark.sql import SparkSession as _SparkSession
    except ImportError:
        raise ImportError(
            "PySpark is required for Spark operations. "
            "Install with: pip install pyspark"
        ) from None

    lake_root = _validate_lake_root(LAKE_ROOT)
    logger.debug("Forge lake root: %s", lake_root)

    # Build packages list — conditionally include OpenLineage JAR
    packages = list(PACKAGES)
    if OPENLINEAGE_URL:
        packages.append("io.openlineage:openlineage-spark_2.13:1.7.0")

    builder = (
        _SparkSession.builder
        .appName(app_name)
        # ── Python executable ────────────────────────────────────
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        # ── Hive Metastore ───────────────────────────────────────
        .config("spark.sql.catalogImplementation", "hive")
        .config("hive.metastore.uris", METASTORE_URI)
        # ── Delta Lake ───────────────────────────────────────────
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # ── S3A / MinIO ─────────────────────────────────────────
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # ── Packages ─────────────────────────────────────────────
        .config("spark.jars.packages", ",".join(packages))
        # ── Performance (local dev) ──────────────────────────────
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        # ── Delta defaults ───────────────────────────────────────
        .config(
            "spark.databricks.delta.schema.autoMerge.enabled", "true",
        )
        # ── Local mode ───────────────────────────────────────────
        .master("local[*]")
    )

    # ── OpenLineage (optional) ───────────────────────────────────
    if OPENLINEAGE_URL:
        builder = (
            builder
            .config(
                "spark.extraListeners",
                "io.openlineage.spark.agent.OpenLineageSparkListener",
            )
            .config(
                "spark.openlineage.transport.type",
                "http",
            )
            .config(
                "spark.openlineage.transport.url",
                OPENLINEAGE_URL,
            )
            .config(
                "spark.openlineage.namespace",
                "north_star",
            )
        )

    _spark = builder.getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")

    return _spark


def stop_spark() -> None:
    """Stop the global Spark session."""
    global _spark  # noqa: PLW0603
    if _spark is not None:
        _spark.stop()
        _spark = None
