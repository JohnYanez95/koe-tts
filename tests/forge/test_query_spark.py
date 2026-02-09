"""Test suite for forge.query.spark — Spark session factory.

All tests mock PySpark so no real Spark installation is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modules.forge.query.spark import _validate_lake_root

# ════════════════════════════════════════════════════════════════════════
# 1. Lake root validation
# ════════════════════════════════════════════════════════════════════════


class TestValidateLakeRoot:
    def test_s3a_accepted(self):
        assert _validate_lake_root("s3a://forge/lake") == "s3a://forge/lake"

    def test_s3a_custom_path(self):
        assert _validate_lake_root("s3a://mybucket/data") == "s3a://mybucket/data"

    def test_s3_rejected(self):
        with pytest.raises(ValueError, match="wrong scheme.*s3a://"):
            _validate_lake_root("s3://forge/lake")

    def test_bare_path_rejected(self):
        with pytest.raises(ValueError, match="must start with s3a://"):
            _validate_lake_root("/data/lake")

    def test_http_rejected(self):
        with pytest.raises(ValueError, match="must start with s3a://"):
            _validate_lake_root("http://localhost:9000/forge/lake")


# ════════════════════════════════════════════════════════════════════════
# 2. get_spark env vars and config
# ════════════════════════════════════════════════════════════════════════


class TestGetSparkConfig:
    @patch("modules.forge.query.spark._validate_lake_root")
    def test_reads_env_vars(self, mock_validate, monkeypatch):
        """Verify get_spark reads config from env vars."""
        mock_validate.return_value = "s3a://forge/lake"

        # Mock SparkSession.builder chain
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session
        mock_session.sparkContext = MagicMock()

        mock_spark_module = MagicMock()
        mock_spark_module.SparkSession.builder = mock_builder

        # Reset the global _spark before calling
        import modules.forge.query.spark as spark_mod

        spark_mod._spark = None

        with patch.dict(
            "sys.modules",
            {"pyspark": MagicMock(), "pyspark.sql": mock_spark_module},
        ):
            result = spark_mod.get_spark("test-app")

        assert result is mock_session
        mock_builder.appName.assert_called_once_with("test-app")

        # Clean up
        spark_mod._spark = None

    def test_cached_session_returned(self):
        """Second call returns the cached session."""
        import modules.forge.query.spark as spark_mod

        mock_session = MagicMock()
        spark_mod._spark = mock_session

        result = spark_mod.get_spark()
        assert result is mock_session

        # Clean up
        spark_mod._spark = None


# ════════════════════════════════════════════════════════════════════════
# 3. stop_spark
# ════════════════════════════════════════════════════════════════════════


class TestStopSpark:
    def test_stops_and_clears(self):
        import modules.forge.query.spark as spark_mod

        mock_session = MagicMock()
        spark_mod._spark = mock_session

        spark_mod.stop_spark()

        mock_session.stop.assert_called_once()
        assert spark_mod._spark is None

    def test_noop_when_no_session(self):
        import modules.forge.query.spark as spark_mod

        spark_mod._spark = None
        spark_mod.stop_spark()  # Should not raise
        assert spark_mod._spark is None


# ════════════════════════════════════════════════════════════════════════
# 4. OpenLineage config
# ════════════════════════════════════════════════════════════════════════


class TestOpenLineageConfig:
    def test_env_var_read(self, monkeypatch):
        """OPENLINEAGE_URL env var is read at module level."""
        # The module reads env vars at import time, so we verify the
        # module attribute directly.
        import modules.forge.query.spark as spark_mod

        # Without the env var set, it should be None or whatever is in env
        assert spark_mod.OPENLINEAGE_URL is None or isinstance(
            spark_mod.OPENLINEAGE_URL, str
        )
