"""Test suite for forge.query.duckdb — DuckDB client.

All tests mock DuckDB so no real DuckDB or Delta tables are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modules.forge.query.duckdb import (
    ALLOWED_LAYERS,
    MAX_LIMIT,
    DuckDBClient,
    create_duckdb_client,
    create_s3_duckdb_client,
    validate_layer,
)

# ════════════════════════════════════════════════════════════════════════
# 1. validate_layer
# ════════════════════════════════════════════════════════════════════════


class TestValidateLayer:
    def test_bronze(self):
        assert validate_layer("bronze") == "bronze"

    def test_silver(self):
        assert validate_layer("silver") == "silver"

    def test_gold(self):
        assert validate_layer("gold") == "gold"

    def test_case_insensitive(self):
        assert validate_layer("BRONZE") == "bronze"

    def test_strips_whitespace(self):
        assert validate_layer("  silver  ") == "silver"

    def test_invalid_rejected(self):
        with pytest.raises(ValueError, match="Invalid layer"):
            validate_layer("copper")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="Invalid layer"):
            validate_layer("")


# ════════════════════════════════════════════════════════════════════════
# 2. Constants
# ════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_allowed_layers(self):
        assert ALLOWED_LAYERS == frozenset({"bronze", "silver", "gold"})

    def test_max_limit(self):
        assert MAX_LIMIT == 500_000


# ════════════════════════════════════════════════════════════════════════
# 3. DuckDBClient — safe_table_path
# ════════════════════════════════════════════════════════════════════════


class TestSafeTablePath:
    def test_valid_path(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        path = client._safe_table_path("bronze", "jsut", "utterances")
        assert path == (tmp_path / "bronze" / "jsut" / "utterances").resolve()

    def test_traversal_rejected(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        with pytest.raises(ValueError):
            client._safe_table_path("bronze", "jsut", "../../etc")

    def test_invalid_layer_rejected(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        with pytest.raises(ValueError, match="Invalid layer"):
            client._safe_table_path("platinum", "jsut", "utterances")

    def test_invalid_dataset_rejected(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        with pytest.raises(ValueError):
            client._safe_table_path("bronze", "DROP TABLE", "utterances")

    def test_invalid_table_rejected(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        with pytest.raises(ValueError):
            client._safe_table_path("bronze", "jsut", "../bad")


# ════════════════════════════════════════════════════════════════════════
# 4. DuckDBClient — connection and queries
# ════════════════════════════════════════════════════════════════════════


class TestDuckDBClient:
    @patch("modules.forge.query.duckdb.duckdb", create=True)
    def test_get_connection_caches(self, mock_duckdb_mod, tmp_path):
        """Connection is created once and cached."""
        mock_conn = MagicMock()
        mock_duckdb_mod.connect.return_value = mock_conn

        # Patch the import inside get_connection
        import sys

        with patch.dict(sys.modules, {"duckdb": mock_duckdb_mod}):
            client = DuckDBClient(lake_root=tmp_path)
            conn1 = client.get_connection()
            conn2 = client.get_connection()

        assert conn1 is conn2
        mock_duckdb_mod.connect.assert_called_once()

    def test_query_raw_delegates(self, tmp_path):
        """query_raw executes SQL via the connection."""
        mock_conn = MagicMock()
        mock_df = MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = mock_df

        client = DuckDBClient(lake_root=tmp_path)
        client._conn = mock_conn

        result = client.query_raw("SELECT 1")
        assert result is mock_df
        mock_conn.execute.assert_called_once_with("SELECT 1")

    def test_lake_root_stored(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        assert client.lake_root == tmp_path

    def test_s3_lake_root_stored_as_string(self):
        client = DuckDBClient(lake_root="s3://forge/lake")
        assert client.lake_root == "s3://forge/lake"


# ════════════════════════════════════════════════════════════════════════
# 5. DuckDBClient — list_tables
# ════════════════════════════════════════════════════════════════════════


class TestDuckDBClientListTables:
    def test_empty_lake(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path)
        assert client.list_tables() == []

    def test_nonexistent_lake(self, tmp_path):
        client = DuckDBClient(lake_root=tmp_path / "nonexistent")
        assert client.list_tables() == []

    def test_finds_delta_tables(self, tmp_path):
        # Create fake Delta table structure
        table_path = tmp_path / "bronze" / "jsut" / "utterances" / "_delta_log"
        table_path.mkdir(parents=True)

        client = DuckDBClient(lake_root=tmp_path)
        tables = client.list_tables()

        assert len(tables) == 1
        assert tables[0]["layer"] == "bronze"
        assert tables[0]["dataset"] == "jsut"
        assert tables[0]["table"] == "utterances"

    def test_multiple_tables(self, tmp_path):
        for layer, ds, tbl in [
            ("bronze", "jsut", "utterances"),
            ("silver", "jsut", "utterances"),
            ("gold", "jsut", "manifests"),
        ]:
            (tmp_path / layer / ds / tbl / "_delta_log").mkdir(parents=True)

        client = DuckDBClient(lake_root=tmp_path)
        tables = client.list_tables()
        assert len(tables) == 3

    def test_ignores_non_delta_dirs(self, tmp_path):
        # Dir without _delta_log
        (tmp_path / "bronze" / "jsut" / "random_dir").mkdir(parents=True)

        client = DuckDBClient(lake_root=tmp_path)
        assert client.list_tables() == []


# ════════════════════════════════════════════════════════════════════════
# 6. DuckDBClient — S3 config
# ════════════════════════════════════════════════════════════════════════


class TestDuckDBClientS3Config:
    def test_s3_config_applied(self):
        """S3 credentials are set on the connection."""
        import sys

        mock_duckdb_mod = MagicMock()
        mock_conn = MagicMock()
        mock_duckdb_mod.connect.return_value = mock_conn

        s3_config = {
            "endpoint": "http://localhost:9000",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
            "region": "us-east-1",
        }

        with patch.dict(sys.modules, {"duckdb": mock_duckdb_mod}):
            client = DuckDBClient(lake_root="s3://forge/lake", s3_config=s3_config)
            client.get_connection()

        # Verify S3 settings were applied
        execute_calls = [str(c) for c in mock_conn.execute.call_args_list]
        joined = " ".join(execute_calls)
        assert "s3_endpoint" in joined
        assert "s3_access_key_id" in joined
        assert "s3_secret_access_key" in joined
        assert "s3_region" in joined
        assert "s3_url_style" in joined

    def test_no_s3_config(self):
        """Without S3 config, no S3 settings are applied."""
        import sys

        mock_duckdb_mod = MagicMock()
        mock_conn = MagicMock()
        mock_duckdb_mod.connect.return_value = mock_conn

        with patch.dict(sys.modules, {"duckdb": mock_duckdb_mod}):
            client = DuckDBClient(lake_root="/local/lake")
            client.get_connection()

        # Only INSTALL + LOAD calls, no S3 config
        assert mock_conn.execute.call_count == 2


# ════════════════════════════════════════════════════════════════════════
# 7. Factory functions
# ════════════════════════════════════════════════════════════════════════


class TestCreateDuckDBClient:
    def test_returns_client(self, tmp_path):
        client = create_duckdb_client(tmp_path)
        assert isinstance(client, DuckDBClient)
        assert client.lake_root == tmp_path

    def test_string_path(self, tmp_path):
        client = create_duckdb_client(str(tmp_path))
        assert isinstance(client, DuckDBClient)


class TestCreateS3DuckDBClient:
    def test_valid_s3_uri(self):
        client = create_s3_duckdb_client(
            lake_root="s3://forge/lake",
            endpoint="http://localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
        )
        assert isinstance(client, DuckDBClient)
        assert client.lake_root == "s3://forge/lake"
        assert client._s3_config is not None
        assert client._s3_config["endpoint"] == "http://localhost:9000"

    def test_s3a_rejected(self):
        with pytest.raises(ValueError, match="wrong scheme.*s3://"):
            create_s3_duckdb_client(lake_root="s3a://forge/lake")

    def test_bare_path_rejected(self):
        with pytest.raises(ValueError, match="must start with s3://"):
            create_s3_duckdb_client(lake_root="/data/lake")

    def test_http_rejected(self):
        with pytest.raises(ValueError, match="must start with s3://"):
            create_s3_duckdb_client(lake_root="http://localhost:9000/lake")

    def test_no_credentials(self):
        """Factory works with just a lake_root (no credentials)."""
        client = create_s3_duckdb_client(lake_root="s3://forge/lake")
        assert isinstance(client, DuckDBClient)
        assert client._s3_config is None

    def test_region_passed(self):
        client = create_s3_duckdb_client(
            lake_root="s3://forge/lake",
            region="ap-northeast-1",
        )
        assert client._s3_config["region"] == "ap-northeast-1"


# ════════════════════════════════════════════════════════════════════════
# 8. Lazy __init__.py re-exports
# ════════════════════════════════════════════════════════════════════════


class TestLazyReexports:
    def test_duckdb_client_accessible(self):
        from modules.forge.query import DuckDBClient as Cls

        assert Cls is DuckDBClient

    def test_create_duckdb_client_accessible(self):
        from modules.forge.query import create_duckdb_client as fn

        assert fn is create_duckdb_client

    def test_create_s3_duckdb_client_accessible(self):
        from modules.forge.query import create_s3_duckdb_client as fn

        assert fn is create_s3_duckdb_client

    def test_unknown_attr_raises(self):
        import modules.forge.query as query_mod

        with pytest.raises(AttributeError, match="no attribute"):
            _ = query_mod.nonexistent_thing
