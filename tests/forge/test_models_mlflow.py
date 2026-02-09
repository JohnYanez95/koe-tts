"""Test suite for forge.models.mlflow — MLflow model registry wrapper.

All tests mock MLflow — no real MLflow installation needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modules.forge.models.mlflow import ModelRegistry, parse_forge_uri

# ════════════════════════════════════════════════════════════════════════
# 1. forge:// URI parsing
# ════════════════════════════════════════════════════════════════════════


class TestParseForgeUri:
    def test_alias_prod(self):
        assert parse_forge_uri("forge://vits@prod") == "models:/vits@prod"

    def test_alias_best(self):
        assert parse_forge_uri("forge://vits@best") == "models:/vits@best"

    def test_version_number(self):
        assert parse_forge_uri("forge://vits@3") == "models:/vits/3"

    def test_version_with_v_prefix(self):
        assert parse_forge_uri("forge://vits@v3") == "models:/vits/3"

    def test_mlflow_passthrough(self):
        assert parse_forge_uri("models:/vits@best") == "models:/vits@best"

    def test_mlflow_version_passthrough(self):
        assert parse_forge_uri("models:/vits/3") == "models:/vits/3"

    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="Invalid model URI"):
            parse_forge_uri("bad://vits")

    def test_missing_ref_raises(self):
        with pytest.raises(ValueError, match="Invalid model URI"):
            parse_forge_uri("forge://vits")

    def test_complex_name(self):
        assert parse_forge_uri("forge://my-model@staging") == "models:/my-model@staging"

    def test_large_version(self):
        assert parse_forge_uri("forge://vits@v100") == "models:/vits/100"


# ════════════════════════════════════════════════════════════════════════
# 2. ModelRegistry construction
# ════════════════════════════════════════════════════════════════════════


class TestModelRegistryInit:
    def test_default_tracking_uri(self):
        registry = ModelRegistry()
        assert registry.tracking_uri == "http://localhost:5002"

    def test_explicit_tracking_uri(self):
        registry = ModelRegistry(tracking_uri="http://mlflow:5000")
        assert registry.tracking_uri == "http://mlflow:5000"

    def test_env_tracking_uri(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom:5003")
        registry = ModelRegistry()
        assert registry.tracking_uri == "http://custom:5003"

    def test_from_env(self):
        registry = ModelRegistry.from_env()
        assert isinstance(registry, ModelRegistry)


# ════════════════════════════════════════════════════════════════════════
# 3. load_model
# ════════════════════════════════════════════════════════════════════════


class TestLoadModel:
    def test_load_with_forge_uri(self):
        mock_mlflow = MagicMock()
        mock_pyfunc = MagicMock()
        mock_model = MagicMock()
        mock_pyfunc.load_model.return_value = mock_model
        # Ensure mlflow.pyfunc attribute matches the pyfunc module mock
        mock_mlflow.pyfunc = mock_pyfunc

        with patch.dict(
            "sys.modules",
            {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc},
        ):
            registry = ModelRegistry()
            result = registry.load_model("forge://vits@prod")

        assert result is mock_model
        mock_pyfunc.load_model.assert_called_once_with("models:/vits@prod")
        mock_mlflow.set_tracking_uri.assert_called_once()


# ════════════════════════════════════════════════════════════════════════
# 4. register
# ════════════════════════════════════════════════════════════════════════


class TestRegister:
    def test_register_existing_model(self, tmp_path):
        mock_mlflow = MagicMock()
        mock_version = MagicMock()
        mock_mlflow.register_model.return_value = mock_version

        model_file = tmp_path / "model.pt"
        model_file.touch()

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            registry = ModelRegistry()
            result = registry.register(model_file, "vits", tags={"stage": "prod"})

        assert result is mock_version
        mock_mlflow.register_model.assert_called_once_with(
            model_uri=str(model_file),
            name="vits",
            tags={"stage": "prod"},
        )

    def test_register_missing_model_raises(self, tmp_path):
        registry = ModelRegistry()
        with pytest.raises(FileNotFoundError, match="Model not found"):
            registry.register(tmp_path / "nonexistent.pt", "vits")


# ════════════════════════════════════════════════════════════════════════
# 5. list_models
# ════════════════════════════════════════════════════════════════════════


class TestListModels:
    def test_list_models(self):
        mock_mlflow = MagicMock()
        mock_tracking = MagicMock()
        mock_mlflow.tracking = mock_tracking
        mock_client = MagicMock()
        mock_tracking.MlflowClient.return_value = mock_client

        mock_model = MagicMock()
        mock_model.name = "vits"
        mock_model.description = "VITS TTS model"
        mock_version = MagicMock()
        mock_version.version = "3"
        mock_version.aliases = ["prod"]
        mock_model.latest_versions = [mock_version]

        mock_client.search_registered_models.return_value = [mock_model]

        with patch.dict(
            "sys.modules",
            {"mlflow": mock_mlflow, "mlflow.tracking": mock_tracking},
        ):
            registry = ModelRegistry()
            result = registry.list_models()

        assert len(result) == 1
        assert result[0]["name"] == "vits"
        assert result[0]["description"] == "VITS TTS model"
        assert result[0]["latest_versions"][0]["version"] == "3"
        assert result[0]["latest_versions"][0]["aliases"] == ["prod"]


# ════════════════════════════════════════════════════════════════════════
# 6. Lazy __init__.py re-exports
# ════════════════════════════════════════════════════════════════════════


class TestLazyImports:
    def test_model_registry_importable(self):
        from modules.forge.models import ModelRegistry as MR

        assert MR is ModelRegistry

    def test_parse_forge_uri_importable(self):
        from modules.forge.models import parse_forge_uri as fn

        assert fn is parse_forge_uri

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            from modules.forge import models

            _ = models.NonExistent
