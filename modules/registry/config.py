"""
MLflow configuration for koe-tts.

Centralizes MLflow tracking URI, artifact location, and experiment naming.
All other registry modules import from here.

Configuration sources (in order):
1. Environment variables (MLFLOW_TRACKING_URI, etc.)
2. Derived from DATA_ROOT (default)

Layout under DATA_ROOT:
    mlflow/
        mlruns/         # tracking store (file-based)
        artifacts/      # artifact store
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow

from modules.platform.paths import paths


@dataclass
class RegistryConfig:
    """MLflow registry configuration."""

    tracking_uri: str
    artifact_location: str
    default_experiment: str = "koe-tts"

    # Model naming conventions
    model_prefix: str = "tts-ja"

    @property
    def mlflow_root(self) -> Path:
        """Root directory for MLflow data."""
        return paths.data_root / "mlflow"


def _get_tracking_uri() -> str:
    """
    Get MLflow tracking URI.

    Priority:
    1. MLFLOW_TRACKING_URI env var
    2. File-based under DATA_ROOT/mlflow/mlruns
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    # Default: file-based tracking
    mlruns_path = paths.data_root / "mlflow" / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    return str(mlruns_path)


def _get_artifact_location() -> str:
    """
    Get MLflow artifact storage location.

    Priority:
    1. MLFLOW_ARTIFACT_LOCATION env var
    2. Directory under DATA_ROOT/mlflow/artifacts
    """
    env_loc = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
    if env_loc:
        return env_loc

    artifacts_path = paths.data_root / "mlflow" / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return str(artifacts_path)


# Singleton config
_config: Optional[RegistryConfig] = None


def get_config() -> RegistryConfig:
    """Get registry configuration (singleton)."""
    global _config
    if _config is None:
        _config = RegistryConfig(
            tracking_uri=_get_tracking_uri(),
            artifact_location=_get_artifact_location(),
        )
    return _config


def configure_mlflow(experiment_name: Optional[str] = None) -> None:
    """
    Configure MLflow with koe-tts settings.

    Call this once at the start of your script/notebook.

    Args:
        experiment_name: Override default experiment name
    """
    config = get_config()

    mlflow.set_tracking_uri(config.tracking_uri)

    exp_name = experiment_name or config.default_experiment
    mlflow.set_experiment(exp_name)


def get_model_name(architecture: str, variant: Optional[str] = None) -> str:
    """
    Get standardized model name for registry.

    Args:
        architecture: Model architecture (e.g., "vits", "xtts")
        variant: Optional variant (e.g., "single-speaker", "multi")

    Returns:
        Registered model name like "tts-ja-vits" or "tts-ja-vits-multi"

    Examples:
        get_model_name("vits") -> "tts-ja-vits"
        get_model_name("vits", "jsut") -> "tts-ja-vits-jsut"
    """
    config = get_config()
    parts = [config.model_prefix, architecture]
    if variant:
        parts.append(variant)
    return "-".join(parts)
