"""
MLflow run tracking for koe-tts.

Provides helpers for starting runs, logging params/metrics/artifacts,
with koe-tts conventions baked in.

Usage:
    from modules.registry import start_run, log_metrics, log_audio_samples

    with start_run("jsut-finetune-v1") as run:
        log_config(config_dict)
        # ... training loop ...
        log_metrics({"val_loss": 33.76, "epoch": 130})
        log_audio_samples(samples_dir)
"""

import json
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional, Union

import mlflow
from mlflow.entities import Run

from .config import configure_mlflow, get_config


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass
    return None


@contextmanager
def start_run(
    run_name: str,
    experiment_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    nested: bool = False,
) -> Generator[Run, None, None]:
    """
    Start an MLflow run with koe-tts conventions.

    Automatically logs:
    - git commit hash
    - timestamp
    - run_name as tag

    Args:
        run_name: Human-readable run name (e.g., "jsut-finetune-v1")
        experiment_name: Override default experiment
        tags: Additional tags to log
        nested: If True, allow nested runs

    Yields:
        MLflow Run object

    Example:
        with start_run("jsut-finetune-v1") as run:
            log_params({"lr": 1e-5, "batch_size": 48})
            # ... training ...
            log_metrics({"val_loss": 33.76})
    """
    configure_mlflow(experiment_name)

    # Build tags
    run_tags = {
        "run_name": run_name,
        "started_at": datetime.utcnow().isoformat(),
    }

    git_commit = _get_git_commit()
    if git_commit:
        run_tags["git_commit"] = git_commit

    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    """
    Log parameters to the active run.

    Handles nested dicts by flattening with dot notation.

    Args:
        params: Parameter dict (can be nested)
    """
    flat_params = _flatten_dict(params)

    # MLflow has a 500-char limit on param values
    for key, value in flat_params.items():
        str_value = str(value)
        if len(str_value) > 500:
            str_value = str_value[:497] + "..."
        mlflow.log_param(key, str_value)


def log_metrics(
    metrics: dict[str, float],
    step: Optional[int] = None,
) -> None:
    """
    Log metrics to the active run.

    Args:
        metrics: Metric name -> value dict
        step: Optional step number (epoch, global_step, etc.)
    """
    mlflow.log_metrics(metrics, step=step)


def log_config(
    config: dict[str, Any],
    filename: str = "config.yaml",
) -> None:
    """
    Log a config dict as both params and artifact.

    Args:
        config: Configuration dictionary
        filename: Artifact filename (config.yaml or config.json)
    """
    # Log as params (flattened)
    log_params(config)

    # Log as artifact (full structure)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=filename, delete=False) as f:
        if filename.endswith(".json"):
            json.dump(config, f, indent=2, default=str)
        else:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="config")
    Path(temp_path).unlink()


def log_audio_samples(
    samples_dir: Union[str, Path],
    artifact_path: str = "samples",
) -> None:
    """
    Log a directory of audio samples as artifacts.

    Args:
        samples_dir: Directory containing .wav files
        artifact_path: Path within artifacts (e.g., "samples/epoch_100")
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        return

    mlflow.log_artifacts(str(samples_dir), artifact_path=artifact_path)


def log_checkpoint(
    checkpoint_path: Union[str, Path],
    kind: str = "checkpoint",
) -> None:
    """
    Log a model checkpoint as artifact.

    Args:
        checkpoint_path: Path to .ckpt or .pt file
        kind: Artifact subdirectory ("checkpoint", "best", "last")
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mlflow.log_artifact(str(checkpoint_path), artifact_path=f"checkpoints/{kind}")


def log_dashboard(
    dashboard_path: Union[str, Path],
) -> None:
    """Log a training dashboard image."""
    dashboard_path = Path(dashboard_path)
    if dashboard_path.exists():
        mlflow.log_artifact(str(dashboard_path), artifact_path="dashboards")


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten nested dict with dot notation keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_run_artifact_uri(run_id: Optional[str] = None) -> str:
    """
    Get the artifact URI for a run.

    Args:
        run_id: Run ID, or None for active run

    Returns:
        Artifact URI (file path or remote URI)
    """
    if run_id is None:
        return mlflow.get_artifact_uri()
    return mlflow.get_artifact_uri(run_id=run_id)
