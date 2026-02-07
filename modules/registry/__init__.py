"""
Registry module: MLflow experiment tracking and model registry for koe-tts.

Provides a thin, opinionated wrapper around MLflow with koe-tts conventions:
- Centralized configuration (tracking URI, artifact location)
- Standardized model naming (tts-ja-{arch}[-{variant}])
- Alias-based versioning (best, candidate, prod)
- Delta table lineage tracking

Quick Start:
    from modules.registry import start_run, log_metrics, register_checkpoint

    with start_run("jsut-finetune-v1") as run:
        # Log training config
        log_config({"lr": 1e-5, "batch_size": 48})

        # Tag dataset versions for lineage
        tag_dataset_version("gold", "training_set")

        # ... training loop ...
        log_metrics({"val_loss": 33.76, "epoch": 130}, step=130)

        # Register best checkpoint
        version = register_checkpoint(
            "checkpoints/best.ckpt",
            model_name="tts-ja-vits-jsut",
        )
        set_alias("tts-ja-vits-jsut", version, "candidate")

MLflow Setup:
    The module auto-configures MLflow on first use:
    - Tracking URI: $DATA_ROOT/mlflow/mlruns (or MLFLOW_TRACKING_URI)
    - Artifacts: $DATA_ROOT/mlflow/artifacts (or MLFLOW_ARTIFACT_LOCATION)
    - Default experiment: "koe-tts"

    No server required for local development. For multi-user setups,
    point MLFLOW_TRACKING_URI to a Postgres-backed MLflow server.

Model Registry Conventions:
    - Model names: tts-ja-vits, tts-ja-vits-jsut, tts-ja-xtts
    - Aliases:
        - "best": overall best checkpoint across all runs
        - "candidate": current run's best checkpoint
        - "prod": deployed/shipped model
    - Each training run creates a new model version
"""

# Configuration
from .config import configure_mlflow, get_config, get_model_name

# Tracking
from .tracking import (
    get_run_artifact_uri,
    log_audio_samples,
    log_checkpoint,
    log_config,
    log_dashboard,
    log_metrics,
    log_params,
    start_run,
)

# Model Registry
from .models import (
    delete_alias,
    get_latest_version,
    get_model_uri,
    get_model_version,
    list_model_versions,
    promote_to_prod,
    register_checkpoint,
    set_alias,
)

# Lineage
from .lineage import (
    get_run_lineage,
    tag_all_training_datasets,
    tag_dataset_version,
    tag_git_info,
    tag_preprocessing_version,
)

__all__ = [
    # Config
    "configure_mlflow",
    "get_config",
    "get_model_name",
    # Tracking
    "start_run",
    "log_params",
    "log_metrics",
    "log_config",
    "log_audio_samples",
    "log_checkpoint",
    "log_dashboard",
    "get_run_artifact_uri",
    # Model Registry
    "register_checkpoint",
    "set_alias",
    "delete_alias",
    "get_model_uri",
    "get_model_version",
    "list_model_versions",
    "get_latest_version",
    "promote_to_prod",
    # Lineage
    "tag_dataset_version",
    "tag_all_training_datasets",
    "tag_preprocessing_version",
    "tag_git_info",
    "get_run_lineage",
]
