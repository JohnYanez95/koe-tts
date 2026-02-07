"""
MLflow Model Registry for koe-tts.

Handles model versioning, registration, and aliases (best, candidate, prod).

Conventions:
- Model names follow pattern: tts-ja-{architecture}[-{variant}]
  e.g., tts-ja-vits, tts-ja-vits-jsut, tts-ja-xtts
- Aliases:
  - "best": best val_loss checkpoint from any run
  - "candidate": current run's best checkpoint
  - "prod": deployed/shipped model
- Each training run creates a new model version

Usage:
    from modules.registry import register_checkpoint, set_alias, load_model

    # Register a checkpoint
    version = register_checkpoint(
        checkpoint_path="checkpoints/best.ckpt",
        model_name="tts-ja-vits-jsut",
        run_id=run.info.run_id,
    )

    # Set alias
    set_alias("tts-ja-vits-jsut", version, "candidate")

    # Load model by alias
    model_uri = get_model_uri("tts-ja-vits-jsut", alias="best")
"""

from pathlib import Path
from typing import Optional, Union

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from .config import configure_mlflow, get_model_name


def register_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str,
    run_id: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
) -> ModelVersion:
    """
    Register a checkpoint in the model registry.

    Creates the registered model if it doesn't exist, then logs the
    checkpoint as a new version.

    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Registered model name (e.g., "tts-ja-vits-jsut")
        run_id: MLflow run ID to associate with. If None, uses active run.
        description: Version description
        tags: Version tags

    Returns:
        ModelVersion object with version number

    Example:
        version = register_checkpoint(
            "checkpoints/best.ckpt",
            "tts-ja-vits-jsut",
            description="val_loss=33.76, epoch=130"
        )
        print(f"Registered as version {version.version}")
    """
    configure_mlflow()
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Ensure model exists in registry
    _ensure_registered_model(model_name)

    # Log checkpoint as artifact and register
    if run_id:
        with mlflow.start_run(run_id=run_id):
            artifact_path = f"registered_models/{model_name}"
            mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_path)
            artifact_uri = f"runs:/{run_id}/{artifact_path}/{checkpoint_path.name}"
    else:
        # Active run
        artifact_path = f"registered_models/{model_name}"
        mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_path)
        run = mlflow.active_run()
        if run is None:
            raise RuntimeError("No active run. Pass run_id or use within start_run().")
        artifact_uri = f"runs:/{run.info.run_id}/{artifact_path}/{checkpoint_path.name}"

    # Register the model version
    result = mlflow.register_model(
        model_uri=artifact_uri,
        name=model_name,
        tags=tags,
    )

    # Add description if provided
    if description:
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=description,
        )

    return result


def set_alias(
    model_name: str,
    version: Union[int, str, ModelVersion],
    alias: str,
) -> None:
    """
    Set an alias for a model version.

    Common aliases:
    - "best": best overall checkpoint
    - "candidate": current experiment's best
    - "prod": production/deployed model

    Args:
        model_name: Registered model name
        version: Version number or ModelVersion object
        alias: Alias name (e.g., "best", "candidate", "prod")
    """
    client = mlflow.tracking.MlflowClient()

    if isinstance(version, ModelVersion):
        version_num = version.version
    else:
        version_num = str(version)

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version_num,
    )


def delete_alias(model_name: str, alias: str) -> None:
    """Remove an alias from a model."""
    client = mlflow.tracking.MlflowClient()
    client.delete_registered_model_alias(name=model_name, alias=alias)


def get_model_uri(
    model_name: str,
    version: Optional[Union[int, str]] = None,
    alias: Optional[str] = None,
) -> str:
    """
    Get the URI for a registered model.

    Args:
        model_name: Registered model name
        version: Specific version number
        alias: Alias name (e.g., "best", "prod")

    Returns:
        Model URI for loading

    Raises:
        ValueError: If neither version nor alias provided
    """
    if alias:
        return f"models:/{model_name}@{alias}"
    elif version:
        return f"models:/{model_name}/{version}"
    else:
        raise ValueError("Must provide either version or alias")


def get_model_version(
    model_name: str,
    version: Optional[Union[int, str]] = None,
    alias: Optional[str] = None,
) -> ModelVersion:
    """
    Get ModelVersion metadata.

    Args:
        model_name: Registered model name
        version: Specific version number
        alias: Alias name

    Returns:
        ModelVersion with metadata
    """
    client = mlflow.tracking.MlflowClient()

    if alias:
        return client.get_model_version_by_alias(name=model_name, alias=alias)
    elif version:
        return client.get_model_version(name=model_name, version=str(version))
    else:
        raise ValueError("Must provide either version or alias")


def list_model_versions(
    model_name: str,
    max_results: int = 100,
) -> list[ModelVersion]:
    """
    List all versions of a registered model.

    Args:
        model_name: Registered model name
        max_results: Maximum versions to return

    Returns:
        List of ModelVersion objects, newest first
    """
    client = mlflow.tracking.MlflowClient()

    # Search for versions
    versions = client.search_model_versions(
        filter_string=f"name='{model_name}'",
        max_results=max_results,
        order_by=["version_number DESC"],
    )

    return list(versions)


def get_latest_version(model_name: str) -> Optional[ModelVersion]:
    """Get the latest version of a model."""
    versions = list_model_versions(model_name, max_results=1)
    return versions[0] if versions else None


def _ensure_registered_model(model_name: str) -> None:
    """Create registered model if it doesn't exist."""
    client = mlflow.tracking.MlflowClient()

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        # Model doesn't exist, create it
        client.create_registered_model(
            name=model_name,
            description=f"koe-tts model: {model_name}",
        )


def promote_to_prod(
    model_name: str,
    from_alias: str = "best",
) -> ModelVersion:
    """
    Promote a model version to production.

    Copies the "best" (or specified) alias to "prod".

    Args:
        model_name: Registered model name
        from_alias: Source alias to promote (default: "best")

    Returns:
        The promoted ModelVersion
    """
    # Get the version currently at from_alias
    version = get_model_version(model_name, alias=from_alias)

    # Set prod alias
    set_alias(model_name, version.version, "prod")

    return version
