"""
Data lineage tracking for koe-tts.

Connects MLflow runs to Delta table versions for full reproducibility.
Tracks which dataset snapshots, phonemizer versions, and configs
produced each model.

Usage:
    from modules.registry import tag_dataset_version, get_dataset_info

    # Tag current run with dataset versions
    tag_dataset_version("bronze", "jsut")
    tag_dataset_version("silver", "utterances")
    tag_dataset_version("gold", "training_set")

    # Tag phonemizer/preprocessing info
    tag_preprocessing_version()

    # Later: get lineage info
    info = get_dataset_info(run_id)
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import mlflow
from delta import DeltaTable

from modules.platform import get_spark, paths
from modules.platform.delta import _get_table_path, table_exists


def tag_dataset_version(
    layer: str,
    table_name: str,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Tag the current run with a Delta table's version info.

    Records:
    - Table path
    - Current version number
    - Timestamp of version
    - Row count

    Args:
        layer: Lake layer (bronze, silver, gold)
        table_name: Delta table name
        run_id: Run to tag. If None, uses active run.

    Returns:
        Dict with version info that was tagged

    Example:
        tag_dataset_version("gold", "training_set")
        # Tags: dataset.gold.training_set.version = 42
        #       dataset.gold.training_set.path = /path/to/table
    """
    if not table_exists(layer, table_name):
        raise FileNotFoundError(f"Table not found: {layer}/{table_name}")

    spark = get_spark()
    table_path = _get_table_path(layer, table_name)
    delta_table = DeltaTable.forPath(spark, str(table_path))

    # Get version info
    history = delta_table.history(1).collect()[0]
    version = history["version"]
    timestamp = history["timestamp"].isoformat()

    # Get row count
    row_count = spark.read.format("delta").load(str(table_path)).count()

    # Build tags
    prefix = f"dataset.{layer}.{table_name}"
    tags = {
        f"{prefix}.path": str(table_path),
        f"{prefix}.version": str(version),
        f"{prefix}.timestamp": timestamp,
        f"{prefix}.row_count": str(row_count),
    }

    # Log to MLflow
    if run_id:
        client = mlflow.tracking.MlflowClient()
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    else:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    return {
        "layer": layer,
        "table": table_name,
        "path": str(table_path),
        "version": version,
        "timestamp": timestamp,
        "row_count": row_count,
    }


def tag_all_training_datasets(run_id: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Tag all standard training datasets.

    Convenience function to tag the typical set of tables used for training.
    """
    results = []

    # Standard tables to tag
    tables = [
        ("silver", "utterances"),
        ("gold", "training_set"),
    ]

    for layer, table_name in tables:
        if table_exists(layer, table_name):
            info = tag_dataset_version(layer, table_name, run_id)
            results.append(info)

    return results


def tag_preprocessing_version(
    run_id: Optional[str] = None,
    extra_info: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Tag the run with preprocessing/phonemizer version info.

    Records:
    - pyopenjtalk version
    - phoneme inventory hash
    - reading corrections hash

    Args:
        run_id: Run to tag. If None, uses active run.
        extra_info: Additional preprocessing info to tag

    Returns:
        Dict with version info that was tagged
    """
    tags = {}

    # pyopenjtalk version
    try:
        import pyopenjtalk
        tags["preprocessing.pyopenjtalk.version"] = getattr(
            pyopenjtalk, "__version__", "unknown"
        )
    except ImportError:
        tags["preprocessing.pyopenjtalk.version"] = "not_installed"

    # Hash the phoneme inventory (if we have one)
    try:
        from modules.platform.paths import paths as p
        inventory_path = p.data_root / "configs" / "phoneme_inventory.json"
        if inventory_path.exists():
            tags["preprocessing.phoneme_inventory.hash"] = _file_hash(inventory_path)
    except Exception:
        pass

    # Hash reading corrections (from legacy code if present)
    try:
        corrections_path = (
            Path(__file__).parent.parent.parent
            / "legacy"
            / "src"
            / "preprocessing"
            / "reading_corrections.py"
        )
        if corrections_path.exists():
            tags["preprocessing.reading_corrections.hash"] = _file_hash(corrections_path)
    except Exception:
        pass

    # Add extra info
    if extra_info:
        for key, value in extra_info.items():
            tags[f"preprocessing.{key}"] = value

    # Log to MLflow
    if run_id:
        client = mlflow.tracking.MlflowClient()
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    else:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    return tags


def tag_git_info(run_id: Optional[str] = None) -> dict[str, str]:
    """
    Tag the run with git repository info.

    Records:
    - Commit hash
    - Branch name
    - Dirty status
    """
    tags = {}

    # Commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            tags["git.commit"] = result.stdout.strip()
    except Exception:
        pass

    # Branch name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            tags["git.branch"] = result.stdout.strip()
    except Exception:
        pass

    # Dirty status
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            tags["git.dirty"] = "true" if result.stdout.strip() else "false"
    except Exception:
        pass

    # Log to MLflow
    if run_id:
        client = mlflow.tracking.MlflowClient()
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    else:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    return tags


def get_run_lineage(run_id: str) -> dict[str, Any]:
    """
    Get all lineage info for a run.

    Returns:
        Dict with datasets, preprocessing, and git info
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tags = run.data.tags

    lineage = {
        "datasets": {},
        "preprocessing": {},
        "git": {},
    }

    for key, value in tags.items():
        if key.startswith("dataset."):
            parts = key.split(".")
            if len(parts) >= 4:
                layer, table, field = parts[1], parts[2], parts[3]
                if layer not in lineage["datasets"]:
                    lineage["datasets"][layer] = {}
                if table not in lineage["datasets"][layer]:
                    lineage["datasets"][layer][table] = {}
                lineage["datasets"][layer][table][field] = value

        elif key.startswith("preprocessing."):
            field = key[len("preprocessing."):]
            lineage["preprocessing"][field] = value

        elif key.startswith("git."):
            field = key[len("git."):]
            lineage["git"][field] = value

    return lineage


def _file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]  # Short hash
