"""
Checkpoint management utilities.
"""

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    checkpoint_path: Path,
    model_state: dict,
    optimizer_state: Optional[dict] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[dict] = None,
    config: Optional[dict] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model_state: Model state dict
        optimizer_state: Optimizer state dict
        epoch: Current epoch
        step: Current global step
        metrics: Training metrics
        config: Training config
    """
    checkpoint = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "step": step,
    }

    if optimizer_state:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if metrics:
        checkpoint["metrics"] = metrics
    if config:
        checkpoint["config"] = config

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to map tensors to

    Returns:
        Checkpoint dict
    """
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
