"""
Training metrics computation.
"""

from typing import Optional

import numpy as np


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Compute training/evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metric_names: Metrics to compute (default: all)

    Returns:
        Dict of metric name -> value
    """
    if metric_names is None:
        metric_names = ["mse", "mae"]

    metrics = {}

    if "mse" in metric_names:
        metrics["mse"] = float(np.mean((predictions - targets) ** 2))

    if "mae" in metric_names:
        metrics["mae"] = float(np.mean(np.abs(predictions - targets)))

    if "rmse" in metric_names:
        metrics["rmse"] = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    return metrics


def compute_mcd(
    pred_mel: np.ndarray,
    target_mel: np.ndarray,
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD).

    Args:
        pred_mel: Predicted mel spectrogram
        target_mel: Target mel spectrogram

    Returns:
        MCD value in dB
    """
    # TODO: Implement proper MCD computation
    # This requires mel -> mfcc conversion
    return 0.0


def compute_f0_metrics(
    pred_f0: np.ndarray,
    target_f0: np.ndarray,
) -> dict[str, float]:
    """
    Compute F0 (pitch) metrics.

    Args:
        pred_f0: Predicted F0 contour
        target_f0: Target F0 contour

    Returns:
        Dict with F0 RMSE, correlation, etc.
    """
    # Filter voiced frames
    voiced_mask = (pred_f0 > 0) & (target_f0 > 0)

    if not np.any(voiced_mask):
        return {"f0_rmse": 0.0, "f0_corr": 0.0}

    pred_voiced = pred_f0[voiced_mask]
    target_voiced = target_f0[voiced_mask]

    # RMSE in Hz
    f0_rmse = float(np.sqrt(np.mean((pred_voiced - target_voiced) ** 2)))

    # Correlation
    f0_corr = float(np.corrcoef(pred_voiced, target_voiced)[0, 1])

    return {
        "f0_rmse": f0_rmse,
        "f0_corr": f0_corr,
    }
