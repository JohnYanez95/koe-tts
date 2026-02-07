"""
Evaluation metrics for TTS.

Computes mel-spectrogram and audio quality metrics.
"""

import torch
import numpy as np
from typing import Optional


def compute_mel_metrics(
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor,
    mel_mask: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute mel-spectrogram evaluation metrics.

    Args:
        pred_mel: Predicted mel [n_mels, T] or [B, n_mels, T]
        target_mel: Target mel [n_mels, T] or [B, n_mels, T]
        mel_mask: Valid frame mask [T] or [B, T], True for valid

    Returns:
        Dict with metrics: mel_l1, mel_l2, mel_mean_pred, mel_std_pred,
                          mel_mean_tgt, mel_std_tgt, snr_proxy
    """
    # Handle batch dimension
    if pred_mel.dim() == 2:
        pred_mel = pred_mel.unsqueeze(0)
        target_mel = target_mel.unsqueeze(0)
        if mel_mask is not None:
            mel_mask = mel_mask.unsqueeze(0)

    B, n_mels, T = pred_mel.shape

    # Create mask if not provided
    if mel_mask is None:
        mel_mask = torch.ones(B, T, dtype=torch.bool, device=pred_mel.device)

    # Expand mask to [B, n_mels, T]
    mask = mel_mask.unsqueeze(1).expand(-1, n_mels, -1).float()
    n_valid = mask.sum()

    # Masked L1 and L2
    diff = pred_mel - target_mel
    l1 = (diff.abs() * mask).sum() / n_valid
    l2 = ((diff ** 2) * mask).sum() / n_valid

    # Mel statistics (on valid frames only)
    pred_masked = pred_mel * mask
    tgt_masked = target_mel * mask

    mel_mean_pred = pred_masked.sum() / n_valid
    mel_mean_tgt = tgt_masked.sum() / n_valid

    # Variance (using masked mean)
    pred_var = ((pred_mel - mel_mean_pred) ** 2 * mask).sum() / n_valid
    tgt_var = ((target_mel - mel_mean_tgt) ** 2 * mask).sum() / n_valid

    mel_std_pred = pred_var.sqrt()
    mel_std_tgt = tgt_var.sqrt()

    # SNR proxy: 10 * log10(var(target) / var(error))
    error_var = ((diff ** 2) * mask).sum() / n_valid
    snr_proxy = 10 * torch.log10(tgt_var / (error_var + 1e-8))

    return {
        "mel_l1": l1.item(),
        "mel_l2": l2.item(),
        "mel_mean_pred": mel_mean_pred.item(),
        "mel_std_pred": mel_std_pred.item(),
        "mel_mean_tgt": mel_mean_tgt.item(),
        "mel_std_tgt": mel_std_tgt.item(),
        "snr_proxy_db": snr_proxy.item(),
    }


def compute_audio_metrics(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    silence_threshold: float = 0.01,
) -> dict:
    """
    Compute audio quality metrics.

    Args:
        waveform: Audio tensor [T] or [C, T]
        sample_rate: Sample rate in Hz
        silence_threshold: Threshold for silence detection (fraction of peak)

    Returns:
        Dict with metrics: duration_sec, rms, peak, silence_pct
    """
    # Handle channel dimension
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)  # Mix to mono

    waveform = waveform.float()
    n_samples = waveform.numel()

    # Duration
    duration_sec = n_samples / sample_rate

    # RMS (root mean square)
    rms = torch.sqrt((waveform ** 2).mean())

    # Peak (maximum absolute amplitude)
    peak = waveform.abs().max()

    # Silence percentage
    # Silence = frames where |sample| < threshold * peak
    threshold = silence_threshold * peak
    silence_frames = (waveform.abs() < threshold).sum()
    silence_pct = 100.0 * silence_frames / n_samples

    return {
        "duration_sec": duration_sec,
        "rms": rms.item(),
        "peak": peak.item(),
        "silence_pct": silence_pct.item(),
    }


def aggregate_metrics(sample_metrics: list[dict]) -> dict:
    """
    Aggregate per-sample metrics into summary statistics.

    Args:
        sample_metrics: List of per-sample metric dicts

    Returns:
        Dict with mean, std, min, max for each metric
    """
    if not sample_metrics:
        return {}

    # Collect all metric keys (excluding string fields)
    keys = sample_metrics[0].keys()
    numeric_keys = []
    for key in keys:
        val = sample_metrics[0].get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            numeric_keys.append(key)

    result = {}
    for key in numeric_keys:
        values = [m[key] for m in sample_metrics if key in m and isinstance(m[key], (int, float))]
        if not values:
            continue

        arr = np.array(values, dtype=np.float64)
        result[f"{key}_mean"] = float(np.mean(arr))
        result[f"{key}_std"] = float(np.std(arr))
        result[f"{key}_min"] = float(np.min(arr))
        result[f"{key}_max"] = float(np.max(arr))

    return result
