"""
VITS Loss Functions.

Stage 4.0 (core): mel reconstruction + KL divergence
Stage 4.1 (gan): + adversarial + feature matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_mel_from_audio(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    sample_rate: int = 22050,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """
    Compute mel spectrogram from audio.

    Args:
        audio: [B, 1, T] or [B, T] waveform

    Returns:
        mel: [B, n_mels, T_mel] mel spectrogram
    """
    if audio.dim() == 3:
        audio = audio.squeeze(1)

    # Create mel filterbank
    import torchaudio.transforms as T

    # Use torchaudio's MelSpectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=1.0,  # Magnitude spectrogram
    ).to(audio.device)

    mel = mel_transform(audio)

    # Log mel
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel


def mel_reconstruction_loss(
    pred_audio: torch.Tensor,
    target_mel: torch.Tensor,
    mel_config: dict,
) -> torch.Tensor:
    """
    Mel reconstruction loss.

    Compute mel from predicted audio and compare to target mel.

    Args:
        pred_audio: [B, 1, T_audio] predicted waveform
        target_mel: [B, n_mels, T_mel] target mel spectrogram
        mel_config: Dict with mel parameters

    Returns:
        loss: Scalar L1 loss
    """
    # Compute mel from predicted audio
    pred_mel = compute_mel_from_audio(
        pred_audio,
        n_fft=mel_config.get("n_fft", 1024),
        hop_length=mel_config.get("hop_length", 256),
        win_length=mel_config.get("win_length", 1024),
        n_mels=mel_config.get("n_mels", 80),
        sample_rate=mel_config.get("sample_rate", 22050),
        f_min=mel_config.get("f_min", 0.0),
        f_max=mel_config.get("f_max", 8000.0),
    )

    # Handle length mismatch (due to hop_length rounding)
    min_len = min(pred_mel.size(-1), target_mel.size(-1))
    pred_mel = pred_mel[..., :min_len]
    target_mel = target_mel[..., :min_len]

    return F.l1_loss(pred_mel, target_mel)


def kl_divergence_loss(
    posterior_mean: torch.Tensor,
    posterior_log_var: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_var: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    KL divergence between posterior and prior.

    KL(q || p) where q = N(posterior_mean, posterior_var)
                     p = N(prior_mean, prior_var)

    Args:
        posterior_mean: [B, C, T_mel] posterior mean
        posterior_log_var: [B, C, T_mel] posterior log variance
        prior_mean: [B, C, T_mel] prior mean (aligned via MAS)
        prior_log_var: [B, C, T_mel] prior log variance (aligned via MAS)
        mask: [B, T_mel] valid positions

    Returns:
        kl: Scalar KL divergence loss
    """
    posterior_var = torch.exp(posterior_log_var)
    prior_var = torch.exp(prior_log_var)

    # KL divergence for Gaussians
    kl = 0.5 * (
        prior_log_var - posterior_log_var
        + posterior_var / prior_var
        + (posterior_mean - prior_mean) ** 2 / prior_var
        - 1.0
    )

    if mask is not None:
        kl = kl * mask.unsqueeze(1).float()
        kl = kl.sum() / mask.sum() / kl.size(1)  # Average over valid positions and channels
    else:
        kl = kl.mean()

    return kl


def duration_loss(
    log_dur_pred: torch.Tensor,
    dur_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Duration prediction loss.

    MSE on log-scale durations.

    Args:
        log_dur_pred: [B, T_text] predicted log durations
        dur_target: [B, T_text] target durations (frames)
        mask: [B, T_text] valid positions

    Returns:
        loss: Scalar MSE loss
    """
    log_dur_target = torch.log(dur_target.float() + 1.0)

    loss = (log_dur_pred - log_dur_target) ** 2

    if mask is not None:
        loss = (loss * mask.float()).sum() / mask.sum()
    else:
        loss = loss.mean()

    return loss


def vits_loss_core(
    pred_audio: torch.Tensor,
    target_mel: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_log_var: torch.Tensor,
    prior_mean_aligned: torch.Tensor,
    prior_log_var_aligned: torch.Tensor,
    log_dur_pred: torch.Tensor,
    dur_target: torch.Tensor,
    mel_mask: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
    mel_config: Optional[dict] = None,
    kl_weight: float = 1.0,
    dur_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Complete VITS core loss (Stage 4.0, no discriminator).

    Args:
        pred_audio: [B, 1, T_audio] predicted waveform
        target_mel: [B, n_mels, T_mel] target mel spectrogram
        posterior_mean/log_var: Posterior distribution params
        prior_mean/log_var_aligned: Prior params aligned to mel length
        log_dur_pred: [B, T_text] predicted log durations
        dur_target: [B, T_text] target durations from MAS
        mel_mask: [B, T_mel] valid mel positions
        text_mask: [B, T_text] valid text positions
        mel_config: Mel spectrogram config dict
        kl_weight: Weight for KL loss
        dur_weight: Weight for duration loss

    Returns:
        total_loss: Scalar total loss
        loss_dict: Dict of individual losses
    """
    mel_config = mel_config or {}

    # Mel reconstruction loss
    mel_loss = mel_reconstruction_loss(pred_audio, target_mel, mel_config)

    # KL divergence loss
    kl_loss = kl_divergence_loss(
        posterior_mean, posterior_log_var,
        prior_mean_aligned, prior_log_var_aligned,
        mel_mask,
    )

    # Duration loss
    dur_loss = duration_loss(log_dur_pred, dur_target, text_mask)

    # Total loss
    total_loss = mel_loss + kl_weight * kl_loss + dur_weight * dur_loss

    loss_dict = {
        "mel_loss": mel_loss.item(),
        "kl_loss": kl_loss.item(),
        "dur_loss": dur_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


# =============================================================================
# Stage 4.1: GAN losses (to be added later)
# =============================================================================


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """
    Discriminator loss (hinge).

    Args:
        disc_real_outputs: List of discriminator outputs for real audio
        disc_fake_outputs: List of discriminator outputs for fake audio

    Returns:
        loss: Total discriminator loss
        loss_dict: Per-discriminator losses
    """
    loss = 0
    loss_dict = {}

    for i, (real, fake) in enumerate(zip(disc_real_outputs, disc_fake_outputs)):
        r_loss = torch.mean(F.relu(1 - real))
        f_loss = torch.mean(F.relu(1 + fake))
        loss_dict[f"d_real_{i}"] = r_loss.item()
        loss_dict[f"d_fake_{i}"] = f_loss.item()
        loss = loss + r_loss + f_loss

    loss_dict["d_total"] = loss.item()
    return loss, loss_dict


def generator_adversarial_loss(
    disc_fake_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """
    Generator adversarial loss.

    Args:
        disc_fake_outputs: List of discriminator outputs for fake audio

    Returns:
        loss: Adversarial loss for generator
        loss_dict: Per-discriminator losses
    """
    loss = 0
    loss_dict = {}

    for i, fake in enumerate(disc_fake_outputs):
        l = torch.mean(F.relu(1 - fake))
        loss_dict[f"g_adv_{i}"] = l.item()
        loss = loss + l

    loss_dict["g_adv_total"] = loss.item()
    return loss, loss_dict


def feature_matching_loss(
    disc_real_fmaps: list[list[torch.Tensor]],
    disc_fake_fmaps: list[list[torch.Tensor]],
) -> tuple[torch.Tensor, dict]:
    """
    Feature matching loss.

    L1 loss between discriminator feature maps.

    Args:
        disc_real_fmaps: Feature maps from discriminator for real audio
        disc_fake_fmaps: Feature maps from discriminator for fake audio

    Returns:
        loss: Feature matching loss
        loss_dict: Per-layer losses
    """
    loss = 0
    loss_dict = {}

    for i, (real_fmaps, fake_fmaps) in enumerate(zip(disc_real_fmaps, disc_fake_fmaps)):
        for j, (real_fmap, fake_fmap) in enumerate(zip(real_fmaps, fake_fmaps)):
            l = F.l1_loss(fake_fmap, real_fmap.detach())
            loss_dict[f"fm_{i}_{j}"] = l.item()
            loss = loss + l

    loss_dict["fm_total"] = loss.item()
    return loss, loss_dict
