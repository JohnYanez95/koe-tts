"""
Vocoder utilities for mel-to-waveform conversion.

v0: Griffin-Lim algorithm (fast, low quality - for sanity checks)
Future: Neural vocoder (HiFi-GAN, etc.)
"""

import torch
import torchaudio
from typing import Optional

from .features import MelConfig, DEFAULT_MEL_CONFIG


class GriffinLimVocoder:
    """
    Griffin-Lim vocoder for mel-to-waveform conversion.

    This is NOT high quality - use only for sanity checks during development.
    Replace with neural vocoder for actual synthesis.
    """

    def __init__(
        self,
        config: Optional[MelConfig] = None,
        n_iter: int = 32,
        device: str = "cpu",
    ):
        self.config = config or DEFAULT_MEL_CONFIG
        self.n_iter = n_iter
        self.device = device

        # Inverse mel transform
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        ).to(device)

        # Griffin-Lim
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            n_iter=self.n_iter,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            power=self.config.power,
        ).to(device)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform.

        Args:
            mel: Log mel-spectrogram [n_mels, frames] or [batch, n_mels, frames]

        Returns:
            Waveform [samples] or [batch, samples]
        """
        # Handle batch dimension
        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        mel = mel.to(self.device)

        # Undo log compression
        mel = torch.exp(mel)

        # Inverse mel to linear spectrogram
        spec = self.inverse_mel(mel)

        # Griffin-Lim to waveform
        waveforms = []
        for i in range(spec.shape[0]):
            waveform = self.griffin_lim(spec[i])
            waveforms.append(waveform)

        waveform = torch.stack(waveforms)

        if squeeze:
            waveform = waveform.squeeze(0)

        return waveform


def mel_to_audio(
    mel: torch.Tensor,
    config: Optional[MelConfig] = None,
    n_iter: int = 32,
) -> torch.Tensor:
    """
    Convenience function: convert mel to audio via Griffin-Lim.

    Args:
        mel: Log mel-spectrogram [n_mels, frames]
        config: Mel config
        n_iter: Griffin-Lim iterations

    Returns:
        Waveform tensor [samples]
    """
    vocoder = GriffinLimVocoder(config=config, n_iter=n_iter)
    return vocoder(mel)


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 22050,
) -> None:
    """
    Save waveform to file.

    Args:
        waveform: Audio tensor [samples] or [C, samples]
        path: Output path
        sample_rate: Sample rate
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Normalize to prevent clipping
    waveform = waveform / (waveform.abs().max() + 1e-8)

    torchaudio.save(path, waveform.cpu(), sample_rate)
