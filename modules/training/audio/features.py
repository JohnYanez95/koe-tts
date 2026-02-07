"""
Audio feature extraction (mel-spectrogram).

Provides consistent mel extraction for training and inference.
Uses torchaudio for GPU-compatible extraction.
"""

import torch
import torchaudio
from dataclasses import dataclass
from typing import Optional


@dataclass
class MelConfig:
    """Mel-spectrogram configuration."""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    power: float = 1.0  # 1 for energy, 2 for power
    normalized: bool = False
    center: bool = True
    pad_mode: str = "reflect"

    # Normalization (applied after mel extraction)
    mel_min: float = 1e-5  # Floor before log

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length


# Default config matching common TTS settings
DEFAULT_MEL_CONFIG = MelConfig()


class MelExtractor:
    """
    Extract mel-spectrograms from audio waveforms.

    Thread-safe and GPU-compatible via torchaudio.
    """

    def __init__(self, config: Optional[MelConfig] = None, device: str = "cpu"):
        self.config = config or DEFAULT_MEL_CONFIG
        self.device = device

        # Create mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=self.config.power,
            normalized=self.config.normalized,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
        ).to(device)

        # Resampler cache (created on demand)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Get or create resampler for given sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, self.config.sample_rate
            ).to(self.device)
        return self._resamplers[orig_sr]

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract mel-spectrogram from waveform.

        Args:
            waveform: Audio tensor [C, T] or [T]
            sample_rate: Original sample rate (resamples if different from config)

        Returns:
            Mel-spectrogram [n_mels, frames]
        """
        # Ensure 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Move to device
        waveform = waveform.to(self.device)

        # Resample if needed
        if sample_rate is not None and sample_rate != self.config.sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        # Extract mel
        mel = self.mel_transform(waveform)  # [1, n_mels, frames]
        mel = mel.squeeze(0)  # [n_mels, frames]

        # Apply log compression with floor
        mel = torch.clamp(mel, min=self.config.mel_min)
        mel = torch.log(mel)

        return mel

    def frames_to_samples(self, n_frames: int) -> int:
        """Convert frame count to sample count."""
        return n_frames * self.config.hop_length

    def samples_to_frames(self, n_samples: int) -> int:
        """Convert sample count to frame count."""
        return n_samples // self.config.hop_length


def load_audio(
    audio_path: str,
    target_sr: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    """
    Load audio file and optionally resample.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (None = keep original)

    Returns:
        Tuple of (waveform [C, T], sample_rate)
    """
    waveform, sr = torchaudio.load(audio_path)

    if target_sr is not None and sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform, sr


def compute_mel(
    audio_path: str,
    config: Optional[MelConfig] = None,
) -> torch.Tensor:
    """
    Convenience function: load audio and compute mel.

    Args:
        audio_path: Path to audio file
        config: Mel config (uses default if None)

    Returns:
        Mel-spectrogram [n_mels, frames]
    """
    config = config or DEFAULT_MEL_CONFIG
    waveform, sr = load_audio(audio_path)
    extractor = MelExtractor(config)
    return extractor(waveform, sample_rate=sr)
