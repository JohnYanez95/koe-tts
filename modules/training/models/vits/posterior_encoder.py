"""
VITS Posterior Encoder.

Encodes mel spectrograms to latent z using WaveNet-style architecture.
Outputs mean and log-variance for the posterior distribution.

Architecture:
- Input projection (n_mels -> hidden_channels)
- WaveNet blocks with dilated convolutions
- Output projection to mean/log_var
"""

import torch
import torch.nn as nn
from typing import Optional


class WaveNetBlock(nn.Module):
    """Single WaveNet residual block with gated activation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2

        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, C, T] input
            mask: [B, T] True for valid positions

        Returns:
            residual: [B, C, T] to add to next layer
            skip: [B, C, T] for skip connection
        """
        # Gated activation
        h = self.conv(x)
        h_tanh, h_sigmoid = h.chunk(2, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)
        h = self.dropout(h)

        # Apply mask if provided
        if mask is not None:
            h = h * mask.unsqueeze(1).float()

        # Residual and skip connections
        residual = self.res_conv(h)
        skip = self.skip_conv(h)

        # Layer norm (transpose for channel-last)
        residual = self.norm(residual.transpose(1, 2)).transpose(1, 2)

        return x + residual, skip


class PosteriorEncoder(nn.Module):
    """
    VITS posterior encoder.

    Encodes mel spectrograms to latent representations with
    mean and log-variance for VAE.

    Args:
        n_mels: Number of mel channels
        hidden_channels: Hidden dimension
        latent_dim: Latent space dimension
        kernel_size: Convolution kernel size
        n_layers: Number of WaveNet blocks
        dilation_rate: Dilation rate multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_channels: int = 192,
        latent_dim: int = 192,
        kernel_size: int = 5,
        n_layers: int = 16,
        dilation_rate: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Conv1d(n_mels, hidden_channels, 1)

        # WaveNet blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** (i % 4)  # Cycle dilations: 1, 2, 4, 8, 1, 2, ...
            self.blocks.append(
                WaveNetBlock(hidden_channels, kernel_size, dilation, dropout)
            )

        # Output projection
        self.output_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.proj_mean = nn.Conv1d(hidden_channels, latent_dim, 1)
        self.proj_log_var = nn.Conv1d(hidden_channels, latent_dim, 1)

    def forward(
        self,
        mel: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode mel spectrogram to latent.

        Args:
            mel: [B, n_mels, T_mel] mel spectrogram
            mel_mask: [B, T_mel] True for valid frames

        Returns:
            posterior_mean: [B, latent_dim, T_mel]
            posterior_log_var: [B, latent_dim, T_mel]
        """
        # Project to hidden
        x = self.input_proj(mel)

        if mel_mask is not None:
            x = x * mel_mask.unsqueeze(1).float()

        # WaveNet blocks with skip connections
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x, mel_mask)
            skip_sum = skip_sum + skip

        # Output projection
        x = self.output_proj(skip_sum)
        if mel_mask is not None:
            x = x * mel_mask.unsqueeze(1).float()

        # Project to mean/log_var
        posterior_mean = self.proj_mean(x)
        posterior_log_var = self.proj_log_var(x)

        return posterior_mean, posterior_log_var

    def sample(
        self,
        mel: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample z from posterior.

        Args:
            mel: [B, n_mels, T_mel] mel spectrogram
            mel_mask: [B, T_mel] True for valid frames

        Returns:
            z: [B, latent_dim, T_mel] sampled latent
        """
        mean, log_var = self.forward(mel, mel_mask)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        if mel_mask is not None:
            z = z * mel_mask.unsqueeze(1).float()

        return z
