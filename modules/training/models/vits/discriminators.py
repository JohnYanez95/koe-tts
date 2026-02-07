"""
VITS Discriminators (Stage 4.1).

Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD)
for adversarial training.

These are not used in Stage 4.0 (core) - only mel reconstruction + KL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiscriminatorP(nn.Module):
    """
    Period discriminator sub-network.

    Reshapes audio to 2D with given period and applies 2D convolutions.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period

        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] audio

        Returns:
            output: [B, 1, T', 1] discriminator output
            fmaps: List of feature maps for feature matching loss
        """
        fmaps = []

        # Reshape to 2D
        B, C, T = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            T = T + n_pad
        x = x.view(B, C, T // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD).

    Uses multiple period discriminators to capture periodic patterns.
    """

    def __init__(
        self,
        periods: tuple = (2, 3, 5, 7, 11),
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm)
            for p in periods
        ])

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[list, list, list, list]:
        """
        Args:
            y: [B, 1, T] real audio
            y_hat: [B, 1, T] generated audio

        Returns:
            y_d_rs: Real outputs per discriminator
            y_d_gs: Generated outputs per discriminator
            fmap_rs: Real feature maps per discriminator
            fmap_gs: Generated feature maps per discriminator
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """
    Scale discriminator sub-network.

    Standard 1D convolutions at different scales.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm_f(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm_f(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] audio

        Returns:
            output: [B, 1, T'] discriminator output
            fmaps: List of feature maps
        """
        fmaps = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD).

    Operates on audio at different scales (original + downsampled).
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),  # First one uses spectral norm
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
        ])

        self.pooling = nn.AvgPool1d(4, 2, 2)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[list, list, list, list]:
        """
        Args:
            y: [B, 1, T] real audio
            y_hat: [B, 1, T] generated audio

        Returns:
            y_d_rs: Real outputs per scale
            y_d_gs: Generated outputs per scale
            fmap_rs: Real feature maps per scale
            fmap_gs: Generated feature maps per scale
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                y = self.pooling(y)
                y_hat = self.pooling(y_hat)

            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
