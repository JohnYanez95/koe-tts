"""
VITS Generator (HiFiGAN-style).

Converts latent z to waveform using upsampling + residual blocks.

Architecture:
- Input projection
- Upsampling blocks (transposed convolutions)
- Multi-Receptive Field Fusion (MRF) blocks
- Output projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return (kernel_size * dilation - dilation) // 2


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    """Initialize conv weights."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple = (1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=get_padding(kernel_size, d)
            )
            for d in dilations
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1, padding=get_padding(kernel_size, 1)
            )
            for _ in dilations
        ])

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = x + xt
        return x


class MRFBlock(nn.Module):
    """Multi-Receptive Field Fusion block."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple = (3, 7, 11),
        dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d)
            for k, d in zip(kernel_sizes, dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for resblock in self.resblocks:
            if out is None:
                out = resblock(x)
            else:
                out = out + resblock(x)
        return out / len(self.resblocks)


class Generator(nn.Module):
    """
    HiFiGAN-style generator for VITS.

    Converts latent representations to waveform.

    Args:
        latent_dim: Input latent dimension
        upsample_rates: Upsampling rates per block
        upsample_kernel_sizes: Kernel sizes for transposed convs
        upsample_initial_channel: Initial number of channels
        resblock_kernel_sizes: Kernel sizes for MRF blocks
        resblock_dilations: Dilations for MRF blocks
    """

    def __init__(
        self,
        latent_dim: int = 192,
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)

        # Input projection
        self.input_proj = nn.Conv1d(latent_dim, upsample_initial_channel, 7, padding=3)

        # Upsampling blocks
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        channels = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    channels, channels // 2,
                    kernel, stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            channels = channels // 2

            self.mrfs.append(
                MRFBlock(channels, resblock_kernel_sizes, resblock_dilations)
            )

        # Output projection
        self.output_proj = nn.Conv1d(channels, 1, 7, padding=3)

        # Initialize
        self.input_proj.apply(init_weights)
        self.ups.apply(init_weights)
        self.output_proj.apply(init_weights)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate waveform from latent.

        Args:
            z: [B, latent_dim, T_latent] latent representation
            mask: [B, T_latent] valid positions (optional)

        Returns:
            audio: [B, 1, T_audio] generated waveform
        """
        x = self.input_proj(z)

        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = mrf(x)

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.output_proj(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for up in self.ups:
            if hasattr(up, 'weight_g'):
                nn.utils.remove_weight_norm(up)
        for mrf in self.mrfs:
            for resblock in mrf.resblocks:
                for conv in resblock.convs1:
                    if hasattr(conv, 'weight_g'):
                        nn.utils.remove_weight_norm(conv)
                for conv in resblock.convs2:
                    if hasattr(conv, 'weight_g'):
                        nn.utils.remove_weight_norm(conv)
