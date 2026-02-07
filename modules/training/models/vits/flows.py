"""
VITS Normalizing Flows.

Residual coupling blocks that transform between prior and posterior
distributions. Used to increase the expressiveness of the latent space.

Architecture:
- Residual coupling layers with WaveNet-style transforms
- Flip operation to alternate which half is transformed
"""

import torch
import torch.nn as nn
from typing import Optional


class WaveNetTransform(nn.Module):
    """WaveNet-style transform network for coupling layer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        n_layers: int = 4,
        dilation_rate: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_channels // 2, hidden_channels, 1)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels, 2 * hidden_channels, kernel_size,
                        dilation=dilation, padding=padding
                    ),
                    nn.Dropout(dropout),
                )
            )

        # Output: mean and log_scale for affine transform
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, 1)
        self.output_proj.weight.data.zero_()
        self.output_proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C/2, T] input (half of channels)
            mask: [B, T] valid positions

        Returns:
            params: [B, C, T] affine parameters (mean, log_scale)
        """
        h = self.input_proj(x)

        for layer in self.layers:
            h_out = layer(h)
            h_tanh, h_sigmoid = h_out.chunk(2, dim=1)
            h = h + torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)

        params = self.output_proj(h)

        if mask is not None:
            params = params * mask.unsqueeze(1).float()

        return params


class ResidualCouplingBlock(nn.Module):
    """
    Single residual coupling layer.

    Splits input into two halves, uses one to parameterize
    an affine transform of the other.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        n_layers: int = 4,
        dilation_rate: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.half_channels = channels // 2

        self.transform = WaveNetTransform(
            channels, hidden_channels, kernel_size, n_layers, dilation_rate, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or reverse pass.

        Args:
            x: [B, C, T] input
            mask: [B, T] valid positions
            reverse: If True, run inverse transform

        Returns:
            y: [B, C, T] transformed output
            log_det: [B] log determinant of Jacobian
        """
        x0, x1 = x.split(self.half_channels, dim=1)

        # Get affine parameters from x0
        params = self.transform(x0, mask)
        mean, log_scale = params.split(self.half_channels, dim=1)

        if not reverse:
            # Forward: y1 = x1 * exp(log_scale) + mean
            y1 = x1 * torch.exp(log_scale) + mean
            log_det = log_scale.sum(dim=[1, 2])
        else:
            # Reverse: x1 = (y1 - mean) * exp(-log_scale)
            y1 = (x1 - mean) * torch.exp(-log_scale)
            log_det = -log_scale.sum(dim=[1, 2])

        y = torch.cat([x0, y1], dim=1)

        if mask is not None:
            y = y * mask.unsqueeze(1).float()

        return y, log_det


class Flip(nn.Module):
    """Flip channels to alternate which half is transformed."""

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Flip is its own inverse
        x = torch.flip(x, dims=[1])
        log_det = torch.zeros(x.size(0), device=x.device)
        return x, log_det


class ResidualCouplingBlocks(nn.Module):
    """
    Stack of residual coupling blocks with flips.

    This is the complete flow that transforms between prior and posterior.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        n_layers: int = 4,
        n_flows: int = 4,
        dilation_rate: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels, hidden_channels, kernel_size, n_layers, dilation_rate, dropout
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or reverse through all flows.

        Args:
            x: [B, C, T] input
            mask: [B, T] valid positions
            reverse: If True, run in reverse order

        Returns:
            y: [B, C, T] output
            total_log_det: [B] total log determinant
        """
        total_log_det = torch.zeros(x.size(0), device=x.device)

        flows = reversed(self.flows) if reverse else self.flows

        for flow in flows:
            x, log_det = flow(x, mask, reverse=reverse)
            total_log_det = total_log_det + log_det

        return x, total_log_det
