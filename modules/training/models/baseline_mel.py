"""
Baseline Mel Model - Simple encoder-decoder for pipeline validation.

Architecture:
- Phoneme encoder: Embedding + ConvBank
- Length regulator: Interpolate to target mel length (teacher-forced)
- Mel decoder: ConvBank + Linear projection

This is NOT a production model. It validates:
1. Data pipeline works
2. Training loop works
3. Checkpointing works
4. Loss converges

For real TTS, use VITS or similar.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaselineModelConfig:
    """Baseline model configuration."""
    vocab_size: int = 45  # From PHONEME_VOCAB_SIZE
    n_mels: int = 80
    d_model: int = 256
    n_conv_layers_enc: int = 6
    n_conv_layers_dec: int = 4
    kernel_size: int = 5
    dropout: float = 0.1


class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        return self.dropout(F.relu(self.bn(self.conv(x))))


class ConvBank(nn.Module):
    """Stack of ConvBlocks with residual connections."""

    def __init__(
        self,
        channels: int,
        n_layers: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock(channels, channels, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        for layer in self.layers:
            x = x + layer(x)  # Residual
        return x


class PhonemeEncoder(nn.Module):
    """Encode phoneme sequence to hidden representations."""

    def __init__(self, config: BaselineModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout)
        self.conv_bank = ConvBank(
            config.d_model,
            config.n_conv_layers_enc,
            config.kernel_size,
            config.dropout,
        )

    def forward(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            phonemes: [B, T_phone] token IDs
            phoneme_mask: [B, T_phone] True for valid positions

        Returns:
            [B, T_phone, d_model] encoded representations
        """
        # Embed and add positional encoding
        x = self.embedding(phonemes)  # [B, T, d_model]
        x = self.pos_encoding(x)

        # Conv bank expects [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv_bank(x)
        x = x.transpose(1, 2)  # [B, T, d_model]

        # Apply mask
        if phoneme_mask is not None:
            x = x * phoneme_mask.unsqueeze(-1).float()

        return x


class LengthRegulator(nn.Module):
    """
    Expand phoneme sequence to mel length via interpolation.

    This is a crude approximation - real models use duration prediction
    or learned upsampling. For baseline validation only.
    """

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_len: int,
        phoneme_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: [B, T_phone, d_model]
            target_len: Target sequence length (mel frames)
            phoneme_mask: [B, T_phone] True for valid positions

        Returns:
            [B, target_len, d_model]
        """
        # Simple interpolation - works but loses temporal detail
        # [B, T_phone, d_model] -> [B, d_model, T_phone]
        x = encoder_out.transpose(1, 2)

        # Interpolate to target length
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)

        # [B, d_model, target_len] -> [B, target_len, d_model]
        return x.transpose(1, 2)


class MelDecoder(nn.Module):
    """Decode to mel spectrogram."""

    def __init__(self, config: BaselineModelConfig):
        super().__init__()
        self.conv_bank = ConvBank(
            config.d_model,
            config.n_conv_layers_dec,
            config.kernel_size,
            config.dropout,
        )
        self.mel_linear = nn.Linear(config.d_model, config.n_mels)

    def forward(
        self,
        x: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T_mel, d_model]
            mel_mask: [B, T_mel] True for valid frames

        Returns:
            [B, n_mels, T_mel] predicted mel spectrogram
        """
        # Conv bank expects [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv_bank(x)
        x = x.transpose(1, 2)  # [B, T_mel, d_model]

        # Project to mel
        mel = self.mel_linear(x)  # [B, T_mel, n_mels]
        mel = mel.transpose(1, 2)  # [B, n_mels, T_mel]

        # Apply mask (set padded frames to 0)
        if mel_mask is not None:
            mel = mel * mel_mask.unsqueeze(1).float()

        return mel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaselineMelModel(nn.Module):
    """
    Baseline TTS model for pipeline validation.

    Non-autoregressive teacher-forced mel prediction.
    """

    def __init__(self, config: BaselineModelConfig):
        super().__init__()
        self.config = config
        self.encoder = PhonemeEncoder(config)
        self.length_regulator = LengthRegulator()
        self.decoder = MelDecoder(config)

    def forward(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: torch.Tensor,
        mel_lens: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with teacher-forced target length.

        Args:
            phonemes: [B, T_phone] token IDs
            phoneme_mask: [B, T_phone] True for valid positions
            mel_lens: [B] target mel lengths (for length regulation)
            mel_mask: [B, T_mel] True for valid mel frames

        Returns:
            [B, n_mels, T_mel] predicted mel spectrogram
        """
        # Encode phonemes
        encoder_out = self.encoder(phonemes, phoneme_mask)

        # Expand to target mel length
        max_mel_len = mel_lens.max().item()
        expanded = self.length_regulator(encoder_out, max_mel_len, phoneme_mask)

        # Decode to mel
        mel_pred = self.decoder(expanded, mel_mask)

        return mel_pred

    @torch.no_grad()
    def inference(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference without ground truth mel length.

        Args:
            phonemes: [B, T_phone] token IDs
            phoneme_mask: [B, T_phone] True for valid positions
            target_len: Override target length (default: estimate from phonemes)
            length_scale: Multiply estimated length (slower/faster speech)

        Returns:
            [B, n_mels, T_mel] predicted mel spectrogram
        """
        self.eval()

        # Encode
        encoder_out = self.encoder(phonemes, phoneme_mask)

        # Estimate target length if not provided
        # Heuristic: ~10 mel frames per phoneme (86 fps / ~8 phones per second)
        if target_len is None:
            if phoneme_mask is not None:
                phone_lens = phoneme_mask.sum(dim=1)
            else:
                phone_lens = torch.tensor([phonemes.size(1)], device=phonemes.device)
            target_len = int(phone_lens.max().item() * 10 * length_scale)

        # Expand and decode
        expanded = self.length_regulator(encoder_out, target_len, phoneme_mask)
        mel_pred = self.decoder(expanded, None)

        return mel_pred

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> BaselineMelModel:
    """Create model from config dict."""
    from modules.training.dataloading import PHONEME_VOCAB_SIZE

    model_config = BaselineModelConfig(
        vocab_size=PHONEME_VOCAB_SIZE,
        n_mels=config.get("mel", {}).get("n_mels", 80),
        d_model=config.get("model", {}).get("d_model", 256),
        n_conv_layers_enc=config.get("model", {}).get("n_conv_layers_enc", 6),
        n_conv_layers_dec=config.get("model", {}).get("n_conv_layers_dec", 4),
        kernel_size=config.get("model", {}).get("kernel_size", 5),
        dropout=config.get("model", {}).get("dropout", 0.1),
    )

    return BaselineMelModel(model_config)
