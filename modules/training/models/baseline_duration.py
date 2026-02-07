"""
Baseline Duration Model - Encoder-decoder with duration prediction.

Architecture:
- Phoneme encoder: Embedding + ConvBank (shared with baseline_mel)
- Duration head: Per-token duration prediction (log-scale)
- Length regulator: Expand by predicted/teacher durations
- Mel decoder: ConvBank + Linear projection

This enables text-to-mel synthesis without forced alignments.
Uses uniform duration bootstrap as training target.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from .baseline_mel import (
    ConvBlock,
    ConvBank,
    PhonemeEncoder,
    MelDecoder,
    PositionalEncoding,
    BaselineModelConfig,
)


@dataclass
class DurationModelConfig(BaselineModelConfig):
    """Duration model configuration (extends baseline)."""
    dur_hidden: int = 256       # Hidden dim for duration predictor
    dur_kernel: int = 3         # Conv kernel for duration predictor
    dur_layers: int = 2         # Num conv layers in duration predictor
    dur_dropout: float = 0.1
    dur_min: int = 1            # Min frames per token (inference)
    dur_max: int = 300          # Max frames per token (inference)


class DurationPredictor(nn.Module):
    """
    Predict duration (frames) per phoneme token.

    Outputs log-scale durations for stability.
    Use softplus at inference to ensure positive values.
    """

    def __init__(self, config: DurationModelConfig):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    config.d_model if i == 0 else config.dur_hidden,
                    config.dur_hidden,
                    config.dur_kernel,
                    padding=config.dur_kernel // 2,
                ),
                nn.ReLU(),
                nn.BatchNorm1d(config.dur_hidden),
                nn.Dropout(config.dur_dropout),
            )
            for i in range(config.dur_layers)
        ])

        self.out = nn.Linear(config.dur_hidden, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Encoder output [B, T, d_model]
            mask: Valid token mask [B, T]

        Returns:
            Log-durations [B, T]
        """
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        for conv in self.convs:
            x = conv(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        # Project to scalar
        log_dur = self.out(x).squeeze(-1)  # [B, T]

        if mask is not None:
            log_dur = log_dur.masked_fill(~mask, 0.0)

        return log_dur


class LengthRegulatorWithDurations(nn.Module):
    """
    Expand encoder output by durations (discrete frame counts).

    Each token embedding is repeated by its duration.
    """

    def forward(
        self,
        encoder_out: torch.Tensor,
        durations: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: [B, T, d_model]
            durations: [B, T] integer frame counts per token
            max_len: Optional max output length (for padding)

        Returns:
            expanded: [B, L, d_model] where L = sum of durations (per sample)
            mel_lens: [B] actual expanded lengths
        """
        B, T, D = encoder_out.shape
        device = encoder_out.device

        # Compute output lengths
        mel_lens = durations.sum(dim=1)  # [B]

        if max_len is None:
            max_len = mel_lens.max().item()

        # Expand each sample
        expanded = torch.zeros(B, max_len, D, device=device, dtype=encoder_out.dtype)

        for b in range(B):
            pos = 0
            for t in range(T):
                dur = durations[b, t].item()
                if dur > 0 and pos < max_len:
                    end = min(pos + dur, max_len)
                    expanded[b, pos:end] = encoder_out[b, t]
                    pos = end

        return expanded, mel_lens


class BaselineDurationModel(nn.Module):
    """
    Baseline TTS model with duration prediction.

    Can run in two modes:
    1. Training: Use teacher durations, compute duration loss
    2. Inference: Predict durations, generate arbitrary length mel
    """

    def __init__(self, config: DurationModelConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = PhonemeEncoder(config)

        # Duration predictor
        self.duration_predictor = DurationPredictor(config)

        # Length regulator
        self.length_regulator = LengthRegulatorWithDurations()

        # Mel decoder
        self.decoder = MelDecoder(config)

    def forward(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: torch.Tensor,
        durations: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher durations.

        Args:
            phonemes: [B, T_phone] token IDs
            phoneme_mask: [B, T_phone] True for valid positions
            durations: [B, T_phone] teacher duration targets (frames)
            mel_mask: [B, T_mel] True for valid mel frames

        Returns:
            mel_pred: [B, n_mels, T_mel]
            log_dur_pred: [B, T_phone] predicted log-durations
        """
        # Encode phonemes
        encoder_out = self.encoder(phonemes, phoneme_mask)  # [B, T, d_model]

        # Predict durations (for loss computation)
        log_dur_pred = self.duration_predictor(encoder_out, phoneme_mask)

        # Expand using teacher durations
        max_mel_len = durations.sum(dim=1).max().item()
        expanded, _ = self.length_regulator(encoder_out, durations, max_mel_len)

        # Decode to mel
        mel_pred = self.decoder(expanded, mel_mask)

        return mel_pred, log_dur_pred

    @torch.no_grad()
    def inference(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with predicted durations.

        Args:
            phonemes: [B, T_phone] token IDs
            phoneme_mask: [B, T_phone] True for valid positions
            duration_scale: Multiply durations (slower/faster speech)

        Returns:
            mel_pred: [B, n_mels, T_mel]
            durations: [B, T_phone] predicted integer durations
        """
        self.eval()

        if phoneme_mask is None:
            phoneme_mask = torch.ones_like(phonemes, dtype=torch.bool)

        # Encode
        encoder_out = self.encoder(phonemes, phoneme_mask)

        # Predict durations
        log_dur_pred = self.duration_predictor(encoder_out, phoneme_mask)

        # Convert to integer durations via softplus + rounding
        dur_pred = F.softplus(log_dur_pred) * duration_scale
        dur_pred = torch.clamp(dur_pred, min=0.5)  # Ensure positive before round
        dur_int = torch.round(dur_pred).long()

        # Apply constraints
        dur_int = torch.clamp(dur_int, min=self.config.dur_min, max=self.config.dur_max)

        # Zero out padding positions
        dur_int = dur_int * phoneme_mask.long()

        # Expand
        expanded, mel_lens = self.length_regulator(encoder_out, dur_int)

        # Create mel mask
        max_len = mel_lens.max().item()
        mel_mask = torch.arange(max_len, device=phonemes.device).unsqueeze(0) < mel_lens.unsqueeze(1)

        # Decode
        mel_pred = self.decoder(expanded, mel_mask)

        return mel_pred, dur_int

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_duration_model(config: dict) -> BaselineDurationModel:
    """Create duration model from config dict."""
    from modules.training.dataloading import PHONEME_VOCAB_SIZE

    model_config = DurationModelConfig(
        vocab_size=PHONEME_VOCAB_SIZE,
        n_mels=config.get("mel", {}).get("n_mels", 80),
        d_model=config.get("model", {}).get("d_model", 256),
        n_conv_layers_enc=config.get("model", {}).get("n_conv_layers_enc", 6),
        n_conv_layers_dec=config.get("model", {}).get("n_conv_layers_dec", 4),
        kernel_size=config.get("model", {}).get("kernel_size", 5),
        dropout=config.get("model", {}).get("dropout", 0.1),
        dur_hidden=config.get("model", {}).get("dur_hidden", 256),
        dur_kernel=config.get("model", {}).get("dur_kernel", 3),
        dur_layers=config.get("model", {}).get("dur_layers", 2),
    )

    return BaselineDurationModel(model_config)
