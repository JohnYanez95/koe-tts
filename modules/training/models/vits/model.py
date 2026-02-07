"""
VITS Model.

Wires together all VITS components:
- Text encoder (phoneme → prior distribution)
- Posterior encoder (mel → latent z)
- Flow (prior ↔ posterior transform)
- Duration predictor (text → durations)
- Generator (latent → waveform)

Training:
- Forward: mel + phonemes → alignment → losses
- Inference: phonemes → predicted durations → waveform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .text_encoder import TextEncoder
from .posterior_encoder import PosteriorEncoder
from .flows import ResidualCouplingBlocks
from .generator import Generator
from .aligner_mas import (
    monotonic_alignment_search,
    generate_path,
    compute_log_probs_for_mas,
)


@dataclass
class VITSConfig:
    """VITS model configuration."""

    # Vocab
    vocab_size: int = 45

    # Audio
    n_mels: int = 80
    sample_rate: int = 22050
    hop_length: int = 256

    # Latent
    latent_dim: int = 192

    # Multi-speaker
    num_speakers: int = 1  # 1 = single-speaker, >1 = multi-speaker
    speaker_emb_dim: int = 128  # Speaker embedding dimension

    # Text encoder
    text_d_model: int = 192
    text_n_layers: int = 6
    text_n_heads: int = 2
    text_d_ff: int = 768
    text_dropout: float = 0.1

    # Posterior encoder
    posterior_hidden: int = 192
    posterior_kernel: int = 5
    posterior_n_layers: int = 16
    posterior_dilation_rate: int = 2

    # Flow
    flow_hidden: int = 192
    flow_kernel: int = 5
    flow_n_layers: int = 4
    flow_n_flows: int = 4
    flow_dilation_rate: int = 2

    # Duration predictor
    dur_hidden: int = 256
    dur_kernel: int = 3
    dur_n_layers: int = 2
    dur_dropout: float = 0.5

    # Generator
    gen_upsample_rates: tuple = (8, 8, 2, 2)
    gen_upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    gen_upsample_initial_channel: int = 512
    gen_resblock_kernel_sizes: tuple = (3, 7, 11)
    gen_resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))


class DurationPredictor(nn.Module):
    """Predict duration (frames) per phoneme token."""

    def __init__(
        self,
        d_model: int,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_ch = d_model if i == 0 else hidden_channels
            self.convs.append(
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] text encoder hidden states
            mask: [B, T] valid positions

        Returns:
            log_dur: [B, T] predicted log durations
        """
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = x.transpose(1, 2)  # [B, T, C]
            x = norm(x)
            x = x.transpose(1, 2)  # [B, C, T]
            x = F.relu(x)
            x = self.dropout(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        log_dur = self.proj(x).squeeze(-1)  # [B, T]

        if mask is not None:
            log_dur = log_dur.masked_fill(~mask, 0.0)

        return log_dur


class VITSModel(nn.Module):
    """
    VITS: Variational Inference Text-to-Speech.

    Training forward pass:
    1. Encode text → prior (mean, log_var)
    2. Encode mel → posterior (mean, log_var) → sample z
    3. Run MAS to find optimal alignment
    4. Transform z through flow (for KL)
    5. Generate waveform from z

    Inference:
    1. Encode text → prior
    2. Predict durations
    3. Sample from prior, expand by durations
    4. Transform through flow (reverse)
    5. Generate waveform
    """

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        # Multi-speaker embedding
        self.num_speakers = config.num_speakers
        if config.num_speakers > 1:
            self.speaker_emb = nn.Embedding(config.num_speakers, config.speaker_emb_dim)
            # Project speaker embedding to text encoder dimension for additive bias
            self.speaker_proj = nn.Linear(config.speaker_emb_dim, config.text_d_model)
        else:
            self.speaker_emb = None
            self.speaker_proj = None

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            d_model=config.text_d_model,
            n_layers=config.text_n_layers,
            n_heads=config.text_n_heads,
            d_ff=config.text_d_ff,
            dropout=config.text_dropout,
            latent_dim=config.latent_dim,
        )

        # Posterior encoder
        self.posterior_encoder = PosteriorEncoder(
            n_mels=config.n_mels,
            hidden_channels=config.posterior_hidden,
            latent_dim=config.latent_dim,
            kernel_size=config.posterior_kernel,
            n_layers=config.posterior_n_layers,
            dilation_rate=config.posterior_dilation_rate,
        )

        # Flow
        self.flow = ResidualCouplingBlocks(
            channels=config.latent_dim,
            hidden_channels=config.flow_hidden,
            kernel_size=config.flow_kernel,
            n_layers=config.flow_n_layers,
            n_flows=config.flow_n_flows,
            dilation_rate=config.flow_dilation_rate,
        )

        # Duration predictor
        self.duration_predictor = DurationPredictor(
            d_model=config.text_d_model,
            hidden_channels=config.dur_hidden,
            kernel_size=config.dur_kernel,
            n_layers=config.dur_n_layers,
            dropout=config.dur_dropout,
        )

        # Generator
        self.generator = Generator(
            latent_dim=config.latent_dim,
            upsample_rates=config.gen_upsample_rates,
            upsample_kernel_sizes=config.gen_upsample_kernel_sizes,
            upsample_initial_channel=config.gen_upsample_initial_channel,
            resblock_kernel_sizes=config.gen_resblock_kernel_sizes,
            resblock_dilations=config.gen_resblock_dilations,
        )

    def forward(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: torch.Tensor,
        mel: torch.Tensor,
        mel_mask: torch.Tensor,
        speaker_idxs: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass.

        Args:
            phonemes: [B, T_text] phoneme token IDs
            phoneme_mask: [B, T_text] True for valid positions
            mel: [B, n_mels, T_mel] mel spectrogram
            mel_mask: [B, T_mel] True for valid frames
            speaker_idxs: [B] speaker indices (optional, for multi-speaker)

        Returns:
            Dict with all outputs needed for loss computation
        """
        # 1. Encode text → prior
        text_hidden, prior_mean, prior_log_var = self.text_encoder(phonemes, phoneme_mask)

        # 1b. Add speaker embedding as additive bias (multi-speaker)
        if self.speaker_emb is not None and speaker_idxs is not None:
            # speaker_emb: [B, speaker_emb_dim]
            spk_emb = self.speaker_emb(speaker_idxs)
            # Project to text encoder dimension: [B, text_d_model]
            spk_bias = self.speaker_proj(spk_emb)
            # Add as bias: [B, T, D] + [B, 1, D] → broadcast
            text_hidden = text_hidden + spk_bias.unsqueeze(1)

        # prior_mean/log_var: [B, T_text, latent_dim] -> [B, latent_dim, T_text]
        prior_mean = prior_mean.transpose(1, 2)
        prior_log_var = prior_log_var.transpose(1, 2)

        # 2. Encode mel → posterior, sample z
        posterior_mean, posterior_log_var = self.posterior_encoder(mel, mel_mask)
        # posterior_mean/log_var: [B, latent_dim, T_mel]

        # Sample z using reparameterization
        posterior_std = torch.exp(0.5 * posterior_log_var)
        eps = torch.randn_like(posterior_std)
        z = posterior_mean + eps * posterior_std
        if mel_mask is not None:
            z = z * mel_mask.unsqueeze(1).float()

        # 3. Run MAS to find alignment
        # Compute log p(z | prior) for alignment
        with torch.no_grad():
            log_probs = compute_log_probs_for_mas(prior_mean, prior_log_var, z)
            neg_log_probs = -log_probs
            alignment_path = monotonic_alignment_search(neg_log_probs, phoneme_mask, mel_mask)
            # alignment_path: [B, T_text, T_mel]

        # Extract durations from alignment (sum over mel for each text)
        durations = alignment_path.sum(dim=2)  # [B, T_text]

        # 4. Align prior to mel length using alignment path
        # prior_mean/log_var: [B, latent_dim, T_text]
        # We need to expand to [B, latent_dim, T_mel] using alignment
        prior_mean_aligned = torch.bmm(prior_mean, alignment_path)  # [B, C, T_mel]
        prior_log_var_aligned = torch.bmm(prior_log_var, alignment_path)

        # 5. Transform z through flow (for KL computation)
        z_flow, _ = self.flow(z, mel_mask, reverse=False)

        # 6. Predict durations (for duration loss)
        log_dur_pred = self.duration_predictor(text_hidden, phoneme_mask)

        # 7. Generate waveform from z
        audio = self.generator(z, mel_mask)

        return {
            "audio": audio,
            "z": z,
            "z_flow": z_flow,
            "posterior_mean": posterior_mean,
            "posterior_log_var": posterior_log_var,
            "prior_mean_aligned": prior_mean_aligned,
            "prior_log_var_aligned": prior_log_var_aligned,
            "log_dur_pred": log_dur_pred,
            "durations": durations,
            "alignment_path": alignment_path,
        }

    @torch.no_grad()
    def inference(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0,
        noise_scale: float = 0.667,
        speaker_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference: phonemes → waveform.

        Args:
            phonemes: [B, T_text] phoneme token IDs
            phoneme_mask: [B, T_text] True for valid positions
            duration_scale: Scale factor for durations
            noise_scale: Scale factor for sampling noise
            speaker_idx: Speaker index for multi-speaker (optional)

        Returns:
            audio: [B, 1, T_audio] generated waveform
        """
        self.eval()

        if phoneme_mask is None:
            phoneme_mask = torch.ones_like(phonemes, dtype=torch.bool)

        # 1. Encode text
        text_hidden, prior_mean, prior_log_var = self.text_encoder(phonemes, phoneme_mask)

        # 1b. Add speaker embedding (multi-speaker)
        if self.speaker_emb is not None and speaker_idx is not None:
            batch_size = phonemes.size(0)
            speaker_idxs = torch.full(
                (batch_size,), speaker_idx, dtype=torch.long, device=phonemes.device
            )
            spk_emb = self.speaker_emb(speaker_idxs)
            spk_bias = self.speaker_proj(spk_emb)
            text_hidden = text_hidden + spk_bias.unsqueeze(1)

        prior_mean = prior_mean.transpose(1, 2)  # [B, C, T_text]
        prior_log_var = prior_log_var.transpose(1, 2)

        # 2. Predict durations
        log_dur_pred = self.duration_predictor(text_hidden, phoneme_mask)
        dur_pred = torch.exp(log_dur_pred) * duration_scale
        dur_pred = torch.clamp(torch.round(dur_pred), min=1).long()
        dur_pred = dur_pred * phoneme_mask.long()

        # 3. Generate alignment path from durations
        T_mel = dur_pred.sum(dim=1).max().item()
        mel_mask = torch.arange(T_mel, device=phonemes.device).unsqueeze(0) < dur_pred.sum(dim=1).unsqueeze(1)
        alignment_path = generate_path(dur_pred, phoneme_mask, mel_mask)

        # 4. Expand prior to mel length
        prior_mean_expanded = torch.bmm(prior_mean, alignment_path)  # [B, C, T_mel]
        prior_log_var_expanded = torch.bmm(prior_log_var, alignment_path)

        # 5. Sample from prior
        prior_std = torch.exp(0.5 * prior_log_var_expanded)
        eps = torch.randn_like(prior_std) * noise_scale
        z = prior_mean_expanded + eps * prior_std

        # 6. Transform through flow (reverse direction)
        z, _ = self.flow(z, mel_mask, reverse=True)

        # 7. Generate waveform
        audio = self.generator(z, mel_mask)

        return audio

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vits_model(config: dict) -> VITSModel:
    """Create VITS model from config dict."""
    from modules.training.dataloading import PHONEME_VOCAB_SIZE

    model_cfg = config.get("model", {})
    mel_cfg = config.get("mel", {})

    vits_config = VITSConfig(
        vocab_size=PHONEME_VOCAB_SIZE,
        n_mels=mel_cfg.get("n_mels", 80),
        sample_rate=mel_cfg.get("sample_rate", 22050),
        hop_length=mel_cfg.get("hop_length", 256),

        latent_dim=model_cfg.get("latent_dim", 192),

        # Multi-speaker
        num_speakers=model_cfg.get("num_speakers", 1),
        speaker_emb_dim=model_cfg.get("speaker_emb_dim", 128),

        text_d_model=model_cfg.get("text_d_model", 192),
        text_n_layers=model_cfg.get("text_n_layers", 6),
        text_n_heads=model_cfg.get("text_n_heads", 2),
        text_d_ff=model_cfg.get("text_d_ff", 768),
        text_dropout=model_cfg.get("text_dropout", 0.1),

        posterior_hidden=model_cfg.get("posterior_hidden", 192),
        posterior_kernel=model_cfg.get("posterior_kernel", 5),
        posterior_n_layers=model_cfg.get("posterior_n_layers", 16),
        posterior_dilation_rate=model_cfg.get("posterior_dilation_rate", 2),

        flow_hidden=model_cfg.get("flow_hidden", 192),
        flow_kernel=model_cfg.get("flow_kernel", 5),
        flow_n_layers=model_cfg.get("flow_n_layers", 4),
        flow_n_flows=model_cfg.get("flow_n_flows", 4),
        flow_dilation_rate=model_cfg.get("flow_dilation_rate", 2),

        dur_hidden=model_cfg.get("dur_hidden", 256),
        dur_kernel=model_cfg.get("dur_kernel", 3),
        dur_n_layers=model_cfg.get("dur_n_layers", 2),
        dur_dropout=model_cfg.get("dur_dropout", 0.5),

        gen_upsample_rates=tuple(model_cfg.get("gen_upsample_rates", [8, 8, 2, 2])),
        gen_upsample_kernel_sizes=tuple(model_cfg.get("gen_upsample_kernel_sizes", [16, 16, 4, 4])),
        gen_upsample_initial_channel=model_cfg.get("gen_upsample_initial_channel", 512),
        gen_resblock_kernel_sizes=tuple(model_cfg.get("gen_resblock_kernel_sizes", [3, 7, 11])),
        gen_resblock_dilations=tuple(
            tuple(d) for d in model_cfg.get("gen_resblock_dilations", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        ),
    )

    return VITSModel(vits_config)
