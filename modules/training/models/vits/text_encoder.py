"""
VITS Text Encoder.

Converts phoneme IDs to hidden representations with mean/log-variance
for the prior distribution.

Architecture:
- Phoneme embedding
- Relative positional encoding
- Transformer encoder blocks
- Linear projection to (mean, log_var)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create relative position embeddings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1)]


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TextEncoder(nn.Module):
    """
    VITS text encoder.

    Encodes phoneme sequences to hidden representations and outputs
    mean/log-variance for the prior distribution.

    Args:
        vocab_size: Number of phoneme tokens
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        latent_dim: Dimension of latent space (for mean/log_var projection)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 2,
        d_ff: int = 768,
        dropout: float = 0.1,
        latent_dim: int = 192,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = RelativePositionalEncoding(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Project to latent space (mean and log_var)
        self.proj_mean = nn.Linear(d_model, latent_dim)
        self.proj_log_var = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        phonemes: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode phoneme sequence.

        Args:
            phonemes: [B, T_text] phoneme token IDs
            phoneme_mask: [B, T_text] True for valid positions

        Returns:
            hidden: [B, T_text, d_model] encoder hidden states
            prior_mean: [B, T_text, latent_dim] prior mean
            prior_log_var: [B, T_text, latent_dim] prior log variance
        """
        # Embed and add positional encoding
        x = self.embedding(phonemes) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)

        # Convert mask to key_padding_mask format (True = ignore)
        key_padding_mask = None
        if phoneme_mask is not None:
            key_padding_mask = ~phoneme_mask

        # Transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Project to prior distribution parameters
        prior_mean = self.proj_mean(x)
        prior_log_var = self.proj_log_var(x)

        return x, prior_mean, prior_log_var
