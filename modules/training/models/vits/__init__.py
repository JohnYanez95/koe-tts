"""
VITS: Variational Inference Text-to-Speech.

Staged implementation:
- Stage 4.0 (core): Text encoder, posterior/prior, flow, MAS alignment, generator
- Stage 4.1 (gan): Add MPD/MSD discriminators

Reference: Kim et al., "Conditional Variational Autoencoder with Adversarial
Learning for End-to-End Text-to-Speech" (2021)
"""

from .text_encoder import TextEncoder
from .posterior_encoder import PosteriorEncoder
from .flows import ResidualCouplingBlock, ResidualCouplingBlocks
from .aligner_mas import monotonic_alignment_search, generate_path
from .generator import Generator
from .losses import vits_loss_core
from .model import VITSModel, create_vits_model

__all__ = [
    # Encoders
    "TextEncoder",
    "PosteriorEncoder",
    # Flow
    "ResidualCouplingBlock",
    "ResidualCouplingBlocks",
    # Alignment
    "monotonic_alignment_search",
    "generate_path",
    # Generator
    "Generator",
    # Losses
    "vits_loss_core",
    # Model
    "VITSModel",
    "create_vits_model",
]
