"""Shared plotting utilities for documentation image generation."""

from pathlib import Path

import matplotlib.pyplot as plt

# Project root (two levels up from commons/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"


def apply_style() -> None:
    """Apply consistent plot style for documentation images."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def get_output_dir(subdir: str) -> Path:
    """Get output directory for a specific doc section, creating it if needed."""
    out = IMAGES_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out
