"""
Visualize asymmetric threshold derivation for trim detection.

Shows how onset and offset thresholds vary as a function of the noise floor
estimate, with different margin settings. Demonstrates why asymmetric margins
are necessary (sharp onsets vs gradual offsets).

Output: images/trim_optimizer/asymmetric_thresholds.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from commons.style import apply_style, get_output_dir


def compute_threshold(noise_floor_db: np.ndarray, margin_db: float, floor_db: float) -> np.ndarray:
    """theta = max(f, p_hat - m)"""
    return np.maximum(floor_db, noise_floor_db - margin_db)


def main() -> None:
    apply_style()

    out_dir = get_output_dir("trim_optimizer")

    # Noise floor range (typical recording environments)
    p_hat = np.linspace(-70, -20, 200)

    # Default parameters
    f_default = -60.0
    m_onset_default = 8.0
    m_offset_default = 10.0

    # Optimized parameters (from JSUT results)
    f_optimized = -36.3
    m_onset_optimized = 14.6
    m_offset_optimized = 13.1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # --- Left panel: Default parameters ---
    ax = axes[0]
    onset_default = compute_threshold(p_hat, m_onset_default, f_default)
    offset_default = compute_threshold(p_hat, m_offset_default, f_default)

    ax.plot(p_hat, onset_default, color="#2563eb", linewidth=2, label=r"$\theta_{\mathrm{onset}}$"
            f" ($m_{{onset}}$={m_onset_default})")
    ax.plot(p_hat, offset_default, color="#dc2626", linewidth=2, label=r"$\theta_{\mathrm{offset}}$"
            f" ($m_{{offset}}$={m_offset_default})")
    ax.axhline(y=f_default, color="#6b7280", linestyle="--", linewidth=1,
               label=f"floor $f$={f_default} dB")
    ax.fill_between(p_hat, onset_default, offset_default, alpha=0.1, color="#7c3aed",
                    label="asymmetric gap")

    ax.set_xlabel(r"Noise floor estimate $\hat{p}$ (dB)")
    ax.set_ylabel("Threshold (dB)")
    ax.set_title("Default Parameters")
    ax.legend(loc="upper left")
    ax.set_xlim(-70, -20)
    ax.set_ylim(-75, -10)

    # --- Right panel: Optimized parameters ---
    ax = axes[1]
    onset_opt = compute_threshold(p_hat, m_onset_optimized, f_optimized)
    offset_opt = compute_threshold(p_hat, m_offset_optimized, f_optimized)

    ax.plot(p_hat, onset_opt, color="#2563eb", linewidth=2, label=r"$\theta_{\mathrm{onset}}$"
            f" ($m_{{onset}}$={m_onset_optimized})")
    ax.plot(p_hat, offset_opt, color="#dc2626", linewidth=2, label=r"$\theta_{\mathrm{offset}}$"
            f" ($m_{{offset}}$={m_offset_optimized})")
    ax.axhline(y=f_optimized, color="#6b7280", linestyle="--", linewidth=1,
               label=f"floor $f$={f_optimized} dB")
    ax.fill_between(p_hat, onset_opt, offset_opt, alpha=0.1, color="#7c3aed",
                    label="asymmetric gap")

    ax.set_xlabel(r"Noise floor estimate $\hat{p}$ (dB)")
    ax.set_title("Optimized Parameters (JSUT)")
    ax.legend(loc="upper left")
    ax.set_xlim(-70, -20)

    fig.suptitle("Asymmetric Threshold Derivation: Default vs Optimized", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = out_dir / "asymmetric_thresholds.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
