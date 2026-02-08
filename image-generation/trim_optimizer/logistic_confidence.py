"""
Visualize the logistic boundary confidence function.

Shows how confidence varies with the energy transition (delta_dB) at a
detected boundary. Demonstrates the effect of varying the steepness (k)
and midpoint (b) parameters.

Output: images/trim_optimizer/logistic_confidence.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from commons.style import apply_style, get_output_dir


def logistic(delta_db: np.ndarray, k: float, b: float) -> np.ndarray:
    """c = 1 / (1 + exp(-k * (delta_dB - b)))"""
    return 1.0 / (1.0 + np.exp(-k * (delta_db - b)))


def main() -> None:
    apply_style()

    out_dir = get_output_dir("trim_optimizer")
    delta_db = np.linspace(-5, 25, 300)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # --- Left panel: Vary steepness k (fixed b=6.0) ---
    ax = axes[0]
    b_fixed = 6.0
    k_values = [0.25, 0.5, 1.0, 2.0]
    colors = ["#94a3b8", "#2563eb", "#7c3aed", "#dc2626"]

    for k, color in zip(k_values, colors):
        c = logistic(delta_db, k, b_fixed)
        lw = 2.5 if k == 0.5 else 1.5
        ls = "-" if k == 0.5 else "--"
        label = f"$k$={k}" + (" (default)" if k == 0.5 else "")
        ax.plot(delta_db, c, color=color, linewidth=lw, linestyle=ls, label=label)

    ax.axvline(x=b_fixed, color="#6b7280", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(y=0.5, color="#6b7280", linestyle=":", linewidth=1, alpha=0.7)
    ax.annotate(f"$b$={b_fixed} dB", xy=(b_fixed, 0.5), xytext=(b_fixed + 3, 0.35),
                fontsize=9, color="#6b7280",
                arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8})

    ax.set_xlabel(r"Energy transition $\Delta_{\mathrm{dB}}$ (dB)")
    ax.set_ylabel("Confidence $c$")
    ax.set_title(f"Varying steepness $k$ (fixed $b$={b_fixed})")
    ax.legend(loc="lower right")
    ax.set_xlim(-5, 25)
    ax.set_ylim(-0.05, 1.05)

    # --- Right panel: Vary midpoint b (fixed k=0.5) ---
    ax = axes[1]
    k_fixed = 0.5
    b_values = [3.0, 6.0, 10.0, 15.0]
    colors = ["#94a3b8", "#2563eb", "#7c3aed", "#dc2626"]

    for b, color in zip(b_values, colors):
        c = logistic(delta_db, k_fixed, b)
        lw = 2.5 if b == 6.0 else 1.5
        ls = "-" if b == 6.0 else "--"
        label = f"$b$={b} dB" + (" (default)" if b == 6.0 else "")
        ax.plot(delta_db, c, color=color, linewidth=lw, linestyle=ls, label=label)

    ax.axhline(y=0.5, color="#6b7280", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel(r"Energy transition $\Delta_{\mathrm{dB}}$ (dB)")
    ax.set_title(f"Varying midpoint $b$ (fixed $k$={k_fixed})")
    ax.legend(loc="lower right")
    ax.set_xlim(-5, 25)

    fig.suptitle("Logistic Boundary Confidence Score", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = out_dir / "logistic_confidence.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
