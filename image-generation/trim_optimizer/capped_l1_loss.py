"""
Visualize the capped L1 loss function used in trim optimization.

Shows how the loss saturates at 1.0 beyond the tolerance window tau,
preventing outliers from dominating the optimization. Also shows the
duration regularization term.

Output: images/trim_optimizer/capped_l1_loss.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from commons.style import apply_style, get_output_dir


def capped_l1(error_ms: np.ndarray, tau: float) -> np.ndarray:
    """L = min(|error| / tau, 1)"""
    return np.minimum(np.abs(error_ms) / tau, 1.0)


def duration_loss(
    gt_dur: float, pred_dur: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """L_dur = min(|d_hat - d| / (d + eps), 1)"""
    return np.minimum(np.abs(pred_dur - gt_dur) / (gt_dur + eps), 1.0)


def main() -> None:
    apply_style()

    out_dir = get_output_dir("trim_optimizer")
    error_ms = np.linspace(0, 1200, 400)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- Left panel: Capped L1 with different tau ---
    ax = axes[0]
    tau_values = [200, 500, 800]
    colors = ["#dc2626", "#2563eb", "#94a3b8"]
    labels = [
        r"$\tau$=200 ms (strict)",
        r"$\tau$=500 ms (default)",
        r"$\tau$=800 ms (lenient)",
    ]

    for tau, color, label in zip(tau_values, colors, labels):
        loss = capped_l1(error_ms, tau)
        lw = 2.5 if tau == 500 else 1.5
        ls = "-" if tau == 500 else "--"
        ax.plot(error_ms, loss, color=color, linewidth=lw, linestyle=ls, label=label)

    # Annotate saturation
    ax.annotate(
        "saturation at 1.0\n(outlier protection)",
        xy=(600, 1.0), xytext=(700, 0.75),
        fontsize=9, color="#6b7280",
        arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
    )

    ax.set_xlabel("Boundary error $|\\hat{s} - s|$ (ms)")
    ax.set_ylabel("Loss $\\mathcal{L}$")
    ax.set_title("Capped L1 Loss (positional)")
    ax.legend(loc="center right")
    ax.set_xlim(0, 1200)
    ax.set_ylim(-0.05, 1.15)

    # --- Right panel: Duration regularization ---
    ax = axes[1]
    gt_dur = 3000.0  # 3 second utterance
    pred_dur = np.linspace(0, 6000, 400)

    dur_loss = duration_loss(gt_dur, pred_dur)
    ax.plot(pred_dur, dur_loss, color="#7c3aed", linewidth=2.5)

    # Mark ground truth
    ax.axvline(x=gt_dur, color="#16a34a", linestyle="--", linewidth=1.5, label=f"GT duration $d$={gt_dur:.0f} ms")
    ax.axhline(y=1.0, color="#6b7280", linestyle=":", linewidth=1, alpha=0.5)

    # Annotate asymmetry
    ax.annotate(
        "asymmetric: shortening\npenalized more than\nlengthening (relative to $d$)",
        xy=(800, 0.73), xytext=(1500, 0.5),
        fontsize=9, color="#6b7280",
        arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
    )

    ax.set_xlabel("Predicted duration $\\hat{d}$ (ms)")
    ax.set_ylabel("Loss $\\mathcal{L}_{\\mathrm{dur}}$")
    ax.set_title(f"Duration Regularization ($d$={gt_dur:.0f} ms)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 6000)
    ax.set_ylim(-0.05, 1.15)

    fig.suptitle("Trim Optimization Loss Functions", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = out_dir / "capped_l1_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
