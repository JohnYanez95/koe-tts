"""
Run-to-run comparison and regression gates.

Compares eval metrics between two runs to detect improvements/regressions.
Supports threshold-based gates for CI/automation.

Usage:
    koe train compare --a run_a --b run_b
    koe train compare --a baseline --b new --gate --mel-l1-max-delta 3%
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modules.data_engineering.common.paths import paths


@dataclass
class CompareThresholds:
    """Regression gate thresholds (percentage)."""
    mel_l1_max_increase: float = 5.0      # Fail if L1 increases by more than 5%
    mel_l2_max_increase: float = 5.0      # Fail if L2 increases by more than 5%
    snr_min_decrease: float = 10.0        # Fail if SNR decreases by more than 10%
    silence_max_increase: float = 3.0     # Fail if silence% increases by more than 3%


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    name: str
    value_a: float
    value_b: float
    delta: float
    delta_pct: float
    higher_is_better: bool = False
    threshold_pct: Optional[float] = None
    threshold_exceeded: bool = False

    @property
    def improved(self) -> bool:
        """Did the metric improve from A to B?"""
        if self.higher_is_better:
            return self.delta > 0
        return self.delta < 0

    @property
    def status_icon(self) -> str:
        """Status icon for display."""
        if self.threshold_exceeded:
            return "✗"
        if self.improved:
            return "✓"
        return "~"


@dataclass
class CompareResult:
    """Full comparison result."""
    run_a: str
    run_b: str
    eval_tag_a: str
    eval_tag_b: str
    metrics: list[MetricComparison] = field(default_factory=list)
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_improved(self) -> int:
        return sum(1 for m in self.metrics if m.improved)

    @property
    def n_regressed(self) -> int:
        return sum(1 for m in self.metrics if not m.improved and abs(m.delta_pct) > 0.5)

    @property
    def n_threshold_exceeded(self) -> int:
        return sum(1 for m in self.metrics if m.threshold_exceeded)


def find_latest_eval(run_dir: Path) -> Optional[Path]:
    """Find the most recent eval directory in a run."""
    eval_base = run_dir / "eval"
    if not eval_base.exists():
        return None

    eval_dirs = sorted(
        [d for d in eval_base.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return eval_dirs[0] if eval_dirs else None


def find_eval_by_tag(run_dir: Path, eval_tag: str) -> Optional[Path]:
    """Find eval directory by tag name."""
    eval_dir = run_dir / "eval" / eval_tag
    if eval_dir.exists():
        return eval_dir
    return None


def load_metrics(eval_dir: Path) -> dict:
    """Load aggregate metrics from eval directory."""
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    with open(metrics_path) as f:
        return json.load(f)


def compare_runs(
    run_a: Path,
    run_b: Path,
    eval_tag: Optional[str] = None,
    thresholds: Optional[CompareThresholds] = None,
    apply_gates: bool = False,
) -> CompareResult:
    """
    Compare metrics between two runs.

    Args:
        run_a: Path to baseline run directory
        run_b: Path to new run directory
        eval_tag: Specific eval tag to compare (default: latest)
        thresholds: Regression thresholds
        apply_gates: Apply threshold checks

    Returns:
        CompareResult with all comparisons
    """
    if thresholds is None:
        thresholds = CompareThresholds()

    # Find eval directories
    if eval_tag:
        eval_a = find_eval_by_tag(run_a, eval_tag)
        eval_b = find_eval_by_tag(run_b, eval_tag)
        tag_a = eval_tag
        tag_b = eval_tag
    else:
        eval_a = find_latest_eval(run_a)
        eval_b = find_latest_eval(run_b)
        tag_a = eval_a.name if eval_a else "none"
        tag_b = eval_b.name if eval_b else "none"

    result = CompareResult(
        run_a=run_a.name,
        run_b=run_b.name,
        eval_tag_a=tag_a,
        eval_tag_b=tag_b,
    )

    if not eval_a:
        result.errors.append(f"No eval found for run A: {run_a.name}")
        result.passed = False
        return result

    if not eval_b:
        result.errors.append(f"No eval found for run B: {run_b.name}")
        result.passed = False
        return result

    # Load metrics
    try:
        metrics_a = load_metrics(eval_a)
        metrics_b = load_metrics(eval_b)
    except FileNotFoundError as e:
        result.errors.append(str(e))
        result.passed = False
        return result

    # Define metrics to compare with their properties
    # (key_prefix, display_name, higher_is_better, threshold_attr)
    metric_defs = [
        ("mel_l1", "mel_l1", False, "mel_l1_max_increase"),
        ("mel_l2", "mel_l2", False, "mel_l2_max_increase"),
        ("snr_proxy_db", "snr_proxy_db", True, "snr_min_decrease"),
        ("pred_silence_pct", "pred_silence_%", False, "silence_max_increase"),
        ("pred_rms", "pred_rms", False, None),
        ("pred_peak", "pred_peak", False, None),
    ]

    for key_prefix, display_name, higher_is_better, threshold_attr in metric_defs:
        mean_key = f"{key_prefix}_mean"

        if mean_key not in metrics_a or mean_key not in metrics_b:
            continue

        val_a = metrics_a[mean_key]
        val_b = metrics_b[mean_key]
        delta = val_b - val_a

        # Calculate percentage change
        if abs(val_a) > 1e-8:
            delta_pct = 100.0 * delta / abs(val_a)
        else:
            delta_pct = 0.0 if abs(delta) < 1e-8 else float('inf')

        # Get threshold
        threshold_pct = None
        threshold_exceeded = False

        if apply_gates and threshold_attr:
            threshold_pct = getattr(thresholds, threshold_attr, None)
            if threshold_pct is not None:
                # For higher_is_better metrics, regression is negative delta
                # For lower_is_better metrics, regression is positive delta
                if higher_is_better:
                    # SNR: regression if it decreases
                    threshold_exceeded = delta_pct < -threshold_pct
                else:
                    # L1/L2/silence: regression if it increases
                    threshold_exceeded = delta_pct > threshold_pct

        comparison = MetricComparison(
            name=display_name,
            value_a=val_a,
            value_b=val_b,
            delta=delta,
            delta_pct=delta_pct,
            higher_is_better=higher_is_better,
            threshold_pct=threshold_pct,
            threshold_exceeded=threshold_exceeded,
        )

        result.metrics.append(comparison)

        if threshold_exceeded:
            result.passed = False
            result.warnings.append(
                f"{display_name}: {delta_pct:+.1f}% exceeds threshold of "
                f"{'+' if not higher_is_better else '-'}{threshold_pct:.1f}%"
            )

    return result


def format_comparison_table(result: CompareResult) -> str:
    """Format comparison result as a table."""
    lines = []

    # Header
    lines.append("=" * 75)
    lines.append(f"RUN COMPARISON: {result.run_a} vs {result.run_b}")
    lines.append(f"Eval tags: {result.eval_tag_a} | {result.eval_tag_b}")
    lines.append("=" * 75)

    if result.errors:
        for err in result.errors:
            lines.append(f"ERROR: {err}")
        return "\n".join(lines)

    # Table header
    lines.append(f"{'Metric':<18} | {'A':>10} | {'B':>10} | {'Δ':>10} | {'Δ%':>8} | {'':>3}")
    lines.append("-" * 75)

    # Metrics
    for m in result.metrics:
        delta_str = f"{m.delta:+.4f}" if abs(m.delta) < 100 else f"{m.delta:+.2e}"
        pct_str = f"{m.delta_pct:+.1f}%"

        lines.append(
            f"{m.name:<18} | {m.value_a:>10.4f} | {m.value_b:>10.4f} | "
            f"{delta_str:>10} | {pct_str:>8} | {m.status_icon:>3}"
        )

    lines.append("-" * 75)

    # Summary
    lines.append(f"Improved: {result.n_improved}/{len(result.metrics)} | "
                 f"Regressed: {result.n_regressed}/{len(result.metrics)} | "
                 f"Threshold exceeded: {result.n_threshold_exceeded}")

    if result.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for warn in result.warnings:
            lines.append(f"  ⚠ {warn}")

    lines.append("")
    if result.passed:
        lines.append("RESULT: PASS")
    else:
        lines.append("RESULT: FAIL")

    lines.append("=" * 75)

    return "\n".join(lines)


def compare_and_print(
    run_a_id: str,
    run_b_id: str,
    eval_tag: Optional[str] = None,
    apply_gates: bool = False,
    thresholds: Optional[CompareThresholds] = None,
    runs_dir: Optional[Path] = None,
) -> CompareResult:
    """
    Compare two runs and print formatted results.

    Args:
        run_a_id: Run A identifier (baseline)
        run_b_id: Run B identifier (new)
        eval_tag: Specific eval tag
        apply_gates: Apply regression gates
        thresholds: Custom thresholds
        runs_dir: Base runs directory (defaults to paths.runs)

    Returns:
        CompareResult
    """
    if runs_dir is None:
        runs_dir = paths.runs
    # Resolve run directories
    run_a = runs_dir / run_a_id
    run_b = runs_dir / run_b_id

    if not run_a.exists():
        # Try fuzzy match
        candidates = list(runs_dir.glob(f"*{run_a_id}*"))
        if candidates:
            run_a = candidates[0]
        else:
            print(f"Run not found: {run_a_id}")
            return CompareResult(
                run_a=run_a_id, run_b=run_b_id,
                eval_tag_a="", eval_tag_b="",
                passed=False, errors=[f"Run not found: {run_a_id}"]
            )

    if not run_b.exists():
        candidates = list(runs_dir.glob(f"*{run_b_id}*"))
        if candidates:
            run_b = candidates[0]
        else:
            print(f"Run not found: {run_b_id}")
            return CompareResult(
                run_a=run_a_id, run_b=run_b_id,
                eval_tag_a="", eval_tag_b="",
                passed=False, errors=[f"Run not found: {run_b_id}"]
            )

    # Compare
    result = compare_runs(
        run_a=run_a,
        run_b=run_b,
        eval_tag=eval_tag,
        thresholds=thresholds,
        apply_gates=apply_gates,
    )

    # Print
    print(format_comparison_table(result))

    return result
