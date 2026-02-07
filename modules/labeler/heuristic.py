"""
Lightweight heuristic runner for the labeling app.

Runs the RMS energy heuristic on utterances from the gold manifest,
writes results as JSONL for fast iteration without Spark/Delta.

Usage:
    from modules.labeler.heuristic import run_heuristic
    run_heuristic("jsut", min_pause_ms=50)
"""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from modules.data_engineering.common.audio import (
    PauseDetectionConfig,
    TrimDetectionConfig,
    detect_silence_regions,
    detect_trim_region,
    regions_to_breakpoints,
)
from modules.data_engineering.common.paths import paths
from modules.data_engineering.silver.segments import (
    config_to_hash,
    config_to_method_name,
    load_audio_for_segmentation,
    process_utterance,
)
from modules.labeler.app.data import load_manifest


def _heuristic_cache_dir() -> Path:
    return paths.runs / "labeling" / "heuristics"


def run_heuristic(
    dataset: str,
    limit: int = 0,
    min_pause_ms: int = 50,
    margin_db: float = 8.0,
    floor_db: float = -60.0,
    merge_gap_ms: int = 80,
    pad_ms: int = 30,
    threshold_db: float | None = None,
    name: str | None = None,
) -> dict:
    """
    Run RMS energy heuristic on a dataset and write JSONL cache.

    Args:
        dataset: Dataset name (e.g. "jsut")
        limit: Max utterances to process (0 = all)
        min_pause_ms: Minimum pause duration
        margin_db: Margin below p10 for adaptive threshold
        floor_db: Absolute floor threshold
        merge_gap_ms: Merge regions closer than this
        pad_ms: Pad regions by this amount
        threshold_db: If set, use manual threshold (disables adaptive)
        name: Human-readable name for this run (e.g. "baseline", "sensitive")

    Returns:
        Summary dict with stats
    """
    config = PauseDetectionConfig(
        min_pause_ms=min_pause_ms,
        margin_db=margin_db,
        floor_db=floor_db,
        merge_gap_ms=merge_gap_ms,
        pad_ms=pad_ms,
    )
    if threshold_db is not None:
        config.adaptive = False
        config.silence_threshold_db = threshold_db

    method = config_to_method_name(config)
    params_hash = config_to_hash(config)

    # Auto-generate name from key params if not provided
    if not name:
        name = f"{method}_n{limit or 'all'}"

    print(f"Heuristic: {method}")
    print(f"Name: {name}")
    print(f"Params hash: {params_hash}")
    print(f"Config: {json.dumps(asdict(config), indent=2)}")
    print()

    # Load manifest
    utterances = load_manifest(dataset)
    if not utterances:
        print(f"No manifest found for dataset: {dataset}")
        return {"status": "error", "message": "no manifest"}

    if limit > 0:
        utterances = utterances[:limit]

    print(f"Processing {len(utterances)} utterances from {dataset}...")

    # Output path
    cache_dir = _heuristic_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{dataset}_{params_hash}.jsonl"

    processed = 0
    with_breaks = 0
    failed = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for i, utt in enumerate(utterances):
            uid = utt["utterance_id"]
            audio_abspath = utt.get("audio_abspath", "")

            if not audio_abspath or not Path(audio_abspath).exists():
                # Try constructing from relpath
                audio_relpath = utt.get("audio_relpath", "")
                if audio_relpath:
                    audio_abspath = str(paths.data / audio_relpath)

            if not audio_abspath or not Path(audio_abspath).exists():
                failed += 1
                continue

            try:
                _regions, breakpoints, debug_info = process_utterance(
                    audio_abspath, config
                )
            except Exception as e:
                print(f"  [{i+1}] FAILED {uid}: {e}")
                failed += 1
                continue

            record = {
                "utterance_id": uid,
                "breakpoints_ms": breakpoints,
                "n_breakpoints": len(breakpoints),
                "duration_ms": debug_info.get("duration_ms", 0),
                "threshold_db_used": debug_info.get("threshold_db_used"),
                "rms_db_p10": debug_info.get("rms_db_p10"),
                "method": method,
                "params_hash": params_hash,
            }
            f.write(json.dumps(record) + "\n")

            processed += 1
            if breakpoints:
                with_breaks += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(utterances)}] {rate:.1f} utt/s, {with_breaks} with breaks")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Processed: {processed}")
    print(f"  With breaks: {with_breaks}")
    print(f"  Failed: {failed}")
    print(f"  Output: {out_path}")

    # Write a metadata sidecar
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "name": name,
                "method": method,
                "params_hash": params_hash,
                "config": asdict(config),
                "processed": processed,
                "with_breaks": with_breaks,
                "failed": failed,
                "elapsed_s": round(elapsed, 1),
                "output_path": str(out_path),
            },
            f,
            indent=2,
        )

    return {
        "status": "ok",
        "output_path": str(out_path),
        "name": name,
        "params_hash": params_hash,
        "method": method,
        "processed": processed,
        "with_breaks": with_breaks,
        "failed": failed,
    }


def list_heuristic_runs(dataset: str) -> list[dict]:
    """
    List available heuristic runs for a dataset.

    Returns list of dicts with: params_hash, method, processed, with_breaks,
    elapsed_s, created_at, file_path.
    """
    cache_dir = _heuristic_cache_dir()
    if not cache_dir.exists():
        return []

    runs = []
    for meta_path in sorted(
        cache_dir.glob(f"{dataset}_*.meta.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["created_at"] = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(meta_path.stat().st_mtime),
        )
        runs.append(meta)

    return runs


def load_heuristic_cache(
    dataset: str,
    params_hash: str | None = None,
) -> tuple[dict[str, list[int]], str, str]:
    """
    Load a heuristic cache for a dataset.

    Args:
        params_hash: Specific run to load. None = most recent.

    Returns (utterance_id -> breakpoints_ms, method, params_hash).
    """
    cache_dir = _heuristic_cache_dir()
    if not cache_dir.exists():
        return {}, "unknown", "unknown"

    if params_hash:
        cache_path = cache_dir / f"{dataset}_{params_hash}.jsonl"
        if not cache_path.exists():
            return {}, "unknown", "unknown"
    else:
        # Find most recent cache file for this dataset
        candidates = sorted(
            cache_dir.glob(f"{dataset}_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return {}, "unknown", "unknown"
        cache_path = candidates[0]

    result: dict[str, list[int]] = {}
    method = "unknown"
    out_hash = "unknown"

    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = record["utterance_id"]
            result[uid] = record.get("breakpoints_ms", [])
            if method == "unknown":
                method = record.get("method", "unknown")
                out_hash = record.get("params_hash", "unknown")

    return result, method, out_hash


def eval_labels(dataset: str) -> dict:
    """
    Evaluate saved labels against heuristic proposals.

    Reads all labeled sessions for the dataset, computes:
    - MAE of delta_ms
    - Within-tolerance accuracy (50ms, 100ms)
    - False positive rate (use_break=false %)
    - Edit stats (mean drag distance)
    """
    from modules.labeler.app.data import _sessions_root

    sessions_dir = _sessions_root()
    if not sessions_dir.exists():
        return {"status": "error", "message": "no sessions directory"}

    all_breaks: list[dict] = []
    utterances_labeled = 0
    utterances_skipped = 0

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue

        with open(session_file) as f:
            meta = json.load(f)

        if meta.get("dataset") != dataset:
            continue

        labels_file = session_dir / "labels.jsonl"
        if not labels_file.exists():
            continue

        with open(labels_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label = json.loads(line)
                status = label.get("status", "labeled")

                if status == "skipped":
                    utterances_skipped += 1
                    continue

                if status == "labeled" and "breaks" in label:
                    utterances_labeled += 1
                    for b in label["breaks"]:
                        all_breaks.append(b)

    if not all_breaks:
        print(f"No labeled breaks found for dataset: {dataset}")
        return {"status": "no_data", "utterances_labeled": 0}

    # Compute metrics
    deltas = []
    abs_deltas = []
    use_break_true = 0
    use_break_false = 0
    dragged = 0

    for b in all_breaks:
        ub = b.get("use_break", True)
        if ub:
            use_break_true += 1
        else:
            use_break_false += 1

        delta = b.get("delta_ms")
        if delta is not None:
            deltas.append(delta)
            abs_deltas.append(abs(delta))
            if delta != 0:
                dragged += 1

    total_breaks = len(all_breaks)
    fp_rate = use_break_false / total_breaks if total_breaks > 0 else 0

    metrics: dict = {
        "dataset": dataset,
        "utterances_labeled": utterances_labeled,
        "utterances_skipped": utterances_skipped,
        "total_pau_breaks": total_breaks,
        "use_break_true": use_break_true,
        "use_break_false": use_break_false,
        "rejection_rate": round(fp_rate * 100, 1),
    }

    if abs_deltas:
        mae = sum(abs_deltas) / len(abs_deltas)
        within_50 = sum(1 for d in abs_deltas if d <= 50) / len(abs_deltas)
        within_100 = sum(1 for d in abs_deltas if d <= 100) / len(abs_deltas)
        mean_drag = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0

        metrics.update({
            "n_with_delta": len(abs_deltas),
            "n_dragged": dragged,
            "mae_ms": round(mae, 1),
            "within_50ms": round(within_50 * 100, 1),
            "within_100ms": round(within_100 * 100, 1),
            "mean_drag_ms": round(mean_drag, 1),
            "max_delta_ms": max(abs_deltas),
            "mean_delta_ms": round(sum(deltas) / len(deltas), 1),
        })

    # Print report
    print(f"Label Eval: {dataset}")
    print(f"  Utterances labeled: {utterances_labeled}")
    print(f"  Utterances skipped: {utterances_skipped}")
    print(f"  Total pau breaks reviewed: {total_breaks}")
    print(f"  Accepted (use_break=true): {use_break_true}")
    print(f"  Rejected (use_break=false): {use_break_false} ({metrics['rejection_rate']}%)")

    if abs_deltas:
        print(f"  MAE: {metrics['mae_ms']}ms")
        print(f"  Within 50ms: {metrics['within_50ms']}%")
        print(f"  Within 100ms: {metrics['within_100ms']}%")
        print(f"  Dragged (delta != 0): {dragged}")
        print(f"  Mean drag distance: {metrics['mean_drag_ms']}ms")
        print(f"  Max delta: {metrics['max_delta_ms']}ms")
        print(f"  Mean delta (signed): {metrics['mean_delta_ms']}ms")

    return metrics


# ============================================================================
# Heuristic Optimization
# ============================================================================

# Stage-2 guardrail — independent of trim config hyperparameters.
# If a predicted trim window is smaller than this, Stage 2 treats
# it as invalid and falls back to full audio (without filtering GT).
MIN_TRIM_WINDOW_MS_FOR_PAUSE = 200


def _load_published_labels(dataset: str) -> list[dict]:
    """Load published labels, deduped by utterance_id (last wins)."""
    pub_path = paths.runs / "labeling" / "published" / dataset / "labels.jsonl"
    if not pub_path.exists():
        return []

    by_uid: dict[str, dict] = {}
    with open(pub_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = json.loads(line)
            if label.get("status") == "labeled":
                by_uid[label["utterance_id"]] = label

    return list(by_uid.values())


def _huber(x: float, delta: float = 1.0) -> float:
    """Huber loss: quadratic near 0, linear for outliers."""
    if abs(x) <= delta:
        return 0.5 * x * x
    return delta * (abs(x) - 0.5 * delta)


def _compute_loss_utterance(
    gt_accepted_ms: list[int],
    gt_rejected_ms: list[int],
    candidates_ms: list[int],
    tau_ms: float = 120.0,
    alpha: float = 1.0,
    beta: float = 3.0,
    gamma: float = 1.0,
    lambda_count: float = 0.3,
    delta_neg: float = 0.5,
    sigma_neg: float | None = None,
) -> tuple[float, dict]:
    """
    Per-utterance loss with Hungarian matching.

    L_u = α * (1/|M|+ε) Σ ρ(|g-c|/τ)           [positional]
        + β * |U+| / (|G+|+ε)                     [miss rate]
        + γ * |O| / (|G+|+ε)                       [orphan rate]
        + λ * ||C| - |G+|| / (|G+|+ε)             [count regularization]
        + δ * (1/|G-|+ε) Σ exp(-d²/2σ²)           [neg proximity]

    Returns (loss, metrics_dict) for enriched reporting.
    """
    import math

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    eps = 1e-6
    if sigma_neg is None:
        sigma_neg = tau_ms / 2.0

    n_gt = len(gt_accepted_ms)
    n_cand = len(candidates_ms)

    # --- Hungarian matching within τ ---
    matched_pairs: list[tuple[int, int]] = []
    matched_gt_idx: set[int] = set()
    matched_cand_idx: set[int] = set()

    if n_gt > 0 and n_cand > 0:
        cost = np.full((n_gt, n_cand), tau_ms + 1.0, dtype=np.float64)
        for i, g in enumerate(gt_accepted_ms):
            for k, c in enumerate(candidates_ms):
                d = abs(g - c)
                if d <= tau_ms:
                    cost[i, k] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, k in zip(row_ind, col_ind, strict=True):
            if cost[i, k] <= tau_ms:
                matched_pairs.append((gt_accepted_ms[i], candidates_ms[k]))
                matched_gt_idx.add(i)
                matched_cand_idx.add(k)

    n_matched = len(matched_pairs)
    n_missed = n_gt - n_matched
    n_orphan = n_cand - n_matched

    # --- Positional loss (Huber on |g-c|/τ) ---
    pos_loss = 0.0
    position_errors = []
    if n_matched > 0:
        for g, c in matched_pairs:
            err = abs(g - c)
            position_errors.append(err)
            pos_loss += _huber(err / tau_ms)
        pos_loss /= (n_matched + eps)

    # --- Miss rate ---
    miss_rate = n_missed / (n_gt + eps)

    # --- Orphan rate ---
    orphan_rate = n_orphan / (n_gt + eps)

    # --- Count regularization ---
    count_penalty = abs(n_cand - n_gt) / (n_gt + eps)

    # --- Negative proximity ---
    neg_loss = 0.0
    if gt_rejected_ms and candidates_ms:
        for r in gt_rejected_ms:
            min_dist = min(abs(r - c) for c in candidates_ms)
            neg_loss += math.exp(-(min_dist ** 2) / (2 * sigma_neg ** 2))
        neg_loss /= (len(gt_rejected_ms) + eps)

    total_loss = (
        alpha * pos_loss
        + beta * miss_rate
        + gamma * orphan_rate
        + lambda_count * count_penalty
        + delta_neg * neg_loss
    )

    # Enriched metrics
    mae = sum(position_errors) / len(position_errors) if position_errors else float("nan")
    recall = n_matched / (n_gt + eps)
    acc_50 = sum(1 for err in position_errors if err <= 50) / (n_gt + eps) if position_errors else 0.0
    acc_100 = sum(1 for err in position_errors if err <= 100) / (n_gt + eps) if position_errors else 0.0

    metrics = {
        "loss": round(total_loss, 4),
        "pos_loss": round(pos_loss, 4),
        "miss_rate": round(miss_rate, 4),
        "orphan_rate": round(orphan_rate, 4),
        "count_penalty": round(count_penalty, 4),
        "neg_loss": round(neg_loss, 4),
        "mae_ms": round(mae, 1) if position_errors else None,
        "recall": round(recall, 4),
        "acc_50": round(acc_50, 4),
        "acc_100": round(acc_100, 4),
        "total_err": sum(position_errors),
        "n_matched": n_matched,
        "n_missed": n_missed,
        "n_orphan": n_orphan,
        "n_gt": n_gt,
        "n_cand": n_cand,
    }

    return total_loss, metrics


def _evaluate_config(
    config: PauseDetectionConfig,
    labels: list[dict],
    preloaded_audio: dict[str, tuple[torch.Tensor, int, int]],
    tau_ms: float,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_count: float,
    delta_neg: float,
    trim_predictions: dict[str, tuple[int, int]] | None = None,
    trim_margin_ms: int = 200,
) -> tuple[float, dict]:
    """
    Evaluate a pause config against published labels.

    Labels store ms in absolute (full-audio) coordinates.

    Windowing modes:
    - trim_predictions=None (default): Legacy mode. Candidates are filtered
      to each utterance's label [trim_start, trim_end] range. GT is NOT
      filtered. Preserves existing behavior.
    - trim_predictions provided (Stage 2 cascaded mode): Both candidates
      AND GT are filtered to [ŝ-m, ê+m] where ŝ,ê are predicted trims
      and m = trim_margin_ms (slack band). Matches inference distribution.

    Invariant: if a predicted trim window is invalid (< MIN_TRIM_WINDOW_MS_FOR_PAUSE),
    Stage 2 falls back to full audio and does NOT filter GT.

    preloaded_audio: uid -> (waveform, sample_rate, duration_ms) — avoids
    re-reading WAVs on every DE evaluation.

    Returns (mean_loss, aggregated_metrics).
    """
    losses = []
    agg_matched = 0
    agg_missed = 0
    agg_orphan = 0
    agg_gt = 0
    agg_cand = 0
    agg_total_err = 0.0
    n_pred_windowed = 0
    n_pred_fallback = 0
    n_pred_missing = 0
    n_legacy = 0

    for label in labels:
        uid = label["utterance_id"]
        audio_data = preloaded_audio.get(uid)
        if audio_data is None:
            continue

        waveform, sr, duration_ms = audio_data

        breaks = label.get("breaks", [])
        gt_accepted = [b["ms"] for b in breaks if b.get("use_break", True) and b.get("ms") is not None]
        gt_rejected = [b["ms"] for b in breaks if not b.get("use_break", True) and b.get("ms") is not None]

        if not gt_accepted and not gt_rejected:
            continue

        try:
            regions, _debug = detect_silence_regions(waveform, sr, config)
            breakpoints = regions_to_breakpoints(regions, duration_ms)
        except Exception:
            continue

        # Determine evaluation window
        if trim_predictions is not None:
            # Stage 2 cascaded mode: use predicted trims + slack band
            pred_trim = trim_predictions.get(uid)
            if pred_trim:
                s, e = pred_trim
                s = max(0, min(s, duration_ms))
                e = max(0, min(e, duration_ms))

                if e - s < MIN_TRIM_WINDOW_MS_FOR_PAUSE:
                    # Trim invalid → full audio, do NOT filter GT
                    candidates = breakpoints
                    gt_acc = gt_accepted
                    gt_rej = gt_rejected
                    n_pred_fallback += 1
                else:
                    m = trim_margin_ms
                    window_start = max(0, s - m)
                    window_end = min(duration_ms, e + m)
                    candidates = [bp for bp in breakpoints if window_start <= bp <= window_end]
                    gt_acc = [g for g in gt_accepted if window_start <= g <= window_end]
                    gt_rej = [r for r in gt_rejected if window_start <= r <= window_end]
                    n_pred_windowed += 1

                    if not gt_acc and not gt_rej:
                        continue
            else:
                # No trim prediction for this utterance → full audio
                candidates = breakpoints
                gt_acc = gt_accepted
                gt_rej = gt_rejected
                n_pred_missing += 1
        else:
            # Legacy mode: filter candidates to label trims, GT unfiltered
            trim_start = label.get("trim_start_ms") or 0
            trim_end = label.get("trim_end_ms") or duration_ms
            trim_start = max(0, min(trim_start, duration_ms))
            trim_end = max(0, min(trim_end, duration_ms))
            if trim_end <= trim_start:
                candidates = breakpoints
            else:
                candidates = [bp for bp in breakpoints if trim_start <= bp <= trim_end]
            gt_acc = gt_accepted
            gt_rej = gt_rejected
            n_legacy += 1

        loss, metrics = _compute_loss_utterance(
            gt_acc, gt_rej, candidates,
            tau_ms=tau_ms, alpha=alpha, beta=beta,
            gamma=gamma, lambda_count=lambda_count,
            delta_neg=delta_neg,
        )
        losses.append(loss)
        agg_matched += metrics["n_matched"]
        agg_missed += metrics["n_missed"]
        agg_orphan += metrics["n_orphan"]
        agg_gt += metrics["n_gt"]
        agg_cand += metrics["n_cand"]
        agg_total_err += metrics["total_err"]

    if not losses:
        return float("inf"), {}

    mean_loss = sum(losses) / len(losses)
    eps = 1e-6
    recall = agg_matched / (agg_gt + eps)
    weighted_mae = agg_total_err / (agg_matched + eps) if agg_matched > 0 else float("nan")
    agg = {
        "mean_loss": round(mean_loss, 4),
        "n_utterances": len(losses),
        "total_matched": agg_matched,
        "total_missed": agg_missed,
        "total_orphan": agg_orphan,
        "total_gt": agg_gt,
        "total_cand": agg_cand,
        "recall": round(recall, 3),
        "mae_ms": round(weighted_mae, 1) if agg_matched > 0 else None,
        "n_pred_windowed": n_pred_windowed,
        "n_pred_fallback": n_pred_fallback,
        "n_pred_missing": n_pred_missing,
        "n_legacy": n_legacy,
    }
    return mean_loss, agg


def optimize_heuristic(
    dataset: str,
    tau_ms: float = 120.0,
    alpha: float = 1.0,
    beta: float = 3.0,
    gamma: float | None = None,
    lambda_count: float = 0.3,
    delta_neg: float = 0.5,
    n_folds: int = 3,
    max_iter: int = 50,
    seed: int = 42,
    name: str | None = None,
    trim_predictions: dict[str, tuple[int, int]] | None = None,
    trim_margin_ms: int = 200,
    trim_params_hash: str | None = None,
    trim_method: str | None = None,
) -> dict:
    """
    Optimize pause heuristic parameters against published labels.

    Uses differential evolution over 5 parameters with k-fold CV:
        min_pause_ms, margin_db, floor_db, merge_gap_ms, pad_ms

    Loss: Hungarian-matched positional Huber + miss rate + orphan rate
          + count regularization + negative proximity.

    gamma auto-schedules: 1.0 when no G- data (prevent spam), 0.25 otherwise.

    When trim_predictions is provided (Stage 2 cascaded mode), candidates and
    GT are filtered to [ŝ-m, ê+m] where m=trim_margin_ms. When None, legacy
    mode uses label-based trim filtering for backward compatibility.

    Saves winner as a new heuristic run, immediately selectable in the UI.
    """
    import random as rng_mod

    from scipy.optimize import differential_evolution

    trim_mode = "predicted" if trim_predictions is not None else "legacy_label_trim"

    if trim_predictions is not None and not trim_predictions:
        print("  WARN: trim_predictions provided but empty; "
              "evaluating on full audio for all utterances.", flush=True)

    labels = _load_published_labels(dataset)
    if not labels:
        raise ValueError(f"No published labels for dataset: {dataset}")

    # Count G+ and G-
    all_breaks = [b for lab in labels for b in lab.get("breaks", [])]
    n_accepted = sum(1 for b in all_breaks if b.get("use_break", True))
    n_rejected = sum(1 for b in all_breaks if not b.get("use_break", True))

    # Auto-schedule gamma: stronger orphan penalty when no negatives
    if gamma is None:
        gamma = 1.0 if n_rejected == 0 else 0.25

    print(f"Optimizing heuristic for {dataset}", flush=True)
    print(f"  Labels: {len(labels)} utterances, {n_accepted} G+, {n_rejected} G-", flush=True)
    print(f"  Loss: τ={tau_ms}ms α={alpha} β={beta} γ={gamma} λ={lambda_count} δ_neg={delta_neg}", flush=True)
    print(f"  CV: {n_folds}-fold, DE maxiter={max_iter}", flush=True)
    print(f"  Trim mode: {trim_mode}", flush=True)
    if trim_predictions is not None:
        print(f"  Trim predictions: {len(trim_predictions)} utterances, "
              f"margin={trim_margin_ms}ms, method={trim_method}, "
              f"hash={trim_params_hash}", flush=True)

    # Resolve audio paths and preload waveforms (avoids re-reading on every DE eval)
    manifest = load_manifest(dataset)
    manifest_by_id = {u["utterance_id"]: u for u in manifest}
    preloaded_audio: dict[str, tuple[torch.Tensor, int, int]] = {}
    for label in labels:
        uid = label["utterance_id"]
        utt = manifest_by_id.get(uid)
        if not utt:
            continue
        abspath = utt.get("audio_abspath", "")
        if not abspath or not Path(abspath).exists():
            relpath = utt.get("audio_relpath", "")
            if relpath:
                abspath = str(paths.data / relpath)
        if abspath and Path(abspath).exists():
            try:
                waveform, sr = load_audio_for_segmentation(abspath)
                duration_ms = int(len(waveform) * 1000 / sr)
                preloaded_audio[uid] = (waveform, sr, duration_ms)
            except Exception as e:
                print(f"  WARN: failed to load {uid}: {e}", flush=True)

    print(f"  Audio: {len(preloaded_audio)}/{len(labels)} loaded", flush=True)

    # Build k-fold splits
    fold_rng = rng_mod.Random(seed)
    indices = list(range(len(labels)))
    fold_rng.shuffle(indices)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, idx in enumerate(indices):
        folds[i % n_folds].append(idx)

    # Evaluate baseline
    baseline_config = PauseDetectionConfig()
    baseline_loss, baseline_agg = _evaluate_config(
        baseline_config, labels, preloaded_audio,
        tau_ms=tau_ms, alpha=alpha, beta=beta, gamma=gamma,
        lambda_count=lambda_count, delta_neg=delta_neg,
        trim_predictions=trim_predictions,
        trim_margin_ms=trim_margin_ms,
    )
    print(f"\n  Baseline: loss={baseline_loss:.4f} recall={baseline_agg.get('recall', '?')} "
          f"mae={baseline_agg.get('mae_ms', '?')}ms", flush=True)
    print(f"    min_pause={baseline_config.min_pause_ms} margin={baseline_config.margin_db} "
          f"floor={baseline_config.floor_db} merge={baseline_config.merge_gap_ms} "
          f"pad={baseline_config.pad_ms}", flush=True)

    # DE bounds: [min_pause_ms, margin_db, floor_db, merge_gap_ms, pad_ms]
    bounds = [
        (20, 250),
        (2.0, 20.0),
        (-80.0, -35.0),
        (0, 200),
        (0, 80),
    ]

    eval_count = [0]
    best_so_far = [float("inf")]

    def objective(params):
        min_pause_ms, margin_db, floor_db, merge_gap_ms, pad_ms = params
        config = PauseDetectionConfig(
            min_pause_ms=int(round(min_pause_ms)),
            margin_db=float(margin_db),
            floor_db=float(floor_db),
            merge_gap_ms=int(round(merge_gap_ms)),
            pad_ms=int(round(pad_ms)),
        )

        # K-fold CV: mean loss across partitions
        fold_losses = []
        for fold_indices in folds:
            fold_labels = [labels[i] for i in fold_indices]
            fold_loss, _ = _evaluate_config(
                config, fold_labels, preloaded_audio,
                tau_ms=tau_ms, alpha=alpha, beta=beta, gamma=gamma,
                lambda_count=lambda_count, delta_neg=delta_neg,
                trim_predictions=trim_predictions,
                trim_margin_ms=trim_margin_ms,
            )
            if fold_loss < float("inf"):
                fold_losses.append(fold_loss)

        if not fold_losses:
            return float("inf")

        mean_loss = sum(fold_losses) / len(fold_losses)
        eval_count[0] += 1

        if mean_loss < best_so_far[0]:
            best_so_far[0] = mean_loss
            print(f"  [{eval_count[0]}] best={mean_loss:.4f} "
                  f"min_pause={int(round(min_pause_ms))} margin={margin_db:.1f} "
                  f"floor={floor_db:.1f} merge={int(round(merge_gap_ms))} "
                  f"pad={int(round(pad_ms))}", flush=True)
        return mean_loss

    print("\nRunning differential evolution...", flush=True)
    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max_iter,
        seed=seed,
        tol=1e-4,
        atol=1e-4,
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({eval_count[0]} evals)")

    # Extract optimized config
    opt_min_pause, opt_margin, opt_floor, opt_merge, opt_pad = result.x
    opt_config = PauseDetectionConfig(
        min_pause_ms=int(round(opt_min_pause)),
        margin_db=round(float(opt_margin), 1),
        floor_db=round(float(opt_floor), 1),
        merge_gap_ms=int(round(opt_merge)),
        pad_ms=int(round(opt_pad)),
    )

    # Full evaluation on all labels
    opt_loss, opt_agg = _evaluate_config(
        opt_config, labels, preloaded_audio,
        tau_ms=tau_ms, alpha=alpha, beta=beta, gamma=gamma,
        lambda_count=lambda_count, delta_neg=delta_neg,
        trim_predictions=trim_predictions,
        trim_margin_ms=trim_margin_ms,
    )

    improvement = ((baseline_loss - opt_loss) / baseline_loss * 100) if baseline_loss > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Optimized: loss={opt_loss:.4f} recall={opt_agg.get('recall', '?')} "
          f"mae={opt_agg.get('mae_ms', '?')}ms")
    print(f"  Baseline:  loss={baseline_loss:.4f} recall={baseline_agg.get('recall', '?')} "
          f"mae={baseline_agg.get('mae_ms', '?')}ms")
    print(f"  Improvement: {improvement:.1f}%")
    print()
    print(f"  min_pause_ms : {baseline_config.min_pause_ms:>6} → {opt_config.min_pause_ms}")
    print(f"  margin_db    : {baseline_config.margin_db:>6} → {opt_config.margin_db}")
    print(f"  floor_db     : {baseline_config.floor_db:>6} → {opt_config.floor_db}")
    print(f"  merge_gap_ms : {baseline_config.merge_gap_ms:>6} → {opt_config.merge_gap_ms}")
    print(f"  pad_ms       : {baseline_config.pad_ms:>6} → {opt_config.pad_ms}")
    print(f"{'='*60}")

    # Save as new heuristic run
    if not name:
        name = f"optimized_n{len(labels)}"

    print(f"\nSaving heuristic run: {name}")
    run_result = run_heuristic(
        dataset,
        min_pause_ms=opt_config.min_pause_ms,
        margin_db=opt_config.margin_db,
        floor_db=opt_config.floor_db,
        merge_gap_ms=opt_config.merge_gap_ms,
        pad_ms=opt_config.pad_ms,
        name=name,
    )

    # Augment pause .meta.json sidecar with trim provenance
    trim_provenance = {
        "mode": trim_mode,
        "margin_ms": trim_margin_ms,
        "params_hash": trim_params_hash,
        "method": trim_method,
        "n_predictions": len(trim_predictions) if trim_predictions else 0,
    }
    out_path_str = run_result.get("output_path", "")
    if out_path_str:
        meta_path = Path(out_path_str).with_suffix(".meta.json")
        try:
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["trim"] = trim_provenance
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        except Exception as _e:
            print(f"  WARN: failed to augment {meta_path} with trim provenance: {_e}",
                  flush=True)

    return {
        "status": "ok",
        "baseline_loss": round(baseline_loss, 4),
        "optimized_loss": round(opt_loss, 4),
        "improvement_pct": round(improvement, 1),
        "baseline_metrics": baseline_agg,
        "optimized_metrics": opt_agg,
        "optimized_config": asdict(opt_config),
        "evaluations": eval_count[0],
        "elapsed_s": round(elapsed, 1),
        "n_folds": n_folds,
        "heuristic_run": run_result,
        "trim": trim_provenance,
    }


# ============================================================================
# Trim Detection Optimization (Stage 1)
# ============================================================================


def _trim_config_hash(config: TrimDetectionConfig) -> str:
    """Generate deterministic hash of trim config."""
    params = asdict(config)
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(params_json.encode()).hexdigest()[:12]


def _compute_trim_loss_utterance(
    gt_start: int,
    gt_end: int,
    pred_start: int,
    pred_end: int,
    tau_ms: float = 500.0,
    alpha_start: float = 1.0,
    alpha_end: float = 1.0,
    lambda_dur: float = 0.5,
) -> tuple[float, dict]:
    """
    Per-utterance capped trim loss.

    L = α_start * min(|gt_start - pred_start| / τ, 1)
      + α_end   * min(|gt_end - pred_end| / τ, 1)
      + λ_dur   * min(|gt_dur - pred_dur| / (gt_dur + ε), 1)

    Capping makes loss robust to outliers and well-behaved for DE.

    Returns (loss, metrics_dict).
    """
    eps = 1e-6

    err_start = abs(gt_start - pred_start)
    err_end = abs(gt_end - pred_end)

    gt_dur = gt_end - gt_start
    pred_dur = pred_end - pred_start
    err_dur = abs(gt_dur - pred_dur)

    l_start = min(err_start / tau_ms, 1.0)
    l_end = min(err_end / tau_ms, 1.0)
    l_dur = min(err_dur / (gt_dur + eps), 1.0)

    loss = alpha_start * l_start + alpha_end * l_end + lambda_dur * l_dur

    metrics = {
        "err_start_ms": err_start,
        "err_end_ms": err_end,
        "err_dur_ms": err_dur,
        "l_start": round(l_start, 4),
        "l_end": round(l_end, 4),
        "l_dur": round(l_dur, 4),
        "loss": round(loss, 4),
    }

    return loss, metrics


def _evaluate_trim_config(
    config: TrimDetectionConfig,
    labels: list[dict],
    preloaded_audio: dict[str, tuple[torch.Tensor, int, int]],
    tau_ms: float = 500.0,
    alpha_start: float = 1.0,
    alpha_end: float = 1.0,
    lambda_dur: float = 0.5,
    lambda_fallback: float = 0.1,
) -> tuple[float, dict]:
    """
    Evaluate a trim config against published labels with trim GT.

    Filters to labels where both trim_start_ms and trim_end_ms are set.
    Skips labels with inverted GT (gt_end <= gt_start).
    Includes fallback penalty to discourage configs that trigger frequent
    fallback-to-full-audio.

    Returns (total_loss, aggregated_metrics).
    """
    losses = []
    n_evaluated = 0
    n_fallback = 0
    n_invalid_gt = 0
    n_failed_eval = 0
    agg_err_start = 0.0
    agg_err_end = 0.0

    # Confidence-stratified buckets (reported, not used in loss)
    buckets: dict[str, list[float]] = {"low_conf": [], "high_conf": []}
    bucket_errs: dict[str, dict[str, list[float]]] = {
        "low_conf": {"start": [], "end": []},
        "high_conf": {"start": [], "end": []},
    }

    for label in labels:
        uid = label["utterance_id"]
        gt_start = label.get("trim_start_ms")
        gt_end = label.get("trim_end_ms")

        # Only evaluate utterances with trim GT
        if gt_start is None or gt_end is None:
            continue

        # Guard against bad / inverted GT
        if gt_end <= gt_start:
            n_invalid_gt += 1
            continue

        audio_data = preloaded_audio.get(uid)
        if audio_data is None:
            continue

        waveform, sr, _duration_ms = audio_data

        try:
            pred_start, pred_end, debug_info = detect_trim_region(
                waveform, sr, config
            )
        except Exception as _e:
            n_failed_eval += 1
            continue

        n_evaluated += 1

        if debug_info.get("fallback_to_full_audio", False):
            n_fallback += 1

        loss, metrics = _compute_trim_loss_utterance(
            gt_start, gt_end, pred_start, pred_end,
            tau_ms=tau_ms, alpha_start=alpha_start,
            alpha_end=alpha_end, lambda_dur=lambda_dur,
        )
        losses.append(loss)
        agg_err_start += metrics["err_start_ms"]
        agg_err_end += metrics["err_end_ms"]

        # Stratify by confidence
        conf_start = debug_info.get("confidence_start", 0.0)
        conf_end = debug_info.get("confidence_end", 0.0)
        bucket = "low_conf" if min(conf_start, conf_end) < 0.3 else "high_conf"
        buckets[bucket].append(loss)
        bucket_errs[bucket]["start"].append(metrics["err_start_ms"])
        bucket_errs[bucket]["end"].append(metrics["err_end_ms"])

    if not losses:
        return float("inf"), {}

    mean_loss = sum(losses) / len(losses)
    fallback_rate = n_fallback / n_evaluated if n_evaluated > 0 else 0.0
    n_attempted = n_evaluated + n_failed_eval
    failed_eval_rate = n_failed_eval / n_attempted if n_attempted > 0 else 0.0

    # Total loss includes fallback penalty
    total_loss = mean_loss + lambda_fallback * fallback_rate

    mae_start = agg_err_start / len(losses)
    mae_end = agg_err_end / len(losses)

    agg = {
        "total_loss": round(total_loss, 4),
        "mean_loss": round(mean_loss, 4),
        "fallback_rate": round(fallback_rate, 4),
        "fallback_penalty": round(lambda_fallback * fallback_rate, 4),
        "n_evaluated": n_evaluated,
        "n_fallback": n_fallback,
        "n_invalid_gt": n_invalid_gt,
        "n_failed_eval": n_failed_eval,
        "failed_eval_rate": round(failed_eval_rate, 4),
        "mae_start_ms": round(mae_start, 1),
        "mae_end_ms": round(mae_end, 1),
    }

    # Confidence-stratified metrics (diagnostic only)
    for bucket_name in ("low_conf", "high_conf"):
        b_losses = buckets[bucket_name]
        b_starts = bucket_errs[bucket_name]["start"]
        b_ends = bucket_errs[bucket_name]["end"]
        n = len(b_losses)
        agg[f"{bucket_name}_n"] = n
        if n > 0:
            agg[f"{bucket_name}_mae_start"] = round(sum(b_starts) / n, 1)
            agg[f"{bucket_name}_mae_end"] = round(sum(b_ends) / n, 1)
            agg[f"{bucket_name}_mean_loss"] = round(sum(b_losses) / n, 4)

    return total_loss, agg


def optimize_trim(
    dataset: str,
    tau_ms: float = 500.0,
    alpha_start: float = 1.0,
    alpha_end: float = 1.0,
    lambda_dur: float = 0.5,
    lambda_fallback: float = 0.1,
    n_folds: int = 3,
    max_iter: int = 50,
    seed: int = 42,
    name: str | None = None,
) -> dict:
    """
    Optimize trim detection parameters against published labels (Stage 1).

    Uses differential evolution over 7 parameters with k-fold partitioned
    evaluation (no train/val distinction; objective is mean loss over
    partitions for robustness):
        onset_margin_db, offset_margin_db, floor_db, percentile,
        min_content_ms, pad_start_ms, pad_end_ms

    Loss: capped positional error on start/end + duration penalty
          + fallback rate penalty.

    Saves winner as a trim heuristic run.
    """
    import random as rng_mod

    from scipy.optimize import differential_evolution

    labels = _load_published_labels(dataset)
    if not labels:
        raise ValueError(f"No published labels for dataset: {dataset}")

    # Filter to labels with trim GT
    trim_labels = [
        lab for lab in labels
        if lab.get("trim_start_ms") is not None
        and lab.get("trim_end_ms") is not None
    ]
    if not trim_labels:
        raise ValueError(f"No labels with trim data for dataset: {dataset}")

    print(f"Optimizing trim detection for {dataset}", flush=True)
    print(f"  Labels: {len(trim_labels)}/{len(labels)} with trim GT", flush=True)
    print(f"  Loss: τ={tau_ms}ms α_s={alpha_start} α_e={alpha_end} "
          f"λ_dur={lambda_dur} λ_fb={lambda_fallback}", flush=True)
    print(f"  CV: {n_folds}-fold partitioned, DE maxiter={max_iter}", flush=True)

    # Preload audio
    manifest = load_manifest(dataset)
    manifest_by_id = {u["utterance_id"]: u for u in manifest}
    preloaded_audio: dict[str, tuple[torch.Tensor, int, int]] = {}
    for label in trim_labels:
        uid = label["utterance_id"]
        utt = manifest_by_id.get(uid)
        if not utt:
            continue
        abspath = utt.get("audio_abspath", "")
        if not abspath or not Path(abspath).exists():
            relpath = utt.get("audio_relpath", "")
            if relpath:
                abspath = str(paths.data / relpath)
        if abspath and Path(abspath).exists():
            try:
                waveform, sr = load_audio_for_segmentation(abspath)
                duration_ms = int(len(waveform) * 1000 / sr)
                preloaded_audio[uid] = (waveform, sr, duration_ms)
            except Exception as e:
                print(f"  WARN: failed to load {uid}: {e}", flush=True)

    print(f"  Audio: {len(preloaded_audio)}/{len(trim_labels)} loaded", flush=True)

    # Build k-fold partitions (partitioned evaluation, not train/val)
    fold_rng = rng_mod.Random(seed)
    indices = list(range(len(trim_labels)))
    fold_rng.shuffle(indices)
    partitions: list[list[int]] = [[] for _ in range(n_folds)]
    for i, idx in enumerate(indices):
        partitions[i % n_folds].append(idx)

    # Evaluate baseline
    baseline_config = TrimDetectionConfig()
    baseline_loss, baseline_agg = _evaluate_trim_config(
        baseline_config, trim_labels, preloaded_audio,
        tau_ms, alpha_start, alpha_end, lambda_dur, lambda_fallback,
    )
    print(f"\n  Baseline: loss={baseline_loss:.4f} "
          f"mae_start={baseline_agg.get('mae_start_ms', '?')}ms "
          f"mae_end={baseline_agg.get('mae_end_ms', '?')}ms "
          f"fallback={baseline_agg.get('fallback_rate', '?')}", flush=True)

    # DE bounds: 7 parameters
    bounds = [
        (2.0, 20.0),    # onset_margin_db
        (2.0, 25.0),    # offset_margin_db
        (-80.0, -35.0), # floor_db
        (5.0, 30.0),    # percentile
        (150, 1200),     # min_content_ms
        (0, 150),        # pad_start_ms
        (0, 250),        # pad_end_ms
    ]

    eval_count = [0]
    best_so_far = [float("inf")]

    def objective(params):
        (onset_margin_db, offset_margin_db, floor_db, percentile,
         min_content_ms, pad_start_ms, pad_end_ms) = params

        config = TrimDetectionConfig(
            onset_margin_db=float(onset_margin_db),
            offset_margin_db=float(offset_margin_db),
            floor_db=float(floor_db),
            percentile=float(percentile),
            min_content_ms=int(round(min_content_ms)),
            pad_start_ms=int(round(pad_start_ms)),
            pad_end_ms=int(round(pad_end_ms)),
        )

        # k-fold partitioned evaluation: mean loss across partitions
        partition_losses = []
        for part_indices in partitions:
            part_labels = [trim_labels[i] for i in part_indices]
            part_loss, _ = _evaluate_trim_config(
                config, part_labels, preloaded_audio,
                tau_ms, alpha_start, alpha_end, lambda_dur, lambda_fallback,
            )
            if part_loss < float("inf"):
                partition_losses.append(part_loss)

        if not partition_losses:
            return float("inf")

        mean_loss = sum(partition_losses) / len(partition_losses)
        eval_count[0] += 1

        if mean_loss < best_so_far[0]:
            best_so_far[0] = mean_loss
            print(f"  [{eval_count[0]}] best={mean_loss:.4f} "
                  f"onset_m={onset_margin_db:.1f} offset_m={offset_margin_db:.1f} "
                  f"floor={floor_db:.1f} pct={percentile:.0f} "
                  f"min_c={int(round(min_content_ms))} "
                  f"pad_s={int(round(pad_start_ms))} "
                  f"pad_e={int(round(pad_end_ms))}", flush=True)
        return mean_loss

    print("\nRunning differential evolution (trim)...", flush=True)
    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max_iter,
        seed=seed,
        tol=1e-4,
        atol=1e-4,
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({eval_count[0]} evals)")

    # Extract optimized config
    (opt_onset_m, opt_offset_m, opt_floor, opt_pct,
     opt_min_content, opt_pad_s, opt_pad_e) = result.x

    opt_config = TrimDetectionConfig(
        onset_margin_db=round(float(opt_onset_m), 1),
        offset_margin_db=round(float(opt_offset_m), 1),
        floor_db=round(float(opt_floor), 1),
        percentile=round(float(opt_pct), 1),
        min_content_ms=int(round(opt_min_content)),
        pad_start_ms=int(round(opt_pad_s)),
        pad_end_ms=int(round(opt_pad_e)),
    )

    # Full evaluation on all trim labels
    opt_loss, opt_agg = _evaluate_trim_config(
        opt_config, trim_labels, preloaded_audio,
        tau_ms, alpha_start, alpha_end, lambda_dur, lambda_fallback,
    )

    improvement = ((baseline_loss - opt_loss) / baseline_loss * 100) if baseline_loss > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Optimized: loss={opt_loss:.4f} "
          f"mae_start={opt_agg.get('mae_start_ms', '?')}ms "
          f"mae_end={opt_agg.get('mae_end_ms', '?')}ms "
          f"fallback={opt_agg.get('fallback_rate', '?')}")
    print(f"  Baseline:  loss={baseline_loss:.4f} "
          f"mae_start={baseline_agg.get('mae_start_ms', '?')}ms "
          f"mae_end={baseline_agg.get('mae_end_ms', '?')}ms "
          f"fallback={baseline_agg.get('fallback_rate', '?')}")
    print(f"  Improvement: {improvement:.1f}%")
    print()
    print(f"  onset_margin_db : {baseline_config.onset_margin_db:>6} → {opt_config.onset_margin_db}")
    print(f"  offset_margin_db: {baseline_config.offset_margin_db:>6} → {opt_config.offset_margin_db}")
    print(f"  floor_db        : {baseline_config.floor_db:>6} → {opt_config.floor_db}")
    print(f"  percentile      : {baseline_config.percentile:>6} → {opt_config.percentile}")
    print(f"  min_content_ms  : {baseline_config.min_content_ms:>6} → {opt_config.min_content_ms}")
    print(f"  pad_start_ms    : {baseline_config.pad_start_ms:>6} → {opt_config.pad_start_ms}")
    print(f"  pad_end_ms      : {baseline_config.pad_end_ms:>6} → {opt_config.pad_end_ms}")

    # Confidence-stratified report
    for bucket in ("low_conf", "high_conf"):
        n = opt_agg.get(f"{bucket}_n", 0)
        if n > 0:
            print(f"  {bucket}: n={n} "
                  f"mae_start={opt_agg.get(f'{bucket}_mae_start', '?')}ms "
                  f"mae_end={opt_agg.get(f'{bucket}_mae_end', '?')}ms")
    print(f"{'='*60}")

    # Save as trim heuristic run
    if not name:
        name = f"trim_optimized_n{len(trim_labels)}"

    print(f"\nSaving trim heuristic run: {name}")
    run_result = run_trim_heuristic(
        dataset,
        config=opt_config,
        name=name,
    )

    return {
        "status": "ok",
        "baseline_loss": round(baseline_loss, 4),
        "optimized_loss": round(opt_loss, 4),
        "improvement_pct": round(improvement, 1),
        "baseline_metrics": baseline_agg,
        "optimized_metrics": opt_agg,
        "optimized_config": asdict(opt_config),
        "evaluations": eval_count[0],
        "elapsed_s": round(elapsed, 1),
        "n_folds": n_folds,
        "trim_run": run_result,
    }


def run_trim_heuristic(
    dataset: str,
    config: TrimDetectionConfig | None = None,
    limit: int = 0,
    name: str | None = None,
) -> dict:
    """
    Run trim detection on a dataset and write JSONL cache.

    Parallel to run_heuristic() but outputs trim predictions.
    Applies clamp + validity check before writing (no garbage trims).

    Output: runs/labeling/heuristics/{dataset}_trim_{params_hash}.jsonl
    """
    if config is None:
        config = TrimDetectionConfig()

    params_hash = _trim_config_hash(config)
    method = "trim_v1_adaptive"

    if not name:
        name = f"{method}_n{limit or 'all'}"

    print(f"Trim heuristic: {method}")
    print(f"Name: {name}")
    print(f"Params hash: {params_hash}")
    print(f"Config: {json.dumps(asdict(config), indent=2)}")
    print()

    utterances = load_manifest(dataset)
    if not utterances:
        print(f"No manifest found for dataset: {dataset}")
        return {"status": "error", "message": "no manifest"}

    if limit > 0:
        utterances = utterances[:limit]

    print(f"Processing {len(utterances)} utterances from {dataset}...")

    cache_dir = _heuristic_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{dataset}_trim_{params_hash}.jsonl"

    processed = 0
    n_fallback = 0
    failed = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for i, utt in enumerate(utterances):
            uid = utt["utterance_id"]
            audio_abspath = utt.get("audio_abspath", "")

            if not audio_abspath or not Path(audio_abspath).exists():
                audio_relpath = utt.get("audio_relpath", "")
                if audio_relpath:
                    audio_abspath = str(paths.data / audio_relpath)

            if not audio_abspath or not Path(audio_abspath).exists():
                failed += 1
                continue

            try:
                waveform, sr = load_audio_for_segmentation(audio_abspath)
                trim_start, trim_end, debug_info = detect_trim_region(
                    waveform, sr, config
                )
            except Exception as e:
                print(f"  [{i+1}] FAILED {uid}: {e}")
                failed += 1
                continue

            fallback = debug_info.get("fallback_to_full_audio", False)
            if fallback:
                n_fallback += 1

            record = {
                "utterance_id": uid,
                "trim_start_ms": trim_start,
                "trim_end_ms": trim_end,
                "confidence_start": debug_info.get("confidence_start", 0.0),
                "confidence_end": debug_info.get("confidence_end", 0.0),
                "fallback_to_full_audio": fallback,
                "fallback_reason": debug_info.get("fallback_reason"),
                "onset_threshold_db": debug_info.get("onset_threshold_db"),
                "offset_threshold_db": debug_info.get("offset_threshold_db"),
                "rms_db_at_percentile": debug_info.get("rms_db_at_percentile"),
                "percentile_effective": debug_info.get("percentile_effective"),
                "duration_ms": debug_info.get("duration_ms", 0),
                "method": method,
                "params_hash": params_hash,
            }
            f.write(json.dumps(record) + "\n")

            processed += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(utterances)}] {rate:.1f} utt/s, "
                      f"{n_fallback} fallbacks")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Processed: {processed}")
    print(f"  Fallbacks: {n_fallback}")
    print(f"  Failed: {failed}")
    print(f"  Output: {out_path}")

    # Write metadata sidecar
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "name": name,
                "method": method,
                "params_hash": params_hash,
                "config": asdict(config),
                "processed": processed,
                "n_fallback": n_fallback,
                "failed": failed,
                "elapsed_s": round(elapsed, 1),
                "output_path": str(out_path),
            },
            f,
            indent=2,
        )

    return {
        "status": "ok",
        "output_path": str(out_path),
        "name": name,
        "params_hash": params_hash,
        "method": method,
        "processed": processed,
        "n_fallback": n_fallback,
        "failed": failed,
    }


def load_trim_cache(
    dataset: str,
    params_hash: str | None = None,
) -> tuple[dict[str, tuple[int, int]], str, str]:
    """
    Load trim predictions from cache.

    Args:
        dataset: Dataset name
        params_hash: Specific trim run to load. None = most recent.

    Returns:
        (uid -> (trim_start_ms, trim_end_ms), method, params_hash)
    """
    cache_dir = _heuristic_cache_dir()
    if not cache_dir.exists():
        return {}, "unknown", "unknown"

    if params_hash:
        cache_path = cache_dir / f"{dataset}_trim_{params_hash}.jsonl"
        if not cache_path.exists():
            return {}, "unknown", "unknown"
    else:
        candidates = sorted(
            cache_dir.glob(f"{dataset}_trim_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return {}, "unknown", "unknown"
        cache_path = candidates[0]

    result: dict[str, tuple[int, int]] = {}
    method = "unknown"
    out_hash = "unknown"

    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = record["utterance_id"]
            result[uid] = (record["trim_start_ms"], record["trim_end_ms"])
            if method == "unknown":
                method = record.get("method", "unknown")
                out_hash = record.get("params_hash", "unknown")

    return result, method, out_hash
