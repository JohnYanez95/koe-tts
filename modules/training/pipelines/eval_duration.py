"""
Duration Model Evaluation Pipeline.

Supports two modes:
1. Teacher-forced (like baseline): Use ground-truth durations, compare mel outputs
2. Free-running: Predict durations, evaluate synthesis quality

Usage:
    koe train eval duration jsut --run-id jsut_duration_20260125 --mode teacher
    koe train eval duration jsut --run-id <run> --mode free --write-audio
"""

import json
from pathlib import Path
from typing import Optional

import torch

from modules.training.audio import MelConfig, GriffinLimVocoder, load_audio
from modules.training.dataloading import TTSDataset
from modules.training.eval import compute_mel_metrics, compute_audio_metrics, EvalWriter
from modules.training.eval.metrics import aggregate_metrics
from modules.training.models.baseline_duration import create_duration_model


def select_eval_samples(
    dataset: TTSDataset,
    n_samples: int,
    seed: int,
) -> list[dict]:
    """
    Deterministically select eval samples.

    Selection is reproducible given the same seed and dataset.
    Samples are selected by sorting utterance_ids and taking every k-th.
    """
    sorted_items = sorted(dataset.items, key=lambda x: x["utterance_id"])

    if n_samples >= len(sorted_items):
        return sorted_items

    torch.manual_seed(seed)
    offset = torch.randint(0, len(sorted_items), (1,)).item()

    step = len(sorted_items) / n_samples
    indices = [int((offset + i * step) % len(sorted_items)) for i in range(n_samples)]
    indices = sorted(set(indices))[:n_samples]

    return [sorted_items[i] for i in indices]


def load_checkpoint(checkpoint_path: Path, device: str) -> tuple[dict, dict]:
    """Load checkpoint and return model state and config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return ckpt["model_state"], ckpt.get("config", {})


def print_summary_table(per_sample_metrics: list[dict], mode: str) -> None:
    """Print a summary table to console."""
    if not per_sample_metrics:
        print("No samples evaluated.")
        return

    print()
    print("=" * 110)
    print(f"PER-SAMPLE RESULTS ({mode.upper()} mode)")
    print("=" * 110)

    if mode == "teacher":
        print(f"{'Utterance ID':<20} | {'L1':>6} | {'L2':>6} | {'SNR':>6} | {'Dur':>5} | {'RMS':>6} | {'Peak':>5} | {'Sil%':>5}")
    else:
        # Free mode includes duration prediction metrics
        print(f"{'Utterance ID':<20} | {'L1':>6} | {'L2':>6} | {'DurMAE':>6} | {'PredLen':>7} | {'TgtLen':>6} | {'RMS':>6} | {'Sil%':>5}")
    print("-" * 110)

    for m in per_sample_metrics:
        uid = m.get("utterance_id", "?")[:20]
        l1 = m.get("mel_l1", 0)
        l2 = m.get("mel_l2", 0)

        if mode == "teacher":
            snr = m.get("snr_proxy_db", 0)
            dur = m.get("duration_sec", 0)
            rms = m.get("pred_rms", 0)
            peak = m.get("pred_peak", 0)
            sil = m.get("pred_silence_pct", 0)
            print(f"{uid:<20} | {l1:>6.3f} | {l2:>6.3f} | {snr:>6.2f} | {dur:>5.2f} | {rms:>6.4f} | {peak:>5.3f} | {sil:>5.1f}")
        else:
            dur_mae = m.get("dur_mae_frames", 0)
            pred_len = m.get("pred_mel_len", 0)
            tgt_len = m.get("target_mel_len", 0)
            rms = m.get("pred_rms", 0)
            sil = m.get("pred_silence_pct", 0)
            print(f"{uid:<20} | {l1:>6.3f} | {l2:>6.3f} | {dur_mae:>6.1f} | {pred_len:>7} | {tgt_len:>6} | {rms:>6.4f} | {sil:>5.1f}")

    print("-" * 110)


def print_aggregate_summary(agg_metrics: dict, mode: str) -> None:
    """Print aggregate metrics summary."""
    print()
    print("=" * 60)
    print(f"AGGREGATE METRICS ({mode.upper()} mode)")
    print("=" * 60)

    mel_metrics = ["mel_l1", "mel_l2", "snr_proxy_db"]
    duration_metrics = ["dur_mae_frames", "length_ratio"]
    audio_metrics = ["pred_rms", "pred_peak", "pred_silence_pct", "pred_duration_sec"]

    print("\nMel Metrics:")
    for base in mel_metrics:
        mean_key = f"{base}_mean"
        std_key = f"{base}_std"
        if mean_key in agg_metrics:
            print(f"  {base}: {agg_metrics[mean_key]:.4f} +/- {agg_metrics.get(std_key, 0):.4f}")

    if mode == "free":
        print("\nDuration Metrics:")
        for base in duration_metrics:
            mean_key = f"{base}_mean"
            std_key = f"{base}_std"
            if mean_key in agg_metrics:
                print(f"  {base}: {agg_metrics[mean_key]:.4f} +/- {agg_metrics.get(std_key, 0):.4f}")

    print("\nAudio Metrics (predicted):")
    for base in audio_metrics:
        mean_key = f"{base}_mean"
        std_key = f"{base}_std"
        if mean_key in agg_metrics:
            print(f"  {base}: {agg_metrics[mean_key]:.4f} +/- {agg_metrics.get(std_key, 0):.4f}")

    print("=" * 60)


def evaluate(
    run_dir: Path,
    checkpoint_name: str = "best.pt",
    mode: str = "teacher",  # "teacher" or "free"
    n_samples: int = 20,
    seed: int = 42,
    write_mels: bool = True,
    write_audio: bool = False,
    write_target_audio: Optional[bool] = None,
    duration_scale: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Run evaluation on duration model.

    Args:
        run_dir: Path to training run directory
        checkpoint_name: Checkpoint filename
        mode: "teacher" for teacher-forced, "free" for predicted durations
        n_samples: Number of samples to evaluate
        seed: Random seed for sample selection
        write_mels: Write mel .npy files
        write_audio: Write predicted audio .wav files
        write_target_audio: Write target audio .wav files
        duration_scale: Scale factor for predicted durations (free mode only)
        device: Device to run on

    Returns:
        Dict with evaluation results
    """
    run_dir = Path(run_dir)

    if write_target_audio is None:
        write_target_audio = write_audio

    checkpoint_path = run_dir / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 70)
    print("DURATION MODEL EVALUATION")
    print("=" * 70)
    print(f"Run: {run_dir.name}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Mode: {mode}")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    if mode == "free":
        print(f"Duration scale: {duration_scale}")
    print(f"Device: {device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    model_state, config = load_checkpoint(checkpoint_path, device)

    # Get dataset info from config
    dataset_name = config.get("data", {}).get("dataset", "jsut")

    # Find cache directory
    from modules.training.pipelines.train_duration import find_cache_dir
    cache_dir = find_cache_dir(dataset_name)
    print(f"Cache: {cache_dir}")

    # Create mel config
    mel_cfg = config.get("mel", {})
    mel_config = MelConfig(
        sample_rate=mel_cfg.get("sample_rate", 22050),
        n_mels=mel_cfg.get("n_mels", 80),
        n_fft=mel_cfg.get("n_fft", 1024),
        hop_length=mel_cfg.get("hop_length", 256),
        win_length=mel_cfg.get("win_length", 1024),
        f_min=mel_cfg.get("f_min", 0),
        f_max=mel_cfg.get("f_max", 8000),
    )

    # Load val dataset
    print("Loading validation dataset...")
    val_dataset = TTSDataset(
        cache_dir=cache_dir,
        split="val",
        mel_config=mel_config,
    )
    print(f"Val dataset size: {len(val_dataset)}")

    # Select eval samples
    print(f"Selecting {n_samples} eval samples (seed={seed})...")
    eval_samples = select_eval_samples(val_dataset, n_samples, seed)
    print(f"Selected: {len(eval_samples)} samples")

    # Create model
    print("Creating model...")
    model = create_duration_model(config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # Create vocoder if needed
    vocoder = None
    if write_audio:
        print("Creating Griffin-Lim vocoder...")
        vocoder = GriffinLimVocoder(config=mel_config, device=device)

    # Setup eval writer
    mode_suffix = f"_{mode}"
    if mode == "free" and duration_scale != 1.0:
        mode_suffix += f"_scale{duration_scale}"
    eval_dir = run_dir / "eval" / f"eval_{checkpoint_name.replace('.pt', '')}{mode_suffix}_n{n_samples}_s{seed}"

    writer = EvalWriter(
        eval_dir=eval_dir,
        write_mels=write_mels,
        write_audio=write_audio,
        write_target_audio=write_target_audio,
    )

    # Write manifest
    manifest_info = {
        "mode": mode,
        "duration_scale": duration_scale if mode == "free" else None,
    }
    writer.write_manifest(
        run_id=run_dir.name,
        checkpoint_path=str(checkpoint_path),
        config=config,
        n_samples=len(eval_samples),
        seed=seed,
        extra_info=manifest_info,
    )
    writer.write_eval_set(eval_samples)

    # Evaluate each sample
    print()
    print(f"Evaluating {len(eval_samples)} samples ({mode} mode)...")
    print("-" * 70)

    per_sample_metrics = []

    for i, sample_info in enumerate(eval_samples):
        utterance_id = sample_info["utterance_id"]
        print(f"  [{i+1}/{len(eval_samples)}] {utterance_id}...", end=" ", flush=True)

        try:
            # Find sample in dataset
            sample_idx = None
            for idx, item in enumerate(val_dataset.items):
                if item["utterance_id"] == utterance_id:
                    sample_idx = idx
                    break

            if sample_idx is None:
                print("SKIP (not found)")
                continue

            sample = val_dataset[sample_idx]
            if sample is None:
                print("SKIP (load failed)")
                continue

            # Prepare inputs
            phoneme_ids = sample["phoneme_ids"].unsqueeze(0).to(device)  # [1, T_phone]
            target_mel = sample["mel"].to(device)  # [n_mels, T_mel]
            durations = sample["durations"].unsqueeze(0).to(device)  # [1, T_phone]

            phoneme_mask = torch.ones(1, phoneme_ids.shape[1], dtype=torch.bool, device=device)
            target_mel_len = target_mel.shape[1]

            with torch.no_grad():
                if mode == "teacher":
                    # Teacher-forced: use ground-truth durations
                    mel_mask = torch.ones(1, target_mel_len, dtype=torch.bool, device=device)
                    pred_mel, log_dur_pred = model(
                        phonemes=phoneme_ids,
                        phoneme_mask=phoneme_mask,
                        durations=durations,
                        mel_mask=mel_mask,
                    )
                    pred_mel = pred_mel.squeeze(0)  # [n_mels, T_mel]
                    pred_durations = None

                else:
                    # Free-running: predict durations
                    pred_mel, pred_durations = model.inference(
                        phonemes=phoneme_ids,
                        phoneme_mask=phoneme_mask,
                        duration_scale=duration_scale,
                    )
                    pred_mel = pred_mel.squeeze(0)  # [n_mels, T_pred]
                    pred_durations = pred_durations.squeeze(0)  # [T_phone]

            # Compute mel metrics
            # For free mode, need to handle length mismatch
            if mode == "free":
                pred_len = pred_mel.shape[1]
                min_len = min(pred_len, target_mel_len)
                pred_mel_eval = pred_mel[:, :min_len]
                target_mel_eval = target_mel[:, :min_len]
                eval_mask = torch.ones(min_len, dtype=torch.bool, device=device)
            else:
                pred_mel_eval = pred_mel
                target_mel_eval = target_mel
                eval_mask = torch.ones(target_mel_len, dtype=torch.bool, device=device)

            mel_metrics = compute_mel_metrics(pred_mel_eval, target_mel_eval, eval_mask)

            # Duration metrics for free mode
            duration_metrics = {}
            if mode == "free" and pred_durations is not None:
                target_durations = durations.squeeze(0)
                dur_diff = (pred_durations.float() - target_durations.float()).abs()
                dur_mae = dur_diff[phoneme_mask.squeeze(0)].mean().item()
                length_ratio = pred_mel.shape[1] / target_mel_len

                duration_metrics = {
                    "dur_mae_frames": dur_mae,
                    "pred_mel_len": pred_mel.shape[1],
                    "target_mel_len": target_mel_len,
                    "length_ratio": length_ratio,
                }

            # Vocode if requested
            pred_audio = None
            target_audio = None
            audio_metrics = {}

            if vocoder is not None:
                pred_audio = vocoder(pred_mel)
                audio_metrics_pred = compute_audio_metrics(
                    pred_audio,
                    sample_rate=mel_config.sample_rate,
                )
                audio_metrics = {f"pred_{k}": v for k, v in audio_metrics_pred.items()}

            if write_target_audio:
                audio_path = sample_info.get("audio_path")
                if audio_path:
                    target_audio, _ = load_audio(audio_path, target_sr=mel_config.sample_rate)
                    target_audio = target_audio.squeeze(0)

            # Combine metrics
            sample_metrics = {
                "utterance_id": utterance_id,
                **mel_metrics,
                **duration_metrics,
                **audio_metrics,
                "duration_sec": sample_info.get("duration_sec", 0),
            }
            per_sample_metrics.append(sample_metrics)

            # Write artifacts
            writer.write_sample(
                utterance_id=utterance_id,
                pred_mel=pred_mel,
                target_mel=target_mel,
                pred_audio=pred_audio,
                target_audio=target_audio,
                sample_rate=mel_config.sample_rate,
                metrics=sample_metrics,
            )

            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Compute aggregates
    agg_metrics = aggregate_metrics(per_sample_metrics)

    # Write aggregate metrics
    writer.write_aggregate_metrics(agg_metrics)

    # Print results
    print_summary_table(per_sample_metrics, mode)
    print_aggregate_summary(agg_metrics, mode)

    print()
    print(f"Results written to: {eval_dir}")

    return {
        "n_samples": len(per_sample_metrics),
        "eval_dir": str(eval_dir),
        "mode": mode,
        "aggregate_metrics": agg_metrics,
        "per_sample_metrics": per_sample_metrics,
    }
