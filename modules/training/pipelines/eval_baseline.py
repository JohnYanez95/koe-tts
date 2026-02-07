"""
Baseline Model Evaluation Pipeline.

Teacher-forced evaluation on fixed val subset for reproducible metrics.

Usage:
    koe train eval baseline jsut --run-id jsut_baseline_mel_20260125_044418
    koe train eval baseline jsut --run-id <run> --ckpt best.pt --write-audio
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

from modules.data_engineering.common.paths import paths
from modules.training.audio import MelConfig, GriffinLimVocoder, load_audio
from modules.training.dataloading import TTSDataset, PHONEME_VOCAB
from modules.training.eval import compute_mel_metrics, compute_audio_metrics, EvalWriter
from modules.training.eval.metrics import aggregate_metrics
from modules.training.models.baseline_mel import create_model


def select_eval_samples(
    dataset: TTSDataset,
    n_samples: int,
    seed: int,
) -> list[dict]:
    """
    Deterministically select eval samples.

    Selection is reproducible given the same seed and dataset.
    Samples are selected by sorting utterance_ids and taking every k-th.

    Args:
        dataset: TTSDataset with val split loaded
        n_samples: Number of samples to select
        seed: Random seed for selection

    Returns:
        List of sample dicts from dataset.items
    """
    # Sort by utterance_id for determinism
    sorted_items = sorted(dataset.items, key=lambda x: x["utterance_id"])

    if n_samples >= len(sorted_items):
        return sorted_items

    # Use seed to determine starting offset, then take evenly spaced samples
    torch.manual_seed(seed)
    offset = torch.randint(0, len(sorted_items), (1,)).item()

    # Evenly spaced selection
    step = len(sorted_items) / n_samples
    indices = [int((offset + i * step) % len(sorted_items)) for i in range(n_samples)]

    # Sort indices for deterministic ordering
    indices = sorted(set(indices))[:n_samples]

    return [sorted_items[i] for i in indices]


def load_checkpoint(checkpoint_path: Path, device: str) -> tuple[dict, dict]:
    """
    Load checkpoint and return model state and config.

    Returns:
        Tuple of (model_state_dict, config_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return ckpt["model_state"], ckpt.get("config", {})


def print_summary_table(per_sample_metrics: list[dict]) -> None:
    """Print a summary table to console."""
    if not per_sample_metrics:
        print("No samples evaluated.")
        return

    # Header
    print()
    print("=" * 100)
    print("PER-SAMPLE RESULTS")
    print("=" * 100)
    print(f"{'Utterance ID':<20} | {'L1':>6} | {'L2':>6} | {'SNR':>6} | {'Dur':>5} | {'RMS':>6} | {'Peak':>5} | {'Sil%':>5}")
    print("-" * 100)

    for m in per_sample_metrics:
        uid = m.get("utterance_id", "?")[:20]
        l1 = m.get("mel_l1", 0)
        l2 = m.get("mel_l2", 0)
        snr = m.get("snr_proxy_db", 0)
        dur = m.get("pred_duration_sec", m.get("duration_sec", 0))
        rms = m.get("pred_rms", 0)
        peak = m.get("pred_peak", 0)
        sil = m.get("pred_silence_pct", 0)

        print(f"{uid:<20} | {l1:>6.3f} | {l2:>6.3f} | {snr:>6.2f} | {dur:>5.2f} | {rms:>6.4f} | {peak:>5.3f} | {sil:>5.1f}")

    print("-" * 100)


def print_aggregate_summary(agg_metrics: dict) -> None:
    """Print aggregate metrics summary."""
    print()
    print("=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)

    # Group by metric type
    mel_metrics = ["mel_l1", "mel_l2", "snr_proxy_db"]
    audio_metrics = ["pred_rms", "pred_peak", "pred_silence_pct", "pred_duration_sec"]

    print("\nMel Metrics:")
    for base in mel_metrics:
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
    n_samples: int = 20,
    seed: int = 42,
    write_mels: bool = True,
    write_audio: bool = False,
    write_target_audio: Optional[bool] = None,
    device: str = "cuda",
) -> dict:
    """
    Run evaluation on baseline model.

    Args:
        run_dir: Path to training run directory
        checkpoint_name: Checkpoint filename (best.pt, final.pt, etc.)
        n_samples: Number of samples to evaluate
        seed: Random seed for sample selection
        write_mels: Write mel .npy files
        write_audio: Write predicted audio .wav files
        write_target_audio: Write target audio .wav files (default: same as write_audio for A/B)
        device: Device to run on

    Returns:
        Dict with evaluation results
    """
    run_dir = Path(run_dir)

    # Default: write target audio when writing predicted audio (for A/B comparison)
    if write_target_audio is None:
        write_target_audio = write_audio
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)
    print(f"Run: {run_dir.name}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    model_state, config = load_checkpoint(checkpoint_path, device)

    # Get dataset info from config
    dataset_name = config.get("data", {}).get("dataset", "jsut")

    # Find cache directory
    from modules.training.pipelines.train_baseline import find_cache_dir
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

    # Select eval samples deterministically
    print(f"Selecting {n_samples} eval samples (seed={seed})...")
    eval_samples = select_eval_samples(val_dataset, n_samples, seed)
    print(f"Selected: {len(eval_samples)} samples")

    # Create model
    print("Creating model...")
    model = create_model(config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # Create vocoder if needed
    vocoder = None
    if write_audio:
        print("Creating Griffin-Lim vocoder...")
        vocoder = GriffinLimVocoder(config=mel_config, device=device)

    # Setup eval writer
    eval_dir = run_dir / "eval" / f"eval_{checkpoint_name.replace('.pt', '')}_n{n_samples}_s{seed}"
    writer = EvalWriter(
        eval_dir=eval_dir,
        write_mels=write_mels,
        write_audio=write_audio,
        write_target_audio=write_target_audio,
    )

    # Write manifest and eval set
    writer.write_manifest(
        run_id=run_dir.name,
        checkpoint_path=str(checkpoint_path),
        config=config,
        n_samples=len(eval_samples),
        seed=seed,
    )
    writer.write_eval_set(eval_samples)

    # Evaluate each sample
    print()
    print(f"Evaluating {len(eval_samples)} samples...")
    print("-" * 70)

    from modules.training.dataloading.dataset import phonemes_to_ids

    for i, sample_info in enumerate(eval_samples):
        utterance_id = sample_info["utterance_id"]
        print(f"  [{i+1}/{len(eval_samples)}] {utterance_id}...", end=" ", flush=True)

        try:
            # Load the actual sample from dataset
            # Find index in dataset
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
            mel_len = torch.tensor([target_mel.shape[1]], device=device)

            # Create masks
            phoneme_mask = torch.ones(1, phoneme_ids.shape[1], dtype=torch.bool, device=device)
            mel_mask = torch.ones(1, target_mel.shape[1], dtype=torch.bool, device=device)

            # Forward pass (teacher-forced)
            with torch.no_grad():
                pred_mel = model(
                    phonemes=phoneme_ids,
                    phoneme_mask=phoneme_mask,
                    mel_lens=mel_len,
                    mel_mask=mel_mask,
                )  # [1, n_mels, T_mel]

            pred_mel = pred_mel.squeeze(0)  # [n_mels, T_mel]

            # Compute mel metrics
            mel_metrics = compute_mel_metrics(
                pred_mel,
                target_mel,
                mel_mask.squeeze(0),
            )

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
                # Prefix with 'pred_'
                audio_metrics = {f"pred_{k}": v for k, v in audio_metrics_pred.items()}

            if write_target_audio:
                # Load original audio
                audio_path = sample_info.get("audio_path")
                if audio_path:
                    target_audio, _ = load_audio(audio_path, target_sr=mel_config.sample_rate)
                    target_audio = target_audio.squeeze(0)  # [T]

            # Combine metrics
            sample_metrics = {
                **mel_metrics,
                **audio_metrics,
                "duration_sec": sample_info.get("duration_sec", 0),
            }

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

            print(f"L1={mel_metrics['mel_l1']:.3f} L2={mel_metrics['mel_l2']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print("-" * 70)

    # Write per-sample metrics
    writer.write_per_sample_metrics()

    # Compute and write aggregate metrics
    per_sample = writer.get_per_sample_metrics()
    agg_metrics = aggregate_metrics(per_sample)
    writer.write_aggregate_metrics(agg_metrics)

    # Print summary
    print_summary_table(per_sample)
    print_aggregate_summary(agg_metrics)

    print()
    print(f"Eval artifacts written to: {eval_dir}")
    print()

    return {
        "eval_dir": str(eval_dir),
        "n_samples": len(per_sample),
        "aggregate_metrics": agg_metrics,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate baseline mel model")
    parser.add_argument("dataset", help="Dataset name (for finding runs)")
    parser.add_argument("--run-id", "-r", required=True, help="Run ID (directory name)")
    parser.add_argument("--ckpt", "-c", default="best.pt", help="Checkpoint name")
    parser.add_argument("--n-samples", "-n", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--write-audio", "-a", action="store_true", help="Write predicted audio")
    parser.add_argument("--write-target-audio", "-t", action="store_true", help="Write target audio")
    parser.add_argument("--no-mels", action="store_true", help="Don't write mel files")
    parser.add_argument("--device", "-d", default="cuda", help="Device")

    args = parser.parse_args()

    # Find run directory
    run_dir = paths.runs / args.run_id
    if not run_dir.exists():
        # Try with dataset prefix
        candidates = list(paths.runs.glob(f"{args.dataset}*{args.run_id}*"))
        if candidates:
            run_dir = candidates[0]
        else:
            print(f"Run not found: {args.run_id}")
            print(f"Available runs: {[p.name for p in paths.runs.glob('*') if p.is_dir()]}")
            sys.exit(1)

    result = evaluate(
        run_dir=run_dir,
        checkpoint_name=args.ckpt,
        n_samples=args.n_samples,
        seed=args.seed,
        write_mels=not args.no_mels,
        write_audio=args.write_audio,
        write_target_audio=args.write_target_audio,
        device=args.device,
    )

    return 0 if result["n_samples"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
