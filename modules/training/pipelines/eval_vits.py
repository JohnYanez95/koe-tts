"""
VITS Model Evaluation Pipeline.

Two modes:
- teacher: Uses real mel (posterior path) - validates reconstruction quality
- inference: Text-only synthesis (prior path) - validates end-to-end TTS

Usage:
    koe train eval vits jsut --run-id jsut_vits_core_20260125_121219
    koe train eval vits jsut --run-id <run> --mode inference --write-audio
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
import torchaudio
import yaml

if TYPE_CHECKING:
    from modules.training.common.events import EventLogger

from modules.data_engineering.common.paths import paths
from modules.training.audio import MelConfig, MelExtractor, load_audio
from modules.training.dataloading import TTSDataset, PHONEME_VOCAB
from modules.training.eval import compute_mel_metrics, compute_audio_metrics, EvalWriter
from modules.training.eval.metrics import aggregate_metrics
from modules.training.models.vits import create_vits_model
from modules.training.models.vits.losses import (
    mel_reconstruction_loss,
    kl_divergence_loss,
    duration_loss,
)


def compute_spectral_centroid(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """
    Compute mean spectral centroid as bandwidth proxy.

    Higher values indicate more high-frequency content (less muffled).

    Args:
        audio: [T] waveform
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Mean spectral centroid in Hz
    """
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=audio.device),
        return_complex=True,
    )
    mag = spec.abs()  # [freq, time]

    # Compute centroid
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[0], device=audio.device)
    centroid = (freqs.unsqueeze(1) * mag).sum(dim=0) / (mag.sum(dim=0) + 1e-8)

    return centroid.mean().item()


def compute_training_losses(
    outputs: dict,
    target_mel: torch.Tensor,
    mel_mask: torch.Tensor,
    phoneme_mask: torch.Tensor,
    mel_config: dict,
) -> dict:
    """
    Compute training-comparable losses from model outputs.

    These are the same losses used during training, allowing direct
    comparison between training curves and evaluation checkpoints.

    Args:
        outputs: Dict from model forward pass (teacher mode)
        target_mel: [B, n_mels, T_mel] target mel spectrogram
        mel_mask: [B, T_mel] valid mel frames
        phoneme_mask: [B, T_text] valid phoneme positions
        mel_config: Dict with mel parameters

    Returns:
        Dict with mel_loss, kl_loss, dur_loss (floats)
    """
    # Mel reconstruction loss
    mel_loss = mel_reconstruction_loss(
        pred_audio=outputs["audio"],
        target_mel=target_mel,
        mel_config=mel_config,
    )

    # KL divergence loss
    kl_loss = kl_divergence_loss(
        posterior_mean=outputs["posterior_mean"],
        posterior_log_var=outputs["posterior_log_var"],
        prior_mean=outputs["prior_mean_aligned"],
        prior_log_var=outputs["prior_log_var_aligned"],
        mask=mel_mask,
    )

    # Duration loss
    dur_loss = duration_loss(
        log_dur_pred=outputs["log_dur_pred"],
        dur_target=outputs["durations"],
        mask=phoneme_mask,
    )

    return {
        "mel_loss": mel_loss.item(),
        "kl_loss": kl_loss.item(),
        "dur_loss": dur_loss.item(),
    }


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


def print_summary_table(per_sample_metrics: list[dict], mode: str) -> None:
    """Print a summary table to console."""
    if not per_sample_metrics:
        print("No samples evaluated.")
        return

    # Header
    print()
    print("=" * 110)
    print(f"PER-SAMPLE RESULTS ({mode.upper()} mode)")
    print("=" * 110)
    print(f"{'Utterance ID':<20} | {'L1':>6} | {'L2':>6} | {'SNR':>6} | {'Dur':>5} | {'RMS':>6} | {'Peak':>5} | {'Sil%':>5} | {'SC':>6}")
    print("-" * 110)

    for m in per_sample_metrics:
        uid = m.get("utterance_id", "?")[:20]
        l1 = m.get("mel_l1", 0)
        l2 = m.get("mel_l2", 0)
        snr = m.get("snr_proxy_db", 0)
        dur = m.get("pred_duration_sec", m.get("duration_sec", 0))
        rms = m.get("pred_rms", 0)
        peak = m.get("pred_peak", 0)
        sil = m.get("pred_silence_pct", 0)
        sc = m.get("pred_spectral_centroid", 0)

        print(f"{uid:<20} | {l1:>6.3f} | {l2:>6.3f} | {snr:>6.2f} | {dur:>5.2f} | {rms:>6.4f} | {peak:>5.3f} | {sil:>5.1f} | {sc:>6.0f}")

    print("-" * 110)


def print_aggregate_summary(agg_metrics: dict, mode: str) -> None:
    """Print aggregate metrics summary."""
    print()
    print("=" * 60)
    print(f"AGGREGATE METRICS ({mode.upper()} mode)")
    print("=" * 60)

    # Group by metric type
    mel_metrics = ["mel_l1", "mel_l2", "snr_proxy_db"]
    audio_metrics = ["pred_rms", "pred_peak", "pred_silence_pct", "pred_duration_sec", "pred_spectral_centroid"]

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
            if "spectral_centroid" in base:
                print(f"  {base}: {agg_metrics[mean_key]:.1f} +/- {agg_metrics.get(std_key, 0):.1f} Hz")
            else:
                print(f"  {base}: {agg_metrics[mean_key]:.4f} +/- {agg_metrics.get(std_key, 0):.4f}")

    # Duration ratio (inference mode)
    if "duration_ratio_mean" in agg_metrics:
        print(f"\nDuration Ratio (pred/target): {agg_metrics['duration_ratio_mean']:.3f} +/- {agg_metrics.get('duration_ratio_std', 0):.3f}")

    print("=" * 60)


def evaluate(
    run_dir: Path,
    checkpoint_name: str = "best.pt",
    mode: str = "teacher",
    n_samples: int = 20,
    seed: int = 42,
    write_mels: bool = True,
    write_audio: bool = False,
    write_target_audio: Optional[bool] = None,
    duration_scale: float = 1.0,
    noise_scale: float = 0.667,
    device: str = "cuda",
    events: Optional["EventLogger"] = None,
) -> dict:
    """
    Run evaluation on VITS model.

    Args:
        run_dir: Path to training run directory
        checkpoint_name: Checkpoint filename (best.pt, final.pt, etc.)
        mode: Evaluation mode ('teacher' or 'inference')
        n_samples: Number of samples to evaluate
        seed: Random seed for sample selection
        write_mels: Write mel .npy files
        write_audio: Write predicted audio .wav files
        write_target_audio: Write target audio .wav files (default: same as write_audio for A/B)
        duration_scale: Duration scale for inference mode
        noise_scale: Noise scale for inference mode
        device: Device to run on
        events: Optional EventLogger for logging eval events to dashboard

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
    print(f"VITS MODEL EVALUATION ({mode.upper()} mode)")
    print("=" * 70)
    print(f"Run: {run_dir.name}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Mode: {mode}")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    if mode == "inference":
        print(f"Duration scale: {duration_scale}")
        print(f"Noise scale: {noise_scale}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    model_state, config = load_checkpoint(checkpoint_path, device)

    # Extract step from checkpoint for event logging
    ckpt_full = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_step = ckpt_full.get("step", 0)

    # Generate eval_id and log start event
    eval_id = f"eval_{mode}_{uuid.uuid4().hex[:8]}"
    if events:
        events.eval_started(
            eval_id=eval_id,
            run_id=run_dir.name,
            step=checkpoint_step,
            mode=mode,
            seed=seed,
            tag=f"{mode}_{checkpoint_name}",
        )

    # Get dataset info from config
    dataset_name = config.get("data", {}).get("dataset", "jsut")

    # Find cache directory
    from modules.training.pipelines.train_vits import find_cache_dir
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
    sample_rate = mel_config.sample_rate
    hop_length = mel_config.hop_length

    # Mel extractor for computing mel from generated audio
    mel_extractor = MelExtractor(mel_config)

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
    model = create_vits_model(config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # Setup eval writer
    mode_suffix = mode
    if mode == "inference":
        mode_suffix = f"inference_ds{duration_scale}"
    eval_dir = run_dir / "eval" / f"eval_{checkpoint_name.replace('.pt', '')}_{mode_suffix}_n{n_samples}_s{seed}"

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
        extra_info={
            "mode": mode,
            "duration_scale": duration_scale,
            "noise_scale": noise_scale,
        },
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

            # Create masks
            phoneme_mask = torch.ones(1, phoneme_ids.shape[1], dtype=torch.bool, device=device)
            mel_mask = torch.ones(1, target_mel.shape[1], dtype=torch.bool, device=device)

            # Generate audio
            with torch.no_grad():
                if mode == "teacher":
                    # Teacher-forced: use real mel to get posterior
                    outputs = model(
                        phonemes=phoneme_ids,
                        phoneme_mask=phoneme_mask,
                        mel=target_mel.unsqueeze(0),
                        mel_mask=mel_mask,
                    )
                    pred_audio = outputs["audio"]  # [1, 1, T_audio]
                else:
                    # Inference: text-only
                    pred_audio = model.inference(
                        phonemes=phoneme_ids,
                        phoneme_mask=phoneme_mask,
                        duration_scale=duration_scale,
                        noise_scale=noise_scale,
                    )  # [1, 1, T_audio]

            pred_audio = pred_audio.squeeze(0).squeeze(0)  # [T_audio]

            # Compute mel from generated audio
            pred_mel = mel_extractor(pred_audio.cpu(), sample_rate=sample_rate)
            pred_mel = pred_mel.to(device)  # [n_mels, T_mel_pred]

            # Match lengths for mel comparison (min of pred and target)
            min_mel_len = min(pred_mel.shape[1], target_mel.shape[1])
            pred_mel_matched = pred_mel[:, :min_mel_len]
            target_mel_matched = target_mel[:, :min_mel_len]
            mel_mask_matched = torch.ones(min_mel_len, dtype=torch.bool, device=device)

            # Compute mel metrics
            mel_metrics = compute_mel_metrics(
                pred_mel_matched,
                target_mel_matched,
                mel_mask_matched,
            )

            # Compute audio metrics
            pred_audio_cpu = pred_audio.cpu()
            audio_metrics_raw = compute_audio_metrics(
                pred_audio_cpu,
                sample_rate=sample_rate,
            )
            audio_metrics = {f"pred_{k}": v for k, v in audio_metrics_raw.items()}

            # Add spectral centroid
            spectral_centroid = compute_spectral_centroid(
                pred_audio_cpu,
                sample_rate=sample_rate,
                hop_length=hop_length,
            )
            audio_metrics["pred_spectral_centroid"] = spectral_centroid

            # Compute training-comparable losses (teacher mode only)
            if mode == "teacher":
                mel_cfg_dict = {
                    "n_fft": mel_config.n_fft,
                    "hop_length": mel_config.hop_length,
                    "win_length": mel_config.win_length,
                    "n_mels": mel_config.n_mels,
                    "sample_rate": mel_config.sample_rate,
                    "f_min": mel_config.f_min,
                    "f_max": mel_config.f_max,
                }
                training_losses = compute_training_losses(
                    outputs=outputs,
                    target_mel=target_mel.unsqueeze(0),  # Add batch dim
                    mel_mask=mel_mask,
                    phoneme_mask=phoneme_mask,
                    mel_config=mel_cfg_dict,
                )
            else:
                # Inference mode: no posterior/prior available for KL/dur loss
                training_losses = {
                    "mel_loss": None,
                    "kl_loss": None,
                    "dur_loss": None,
                }

            # Duration ratio (for inference mode)
            target_duration_sec = sample_info.get("duration_sec", 0)
            pred_duration_sec = len(pred_audio) / sample_rate
            duration_ratio = pred_duration_sec / target_duration_sec if target_duration_sec > 0 else 1.0

            # Load target audio for A/B comparison
            target_audio = None
            if write_target_audio:
                audio_path = sample_info.get("audio_path")
                if audio_path:
                    target_audio, _ = load_audio(audio_path, target_sr=sample_rate)
                    target_audio = target_audio.squeeze(0)  # [T]

            # Combine metrics
            sample_metrics = {
                **mel_metrics,
                **audio_metrics,
                **training_losses,
                "duration_sec": target_duration_sec,
                "pred_duration_sec": pred_duration_sec,
                "duration_ratio": duration_ratio,
            }

            # Write artifacts
            writer.write_sample(
                utterance_id=utterance_id,
                pred_mel=pred_mel.cpu(),
                target_mel=target_mel.cpu(),
                pred_audio=pred_audio_cpu,
                target_audio=target_audio,
                sample_rate=sample_rate,
                metrics=sample_metrics,
            )

            if mode == "teacher" and training_losses["mel_loss"] is not None:
                print(f"L1={mel_metrics['mel_l1']:.3f} mel={training_losses['mel_loss']:.3f} kl={training_losses['kl_loss']:.3f} dur={training_losses['dur_loss']:.3f}")
            else:
                print(f"L1={mel_metrics['mel_l1']:.3f} L2={mel_metrics['mel_l2']:.3f} SC={spectral_centroid:.0f}Hz")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("-" * 70)

    # Write per-sample metrics
    writer.write_per_sample_metrics()

    # Compute and write aggregate metrics
    per_sample = writer.get_per_sample_metrics()
    agg_metrics = aggregate_metrics(per_sample)
    writer.write_aggregate_metrics(agg_metrics)

    # Print summary
    print_summary_table(per_sample, mode)
    print_aggregate_summary(agg_metrics, mode)

    print()
    print(f"Eval artifacts written to: {eval_dir}")
    print()

    # Log eval completion
    if events:
        # Extract training-comparable losses for dashboard chart markers
        eval_losses = None
        if mode == "teacher":
            mel_loss = agg_metrics.get("mel_loss_mean")
            kl_loss = agg_metrics.get("kl_loss_mean")
            dur_loss = agg_metrics.get("dur_loss_mean")
            if mel_loss is not None:
                eval_losses = {
                    "mel_loss": mel_loss,
                    "kl_loss": kl_loss,
                    "dur_loss": dur_loss,
                }

        events.eval_complete(
            eval_id=eval_id,
            run_id=run_dir.name,
            step=checkpoint_step,
            artifact_dir=str(eval_dir),
            summary={
                "n_samples": len(per_sample),
                "mel_l1_mean": agg_metrics.get("mel_l1_mean"),
                "mel_loss_mean": agg_metrics.get("mel_loss_mean"),
                "kl_loss_mean": agg_metrics.get("kl_loss_mean"),
                "dur_loss_mean": agg_metrics.get("dur_loss_mean"),
            },
            losses=eval_losses,
        )

    return {
        "eval_dir": str(eval_dir),
        "eval_id": eval_id,
        "step": checkpoint_step,
        "n_samples": len(per_sample),
        "aggregate_metrics": agg_metrics,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate VITS model")
    parser.add_argument("dataset", help="Dataset name (for finding runs)")
    parser.add_argument("--run-id", "-r", required=True, help="Run ID (directory name)")
    parser.add_argument("--ckpt", "-c", default="best.pt", help="Checkpoint name")
    parser.add_argument("--mode", "-m", choices=["teacher", "inference"], default="teacher", help="Eval mode")
    parser.add_argument("--n-samples", "-n", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--write-audio", "-a", action="store_true", help="Write predicted audio")
    parser.add_argument("--write-target-audio", "-t", action="store_true", help="Write target audio")
    parser.add_argument("--no-mels", action="store_true", help="Don't write mel files")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Duration scale (inference mode)")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="Noise scale (inference mode)")
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
            candidates = list(paths.runs.glob(f"*{args.run_id}*"))
            if candidates:
                run_dir = candidates[0]
            else:
                print(f"Run not found: {args.run_id}")
                print(f"Available runs: {[p.name for p in paths.runs.glob('*') if p.is_dir()]}")
                sys.exit(1)

    # Create EventLogger for dashboard integration
    from modules.training.common.events import EventLogger, get_latest_session

    session_dir = get_latest_session(run_dir)
    session_id = session_dir.name if session_dir else None
    events = EventLogger(run_dir, session_id=session_id)

    result = evaluate(
        run_dir=run_dir,
        checkpoint_name=args.ckpt,
        mode=args.mode,
        n_samples=args.n_samples,
        seed=args.seed,
        write_mels=not args.no_mels,
        write_audio=args.write_audio,
        write_target_audio=args.write_target_audio,
        duration_scale=args.duration_scale,
        noise_scale=args.noise_scale,
        device=args.device,
        events=events,
    )

    return 0 if result["n_samples"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
