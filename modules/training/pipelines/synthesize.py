"""
Synthesis pipeline for koe synth command.

Synthesizes audio from text using a trained VITS model.

Usage:
    koe synth jsut --run-id jsut_vits_gan_20260125_... --text "水をマレーシアから買わなくてはならない。"
    koe synth jsut -r <run_id> --text "..." -o out.wav
    koe synth compare --a <run_a> --b <run_b> --text-file prompts.txt

Output:
    - <output>.wav: Synthesized audio
    - <output>.json: Metadata (run_id, checkpoint, phonemes, tokens, etc.)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.phonemes import (
    generate_phonemes_normalized,
    tokenize,
    CANONICAL_INVENTORY,
)
from modules.training.dataloading.dataset import phonemes_to_ids, PHONEME_VOCAB
from modules.training.models.vits import create_vits_model


# =============================================================================
# Audio Validation Helpers
# =============================================================================


def compute_audio_stats(audio: torch.Tensor, sample_rate: int) -> dict:
    """
    Compute quick audio statistics for validation.

    Args:
        audio: [T] waveform tensor
        sample_rate: Sample rate

    Returns:
        Dict with rms, peak, silence_pct, is_valid
    """
    audio_abs = audio.abs()
    rms = audio.pow(2).mean().sqrt().item()
    peak = audio_abs.max().item()

    # Silence detection (frames below -60dB)
    silence_threshold = 0.001  # ~-60dB
    silence_pct = (audio_abs < silence_threshold).float().mean().item() * 100

    # Check for invalid values
    is_finite = torch.isfinite(audio).all().item()

    return {
        "rms": rms,
        "peak": peak,
        "silence_pct": silence_pct,
        "is_finite": is_finite,
        "is_valid": is_finite and rms > 0.0001 and silence_pct < 99.0,
    }


def normalize_rms(
    audio: torch.Tensor,
    target_rms: float = 0.05,
    max_peak: float = 0.95,
) -> tuple[torch.Tensor, dict]:
    """
    Normalize audio to target RMS with clipping protection.

    Args:
        audio: [T] waveform tensor
        target_rms: Target RMS level (default 0.05, typical speech)
        max_peak: Maximum allowed peak after normalization (default 0.95)

    Returns:
        Tuple of (normalized_audio, normalization_stats)
    """
    current_rms = audio.pow(2).mean().sqrt().item()

    if current_rms < 1e-8:
        # Silent audio, can't normalize
        return audio, {
            "normalized": False,
            "reason": "silent",
            "original_rms": current_rms,
            "target_rms": target_rms,
            "gain_applied": 1.0,
        }

    # Compute gain needed
    gain = target_rms / current_rms

    # Apply gain
    normalized = audio * gain

    # Check for clipping
    peak = normalized.abs().max().item()
    clipped = False

    if peak > max_peak:
        # Scale down to avoid clipping
        clip_gain = max_peak / peak
        normalized = normalized * clip_gain
        gain = gain * clip_gain
        clipped = True

    final_rms = normalized.pow(2).mean().sqrt().item()
    final_peak = normalized.abs().max().item()

    return normalized, {
        "normalized": True,
        "original_rms": round(current_rms, 6),
        "target_rms": target_rms,
        "final_rms": round(final_rms, 6),
        "gain_applied": round(gain, 4),
        "final_peak": round(final_peak, 4),
        "clipped": clipped,
    }


def validate_tokens(phonemes: str, token_ids: list[int]) -> tuple[bool, list[str]]:
    """
    Validate that all phonemes are in vocabulary.

    Args:
        phonemes: Space-separated phoneme string
        token_ids: Token ID list

    Returns:
        Tuple of (is_valid, list_of_oov_phonemes)
    """
    tokens = tokenize(phonemes)
    oov = [t for t in tokens if t not in CANONICAL_INVENTORY]

    # Check for empty or only special tokens (BOS/EOS/PAD)
    # BOS=2, EOS=3, PAD=0, UNK=1
    content_tokens = [t for t in token_ids if t not in (0, 1, 2, 3)]
    is_empty = len(content_tokens) == 0

    return (len(oov) == 0 and not is_empty), oov


# =============================================================================
# Core Functions
# =============================================================================


def resolve_checkpoint(run_id: str, checkpoint: Optional[str] = None) -> Path:
    """
    Resolve run_id to checkpoint path.

    Looks in runs/<run_id>/checkpoints/ and prefers:
    1. Specified checkpoint name (can be full path or just filename)
    2. best.pt
    3. final.pt
    4. Newest *.pt file (by mtime, not lexical sort)

    Args:
        run_id: Run directory name or partial match
        checkpoint: Specific checkpoint name or path (optional)

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If run or checkpoint not found
    """
    runs_dir = paths.runs

    # If checkpoint is a full path, use it directly
    if checkpoint and Path(checkpoint).is_absolute():
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    # Try exact match first
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        # Try glob match, sorted by mtime (newest first)
        candidates = sorted(
            runs_dir.glob(f"*{run_id}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            available = [p.name for p in runs_dir.glob("*") if p.is_dir()]
            raise FileNotFoundError(
                f"Run not found: {run_id}\n"
                f"Available runs: {available[:10]}"
            )
        run_dir = candidates[0]

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

    # Resolve checkpoint file
    if checkpoint:
        ckpt_path = ckpt_dir / checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    # Try in order: best.pt, final.pt, newest by mtime
    for name in ["best.pt", "final.pt"]:
        ckpt_path = ckpt_dir / name
        if ckpt_path.exists():
            return ckpt_path

    # Fallback: newest .pt file by mtime
    pt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pt_files:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    return pt_files[0]


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """
    Load VITS model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config, checkpoint_info)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    config = ckpt.get("config", {})
    model = create_vits_model(config)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Extract speaker info if multi-speaker
    num_speakers = config.get("model", {}).get("num_speakers", 1)
    speaker_list = ckpt.get("speaker_list", None)  # May be saved in checkpoint

    checkpoint_info = {
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "num_speakers": num_speakers,
        "speaker_list": speaker_list,
    }

    return model, config, checkpoint_info


def resolve_speaker(
    speaker: Optional[str | int],
    speaker_list: Optional[list[str]],
    num_speakers: int,
) -> Optional[int]:
    """
    Resolve speaker argument to speaker index.

    Args:
        speaker: Speaker ID (string like "jvs001") or index (int)
        speaker_list: List of speaker IDs from checkpoint
        num_speakers: Number of speakers in model

    Returns:
        Speaker index (int) or None for single-speaker models

    Raises:
        ValueError: If speaker not found or out of range
    """
    if speaker is None:
        return 0 if num_speakers > 1 else None

    if num_speakers == 1:
        return None  # Single-speaker model, ignore speaker arg

    # If it's already an int, validate range
    if isinstance(speaker, int):
        if speaker < 0 or speaker >= num_speakers:
            raise ValueError(f"Speaker index {speaker} out of range [0, {num_speakers})")
        return speaker

    # String: look up in speaker_list
    if speaker_list is None:
        raise ValueError(
            f"Cannot resolve speaker '{speaker}': no speaker_list in checkpoint. "
            f"Use integer index instead (0-{num_speakers - 1})."
        )

    try:
        idx = speaker_list.index(speaker)
        return idx
    except ValueError:
        # Show available speakers in error
        preview = speaker_list[:10]
        suffix = f"... ({len(speaker_list)} total)" if len(speaker_list) > 10 else ""
        raise ValueError(
            f"Speaker '{speaker}' not found. Available: {preview}{suffix}"
        )


def text_to_tokens(text: str) -> tuple[torch.Tensor, str, list[int]]:
    """
    Convert Japanese text to phoneme tokens with validation.

    Args:
        text: Japanese text

    Returns:
        Tuple of (token_tensor, phoneme_string, token_ids_list)

    Raises:
        ValueError: If phonemization fails or produces invalid output
    """
    # Generate phonemes using pyopenjtalk
    phonemes = generate_phonemes_normalized(text)
    if not phonemes:
        raise ValueError(f"Failed to phonemize text (empty output): {text}")

    # Convert to token IDs
    token_ids = phonemes_to_ids(phonemes)
    token_list = token_ids.tolist()

    # Validate tokens
    is_valid, oov = validate_tokens(phonemes, token_list)
    if oov:
        raise ValueError(f"Out-of-vocabulary phonemes: {oov} in text: {text}")
    if not is_valid:
        raise ValueError(f"Empty phoneme sequence (only BOS/EOS) for text: {text}")

    return token_ids, phonemes, token_list


def synthesize_single(
    model,
    text: str,
    config: dict,
    device: str = "cuda",
    duration_scale: float = 1.0,
    noise_scale: float = 0.667,
    seed: Optional[int] = None,
    speaker_idx: Optional[int] = None,
    target_rms: Optional[float] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Synthesize audio from a single text input with validation.

    Args:
        model: Loaded VITS model
        text: Japanese text to synthesize
        config: Model config
        device: Device
        duration_scale: Duration multiplier (1.0 = normal speed)
        noise_scale: Noise scale for sampling (0.667 typical)
        seed: Random seed for reproducibility
        speaker_idx: Speaker index for multi-speaker model (optional)
        target_rms: Optional target RMS for loudness normalization (e.g., 0.05)

    Returns:
        Tuple of (audio_tensor, metadata_dict)
        metadata includes 'is_valid' and 'failure_reason' if failed
    """
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    # Get config values
    mel_cfg = config.get("mel", {})
    sample_rate = mel_cfg.get("sample_rate", 22050)
    hop_length = mel_cfg.get("hop_length", 256)

    # Convert text to tokens
    try:
        token_ids, phonemes, token_list = text_to_tokens(text)
    except ValueError as e:
        # Return empty audio with failure metadata
        return torch.zeros(sample_rate), {
            "text": text,
            "phonemes": None,
            "tokens": [],
            "n_tokens": 0,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "n_samples": 0,
            "duration_sec": 0.0,
            "duration_scale": duration_scale,
            "noise_scale": noise_scale,
            "seed": seed,
            "is_valid": False,
            "failure_reason": str(e),
        }

    # Prepare input
    phoneme_input = token_ids.unsqueeze(0).to(device)  # [1, T]
    phoneme_mask = torch.ones(1, len(token_ids), dtype=torch.bool, device=device)

    # Run inference
    with torch.inference_mode():
        audio = model.inference(
            phonemes=phoneme_input,
            phoneme_mask=phoneme_mask,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
            speaker_idx=speaker_idx,
        )  # [1, 1, T_audio]

    audio = audio.squeeze(0).squeeze(0).cpu()  # [T_audio]

    # Apply RMS normalization if requested
    norm_stats = None
    if target_rms is not None:
        audio, norm_stats = normalize_rms(audio, target_rms=target_rms)

    # Validate output
    audio_stats = compute_audio_stats(audio, sample_rate)

    # Build metadata
    metadata = {
        "text": text,
        "phonemes": phonemes,
        "tokens": token_list,
        "n_tokens": len(token_list),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "n_samples": len(audio),
        "duration_sec": len(audio) / sample_rate,
        "duration_scale": duration_scale,
        "noise_scale": noise_scale,
        "seed": seed,
        "speaker_idx": speaker_idx,
        # Audio stats
        "rms": audio_stats["rms"],
        "peak": audio_stats["peak"],
        "silence_pct": audio_stats["silence_pct"],
        "is_valid": audio_stats["is_valid"],
        # Normalization stats (if applied)
        "normalization": norm_stats,
    }

    if not audio_stats["is_valid"]:
        if not audio_stats["is_finite"]:
            metadata["failure_reason"] = "Audio contains NaN/Inf values"
        elif audio_stats["rms"] < 0.0001:
            metadata["failure_reason"] = f"Audio too quiet (RMS={audio_stats['rms']:.6f})"
        else:
            metadata["failure_reason"] = f"Audio mostly silent ({audio_stats['silence_pct']:.1f}% silence)"

    return audio, metadata


def synthesize(
    text: Optional[str] = None,
    text_file: Optional[str] = None,
    run_id: Optional[str] = None,
    checkpoint: Optional[str] = None,
    output: Optional[str] = None,
    duration_scale: float = 1.0,
    noise_scale: float = 0.667,
    seed: int = 42,
    device: str = "cuda",
    write_json: bool = True,
    speaker: Optional[int | str] = None,
    target_rms: Optional[float] = None,
) -> dict:
    """
    Synthesize audio from text using a trained VITS model.

    Args:
        text: Text to synthesize (single utterance)
        text_file: File with text lines (one per line)
        run_id: Run ID to load checkpoint from
        checkpoint: Specific checkpoint name or path (default: best.pt)
        output: Output path (default: ./synth_000.wav)
        duration_scale: Duration multiplier
        noise_scale: Noise scale
        seed: Random seed
        device: Device (cuda/cpu)
        write_json: Write metadata JSON alongside WAV
        speaker: Speaker index (int) or ID (str like "jvs001") for multi-speaker
        target_rms: Optional target RMS for loudness normalization (e.g., 0.05)

    Returns:
        Dict with synthesis results
    """
    # Validate inputs
    if not text and not text_file:
        raise ValueError("Must provide --text or --text-file")

    if not run_id and not checkpoint:
        raise ValueError("Must provide --run-id or --checkpoint")

    # Resolve checkpoint
    if run_id:
        checkpoint_path = resolve_checkpoint(run_id, checkpoint)
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_dir = checkpoint_path.parent.parent
    run_name = run_dir.name

    print("=" * 60)
    print("KOE SYNTH")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Device: {device}")
    print(f"Duration scale: {duration_scale}")
    print(f"Noise scale: {noise_scale}")
    print(f"Seed: {seed}")
    if target_rms is not None:
        print(f"Target RMS: {target_rms}")
    print()

    # Load model
    print("Loading model...")
    model, config, ckpt_info = load_model(checkpoint_path, device)
    sample_rate = config.get("mel", {}).get("sample_rate", 22050)
    num_speakers = ckpt_info.get("num_speakers", 1)
    speaker_list = ckpt_info.get("speaker_list")
    print(f"  Step: {ckpt_info['step']}, Best val loss: {ckpt_info['best_val_loss']:.4f}")
    print(f"  Speakers: {num_speakers}")

    # Resolve speaker (supports both int index and string ID)
    speaker_idx = resolve_speaker(speaker, speaker_list, num_speakers)
    if speaker_idx is not None:
        speaker_name = speaker_list[speaker_idx] if speaker_list else str(speaker_idx)
        print(f"  Using speaker: {speaker_name} (idx={speaker_idx})")
    print()

    # Collect texts to synthesize
    texts = []
    if text:
        texts.append(text)
    if text_file:
        with open(text_file, encoding="utf-8") as f:
            texts.extend([line.strip() for line in f if line.strip()])

    if not texts:
        raise ValueError("No text to synthesize")

    print(f"Synthesizing {len(texts)} utterance(s)...")
    print("-" * 60)

    results = []
    n_failed = 0

    for i, txt in enumerate(texts):
        # Determine output path
        if output:
            if len(texts) == 1:
                out_path = Path(output)
            else:
                base = Path(output)
                out_path = base.parent / f"{base.stem}_{i:03d}{base.suffix}"
        else:
            out_path = Path(f"synth_{i:03d}.wav")

        # Ensure .wav extension
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")

        print(f"  [{i+1}/{len(texts)}] {txt[:40]}{'...' if len(txt) > 40 else ''}")

        # Synthesize
        audio, metadata = synthesize_single(
            model=model,
            text=txt,
            config=config,
            device=device,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
            seed=seed + i,  # Different seed per utterance for variety
            speaker_idx=speaker_idx,
            target_rms=target_rms,
        )

        # Add run info to metadata
        metadata.update({
            "run_id": run_name,
            "checkpoint": checkpoint_path.name,
            "checkpoint_step": ckpt_info["step"],
            "dataset": config.get("data", {}).get("dataset", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

        # Save audio (even if invalid, for debugging)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio.numpy(), sample_rate)

        # Save metadata JSON
        if write_json:
            json_path = out_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Print status
        if metadata.get("is_valid", True):
            print(f"      Phonemes: {metadata.get('phonemes', 'N/A')}")
            norm_info = ""
            norm_stats = metadata.get("normalization")
            if norm_stats and norm_stats.get("normalized"):
                norm_info = f" | Gain: {norm_stats['gain_applied']:.2f}x"
                if norm_stats.get("clipped"):
                    norm_info += " (clipped)"
            print(f"      Duration: {metadata['duration_sec']:.2f}s | RMS: {metadata.get('rms', 0):.4f}{norm_info}")
            print(f"      Output: {out_path}")
        else:
            n_failed += 1
            print(f"      FAILED: {metadata.get('failure_reason', 'Unknown')}")
            print(f"      Output: {out_path} (invalid)")

        results.append({
            "text": txt,
            "output_path": str(out_path),
            "json_path": str(out_path.with_suffix(".json")) if write_json else None,
            "duration_sec": metadata["duration_sec"],
            "phonemes": metadata.get("phonemes"),
            "is_valid": metadata.get("is_valid", True),
        })

    print("-" * 60)
    print(f"Done. Synthesized {len(results)} utterance(s).")
    if n_failed > 0:
        print(f"  WARNING: {n_failed} failed (see .json files for details)")
    print()

    return {
        "status": "success" if n_failed == 0 else "partial",
        "run_id": run_name,
        "checkpoint": str(checkpoint_path),
        "n_utterances": len(results),
        "n_failed": n_failed,
        "results": results,
    }


# =============================================================================
# A/B Comparison
# =============================================================================


def synth_compare(
    run_a: str,
    run_b: str,
    text_file: str,
    output_dir: Optional[str] = None,
    checkpoint_a: Optional[str] = None,
    checkpoint_b: Optional[str] = None,
    duration_scale: float = 1.0,
    noise_scale: float = 0.667,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    A/B comparison synthesis for two runs.

    Generates identical prompts with identical seeds for fair comparison.

    Args:
        run_a: Run A identifier (baseline)
        run_b: Run B identifier (new)
        text_file: File with prompts (one per line)
        output_dir: Output directory (default: runs/synth_compare/<timestamp>)
        checkpoint_a: Specific checkpoint for A
        checkpoint_b: Specific checkpoint for B
        duration_scale: Duration multiplier
        noise_scale: Noise scale
        seed: Base random seed (same for both)
        device: Device

    Returns:
        Dict with comparison results and paths
    """
    # Load prompts
    prompts_path = Path(text_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {text_file}")

    with open(prompts_path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError("No prompts found in file")

    # Resolve checkpoints
    ckpt_a = resolve_checkpoint(run_a, checkpoint_a)
    ckpt_b = resolve_checkpoint(run_b, checkpoint_b)

    run_a_name = ckpt_a.parent.parent.name
    run_b_name = ckpt_b.parent.parent.name

    # Setup output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = paths.runs / "synth_compare" / f"compare_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)
    dir_a = out_dir / "a"
    dir_b = out_dir / "b"
    dir_a.mkdir(exist_ok=True)
    dir_b.mkdir(exist_ok=True)

    print("=" * 70)
    print("KOE SYNTH COMPARE (A/B)")
    print("=" * 70)
    print(f"Run A: {run_a_name} ({ckpt_a.name})")
    print(f"Run B: {run_b_name} ({ckpt_b.name})")
    print(f"Prompts: {len(prompts)}")
    print(f"Seed: {seed}")
    print(f"Output: {out_dir}")
    print()

    # Load models
    print("Loading model A...")
    model_a, config_a, info_a = load_model(ckpt_a, device)
    sample_rate_a = config_a.get("mel", {}).get("sample_rate", 22050)
    print(f"  Step: {info_a['step']}")

    print("Loading model B...")
    model_b, config_b, info_b = load_model(ckpt_b, device)
    sample_rate_b = config_b.get("mel", {}).get("sample_rate", 22050)
    print(f"  Step: {info_b['step']}")
    print()

    # Synthesize all prompts
    print(f"Synthesizing {len(prompts)} prompts...")
    print("-" * 70)

    results = []

    for i, prompt in enumerate(prompts):
        prompt_seed = seed + i
        prompt_id = f"{i:03d}"

        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        # Synthesize A
        audio_a, meta_a = synthesize_single(
            model=model_a,
            text=prompt,
            config=config_a,
            device=device,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
            seed=prompt_seed,
        )

        # Synthesize B (same seed!)
        audio_b, meta_b = synthesize_single(
            model=model_b,
            text=prompt,
            config=config_b,
            device=device,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
            seed=prompt_seed,
        )

        # Save audio
        path_a = dir_a / f"{prompt_id}.wav"
        path_b = dir_b / f"{prompt_id}.wav"
        sf.write(str(path_a), audio_a.numpy(), sample_rate_a)
        sf.write(str(path_b), audio_b.numpy(), sample_rate_b)

        # Save metadata
        meta_a["run_id"] = run_a_name
        meta_a["checkpoint"] = ckpt_a.name
        meta_b["run_id"] = run_b_name
        meta_b["checkpoint"] = ckpt_b.name

        with open(dir_a / f"{prompt_id}.json", "w", encoding="utf-8") as f:
            json.dump(meta_a, f, indent=2, ensure_ascii=False)
        with open(dir_b / f"{prompt_id}.json", "w", encoding="utf-8") as f:
            json.dump(meta_b, f, indent=2, ensure_ascii=False)

        results.append({
            "id": prompt_id,
            "text": prompt,
            "seed": prompt_seed,
            "a": {
                "path": f"a/{prompt_id}.wav",
                "duration_sec": meta_a["duration_sec"],
                "rms": meta_a.get("rms", 0),
                "is_valid": meta_a.get("is_valid", True),
            },
            "b": {
                "path": f"b/{prompt_id}.wav",
                "duration_sec": meta_b["duration_sec"],
                "rms": meta_b.get("rms", 0),
                "is_valid": meta_b.get("is_valid", True),
            },
        })

        status_a = "OK" if meta_a.get("is_valid", True) else "FAIL"
        status_b = "OK" if meta_b.get("is_valid", True) else "FAIL"
        print(f"      A: {meta_a['duration_sec']:.2f}s [{status_a}] | B: {meta_b['duration_sec']:.2f}s [{status_b}]")

    print("-" * 70)

    # Write comparison manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_a": {
            "name": run_a_name,
            "checkpoint": ckpt_a.name,
            "step": info_a["step"],
        },
        "run_b": {
            "name": run_b_name,
            "checkpoint": ckpt_b.name,
            "step": info_b["step"],
        },
        "params": {
            "duration_scale": duration_scale,
            "noise_scale": noise_scale,
            "seed": seed,
        },
        "n_prompts": len(prompts),
        "results": results,
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write HTML comparison page
    html = _generate_comparison_html(manifest, run_a_name, run_b_name)
    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)

    # Write markdown table
    md = _generate_comparison_markdown(manifest, run_a_name, run_b_name)
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Comparison complete!")
    print(f"  HTML: {out_dir / 'index.html'}")
    print(f"  Markdown: {out_dir / 'README.md'}")
    print()

    return {
        "status": "success",
        "output_dir": str(out_dir),
        "n_prompts": len(prompts),
        "manifest": manifest,
    }


def _generate_comparison_html(manifest: dict, run_a: str, run_b: str) -> str:
    """Generate HTML page for A/B listening comparison."""
    rows = []
    for r in manifest["results"]:
        text_preview = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
        rows.append(f"""
        <tr>
            <td>{r['id']}</td>
            <td title="{r['text']}">{text_preview}</td>
            <td><audio controls src="{r['a']['path']}"></audio><br><small>{r['a']['duration_sec']:.2f}s</small></td>
            <td><audio controls src="{r['b']['path']}"></audio><br><small>{r['b']['duration_sec']:.2f}s</small></td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Synth Compare: {run_a} vs {run_b}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tr:hover {{ background-color: #f9f9f9; }}
        audio {{ width: 200px; }}
        .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>A/B Synthesis Comparison</h1>
    <div class="meta">
        <p><strong>Run A:</strong> {run_a} (step {manifest['run_a']['step']})</p>
        <p><strong>Run B:</strong> {run_b} (step {manifest['run_b']['step']})</p>
        <p><strong>Params:</strong> seed={manifest['params']['seed']}, noise_scale={manifest['params']['noise_scale']}, duration_scale={manifest['params']['duration_scale']}</p>
        <p><strong>Generated:</strong> {manifest['created_at']}</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Text</th>
                <th>A ({run_a[:20]})</th>
                <th>B ({run_b[:20]})</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>"""


def _generate_comparison_markdown(manifest: dict, run_a: str, run_b: str) -> str:
    """Generate markdown table for comparison."""
    lines = [
        f"# Synth Compare: {run_a} vs {run_b}",
        "",
        f"**Run A:** {run_a} (step {manifest['run_a']['step']})",
        f"**Run B:** {run_b} (step {manifest['run_b']['step']})",
        f"**Params:** seed={manifest['params']['seed']}, noise_scale={manifest['params']['noise_scale']}",
        f"**Generated:** {manifest['created_at']}",
        "",
        "| ID | Text | A (dur) | B (dur) |",
        "|----|------|---------|---------|",
    ]

    for r in manifest["results"]:
        text_preview = r["text"][:40] + "..." if len(r["text"]) > 40 else r["text"]
        lines.append(f"| {r['id']} | {text_preview} | {r['a']['duration_sec']:.2f}s | {r['b']['duration_sec']:.2f}s |")

    lines.extend([
        "",
        "## Audio Files",
        "",
        "- `a/` - Run A outputs",
        "- `b/` - Run B outputs",
        "",
        "Open `index.html` for interactive listening.",
    ])

    return "\n".join(lines)


# =============================================================================
# Multi-Speaker Eval Grid
# =============================================================================

# Default eval prompts (diverse phonetic coverage)
DEFAULT_EVAL_PROMPTS = [
    "水をマレーシアから買わなくてはならない。",
    "あらゆる現実を、すべて自分の方へねじ曲げたのだ。",
    "本日は晴天なり。",
    "吾輩は猫である。名前はまだ無い。",
    "これはテストです。",
]

# Known "anchor" speakers to always include if present
ANCHOR_SPEAKERS = ["spk00"]  # JSUT single speaker


def select_default_speakers(
    speaker_list: list[str],
    n_speakers: int = 7,
    anchors: list[str] = ANCHOR_SPEAKERS,
) -> list[str]:
    """
    Select default speakers for eval: anchors + uniformly spaced others.

    Args:
        speaker_list: Full list of speaker IDs from checkpoint
        n_speakers: Target number of speakers (including anchors)
        anchors: Speaker IDs to always include if present

    Returns:
        List of speaker IDs for eval
    """
    if len(speaker_list) <= n_speakers:
        return speaker_list

    selected = []

    # Add anchors first (if they exist)
    for anchor in anchors:
        if anchor in speaker_list:
            selected.append(anchor)

    # Fill remaining slots with uniformly spaced speakers
    remaining = n_speakers - len(selected)
    if remaining > 0:
        # Exclude anchors from spacing selection
        other_speakers = [s for s in speaker_list if s not in selected]
        if other_speakers:
            # Pick uniformly spaced indices
            step = len(other_speakers) / remaining
            for i in range(remaining):
                idx = int(i * step)
                selected.append(other_speakers[idx])

    return selected


def find_closest_speaker(query: str, speaker_list: list[str], n: int = 3) -> list[str]:
    """Find closest matching speaker IDs (simple prefix/substring match)."""
    # Exact match
    if query in speaker_list:
        return [query]

    # Prefix match
    prefix_matches = [s for s in speaker_list if s.startswith(query)]
    if prefix_matches:
        return prefix_matches[:n]

    # Substring match
    substr_matches = [s for s in speaker_list if query in s]
    if substr_matches:
        return substr_matches[:n]

    # Levenshtein-ish: speakers that share common chars
    def overlap_score(s):
        return len(set(query) & set(s))

    scored = sorted(speaker_list, key=overlap_score, reverse=True)
    return scored[:n]


def eval_multispeaker(
    run_id: str,
    checkpoint: Optional[str] = None,
    speakers: Optional[list[str]] = None,
    prompts: Optional[list[str]] = None,
    prompts_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    duration_scale: float = 1.0,
    noise_scale: float = 0.667,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    Multi-speaker evaluation grid.

    Synthesizes each prompt with each speaker, producing a grid for comparison.
    Answers: Does speaker control work? Does text fidelity hold across speakers?

    Args:
        run_id: Run ID to evaluate
        checkpoint: Specific checkpoint (default: best.pt)
        speakers: List of speaker IDs (default: JSUT + 6 JVS)
        prompts: List of prompts (default: 5 diverse prompts)
        prompts_file: File with prompts (one per line)
        output_dir: Output directory (default: runs/<run>/eval/multispeaker_<seed>)
        duration_scale: Duration multiplier
        noise_scale: Noise scale
        seed: Random seed (same for all, for fair comparison)
        device: Device

    Returns:
        Dict with eval results and paths
    """
    # Resolve checkpoint
    ckpt_path = resolve_checkpoint(run_id, checkpoint)
    run_dir = ckpt_path.parent.parent
    run_name = run_dir.name

    # Setup output directory (include checkpoint name to avoid overwrites)
    ckpt_base = ckpt_path.stem  # e.g. "best", "step_025000"
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = run_dir / "eval" / f"multispeaker_{ckpt_base}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KOE EVAL MULTISPEAKER")
    print("=" * 70)
    print(f"Run: {run_name}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Seed: {seed}")
    print(f"Output: {out_dir}")
    print()

    # Load model
    print("Loading model...")
    model, config, ckpt_info = load_model(ckpt_path, device)
    sample_rate = config.get("mel", {}).get("sample_rate", 22050)
    num_speakers = ckpt_info.get("num_speakers", 1)
    speaker_list = ckpt_info.get("speaker_list")
    print(f"  Step: {ckpt_info['step']}")
    print(f"  Speakers in model: {num_speakers}")

    if num_speakers == 1:
        raise ValueError("Multi-speaker eval requires a multi-speaker model (num_speakers > 1)")

    if speaker_list is None:
        raise ValueError("Checkpoint missing speaker_list - cannot resolve speaker names")

    # Resolve speakers: use provided list or auto-select from checkpoint
    if speakers is None:
        speakers = select_default_speakers(speaker_list, n_speakers=7)
        print(f"  Auto-selected speakers: {speakers}")

    # Filter to speakers that exist in the model (with helpful warnings)
    valid_speakers = []
    for spk in speakers:
        try:
            idx = resolve_speaker(spk, speaker_list, num_speakers)
            valid_speakers.append((spk, idx))
        except ValueError:
            closest = find_closest_speaker(spk, speaker_list)
            print(f"  Warning: Speaker '{spk}' not found. Did you mean: {closest}?")

    if not valid_speakers:
        raise ValueError(
            f"No valid speakers found. Available: {speaker_list[:10]}..."
            if len(speaker_list) > 10 else f"Available: {speaker_list}"
        )

    print(f"  Eval speakers: {[s[0] for s in valid_speakers]}")

    # Resolve prompts
    if prompts_file:
        with open(prompts_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif prompts is None:
        prompts = DEFAULT_EVAL_PROMPTS

    print(f"  Prompts: {len(prompts)}")
    print()

    # Create speaker directories
    for spk_id, _ in valid_speakers:
        (out_dir / spk_id).mkdir(exist_ok=True)

    # Synthesize grid
    print(f"Synthesizing {len(prompts)} x {len(valid_speakers)} = {len(prompts) * len(valid_speakers)} samples...")
    print("-" * 70)

    results = []
    per_speaker_metrics = {spk_id: [] for spk_id, _ in valid_speakers}

    for prompt_idx, prompt in enumerate(prompts):
        prompt_id = f"{prompt_idx:03d}"
        print(f"  [{prompt_idx + 1}/{len(prompts)}] {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        # Compute phonemes once per prompt (same for all speakers)
        try:
            _, phonemes_str, _ = text_to_tokens(prompt)
        except ValueError:
            phonemes_str = None

        prompt_result = {
            "id": prompt_id,
            "text": prompt,
            "phonemes": phonemes_str,
            "speakers": {},
        }

        for spk_id, spk_idx in valid_speakers:
            # Synthesize with fixed seed for reproducibility
            audio, meta = synthesize_single(
                model=model,
                text=prompt,
                config=config,
                device=device,
                duration_scale=duration_scale,
                noise_scale=noise_scale,
                seed=seed,  # Same seed for all speakers
                speaker_idx=spk_idx,
            )

            # Save audio
            audio_path = out_dir / spk_id / f"{prompt_id}.wav"
            sf.write(str(audio_path), audio.numpy(), sample_rate)

            # Save metadata
            meta["speaker_id"] = spk_id
            meta["speaker_idx"] = spk_idx
            with open(audio_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            prompt_result["speakers"][spk_id] = {
                "path": f"{spk_id}/{prompt_id}.wav",
                "duration_sec": meta["duration_sec"],
                "rms": meta.get("rms", 0),
                "silence_pct": meta.get("silence_pct", 0),
                "is_valid": meta.get("is_valid", True),
            }

            per_speaker_metrics[spk_id].append({
                "prompt_id": prompt_id,
                "duration_sec": meta["duration_sec"],
                "rms": meta.get("rms", 0),
                "silence_pct": meta.get("silence_pct", 0),
                "is_valid": meta.get("is_valid", True),
            })

        results.append(prompt_result)

    print("-" * 70)

    # Compute per-speaker summary stats
    speaker_summary = {}
    for spk_id, metrics in per_speaker_metrics.items():
        durations = [m["duration_sec"] for m in metrics]
        rms_vals = [m["rms"] for m in metrics if m["rms"] > 0]
        silence_vals = [m["silence_pct"] for m in metrics]
        n_valid = sum(1 for m in metrics if m["is_valid"])
        speaker_summary[spk_id] = {
            "n_samples": len(metrics),
            "n_valid": n_valid,
            "mean_duration_sec": sum(durations) / len(durations) if durations else 0,
            "mean_rms": sum(rms_vals) / len(rms_vals) if rms_vals else 0,
            "mean_silence_pct": sum(silence_vals) / len(silence_vals) if silence_vals else 0,
        }

    # Compute inter-speaker mel distances (identity separation proxy)
    print("Computing inter-speaker distances...")
    separation_metrics = _compute_speaker_separation(
        out_dir, results, valid_speakers, config
    )

    # Write manifest
    manifest = {
        "schema_version": 2,  # v2: added phonemes field to results
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "name": run_name,
            "checkpoint": ckpt_path.name,
            "step": ckpt_info["step"],
            "num_speakers": num_speakers,
        },
        "params": {
            "duration_scale": duration_scale,
            "noise_scale": noise_scale,
            "seed": seed,
        },
        "speakers": [s[0] for s in valid_speakers],
        "n_prompts": len(prompts),
        "results": results,
        "per_speaker_summary": speaker_summary,
        "separation_metrics": separation_metrics,
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write per-speaker metrics
    with open(out_dir / "per_speaker_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_speaker_metrics, f, indent=2, ensure_ascii=False)

    # Generate HTML grid
    html = _generate_multispeaker_html(manifest, prompts, valid_speakers)
    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Eval complete!")
    print(f"  HTML: {out_dir / 'index.html'}")
    print(f"  Manifest: {out_dir / 'manifest.json'}")
    print()

    # Print speaker summary with self-consistency
    print("Per-speaker summary:")
    per_spk_consistency = separation_metrics.get("per_speaker_consistency", {})
    for spk_id, stats in speaker_summary.items():
        consistency = per_spk_consistency.get(spk_id, {})
        intra_dist = consistency.get("mean_intra_distance", 0.0)
        is_anomalous = consistency.get("is_anomalous", False)
        anomaly_flag = " [ANOMALOUS]" if is_anomalous else ""
        print(f"  {spk_id}: {stats['n_valid']}/{stats['n_samples']} valid, "
              f"dur={stats['mean_duration_sec']:.2f}s, rms={stats['mean_rms']:.4f}, "
              f"silence={stats['mean_silence_pct']:.1f}%, consist={intra_dist:.4f}{anomaly_flag}")

    # Print separation metrics
    print()
    print("Speaker separation (identity proxy):")
    print(f"  Mean inter-speaker mel L1: {separation_metrics['mean_inter_speaker_distance']:.4f}")
    print(f"  Per-prompt std: {separation_metrics['inter_speaker_std']:.4f}")
    print(f"  Median intra-speaker: {separation_metrics['median_intra_speaker_distance']:.4f}")
    if separation_metrics["mean_inter_speaker_distance"] < 0.05:
        print("  WARNING: Low separation - speaker conditioning may not be effective")

    # Warn about anomalous speakers
    anomalous = [spk for spk, c in per_spk_consistency.items() if c.get("is_anomalous", False)]
    if anomalous:
        print(f"  WARNING: Anomalous speakers (high intra-variance): {anomalous}")

    return {
        "status": "success",
        "output_dir": str(out_dir),
        "n_prompts": len(prompts),
        "n_speakers": len(valid_speakers),
        "manifest": manifest,
    }


def _compute_speaker_separation(
    out_dir: Path,
    results: list[dict],
    speakers: list[tuple[str, int]],
    config: dict,
) -> dict:
    """
    Compute inter-speaker mel distances as a proxy for identity separation.

    For each prompt, computes pairwise mel L1 distances between speakers.
    Higher mean distance = better speaker differentiation.

    Returns:
        Dict with separation metrics
    """
    from modules.training.audio import MelExtractor, MelConfig
    import numpy as np

    mel_cfg = config.get("mel", {})
    mel_extractor = MelExtractor(MelConfig(
        sample_rate=mel_cfg.get("sample_rate", 22050),
        n_mels=mel_cfg.get("n_mels", 80),
        n_fft=mel_cfg.get("n_fft", 1024),
        hop_length=mel_cfg.get("hop_length", 256),
    ))

    speaker_ids = [s[0] for s in speakers]
    per_prompt_distances = []

    # Also collect mels per speaker for self-consistency computation
    speaker_all_mels: dict[str, list] = {spk_id: [] for spk_id in speaker_ids}

    for result in results:
        prompt_id = result["id"]

        # Load mels for all speakers
        speaker_mels = {}
        for spk_id in speaker_ids:
            audio_path = out_dir / spk_id / f"{prompt_id}.wav"
            if audio_path.exists():
                import soundfile as sf_read
                audio, sr = sf_read.read(str(audio_path))
                audio_tensor = torch.from_numpy(audio).float()
                mel = mel_extractor(audio_tensor)
                speaker_mels[spk_id] = mel
                speaker_all_mels[spk_id].append(mel)

        # Compute pairwise distances
        pairwise_dists = []
        spk_list = list(speaker_mels.keys())
        for i in range(len(spk_list)):
            for j in range(i + 1, len(spk_list)):
                mel_a = speaker_mels[spk_list[i]]
                mel_b = speaker_mels[spk_list[j]]
                # Truncate to same length
                min_frames = min(mel_a.shape[1], mel_b.shape[1])
                dist = (mel_a[:, :min_frames] - mel_b[:, :min_frames]).abs().mean().item()
                pairwise_dists.append(dist)

        if pairwise_dists:
            per_prompt_distances.append(np.mean(pairwise_dists))

    # Aggregate inter-speaker
    if per_prompt_distances:
        mean_dist = float(np.mean(per_prompt_distances))
        std_dist = float(np.std(per_prompt_distances))
    else:
        mean_dist = 0.0
        std_dist = 0.0

    # Compute per-speaker self-consistency (intra-speaker variance across prompts)
    # Higher value = less consistent outputs for that speaker
    per_speaker_consistency = {}
    for spk_id, mels in speaker_all_mels.items():
        if len(mels) < 2:
            per_speaker_consistency[spk_id] = {"mean_intra_distance": 0.0, "n_pairs": 0}
            continue

        # Pairwise distances within this speaker's outputs
        intra_dists = []
        for i in range(len(mels)):
            for j in range(i + 1, len(mels)):
                min_frames = min(mels[i].shape[1], mels[j].shape[1])
                dist = (mels[i][:, :min_frames] - mels[j][:, :min_frames]).abs().mean().item()
                intra_dists.append(dist)

        per_speaker_consistency[spk_id] = {
            "mean_intra_distance": float(np.mean(intra_dists)),
            "n_pairs": len(intra_dists),
        }

    # Flag speakers with anomalous consistency (much higher than median)
    intra_values = [v["mean_intra_distance"] for v in per_speaker_consistency.values() if v["n_pairs"] > 0]
    if intra_values:
        median_intra = float(np.median(intra_values))
        for spk_id, stats in per_speaker_consistency.items():
            # Flag if > 2x median (likely unstable speaker)
            stats["is_anomalous"] = stats["mean_intra_distance"] > 2 * median_intra if median_intra > 0 else False
    else:
        median_intra = 0.0

    return {
        "mean_inter_speaker_distance": mean_dist,
        "inter_speaker_std": std_dist,
        "n_prompts_computed": len(per_prompt_distances),
        "per_prompt_distances": per_prompt_distances,
        "per_speaker_consistency": per_speaker_consistency,
        "median_intra_speaker_distance": median_intra,
    }


def _format_speaker_row(spk: str, manifest: dict) -> str:
    """Format a speaker row for the HTML summary table."""
    import html as html_lib

    summary = manifest["per_speaker_summary"][spk]
    consistency = manifest["separation_metrics"]["per_speaker_consistency"].get(spk, {})
    intra_dist = consistency.get("mean_intra_distance", 0.0)
    is_anomalous = consistency.get("is_anomalous", False)
    silence_pct = summary.get("mean_silence_pct", 0.0)

    # Style anomalous speakers with warning color
    row_class = ' class="anomalous-row"' if is_anomalous else ""
    warning = " ⚠" if is_anomalous else ""

    # High silence (>80%) is also a warning sign
    silence_style = ' style="color: #b45309; font-weight: 600;"' if silence_pct > 80 else ""

    spk_escaped = html_lib.escape(spk)

    return (
        f"<tr{row_class}>"
        f"<td>{spk_escaped}{warning}</td>"
        f"<td>{summary['n_valid']}/{summary['n_samples']}</td>"
        f"<td>{summary['mean_duration_sec']:.2f}s</td>"
        f"<td>{summary['mean_rms']:.4f}</td>"
        f"<td{silence_style}>{silence_pct:.1f}%</td>"
        f"<td>{intra_dist:.4f}</td>"
        f"</tr>"
    )


def _generate_multispeaker_html(
    manifest: dict,
    prompts: list[str],
    speakers: list[tuple[str, int]],
) -> str:
    """Generate HTML grid for multi-speaker eval with phonemes display."""
    import html as html_lib

    speaker_ids = [s[0] for s in speakers]
    run_name_escaped = html_lib.escape(manifest['run']['name'])

    # Header row with sticky positioning
    header_cells = "<th class=\"sticky-col\">ID</th><th class=\"sticky-col text-col\">Text</th>" + "".join(
        f"<th>{html_lib.escape(spk)}</th>" for spk in speaker_ids
    )

    # Data rows with zebra striping
    rows = []
    for i, r in enumerate(manifest["results"]):
        text_escaped = html_lib.escape(r["text"])
        text_preview = text_escaped[:60] + "..." if len(text_escaped) > 60 else text_escaped
        phonemes_raw = r.get("phonemes")
        phonemes_escaped = html_lib.escape(phonemes_raw) if phonemes_raw else ""
        row_class = "even-row" if i % 2 == 0 else "odd-row"

        # Text cell with phonemes underneath (clamp + expand on click)
        # For backwards-compat: show placeholder if phonemes missing (v1 manifests)
        if phonemes_escaped:
            phonemes_div = f'<div class="phonemes" onclick="this.classList.toggle(\'expanded\')" title="Click to expand">{phonemes_escaped}</div>'
        else:
            phonemes_div = '<div class="phonemes unavailable">(phonemes unavailable)</div>'

        text_cell = f'''<td class="sticky-col text-col" title="{text_escaped}">
            <div class="text-content">{text_preview}</div>
            {phonemes_div}
        </td>'''

        cells = [f"<td class=\"sticky-col\">{r['id']}</td>", text_cell]

        for spk_id in speaker_ids:
            spk_data = r["speakers"].get(spk_id, {})
            path = spk_data.get("path", "")
            dur = spk_data.get("duration_sec", 0)
            is_valid = spk_data.get("is_valid", True)
            status_class = "" if is_valid else " invalid"
            cells.append(
                f'<td class="audio-cell{status_class}">'
                f'<audio controls src="{html_lib.escape(path)}"></audio>'
                f'<span class="duration">{dur:.2f}s</span>'
                f'</td>'
            )

        rows.append(f'<tr class="{row_class}">' + "".join(cells) + "</tr>")

    # Format speaker summary rows
    summary_rows = "".join(_format_speaker_row(spk, manifest) for spk in speaker_ids)

    # Separation status
    sep_dist = manifest['separation_metrics']['mean_inter_speaker_distance']
    sep_status_class = "warning" if sep_dist < 0.05 else "success"
    sep_status_text = "Low separation - conditioning may not be effective" if sep_dist < 0.05 else "Separation looks reasonable"

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Multi-Speaker Eval: {run_name_escaped}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #fafafa;
            color: #333;
        }}
        h1 {{
            color: #1a1a1a;
            font-size: 1.5rem;
            margin-bottom: 16px;
        }}
        h3 {{
            color: #444;
            font-size: 1rem;
            margin: 20px 0 12px 0;
        }}

        /* Metadata header */
        .meta {{
            background: #fff;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            font-size: 0.9rem;
            color: #555;
        }}
        .meta p {{ margin: 4px 0; }}
        .meta strong {{ color: #333; }}
        .meta-hint {{
            margin-top: 12px;
            font-size: 0.8rem;
            color: #888;
            font-style: italic;
        }}

        /* Table container for horizontal scroll */
        .table-container {{
            overflow-x: auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        /* Main table */
        table {{
            border-collapse: collapse;
            width: 100%;
            min-width: max-content;
        }}
        th, td {{
            border: 1px solid #e5e5e5;
            padding: 10px 12px;
            text-align: center;
            vertical-align: top;
        }}

        /* Sticky header */
        thead th {{
            background: #f5f5f5;
            position: sticky;
            top: 0;
            z-index: 20;
            font-weight: 600;
            font-size: 0.85rem;
            color: #555;
        }}

        /* Sticky first columns (ID + Text) */
        .sticky-col {{
            position: sticky;
            background: inherit;
            z-index: 10;
        }}
        td.sticky-col:first-child,
        th.sticky-col:first-child {{
            left: 0;
            min-width: 50px;
        }}
        .text-col {{
            left: 50px;
            min-width: 280px;
            max-width: 320px;
            text-align: left !important;
        }}
        thead th.sticky-col {{
            z-index: 30;
            background: #f5f5f5;
        }}

        /* Zebra striping */
        .even-row {{ background: #fff; }}
        .odd-row {{ background: #fafafa; }}
        tr:hover {{ background: #f0f7ff !important; }}

        /* Text content styling */
        .text-content {{
            font-size: 0.9rem;
            color: #333;
            margin-bottom: 6px;
        }}

        /* Phonemes display (clamped by default, expand on click) */
        .phonemes {{
            font-family: "SF Mono", Monaco, Consolas, monospace;
            font-size: 0.75rem;
            color: #888;
            background: #f0f0f0;
            padding: 4px 6px;
            border-radius: 4px;
            max-height: 1.8em;
            overflow: hidden;
            cursor: pointer;
            transition: max-height 0.2s ease;
            word-break: break-all;
        }}
        .phonemes:hover {{
            background: #e8e8e8;
        }}
        .phonemes.expanded {{
            max-height: none;
        }}
        .phonemes:empty {{
            display: none;
        }}
        .phonemes.unavailable {{
            font-style: italic;
            color: #aaa;
            cursor: default;
        }}

        /* Audio cells */
        .audio-cell {{
            min-width: 180px;
        }}
        .audio-cell audio {{
            width: 160px;
            height: 32px;
        }}
        .audio-cell .duration {{
            display: block;
            font-size: 0.75rem;
            color: #888;
            margin-top: 4px;
        }}
        .audio-cell.invalid {{
            background: #fef2f2 !important;
        }}

        /* Summary section */
        .summary {{
            margin-top: 24px;
            padding: 16px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary table {{
            width: auto;
            min-width: 600px;
        }}
        .summary th {{
            position: static;
            background: #f5f5f5;
        }}
        .summary td {{
            font-size: 0.85rem;
        }}

        /* Status indicators */
        .status-warning {{
            color: #b45309;
            background: #fef3c7;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid #f59e0b;
        }}
        .status-success {{
            color: #047857;
            background: #d1fae5;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid #10b981;
        }}
        .anomalous-row {{
            background: #fef2f2 !important;
        }}
    </style>
</head>
<body>
    <h1>Multi-Speaker Evaluation Grid</h1>
    <div class="meta">
        <p><strong>Run:</strong> {run_name_escaped} (step {manifest['run']['step']})</p>
        <p><strong>Speakers:</strong> {', '.join(html_lib.escape(s) for s in speaker_ids)}</p>
        <p><strong>Params:</strong> seed={manifest['params']['seed']}, noise_scale={manifest['params']['noise_scale']}</p>
        <p><strong>Generated:</strong> {manifest['created_at']}</p>
        <p class="meta-hint">Phonemes: visible below text (click to expand)</p>
    </div>

    <div class="table-container">
        <table>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>

    <div class="summary">
        <h3>Per-Speaker Summary</h3>
        <table>
            <tr><th>Speaker</th><th>Valid</th><th>Duration</th><th>RMS</th><th>Silence%</th><th>Consistency</th></tr>
            {summary_rows}
        </table>

        <h3>Speaker Separation (Identity Proxy)</h3>
        <p><strong>Mean inter-speaker mel L1:</strong> {sep_dist:.4f}</p>
        <p><strong>Std across prompts:</strong> {manifest['separation_metrics']['inter_speaker_std']:.4f}</p>
        <p class="status-{sep_status_class}">{sep_status_text}</p>
        <p><strong>Median intra-speaker distance:</strong> {manifest['separation_metrics']['median_intra_speaker_distance']:.4f}</p>
    </div>
</body>
</html>'''


# =============================================================================
# Sanity Probes
# =============================================================================


def probe_speaker_determinism(
    run_id: str,
    speaker: str | int,
    text: str = "本日は晴天なり。",
    checkpoint: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    Probe A: Same speaker + same seed + same text → identical audio.

    Verifies that synthesis is deterministic given fixed inputs.

    Returns:
        Dict with match status and max absolute difference
    """
    ckpt_path = resolve_checkpoint(run_id, checkpoint)
    model, config, ckpt_info = load_model(ckpt_path, device)
    speaker_list = ckpt_info.get("speaker_list")
    num_speakers = ckpt_info.get("num_speakers", 1)

    speaker_idx = resolve_speaker(speaker, speaker_list, num_speakers)

    # Run twice with same seed
    audio1, _ = synthesize_single(model, text, config, device, seed=seed, speaker_idx=speaker_idx)
    audio2, _ = synthesize_single(model, text, config, device, seed=seed, speaker_idx=speaker_idx)

    # Compare
    max_diff = (audio1 - audio2).abs().max().item()
    is_match = max_diff < 1e-5  # Allow tiny float noise

    return {
        "probe": "speaker_determinism",
        "speaker": speaker,
        "seed": seed,
        "text": text,
        "max_abs_diff": max_diff,
        "is_match": is_match,
        "status": "PASS" if is_match else "FAIL",
    }


def probe_speaker_difference(
    run_id: str,
    speaker_a: str | int,
    speaker_b: str | int,
    text: str = "本日は晴天なり。",
    checkpoint: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda",
    mel_l1_threshold: float = 0.1,
) -> dict:
    """
    Probe B: Different speaker + same seed + same text → different audio.

    Verifies that speaker conditioning actually changes output.

    Returns:
        Dict with difference metrics and pass/fail status
    """
    from modules.training.audio import MelExtractor, MelConfig

    ckpt_path = resolve_checkpoint(run_id, checkpoint)
    model, config, ckpt_info = load_model(ckpt_path, device)
    speaker_list = ckpt_info.get("speaker_list")
    num_speakers = ckpt_info.get("num_speakers", 1)

    idx_a = resolve_speaker(speaker_a, speaker_list, num_speakers)
    idx_b = resolve_speaker(speaker_b, speaker_list, num_speakers)

    if idx_a == idx_b:
        raise ValueError(f"speaker_a and speaker_b resolve to same index: {idx_a}")

    # Synthesize both
    audio_a, _ = synthesize_single(model, text, config, device, seed=seed, speaker_idx=idx_a)
    audio_b, _ = synthesize_single(model, text, config, device, seed=seed, speaker_idx=idx_b)

    # Compute mel L1 distance
    mel_cfg = config.get("mel", {})
    mel_extractor = MelExtractor(MelConfig(
        sample_rate=mel_cfg.get("sample_rate", 22050),
        n_mels=mel_cfg.get("n_mels", 80),
        n_fft=mel_cfg.get("n_fft", 1024),
        hop_length=mel_cfg.get("hop_length", 256),
    ))

    # Truncate to same length for comparison
    min_len = min(len(audio_a), len(audio_b))
    mel_a = mel_extractor(audio_a[:min_len])
    mel_b = mel_extractor(audio_b[:min_len])

    # Truncate mel frames to match
    min_frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_l1 = (mel_a[:, :min_frames] - mel_b[:, :min_frames]).abs().mean().item()

    # Also compute waveform correlation (should be low for different speakers)
    audio_corr = torch.corrcoef(torch.stack([audio_a[:min_len], audio_b[:min_len]]))[0, 1].item()

    is_different = mel_l1 > mel_l1_threshold

    return {
        "probe": "speaker_difference",
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "seed": seed,
        "text": text,
        "mel_l1": mel_l1,
        "mel_l1_threshold": mel_l1_threshold,
        "waveform_correlation": audio_corr,
        "is_different": is_different,
        "status": "PASS" if is_different else "FAIL",
    }


# =============================================================================
# CLI Entry Points
# =============================================================================


def main():
    """CLI entry point (for direct module invocation)."""
    import argparse

    parser = argparse.ArgumentParser(description="Synthesize audio with VITS")
    parser.add_argument("--text", "-t", help="Text to synthesize")
    parser.add_argument("--text-file", "-f", help="File with text lines")
    parser.add_argument("--run-id", "-r", help="Run ID")
    parser.add_argument("--checkpoint", "-c", help="Checkpoint path or name")
    parser.add_argument("--output", "-o", default="synth_output.wav", help="Output path")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Duration scale")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="Noise scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", "-d", default="cuda", help="Device")
    parser.add_argument("--no-json", action="store_true", help="Don't write metadata JSON")
    parser.add_argument("--speaker", "-s", type=int, default=None,
                        help="Speaker index for multi-speaker models (0-indexed)")
    parser.add_argument("--target-rms", type=float, default=None,
                        help="Target RMS for loudness normalization (e.g., 0.05 for typical speech)")

    args = parser.parse_args()

    result = synthesize(
        text=args.text,
        text_file=args.text_file,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        output=args.output,
        duration_scale=args.duration_scale,
        noise_scale=args.noise_scale,
        seed=args.seed,
        device=args.device,
        write_json=not args.no_json,
        speaker=args.speaker,
        target_rms=args.target_rms,
    )

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
