"""
Eval artifact writer.

Writes evaluation outputs: mels, audio, manifests, summaries.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class EvalWriter:
    """
    Write evaluation artifacts to disk.

    Structure:
        eval_dir/
        ├── manifest.json      # Eval configuration
        ├── eval_set.jsonl     # Selected samples (for reproducibility)
        ├── metrics.json       # Aggregate metrics
        ├── per_sample.jsonl   # Per-sample metrics
        ├── mels/              # Mel spectrograms
        │   ├── {uid}_pred.npy
        │   └── {uid}_tgt.npy
        └── audio/             # Vocoded audio
            ├── {uid}_pred.wav
            └── {uid}_tgt.wav
    """

    def __init__(
        self,
        eval_dir: Path,
        write_mels: bool = True,
        write_audio: bool = False,
        write_target_audio: bool = False,
    ):
        """
        Args:
            eval_dir: Output directory for eval artifacts
            write_mels: Write predicted/target mel .npy files
            write_audio: Write predicted audio .wav files
            write_target_audio: Write target audio .wav files (for A/B comparison)
        """
        self.eval_dir = Path(eval_dir)
        self.write_mels = write_mels
        self.write_audio = write_audio
        self.write_target_audio = write_target_audio

        # Create directories
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        if write_mels:
            (self.eval_dir / "mels").mkdir(exist_ok=True)
        if write_audio or write_target_audio:
            (self.eval_dir / "audio").mkdir(exist_ok=True)

        # Track samples
        self._samples = []
        self._per_sample_metrics = []

    def write_manifest(
        self,
        run_id: str,
        checkpoint_path: str,
        config: dict,
        n_samples: int,
        seed: int,
        extra_info: dict | None = None,
    ) -> None:
        """Write eval manifest (configuration record)."""
        manifest = {
            "run_id": run_id,
            "checkpoint": checkpoint_path,
            "config": config,
            "n_samples": n_samples,
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "write_mels": self.write_mels,
            "write_audio": self.write_audio,
            "write_target_audio": self.write_target_audio,
        }

        if extra_info:
            manifest.update(extra_info)

        with open(self.eval_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def write_eval_set(self, samples: list[dict]) -> None:
        """
        Write the selected eval samples for reproducibility.

        This is the source of truth for what was evaluated.
        """
        with open(self.eval_dir / "eval_set.jsonl", "w") as f:
            for sample in samples:
                # Write minimal info needed to reproduce
                row = {
                    "utterance_id": sample["utterance_id"],
                    "phonemes": sample.get("phonemes", ""),
                    "duration_sec": sample.get("duration_sec", 0),
                    "audio_path": sample.get("audio_path", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._samples = samples

    def write_sample(
        self,
        utterance_id: str,
        pred_mel: torch.Tensor,
        target_mel: Optional[torch.Tensor] = None,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        sample_rate: int = 22050,
        metrics: Optional[dict] = None,
    ) -> None:
        """
        Write artifacts for a single sample.

        Args:
            utterance_id: Sample identifier
            pred_mel: Predicted mel [n_mels, T]
            target_mel: Target mel [n_mels, T] (optional)
            pred_audio: Predicted audio [T] (optional)
            target_audio: Target audio [T] (optional)
            sample_rate: Audio sample rate
            metrics: Per-sample metrics dict
        """
        # Write mels
        if self.write_mels:
            mel_dir = self.eval_dir / "mels"
            np.save(mel_dir / f"{utterance_id}_pred.npy", pred_mel.cpu().numpy())
            if target_mel is not None:
                np.save(mel_dir / f"{utterance_id}_tgt.npy", target_mel.cpu().numpy())

        # Write audio
        if self.write_audio and pred_audio is not None:
            self._save_audio(
                self.eval_dir / "audio" / f"{utterance_id}_pred.wav",
                pred_audio,
                sample_rate,
            )

        if self.write_target_audio and target_audio is not None:
            self._save_audio(
                self.eval_dir / "audio" / f"{utterance_id}_tgt.wav",
                target_audio,
                sample_rate,
            )

        # Track metrics
        if metrics:
            self._per_sample_metrics.append({
                "utterance_id": utterance_id,
                **metrics,
            })

    def _save_audio(
        self,
        path: Path,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> None:
        """Save audio file with normalization."""
        import torchaudio

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Normalize to prevent clipping
        waveform = waveform / (waveform.abs().max() + 1e-8)

        torchaudio.save(str(path), waveform.cpu(), sample_rate)

    def write_per_sample_metrics(self) -> None:
        """Write per-sample metrics to JSONL."""
        if not self._per_sample_metrics:
            return

        with open(self.eval_dir / "per_sample.jsonl", "w") as f:
            for m in self._per_sample_metrics:
                f.write(json.dumps(m) + "\n")

    def write_aggregate_metrics(self, metrics: dict) -> None:
        """Write aggregate metrics summary."""
        with open(self.eval_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def get_per_sample_metrics(self) -> list[dict]:
        """Get collected per-sample metrics."""
        return self._per_sample_metrics
