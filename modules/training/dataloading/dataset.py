"""
TTS Dataset for training.

Loads from cache snapshot manifest, extracts mel on-the-fly (v0).
Future: optional precomputed mels for speed.

Multi-speaker support:
- Each sample includes speaker_id (string) and speaker_idx (int)
- Use SpeakerVocab to map string IDs to indices
- Use SpeakerBalancedBatchSampler for balanced training

Segment training:
- Use segment_samples to randomly crop fixed-length audio segments
- This bounds GPU memory and is the standard approach for VITS/HiFiGAN
- Recommended: segment_samples = 3.0 * sample_rate (3 seconds)
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from modules.data_engineering.common.phonemes import CANONICAL_INVENTORY, tokenize
from modules.training.audio import MelConfig, MelExtractor, load_audio
from .speaker import SpeakerVocab, build_speaker_vocab_from_cache


def build_phoneme_vocab() -> dict[str, int]:
    """
    Build phoneme to token ID mapping.

    Uses canonical inventory from data engineering.
    """
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }

    for phone in sorted(CANONICAL_INVENTORY):
        vocab[phone] = len(vocab)

    return vocab


# Singleton vocab
PHONEME_VOCAB = build_phoneme_vocab()
PHONEME_VOCAB_SIZE = len(PHONEME_VOCAB)


def phonemes_to_ids(phoneme_str: Optional[str]) -> torch.Tensor:
    """
    Convert phoneme string to token IDs.

    Args:
        phoneme_str: Space-separated phonemes

    Returns:
        Tensor of token IDs with BOS/EOS
    """
    if not phoneme_str:
        return torch.tensor([PHONEME_VOCAB["<bos>"], PHONEME_VOCAB["<eos>"]], dtype=torch.long)

    tokens = tokenize(phoneme_str)
    ids = [PHONEME_VOCAB["<bos>"]]
    for t in tokens:
        ids.append(PHONEME_VOCAB.get(t, PHONEME_VOCAB["<unk>"]))
    ids.append(PHONEME_VOCAB["<eos>"])

    return torch.tensor(ids, dtype=torch.long)


def compute_uniform_durations(n_tokens: int, n_frames: int) -> torch.Tensor:
    """
    Compute uniform duration targets for duration predictor bootstrap.

    Distributes mel frames evenly across tokens, with remainder
    distributed to first r tokens.

    Args:
        n_tokens: Number of phoneme tokens (including BOS/EOS)
        n_frames: Number of mel frames

    Returns:
        Tensor of integer durations [n_tokens] summing to n_frames
    """
    if n_tokens == 0:
        return torch.tensor([], dtype=torch.long)

    base = n_frames // n_tokens
    remainder = n_frames % n_tokens

    durations = torch.full((n_tokens,), base, dtype=torch.long)
    durations[:remainder] += 1

    return durations


class TTSDataset(Dataset):
    """
    TTS Dataset that reads from cache snapshot manifest.

    Loads audio and computes mel on-the-fly.
    Supports multi-speaker training with speaker vocabulary.

    Attributes:
        speaker_vocab: SpeakerVocab mapping speaker_id strings to indices
        num_speakers: Number of unique speakers
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        mel_config: Optional[MelConfig] = None,
        max_audio_len: Optional[int] = None,
        max_mel_len: Optional[int] = None,
        segment_samples: Optional[int] = None,
        speaker_vocab: Optional[SpeakerVocab] = None,
    ):
        """
        Args:
            cache_dir: Path to cache snapshot directory
            split: Data split (train, val)
            mel_config: Mel extraction config
            max_audio_len: Max audio samples (truncate longer, applied before segment crop)
            max_mel_len: Max mel frames (truncate longer)
            segment_samples: If set, randomly crop audio to this length (samples).
                            This bounds GPU memory. Recommended: 3.0 * sample_rate.
                            If audio is shorter, it's kept as-is and padded in collator.
            speaker_vocab: Pre-built speaker vocabulary (built from manifest if None)
        """
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.mel_config = mel_config or MelConfig()
        self.max_audio_len = max_audio_len
        self.max_mel_len = max_mel_len
        self.segment_samples = segment_samples

        # Mel extractor (shared across samples)
        self.mel_extractor = MelExtractor(self.mel_config)

        # Load manifest
        manifest_path = self.cache_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.items = []
        with open(manifest_path) as f:
            for line in f:
                item = json.loads(line)
                if item.get("split") == split:
                    self.items.append(item)

        if not self.items:
            raise ValueError(f"No items found for split '{split}' in {manifest_path}")

        # Build or use provided speaker vocabulary
        if speaker_vocab is not None:
            self.speaker_vocab = speaker_vocab
        else:
            # Build from all splits to ensure consistent vocab
            self.speaker_vocab = build_speaker_vocab_from_cache(cache_dir, split=None)

        self.num_speakers = len(self.speaker_vocab)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[dict]:
        """
        Load a single sample.

        Returns:
            Dict with phoneme_ids, mel, utterance_id, duration_sec
            or None if loading fails

        Supports segment manifests with start_ms/end_ms for sub-utterance training.
        Segments are sliced after resampling to ensure consistent sample rate.
        """
        item = self.items[idx]

        try:
            # Load audio at target sample rate for GAN training
            audio_path = item.get("audio_path") or item.get("audio_abspath")
            waveform, sr = load_audio(audio_path, target_sr=self.mel_config.sample_rate)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)  # [T]

            # Segment slicing (after resample, before random crop)
            # start_ms/end_ms are always interpreted at target sample rate
            start_ms = item.get("start_ms")
            end_ms = item.get("end_ms")
            if start_ms is not None and end_ms is not None:
                start_sample = int(start_ms * sr / 1000)
                end_sample = int(end_ms * sr / 1000)

                # Clamp to valid bounds (gold build should prevent this, but safety first)
                start_sample = max(0, min(start_sample, len(waveform)))
                end_sample = max(start_sample, min(end_sample, len(waveform)))

                waveform = waveform[start_sample:end_sample]

            # Hard truncate if way too long (safety guard)
            if self.max_audio_len and len(waveform) > self.max_audio_len:
                waveform = waveform[:self.max_audio_len]

            # Random segment crop for bounded GPU memory (standard VITS/HiFiGAN approach)
            if self.segment_samples and len(waveform) > self.segment_samples:
                max_start = len(waveform) - self.segment_samples
                start = random.randint(0, max_start)
                waveform = waveform[start : start + self.segment_samples]

            # Compute mel from (potentially cropped) audio
            mel = self.mel_extractor(waveform, sample_rate=sr)  # [n_mels, frames]

            # Truncate mel if too long (rarely needed with segment crop)
            if self.max_mel_len and mel.shape[1] > self.max_mel_len:
                mel = mel[:, :self.max_mel_len]

            # Check if segment is labeled (Tier 1: always False for segments)
            # Full utterances from standard gold manifests are always labeled
            segment_label_status = item.get("segment_label_status")
            is_labeled = segment_label_status != "unlabeled"

            # For unlabeled segments, we don't have text alignment
            if is_labeled:
                # Labeled segment or full utterance: use phonemes
                # For segments, prefer phonemes_span; fall back to phonemes for full utterances
                phoneme_str = item.get("phonemes_span") or item.get("phonemes", "")
                phoneme_ids = phonemes_to_ids(phoneme_str)

                # Compute uniform duration targets (bootstrap for duration predictor)
                n_tokens = len(phoneme_ids)
                n_frames = mel.shape[1]
                durations = compute_uniform_durations(n_tokens, n_frames)
            else:
                # Unlabeled segment: no phoneme alignment
                phoneme_ids = None
                durations = None

            # Get speaker info
            speaker_id = item.get("speaker_id", "unknown")
            speaker_idx = self.speaker_vocab.get_idx_safe(speaker_id, default=0)

            # Use segment_id if present, else utterance_id
            sample_id = item.get("segment_id") or item.get("utterance_id")

            return {
                "phoneme_ids": phoneme_ids,
                "mel": mel,
                "audio": waveform,  # [T] raw audio for GAN training
                "durations": durations,  # [n_tokens] uniform bootstrap or None
                "utterance_id": sample_id,
                "duration_sec": item.get("duration_sec", 0.0),
                "speaker_id": speaker_id,
                "speaker_idx": speaker_idx,
                "is_labeled": is_labeled,
            }

        except Exception as e:
            # Log and return None (filtered out by collator)
            print(f"Warning: Failed to load {item.get('utterance_id', 'unknown')}: {e}")
            return None


def create_dataloader(
    cache_dir: Path,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    mel_config: Optional[MelConfig] = None,
    max_audio_len: Optional[int] = None,
    max_mel_len: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for TTS training.

    Args:
        cache_dir: Path to cache snapshot
        split: Data split
        batch_size: Batch size
        num_workers: Number of data loading workers
        mel_config: Mel extraction config
        max_audio_len: Max audio samples
        max_mel_len: Max mel frames
        shuffle: Shuffle data (default: True for train, False for val)

    Returns:
        DataLoader with TTSCollator
    """
    from .collate import TTSCollator

    dataset = TTSDataset(
        cache_dir=cache_dir,
        split=split,
        mel_config=mel_config,
        max_audio_len=max_audio_len,
        max_mel_len=max_mel_len,
    )

    if shuffle is None:
        shuffle = (split == "train")

    collator = TTSCollator(pad_id=PHONEME_VOCAB["<pad>"])

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=(split == "train"),
    )
