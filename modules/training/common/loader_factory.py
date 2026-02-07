"""
Unified dataloader factory for all training pipelines.

Provides consistent memory-safe behavior (segment cropping, speaker balancing)
across baseline, duration, and VITS training.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from modules.training.audio import MelConfig
from modules.training.dataloading import (
    TTSDataset,
    TTSCollator,
    PHONEME_VOCAB,
    SpeakerBalancedBatchSampler,
)
from modules.training.dataloading.speaker import SpeakerVocab, build_speaker_vocab_from_cache


def create_train_val_loaders(
    cache_dir: Path,
    mel_config: Optional[MelConfig] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    segment_seconds: Optional[float] = None,
    max_audio_len: Optional[int] = None,
    speaker_balanced: Optional[bool] = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, SpeakerVocab, int]:
    """
    Create train and validation dataloaders with unified memory-safe behavior.

    This is the single source of truth for dataloader creation across all
    training pipelines (baseline, duration, VITS core, VITS GAN).

    Args:
        cache_dir: Path to cache snapshot directory
        mel_config: Mel extraction config (uses default if None)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        segment_seconds: If set, randomly crop audio to this duration (seconds).
                        Bounds GPU memory. Recommended: 3.0 for multi-speaker.
                        None = use full utterance (may OOM on long sequences).
        max_audio_len: If set, truncate audio longer than this (in samples).
                      Use to cap utterance length and prevent OOM.
                      Example: 220500 = 10 seconds @ 22050 Hz.
        speaker_balanced: Whether to balance speakers in each batch.
                         None = auto (enabled if num_speakers > 1)
                         True = always enable
                         False = never enable (standard shuffle)
        seed: Random seed for reproducibility
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, speaker_vocab, num_speakers)

    Example:
        train_loader, val_loader, speaker_vocab, num_speakers = create_train_val_loaders(
            cache_dir=Path("data/cache/multi/snapshot-xxx"),
            batch_size=8,
            segment_seconds=3.0,
        )
    """
    cache_dir = Path(cache_dir)
    mel_config = mel_config or MelConfig()

    # Compute segment_samples from segment_seconds
    segment_samples = None
    if segment_seconds is not None:
        segment_samples = int(segment_seconds * mel_config.sample_rate)

    # Build speaker vocabulary from full manifest (all splits)
    speaker_vocab = build_speaker_vocab_from_cache(cache_dir, split=None)
    num_speakers = len(speaker_vocab)

    # Create train dataset with segment cropping and max length
    train_dataset = TTSDataset(
        cache_dir=cache_dir,
        split="train",
        mel_config=mel_config,
        max_audio_len=max_audio_len,
        segment_samples=segment_samples,
        speaker_vocab=speaker_vocab,
    )

    # Create val dataset WITHOUT segment cropping (evaluate on full utterances)
    # But still respect max_audio_len to prevent OOM during validation
    val_dataset = TTSDataset(
        cache_dir=cache_dir,
        split="val",
        mel_config=mel_config,
        max_audio_len=max_audio_len,
        segment_samples=None,  # Always full utterance for validation
        speaker_vocab=speaker_vocab,
    )

    # Determine speaker balancing strategy
    use_speaker_balanced = speaker_balanced
    if use_speaker_balanced is None:
        # Auto: enable if multi-speaker
        use_speaker_balanced = num_speakers > 1

    # Collator for padding
    collator = TTSCollator(pad_id=PHONEME_VOCAB["<pad>"])

    # Create train dataloader
    if use_speaker_balanced and num_speakers > 1:
        # Speaker-balanced sampling for multi-speaker training
        train_sampler = SpeakerBalancedBatchSampler.from_dataset(
            items=train_dataset.items,
            batch_size=batch_size,
            drop_last=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
        )
    else:
        # Standard shuffle for single-speaker
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            drop_last=True,
        )

    # Validation loader (no shuffle, no drop_last)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, speaker_vocab, num_speakers
