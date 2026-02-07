"""
Collation utilities for TTS training.

Handles padding and mask creation for variable-length sequences.
Reusable across baseline, VITS, and other models.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSBatch:
    """
    Batched TTS data with padding masks.

    All tensors are padded to the max length in the batch.
    Masks indicate valid (non-padded) positions.

    Unlabeled segments (Tier 1):
        When is_labeled is False for any sample in batch, phonemes/durations are None.
        Use is_labeled and n_labeled to check batch status.
    """
    # Phoneme input (None if batch contains unlabeled samples)
    phonemes: Optional[torch.Tensor] = None  # [B, max_phone_len] - token IDs
    phoneme_lens: Optional[torch.Tensor] = None  # [B] - actual lengths
    phoneme_mask: Optional[torch.Tensor] = None  # [B, max_phone_len] - True for valid

    # Duration targets (None if batch contains unlabeled samples)
    durations: Optional[torch.Tensor] = None  # [B, max_phone_len] - frames per token

    # Mel target
    mels: torch.Tensor = None  # [B, n_mels, max_mel_frames]
    mel_lens: torch.Tensor = None  # [B] - actual frame counts
    mel_mask: torch.Tensor = None  # [B, max_mel_frames] - True for valid frames

    # Raw audio (for GAN training)
    audio: Optional[torch.Tensor] = None  # [B, max_audio_len] - raw waveform
    audio_lens: Optional[torch.Tensor] = None  # [B] - actual audio lengths

    # Speaker info (for multi-speaker)
    speaker_ids: Optional[list[str]] = None  # [B] - speaker ID strings
    speaker_idxs: Optional[torch.Tensor] = None  # [B] - speaker indices for embedding

    # Label status (for segment training)
    is_labeled: Optional[torch.Tensor] = None  # [B] - True if sample has text labels
    n_labeled: int = 0  # Count of labeled samples in batch

    # Metadata (for logging/debugging)
    utterance_ids: list[str] = None
    durations_sec: torch.Tensor = None  # [B] - original audio durations

    def to(self, device: str) -> "TTSBatch":
        """Move all tensors to device."""
        return TTSBatch(
            phonemes=self.phonemes.to(device) if self.phonemes is not None else None,
            phoneme_lens=self.phoneme_lens.to(device) if self.phoneme_lens is not None else None,
            phoneme_mask=self.phoneme_mask.to(device) if self.phoneme_mask is not None else None,
            durations=self.durations.to(device) if self.durations is not None else None,
            mels=self.mels.to(device),
            mel_lens=self.mel_lens.to(device),
            mel_mask=self.mel_mask.to(device),
            audio=self.audio.to(device) if self.audio is not None else None,
            audio_lens=self.audio_lens.to(device) if self.audio_lens is not None else None,
            speaker_ids=self.speaker_ids,
            speaker_idxs=self.speaker_idxs.to(device) if self.speaker_idxs is not None else None,
            is_labeled=self.is_labeled.to(device) if self.is_labeled is not None else None,
            n_labeled=self.n_labeled,
            utterance_ids=self.utterance_ids,
            durations_sec=self.durations_sec.to(device) if self.durations_sec is not None else None,
        )

    def __len__(self) -> int:
        return len(self.utterance_ids) if self.utterance_ids else 0

    @property
    def has_text(self) -> bool:
        """Check if batch has text labels (phonemes/durations)."""
        return self.phonemes is not None


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create padding mask from lengths.

    Args:
        lengths: [B] tensor of actual lengths
        max_len: Maximum length (default: max of lengths)

    Returns:
        [B, max_len] boolean tensor, True for valid positions
    """
    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)

    return mask


def pad_sequence_2d(
    sequences: list[torch.Tensor],
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of 2D tensors (e.g., mel spectrograms).

    Args:
        sequences: List of [C, T_i] tensors
        padding_value: Value to pad with

    Returns:
        Tuple of (padded [B, C, max_T], lengths [B])
    """
    # Get dimensions
    n_channels = sequences[0].shape[0]
    lengths = torch.tensor([s.shape[1] for s in sequences])
    max_len = lengths.max().item()
    batch_size = len(sequences)

    # Create padded tensor
    padded = torch.full(
        (batch_size, n_channels, max_len),
        padding_value,
        dtype=sequences[0].dtype,
    )

    for i, seq in enumerate(sequences):
        padded[i, :, :seq.shape[1]] = seq

    return padded, lengths


def pad_sequence_1d(
    sequences: list[torch.Tensor],
    padding_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of 1D tensors (e.g., phoneme IDs).

    Args:
        sequences: List of [T_i] tensors
        padding_value: Value to pad with

    Returns:
        Tuple of (padded [B, max_T], lengths [B])
    """
    lengths = torch.tensor([len(s) for s in sequences])
    max_len = lengths.max().item()
    batch_size = len(sequences)

    padded = torch.full(
        (batch_size, max_len),
        padding_value,
        dtype=sequences[0].dtype,
    )

    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    return padded, lengths


class TTSCollator:
    """
    Collate function for TTS batches.

    Handles padding and mask creation for variable-length audio/phonemes.
    """

    def __init__(self, pad_id: int = 0):
        """
        Args:
            pad_id: Padding token ID for phonemes
        """
        self.pad_id = pad_id

    def __call__(self, samples: list[dict]) -> Optional[TTSBatch]:
        """
        Collate samples into a batch.

        Each sample dict should have:
            - phoneme_ids: torch.Tensor [T_phone] or None (unlabeled)
            - durations: torch.Tensor [T_phone] or None (unlabeled)
            - mel: torch.Tensor [n_mels, T_mel]
            - audio: torch.Tensor [T_audio] (optional, for GAN training)
            - utterance_id: str
            - duration_sec: float
            - speaker_id: str (optional, for multi-speaker)
            - speaker_idx: int (optional, for multi-speaker)
            - is_labeled: bool (optional, default True for backwards compat)

        Unlabeled handling (strict mode):
            If ANY sample is unlabeled, treat entire batch as unlabeled.
            phonemes/durations will be None in the returned batch.

        Returns:
            TTSBatch or None if batch is empty
        """
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        if not samples:
            return None

        # Determine label status for each sample
        # Default to True for backwards compatibility with full-utterance manifests
        is_labeled_list = [s.get("is_labeled", True) for s in samples]
        is_labeled = torch.tensor(is_labeled_list, dtype=torch.bool)
        n_labeled = is_labeled.sum().item()

        # Extract metadata (always present)
        mel_seqs = [s["mel"] for s in samples]
        utterance_ids = [s["utterance_id"] for s in samples]
        durations_sec = [s["duration_sec"] for s in samples]

        # Pad mels (always present)
        mels, mel_lens = pad_sequence_2d(mel_seqs, padding_value=0.0)
        mel_mask = make_pad_mask(mel_lens)

        # Pad audio if present (for GAN training)
        audio = None
        audio_lens = None
        if samples[0].get("audio") is not None:
            audio_seqs = [s["audio"] for s in samples]
            audio, audio_lens = pad_sequence_1d(audio_seqs, padding_value=0)
            audio = audio.float()  # Ensure float dtype

        # Extract speaker info if present
        speaker_ids = None
        speaker_idxs = None
        if samples[0].get("speaker_idx") is not None:
            speaker_ids = [s.get("speaker_id", "unknown") for s in samples]
            speaker_idxs = torch.tensor([s["speaker_idx"] for s in samples], dtype=torch.long)

        # Handle phonemes/durations based on label status
        # Strict mode: if ANY sample is unlabeled, treat entire batch as unlabeled
        phonemes = None
        phoneme_lens = None
        phoneme_mask = None
        durations = None

        if n_labeled == len(samples):
            # All samples are labeled - pad phonemes and durations
            phoneme_seqs = [s["phoneme_ids"] for s in samples]
            duration_seqs = [s["durations"] for s in samples]

            phonemes, phoneme_lens = pad_sequence_1d(phoneme_seqs, padding_value=self.pad_id)
            phoneme_mask = make_pad_mask(phoneme_lens)
            durations, _ = pad_sequence_1d(duration_seqs, padding_value=0)

        return TTSBatch(
            phonemes=phonemes,
            phoneme_lens=phoneme_lens,
            phoneme_mask=phoneme_mask,
            durations=durations,
            mels=mels,
            mel_lens=mel_lens,
            mel_mask=mel_mask,
            audio=audio,
            audio_lens=audio_lens,
            speaker_ids=speaker_ids,
            speaker_idxs=speaker_idxs,
            is_labeled=is_labeled,
            n_labeled=n_labeled,
            utterance_ids=utterance_ids,
            durations_sec=torch.tensor(durations_sec),
        )
