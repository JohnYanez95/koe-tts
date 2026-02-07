"""
Speaker vocabulary and utilities for multi-speaker TTS.

Speaker IDs are strings in the manifest (e.g., "spk00", "jvs001").
This module provides mapping to integer indices for embedding lookup.
"""

import json
from pathlib import Path
from typing import Optional


class SpeakerVocab:
    """
    Speaker vocabulary mapping string IDs to integer indices.

    Supports building from manifest or loading from saved file.
    """

    def __init__(self):
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._id_to_idx)

    def __contains__(self, speaker_id: str) -> bool:
        return speaker_id in self._id_to_idx

    def add(self, speaker_id: str) -> int:
        """Add speaker to vocab, return index."""
        if speaker_id not in self._id_to_idx:
            idx = len(self._id_to_idx)
            self._id_to_idx[speaker_id] = idx
            self._idx_to_id[idx] = speaker_id
        return self._id_to_idx[speaker_id]

    def get_idx(self, speaker_id: str) -> int:
        """Get index for speaker ID. Raises KeyError if not found."""
        return self._id_to_idx[speaker_id]

    def get_id(self, idx: int) -> str:
        """Get speaker ID for index. Raises KeyError if not found."""
        return self._idx_to_id[idx]

    def get_idx_safe(self, speaker_id: str, default: int = 0) -> int:
        """Get index for speaker ID, returning default if not found."""
        return self._id_to_idx.get(speaker_id, default)

    @property
    def speakers(self) -> list[str]:
        """Get list of speaker IDs in index order."""
        return [self._idx_to_id[i] for i in range(len(self))]

    def save(self, path: Path) -> None:
        """Save vocab to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "speakers": self.speakers,
                "version": 1,
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SpeakerVocab":
        """Load vocab from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        for speaker_id in data["speakers"]:
            vocab.add(speaker_id)
        return vocab

    @classmethod
    def from_manifest(cls, manifest_path: Path, split: Optional[str] = None) -> "SpeakerVocab":
        """
        Build vocab from cache manifest.

        Args:
            manifest_path: Path to manifest.jsonl
            split: Optional split filter (train, val)

        Returns:
            SpeakerVocab with all speakers found
        """
        vocab = cls()
        seen = set()

        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                # Filter by split if specified
                if split and item.get("split") != split:
                    continue

                speaker_id = item.get("speaker_id")
                if speaker_id and speaker_id not in seen:
                    vocab.add(speaker_id)
                    seen.add(speaker_id)

        return vocab


def build_speaker_vocab_from_cache(cache_dir: Path, split: Optional[str] = None) -> SpeakerVocab:
    """
    Build speaker vocab from cache directory.

    Args:
        cache_dir: Path to cache snapshot directory
        split: Optional split filter

    Returns:
        SpeakerVocab
    """
    manifest_path = Path(cache_dir) / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return SpeakerVocab.from_manifest(manifest_path, split=split)
