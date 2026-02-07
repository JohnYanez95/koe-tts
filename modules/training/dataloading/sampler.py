"""
Samplers for multi-speaker TTS training.

Key insight: speaker-balanced sampling is critical for multi-speaker training
because dataset sizes vary dramatically (e.g., JSUT has 7670 utterances from
one speaker, while each JVS speaker has ~150 utterances).
"""

import random
from collections import defaultdict
from typing import Iterator, Optional

from torch.utils.data import Sampler


class SpeakerBalancedBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that balances speakers within each batch.

    Sampling strategy (per batch):
    1. Choose a speaker uniformly at random
    2. Choose an utterance uniformly from that speaker
    3. Repeat until batch is filled

    This ensures each speaker contributes equally per step, regardless of
    how many utterances they have.

    Usage:
        sampler = SpeakerBalancedBatchSampler(
            speaker_to_indices={"spk00": [0, 1, 2], "jvs001": [3, 4]},
            batch_size=16,
            drop_last=True,
        )
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        speaker_to_indices: dict[str, list[int]],
        batch_size: int,
        drop_last: bool = True,
        seed: Optional[int] = None,
        speaker_weights: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            speaker_to_indices: Mapping from speaker_id to list of dataset indices
            batch_size: Number of samples per batch
            drop_last: Drop last incomplete batch
            seed: Random seed for reproducibility
            speaker_weights: Optional per-speaker sampling weights (default: uniform)
        """
        self.speaker_to_indices = speaker_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.speaker_weights = speaker_weights

        self.speakers = list(speaker_to_indices.keys())
        self.total_samples = sum(len(indices) for indices in speaker_to_indices.values())

        # Compute number of batches
        if drop_last:
            self.num_batches = self.total_samples // batch_size
        else:
            self.num_batches = (self.total_samples + batch_size - 1) // batch_size

        # Build weight array for weighted sampling
        if speaker_weights:
            self._weights = [speaker_weights.get(s, 1.0) for s in self.speakers]
        else:
            self._weights = [1.0] * len(self.speakers)

        # Normalize weights
        total_weight = sum(self._weights)
        self._weights = [w / total_weight for w in self._weights]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices."""
        rng = random.Random(self.seed)

        # Track available indices per speaker (copy to allow in-epoch exhaustion)
        available = {
            speaker: list(indices)
            for speaker, indices in self.speaker_to_indices.items()
        }

        # Shuffle within each speaker
        for indices in available.values():
            rng.shuffle(indices)

        for batch_idx in range(self.num_batches):
            batch = []

            while len(batch) < self.batch_size:
                # Choose speaker (weighted random)
                speaker = rng.choices(self.speakers, weights=self._weights, k=1)[0]

                # Get available indices for this speaker
                speaker_indices = available[speaker]

                if not speaker_indices:
                    # Refill if exhausted
                    speaker_indices = list(self.speaker_to_indices[speaker])
                    rng.shuffle(speaker_indices)
                    available[speaker] = speaker_indices

                # Pop one sample
                idx = speaker_indices.pop()
                batch.append(idx)

            yield batch

    @classmethod
    def from_dataset(
        cls,
        items: list[dict],
        batch_size: int,
        drop_last: bool = True,
        seed: Optional[int] = None,
        speaker_weights: Optional[dict[str, float]] = None,
    ) -> "SpeakerBalancedBatchSampler":
        """
        Create sampler from dataset items.

        Args:
            items: List of dicts with 'speaker_id' key
            batch_size: Batch size
            drop_last: Drop last incomplete batch
            seed: Random seed
            speaker_weights: Optional speaker weights

        Returns:
            SpeakerBalancedBatchSampler
        """
        speaker_to_indices: dict[str, list[int]] = defaultdict(list)

        for idx, item in enumerate(items):
            speaker_id = item.get("speaker_id", "unknown")
            speaker_to_indices[speaker_id].append(idx)

        return cls(
            speaker_to_indices=dict(speaker_to_indices),
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            speaker_weights=speaker_weights,
        )


def build_speaker_index(items: list[dict]) -> dict[str, list[int]]:
    """
    Build index mapping speaker_id to list of item indices.

    Args:
        items: List of dicts with 'speaker_id' key

    Returns:
        Dict mapping speaker_id to list of indices
    """
    index: dict[str, list[int]] = defaultdict(list)

    for idx, item in enumerate(items):
        speaker_id = item.get("speaker_id", "unknown")
        index[speaker_id].append(idx)

    return dict(index)
