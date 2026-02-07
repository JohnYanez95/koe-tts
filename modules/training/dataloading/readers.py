"""
Dataset readers for training.

Reads gold manifest (parquet/jsonl) and provides iterators for training.
"""

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


class GoldManifestReader:
    """
    Read gold training manifest for model training.

    Supports both Parquet and JSONL formats.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        cache_snapshot_path: Optional[Path] = None,
    ):
        """
        Initialize manifest reader.

        Args:
            manifest_path: Path to manifest file (parquet or jsonl)
            split: Data split to read (train, val, test)
            cache_snapshot_path: Path to local cache (if using cached audio)
        """
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.cache_snapshot_path = cache_snapshot_path
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load manifest into memory."""
        if self._df is not None:
            return self._df

        if self.manifest_path.suffix == ".parquet":
            self._df = pd.read_parquet(self.manifest_path)
        elif self.manifest_path.suffix == ".jsonl":
            self._df = pd.read_json(self.manifest_path, lines=True)
        else:
            raise ValueError(f"Unsupported format: {self.manifest_path.suffix}")

        # Filter by split
        if "split" in self._df.columns:
            self._df = self._df[self._df["split"] == self.split]

        return self._df

    def __len__(self) -> int:
        """Return number of utterances."""
        return len(self.load())

    def __iter__(self) -> Iterator[dict]:
        """Iterate over utterances."""
        df = self.load()
        for _, row in df.iterrows():
            item = row.to_dict()

            # Resolve audio path from cache if available
            if self.cache_snapshot_path:
                item["audio_path"] = str(
                    self.cache_snapshot_path / Path(item["audio_path"]).name
                )

            yield item

    def get_by_id(self, utterance_id: str) -> Optional[dict]:
        """Get a specific utterance by ID."""
        df = self.load()
        matches = df[df["utterance_id"] == utterance_id]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()
