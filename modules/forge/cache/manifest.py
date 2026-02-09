"""Cache manifest dataclasses.

Tracks what was extracted, when, and with what checksums.  Used by
``CacheManager`` to verify cache integrity and freshness.

Public API
----------
- ``ManifestEntry`` — single file in the cache (path + size)
- ``CacheManifest`` — full manifest with SHA-256, file list, timestamps
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    """A single file in the cache."""

    path: str
    size: int


@dataclass
class CacheManifest:
    """Manifest describing the contents of a cache directory.

    Attributes
    ----------
    archive_sha256
        SHA-256 hex digest of the source archive.
    files
        List of extracted files with relative paths and sizes.
    total_size
        Total size of all extracted files in bytes.
    extracted_at
        ISO 8601 timestamp of extraction.
    forge_version
        Version of forge that created this cache.
    """

    archive_sha256: str
    files: list[ManifestEntry] = field(default_factory=list)
    total_size: int = 0
    extracted_at: str = ""
    forge_version: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CacheManifest:
        """Deserialize from a dict (e.g. loaded from JSON)."""
        files = [ManifestEntry(**entry) for entry in data.get("files", [])]
        return cls(
            archive_sha256=data["archive_sha256"],
            files=files,
            total_size=data.get("total_size", 0),
            extracted_at=data.get("extracted_at", ""),
            forge_version=data.get("forge_version", ""),
        )

    def save(self, path: Path) -> None:
        """Write manifest as JSON to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CacheManifest:
        """Load manifest from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
