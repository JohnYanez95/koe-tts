"""Atomic cache extraction manager.

Downloads archives from a ``StorageBackend``, extracts via ``TarHandler``
or ``ZipHandler``, writes a ``CacheManifest``, and atomically renames to
the final cache path.  Partial extractions are cleaned up on failure.

Flow (``pull``)::

    storage.get_to_path(key, tmp_archive)
    → detect archive type (.tar.gz / .zip)
    → extract to .tmp/<uuid> under cache_root
    → compute SHA-256 of archive, collect file list + sizes
    → write manifest.json
    → os.rename(.tmp/<uuid>, cache_root/dest_name)
    → clean up tmp_archive

Public API
----------
- ``CacheError`` — base exception for cache operations
- ``CacheManager`` — atomic download + extract + manifest + rename
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from modules.forge.archive.tar import TarHandler
from modules.forge.archive.zip import ZipHandler
from modules.forge.cache.manifest import CacheManifest, ManifestEntry
from modules.forge.storage.protocols import StorageBackend
from modules.forge.storage.s3 import validate_path_component

logger = logging.getLogger(__name__)

_HASH_BUFSIZE: int = 262_144  # 256 KB
_FORGE_VERSION: str = "0.1.0"


class CacheError(Exception):
    """Base error for cache operations."""


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_BUFSIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _collect_files(root: Path) -> list[ManifestEntry]:
    """Walk a directory and collect relative paths + sizes."""
    entries: list[ManifestEntry] = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            rel = str(item.relative_to(root))
            entries.append(ManifestEntry(path=rel, size=item.stat().st_size))
    return entries


def _detect_handler(archive_path: Path) -> TarHandler | ZipHandler:
    """Pick the right archive handler based on file extension."""
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        return TarHandler()
    if name.endswith(".zip"):
        return ZipHandler()
    raise CacheError(
        f"Unsupported archive format: {archive_path.name!r}. "
        "Expected .tar.gz, .tgz, .tar.bz2, .tar.xz, .tar, or .zip"
    )


class CacheManager:
    """Atomic cache extraction manager.

    Downloads archives from storage, extracts them safely, and manages
    the local cache directory with manifest-based integrity tracking.

    Parameters
    ----------
    storage
        Storage backend to download archives from.
    cache_root
        Local directory for cached extractions.
    """

    def __init__(self, storage: StorageBackend, cache_root: Path) -> None:
        self.storage = storage
        self.cache_root = Path(cache_root)

    def pull(self, key: str, dest_name: str) -> Path:
        """Download and atomically extract an archive to the cache.

        Parameters
        ----------
        key
            Storage key for the archive (e.g. ``raw/corpora/jsut.tar.gz``).
        dest_name
            Name of the cache directory (validated as a safe path component).

        Returns
        -------
        Path
            Path to the extracted cache directory.

        Raises
        ------
        CacheError
            If download, extraction, or rename fails.
        """
        validate_path_component(dest_name, "dest_name")

        final_dest = self.cache_root / dest_name
        if final_dest.exists():
            logger.info("Cache already exists: %s", final_dest)
            return final_dest

        # Create temp working area
        tmp_dir = self.cache_root / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        staging_id = str(uuid.uuid4())
        staging_dir = tmp_dir / staging_id

        # Infer archive extension from key for temp file naming
        key_suffix = Path(key).suffix
        if key.endswith(".tar.gz"):
            key_suffix = ".tar.gz"
        elif key.endswith(".tar.bz2"):
            key_suffix = ".tar.bz2"
        elif key.endswith(".tar.xz"):
            key_suffix = ".tar.xz"
        tmp_archive = tmp_dir / f"{staging_id}{key_suffix}"

        try:
            # 1. Download
            logger.info("Downloading %s → %s", key, tmp_archive)
            self.storage.get_to_path(key, tmp_archive)

            # 2. Detect handler and extract
            handler = _detect_handler(tmp_archive)
            logger.info("Extracting %s → %s", tmp_archive.name, staging_dir)
            handler.extract(tmp_archive, staging_dir)

            # 3. Compute archive checksum
            archive_sha256 = _sha256_file(tmp_archive)

            # 4. Collect file list and sizes
            files = _collect_files(staging_dir)
            total_size = sum(f.size for f in files)

            # 5. Write manifest
            manifest = CacheManifest(
                archive_sha256=archive_sha256,
                files=files,
                total_size=total_size,
                extracted_at=datetime.now(UTC).isoformat(),
                forge_version=_FORGE_VERSION,
            )
            manifest.save(staging_dir / "manifest.json")

            # 6. Atomic rename to final path
            self.cache_root.mkdir(parents=True, exist_ok=True)
            os.rename(staging_dir, final_dest)
            logger.info("Cache ready: %s (%d files, %d bytes)", final_dest, len(files), total_size)

            return final_dest

        except Exception as e:
            # Clean up staging dir on any failure
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if not isinstance(e, CacheError):
                raise CacheError(f"Cache pull failed for {key!r}: {e}") from e
            raise
        finally:
            # Always clean up temp archive
            tmp_archive.unlink(missing_ok=True)

    def status(self, name: str) -> dict | None:
        """Read manifest for a cached directory.

        Parameters
        ----------
        name
            Cache directory name.

        Returns
        -------
        dict | None
            Manifest as dict if cached, None otherwise.
        """
        validate_path_component(name, "name")
        manifest_path = self.cache_root / name / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = CacheManifest.load(manifest_path)
        return manifest.to_dict()

    def invalidate(self, name: str) -> None:
        """Remove a cached directory.

        Parameters
        ----------
        name
            Cache directory name (validated to prevent traversal).

        Raises
        ------
        CacheError
            If the cache directory does not exist.
        """
        validate_path_component(name, "name")
        target = self.cache_root / name
        if not target.exists():
            raise CacheError(f"Cache not found: {name!r}")
        # Double-check the resolved path is under cache_root
        resolved = target.resolve()
        if not resolved.is_relative_to(self.cache_root.resolve()):
            raise CacheError(f"Invalid cache path: {name!r}")
        shutil.rmtree(resolved)
        logger.info("Invalidated cache: %s", name)

    def list_cached(self) -> list[str]:
        """List all cached directories (those with a manifest.json).

        Returns
        -------
        list[str]
            Sorted list of cache directory names.
        """
        if not self.cache_root.exists():
            return []
        cached = []
        for entry in sorted(self.cache_root.iterdir()):
            if entry.is_dir() and (entry / "manifest.json").exists():
                cached.append(entry.name)
        return cached
