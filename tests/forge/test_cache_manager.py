"""Test suite for forge.cache.manager — atomic cache extraction.

All tests mock StorageBackend — no real storage connection needed.
Archive handlers are exercised with real temp files for integration testing.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.forge.cache.manager import CacheError, CacheManager

# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _create_test_tarball(path: Path, files: dict[str, bytes]) -> None:
    """Create a .tar.gz with the given files."""
    import io

    with tarfile.open(path, "w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def _mock_storage(tmp_path: Path) -> MagicMock:
    """Create a mock StorageBackend that writes a test tarball on get_to_path."""
    storage = MagicMock()

    def fake_get_to_path(key: str, dest: Path) -> Path:
        _create_test_tarball(
            dest,
            {
                "audio/001.wav": b"fake wav data 1",
                "audio/002.wav": b"fake wav data 2",
                "metadata.json": b'{"count": 2}',
            },
        )
        return dest

    storage.get_to_path.side_effect = fake_get_to_path
    return storage


# ════════════════════════════════════════════════════════════════════════
# 1. pull — happy path
# ════════════════════════════════════════════════════════════════════════


class TestPull:
    def test_downloads_extracts_and_creates_manifest(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        result = manager.pull("raw/corpora/test.tar.gz", "test-dataset")

        assert result == cache_root / "test-dataset"
        assert result.exists()

        # Check extracted files
        assert (result / "audio" / "001.wav").exists()
        assert (result / "audio" / "002.wav").exists()
        assert (result / "metadata.json").exists()

        # Check manifest
        manifest_path = result / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["archive_sha256"]  # Non-empty
        assert len(manifest["files"]) == 3
        assert manifest["total_size"] > 0
        assert manifest["forge_version"] == "0.1.0"

    def test_skips_if_already_cached(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        # Pre-create the cache dir
        final = cache_root / "existing"
        final.mkdir(parents=True)

        result = manager.pull("raw/corpora/test.tar.gz", "existing")

        assert result == final
        storage.get_to_path.assert_not_called()

    def test_temp_archive_cleaned_up(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        manager.pull("raw/corpora/test.tar.gz", "my-dataset")

        # .tmp dir should have no archive files left
        tmp_dir = cache_root / ".tmp"
        if tmp_dir.exists():
            archives = list(tmp_dir.glob("*.tar.gz"))
            assert len(archives) == 0


# ════════════════════════════════════════════════════════════════════════
# 2. pull — failure cleanup
# ════════════════════════════════════════════════════════════════════════


class TestPullCleanup:
    def test_cleans_up_on_download_failure(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        storage.get_to_path.side_effect = RuntimeError("network error")
        manager = CacheManager(storage, cache_root)

        with pytest.raises(CacheError, match="Cache pull failed"):
            manager.pull("raw/corpora/test.tar.gz", "failing")

        # No orphaned staging dirs
        tmp_dir = cache_root / ".tmp"
        if tmp_dir.exists():
            staging_dirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
            assert len(staging_dirs) == 0

    def test_cleans_up_on_extraction_failure(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()

        def write_bad_archive(key: str, dest: Path) -> Path:
            dest.write_bytes(b"not a real archive")
            return dest

        storage.get_to_path.side_effect = write_bad_archive
        manager = CacheManager(storage, cache_root)

        with pytest.raises(CacheError):
            manager.pull("raw/corpora/test.tar.gz", "bad-archive")

        # No orphaned staging dirs
        final = cache_root / "bad-archive"
        assert not final.exists()


# ════════════════════════════════════════════════════════════════════════
# 3. pull — path validation
# ════════════════════════════════════════════════════════════════════════


class TestPullValidation:
    def test_rejects_traversal_in_dest_name(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        with pytest.raises(ValueError, match="Invalid dest_name"):
            manager.pull("key", "../escape")

    def test_rejects_absolute_dest_name(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        with pytest.raises(ValueError, match="Invalid dest_name"):
            manager.pull("key", "/etc/passwd")

    def test_rejects_slashes_in_dest_name(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        with pytest.raises(ValueError, match="Invalid dest_name"):
            manager.pull("key", "foo/bar")


# ════════════════════════════════════════════════════════════════════════
# 4. status
# ════════════════════════════════════════════════════════════════════════


class TestStatus:
    def test_returns_manifest_dict(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        # Pull first to create a cached entry
        manager.pull("raw/corpora/test.tar.gz", "my-dataset")

        status = manager.status("my-dataset")
        assert status is not None
        assert status["archive_sha256"]
        assert len(status["files"]) == 3

    def test_returns_none_for_missing(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        assert manager.status("nonexistent") is None

    def test_validates_name(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        with pytest.raises(ValueError, match="Invalid name"):
            manager.status("../escape")


# ════════════════════════════════════════════════════════════════════════
# 5. invalidate
# ════════════════════════════════════════════════════════════════════════


class TestInvalidate:
    def test_removes_cached_dir(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        manager.pull("raw/corpora/test.tar.gz", "to-delete")
        assert (cache_root / "to-delete").exists()

        manager.invalidate("to-delete")
        assert not (cache_root / "to-delete").exists()

    def test_raises_for_nonexistent(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        with pytest.raises(CacheError, match="Cache not found"):
            manager.invalidate("ghost")

    def test_rejects_traversal(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        with pytest.raises(ValueError, match="Invalid name"):
            manager.invalidate("../escape")


# ════════════════════════════════════════════════════════════════════════
# 6. list_cached
# ════════════════════════════════════════════════════════════════════════


class TestListCached:
    def test_empty_when_no_cache_root(self, tmp_path):
        cache_root = tmp_path / "nonexistent"
        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        assert manager.list_cached() == []

    def test_lists_cached_dirs_with_manifests(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        manager.pull("raw/corpora/a.tar.gz", "dataset-a")
        manager.pull("raw/corpora/b.tar.gz", "dataset-b")

        result = manager.list_cached()
        assert result == ["dataset-a", "dataset-b"]

    def test_ignores_dirs_without_manifest(self, tmp_path):
        cache_root = tmp_path / "cache"
        (cache_root / "no-manifest").mkdir(parents=True)
        (cache_root / "has-manifest").mkdir(parents=True)
        (cache_root / "has-manifest" / "manifest.json").write_text("{}")

        storage = MagicMock()
        manager = CacheManager(storage, cache_root)

        result = manager.list_cached()
        assert result == ["has-manifest"]

    def test_ignores_tmp_dir(self, tmp_path):
        cache_root = tmp_path / "cache"
        storage = _mock_storage(tmp_path)
        manager = CacheManager(storage, cache_root)

        manager.pull("raw/corpora/test.tar.gz", "real-dataset")

        # .tmp won't have a manifest.json, so it's ignored
        result = manager.list_cached()
        assert ".tmp" not in result
        assert result == ["real-dataset"]
