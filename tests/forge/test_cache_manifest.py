"""Test suite for forge.cache.manifest — cache manifest dataclasses."""

from __future__ import annotations

import json

import pytest

from modules.forge.cache.manifest import CacheManifest, ManifestEntry

# ════════════════════════════════════════════════════════════════════════
# 1. ManifestEntry
# ════════════════════════════════════════════════════════════════════════


class TestManifestEntry:
    def test_creation(self):
        entry = ManifestEntry(path="audio/001.wav", size=1024)
        assert entry.path == "audio/001.wav"
        assert entry.size == 1024

    def test_frozen(self):
        entry = ManifestEntry(path="a.wav", size=100)
        with pytest.raises(AttributeError):
            entry.path = "b.wav"


# ════════════════════════════════════════════════════════════════════════
# 2. CacheManifest serialization
# ════════════════════════════════════════════════════════════════════════


class TestCacheManifest:
    @pytest.fixture()
    def sample_manifest(self) -> CacheManifest:
        return CacheManifest(
            archive_sha256="abc123def456",
            files=[
                ManifestEntry(path="audio/001.wav", size=1024),
                ManifestEntry(path="audio/002.wav", size=2048),
            ],
            total_size=3072,
            extracted_at="2025-01-01T00:00:00+00:00",
            forge_version="0.1.0",
        )

    def test_to_dict(self, sample_manifest):
        d = sample_manifest.to_dict()
        assert d["archive_sha256"] == "abc123def456"
        assert len(d["files"]) == 2
        assert d["files"][0]["path"] == "audio/001.wav"
        assert d["total_size"] == 3072
        assert d["extracted_at"] == "2025-01-01T00:00:00+00:00"
        assert d["forge_version"] == "0.1.0"

    def test_from_dict_roundtrip(self, sample_manifest):
        d = sample_manifest.to_dict()
        restored = CacheManifest.from_dict(d)
        assert restored.archive_sha256 == sample_manifest.archive_sha256
        assert len(restored.files) == len(sample_manifest.files)
        assert restored.files[0].path == "audio/001.wav"
        assert restored.files[0].size == 1024
        assert restored.total_size == 3072

    def test_from_dict_missing_optional_fields(self):
        data = {"archive_sha256": "abc123"}
        manifest = CacheManifest.from_dict(data)
        assert manifest.archive_sha256 == "abc123"
        assert manifest.files == []
        assert manifest.total_size == 0
        assert manifest.extracted_at == ""
        assert manifest.forge_version == ""

    def test_save_and_load(self, tmp_path, sample_manifest):
        path = tmp_path / "manifest.json"
        sample_manifest.save(path)

        assert path.exists()
        # Verify it's valid JSON
        with open(path) as f:
            data = json.load(f)
        assert data["archive_sha256"] == "abc123def456"

        # Load back
        loaded = CacheManifest.load(path)
        assert loaded.archive_sha256 == sample_manifest.archive_sha256
        assert len(loaded.files) == 2
        assert loaded.total_size == 3072

    def test_save_creates_parent_dirs(self, tmp_path, sample_manifest):
        path = tmp_path / "deep" / "nested" / "manifest.json"
        sample_manifest.save(path)
        assert path.exists()
