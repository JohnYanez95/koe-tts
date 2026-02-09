"""Test suite for forge.storage.protocols — storage abstraction layer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from modules.forge.storage.protocols import (
    NotFoundError,
    StorageBackend,
    StorageError,
)

# ════════════════════════════════════════════════════════════════════════
# 1. Protocol conformance
# ════════════════════════════════════════════════════════════════════════


class _MinimalBackend:
    """Minimal concrete class that satisfies StorageBackend."""

    def put(self, key: str, source: Path) -> str:
        return key

    def get_to_path(self, key: str, dest: Path) -> Path:
        return dest

    def exists(self, key: str) -> bool:
        return False

    def list_keys(self, prefix: str) -> Iterator[str]:
        yield from []


class TestStorageBackendProtocol:
    def test_isinstance_check(self):
        backend = _MinimalBackend()
        assert isinstance(backend, StorageBackend)

    def test_non_conforming_rejected(self):
        class _Incomplete:
            def put(self, key: str, source: Path) -> str:
                return key

        assert not isinstance(_Incomplete(), StorageBackend)


# ════════════════════════════════════════════════════════════════════════
# 2. Error hierarchy
# ════════════════════════════════════════════════════════════════════════


class TestErrorHierarchy:
    def test_not_found_is_storage_error(self):
        assert issubclass(NotFoundError, StorageError)

    def test_storage_error_is_exception(self):
        assert issubclass(StorageError, Exception)

    def test_not_found_caught_by_storage_error(self):
        try:
            raise NotFoundError("missing")
        except StorageError:
            pass  # Should be caught
