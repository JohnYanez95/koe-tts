"""Storage abstraction protocol and error hierarchy.

Defines the ``StorageBackend`` protocol that all storage implementations
must satisfy.  Designed for large objects (multi-GB corpus tarballs) —
all reads go to disk, not memory.

Public API
----------
- ``StorageBackend`` — runtime-checkable protocol for storage backends
- ``StorageError`` — base error for storage operations
- ``NotFoundError`` — raised when a requested key does not exist
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable


class StorageError(Exception):
    """Base error for storage operations."""


class NotFoundError(StorageError):
    """Raised when a requested key does not exist."""


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal, disk-first storage protocol.

    Designed for large objects (multi-GB corpus tarballs).  All reads
    go to disk, not memory.
    """

    def put(self, key: str, source: Path) -> str:
        """Upload file to storage.  Returns the storage key."""
        ...

    def get_to_path(self, key: str, dest: Path) -> Path:
        """Download object directly to disk.  Returns dest path.

        Raises ``NotFoundError`` if key does not exist.
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in storage."""
        ...

    def list_keys(self, prefix: str) -> Iterator[str]:
        """List keys matching prefix."""
        ...
