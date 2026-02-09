"""Archive extraction security gatekeeper.

Validates archive member metadata *before* extraction to prevent:
- Path traversal (``../`` escape, absolute paths)
- Symlink/hardlink attacks (escape to files outside extraction root)
- Zip bombs (file size, total size, file count limits)
- Device file extraction
- Null byte injection (C-level path truncation)
- Windows path separator attacks

All checks are performed on metadata alone — no actual files are read
or extracted.  This module is stdlib-only (no external dependencies).

Public API
----------
- ``validate_archive_member(...)`` — validate a single archive member
- ``is_path_safe(path, root)`` — check if a path resolves within root
- ``ExtractionError`` — raised on any validation failure
- ``ExtractionLimits`` — configurable limits dataclass
- ``ArchiveMember`` — metadata dataclass for archive listing
- ``SAFE_DIR_MODE`` / ``SAFE_FILE_MODE`` — recommended permissions

Example
-------
>>> from modules.forge.archive.safety import validate_archive_member, ExtractionLimits
>>> limits = ExtractionLimits()
>>> dest = validate_archive_member(
...     name="data/file.txt",
...     size=1024,
...     member_type="file",
...     link_target=None,
...     extraction_root=Path("/tmp/extract"),
...     limits=limits,
... )
>>> dest
PosixPath('/tmp/extract/data/file.txt')
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ExtractionError(ValueError):
    """Raised when an archive member fails validation."""


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Configurable limits for archive extraction.

    All size values are in bytes.
    """

    max_file_size: int = 500_000_000  # 500 MB
    max_total_size: int = 10_000_000_000  # 10 GB
    max_files: int = 100_000
    max_path_length: int = 1024
    allowed_extensions: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """Metadata for a single archive member (used by list_members)."""

    name: str
    size: int
    is_file: bool
    is_dir: bool
    is_symlink: bool
    link_target: str | None = None


# Recommended permissions for extracted files/directories.
SAFE_DIR_MODE: int = 0o755
SAFE_FILE_MODE: int = 0o644

# Device file names that must never be extracted (case-insensitive).
_DEVICE_NAMES: frozenset[str] = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})

# Characters forbidden in path components (Windows-specific attack vectors).
_WINDOWS_FORBIDDEN: frozenset[str] = frozenset("\\:")


def _is_device_name(name: str) -> bool:
    """Check if a filename (without extension) is a Windows device name."""
    stem = PurePosixPath(name).stem.lower()
    return stem in _DEVICE_NAMES


def _check_link_containment(
    link_target: str,
    extraction_root: Path,
    member_dir: Path,
) -> None:
    """Verify a symlink/hardlink target resolves within extraction_root.

    Uses pure path arithmetic — does not touch the filesystem — because
    link targets may reference files that don't exist yet during
    extraction.
    """
    normalized = os.path.normpath(link_target)

    # Absolute link targets: must be under extraction_root
    if os.path.isabs(normalized):
        target_path = Path(normalized)
    else:
        # Relative link: resolve from the directory containing the member
        target_path = member_dir / normalized

    # Normalize to remove any remaining ..
    target_resolved = Path(os.path.normpath(target_path))

    if not _is_relative_to(target_resolved, extraction_root):
        raise ExtractionError(
            f"Link target escapes extraction root: {link_target!r}"
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    """Check if *path* is equal to or a child of *root*.

    Pure path check — both paths should already be normalized.
    """
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_archive_member(
    name: str,
    size: int,
    member_type: str,
    link_target: str | None,
    extraction_root: Path,
    limits: ExtractionLimits,
    *,
    cumulative_size: int = 0,
    cumulative_files: int = 0,
    allow_symlinks: bool = False,
    allow_hardlinks: bool = False,
) -> Path:
    """Validate a single archive member and return the safe destination path.

    Parameters
    ----------
    name
        The member name/path as stored in the archive.
    size
        The uncompressed size in bytes (0 for directories/links).
    member_type
        One of ``"file"``, ``"dir"``, ``"symlink"``, ``"hardlink"``,
        ``"device"``, or other archive-specific types.
    link_target
        Target path for symlinks/hardlinks, ``None`` otherwise.
    extraction_root
        The root directory files will be extracted into.
    limits
        Extraction limits to enforce.
    cumulative_size
        Total bytes already extracted (for bomb detection).
    cumulative_files
        Total files already extracted (for bomb detection).
    allow_symlinks
        If ``True``, symlinks are permitted but must target within
        *extraction_root*.
    allow_hardlinks
        If ``True``, hardlinks are permitted but must target within
        *extraction_root*.

    Returns
    -------
    Path
        The resolved, validated destination path.

    Raises
    ------
    ExtractionError
        If any validation check fails.
    """
    # ── 1. Unicode NFC normalization ────────────────────────────────
    name = unicodedata.normalize("NFC", name)

    # ── 2. Empty / whitespace name ──────────────────────────────────
    if not name or not name.strip():
        raise ExtractionError("Empty or whitespace-only member name")

    # ── 3. Null byte rejection ──────────────────────────────────────
    if "\x00" in name:
        raise ExtractionError(
            "Member name contains null byte (path truncation attack)"
        )

    # ── 4. Windows path characters ──────────────────────────────────
    if any(ch in name for ch in _WINDOWS_FORBIDDEN):
        raise ExtractionError(
            f"Member name contains forbidden character: {name!r}"
        )

    # ── 5. Path length limit ────────────────────────────────────────
    if len(name) > limits.max_path_length:
        raise ExtractionError(
            f"Member path too long ({len(name)} > {limits.max_path_length})"
        )

    # ── 6. Absolute path ───────────────────────────────────────────
    if os.path.isabs(name):
        raise ExtractionError(f"Absolute path in archive: {name!r}")

    # ── 7. Path traversal ──────────────────────────────────────────
    normalized = os.path.normpath(name)
    dest = Path(os.path.normpath(extraction_root / normalized))

    if not _is_relative_to(dest, extraction_root):
        raise ExtractionError(
            f"Path traversal detected: {name!r}"
        )

    # ── 8. Device file rejection ───────────────────────────────────
    if member_type == "device":
        raise ExtractionError(
            f"Device file in archive: {name!r}"
        )
    # Also reject Windows device names in any path component
    for part in PurePosixPath(normalized).parts:
        if _is_device_name(part):
            raise ExtractionError(
                f"Device file name in archive path: {name!r}"
            )

    # ── 9. Symlink handling ────────────────────────────────────────
    if member_type == "symlink":
        if not allow_symlinks:
            raise ExtractionError(
                f"Symlinks not allowed: {name!r}"
            )
        if link_target is None:
            raise ExtractionError(
                f"Symlink without target: {name!r}"
            )
        _check_link_containment(link_target, extraction_root, dest.parent)

    # ── 10. Hardlink handling ──────────────────────────────────────
    if member_type == "hardlink":
        if not allow_hardlinks:
            raise ExtractionError(
                f"Hardlinks not allowed: {name!r}"
            )
        if link_target is None:
            raise ExtractionError(
                f"Hardlink without target: {name!r}"
            )
        _check_link_containment(link_target, extraction_root, dest.parent)

    # ── 11. Per-file size limit ────────────────────────────────────
    if size > limits.max_file_size:
        raise ExtractionError(
            f"File too large: {name!r} ({size} > {limits.max_file_size})"
        )

    # ── 12. Cumulative size limit ──────────────────────────────────
    new_total = cumulative_size + size
    if new_total > limits.max_total_size:
        raise ExtractionError(
            f"Total extraction size exceeded "
            f"({new_total} > {limits.max_total_size})"
        )

    # ── 13. File count limit ───────────────────────────────────────
    new_count = cumulative_files + 1
    if new_count > limits.max_files:
        raise ExtractionError(
            f"Too many files ({new_count} > {limits.max_files})"
        )

    # ── 14. Extension filter (directories exempt) ──────────────────
    if (
        limits.allowed_extensions is not None
        and member_type != "dir"
    ):
        ext = PurePosixPath(name).suffix.lower()
        if ext not in limits.allowed_extensions:
            raise ExtractionError(
                f"Extension not allowed: {ext!r} for {name!r}"
            )

    # ── 15. Return resolved safe path ──────────────────────────────
    return dest


def is_path_safe(path: str, root: Path) -> bool:
    """Check whether *path* resolves to a location within *root*.

    Returns ``True`` if the normalized, joined path is equal to or a
    child of *root*.  Returns ``False`` for traversal attempts, absolute
    paths, or null bytes.

    This is a convenience wrapper — for full validation of archive
    members, use ``validate_archive_member`` instead.
    """
    if "\x00" in path:
        return False

    if os.path.isabs(path):
        return False

    normalized = os.path.normpath(path)
    dest = Path(os.path.normpath(root / normalized))
    return _is_relative_to(dest, root)
