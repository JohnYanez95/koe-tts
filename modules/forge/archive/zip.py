"""Safe ZIP archive handler.

Extracts ZIP archives member-by-member — never calls ``extractall()``.
Each member is validated via ``validate_archive_member()`` before any
bytes are written to disk.

Detects Unix symlinks stored in ZIP external attributes and rejects
them by default.

Public API
----------
- ``ZipHandler`` — safe ZIP extraction, listing, and creation
"""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

from modules.forge.archive.safety import (
    SAFE_DIR_MODE,
    SAFE_FILE_MODE,
    ArchiveMember,
    ExtractionError,
    ExtractionLimits,
    validate_archive_member,
)

_COPY_BUFSIZE: int = 262_144  # 256 KB


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    """Detect Unix symlinks stored in ZIP external attributes."""
    # Upper 16 bits of external_attr contain Unix mode bits
    unix_mode = info.external_attr >> 16
    return stat.S_ISLNK(unix_mode) if unix_mode else False


def _zip_member_type(info: zipfile.ZipInfo) -> str:
    """Map a ZipInfo to our canonical member type string."""
    if info.is_dir():
        return "dir"
    if _is_zip_symlink(info):
        return "symlink"
    return "file"


class ZipHandler:
    """Safe ZIP archive handler.

    Parameters
    ----------
    limits
        Extraction limits.  Defaults to ``ExtractionLimits()`` if not
        provided.
    allow_symlinks
        If ``True``, symlinks targeting within the extraction root are
        permitted.
    """

    def __init__(
        self,
        limits: ExtractionLimits | None = None,
        *,
        allow_symlinks: bool = False,
    ) -> None:
        self._limits = limits or ExtractionLimits()
        self._allow_symlinks = allow_symlinks

    def extract(self, archive: Path, dest: Path) -> list[Path]:
        """Safely extract a ZIP archive, returning extracted paths.

        Each member is validated before extraction.  Files are written
        via stream-copy with per-file size enforcement.
        """
        dest = dest.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        cumulative_size = 0
        cumulative_files = 0

        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                mtype = _zip_member_type(info)
                link_target = None

                if mtype == "symlink":
                    # Symlink target is stored as the file's data
                    link_target = zf.read(info).decode("utf-8", errors="replace")

                safe_path = validate_archive_member(
                    name=info.filename,
                    size=info.file_size,
                    member_type=mtype,
                    link_target=link_target,
                    extraction_root=dest,
                    limits=self._limits,
                    cumulative_size=cumulative_size,
                    cumulative_files=cumulative_files,
                    allow_symlinks=self._allow_symlinks,
                )

                if mtype == "dir":
                    os.makedirs(safe_path, mode=SAFE_DIR_MODE, exist_ok=True)
                elif mtype == "file":
                    self._extract_file(zf, info, safe_path)
                elif mtype == "symlink":
                    safe_path.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(link_target, safe_path)

                cumulative_size += info.file_size
                cumulative_files += 1
                extracted.append(safe_path)

        return extracted

    def _extract_file(
        self,
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        safe_path: Path,
    ) -> None:
        """Stream-copy a file member to disk with size enforcement."""
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        bytes_written = 0
        try:
            with zf.open(info) as src, open(safe_path, "wb") as out:
                while True:
                    chunk = src.read(_COPY_BUFSIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > self._limits.max_file_size:
                        raise ExtractionError(
                            f"File exceeds size limit during extraction: "
                            f"{info.filename!r} ({bytes_written} > "
                            f"{self._limits.max_file_size})"
                        )
                    out.write(chunk)
        except ExtractionError:
            # Clean up partial file
            safe_path.unlink(missing_ok=True)
            raise

        os.chmod(safe_path, SAFE_FILE_MODE)

    def list_members(self, archive: Path) -> list[ArchiveMember]:
        """List archive contents without extracting."""
        members: list[ArchiveMember] = []
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                is_sym = _is_zip_symlink(info)
                link_target = None
                if is_sym:
                    link_target = zf.read(info).decode(
                        "utf-8", errors="replace"
                    )
                members.append(ArchiveMember(
                    name=info.filename,
                    size=info.file_size,
                    is_file=not info.is_dir() and not is_sym,
                    is_dir=info.is_dir(),
                    is_symlink=is_sym,
                    link_target=link_target,
                ))
        return members

    def create(self, source: Path, archive: Path) -> None:
        """Create a ZIP archive from a directory.

        Parameters
        ----------
        source
            Directory to archive.
        archive
            Destination archive path.
        """
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(source.rglob("*")):
                arcname = str(item.relative_to(source))
                if item.is_dir():
                    zf.write(item, arcname=arcname + "/")
                else:
                    zf.write(item, arcname=arcname)
