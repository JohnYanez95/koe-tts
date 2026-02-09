"""Safe tar archive handler.

Extracts tar archives member-by-member — never calls ``extractall()``.
Each member is validated via ``validate_archive_member()`` before any
bytes are written to disk.

Stream-copy enforces per-file size limits during extraction, catching
archives that lie about their declared size.

Public API
----------
- ``TarHandler`` — safe tar extraction, listing, and creation
"""

from __future__ import annotations

import os
import tarfile
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


def _tar_member_type(member: tarfile.TarInfo) -> str:
    """Map a TarInfo to our canonical member type string."""
    if member.isdir():
        return "dir"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.isdev() or member.isblk() or member.ischr() or member.isfifo():
        return "device"
    # isfile() covers regular files (including sparse)
    return "file"


def _tar_link_target(member: tarfile.TarInfo) -> str | None:
    """Extract link target from a TarInfo, if applicable."""
    if member.issym() or member.islnk():
        return member.linkname
    return None


class TarHandler:
    """Safe tar archive handler.

    Parameters
    ----------
    limits
        Extraction limits.  Defaults to ``ExtractionLimits()`` if not
        provided.
    allow_symlinks
        If ``True``, symlinks targeting within the extraction root are
        permitted.
    allow_hardlinks
        If ``True``, hardlinks targeting within the extraction root are
        permitted.
    """

    def __init__(
        self,
        limits: ExtractionLimits | None = None,
        *,
        allow_symlinks: bool = False,
        allow_hardlinks: bool = False,
    ) -> None:
        self._limits = limits or ExtractionLimits()
        self._allow_symlinks = allow_symlinks
        self._allow_hardlinks = allow_hardlinks

    def extract(self, archive: Path, dest: Path) -> list[Path]:
        """Safely extract a tar archive, returning extracted paths.

        Each member is validated before extraction.  Files are written
        via stream-copy with per-file size enforcement.
        """
        dest = dest.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        cumulative_size = 0
        cumulative_files = 0

        with tarfile.open(archive, "r:*") as tf:
            for member in tf.getmembers():
                mtype = _tar_member_type(member)
                link_target = _tar_link_target(member)

                safe_path = validate_archive_member(
                    name=member.name,
                    size=member.size,
                    member_type=mtype,
                    link_target=link_target,
                    extraction_root=dest,
                    limits=self._limits,
                    cumulative_size=cumulative_size,
                    cumulative_files=cumulative_files,
                    allow_symlinks=self._allow_symlinks,
                    allow_hardlinks=self._allow_hardlinks,
                )

                if mtype == "dir":
                    os.makedirs(safe_path, mode=SAFE_DIR_MODE, exist_ok=True)
                elif mtype == "file":
                    self._extract_file(tf, member, safe_path)
                elif mtype == "symlink":
                    safe_path.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(link_target, safe_path)
                elif mtype == "hardlink":
                    safe_path.parent.mkdir(parents=True, exist_ok=True)
                    resolved_target = dest / os.path.normpath(member.linkname)
                    os.link(resolved_target, safe_path)

                cumulative_size += member.size
                cumulative_files += 1
                extracted.append(safe_path)

        return extracted

    def _extract_file(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        safe_path: Path,
    ) -> None:
        """Stream-copy a file member to disk with size enforcement."""
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        fileobj = tf.extractfile(member)
        if fileobj is None:
            raise ExtractionError(
                f"Cannot extract file content: {member.name!r}"
            )

        bytes_written = 0
        try:
            with open(safe_path, "wb") as out:
                while True:
                    chunk = fileobj.read(_COPY_BUFSIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > self._limits.max_file_size:
                        raise ExtractionError(
                            f"File exceeds size limit during extraction: "
                            f"{member.name!r} ({bytes_written} > "
                            f"{self._limits.max_file_size})"
                        )
                    out.write(chunk)
        except ExtractionError:
            # Clean up partial file
            safe_path.unlink(missing_ok=True)
            raise
        finally:
            fileobj.close()

        os.chmod(safe_path, SAFE_FILE_MODE)

    def list_members(self, archive: Path) -> list[ArchiveMember]:
        """List archive contents without extracting."""
        members: list[ArchiveMember] = []
        with tarfile.open(archive, "r:*") as tf:
            for m in tf.getmembers():
                members.append(ArchiveMember(
                    name=m.name,
                    size=m.size,
                    is_file=m.isfile(),
                    is_dir=m.isdir(),
                    is_symlink=m.issym(),
                    link_target=_tar_link_target(m),
                ))
        return members

    def create(
        self,
        source: Path,
        archive: Path,
        compression: str = "gz",
    ) -> None:
        """Create a tar archive from a directory.

        Parameters
        ----------
        source
            Directory to archive.
        archive
            Destination archive path.
        compression
            Compression type: ``"gz"``, ``"bz2"``, ``"xz"``, or ``""``
            for no compression.
        """
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, f"w:{compression}") as tf:
            for item in sorted(source.rglob("*")):
                arcname = str(item.relative_to(source))
                tf.add(item, arcname=arcname, recursive=False)
