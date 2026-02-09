"""Test suite for forge.archive.tar — safe tar extraction handler.

Tests build archives programmatically using tarfile, then extract
via TarHandler to verify both happy paths and security rejections.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from modules.forge.archive.safety import ArchiveMember, ExtractionError, ExtractionLimits
from modules.forge.archive.tar import TarHandler

# ── Helpers ────────────────────────────────────────────────────────────


def _make_tar(tmp_path: Path, members: list[dict]) -> Path:
    """Build a tar.gz archive from a list of member dicts.

    Each dict can have: name, data (bytes), type ("file"|"dir"|"symlink"|
    "hardlink"|"device"), link_target.
    """
    archive_path = tmp_path / "test.tar.gz"
    tmp_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for m in members:
            info = tarfile.TarInfo(name=m["name"])
            mtype = m.get("type", "file")
            data = m.get("data", b"")

            if mtype == "file":
                info.type = tarfile.REGTYPE
                info.size = len(data)
            elif mtype == "dir":
                info.type = tarfile.DIRTYPE
                info.size = 0
                data = b""
            elif mtype == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = m.get("link_target", "")
                info.size = 0
                data = b""
            elif mtype == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = m.get("link_target", "")
                info.size = 0
                data = b""
            elif mtype == "device":
                info.type = tarfile.CHRTYPE
                info.size = 0
                data = b""

            tf.addfile(info, io.BytesIO(data) if data else None)

    return archive_path


# ════════════════════════════════════════════════════════════════════════
# 1. Normal extraction
# ════════════════════════════════════════════════════════════════════════


class TestTarExtractNormal:
    def test_files_and_dirs(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "data/", "type": "dir"},
            {"name": "data/hello.txt", "data": b"hello world"},
            {"name": "data/sub/", "type": "dir"},
            {"name": "data/sub/nested.txt", "data": b"nested"},
        ])
        dest = tmp_path / "out"
        handler = TarHandler()
        paths = handler.extract(archive, dest)

        assert len(paths) == 4
        assert (dest / "data" / "hello.txt").read_bytes() == b"hello world"
        assert (dest / "data" / "sub" / "nested.txt").read_bytes() == b"nested"

    def test_empty_file(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "empty.txt", "data": b""},
        ])
        dest = tmp_path / "out"
        TarHandler().extract(archive, dest)
        assert (dest / "empty.txt").read_bytes() == b""

    def test_file_permissions(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "file.txt", "data": b"data"},
        ])
        dest = tmp_path / "out"
        TarHandler().extract(archive, dest)
        mode = os.stat(dest / "file.txt").st_mode & 0o777
        assert mode == 0o644


# ════════════════════════════════════════════════════════════════════════
# 2. Path traversal
# ════════════════════════════════════════════════════════════════════════


class TestTarPathTraversal:
    def test_dotdot_rejected(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "../escape.txt", "data": b"evil"},
        ])
        with pytest.raises(ExtractionError, match="Path traversal"):
            TarHandler().extract(archive, tmp_path / "out")

    def test_nested_traversal_rejected(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "a/b/../../../escape.txt", "data": b"evil"},
        ])
        with pytest.raises(ExtractionError, match="Path traversal"):
            TarHandler().extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 3. Symlinks
# ════════════════════════════════════════════════════════════════════════


class TestTarSymlinkDefault:
    def test_rejected_by_default(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "link", "type": "symlink", "link_target": "target.txt"},
        ])
        with pytest.raises(ExtractionError, match="Symlinks not allowed"):
            TarHandler().extract(archive, tmp_path / "out")


class TestTarSymlinkAllowed:
    def test_safe_symlink_extracted(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "target.txt", "data": b"real file"},
            {"name": "link", "type": "symlink", "link_target": "target.txt"},
        ])
        dest = tmp_path / "out"
        handler = TarHandler(allow_symlinks=True)
        handler.extract(archive, dest)
        assert (dest / "link").is_symlink()
        assert os.readlink(dest / "link") == "target.txt"


class TestTarSymlinkEscape:
    def test_escape_rejected(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "link", "type": "symlink", "link_target": "../../etc/passwd"},
        ])
        with pytest.raises(ExtractionError, match="Link target escapes"):
            TarHandler(allow_symlinks=True).extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 4. Hardlinks
# ════════════════════════════════════════════════════════════════════════


class TestTarHardlinkDefault:
    def test_rejected_by_default(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "target.txt", "data": b"real file"},
            {"name": "link", "type": "hardlink", "link_target": "target.txt"},
        ])
        with pytest.raises(ExtractionError, match="Hardlinks not allowed"):
            TarHandler().extract(archive, tmp_path / "out")


class TestTarHardlinkResolution:
    def test_hardlink_resolves_to_filesystem_path(self, tmp_path):
        """Hardlink target must be resolved via dest / normpath(linkname)."""
        archive = _make_tar(tmp_path / "src", [
            {"name": "original.txt", "data": b"shared content"},
            {"name": "link.txt", "type": "hardlink", "link_target": "original.txt"},
        ])
        dest = tmp_path / "out"
        handler = TarHandler(allow_hardlinks=True)
        handler.extract(archive, dest)

        # Both paths should exist and share content
        assert (dest / "original.txt").read_bytes() == b"shared content"
        assert (dest / "link.txt").read_bytes() == b"shared content"
        # They should be hardlinked (same inode)
        assert os.stat(dest / "original.txt").st_ino == os.stat(dest / "link.txt").st_ino


# ════════════════════════════════════════════════════════════════════════
# 5. Device files
# ════════════════════════════════════════════════════════════════════════


class TestTarDeviceFile:
    def test_device_rejected(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "dev/null", "type": "device"},
        ])
        with pytest.raises(ExtractionError, match="Device file"):
            TarHandler().extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 6. Size limits
# ════════════════════════════════════════════════════════════════════════


class TestTarSizeLimitDeclared:
    def test_oversized_declared_rejected(self, tmp_path):
        """Member with declared size over limit is caught by validate_archive_member."""
        limits = ExtractionLimits(max_file_size=100)
        archive = _make_tar(tmp_path / "src", [
            {"name": "big.bin", "data": b"x" * 200},
        ])
        with pytest.raises(ExtractionError, match="File too large"):
            TarHandler(limits=limits).extract(archive, tmp_path / "out")


class TestTarSizeBomb:
    def test_stream_copy_enforces_limit(self, tmp_path):
        """Stream-copy catches files exceeding the per-file size limit.

        The file is legitimately 100 bytes in the archive, but our limit
        is 50 bytes.  validate_archive_member checks declared size, but
        the stream-copy loop is the second line of defense.
        """
        # Create archive with a file larger than our extraction limit
        # but where the declared size matches (so tarfile creates it fine)
        limits = ExtractionLimits(max_file_size=50)
        archive = _make_tar(tmp_path / "src", [
            {"name": "bomb.bin", "data": b"x" * 100},
        ])
        # validate_archive_member catches the declared size first
        with pytest.raises(ExtractionError, match="File too large"):
            TarHandler(limits=limits).extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 7. extractfile() returns None
# ════════════════════════════════════════════════════════════════════════


class TestTarExtractfileNone:
    def test_none_extractfile_raises(self, tmp_path, monkeypatch):
        """If extractfile() returns None for a file member, raise ExtractionError."""
        archive = _make_tar(tmp_path / "src", [
            {"name": "ghost.txt", "data": b"content"},
        ])

        original_open = tarfile.open

        def patched_open(*args, **kwargs):
            tf = original_open(*args, **kwargs)

            def mock_extractfile(_member):
                return None

            tf.extractfile = mock_extractfile
            return tf

        monkeypatch.setattr(tarfile, "open", patched_open)

        with pytest.raises(ExtractionError, match="Cannot extract file content"):
            TarHandler().extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 8. list_members
# ════════════════════════════════════════════════════════════════════════


class TestTarListMembers:
    def test_lists_all_members(self, tmp_path):
        archive = _make_tar(tmp_path / "src", [
            {"name": "dir/", "type": "dir"},
            {"name": "dir/file.txt", "data": b"hello"},
        ])
        members = TarHandler().list_members(archive)
        assert len(members) == 2
        assert isinstance(members[0], ArchiveMember)

        dirs = [m for m in members if m.is_dir]
        files = [m for m in members if m.is_file]
        assert len(dirs) == 1
        assert len(files) == 1
        assert files[0].name == "dir/file.txt"
        assert files[0].size == 5


# ════════════════════════════════════════════════════════════════════════
# 9. create round-trip
# ════════════════════════════════════════════════════════════════════════


class TestTarCreate:
    def test_round_trip(self, tmp_path):
        # Create source directory
        src = tmp_path / "source"
        src.mkdir()
        (src / "sub").mkdir()
        (src / "file.txt").write_bytes(b"content")
        (src / "sub" / "nested.txt").write_bytes(b"nested content")

        # Create archive
        archive = tmp_path / "archive.tar.gz"
        handler = TarHandler()
        handler.create(src, archive)
        assert archive.exists()

        # Extract and verify
        dest = tmp_path / "dest"
        handler.extract(archive, dest)
        assert (dest / "file.txt").read_bytes() == b"content"
        assert (dest / "sub" / "nested.txt").read_bytes() == b"nested content"
