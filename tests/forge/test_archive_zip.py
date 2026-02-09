"""Test suite for forge.archive.zip — safe ZIP extraction handler.

Tests build archives programmatically using zipfile, then extract
via ZipHandler to verify both happy paths and security rejections.
"""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

import pytest

from modules.forge.archive.safety import ArchiveMember, ExtractionError, ExtractionLimits
from modules.forge.archive.zip import ZipHandler

# ── Helpers ────────────────────────────────────────────────────────────


def _make_zip(tmp_path: Path, members: list[dict]) -> Path:
    """Build a ZIP archive from a list of member dicts.

    Each dict can have: name, data (bytes), type ("file"|"dir"|"symlink"),
    link_target.
    """
    archive_path = tmp_path / "test.zip"
    tmp_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in members:
            name = m["name"]
            mtype = m.get("type", "file")
            data = m.get("data", b"")

            if mtype == "dir":
                # Directories end with /
                if not name.endswith("/"):
                    name += "/"
                zf.writestr(zipfile.ZipInfo(name), "")
            elif mtype == "symlink":
                info = zipfile.ZipInfo(name)
                # Set Unix symlink mode in external_attr
                # 0o120000 is the symlink file type, 0o777 permissions
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                target = m.get("link_target", "")
                zf.writestr(info, target.encode("utf-8"))
            else:
                zf.writestr(name, data)

    return archive_path


def _make_zip_with_raw_filename(tmp_path: Path, filename: str, data: bytes) -> Path:
    """Build a ZIP with a raw filename (for injection tests)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        info = zipfile.ZipInfo(filename)
        zf.writestr(info, data)
    return archive_path


# ════════════════════════════════════════════════════════════════════════
# 1. Normal extraction
# ════════════════════════════════════════════════════════════════════════


class TestZipExtractNormal:
    def test_files_and_dirs(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "data/", "type": "dir"},
            {"name": "data/hello.txt", "data": b"hello world"},
            {"name": "data/sub/", "type": "dir"},
            {"name": "data/sub/nested.txt", "data": b"nested"},
        ])
        dest = tmp_path / "out"
        handler = ZipHandler()
        paths = handler.extract(archive, dest)

        assert len(paths) == 4
        assert (dest / "data" / "hello.txt").read_bytes() == b"hello world"
        assert (dest / "data" / "sub" / "nested.txt").read_bytes() == b"nested"

    def test_empty_file(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "empty.txt", "data": b""},
        ])
        dest = tmp_path / "out"
        ZipHandler().extract(archive, dest)
        assert (dest / "empty.txt").read_bytes() == b""

    def test_file_permissions(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "file.txt", "data": b"data"},
        ])
        dest = tmp_path / "out"
        ZipHandler().extract(archive, dest)
        mode = os.stat(dest / "file.txt").st_mode & 0o777
        assert mode == 0o644


# ════════════════════════════════════════════════════════════════════════
# 2. ZIP Slip (path traversal)
# ════════════════════════════════════════════════════════════════════════


class TestZipSlip:
    def test_dotdot_rejected(self, tmp_path):
        archive = _make_zip_with_raw_filename(
            tmp_path / "src", "../escape.txt", b"evil"
        )
        with pytest.raises(ExtractionError, match="Path traversal"):
            ZipHandler().extract(archive, tmp_path / "out")

    def test_nested_traversal_rejected(self, tmp_path):
        archive = _make_zip_with_raw_filename(
            tmp_path / "src", "a/b/../../../escape.txt", b"evil"
        )
        with pytest.raises(ExtractionError, match="Path traversal"):
            ZipHandler().extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 3. Symlinks via external_attr
# ════════════════════════════════════════════════════════════════════════


class TestZipSymlinkExternal:
    def test_rejected_by_default(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "link", "type": "symlink", "link_target": "target.txt"},
        ])
        with pytest.raises(ExtractionError, match="Symlinks not allowed"):
            ZipHandler().extract(archive, tmp_path / "out")


class TestZipSymlinkAllowed:
    def test_safe_symlink_extracted(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "target.txt", "data": b"real file"},
            {"name": "link", "type": "symlink", "link_target": "target.txt"},
        ])
        dest = tmp_path / "out"
        handler = ZipHandler(allow_symlinks=True)
        handler.extract(archive, dest)
        assert (dest / "link").is_symlink()
        assert os.readlink(dest / "link") == "target.txt"

    def test_escape_rejected(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "link", "type": "symlink", "link_target": "../../etc/passwd"},
        ])
        with pytest.raises(ExtractionError, match="Link target escapes"):
            ZipHandler(allow_symlinks=True).extract(archive, tmp_path / "out")


# ════════════════════════════════════════════════════════════════════════
# 4. Size bomb
# ════════════════════════════════════════════════════════════════════════


class TestZipSizeBomb:
    def test_oversized_content_rejected(self, tmp_path):
        """File with content exceeding limit caught by validation."""
        limits = ExtractionLimits(max_file_size=50)
        archive = _make_zip(tmp_path / "src", [
            {"name": "bomb.bin", "data": b"x" * 100},
        ])
        with pytest.raises(ExtractionError, match="File too large"):
            ZipHandler(limits=limits).extract(archive, tmp_path / "out")

    def test_within_limits_accepted(self, tmp_path):
        limits = ExtractionLimits(max_file_size=200)
        archive = _make_zip(tmp_path / "src", [
            {"name": "ok.bin", "data": b"x" * 100},
        ])
        dest = tmp_path / "out"
        ZipHandler(limits=limits).extract(archive, dest)
        assert (dest / "ok.bin").read_bytes() == b"x" * 100


# ════════════════════════════════════════════════════════════════════════
# 5. list_members
# ════════════════════════════════════════════════════════════════════════


class TestZipListMembers:
    def test_lists_all_members(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "dir/", "type": "dir"},
            {"name": "dir/file.txt", "data": b"hello"},
        ])
        members = ZipHandler().list_members(archive)
        assert len(members) == 2
        assert isinstance(members[0], ArchiveMember)

        dirs = [m for m in members if m.is_dir]
        files = [m for m in members if m.is_file]
        assert len(dirs) == 1
        assert len(files) == 1
        assert files[0].name == "dir/file.txt"
        assert files[0].size == 5

    def test_detects_symlinks(self, tmp_path):
        archive = _make_zip(tmp_path / "src", [
            {"name": "link", "type": "symlink", "link_target": "target.txt"},
        ])
        members = ZipHandler().list_members(archive)
        assert len(members) == 1
        assert members[0].is_symlink
        assert members[0].link_target == "target.txt"


# ════════════════════════════════════════════════════════════════════════
# 6. create round-trip
# ════════════════════════════════════════════════════════════════════════


class TestZipCreate:
    def test_round_trip(self, tmp_path):
        # Create source directory
        src = tmp_path / "source"
        src.mkdir()
        (src / "sub").mkdir()
        (src / "file.txt").write_bytes(b"content")
        (src / "sub" / "nested.txt").write_bytes(b"nested content")

        # Create archive
        archive = tmp_path / "archive.zip"
        handler = ZipHandler()
        handler.create(src, archive)
        assert archive.exists()

        # Extract and verify
        dest = tmp_path / "dest"
        handler.extract(archive, dest)
        assert (dest / "file.txt").read_bytes() == b"content"
        assert (dest / "sub" / "nested.txt").read_bytes() == b"nested content"
