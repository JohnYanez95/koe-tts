"""Test suite for forge.archive.safety — archive extraction security gatekeeper.

All tests call validate_archive_member() with constructed parameters.
No actual archive files are needed — pure metadata validation.
"""

from pathlib import Path

import pytest

from modules.forge.archive.safety import (
    ExtractionError,
    ExtractionLimits,
    is_path_safe,
    validate_archive_member,
)

# ── Fixtures ───────────────────────────────────────────────────────────

ROOT = Path("/tmp/extract")
LIMITS = ExtractionLimits()


def _validate(name, *, size=100, member_type="file", link_target=None,
              root=ROOT, limits=LIMITS, **kwargs):
    """Shorthand for validate_archive_member with sensible defaults."""
    return validate_archive_member(
        name=name,
        size=size,
        member_type=member_type,
        link_target=link_target,
        extraction_root=root,
        limits=limits,
        **kwargs,
    )


# ════════════════════════════════════════════════════════════════════════
# 1. Path traversal
# ════════════════════════════════════════════════════════════════════════

class TestPathTraversal:
    def test_dotdot_rejected(self):
        with pytest.raises(ExtractionError, match="Path traversal"):
            _validate("../etc/passwd")

    def test_nested_dotdot_rejected(self):
        with pytest.raises(ExtractionError, match="Path traversal"):
            _validate("data/../../etc/passwd")

    def test_many_dotdots_rejected(self):
        with pytest.raises(ExtractionError, match="Path traversal"):
            _validate("a/b/c/../../../../etc/shadow")

    def test_normalized_traversal_rejected(self):
        """Even after normalization, path must stay within root."""
        with pytest.raises(ExtractionError, match="Path traversal"):
            _validate("data/subdir/../../../outside")

    def test_clean_nested_path_accepted(self):
        dest = _validate("data/subdir/file.txt")
        assert dest == ROOT / "data" / "subdir" / "file.txt"

    def test_single_file_accepted(self):
        dest = _validate("file.txt")
        assert dest == ROOT / "file.txt"


# ════════════════════════════════════════════════════════════════════════
# 2. Absolute paths
# ════════════════════════════════════════════════════════════════════════

class TestAbsolutePaths:
    def test_absolute_unix_rejected(self):
        with pytest.raises(ExtractionError, match="Absolute path"):
            _validate("/etc/passwd")

    def test_absolute_with_subdir_rejected(self):
        with pytest.raises(ExtractionError, match="Absolute path"):
            _validate("/home/user/data/file.txt")


# ════════════════════════════════════════════════════════════════════════
# 3. Symlink escape
# ════════════════════════════════════════════════════════════════════════

class TestSymlinkEscape:
    def test_symlink_rejected_by_default(self):
        with pytest.raises(ExtractionError, match="Symlinks not allowed"):
            _validate("link", member_type="symlink", link_target="target.txt")

    def test_symlink_allowed_when_opted_in(self):
        dest = _validate(
            "link", member_type="symlink", link_target="target.txt",
            allow_symlinks=True,
        )
        assert dest == ROOT / "link"

    def test_symlink_escape_rejected_even_when_allowed(self):
        with pytest.raises(ExtractionError, match="Link target escapes"):
            _validate(
                "link", member_type="symlink", link_target="../../etc/passwd",
                allow_symlinks=True,
            )

    def test_symlink_absolute_escape_rejected(self):
        with pytest.raises(ExtractionError, match="Link target escapes"):
            _validate(
                "link", member_type="symlink", link_target="/etc/passwd",
                allow_symlinks=True,
            )

    def test_symlink_without_target_rejected(self):
        with pytest.raises(ExtractionError, match="Symlink without target"):
            _validate(
                "link", member_type="symlink", link_target=None,
                allow_symlinks=True,
            )

    def test_symlink_relative_within_root_accepted(self):
        dest = _validate(
            "data/link", member_type="symlink", link_target="sibling.txt",
            allow_symlinks=True,
        )
        assert dest == ROOT / "data" / "link"


# ════════════════════════════════════════════════════════════════════════
# 4. Hardlink attack
# ════════════════════════════════════════════════════════════════════════

class TestHardlinkAttack:
    def test_hardlink_rejected_by_default(self):
        with pytest.raises(ExtractionError, match="Hardlinks not allowed"):
            _validate("link", member_type="hardlink", link_target="target.txt")

    def test_hardlink_allowed_when_opted_in(self):
        dest = _validate(
            "link", member_type="hardlink", link_target="target.txt",
            allow_hardlinks=True,
        )
        assert dest == ROOT / "link"

    def test_hardlink_escape_rejected_even_when_allowed(self):
        with pytest.raises(ExtractionError, match="Link target escapes"):
            _validate(
                "link", member_type="hardlink", link_target="../../etc/passwd",
                allow_hardlinks=True,
            )

    def test_hardlink_without_target_rejected(self):
        with pytest.raises(ExtractionError, match="Hardlink without target"):
            _validate(
                "link", member_type="hardlink", link_target=None,
                allow_hardlinks=True,
            )


# ════════════════════════════════════════════════════════════════════════
# 5. Size bomb
# ════════════════════════════════════════════════════════════════════════

class TestSizeBomb:
    def test_oversized_single_file_rejected(self):
        with pytest.raises(ExtractionError, match="File too large"):
            _validate("big.bin", size=LIMITS.max_file_size + 1)

    def test_cumulative_overflow_rejected(self):
        almost_full = LIMITS.max_total_size - 50
        with pytest.raises(ExtractionError, match="Total extraction size"):
            _validate("last.bin", size=100, cumulative_size=almost_full)

    def test_within_limits_accepted(self):
        dest = _validate("small.bin", size=1024)
        assert dest == ROOT / "small.bin"

    def test_exactly_at_file_limit_accepted(self):
        dest = _validate("exact.bin", size=LIMITS.max_file_size)
        assert dest == ROOT / "exact.bin"


# ════════════════════════════════════════════════════════════════════════
# 6. Device files
# ════════════════════════════════════════════════════════════════════════

class TestDeviceFile:
    def test_device_member_type_rejected(self):
        with pytest.raises(ExtractionError, match="Device file in archive"):
            _validate("dev/sda", member_type="device")

    def test_windows_device_name_rejected(self):
        with pytest.raises(ExtractionError, match="Device file name"):
            _validate("CON")

    def test_windows_device_with_extension_rejected(self):
        with pytest.raises(ExtractionError, match="Device file name"):
            _validate("NUL.txt")

    def test_windows_device_in_subdir_rejected(self):
        with pytest.raises(ExtractionError, match="Device file name"):
            _validate("data/PRN")

    def test_non_device_name_accepted(self):
        dest = _validate("console.log")
        assert dest == ROOT / "console.log"


# ════════════════════════════════════════════════════════════════════════
# 7. Null bytes
# ════════════════════════════════════════════════════════════════════════

class TestNullBytes:
    def test_null_in_name_rejected(self):
        with pytest.raises(ExtractionError, match="null byte"):
            _validate("file\x00.txt")

    def test_null_at_end_rejected(self):
        with pytest.raises(ExtractionError, match="null byte"):
            _validate("file.txt\x00")


# ════════════════════════════════════════════════════════════════════════
# 8. Windows paths
# ════════════════════════════════════════════════════════════════════════

class TestWindowsPaths:
    def test_backslash_rejected(self):
        with pytest.raises(ExtractionError, match="forbidden character"):
            _validate("data\\file.txt")

    def test_drive_letter_rejected(self):
        with pytest.raises(ExtractionError, match="forbidden character"):
            _validate("C:file.txt")

    def test_colon_in_path_rejected(self):
        with pytest.raises(ExtractionError, match="forbidden character"):
            _validate("data:stream")


# ════════════════════════════════════════════════════════════════════════
# 9. Unicode normalization
# ════════════════════════════════════════════════════════════════════════

class TestUnicodeNormalization:
    def test_nfc_normalization_applied(self):
        # U+00E9 (e-acute precomposed) vs U+0065 U+0301 (decomposed)
        precomposed = "caf\u00e9.txt"
        decomposed = "cafe\u0301.txt"
        dest_pre = _validate(precomposed)
        dest_dec = _validate(decomposed)
        # Both should normalize to the same NFC path
        assert dest_pre == dest_dec


# ════════════════════════════════════════════════════════════════════════
# 10. Extension filtering
# ════════════════════════════════════════════════════════════════════════

class TestExtensionFiltering:
    @pytest.fixture
    def restricted_limits(self):
        return ExtractionLimits(
            allowed_extensions=frozenset({".wav", ".txt", ".json"}),
        )

    def test_disallowed_extension_rejected(self, restricted_limits):
        with pytest.raises(ExtractionError, match="Extension not allowed"):
            _validate("script.py", limits=restricted_limits)

    def test_allowed_extension_accepted(self, restricted_limits):
        dest = _validate("audio.wav", limits=restricted_limits)
        assert dest == ROOT / "audio.wav"

    def test_none_allows_all(self):
        dest = _validate("anything.xyz")
        assert dest == ROOT / "anything.xyz"

    def test_case_insensitive(self, restricted_limits):
        dest = _validate("audio.WAV", limits=restricted_limits)
        assert dest == ROOT / "audio.WAV"

    def test_dirs_exempt(self, restricted_limits):
        dest = _validate("subdir", member_type="dir", limits=restricted_limits)
        assert dest == ROOT / "subdir"


# ════════════════════════════════════════════════════════════════════════
# 11. File count limits
# ════════════════════════════════════════════════════════════════════════

class TestFileCountLimits:
    def test_exceeding_count_rejected(self):
        with pytest.raises(ExtractionError, match="Too many files"):
            _validate("file.txt", cumulative_files=LIMITS.max_files)

    def test_at_limit_accepted(self):
        dest = _validate("file.txt", cumulative_files=LIMITS.max_files - 1)
        assert dest == ROOT / "file.txt"


# ════════════════════════════════════════════════════════════════════════
# 12. Path length limits
# ════════════════════════════════════════════════════════════════════════

class TestPathLengthLimits:
    def test_over_limit_rejected(self):
        long_name = "a" * (LIMITS.max_path_length + 1)
        with pytest.raises(ExtractionError, match="path too long"):
            _validate(long_name)

    def test_at_limit_accepted(self):
        name = "a" * LIMITS.max_path_length
        dest = _validate(name)
        assert dest == ROOT / name


# ════════════════════════════════════════════════════════════════════════
# 13. is_path_safe utility
# ════════════════════════════════════════════════════════════════════════

class TestIsPathSafe:
    def test_safe_relative_path(self):
        assert is_path_safe("data/file.txt", ROOT) is True

    def test_traversal_rejected(self):
        assert is_path_safe("../escape", ROOT) is False

    def test_absolute_rejected(self):
        assert is_path_safe("/etc/passwd", ROOT) is False

    def test_null_byte_rejected(self):
        assert is_path_safe("file\x00.txt", ROOT) is False

    def test_current_dir_safe(self):
        assert is_path_safe(".", ROOT) is True

    def test_nested_dotdot_that_stays_inside(self):
        assert is_path_safe("a/b/../c", ROOT) is True


# ════════════════════════════════════════════════════════════════════════
# 14. ExtractionLimits defaults
# ════════════════════════════════════════════════════════════════════════

class TestExtractionLimitsDefaults:
    def test_max_file_size(self):
        assert LIMITS.max_file_size == 500_000_000

    def test_max_total_size(self):
        assert LIMITS.max_total_size == 10_000_000_000

    def test_max_files(self):
        assert LIMITS.max_files == 100_000

    def test_max_path_length(self):
        assert LIMITS.max_path_length == 1024

    def test_allowed_extensions_default_none(self):
        assert LIMITS.allowed_extensions is None

    def test_frozen(self):
        with pytest.raises(AttributeError):
            LIMITS.max_file_size = 0  # type: ignore[misc]
