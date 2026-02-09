"""Archive extraction security gatekeeper.

Re-exports the public API from ``forge.archive.safety``.
"""

from modules.forge.archive.safety import (
    SAFE_DIR_MODE,
    SAFE_FILE_MODE,
    ExtractionError,
    ExtractionLimits,
    is_path_safe,
    validate_archive_member,
)

__all__ = [
    "ExtractionError",
    "ExtractionLimits",
    "SAFE_DIR_MODE",
    "SAFE_FILE_MODE",
    "is_path_safe",
    "validate_archive_member",
]
