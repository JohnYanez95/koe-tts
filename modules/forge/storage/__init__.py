"""Storage abstraction and S3/MinIO backend.

Re-exports the public API from ``forge.storage.protocols`` and
``forge.storage.s3``.
"""

from modules.forge.storage.protocols import (
    NotFoundError,
    StorageBackend,
    StorageError,
)
from modules.forge.storage.s3 import (
    S3ConfigError,
    S3StorageBackend,
    S3UploadError,
    build_raw_zone_prefix,
    collect_upload_files,
    ensure_bucket,
    get_s3_client,
    get_s3_config,
    is_s3_available,
    upload_directory,
    upload_file,
    validate_path_component,
)

__all__ = [
    "NotFoundError",
    "S3ConfigError",
    "S3StorageBackend",
    "S3UploadError",
    "StorageBackend",
    "StorageError",
    "build_raw_zone_prefix",
    "collect_upload_files",
    "ensure_bucket",
    "get_s3_client",
    "get_s3_config",
    "is_s3_available",
    "upload_directory",
    "upload_file",
    "validate_path_component",
]
