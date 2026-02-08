"""S3/MinIO utilities for raw zone uploads.

This module provides S3 operations for the forge lakehouse. It's designed to:
1. Work with MinIO (S3-compatible) on the local NAS
2. Fall back gracefully when S3 is not configured
3. Follow the raw zone path conventions

Path Convention (raw zone):
    s3://forge/lake/raw/koe/corpora/corpus={corpus}/corpus_version={version}/...

Environment Variables:
    MINIO_ENDPOINT      - MinIO endpoint (default: http://localhost:9000)
    MINIO_ROOT_USER     - Access key
    MINIO_ROOT_PASSWORD - Secret key
    FORGE_LAKE_ROOT_S3  - Lake root for DuckDB (s3://forge/lake)

Security Notes:
    - Credentials are read from environment at runtime, never stored
    - Path components are validated before use (no traversal)
    - Checksums are verified after upload when possible

Usage:
    from modules.data_engineering.common.s3 import get_s3_client, upload_directory

    client = get_s3_client()
    if client:
        upload_directory(
            local_path=paths.ingest_extracted("jsut"),
            s3_prefix="lake/raw/koe/corpora/corpus=jsut/corpus_version=v1",
            bucket="forge",
        )
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

# Validation pattern for S3 path components.
# Explicitly ASCII-only to avoid Unicode surprises (Python \w matches Unicode by default).
# Allows: a-z, A-Z, 0-9, underscore, hyphen, dot, equals (for Hive-style partition keys).
# Note: This is intentionally broader than filters.py SAFE_IDENT to support corpus=jsut syntax.
SAFE_PATH_COMPONENT = re.compile(r"^[a-zA-Z0-9_\-\.=]+$")


class S3ConfigError(Exception):
    """Raised when S3 is not properly configured."""

    pass


class S3UploadError(Exception):
    """Raised when an upload fails."""

    pass


def validate_path_component(value: str, name: str = "component") -> str:
    """Validate a path component to prevent traversal attacks.

    Rejects:
    - Empty strings
    - .. (parent traversal)
    - Leading / or \\ (absolute paths)
    - Control characters (ord < 32)
    - Characters outside [a-zA-Z0-9_.-=]

    The = character is allowed for Hive-style partition keys (corpus=jsut).

    This function is part of the public API for use by other modules.
    """
    if not value:
        raise ValueError(f"Invalid {name}: empty string")
    if ".." in value:
        raise ValueError(f"Invalid {name}: contains '..'")
    if value.startswith("/") or value.startswith("\\"):
        raise ValueError(f"Invalid {name}: absolute path not allowed")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {name}: contains control characters")
    if not SAFE_PATH_COMPONENT.match(value):
        raise ValueError(f"Invalid {name}: {value!r} (allowed: [a-zA-Z0-9_.-=])")
    return value


def get_s3_config() -> dict | None:
    """Get S3/MinIO configuration from environment.

    Returns:
        Dict with endpoint, access_key, secret_key, or None if not configured.
    """
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    if not all([endpoint, access_key, secret_key]):
        return None

    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def get_s3_client() -> S3Client | None:
    """Get a boto3 S3 client configured for MinIO.

    Returns:
        boto3 S3 client or None if not configured.

    Note:
        Requires boto3 to be installed. Returns None if boto3 is not available.
    """
    config = get_s3_config()
    if config is None:
        logger.debug("S3 not configured (missing MINIO_* env vars)")
        return None

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.warning("boto3 not installed, S3 operations unavailable")
        return None

    # MinIO requires path-style addressing
    client = boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        config=Config(s3={"addressing_style": "path"}),
        region_name="us-east-1",  # MinIO ignores this but boto3 requires it
    )

    return client


def is_s3_available() -> bool:
    """Check if S3/MinIO is configured and reachable."""
    client = get_s3_client()
    if client is None:
        return False

    try:
        # Try to list buckets as a health check
        client.list_buckets()
        return True
    except Exception as e:
        logger.debug(f"S3 not reachable: {e}")
        return False


def ensure_bucket(bucket: str, client: S3Client | None = None) -> bool:
    """Ensure a bucket exists, creating it if necessary.

    Args:
        bucket: Bucket name
        client: Optional S3 client (uses get_s3_client() if not provided)

    Returns:
        True if bucket exists or was created, False otherwise.
    """
    if client is None:
        client = get_s3_client()
    if client is None:
        return False

    validate_path_component(bucket, "bucket")

    try:
        client.head_bucket(Bucket=bucket)
        return True
    except client.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            # Bucket doesn't exist, create it
            try:
                client.create_bucket(Bucket=bucket)
                logger.info(f"Created bucket: {bucket}")
                return True
            except Exception as create_err:
                logger.error(f"Failed to create bucket {bucket}: {create_err}")
                return False
        else:
            logger.error(f"Error checking bucket {bucket}: {e}")
            return False
    except Exception as e:
        logger.error(f"Error checking bucket {bucket}: {e}")
        return False


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def upload_file(
    local_path: Path,
    bucket: str,
    key: str,
    client: S3Client | None = None,
    compute_checksum: bool = True,
) -> dict:
    """Upload a single file to S3.

    Args:
        local_path: Local file path
        bucket: S3 bucket name
        key: S3 object key
        client: Optional S3 client
        compute_checksum: If True, compute and return SHA256

    Returns:
        Dict with upload metadata (key, size, checksum, etag)

    Raises:
        S3UploadError: If upload fails
        FileNotFoundError: If local file doesn't exist
        ValueError: If bucket or key contains invalid characters

    Note:
        Checksum is computed before upload. If the file changes between
        checksum computation and upload (TOCTOU), the recorded checksum
        won't match S3 contents. For critical integrity checks, verify
        against ETag or re-download and hash.
    """
    if client is None:
        client = get_s3_client()
    if client is None:
        raise S3ConfigError("S3 not configured")

    # Validate bucket name
    validate_path_component(bucket, "bucket")

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    # Validate key components
    for component in key.split("/"):
        if component:  # Skip empty components from leading/trailing slashes
            validate_path_component(component, "key component")

    size = local_path.stat().st_size
    checksum = compute_file_sha256(local_path) if compute_checksum else None

    try:
        client.upload_file(str(local_path), bucket, key)
        # Get ETag from a head request
        head = client.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag", "").strip('"')

        return {
            "key": key,
            "size": size,
            "checksum_sha256": checksum,
            "etag": etag,
            "bucket": bucket,
            "s3_uri": f"s3://{bucket}/{key}",
        }
    except Exception as e:
        raise S3UploadError(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}") from e


def collect_upload_files(
    local_path: Path,
    file_filter: str | None = None,
    max_files: int = 100_000,
) -> list[Path]:
    """Collect files for upload with security filtering.

    This function applies the same filtering logic used by upload_directory,
    making it suitable for dry-run previews that match real upload behavior.

    Args:
        local_path: Local directory path (will be resolved)
        file_filter: Optional glob pattern to filter files (e.g., "*.wav")
        max_files: Maximum files allowed (guard against wrong directory)

    Returns:
        List of resolved file paths that would be uploaded

    Raises:
        NotADirectoryError: If local_path is not a directory
        ValueError: If file count exceeds max_files

    Security:
        - Symlinks that escape local_path are skipped (prevents exfiltration)
        - File count is capped to prevent accidental large operations
    """
    local_path = Path(local_path).resolve()
    if not local_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {local_path}")

    # Collect candidates
    if file_filter:
        candidates = list(local_path.rglob(file_filter))
    else:
        candidates = [f for f in local_path.rglob("*") if f.is_file()]

    # Filter out symlinks that escape the directory (prevents exfiltration)
    files = []
    for file_path in candidates:
        resolved = file_path.resolve()
        if not resolved.is_relative_to(local_path):
            logger.warning(f"Skipping symlink escape: {file_path} -> {resolved}")
            continue
        files.append(file_path)

    # Guard against wrong directory
    if len(files) > max_files:
        raise ValueError(
            f"Too many files ({len(files):,}) in {local_path}. "
            f"Max is {max_files:,}. Use --force or increase max_files if intentional."
        )

    return files


def upload_directory(
    local_path: Path,
    bucket: str,
    s3_prefix: str,
    client: S3Client | None = None,
    file_filter: str | None = None,
    compute_checksums: bool = True,
    progress_callback: callable | None = None,
    max_files: int = 100_000,
) -> list[dict]:
    """Upload a directory to S3.

    Args:
        local_path: Local directory path
        bucket: S3 bucket name
        s3_prefix: S3 key prefix (e.g., "lake/raw/koe/corpora/corpus=jsut/corpus_version=v1")
        client: Optional S3 client
        file_filter: Optional glob pattern to filter files (e.g., "*.wav")
        compute_checksums: If True, compute SHA256 for each file
        progress_callback: Optional callback(current, total, filename) for progress
        max_files: Maximum files to upload (guard against runaway uploads)

    Returns:
        List of upload metadata dicts

    Raises:
        S3ConfigError: If S3 not configured
        S3UploadError: If any upload fails
        ValueError: If file count exceeds max_files

    Security:
        - Symlinks that escape local_path are skipped (prevents exfiltration via
          malicious corpus archives containing symlinks like evil -> /etc/passwd)
        - File count is capped to prevent accidental uploads of wrong directories
    """
    if client is None:
        client = get_s3_client()
    if client is None:
        raise S3ConfigError("S3 not configured")

    local_path = Path(local_path).resolve()

    # Ensure bucket exists
    if not ensure_bucket(bucket, client):
        raise S3ConfigError(f"Bucket {bucket} does not exist and could not be created")

    # Use shared collection logic (applies symlink filter + max_files guard)
    files = collect_upload_files(local_path, file_filter, max_files)

    results = []
    total = len(files)

    for i, file_path in enumerate(files):
        rel_path = file_path.relative_to(local_path)
        key = f"{s3_prefix.rstrip('/')}/{rel_path.as_posix()}"

        if progress_callback:
            progress_callback(i + 1, total, str(rel_path))

        result = upload_file(
            local_path=file_path,
            bucket=bucket,
            key=key,
            client=client,
            compute_checksum=compute_checksums,
        )
        results.append(result)

    return results


def build_raw_zone_prefix(
    corpus: str,
    corpus_version: str,
    domain: str = "koe",
) -> str:
    """Build the S3 prefix for raw zone uploads.

    Args:
        corpus: Corpus name (e.g., "jsut", "jvs")
        corpus_version: Version string (e.g., "v1", "2026-01")
        domain: Domain name (default: "koe")

    Returns:
        S3 prefix like "lake/raw/koe/corpora/corpus=jsut/corpus_version=v1"
    """
    # Validate all components
    validate_path_component(corpus, "corpus")
    validate_path_component(corpus_version, "corpus_version")
    validate_path_component(domain, "domain")

    return f"lake/raw/{domain}/corpora/corpus={corpus}/corpus_version={corpus_version}"
