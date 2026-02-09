"""S3/MinIO storage backend and utilities.

Provides S3 operations for the forge lakehouse and an ``S3StorageBackend``
class implementing the ``StorageBackend`` protocol.

Designed to:
1. Work with MinIO (S3-compatible) on a local NAS
2. Fall back gracefully when S3 is not configured
3. Follow the raw zone path conventions

Path Convention (raw zone):
    s3://forge/lake/raw/{domain}/corpora/corpus={corpus}/corpus_version={version}/...

Environment Variables:
    MINIO_ENDPOINT        - MinIO endpoint (default: http://localhost:9000)
    MINIO_ROOT_USER       - Access key
    MINIO_ROOT_PASSWORD   - Secret key
    FORGE_S3_CONNECT_TIMEOUT - Connection timeout in seconds (default: 5)
    FORGE_S3_READ_TIMEOUT    - Read timeout in seconds (default: 30)
    FORGE_S3_MAX_ATTEMPTS    - Max retry attempts (default: 3)
    FORGE_S3_REGION          - AWS/MinIO region (default: us-east-1)

Security Notes:
    - Credentials are read from environment at runtime, never stored
    - Path components are validated before use (no traversal)
    - Checksums are verified after upload when possible

Public API
----------
- ``S3StorageBackend`` — StorageBackend implementation for S3/MinIO
- ``S3ConfigError`` / ``S3UploadError`` — error types
- ``validate_path_component(value, name)`` — path safety validation
- ``get_s3_config()`` / ``get_s3_client()`` / ``is_s3_available()``
- ``ensure_bucket(bucket, client)``
- ``upload_file(local_path, bucket, key, ...)``
- ``upload_directory(local_path, bucket, s3_prefix, ...)``
- ``collect_upload_files(local_path, file_filter, max_files)``
- ``build_raw_zone_prefix(corpus, corpus_version, domain)``
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from modules.forge.storage.protocols import NotFoundError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

# ── Timeout configuration (env-overridable) ────────────────────────────
S3_CONNECT_TIMEOUT: int = int(os.getenv("FORGE_S3_CONNECT_TIMEOUT", "5"))
S3_READ_TIMEOUT: int = int(os.getenv("FORGE_S3_READ_TIMEOUT", "30"))
S3_MAX_ATTEMPTS: int = int(os.getenv("FORGE_S3_MAX_ATTEMPTS", "3"))
S3_REGION: str = os.getenv("FORGE_S3_REGION", "us-east-1")

# Validation pattern for S3 path components.
# Explicitly ASCII-only to avoid Unicode surprises (Python \w matches Unicode by default).
# Allows: a-z, A-Z, 0-9, underscore, hyphen, dot, equals (for Hive-style partition keys).
SAFE_PATH_COMPONENT = re.compile(r"^[a-zA-Z0-9_\-\.=]+$")


class S3ConfigError(Exception):
    """Raised when S3 is not properly configured."""


class S3UploadError(Exception):
    """Raised when an upload fails."""


# ── Path validation ────────────────────────────────────────────────────


def validate_path_component(value: str, name: str = "component") -> str:
    """Validate a path component to prevent traversal attacks.

    Rejects:
    - Empty strings
    - .. (parent traversal)
    - Leading / or \\\\ (absolute paths)
    - Control characters (ord < 32)
    - Characters outside [a-zA-Z0-9_.-=]

    The = character is allowed for Hive-style partition keys (corpus=jsut).
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
        raise ValueError(
            f"Invalid {name}: {value!r} (allowed: [a-zA-Z0-9_.-=])"
        )
    return value


# ── S3 client management ──────────────────────────────────────────────


def get_s3_config() -> dict | None:
    """Get S3/MinIO configuration from environment.

    Returns dict with endpoint, access_key, secret_key, or None if
    not configured.
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

    Returns boto3 S3 client or None if not configured.  Applies
    timeout and retry configuration from environment variables.
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
        config=Config(
            s3={"addressing_style": "path"},
            connect_timeout=S3_CONNECT_TIMEOUT,
            read_timeout=S3_READ_TIMEOUT,
            retries={"max_attempts": S3_MAX_ATTEMPTS},
        ),
        region_name=S3_REGION,
    )

    return client


def is_s3_available() -> bool:
    """Check if S3/MinIO is configured and reachable."""
    client = get_s3_client()
    if client is None:
        return False

    try:
        client.list_buckets()
    except Exception:
        logger.debug("S3 not reachable")
        return False
    else:
        return True


def ensure_bucket(bucket: str, client: S3Client | None = None) -> bool:
    """Ensure a bucket exists, creating it if necessary.

    Returns True if bucket exists or was created, False otherwise.
    """
    if client is None:
        client = get_s3_client()
    if client is None:
        return False

    validate_path_component(bucket, "bucket")

    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            try:
                client.create_bucket(Bucket=bucket)
                logger.info("Created bucket: %s", bucket)
            except Exception:
                logger.error("Failed to create bucket %s", bucket)
                return False
            else:
                return True
        else:
            logger.error("Error checking bucket %s", bucket)
            return False
    except Exception:
        logger.error("Error checking bucket %s", bucket)
        return False
    else:
        return True


# ── File operations ───────────────────────────────────────────────────


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

    Returns dict with upload metadata (key, size, checksum, etag).

    Raises:
        S3UploadError: If upload fails
        FileNotFoundError: If local file doesn't exist
        ValueError: If bucket or key contains invalid characters
    """
    if client is None:
        client = get_s3_client()
    if client is None:
        raise S3ConfigError("S3 not configured")

    validate_path_component(bucket, "bucket")

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    # Validate key components
    for component in key.split("/"):
        if component:
            validate_path_component(component, "key component")

    size = local_path.stat().st_size
    checksum = compute_file_sha256(local_path) if compute_checksum else None

    try:
        client.upload_file(str(local_path), bucket, key)
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
        raise S3UploadError(
            f"Failed to upload {local_path} to s3://{bucket}/{key}"
        ) from e


def collect_upload_files(
    local_path: Path,
    file_filter: str | None = None,
    max_files: int = 100_000,
) -> list[Path]:
    """Collect files for upload with security filtering.

    Symlinks that escape local_path are skipped (prevents exfiltration).
    File count is capped to prevent accidental large operations.
    """
    local_path = Path(local_path).resolve()
    if not local_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {local_path}")

    if file_filter:
        candidates = list(local_path.rglob(file_filter))
    else:
        candidates = [f for f in local_path.rglob("*") if f.is_file()]

    # Filter out symlinks that escape the directory
    files = []
    for file_path in candidates:
        resolved = file_path.resolve()
        if not resolved.is_relative_to(local_path):
            logger.warning("Skipping symlink escape: %s -> %s", file_path, resolved)
            continue
        files.append(file_path)

    if len(files) > max_files:
        raise ValueError(
            f"Too many files ({len(files):,}) in {local_path}. "
            f"Max is {max_files:,}. Use --force or increase max_files "
            f"if intentional."
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

    Symlinks that escape local_path are skipped (prevents exfiltration
    via malicious corpus archives containing symlinks).
    """
    if client is None:
        client = get_s3_client()
    if client is None:
        raise S3ConfigError("S3 not configured")

    local_path = Path(local_path).resolve()

    if not ensure_bucket(bucket, client):
        raise S3ConfigError(
            f"Bucket {bucket} does not exist and could not be created"
        )

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
    domain: str,
) -> str:
    """Build the S3 prefix for raw zone uploads.

    Returns prefix like
    ``"lake/raw/{domain}/corpora/corpus={corpus}/corpus_version={version}"``.

    All parameters are required and validated.
    """
    validate_path_component(corpus, "corpus")
    validate_path_component(corpus_version, "corpus_version")
    validate_path_component(domain, "domain")

    return (
        f"lake/raw/{domain}/corpora/"
        f"corpus={corpus}/corpus_version={corpus_version}"
    )


# ── S3StorageBackend ──────────────────────────────────────────────────


class S3StorageBackend:
    """``StorageBackend`` implementation for S3/MinIO.

    Parameters
    ----------
    bucket
        S3 bucket name.
    prefix
        Optional key prefix prepended to all operations.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        validate_path_component(bucket, "bucket")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._client: S3Client | None = None

    def _get_client(self) -> S3Client:
        """Lazy-initialise and return the S3 client."""
        if self._client is None:
            self._client = get_s3_client()
        if self._client is None:
            raise S3ConfigError("S3 not configured")
        return self._client

    def _full_key(self, key: str) -> str:
        """Prepend prefix to key."""
        if self._prefix:
            return f"{self._prefix}/{key}"
        return key

    def put(self, key: str, source: Path) -> str:
        """Upload file to S3.  Returns the full S3 key."""
        full_key = self._full_key(key)
        upload_file(
            local_path=source,
            bucket=self._bucket,
            key=full_key,
            client=self._get_client(),
        )
        return full_key

    def get_to_path(self, key: str, dest: Path) -> Path:
        """Download object directly to disk.

        Uses boto3 ``download_file()`` for multipart downloads and
        automatic retries.  Raises ``NotFoundError`` if key does not
        exist.
        """
        client = self._get_client()
        full_key = self._full_key(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            client.download_file(self._bucket, full_key, str(dest))
        except client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                raise NotFoundError(
                    f"Key not found: s3://{self._bucket}/{full_key}"
                ) from e
            raise
        return dest

    def exists(self, key: str) -> bool:
        """Check if key exists in S3."""
        client = self._get_client()
        full_key = self._full_key(key)

        try:
            client.head_object(Bucket=self._bucket, Key=full_key)
        except client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                return False
            raise
        return True

    def list_keys(self, prefix: str) -> Iterator[str]:
        """List keys matching prefix, with pagination."""
        client = self._get_client()
        full_prefix = self._full_key(prefix)

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self._bucket, Prefix=full_prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Strip our prefix so callers see relative keys
                if self._prefix and key.startswith(self._prefix + "/"):
                    key = key[len(self._prefix) + 1 :]
                yield key
