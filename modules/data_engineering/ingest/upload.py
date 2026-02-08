"""Upload ingested data to S3 raw zone.

This module handles uploading extracted corpus data to the S3 raw zone
for durable storage. The raw zone is the source of truth for immutable
corpus blobs.

Path Convention:
    s3://forge/lake/raw/koe/corpora/corpus={corpus}/corpus_version={version}/
        wav/              - Audio files
        transcripts/      - Text files
        metadata/         - Speaker info, etc.

Usage:
    from modules.data_engineering.ingest.upload import upload_to_raw_zone

    result = upload_to_raw_zone(
        dataset="jsut",
        version="v1",
        force=False,
    )

The upload is idempotent - re-running will skip files that already exist
with matching checksums.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.s3 import (
    S3ConfigError,
    S3UploadError,
    build_raw_zone_prefix,
    collect_upload_files,
    get_s3_client,
    is_s3_available,
    upload_directory,
    validate_path_component,
)

logger = logging.getLogger(__name__)

# Default bucket name
DEFAULT_BUCKET = "forge"


def get_upload_manifest_path(dataset: str) -> Path:
    """Get path to upload manifest (tracks what's been uploaded)."""
    return paths.ingest_raw(dataset) / "UPLOAD_MANIFEST.json"


def load_upload_manifest(dataset: str) -> dict | None:
    """Load existing upload manifest if present."""
    manifest_path = get_upload_manifest_path(dataset)
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def save_upload_manifest(dataset: str, manifest: dict) -> Path:
    """Save upload manifest."""
    manifest_path = get_upload_manifest_path(dataset)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def upload_to_raw_zone(
    dataset: str,
    version: str = "v1",
    bucket: str = DEFAULT_BUCKET,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Upload extracted corpus data to S3 raw zone.

    Args:
        dataset: Dataset name (e.g., "jsut", "jvs")
        version: Corpus version (e.g., "v1")
        bucket: S3 bucket name
        force: If True, re-upload even if manifest shows complete
        dry_run: If True, show what would be uploaded without uploading

    Returns:
        Dict with upload results:
            status: "success", "skipped", "unavailable", or "error"
            files_uploaded: Number of files uploaded
            s3_prefix: S3 prefix where files were uploaded
            manifest_path: Path to upload manifest

    Raises:
        FileNotFoundError: If extracted directory doesn't exist
        ValueError: If dataset, version, or bucket contains invalid characters
    """
    # Validate inputs BEFORE touching filesystem (prevents path probing attacks)
    try:
        validate_path_component(dataset, "dataset")
        validate_path_component(version, "version")
        validate_path_component(bucket, "bucket")
    except ValueError as e:
        return {
            "status": "error",
            "message": str(e),
            "files_uploaded": 0,
        }

    extracted_dir = paths.ingest_extracted(dataset)
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Extracted directory not found: {extracted_dir}\n"
            f"Run 'koe ingest {dataset}' first."
        )

    # Check if S3 is available
    if not is_s3_available():
        logger.info("S3 not available, skipping raw zone upload")
        return {
            "status": "unavailable",
            "message": "S3 not configured or not reachable",
            "files_uploaded": 0,
        }

    # Check existing manifest
    existing_manifest = load_upload_manifest(dataset)
    if existing_manifest and not force:
        if existing_manifest.get("status") == "complete":
            if existing_manifest.get("version") == version:
                logger.info(f"Raw zone upload already complete for {dataset} {version}")
                return {
                    "status": "skipped",
                    "message": "Already uploaded",
                    "files_uploaded": 0,
                    "s3_prefix": existing_manifest.get("s3_prefix"),
                    "manifest_path": str(get_upload_manifest_path(dataset)),
                }

    # Build S3 prefix
    s3_prefix = build_raw_zone_prefix(corpus=dataset, corpus_version=version)

    if dry_run:
        # Use same file collection logic as real upload (symlink filter + max_files)
        try:
            files = collect_upload_files(extracted_dir)
        except ValueError as e:
            # max_files exceeded
            return {
                "status": "error",
                "message": str(e),
                "files_to_upload": 0,
            }
        print(f"\n[Dry Run] Would upload {len(files)} files to:")
        print(f"  s3://{bucket}/{s3_prefix}/")
        print(f"\nSource: {extracted_dir}")
        return {
            "status": "dry_run",
            "files_to_upload": len(files),
            "s3_prefix": s3_prefix,
        }

    # Get S3 client
    client = get_s3_client()
    if client is None:
        return {
            "status": "error",
            "message": "Failed to get S3 client",
            "files_uploaded": 0,
        }

    # Progress callback
    def progress(current: int, total: int, filename: str) -> None:
        if current % 100 == 0 or current == total:
            print(f"  Uploading: {current}/{total} - {filename}")

    print(f"\nUploading {dataset} to raw zone...")
    print(f"  Source: {extracted_dir}")
    print(f"  Destination: s3://{bucket}/{s3_prefix}/")

    try:
        results = upload_directory(
            local_path=extracted_dir,
            bucket=bucket,
            s3_prefix=s3_prefix,
            client=client,
            compute_checksums=True,
            progress_callback=progress,
        )
    except (S3ConfigError, S3UploadError) as e:
        logger.error(f"Upload failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "files_uploaded": 0,
        }

    # Build and save manifest
    manifest = {
        "status": "complete",
        "dataset": dataset,
        "version": version,
        "bucket": bucket,
        "s3_prefix": s3_prefix,
        "files_uploaded": len(results),
        "uploaded_at": datetime.now(UTC).isoformat(),
        "files": results,
    }
    manifest_path = save_upload_manifest(dataset, manifest)

    print(f"\n  Uploaded {len(results)} files")
    print(f"  Manifest: {manifest_path}")

    return {
        "status": "success",
        "files_uploaded": len(results),
        "s3_prefix": s3_prefix,
        "s3_uri": f"s3://{bucket}/{s3_prefix}",
        "manifest_path": str(manifest_path),
    }
