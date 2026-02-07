"""
JSUT ingest manifest management.

The manifest records what was ingested and provides provenance
information for the bronze layer.

Output: data/ingest/jsut/raw/MANIFEST.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.data_engineering.common.paths import paths

from .download import JSUT_ARCHIVE_NAME, JSUT_SOURCE_URL, get_raw_dir

# Ingest pipeline version - bump when logic changes
INGEST_VERSION = "2025-01-24"


def get_manifest_path() -> Path:
    """Get path to MANIFEST.json."""
    return get_raw_dir() / "MANIFEST.json"


def load_manifest() -> Optional[dict]:
    """
    Load existing manifest if present.

    Returns:
        Manifest dict or None if not found
    """
    manifest_path = get_manifest_path()
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def save_manifest(manifest: dict) -> Path:
    """
    Save manifest to MANIFEST.json.

    Args:
        manifest: Manifest dict

    Returns:
        Path to saved manifest
    """
    manifest_path = get_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest saved: {manifest_path}")
    return manifest_path


def build_manifest(
    archive_metadata: dict,
    extraction_metadata: dict,
) -> dict:
    """
    Build MANIFEST.json from archive and extraction metadata.

    Args:
        archive_metadata: From download_jsut()
        extraction_metadata: From extract_jsut()

    Returns:
        Complete manifest dict
    """
    return {
        "dataset": "jsut",
        "ingest_version": INGEST_VERSION,
        "source_version": "v1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archives": [
            {
                "filename": archive_metadata["filename"],
                "source_url": archive_metadata["source_url"],
                "size_bytes": archive_metadata["size_bytes"],
                "source_archive_checksum": archive_metadata["source_archive_checksum"],
                "validated_at": archive_metadata.get("validated_at"),
            }
        ],
        "extraction": {
            "extracted_dir": extraction_metadata["extracted_dir"],
            "assets_dir": extraction_metadata["assets_dir"],
            "checksums_path": extraction_metadata["checksums_path"],
            "extracted_at": extraction_metadata["extracted_at"],
        },
        "paths": {
            "raw": str(get_raw_dir()),
            "extracted": str(paths.ingest_extracted("jsut")),
            "assets": str(paths.dataset_assets("jsut")),
        },
    }


def write_jsut_manifest(force: bool = False) -> dict:
    """
    Build and write JSUT ingest manifest.

    This is the main entry point for manifest generation.
    Reads metadata from download and extract steps.

    Args:
        force: If True, rebuild manifest even if exists

    Returns:
        Complete manifest dict
    """
    from .download import download_jsut
    from .extract import extract_jsut

    manifest_path = get_manifest_path()

    # Check for existing manifest
    if manifest_path.exists() and not force:
        existing = load_manifest()
        if existing and existing.get("ingest_version") == INGEST_VERSION:
            print(f"Manifest already up-to-date: {manifest_path}")
            return existing

    # Get metadata from download and extract
    print("Building JSUT ingest manifest...")
    archive_metadata = download_jsut(force=False)
    extraction_metadata = extract_jsut(force=False)

    # Build and save manifest
    manifest = build_manifest(archive_metadata, extraction_metadata)
    save_manifest(manifest)

    return manifest


# Legacy function for backwards compatibility
def build_jsut_manifest(
    output_path: Optional[Path] = None,
    extracted_dir: Optional[Path] = None,
    version: str = "v1.1",
) -> Path:
    """
    Deprecated: Use write_jsut_manifest() instead.

    This function now delegates to write_jsut_manifest().
    """
    manifest = write_jsut_manifest()
    return get_manifest_path()
