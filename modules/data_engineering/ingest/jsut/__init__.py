"""
JSUT corpus ingestion.

Usage:
    koe ingest jsut

Outputs:
    data/ingest/jsut/raw/          - Original archive + MANIFEST.json
    data/ingest/jsut/extracted/    - Extracted audio + transcripts
    data/assets/jsut/              - License, readme files

The ingest process is idempotent - re-running will skip completed steps.
"""

from .download import download_jsut, get_archive_path, get_raw_dir
from .extract import (
    build_audio_checksums,
    extract_jsut,
    get_assets_dir,
    get_extracted_dir,
)
from .manifest import (
    build_jsut_manifest,
    get_manifest_path,
    load_manifest,
    write_jsut_manifest,
)

__all__ = [
    # Download
    "download_jsut",
    "get_archive_path",
    "get_raw_dir",
    # Extract
    "extract_jsut",
    "build_audio_checksums",
    "get_extracted_dir",
    "get_assets_dir",
    # Manifest
    "write_jsut_manifest",
    "load_manifest",
    "get_manifest_path",
    "build_jsut_manifest",  # Legacy
    # Main entry point
    "ingest_jsut",
]


def ingest_jsut(force: bool = False) -> dict:
    """
    Run full JSUT ingest pipeline.

    This is the main entry point for `koe ingest jsut`.

    Steps:
    1. Validate archive exists (manual download required)
    2. Extract archive to extracted directory
    3. Move assets to assets directory
    4. Compute audio checksums
    5. Write MANIFEST.json

    Args:
        force: If True, re-run all steps even if already complete

    Returns:
        Dict with ingest results including manifest
    """
    print("=" * 60)
    print("JSUT Ingest Pipeline")
    print("=" * 60)

    # Step 1: Validate archive
    print("\n[1/3] Validating archive...")
    archive_metadata = download_jsut(force=force)
    print(f"  Archive: {archive_metadata['filename']}")
    print(f"  Size: {archive_metadata['size_bytes']:,} bytes")

    # Step 2: Extract and compute checksums
    print("\n[2/3] Extracting and computing audio checksums...")
    extraction_metadata = extract_jsut(force=force)
    print(f"  Extracted to: {extraction_metadata['extracted_dir']}")
    print(f"  Assets to: {extraction_metadata['assets_dir']}")

    # Step 3: Write manifest
    print("\n[3/3] Writing manifest...")
    manifest = write_jsut_manifest(force=force)

    print("\n" + "=" * 60)
    print("JSUT ingest complete!")
    print(f"  Manifest: {get_manifest_path()}")
    print("=" * 60)

    return {
        "status": "success",
        "manifest": manifest,
        "archive": archive_metadata,
        "extraction": extraction_metadata,
    }
