"""
JVS corpus ingestion.

JVS: Japanese Versatile Speech corpus
https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

Dataset info:
- 100 professional speakers (voice actors)
- ~30 hours total
- 24 kHz sample rate (48 kHz available for commercial license)
- Per speaker: 100 parallel + 30 non-parallel + 10 whispered + 10 falsetto utterances
- Includes phoneme alignments (automatically generated) and speaker tags
  (gender, F0 range, speaker similarity, duration)

License:
- Text: Same as JSUT corpus
- Tags: CC BY-SA 4.0
- Audio: Academic research, non-commercial, personal use permitted
- Commercial: Contact TLO representatives

Citation:
    Takamichi, S., et al. (2019).
    "JVS corpus: free Japanese multi-speaker voice corpus"
    arXiv preprint 1908.06248

Key feature: Ground-truth phoneme alignments (unlike JSUT which has none)

Usage:
    koe ingest jvs

Outputs:
    data/ingest/jvs/raw/          - Original archive + MANIFEST.json
    data/ingest/jvs/extracted/    - Extracted audio + transcripts
    data/assets/jvs/              - License, readme files
"""

from .download import (
    JVS_GDRIVE_ID,
    JVS_SAMPLE_RATE,
    JVS_SOURCE_URL,
    JVS_SPEAKER_COUNT,
    JVS_SUBSETS,
    download_jvs,
    get_archive_path,
    get_raw_dir,
)
from .extract import (
    build_audio_checksums,
    extract_jvs,
    get_assets_dir,
    get_extracted_dir,
    get_jvs_root,
    iter_audio_files,
    iter_speakers,
)
from .manifest import (
    get_manifest_path,
    load_manifest,
    write_jvs_manifest,
)

__all__ = [
    # Constants
    "JVS_SOURCE_URL",
    "JVS_GDRIVE_ID",
    "JVS_SAMPLE_RATE",
    "JVS_SPEAKER_COUNT",
    "JVS_SUBSETS",
    # Download
    "download_jvs",
    "get_archive_path",
    "get_raw_dir",
    # Extract
    "extract_jvs",
    "build_audio_checksums",
    "get_extracted_dir",
    "get_assets_dir",
    "get_jvs_root",
    "iter_audio_files",
    "iter_speakers",
    # Manifest
    "write_jvs_manifest",
    "load_manifest",
    "get_manifest_path",
    # Main entry point
    "ingest_jvs",
]


def ingest_jvs(force: bool = False) -> dict:
    """
    Run full JVS ingest pipeline.

    This is the main entry point for `koe ingest jvs`.

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
    print("JVS Ingest Pipeline")
    print("=" * 60)

    # Step 1: Validate archive
    print("\n[1/3] Validating archive...")
    archive_metadata = download_jvs(force=force)
    print(f"  Archive: {archive_metadata['filename']}")
    print(f"  Size: {archive_metadata['size_bytes']:,} bytes")

    # Step 2: Extract and compute checksums
    print("\n[2/3] Extracting and computing audio checksums...")
    extraction_metadata = extract_jvs(force=force)
    print(f"  Extracted to: {extraction_metadata['extracted_dir']}")
    print(f"  Assets to: {extraction_metadata['assets_dir']}")

    # Step 3: Write manifest
    print("\n[3/3] Writing manifest...")
    manifest = write_jvs_manifest(force=force)

    print("\n" + "=" * 60)
    print("JVS ingest complete!")
    print(f"  Manifest: {get_manifest_path()}")
    print("=" * 60)

    return {
        "status": "success",
        "manifest": manifest,
        "archive": archive_metadata,
        "extraction": extraction_metadata,
    }
