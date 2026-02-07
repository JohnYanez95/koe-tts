"""
Download JVS corpus.

JVS: Japanese Versatile Speech corpus
https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

Dataset info:
- 100 professional speakers (voice actors)
- ~30 hours total
- 24 kHz sample rate (48 kHz available for commercial license)
- Per speaker: 100 parallel + 30 non-parallel + 10 whispered + 10 falsetto
- Includes phoneme alignments (automatically generated)

License:
- Text: Same as JSUT corpus
- Tags: CC BY-SA 4.0
- Audio: Academic research, non-commercial, personal use permitted
- Commercial: Contact TLO representatives

Note: JVS requires manual download due to Google Drive hosting.
This module validates the download and computes checksums.
"""

from datetime import datetime, timezone
from pathlib import Path

from modules.data_engineering.common.ids import make_file_checksum
from modules.data_engineering.common.paths import paths

# JVS download info
JVS_SOURCE_URL = "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"
JVS_GDRIVE_ID = "19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"
JVS_ARCHIVE_NAME = "jvs_ver1.zip"
JVS_EXPECTED_SIZE_BYTES = 3_500_000_000  # ~3.5GB

# Dataset metadata
JVS_SAMPLE_RATE = 24000
JVS_HOURS = 30
JVS_SPEAKER_COUNT = 100
JVS_SUBSETS = ["parallel100", "nonpara30", "whisper10", "falset10"]


def get_raw_dir() -> Path:
    """Get the raw download directory for JVS."""
    return paths.ingest_raw("jvs")


def get_archive_path() -> Path:
    """Get the expected archive path."""
    return get_raw_dir() / JVS_ARCHIVE_NAME


def validate_archive(archive_path: Path) -> dict:
    """
    Validate JVS archive exists and compute checksum.

    Args:
        archive_path: Path to archive file

    Returns:
        Dict with archive metadata (size, checksum, etc.)

    Raises:
        FileNotFoundError: If archive not found with download instructions
    """
    if not archive_path.exists():
        raise FileNotFoundError(
            f"JVS archive not found at {archive_path}\n\n"
            "Please download manually:\n"
            f"1. Visit {JVS_SOURCE_URL}\n"
            "2. Download jvs_ver1.zip from Google Drive\n"
            f"   (Google Drive ID: {JVS_GDRIVE_ID})\n"
            f"3. Place in {archive_path.parent}/{JVS_ARCHIVE_NAME}\n"
        )

    # Get file stats
    stat = archive_path.stat()
    size_bytes = stat.st_size

    # Compute checksum (this takes a while for 3.5GB file)
    print(f"Computing checksum for {archive_path.name}...")
    checksum = make_file_checksum(archive_path, algorithm="sha256")

    return {
        "filename": archive_path.name,
        "source_url": JVS_SOURCE_URL,
        "size_bytes": size_bytes,
        "source_archive_checksum": f"sha256:{checksum}",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def download_jvs(force: bool = False) -> dict:
    """
    Validate JVS corpus archive (manual download required).

    This function:
    1. Ensures raw directory exists
    2. Checks if archive is present
    3. Computes and returns archive metadata

    Args:
        force: If True, recompute checksum even if already validated

    Returns:
        Dict with archive metadata for MANIFEST.json

    Raises:
        FileNotFoundError: If archive not found
    """
    raw_dir = get_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_path = get_archive_path()

    # Check for existing validation
    checksum_file = raw_dir / f"{JVS_ARCHIVE_NAME}.sha256"
    if checksum_file.exists() and not force:
        existing_checksum = checksum_file.read_text().strip()
        stat = archive_path.stat()
        print(f"JVS archive already validated: {archive_path.name}")
        return {
            "filename": archive_path.name,
            "source_url": JVS_SOURCE_URL,
            "size_bytes": stat.st_size,
            "source_archive_checksum": existing_checksum,
            "validated_at": checksum_file.stat().st_mtime,
        }

    # Validate and compute checksum
    metadata = validate_archive(archive_path)

    # Save checksum for future runs
    checksum_file.write_text(metadata["source_archive_checksum"])
    print(f"Checksum saved: {checksum_file}")

    return metadata
