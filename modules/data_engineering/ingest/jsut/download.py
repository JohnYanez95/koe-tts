"""
Download JSUT corpus.

JSUT: Japanese speech corpus of Saruwatari-lab., University of Tokyo
https://sites.google.com/site/shinnosuketakamichi/publication/jsut

Dataset info:
- Single native Japanese female speaker
- ~10 hours of speech, ~7,700 utterances
- 48 kHz sample rate, recorded in anechoic chamber
- Subsets: basic5000, utparaphrase512, onomatopee300, countersuffix26,
           loanword128, voiceactress100, travel1000, precedent130, repeat500

License:
- Text: CC-BY-SA 4.0
- Audio: Academic research, non-commercial, personal use permitted

Citation:
    Sonobe, R., Takamichi, S., & Saruwatari, H. (2017).
    "JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech synthesis"
    arXiv preprint 1711.00354

Note: JSUT requires manual download due to license agreement.
This module validates the download and computes checksums.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.data_engineering.common.ids import make_file_checksum
from modules.data_engineering.common.paths import paths

# JSUT download info
JSUT_SOURCE_URL = "https://sites.google.com/site/shinnosuketakamichi/publication/jsut"
JSUT_DOWNLOAD_URL = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
JSUT_ARCHIVE_NAME = "jsut_ver1.1.zip"
JSUT_EXPECTED_SIZE_BYTES = 2_900_000_000  # ~2.7GB

# Dataset metadata
JSUT_SAMPLE_RATE = 48000
JSUT_HOURS = 10
JSUT_SPEAKER_COUNT = 1
JSUT_UTTERANCE_COUNT_APPROX = 7700


def get_raw_dir() -> Path:
    """Get the raw download directory for JSUT."""
    return paths.ingest_raw("jsut")


def get_archive_path() -> Path:
    """Get the expected archive path."""
    return get_raw_dir() / JSUT_ARCHIVE_NAME


def validate_archive(archive_path: Path) -> dict:
    """
    Validate JSUT archive exists and compute checksum.

    Args:
        archive_path: Path to archive file

    Returns:
        Dict with archive metadata (size, checksum, etc.)

    Raises:
        FileNotFoundError: If archive not found with download instructions
    """
    if not archive_path.exists():
        raise FileNotFoundError(
            f"JSUT archive not found at {archive_path}\n\n"
            "Please download manually:\n"
            f"1. Visit {JSUT_SOURCE_URL}\n"
            "2. Accept the license agreement\n"
            f"3. Download from: {JSUT_DOWNLOAD_URL}\n"
            f"4. Place in {archive_path.parent}/{JSUT_ARCHIVE_NAME}\n"
        )

    # Get file stats
    stat = archive_path.stat()
    size_bytes = stat.st_size

    # Compute checksum (this takes a while for 2GB file)
    print(f"Computing checksum for {archive_path.name}...")
    checksum = make_file_checksum(archive_path, algorithm="sha256")

    return {
        "filename": archive_path.name,
        "source_url": JSUT_SOURCE_URL,
        "size_bytes": size_bytes,
        "source_archive_checksum": f"sha256:{checksum}",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def download_jsut(force: bool = False) -> dict:
    """
    Validate JSUT corpus archive (manual download required).

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
    checksum_file = raw_dir / f"{JSUT_ARCHIVE_NAME}.sha256"
    if checksum_file.exists() and not force:
        existing_checksum = checksum_file.read_text().strip()
        stat = archive_path.stat()
        print(f"JSUT archive already validated: {archive_path.name}")
        return {
            "filename": archive_path.name,
            "source_url": JSUT_SOURCE_URL,
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
