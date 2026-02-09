"""
Extract JVS corpus from archive and compute audio checksums.

JVS directory structure:
    jvs_ver1/
    ├── jvs001/
    │   ├── parallel100/
    │   │   ├── wav24kHz16bit/
    │   │   │   └── VOICEACTRESS100_001.wav
    │   │   └── lab/
    │   │       └── mon/
    │   │           └── VOICEACTRESS100_001.lab
    │   ├── nonpara30/
    │   │   ├── wav24kHz16bit/
    │   │   └── lab/
    │   ├── whisper10/
    │   │   └── wav24kHz16bit/
    │   └── falset10/
    │       └── wav24kHz16bit/
    ├── jvs002/
    ...

Outputs:
- data/ingest/jvs/extracted/ - Extracted audio and metadata files
- data/ingest/jvs/extracted/audio_checksums.parquet - Audio file metadata
- data/assets/jvs/ - License and readme files
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import soundfile as sf

from modules.data_engineering.common.ids import make_file_checksum
from modules.data_engineering.common.paths import paths

from .download import JVS_ARCHIVE_NAME, JVS_SUBSETS, get_archive_path, get_raw_dir


def get_extracted_dir() -> Path:
    """Get the extracted directory for JVS."""
    return paths.ingest_extracted("jvs")


def get_assets_dir() -> Path:
    """Get the assets directory for JVS."""
    return paths.dataset_assets("jvs")


def is_extracted(extracted_dir: Path, archive_checksum: str) -> bool:
    """
    Check if extraction is complete and matches archive checksum.

    Args:
        extracted_dir: Path to extracted directory
        archive_checksum: Expected archive checksum

    Returns:
        True if already extracted with matching checksum
    """
    marker_file = extracted_dir / ".extracted"
    if not marker_file.exists():
        return False

    # Check if checksum matches
    stored_checksum = marker_file.read_text().strip()
    return stored_checksum == archive_checksum


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    archive_checksum: str,
) -> None:
    """
    Extract JVS archive to output directory.

    Args:
        archive_path: Path to zip archive
        output_dir: Directory to extract to
        archive_checksum: Archive checksum for marker file
    """
    from modules.forge.archive.zip import ZipHandler

    print(f"Extracting {archive_path.name} to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ZipHandler().extract(archive_path, output_dir)

    # Write marker file with checksum
    marker_file = output_dir / ".extracted"
    marker_file.write_text(archive_checksum)
    print(f"Extraction complete: {output_dir}")


def move_assets(extracted_dir: Path, assets_dir: Path) -> None:
    """
    Move license and readme files to assets directory.

    Args:
        extracted_dir: Path to extracted files
        assets_dir: Path to assets directory
    """
    assets_dir.mkdir(parents=True, exist_ok=True)

    jvs_root = extracted_dir / "jvs_ver1"
    if not jvs_root.exists():
        return

    # Look for readme/license files at root
    asset_patterns = ["README*", "LICENSE*", "*.txt", "*.md"]
    for pattern in asset_patterns:
        for f in jvs_root.glob(pattern):
            if f.is_file():
                dest = assets_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    print(f"Copied asset: {f.name}")


def get_jvs_root(extracted_dir: Path) -> Path:
    """
    Get the JVS root directory.

    Returns:
        Path to jvs_ver1 directory

    Raises:
        FileNotFoundError: If JVS root not found
    """
    jvs_root = extracted_dir / "jvs_ver1"
    if not jvs_root.exists():
        raise FileNotFoundError(f"JVS root not found: {jvs_root}")
    return jvs_root


def iter_speakers(extracted_dir: Path) -> Iterator[tuple[str, Path]]:
    """
    Iterate over JVS speakers.

    Yields:
        Tuple of (speaker_id, speaker_dir)
    """
    jvs_root = get_jvs_root(extracted_dir)

    for speaker_dir in sorted(jvs_root.iterdir()):
        if speaker_dir.is_dir() and speaker_dir.name.startswith("jvs"):
            yield speaker_dir.name, speaker_dir


def iter_audio_files(extracted_dir: Path) -> Iterator[Path]:
    """
    Iterate over all audio files in extracted directory.

    Yields:
        Path to each audio file
    """
    jvs_root = get_jvs_root(extracted_dir)

    for wav_file in sorted(jvs_root.rglob("*.wav")):
        yield wav_file


def compute_audio_metadata(audio_path: Path, data_root: Path) -> dict:
    """
    Compute metadata for a single audio file.

    Args:
        audio_path: Absolute path to audio file
        data_root: Data root for computing relative path

    Returns:
        Dict with audio metadata
    """
    # Get audio info
    info = sf.info(audio_path)

    # Compute relative path from data/ root
    try:
        audio_relpath = str(audio_path.relative_to(data_root / "data"))
    except ValueError:
        # Fallback if path structure differs
        audio_relpath = str(audio_path.relative_to(data_root))

    # Compute checksum
    checksum = make_file_checksum(audio_path, algorithm="sha256", truncate=64)

    return {
        "audio_relpath": audio_relpath,
        "audio_checksum": f"sha256:{checksum}",
        "audio_format": audio_path.suffix[1:],  # wav
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_sec": info.duration,
        "frames": info.frames,
    }


def build_audio_checksums(
    extracted_dir: Path,
    output_path: Path = None,
    force: bool = False,
) -> Path:
    """
    Build audio checksums parquet file.

    Args:
        extracted_dir: Path to extracted JVS
        output_path: Path to output parquet file
        force: If True, rebuild even if exists

    Returns:
        Path to checksums file
    """
    if output_path is None:
        output_path = extracted_dir / "audio_checksums.parquet"

    if output_path.exists() and not force:
        print(f"Audio checksums already exist: {output_path}")
        return output_path

    print("Computing audio checksums (this may take a while for JVS ~15k files)...")

    # Collect metadata for all audio files
    records = []
    data_root = paths.data_root

    for i, audio_path in enumerate(iter_audio_files(extracted_dir)):
        if i % 500 == 0:
            print(f"  Processing file {i}...")

        try:
            metadata = compute_audio_metadata(audio_path, data_root)
            records.append(metadata)
        except Exception as e:
            print(f"  Warning: Failed to process {audio_path.name}: {e}")

    print(f"Processed {len(records)} audio files")

    # Write to parquet
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Audio checksums saved: {output_path}")

    return output_path


def extract_jvs(force: bool = False) -> dict:
    """
    Extract JVS corpus and compute audio checksums.

    This function:
    1. Extracts archive if not already extracted
    2. Moves assets (license, readme) to assets directory
    3. Computes audio checksums

    Args:
        force: If True, re-extract and recompute even if exists

    Returns:
        Dict with extraction metadata
    """
    from .download import download_jvs

    # Get archive metadata (validates archive exists)
    archive_metadata = download_jvs(force=False)
    archive_checksum = archive_metadata["source_archive_checksum"]

    archive_path = get_archive_path()
    extracted_dir = get_extracted_dir()
    assets_dir = get_assets_dir()

    # Extract if needed
    if force or not is_extracted(extracted_dir, archive_checksum):
        extract_archive(archive_path, extracted_dir, archive_checksum)
    else:
        print(f"JVS already extracted: {extracted_dir}")

    # Move assets
    move_assets(extracted_dir, assets_dir)

    # Build audio checksums
    checksums_path = build_audio_checksums(extracted_dir, force=force)

    return {
        "extracted_dir": str(extracted_dir),
        "assets_dir": str(assets_dir),
        "checksums_path": str(checksums_path),
        "archive_checksum": archive_checksum,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
