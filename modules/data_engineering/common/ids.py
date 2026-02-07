"""
Utterance ID generation for stable, unique identifiers.

ID Strategy:
- utterance_id: Hash-based primary key (16 hex chars)
  - Stable across machines and time
  - Safe for multi-corpus joins
  - Formula: sha1("{dataset}|{speaker_id}|{subset}|{corpus_utt_id}")[:16]

- utterance_key: Human-readable key for debugging
  - Format: {dataset}_{subset}_{corpus_utt_id} or {dataset}_{speaker}_{subset}_{idx:05d}
  - Makes debugging and inspection 10x easier

Examples:
    utterance_id:  "a3f2b8c1d4e5f678"
    utterance_key: "jsut_basic5000_BASIC5000_0001"
    utterance_key: "jvs_jvs042_parallel100_00023" (fallback with index)
"""

import hashlib
from pathlib import Path
from typing import Optional


def make_utterance_id(
    dataset: str,
    speaker_id: str,
    subset: str,
    corpus_utt_id: Optional[str] = None,
    audio_relpath: Optional[str] = None,
) -> str:
    """
    Generate a stable hash-based utterance ID (primary key).

    Uses corpus_utt_id if available, otherwise falls back to audio_relpath.
    At least one of corpus_utt_id or audio_relpath must be provided.

    Args:
        dataset: Dataset name (jsut, jvs, common_voice)
        speaker_id: Speaker identifier
        subset: Subset name (basic5000, parallel100, etc.)
        corpus_utt_id: Original corpus utterance ID (preferred)
        audio_relpath: Relative audio path (fallback)

    Returns:
        16-character hex hash string
    """
    # Use corpus_utt_id if available, otherwise audio_relpath
    unique_part = corpus_utt_id or audio_relpath
    if not unique_part:
        raise ValueError("Either corpus_utt_id or audio_relpath must be provided")

    # Build deterministic input string
    input_str = f"{dataset}|{speaker_id}|{subset}|{unique_part}"

    # SHA1 hash, take first 16 hex chars (64 bits, ~2^64 collision resistance)
    return hashlib.sha1(input_str.encode("utf-8")).hexdigest()[:16]


def make_utterance_key(
    dataset: str,
    subset: str,
    corpus_utt_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    index: Optional[int] = None,
) -> str:
    """
    Generate a human-readable utterance key for debugging.

    Prefers corpus_utt_id format, falls back to speaker+index format.

    Args:
        dataset: Dataset name
        subset: Subset name
        corpus_utt_id: Original corpus utterance ID (preferred)
        speaker_id: Speaker identifier (for fallback)
        index: Utterance index within subset (for fallback)

    Returns:
        Human-readable key string

    Examples:
        "jsut_basic5000_BASIC5000_0001"
        "jvs_parallel100_jvs042_00023"
    """
    # Normalize dataset name
    ds = _normalize_dataset(dataset)

    if corpus_utt_id:
        return f"{ds}_{subset}_{corpus_utt_id}"
    elif speaker_id is not None and index is not None:
        return f"{ds}_{subset}_{speaker_id}_{index:05d}"
    else:
        raise ValueError("Either corpus_utt_id or (speaker_id + index) must be provided")


def _normalize_dataset(dataset: str) -> str:
    """Normalize dataset name for IDs."""
    ds = dataset.lower().replace("_", "").replace("-", "")
    # Abbreviate common_voice
    if ds == "commonvoice":
        return "cv"
    return ds


def make_speaker_id(
    dataset: str,
    raw_id: str,
    hash_long_ids: bool = True,
    max_length: int = 16,
) -> str:
    """
    Normalize a speaker ID.

    For datasets with hash-based IDs (Common Voice), optionally
    truncate to a shorter hash for readability.

    Args:
        dataset: Dataset name
        raw_id: Raw speaker identifier from dataset
        hash_long_ids: If True, hash IDs longer than max_length
        max_length: Maximum ID length before hashing

    Returns:
        Normalized speaker ID
    """
    # Already short enough
    if len(raw_id) <= max_length:
        return raw_id.lower()

    # Hash long IDs (e.g., Common Voice client_id)
    if hash_long_ids:
        return hashlib.md5(raw_id.encode()).hexdigest()[:12]

    return raw_id.lower()[:max_length]


def parse_utterance_key(utterance_key: str) -> dict:
    """
    Parse a human-readable utterance key back into components.

    Note: This is a best-effort parse. The key format varies by corpus.

    Args:
        utterance_key: Human-readable key string

    Returns:
        Dict with dataset, subset, and either corpus_utt_id or (speaker_id, index)
    """
    parts = utterance_key.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid utterance key format: {utterance_key}")

    result = {
        "dataset": parts[0],
        "subset": parts[1],
    }

    # Remaining parts are either corpus_utt_id or speaker_id + index
    remainder = "_".join(parts[2:])

    # Check if last part is a zero-padded number (index)
    if parts[-1].isdigit() and len(parts) >= 4:
        result["speaker_id"] = "_".join(parts[2:-1])
        result["index"] = int(parts[-1])
    else:
        result["corpus_utt_id"] = remainder

    return result


def make_file_checksum(
    file_path: str | Path,
    algorithm: str = "sha256",
    truncate: Optional[int] = None,
) -> str:
    """
    Generate a checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, sha1, md5)
        truncate: If set, truncate to this many hex chars

    Returns:
        Hex digest string
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    digest = h.hexdigest()
    return digest[:truncate] if truncate else digest


# Backwards compatibility alias
def make_audio_hash(audio_path: str, algorithm: str = "sha256") -> str:
    """Deprecated: use make_file_checksum() instead."""
    return make_file_checksum(audio_path, algorithm, truncate=16)
