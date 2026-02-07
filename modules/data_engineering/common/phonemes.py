"""
Canonical phoneme inventory and normalization utilities.

The canonical inventory is derived from JVS corpus (OpenJTalk HTS format).
JVS phonemes == pyopenjtalk output (verified 100% match on 12,734 utterances).

Normalization rules:
- Tokenize by whitespace
- Strip leading/trailing 'sil' (sentence boundary markers)
- Keep internal 'pau' (punctuation pauses - meaningful prosody)
- All other tokens unchanged

Usage:
    from modules.data_engineering.common.phonemes import (
        tokenize,
        detokenize,
        normalize_openjtalk,
        validate_inventory,
        generate_phonemes,
        CANONICAL_INVENTORY,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Set, Tuple

# =============================================================================
# Boundary markers
# =============================================================================

BOUNDARY_SIL = {"sil"}  # Strip at utterance boundaries
# Note: We do NOT strip boundary pau - internal pau is meaningful prosody

# =============================================================================
# Canonical Inventory
# =============================================================================
# Derived from JVS corpus phonemes_raw (12,734 utterances, 902,388 tokens).
# This is the OpenJTalk HTS phoneme set.

CANONICAL_INVENTORY: frozenset[str] = frozenset({
    # Vowels (5)
    "a", "i", "u", "e", "o",
    # Devoiced vowels (2)
    "I", "U",
    # Syllabic nasal
    "N",
    # Geminate (glottal closure)
    "cl",
    # Silence/pause markers
    "sil", "pau",
    # Basic consonants
    "k", "s", "t", "n", "h", "m", "y", "r", "w",
    "g", "z", "d", "b", "p", "f", "v", "j",
    # Palatalized/affricate
    "ky", "sh", "ch", "ny", "hy", "my", "ry",
    "gy", "by", "py", "dy", "ty", "ts",
})

# =============================================================================
# Tokenization
# =============================================================================


def tokenize(phoneme_str: str | None) -> List[str]:
    """
    Tokenize a phoneme string by whitespace.

    Args:
        phoneme_str: Space-separated phoneme string

    Returns:
        List of phoneme tokens
    """
    if not phoneme_str:
        return []
    return [p for p in phoneme_str.strip().split() if p]


def detokenize(tokens: Sequence[str]) -> str | None:
    """
    Join phoneme tokens back into a string.

    Args:
        tokens: List of phoneme tokens

    Returns:
        Space-separated phoneme string, or None if empty
    """
    if not tokens:
        return None
    return " ".join(tokens)


# =============================================================================
# Normalization
# =============================================================================


def strip_boundary_silence(tokens: Sequence[str]) -> List[str]:
    """
    Remove 'sil' markers from start and end of token sequence.

    Internal 'pau' markers are preserved (meaningful prosody).

    Args:
        tokens: List of phoneme tokens

    Returns:
        Tokens with boundary silence removed
    """
    if not tokens:
        return []

    i, j = 0, len(tokens)

    # Strip leading sil
    while i < j and tokens[i] in BOUNDARY_SIL:
        i += 1

    # Strip trailing sil
    while j > i and tokens[j - 1] in BOUNDARY_SIL:
        j -= 1

    return list(tokens[i:j])


def normalize_openjtalk(tokens: Sequence[str]) -> List[str]:
    """
    Normalize OpenJTalk/JVS phoneme tokens to canonical form.

    Currently: strip boundary 'sil' only.
    Future: could add more normalization rules here.

    Args:
        tokens: List of phoneme tokens

    Returns:
        Normalized token list
    """
    return strip_boundary_silence(tokens)


def normalize_phonemes(phoneme_str: str | None) -> str | None:
    """
    Convenience function: normalize a phoneme string.

    Args:
        phoneme_str: Space-separated phoneme string

    Returns:
        Normalized phoneme string
    """
    tokens = tokenize(phoneme_str)
    normalized = normalize_openjtalk(tokens)
    return detokenize(normalized)


# =============================================================================
# Validation
# =============================================================================


def validate_inventory(
    tokens: Sequence[str],
    inventory: Set[str] | None = None,
) -> Tuple[bool, Set[str]]:
    """
    Validate that all tokens are in the canonical inventory.

    Args:
        tokens: List of phoneme tokens
        inventory: Custom inventory (default: CANONICAL_INVENTORY)

    Returns:
        Tuple of (is_valid, set_of_unknown_tokens)
    """
    if inventory is None:
        inventory = CANONICAL_INVENTORY

    unknown = set(tokens) - inventory
    return (len(unknown) == 0, unknown)


def validate_phonemes(phoneme_str: str | None) -> Tuple[bool, Set[str]]:
    """
    Convenience function: validate a phoneme string.

    Args:
        phoneme_str: Space-separated phoneme string

    Returns:
        Tuple of (is_valid, set_of_unknown_tokens)
    """
    tokens = tokenize(phoneme_str)
    return validate_inventory(tokens)


# =============================================================================
# Generation (pyopenjtalk wrapper)
# =============================================================================


def generate_phonemes(text: str | None) -> str | None:
    """
    Generate phonemes from Japanese text using pyopenjtalk.

    Args:
        text: Japanese text

    Returns:
        Space-separated phoneme string (NOT normalized - raw pyopenjtalk output)
    """
    if not text or not text.strip():
        return None

    try:
        import pyopenjtalk
        phonemes = pyopenjtalk.g2p(text)
        return phonemes if phonemes else None
    except Exception:
        return None


def generate_phonemes_normalized(text: str | None) -> str | None:
    """
    Generate and normalize phonemes from Japanese text.

    Args:
        text: Japanese text

    Returns:
        Normalized phoneme string (boundary sil stripped)
    """
    raw = generate_phonemes(text)
    return normalize_phonemes(raw)


# =============================================================================
# Spark UDF factories
# =============================================================================


def create_generate_phonemes_udf():
    """
    Create a Spark UDF for phoneme generation.

    Note: For small datasets (<10k rows), consider driver-side processing
    instead to avoid serialization overhead.

    Returns:
        PySpark UDF
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    @F.udf(StringType())
    def _udf(text: str) -> str | None:
        return generate_phonemes(text)

    return _udf


def create_normalize_phonemes_udf():
    """
    Create a Spark UDF for phoneme normalization.

    Returns:
        PySpark UDF
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    @F.udf(StringType())
    def _udf(phonemes: str) -> str | None:
        return normalize_phonemes(phonemes)

    return _udf


# =============================================================================
# Inventory Export
# =============================================================================


def compute_inventory_stats(
    df,
    phonemes_col: str = "phonemes",
    is_trainable_col: str = "is_trainable",
) -> dict:
    """
    Compute phoneme inventory statistics from a DataFrame.

    Args:
        df: PySpark DataFrame with phonemes column
        phonemes_col: Name of the phonemes column
        is_trainable_col: Name of the trainability column

    Returns:
        Dict with inventory, counts, and coverage stats
    """
    from pyspark.sql import functions as F

    # Total counts
    total_count = df.count()
    with_phonemes_count = df.filter(F.col(phonemes_col).isNotNull()).count()
    trainable_count = df.filter(F.col(is_trainable_col) == True).count()
    trainable_with_phonemes = df.filter(
        (F.col(is_trainable_col) == True) & (F.col(phonemes_col).isNotNull())
    ).count()

    # Collect phonemes for counting
    phoneme_rows = (
        df.filter(F.col(phonemes_col).isNotNull())
        .select(phonemes_col)
        .collect()
    )

    # Count all phoneme tokens
    token_counts: dict[str, int] = {}
    for row in phoneme_rows:
        tokens = tokenize(row[phonemes_col])
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

    # Compute inventory (sorted)
    inventory = sorted(token_counts.keys())

    # Coverage
    coverage_all = with_phonemes_count / total_count if total_count > 0 else 0.0
    coverage_trainable = trainable_with_phonemes / trainable_count if trainable_count > 0 else 0.0

    return {
        "inventory": inventory,
        "counts": token_counts,
        "num_utterances_total": total_count,
        "num_utterances_with_phonemes": with_phonemes_count,
        "num_trainable": trainable_count,
        "num_trainable_with_phonemes": trainable_with_phonemes,
        "coverage_all": coverage_all,
        "coverage_trainable": coverage_trainable,
    }


def export_inventory_json(
    df,
    output_path: str | Path,
    dataset: str,
    layer: str,
    source_table: str,
    phonemes_method: str,
    inventory_version: str = "v1",
    phonemes_col: str = "phonemes",
    is_trainable_col: str = "is_trainable",
) -> Path:
    """
    Export phoneme inventory and statistics to JSON.

    This creates a persisted artifact for tracking the canonical phoneme
    inventory derived from a corpus. The inventory hash enables detecting
    when the phoneme set has changed.

    Args:
        df: PySpark DataFrame with phonemes column
        output_path: Path to write JSON file
        dataset: Dataset name (e.g., "jvs")
        layer: Layer name (e.g., "silver")
        source_table: Source table path (e.g., "lake/silver/jvs/utterances")
        phonemes_method: Method used to generate phonemes
        inventory_version: Version tag for the inventory format
        phonemes_col: Name of the phonemes column
        is_trainable_col: Name of the trainability column

    Returns:
        Path to the written JSON file
    """
    import hashlib
    import json
    from datetime import datetime, timezone

    output_path = Path(output_path)

    # Compute stats
    stats = compute_inventory_stats(
        df,
        phonemes_col=phonemes_col,
        is_trainable_col=is_trainable_col,
    )

    # Compute inventory hash (for change detection)
    inventory_str = json.dumps(sorted(stats["inventory"]), sort_keys=True)
    inventory_hash = f"sha1:{hashlib.sha1(inventory_str.encode()).hexdigest()}"

    # Build output document
    doc = {
        "dataset": dataset,
        "layer": layer,
        "source_table": source_table,
        "phonemes_method": phonemes_method,
        "inventory_version": inventory_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "normalize_rules": {
            "strip_boundary_sil": True,
            "keep_internal_pau": True,
        },
        "inventory": stats["inventory"],
        "counts": dict(sorted(stats["counts"].items())),
        "num_utterances_total": stats["num_utterances_total"],
        "num_utterances_with_phonemes": stats["num_utterances_with_phonemes"],
        "num_trainable": stats["num_trainable"],
        "num_trainable_with_phonemes": stats["num_trainable_with_phonemes"],
        "coverage_all": round(stats["coverage_all"], 4),
        "coverage_trainable": round(stats["coverage_trainable"], 4),
        "inventory_hash": inventory_hash,
    }

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    return output_path
