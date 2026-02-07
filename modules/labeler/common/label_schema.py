"""
Label schema definitions for human annotations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QualityRating(Enum):
    """Audio quality rating."""
    EXCELLENT = "A"  # Production ready
    GOOD = "B"       # Minor issues
    FAIR = "C"       # Usable with caveats
    POOR = "D"       # Not recommended
    UNUSABLE = "F"   # Should be excluded


@dataclass
class LabelSchema:
    """Schema for human labels on an utterance."""

    utterance_id: str

    # Quality assessment
    quality_rating: Optional[QualityRating] = None
    quality_notes: Optional[str] = None

    # Transcription
    transcription_correct: Optional[bool] = None
    transcription_corrected: Optional[str] = None

    # Phonemes
    phonemes_correct: Optional[bool] = None
    phonemes_corrected: Optional[str] = None

    # Audio issues
    has_background_noise: Optional[bool] = None
    has_clipping: Optional[bool] = None
    has_silence_issues: Optional[bool] = None
    has_pronunciation_errors: Optional[bool] = None

    # Labeler info
    labeler_id: Optional[str] = None
    labeled_at: Optional[str] = None
