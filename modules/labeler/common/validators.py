"""
Validators for label data.
"""

from .label_schema import LabelSchema, QualityRating


def validate_label(label: LabelSchema) -> list[str]:
    """
    Validate a label against schema rules.

    Args:
        label: Label to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required field
    if not label.utterance_id:
        errors.append("utterance_id is required")

    # Quality rating validation
    if label.quality_rating is not None:
        if not isinstance(label.quality_rating, QualityRating):
            errors.append(f"Invalid quality_rating: {label.quality_rating}")

    # Correction consistency
    if label.transcription_correct is False and not label.transcription_corrected:
        errors.append("transcription_corrected required when transcription_correct=False")

    if label.phonemes_correct is False and not label.phonemes_corrected:
        errors.append("phonemes_corrected required when phonemes_correct=False")

    return errors
