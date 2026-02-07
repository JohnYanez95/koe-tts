"""
Silver layer schema definitions.

Silver tables contain cleaned, normalized, QC-passed data.
This layer:
- Inherits all columns from bronze (passthrough)
- Adds QC/eligibility columns
- Adds text normalization columns
- Adds phoneme columns (for generated/labeled phonemes)
- Adds split assignment
- Adds labeler tracking columns
"""

from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SILVER_UTTERANCES_SCHEMA = StructType([
    # =========================
    # Bronze passthrough columns
    # =========================
    StructField("utterance_id", StringType(), nullable=False),
    StructField("utterance_key", StringType(), nullable=False),

    StructField("dataset", StringType(), nullable=False),
    StructField("speaker_id", StringType(), nullable=False),
    StructField("speaker_name", StringType(), nullable=True),
    StructField("subset", StringType(), nullable=False),
    StructField("corpus_utt_id", StringType(), nullable=False),

    StructField("audio_relpath", StringType(), nullable=False),
    StructField("audio_format", StringType(), nullable=False),   # wav/flac/mp3
    StructField("sample_rate", IntegerType(), nullable=False),
    StructField("channels", IntegerType(), nullable=False),
    StructField("duration_sec", FloatType(), nullable=False),

    StructField("text_raw", StringType(), nullable=False),
    StructField("text_norm_raw", StringType(), nullable=True),

    StructField("phonemes_source", StringType(), nullable=False),  # ground_truth|corpus_provided|generated|none|unknown
    StructField("phonemes_raw", StringType(), nullable=True),

    StructField("ingest_version", StringType(), nullable=False),
    StructField("source_version", StringType(), nullable=True),
    StructField("source_url", StringType(), nullable=True),
    StructField("source_archive_checksum", StringType(), nullable=True),
    StructField("audio_checksum", StringType(), nullable=True),
    StructField("ingested_at", TimestampType(), nullable=False),

    StructField("meta", MapType(StringType(), StringType(), valueContainsNull=True), nullable=True),

    # =========================
    # Silver enrichment columns
    # =========================
    # QC / filtering
    StructField("is_trainable", BooleanType(), nullable=False),   # default True
    StructField("exclude_reason", StringType(), nullable=True),
    StructField("qc_version", StringType(), nullable=True),
    StructField("qc_checked_at", TimestampType(), nullable=True),

    # Normalized text (canonical)
    StructField("text_norm", StringType(), nullable=True),
    StructField("text_norm_method", StringType(), nullable=True),

    # Canonical phonemes (post-phonemizer or ground truth)
    StructField("phonemes", StringType(), nullable=True),
    StructField("phonemes_method", StringType(), nullable=True),
    StructField("phonemes_checked", BooleanType(), nullable=False),

    # Split assignment (optional in silver, but many pipelines expect it exists)
    StructField("split", StringType(), nullable=True),

    # Labeling workflow hooks
    StructField("label_status", StringType(), nullable=False),    # default "unlabeled"
    StructField("label_batch_id", StringType(), nullable=True),
    StructField("labeled_at", TimestampType(), nullable=True),
    StructField("labeled_by", StringType(), nullable=True),

    # Lineage / versioning
    StructField("bronze_version", StringType(), nullable=True),
    StructField("silver_version", StringType(), nullable=True),
    StructField("processed_at", TimestampType(), nullable=False),
])

# Silver column defaults for stub
SILVER_STUB_DEFAULTS = {
    "is_trainable": True,
    "exclude_reason": None,
    "qc_version": None,
    "qc_checked_at": None,
    "text_norm": None,
    "text_norm_method": None,
    "phonemes": None,
    "phonemes_method": None,
    "phonemes_checked": False,
    "split": None,
    "label_status": "unlabeled",
    "label_batch_id": None,
    "labeled_at": None,
    "labeled_by": None,
}
