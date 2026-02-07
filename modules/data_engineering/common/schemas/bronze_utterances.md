# Bronze Utterances Table

> Raw ingested utterance data with full provenance.

## Location

```
lake/bronze/{dataset}/utterances
```

## Primary Key

`utterance_id` (deterministic hash)

## Partitioning

```python
partitionBy=["dataset"]
```

## Schema

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `utterance_id` | string | No | Stable deterministic ID: `sha1({dataset}\|{speaker_id}\|{subset}\|{corpus_utt_id})[:16]` |
| `utterance_key` | string | No | Human-readable key: `{dataset}_{subset}_{corpus_utt_id}` |
| `dataset` | string | No | Dataset name (`jsut`, `jvs`, `common_voice`) |
| `speaker_id` | string | No | Normalized speaker ID (`spk00`, `spk01`, ...) |
| `speaker_name` | string | Yes | Optional friendly speaker name |
| `subset` | string | No | Corpus subset (`basic5000`, `parallel100`, etc.) |
| `corpus_utt_id` | string | No | Original utterance ID from corpus |
| `audio_relpath` | string | No | Relative path under `DATA_ROOT/data/` |
| `audio_format` | string | No | Audio format (`wav`, `flac`, `mp3`) |
| `sample_rate` | int | No | Sample rate in Hz |
| `channels` | int | No | Number of audio channels |
| `duration_sec` | float | No | Duration in seconds |
| `text_raw` | string | No | Original transcript (unmodified) |
| `text_norm_raw` | string | Yes | Corpus-provided normalized text (if available) |
| `phonemes_source` | string | No | Source of phonemes: `ground_truth`, `corpus_provided`, `generated`, `none`, `unknown` |
| `phonemes_raw` | string | Yes | Phoneme string as provided by corpus |
| `ingest_version` | string | No | Ingest pipeline version tag |
| `source_version` | string | Yes | Corpus version (e.g., `v1.1`) |
| `source_url` | string | Yes | Download URL |
| `source_archive_checksum` | string | Yes | SHA256 of original archive |
| `audio_checksum` | string | Yes | SHA256 of extracted audio file |
| `ingested_at` | timestamp | No | When ingested |
| `meta` | map<string,string> | Yes | Overflow metadata for corpus-specific fields |

## Column Applications

### Identifiers
- **utterance_id**: Join key across all layers. Never changes.
- **utterance_key**: For debugging, UI display, logs.
- **corpus_utt_id**: Maps back to original corpus for reference.

### Audio
- **audio_relpath**: Stable across mounts. Resolve: `DATA_ROOT/data/{audio_relpath}`
- **sample_rate/channels**: Used by QC to detect non-standard audio.
- **duration_sec**: Used for filtering, bucketing, batch balancing.

### Text
- **text_raw**: Original transcript. Never modified in bronze.
- **text_norm_raw**: Only populated if corpus provides normalized version.

### Phonemes
- **phonemes_source**: Tells downstream whether to trust `phonemes_raw`.
- **phonemes_raw**: May be null if corpus doesn't provide phonemes.

### Provenance
- **audio_checksum**: Detect file drift/corruption.
- **source_archive_checksum**: Verify original download.
- **ingest_version**: Ties to MANIFEST for reproducibility.

## Invariants

1. No duplicate `utterance_id`
2. `audio_relpath` points to an existing file
3. `duration_sec > 0`
4. `sample_rate > 0`
5. `phonemes_source` is one of the allowed enum values

## Validation Checks

```python
# Run by bronze pipeline
assert df.select("utterance_id").distinct().count() == df.count()
assert df.filter(col("text_raw").isNull()).count() == 0
assert df.filter(col("duration_sec") <= 0).count() == 0
```
