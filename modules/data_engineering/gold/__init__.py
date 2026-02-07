"""
Gold layer: training-ready data with splits and sampling.

Gold = "what exactly is the dataset I will train on, with stable splits
and a stable snapshot id?"

Operations:
- Filter to trainable rows only
- Deterministic train/val/test splits (hash-based)
- Duration bucketing for length-balanced batching
- Snapshot ID for reproducible training runs
- JSONL manifest export for training

Tables:
- jsut/utterances: JSUT corpus gold data
- jvs/utterances: JVS corpus gold data
- common_voice/utterances: Common Voice gold data (TODO)

Usage:
    from modules.data_engineering.gold import build_gold_jsut, build_gold_jvs
    build_gold_jsut()
    build_gold_jvs()

Or via CLI:
    koe gold jsut
    koe gold jvs
"""

from .jsut import build_gold_jsut, build_gold_df
from .jvs import build_gold_jvs

__all__ = [
    # JSUT
    "build_gold_jsut",
    "build_gold_df",
    # JVS
    "build_gold_jvs",
]
