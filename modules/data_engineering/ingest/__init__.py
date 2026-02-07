"""
Data ingestion for koe-tts.

Handles downloading, extracting, and creating manifests for speech corpora.

Usage:
    python -m modules.data_engineering.ingest.cli jsut --download --extract --manifest
"""

from .jsut import build_jsut_manifest, download_jsut, extract_jsut

__all__ = [
    "download_jsut",
    "extract_jsut",
    "build_jsut_manifest",
]
