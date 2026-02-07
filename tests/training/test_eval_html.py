"""Contract tests for eval HTML generation.

Ensures the eval HTML renderer produces valid output with required elements.
"""

import pytest


def test_multispeaker_html_contains_required_elements():
    """Test that generated HTML contains all required structural elements."""
    from modules.training.pipelines.synthesize import _generate_multispeaker_html

    # Minimal fake manifest
    manifest = {
        "schema_version": 2,
        "created_at": "2026-01-27T00:00:00+00:00",
        "run": {
            "name": "test_run",
            "checkpoint": "best.pt",
            "step": 1000,
            "num_speakers": 2,
        },
        "params": {
            "duration_scale": 1.0,
            "noise_scale": 0.667,
            "seed": 42,
        },
        "speakers": ["spk00", "spk01"],
        "n_prompts": 2,
        "results": [
            {
                "id": "000",
                "text": "テスト文です。",
                "phonemes": "t e s U t o b u N d e s U",
                "speakers": {
                    "spk00": {"path": "spk00/000.wav", "duration_sec": 1.0, "rms": 0.05, "silence_pct": 5.0, "is_valid": True},
                    "spk01": {"path": "spk01/000.wav", "duration_sec": 1.1, "rms": 0.04, "silence_pct": 6.0, "is_valid": True},
                },
            },
            {
                "id": "001",
                "text": "二番目のテスト。",
                "phonemes": "n i b a N m e n o t e s U t o",
                "speakers": {
                    "spk00": {"path": "spk00/001.wav", "duration_sec": 0.9, "rms": 0.05, "silence_pct": 4.0, "is_valid": True},
                    "spk01": {"path": "spk01/001.wav", "duration_sec": 1.0, "rms": 0.04, "silence_pct": 5.0, "is_valid": True},
                },
            },
        ],
        "per_speaker_summary": {
            "spk00": {"n_samples": 2, "n_valid": 2, "mean_duration_sec": 0.95, "mean_rms": 0.05, "mean_silence_pct": 4.5},
            "spk01": {"n_samples": 2, "n_valid": 2, "mean_duration_sec": 1.05, "mean_rms": 0.04, "mean_silence_pct": 5.5},
        },
        "separation_metrics": {
            "mean_inter_speaker_distance": 0.15,
            "inter_speaker_std": 0.02,
            "median_intra_speaker_distance": 0.10,
            "per_speaker_consistency": {
                "spk00": {"mean_intra_distance": 0.09, "n_pairs": 1, "is_anomalous": False},
                "spk01": {"mean_intra_distance": 0.11, "n_pairs": 1, "is_anomalous": False},
            },
        },
    }

    prompts = ["テスト文です。", "二番目のテスト。"]
    speakers = [("spk00", 0), ("spk01", 1)]

    html = _generate_multispeaker_html(manifest, prompts, speakers)

    # Required structural elements
    assert "<!DOCTYPE html>" in html
    assert "<title>Multi-Speaker Eval:" in html

    # Phonemes must be present for each prompt
    assert 't e s U t o b u N d e s U' in html  # First prompt phonemes
    assert 'n i b a N m e n o t e s U t o' in html  # Second prompt phonemes
    assert 'class="phonemes"' in html

    # Sticky header + text column classes
    assert 'class="sticky-col"' in html
    assert 'class="sticky-col text-col"' in html
    assert "position: sticky" in html

    # Zebra striping
    assert 'class="even-row"' in html
    assert 'class="odd-row"' in html

    # Speaker columns
    assert ">spk00<" in html
    assert ">spk01<" in html

    # Audio elements
    assert '<audio controls' in html
    assert 'spk00/000.wav' in html

    # Summary section
    assert "Per-Speaker Summary" in html
    assert "Speaker Separation" in html


def test_multispeaker_html_handles_missing_phonemes():
    """Test backwards-compat: missing phonemes shows placeholder."""
    from modules.training.pipelines.synthesize import _generate_multispeaker_html

    # Manifest without phonemes (v1 format)
    manifest = {
        "created_at": "2026-01-27T00:00:00+00:00",
        "run": {"name": "old_run", "checkpoint": "best.pt", "step": 500, "num_speakers": 1},
        "params": {"duration_scale": 1.0, "noise_scale": 0.667, "seed": 42},
        "speakers": ["spk00"],
        "n_prompts": 1,
        "results": [
            {
                "id": "000",
                "text": "テスト",
                # No "phonemes" field - v1 manifest
                "speakers": {
                    "spk00": {"path": "spk00/000.wav", "duration_sec": 0.5, "rms": 0.05, "silence_pct": 5.0, "is_valid": True},
                },
            },
        ],
        "per_speaker_summary": {
            "spk00": {"n_samples": 1, "n_valid": 1, "mean_duration_sec": 0.5, "mean_rms": 0.05, "mean_silence_pct": 5.0},
        },
        "separation_metrics": {
            "mean_inter_speaker_distance": 0.0,
            "inter_speaker_std": 0.0,
            "median_intra_speaker_distance": 0.0,
            "per_speaker_consistency": {"spk00": {"mean_intra_distance": 0.0, "n_pairs": 0, "is_anomalous": False}},
        },
    }

    prompts = ["テスト"]
    speakers = [("spk00", 0)]

    # Should not raise, should render gracefully
    html = _generate_multispeaker_html(manifest, prompts, speakers)

    # Should contain the unavailable placeholder
    assert "(phonemes unavailable)" in html
    assert 'class="phonemes unavailable"' in html


def test_multispeaker_html_escapes_special_characters():
    """Test that HTML special characters are properly escaped."""
    from modules.training.pipelines.synthesize import _generate_multispeaker_html

    manifest = {
        "schema_version": 2,
        "created_at": "2026-01-27T00:00:00+00:00",
        "run": {"name": "test<script>alert(1)</script>", "checkpoint": "best.pt", "step": 100, "num_speakers": 1},
        "params": {"duration_scale": 1.0, "noise_scale": 0.667, "seed": 42},
        "speakers": ["spk00"],
        "n_prompts": 1,
        "results": [
            {
                "id": "000",
                "text": "<script>alert('xss')</script>",
                "phonemes": "test & phonemes < >",
                "speakers": {
                    "spk00": {"path": "spk00/000.wav", "duration_sec": 0.5, "rms": 0.05, "silence_pct": 5.0, "is_valid": True},
                },
            },
        ],
        "per_speaker_summary": {
            "spk00": {"n_samples": 1, "n_valid": 1, "mean_duration_sec": 0.5, "mean_rms": 0.05, "mean_silence_pct": 5.0},
        },
        "separation_metrics": {
            "mean_inter_speaker_distance": 0.0,
            "inter_speaker_std": 0.0,
            "median_intra_speaker_distance": 0.0,
            "per_speaker_consistency": {"spk00": {"mean_intra_distance": 0.0, "n_pairs": 0, "is_anomalous": False}},
        },
    }

    prompts = ["<script>alert('xss')</script>"]
    speakers = [("spk00", 0)]

    html = _generate_multispeaker_html(manifest, prompts, speakers)

    # Raw script tags should NOT appear (must be escaped)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html  # Escaped version
    assert "test &amp; phonemes &lt; &gt;" in html  # Escaped phonemes
