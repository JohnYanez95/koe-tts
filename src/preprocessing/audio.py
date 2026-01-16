"""
Audio preprocessing for TTS training.

Handles:
- Resampling to target sample rate
- Normalization
- Silence trimming
- Format conversion
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np


def load_audio(
    path: str | Path,
    target_sr: int = 22050
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 22050 for TTS)

    Returns:
        audio: Audio as numpy array
        sr: Sample rate
    """
    import librosa

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    import librosa

    # Calculate current dB
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    return audio * gain


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 30,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Trim silence from beginning and end of audio."""
    import librosa

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return trimmed


def save_audio(
    audio: np.ndarray,
    path: str | Path,
    sr: int = 22050
) -> None:
    """Save audio to file."""
    import soundfile as sf

    sf.write(path, audio, sr)


def get_audio_duration(path: str | Path) -> float:
    """Get duration of audio file in seconds."""
    import librosa

    return librosa.get_duration(path=path)


def process_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    target_sr: int = 22050,
    normalize: bool = True,
    trim: bool = True,
    target_db: float = -20.0
) -> dict:
    """
    Full preprocessing pipeline for a single audio file.

    Returns dict with metadata about the processed file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and resample
    audio, sr = load_audio(input_path, target_sr)

    original_duration = len(audio) / sr

    # Trim silence
    if trim:
        audio = trim_silence(audio, sr)

    # Normalize
    if normalize:
        audio = normalize_audio(audio, target_db)

    # Save
    save_audio(audio, output_path, sr)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_rate": sr,
        "original_duration": original_duration,
        "processed_duration": len(audio) / sr,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            duration = get_audio_duration(path)
            print(f"{path}: {duration:.2f}s")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python audio.py <audio_file>")
