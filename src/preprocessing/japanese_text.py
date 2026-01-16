"""
Japanese text preprocessing for TTS.

Handles:
- Text normalization (fullwidth/halfwidth, numbers)
- Phoneme conversion via pyopenjtalk
- Accent/prosody extraction
"""

from typing import Tuple
import re


def normalize_text(text: str) -> str:
    """Normalize Japanese text for TTS processing."""
    import jaconv

    # Convert fullwidth alphanumeric to halfwidth
    text = jaconv.z2h(text, kana=False, digit=True, ascii=True)

    # Convert halfwidth katakana to fullwidth
    text = jaconv.h2z(text, kana=True, digit=False, ascii=False)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phoneme sequence using OpenJTalk."""
    import pyopenjtalk

    text = normalize_text(text)
    phonemes = pyopenjtalk.g2p(text, kana=False)
    return phonemes


def text_to_phonemes_with_accent(text: str) -> Tuple[str, list]:
    """
    Convert text to phonemes with accent/prosody information.

    Returns:
        phonemes: Phoneme string
        accent_info: List of accent phrase boundaries and pitch patterns
    """
    import pyopenjtalk

    text = normalize_text(text)

    # Get full analysis including accent
    njd_features = pyopenjtalk.run_frontend(text)

    phonemes = []
    accent_info = []

    for feature in njd_features:
        # Extract relevant fields
        phonemes.append(feature['pron'])  # Pronunciation
        accent_info.append({
            'mora': feature['pron'],
            'accent_type': feature.get('acc', 0),
            'is_pause': feature.get('pos', '') == '記号'
        })

    return ' '.join(phonemes), accent_info


def split_into_sentences(text: str) -> list:
    """Split Japanese text into sentences."""
    # Japanese sentence endings
    pattern = r'([。！？\n])'
    parts = re.split(pattern, text)

    sentences = []
    current = ""

    for part in parts:
        current += part
        if part in '。！？\n':
            if current.strip():
                sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences


if __name__ == "__main__":
    # Test
    test_text = "こんにちは、世界！今日は良い天気ですね。"

    print(f"Original: {test_text}")
    print(f"Normalized: {normalize_text(test_text)}")

    try:
        print(f"Phonemes: {text_to_phonemes(test_text)}")
    except ImportError:
        print("pyopenjtalk not installed - run: pip install pyopenjtalk")

    print(f"Sentences: {split_into_sentences(test_text)}")
