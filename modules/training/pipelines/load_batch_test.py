"""
Load One Batch Test - Stage 1 verification.

Verifies the dataloading pipeline works end-to-end:
1. Load dataset from cache
2. Create batch via collator
3. Extract mel spectrograms
4. Print batch statistics

Usage:
    python -m modules.training.pipelines.load_batch_test --cache-dir data/cache/jsut/latest
    python -m modules.training.pipelines.load_batch_test --dataset jsut
"""

import argparse
import sys
from pathlib import Path


def find_latest_cache(dataset: str) -> Path:
    """Find the latest cache directory for a dataset."""
    cache_base = Path("data/cache") / dataset
    if not cache_base.exists():
        raise FileNotFoundError(f"No cache found for dataset '{dataset}' at {cache_base}")

    # Check for 'latest' symlink
    latest = cache_base / "latest"
    if latest.exists():
        return latest.resolve()

    # Find most recent snapshot
    snapshots = sorted(cache_base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    snapshots = [s for s in snapshots if s.is_dir() and s.name != "latest"]
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {cache_base}")

    return snapshots[0]


def main():
    parser = argparse.ArgumentParser(description="Load one batch test")
    parser.add_argument("--cache-dir", type=Path, help="Path to cache directory")
    parser.add_argument("--dataset", type=str, help="Dataset name (finds latest cache)")
    parser.add_argument("--split", type=str, default="train", help="Split to test")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    # Resolve cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.dataset:
        cache_dir = find_latest_cache(args.dataset)
    else:
        parser.error("Either --cache-dir or --dataset must be specified")

    print(f"Cache directory: {cache_dir}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Import here to avoid slow imports if args are wrong
    import torch
    from modules.training.dataloading import (
        TTSDataset,
        TTSCollator,
        PHONEME_VOCAB,
        PHONEME_VOCAB_SIZE,
    )
    from modules.training.audio import MelConfig, DEFAULT_MEL_CONFIG

    print("=" * 60)
    print("STAGE 1: DATALOADING VERIFICATION")
    print("=" * 60)
    print()

    # Step 1: Phoneme vocab
    print("[1/5] Phoneme vocabulary")
    print(f"  Vocab size: {PHONEME_VOCAB_SIZE}")
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    for tok in special_tokens:
        print(f"  {tok}: {PHONEME_VOCAB[tok]}")
    print()

    # Step 2: Mel config
    print("[2/5] Mel configuration")
    config = DEFAULT_MEL_CONFIG
    print(f"  Sample rate: {config.sample_rate}")
    print(f"  N FFT: {config.n_fft}")
    print(f"  Hop length: {config.hop_length}")
    print(f"  N mels: {config.n_mels}")
    print(f"  Frame rate: {config.frame_rate:.1f} fps")
    print()

    # Step 3: Load dataset
    print("[3/5] Loading dataset...")
    try:
        dataset = TTSDataset(
            cache_dir=cache_dir,
            split=args.split,
            mel_config=config,
        )
        print(f"  Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
    print()

    # Step 4: Load first batch
    print("[4/5] Loading batch...")
    collator = TTSCollator(pad_id=PHONEME_VOCAB["<pad>"])

    # Get samples
    samples = []
    for i in range(min(args.batch_size, len(dataset))):
        sample = dataset[i]
        if sample is not None:
            samples.append(sample)

    if not samples:
        print("  FAILED: No valid samples loaded")
        sys.exit(1)

    batch = collator(samples)
    if batch is None:
        print("  FAILED: Collator returned None")
        sys.exit(1)

    print(f"  Batch size: {len(batch)}")
    print()

    # Step 5: Batch statistics
    print("[5/5] Batch statistics")
    print(f"  Phonemes shape: {batch.phonemes.shape}")
    print(f"  Phoneme lengths: {batch.phoneme_lens.tolist()}")
    print(f"  Phoneme mask shape: {batch.phoneme_mask.shape}")
    print()
    print(f"  Mels shape: {batch.mels.shape}")
    print(f"  Mel lengths: {batch.mel_lens.tolist()}")
    print(f"  Mel mask shape: {batch.mel_mask.shape}")
    print()
    print(f"  Mel value range: [{batch.mels.min():.2f}, {batch.mels.max():.2f}]")
    print(f"  Durations (sec): {batch.durations_sec.tolist()}")
    print()
    print(f"  Utterance IDs:")
    for uid in batch.utterance_ids:
        print(f"    - {uid}")
    print()

    # Sanity checks
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    checks_passed = 0
    checks_total = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {msg}")

    # Check shapes
    B = len(batch)
    check(
        "Phoneme batch dim",
        batch.phonemes.shape[0] == B,
        f"Expected {B}, got {batch.phonemes.shape[0]}"
    )
    check(
        "Mel batch dim",
        batch.mels.shape[0] == B,
        f"Expected {B}, got {batch.mels.shape[0]}"
    )
    check(
        "Mel channels",
        batch.mels.shape[1] == config.n_mels,
        f"Expected {config.n_mels}, got {batch.mels.shape[1]}"
    )

    # Check masks
    check(
        "Phoneme mask shape",
        batch.phoneme_mask.shape == batch.phonemes.shape,
        f"{batch.phoneme_mask.shape} != {batch.phonemes.shape}"
    )
    check(
        "Mel mask shape",
        batch.mel_mask.shape == (B, batch.mels.shape[2]),
        f"{batch.mel_mask.shape} != ({B}, {batch.mels.shape[2]})"
    )

    # Check lengths match masks
    for i in range(B):
        expected_phone_len = batch.phoneme_lens[i].item()
        actual_phone_len = batch.phoneme_mask[i].sum().item()
        if expected_phone_len != actual_phone_len:
            check(
                f"Phoneme mask consistency [{i}]",
                False,
                f"len={expected_phone_len}, mask_sum={actual_phone_len}"
            )
            break
    else:
        check("Phoneme mask consistency", True)

    for i in range(B):
        expected_mel_len = batch.mel_lens[i].item()
        actual_mel_len = batch.mel_mask[i].sum().item()
        if expected_mel_len != actual_mel_len:
            check(
                f"Mel mask consistency [{i}]",
                False,
                f"len={expected_mel_len}, mask_sum={actual_mel_len}"
            )
            break
    else:
        check("Mel mask consistency", True)

    # Check mel values are reasonable (log scale)
    check(
        "Mel values reasonable",
        batch.mels.min() > -20 and batch.mels.max() < 10,
        f"Range [{batch.mels.min():.2f}, {batch.mels.max():.2f}] seems wrong"
    )

    # Check BOS/EOS tokens
    bos_id = PHONEME_VOCAB["<bos>"]
    eos_id = PHONEME_VOCAB["<eos>"]
    all_have_bos = all(batch.phonemes[i, 0].item() == bos_id for i in range(B))
    check("BOS token present", all_have_bos, "First token should be <bos>")

    all_have_eos = all(
        batch.phonemes[i, batch.phoneme_lens[i] - 1].item() == eos_id
        for i in range(B)
    )
    check("EOS token present", all_have_eos, "Last valid token should be <eos>")

    print()
    print(f"Checks: {checks_passed}/{checks_total} passed")
    print()

    if checks_passed == checks_total:
        print("=" * 60)
        print("STAGE 1 VERIFICATION COMPLETE")
        print("Dataloading pipeline is working correctly.")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("STAGE 1 VERIFICATION FAILED")
        print(f"{checks_total - checks_passed} check(s) failed.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
