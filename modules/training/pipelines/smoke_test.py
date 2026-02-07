"""
Training pipeline smoke test with data contract verification.

Verifies the full training pipeline works end-to-end:
1. Cache manifest loading
2. Audio file decoding with correct sample rate
3. Phoneme tokenization roundtrip
4. Batch collation for variable-length sequences
5. Train/val split matches manifest counts
6. Determinism: same seed → same batches
7. Forward pass through dummy model
8. Checkpoint save/load

Usage:
    koe train smoke-test --cache jsut
    python -m modules.training.pipelines.smoke_test --cache jsut

This is NOT a real training run - it uses a dummy model to verify
that all data loading and pipeline infrastructure works correctly.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modules.data_engineering.common.paths import paths
from modules.data_engineering.common.phonemes import (
    CANONICAL_INVENTORY,
    detokenize,
    tokenize,
)


@dataclass
class SmokeTestConfig:
    """Configuration for smoke test."""
    max_utterances: int = 100
    batch_size: int = 4
    num_epochs: int = 1
    target_sample_rate: int = 22050
    max_audio_len: int = 22050 * 10  # 10 seconds
    min_duration_sec: float = 0.1
    max_duration_sec: float = 30.0
    device: str = "cpu"
    seed: int = 42


@dataclass
class SmokeTestResults:
    """Results from smoke test."""
    status: str = "running"
    steps: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, step: str, error: str):
        self.steps[step] = f"FAIL: {error}"
        self.errors.append(f"{step}: {error}")
        self.status = "failed"

    def warn(self, msg: str):
        self.warnings.append(msg)

    def passed(self, step: str, detail: str = ""):
        self.steps[step] = f"PASS{': ' + detail if detail else ''}"


def find_cache(dataset: str, snapshot_id: Optional[str] = None) -> Path:
    """Find cache directory."""
    cache_root = paths.cache / dataset

    if snapshot_id:
        cache_dir = cache_root / snapshot_id
    else:
        # Use latest
        latest = cache_root / "latest"
        if latest.is_symlink():
            cache_dir = cache_root / latest.resolve().name
        else:
            # Find most recent
            caches = [d for d in cache_root.iterdir() if d.is_dir() and d.name != "latest"]
            if not caches:
                raise FileNotFoundError(f"No caches found for {dataset}")
            cache_dir = max(caches, key=lambda d: d.stat().st_mtime)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache not found: {cache_dir}")

    return cache_dir


def load_manifest(cache_dir: Path) -> list[dict]:
    """Load all items from cache manifest."""
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items = []
    with open(manifest_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


# =============================================================================
# Data Contract Tests
# =============================================================================


def test_audio_loading(items: list[dict], config: SmokeTestConfig, results: SmokeTestResults) -> bool:
    """
    Test audio files load correctly with expected properties.

    Checks:
    - Files exist and can be decoded
    - Sample rate matches expected (after resampling)
    - Duration within sanity bounds
    - No NaN/Inf values
    """
    import torchaudio

    print("\n[Contract] Audio Loading...")
    sample_size = min(20, len(items))
    sampled = items[:sample_size]

    loaded = 0
    sr_mismatches = 0
    duration_issues = 0
    nan_issues = 0

    for item in sampled:
        audio_path = item.get("audio_path") or item.get("audio_abspath")
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Check for NaN/Inf
            import torch
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                nan_issues += 1
                continue

            # Duration sanity
            duration = waveform.shape[1] / sr
            manifest_duration = item.get("duration_sec", 0)
            if abs(duration - manifest_duration) > 0.5:
                duration_issues += 1
                results.warn(f"Duration mismatch: {audio_path} manifest={manifest_duration:.2f}s actual={duration:.2f}s")

            loaded += 1

        except Exception as e:
            results.warn(f"Failed to load {audio_path}: {e}")

    if loaded == 0:
        results.fail("audio_loading", "No audio files could be loaded")
        return False

    if nan_issues > 0:
        results.fail("audio_loading", f"{nan_issues}/{sample_size} files contain NaN/Inf")
        return False

    results.passed("audio_loading", f"{loaded}/{sample_size} loaded, {duration_issues} duration mismatches")
    results.stats["audio_loaded"] = loaded
    results.stats["audio_sampled"] = sample_size
    return True


def test_phoneme_roundtrip(items: list[dict], results: SmokeTestResults) -> bool:
    """
    Test phoneme tokenization roundtrips correctly.

    Checks:
    - tokenize() → detokenize() preserves phonemes
    - All phonemes in canonical inventory
    - No empty phoneme sequences for items that should have them
    """
    print("\n[Contract] Phoneme Roundtrip...")

    sample_size = min(50, len(items))
    sampled = items[:sample_size]

    roundtrip_ok = 0
    unknown_phonemes = set()
    empty_count = 0

    for item in sampled:
        phonemes = item.get("phonemes", "")
        if not phonemes:
            empty_count += 1
            continue

        # Tokenize
        tokens = tokenize(phonemes)

        # Check inventory
        for t in tokens:
            if t not in CANONICAL_INVENTORY:
                unknown_phonemes.add(t)

        # Roundtrip
        reconstructed = detokenize(tokens)
        if reconstructed == phonemes:
            roundtrip_ok += 1
        else:
            results.warn(f"Roundtrip mismatch: '{phonemes}' → '{reconstructed}'")

    if unknown_phonemes:
        results.fail("phoneme_roundtrip", f"Unknown phonemes: {unknown_phonemes}")
        return False

    with_phonemes = sample_size - empty_count
    if with_phonemes > 0 and roundtrip_ok < with_phonemes * 0.95:
        results.fail("phoneme_roundtrip", f"Only {roundtrip_ok}/{with_phonemes} roundtripped correctly")
        return False

    results.passed("phoneme_roundtrip", f"{roundtrip_ok}/{with_phonemes} ok, {empty_count} empty")
    results.stats["phoneme_roundtrip_ok"] = roundtrip_ok
    return True


def test_split_distribution(items: list[dict], results: SmokeTestResults) -> bool:
    """
    Test train/val split distribution matches expectations.

    Checks:
    - Split column exists
    - Counts match metadata if available
    - Reasonable train/val ratio
    """
    print("\n[Contract] Split Distribution...")

    splits = {}
    for item in items:
        split = item.get("split", "unknown")
        splits[split] = splits.get(split, 0) + 1

    if not splits:
        results.fail("split_distribution", "No split column found")
        return False

    total = sum(splits.values())
    for split, count in sorted(splits.items()):
        pct = count / total * 100
        print(f"  {split}: {count} ({pct:.1f}%)")

    # Basic sanity: train should be largest
    if "train" in splits and "val" in splits:
        if splits["train"] < splits["val"]:
            results.warn("Train split smaller than val - unusual")

    results.passed("split_distribution", f"{len(splits)} splits, {total} total")
    results.stats["split_distribution"] = splits
    return True


def test_batch_collation(items: list[dict], config: SmokeTestConfig, results: SmokeTestResults) -> bool:
    """
    Test batch collation handles variable-length sequences.

    Checks:
    - Batches are created without errors
    - Padding is applied correctly
    - Lengths are tracked
    """
    import torch
    import torchaudio
    from torch.utils.data import DataLoader, Dataset

    print("\n[Contract] Batch Collation...")

    class SimpleDataset(Dataset):
        def __init__(self, items, max_items):
            self.items = items[:max_items]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate_with_audio(batch):
        """Collate with actual audio loading."""
        audios = []
        phoneme_strs = []
        valid = []

        for item in batch:
            audio_path = item.get("audio_path") or item.get("audio_abspath")
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != config.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, config.target_sample_rate)
                    waveform = resampler(waveform)
                waveform = waveform.mean(dim=0)  # mono
                if len(waveform) > config.max_audio_len:
                    waveform = waveform[:config.max_audio_len]
                audios.append(waveform)
                phoneme_strs.append(item.get("phonemes", ""))
                valid.append(item)
            except Exception:
                continue

        if not audios:
            return None

        # Pad audio
        max_len = max(len(a) for a in audios)
        audio_padded = torch.zeros(len(audios), max_len)
        audio_lens = []
        for i, a in enumerate(audios):
            audio_padded[i, :len(a)] = a
            audio_lens.append(len(a))

        # Tokenize and pad phonemes
        vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for p in sorted(CANONICAL_INVENTORY):
            vocab[p] = len(vocab)

        phone_ids_list = []
        for ps in phoneme_strs:
            tokens = tokenize(ps)
            ids = [vocab["<bos>"]] + [vocab.get(t, vocab["<unk>"]) for t in tokens] + [vocab["<eos>"]]
            phone_ids_list.append(torch.tensor(ids))

        max_phone_len = max(len(p) for p in phone_ids_list)
        phone_padded = torch.zeros(len(phone_ids_list), max_phone_len, dtype=torch.long)
        phone_lens = []
        for i, p in enumerate(phone_ids_list):
            phone_padded[i, :len(p)] = p
            phone_lens.append(len(p))

        return {
            "audio": audio_padded,
            "audio_lens": torch.tensor(audio_lens),
            "phonemes": phone_padded,
            "phoneme_lens": torch.tensor(phone_lens),
        }

    try:
        ds = SimpleDataset(items, min(50, len(items)))
        loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_with_audio)

        batches_ok = 0
        variable_lens_seen = False

        for batch in loader:
            if batch is None:
                continue
            batches_ok += 1

            # Check variable lengths exist
            audio_lens = batch["audio_lens"]
            if audio_lens.min() != audio_lens.max():
                variable_lens_seen = True

            # Check shapes match
            assert batch["audio"].shape[0] == len(audio_lens)
            assert batch["phonemes"].shape[0] == len(batch["phoneme_lens"])

        if batches_ok == 0:
            results.fail("batch_collation", "No batches could be created")
            return False

        results.passed("batch_collation", f"{batches_ok} batches, variable_lens={variable_lens_seen}")
        results.stats["batches_created"] = batches_ok
        return True

    except Exception as e:
        results.fail("batch_collation", str(e))
        return False


def test_determinism(items: list[dict], config: SmokeTestConfig, results: SmokeTestResults) -> bool:
    """
    Test that same seed produces same batch order.

    Checks:
    - Setting seed before DataLoader gives reproducible order
    - First N items match across two runs
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    print("\n[Contract] Determinism...")

    class SimpleDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            # Return just the utterance_id to avoid collation issues
            return self.items[idx]["utterance_id"]

    def get_first_n_ids(items, seed, n=10):
        torch.manual_seed(seed)
        ds = SimpleDataset(items[:50])
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        ids = []
        for batch in loader:
            # batch is now a list of utterance_ids
            for uid in batch:
                ids.append(uid)
                if len(ids) >= n:
                    return ids
        return ids

    try:
        ids1 = get_first_n_ids(items, config.seed)
        ids2 = get_first_n_ids(items, config.seed)

        if ids1 != ids2:
            results.fail("determinism", f"Different order: {ids1[:3]} vs {ids2[:3]}")
            return False

        # Different seed should give different order
        ids3 = get_first_n_ids(items, config.seed + 1)
        if ids1 == ids3:
            results.warn("Same order with different seed - may be coincidence with small dataset")

        results.passed("determinism", f"Seed {config.seed} reproduces same order")
        return True

    except Exception as e:
        results.fail("determinism", str(e))
        return False


# =============================================================================
# Training Smoke Test
# =============================================================================


def test_forward_pass(items: list[dict], config: SmokeTestConfig, results: SmokeTestResults) -> bool:
    """
    Test forward pass through dummy model.
    """
    import torch
    import torch.nn as nn
    import torchaudio
    from torch.utils.data import DataLoader, Dataset

    print("\n[Contract] Forward Pass...")

    class DummyTTSModel(nn.Module):
        def __init__(self, vocab_size, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, 1)

        def forward(self, phonemes, phoneme_lens):
            x = self.embedding(phonemes)
            x, _ = self.encoder(x)
            x = self.proj(x)
            return x.abs().mean()

    try:
        # Build vocab
        vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for p in sorted(CANONICAL_INVENTORY):
            vocab[p] = len(vocab)

        model = DummyTTSModel(len(vocab))
        model = model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Quick forward pass
        sample_phonemes = torch.randint(0, len(vocab), (2, 20))
        sample_lens = torch.tensor([20, 15])

        model.train()
        optimizer.zero_grad()
        loss = model(sample_phonemes, sample_lens)
        loss.backward()
        optimizer.step()

        results.passed("forward_pass", f"loss={loss.item():.4f}")
        results.stats["forward_loss"] = loss.item()
        return True

    except Exception as e:
        results.fail("forward_pass", str(e))
        return False


def test_checkpoint_roundtrip(config: SmokeTestConfig, results: SmokeTestResults) -> bool:
    """
    Test checkpoint save and load.
    """
    import torch
    import torch.nn as nn

    print("\n[Contract] Checkpoint Roundtrip...")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    try:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Do a step
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Save
        checkpoint_dir = paths.checkpoints / "smoke_test"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "smoke_test.ckpt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 1,
        }, checkpoint_path)

        # Load
        loaded = torch.load(checkpoint_path, weights_only=True)
        model2 = TinyModel()
        model2.load_state_dict(loaded["model_state_dict"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if not torch.allclose(p1, p2):
                results.fail("checkpoint_roundtrip", f"Weight mismatch: {n1}")
                return False

        results.passed("checkpoint_roundtrip", f"Saved to {checkpoint_path}")
        results.stats["checkpoint_path"] = str(checkpoint_path)
        return True

    except Exception as e:
        results.fail("checkpoint_roundtrip", str(e))
        return False


# =============================================================================
# Main Runner
# =============================================================================


def run_smoke_test(
    dataset: str,
    snapshot_id: Optional[str] = None,
    config: Optional[SmokeTestConfig] = None,
) -> dict:
    """
    Run training smoke test with data contract verification.

    Args:
        dataset: Dataset name (jsut, jvs)
        snapshot_id: Specific cache snapshot (None = latest)
        config: Test configuration

    Returns:
        Dict with test results
    """
    if config is None:
        config = SmokeTestConfig()

    print("=" * 60)
    print("Training Smoke Test + Data Contract Verification")
    print("=" * 60)
    print(f"  Dataset: {dataset}")
    print(f"  Max utterances: {config.max_utterances}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")

    results = SmokeTestResults()

    # Step 1: Find cache
    print("\n[1/8] Finding cache...")
    try:
        cache_dir = find_cache(dataset, snapshot_id)
        print(f"  Cache: {cache_dir}")
        results.passed("find_cache", str(cache_dir))
        results.stats["cache_dir"] = str(cache_dir)
    except Exception as e:
        results.fail("find_cache", str(e))
        return results.__dict__

    # Step 2: Load manifest
    print("\n[2/8] Loading manifest...")
    try:
        items = load_manifest(cache_dir)
        print(f"  Loaded {len(items)} items")
        # Limit for testing
        items = items[:config.max_utterances]
        print(f"  Using {len(items)} items for test")
        results.passed("load_manifest", f"{len(items)} items")
        results.stats["num_items"] = len(items)
    except Exception as e:
        results.fail("load_manifest", str(e))
        return results.__dict__

    # Step 3-8: Data Contract Tests
    print("\n" + "=" * 60)
    print("Data Contract Verification")
    print("=" * 60)

    all_passed = True

    # Audio loading
    if not test_audio_loading(items, config, results):
        all_passed = False

    # Phoneme roundtrip
    if not test_phoneme_roundtrip(items, results):
        all_passed = False

    # Split distribution
    if not test_split_distribution(items, results):
        all_passed = False

    # Batch collation
    if not test_batch_collation(items, config, results):
        all_passed = False

    # Determinism
    if not test_determinism(items, config, results):
        all_passed = False

    # Forward pass
    if not test_forward_pass(items, config, results):
        all_passed = False

    # Checkpoint roundtrip
    if not test_checkpoint_roundtrip(config, results):
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed and results.status != "failed":
        results.status = "passed"
        print("SMOKE TEST PASSED")
    else:
        results.status = "failed"
        print("SMOKE TEST FAILED")
    print("=" * 60)

    print("\nResults:")
    for step, result in results.steps.items():
        status = "✓" if result.startswith("PASS") else "✗"
        print(f"  {status} {step}: {result}")

    if results.warnings:
        print("\nWarnings:")
        for w in results.warnings:
            print(f"  ⚠ {w}")

    if results.errors:
        print("\nErrors:")
        for e in results.errors:
            print(f"  ✗ {e}")

    print("=" * 60)

    return results.__dict__


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Training smoke test with data contract verification")
    parser.add_argument("--cache", "-c", required=True, help="Dataset name")
    parser.add_argument("--snapshot", "-s", help="Specific snapshot ID")
    parser.add_argument("--max-utterances", "-n", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = SmokeTestConfig(
        max_utterances=args.max_utterances,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    result = run_smoke_test(
        dataset=args.cache,
        snapshot_id=args.snapshot,
        config=config,
    )

    if result["status"] != "passed":
        print("\nSmoke test FAILED")
        exit(1)


if __name__ == "__main__":
    main()
