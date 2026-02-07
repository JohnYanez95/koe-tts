"""
Training pipeline.

Usage:
    python -m modules.training.pipelines.train --cache-snapshot jsut_v1 --config configs/training/vits.yaml
"""

import argparse
from pathlib import Path


def train(
    cache_snapshot: str,
    config_path: Path,
    resume_from: Path | None = None,
) -> dict:
    """
    Run model training.

    Args:
        cache_snapshot: Name of cached gold snapshot
        config_path: Path to training config YAML
        resume_from: Optional checkpoint to resume from

    Returns:
        Dict with training results
    """
    print(f"Training from cache: {cache_snapshot}")
    print(f"Config: {config_path}")

    if resume_from:
        print(f"Resuming from: {resume_from}")

    # TODO: Implement training
    # 1. Load config
    # 2. Initialize model
    # 3. Load cache snapshot
    # 4. Create dataloaders
    # 5. Train loop
    # 6. Save checkpoints
    # 7. Log to MLflow

    print("  Training not yet implemented")
    print("  See legacy/src/training/ for reference implementation")

    return {
        "cache_snapshot": cache_snapshot,
        "config_path": str(config_path),
        "status": "not_implemented",
    }


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--cache-snapshot", required=True, help="Cached gold snapshot name")
    parser.add_argument("--config", type=Path, required=True, help="Training config")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint to resume from")

    args = parser.parse_args()

    result = train(
        cache_snapshot=args.cache_snapshot,
        config_path=args.config,
        resume_from=args.resume_from,
    )

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
