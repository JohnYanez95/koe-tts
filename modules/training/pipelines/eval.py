"""
Evaluation pipeline.

Usage:
    python -m modules.training.pipelines.eval --checkpoint runs/jsut-v1/checkpoints/best.ckpt
"""

import argparse
from pathlib import Path


def evaluate(
    checkpoint_path: Path,
    eval_set: str = "test",
) -> dict:
    """
    Evaluate a trained checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        eval_set: Evaluation set to use (test, val, or custom)

    Returns:
        Dict with evaluation metrics
    """
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Eval set: {eval_set}")

    # TODO: Implement evaluation
    # 1. Load checkpoint
    # 2. Load eval set from gold.eval_sets
    # 3. Generate audio for each prompt
    # 4. Compute metrics (MCD, F0 RMSE, etc.)
    # 5. Log results

    print("  Evaluation not yet implemented")

    return {
        "checkpoint_path": str(checkpoint_path),
        "eval_set": eval_set,
        "status": "not_implemented",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-set", default="test")

    args = parser.parse_args()

    result = evaluate(
        checkpoint_path=args.checkpoint,
        eval_set=args.eval_set,
    )

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
