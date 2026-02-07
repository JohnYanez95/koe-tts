"""
Pull a batch of utterances for labeling.

Selects utterances from gold manifest, stratifies by pau count,
and creates a labeling session.

Usage:
    python -m modules.labeler.pipelines.pull_batch --batch-size 100 --strategy random

    Or via CLI:
    koe label pull --dataset jsut --batch-size 100
"""

import argparse
from pathlib import Path

from modules.labeler.app.data import (
    create_session,
    get_session_progress,
    list_datasets,
    load_manifest,
    stratify_utterances,
)


def pull_batch(
    batch_size: int = 100,
    strategy: str = "random",
    output_dir: Path | None = None,
    dataset: str | None = None,
    query: str | None = None,
) -> dict:
    """
    Pull a batch of utterances for labeling.

    Args:
        batch_size: Number of utterances to pull
        strategy: Sampling strategy (random, low_confidence, by_speaker)
        output_dir: Directory to write batch manifest
        dataset: Dataset name
        query: Filter query (unused in V1)

    Returns:
        Dict with batch info
    """
    # Resolve dataset
    if dataset is None:
        available = list_datasets()
        if not available:
            print("No datasets with gold manifests found.")
            return {"status": "error", "message": "no datasets"}
        dataset = available[0]["name"]
        print(f"Auto-selected dataset: {dataset}")

    # Show stratification stats
    utterances = load_manifest(dataset)
    if not utterances:
        print(f"No manifest found for dataset: {dataset}")
        return {"status": "error", "message": "no manifest"}

    strata = stratify_utterances(utterances)
    print(f"\nDataset: {dataset} ({len(utterances)} utterances)")
    print("Strata:")
    for s in sorted(strata.keys()):
        print(f"  {s} pau: {len(strata[s])} utterances")

    # Pick the most populated stratum
    best_stratum = max(strata, key=lambda k: len(strata[k]))
    print(f"\nPulling batch from stratum {best_stratum} (most utterances)")

    # Create session
    session = create_session(
        dataset=dataset,
        batch_size=batch_size,
        stratum=best_stratum,
    )

    progress = get_session_progress(session.session_id)

    print(f"\nSession created: {session.session_id}")
    print(f"  Batch size: {progress['total']}")
    print(f"  Stratum: {session.stratum}")
    print(f"\nTo start labeling: koe label serve {dataset}")

    return {
        "batch_size": progress["total"],
        "strategy": strategy,
        "session_id": session.session_id,
        "dataset": dataset,
        "stratum": session.stratum,
        "status": "success",
    }


def main():
    parser = argparse.ArgumentParser(description="Pull labeling batch")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--strategy", choices=["random", "low_confidence", "by_speaker"], default="random")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", type=str, default=None)

    args = parser.parse_args()

    result = pull_batch(
        batch_size=args.batch_size,
        strategy=args.strategy,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
