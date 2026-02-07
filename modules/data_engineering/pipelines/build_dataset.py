"""
End-to-end dataset build pipeline.

Runs: ingest → bronze → silver → gold in sequence.

Usage:
    koe build jsut
    koe build jsut --skip-ingest
    python -m modules.data_engineering.pipelines.build_dataset --dataset jsut
"""

import argparse
from typing import Optional


def build_dataset(
    dataset: str,
    gold_version: Optional[str] = None,
    skip_ingest: bool = False,
    skip_bronze: bool = False,
    skip_silver: bool = False,
    skip_gold: bool = False,
) -> dict:
    """
    Build dataset end-to-end: ingest → bronze → silver → gold.

    Args:
        dataset: Dataset name (jsut, jvs, common_voice, all)
        gold_version: Gold snapshot ID (auto-generated if None)
        skip_ingest: Skip ingest step (assume already extracted)
        skip_bronze: Skip bronze step
        skip_silver: Skip silver step
        skip_gold: Skip gold step

    Returns:
        Dict with results from each step
    """
    results = {}
    datasets = ["jsut", "jvs", "common_voice"] if dataset == "all" else [dataset]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Building dataset: {ds}")
        print(f"{'='*60}")

        # Step 1: Ingest
        if not skip_ingest:
            print(f"\n[1/4] Ingest: {ds}")
            results[f"{ds}_ingest"] = run_ingest(ds)
        else:
            print(f"\n[1/4] Ingest: SKIPPED")
            results[f"{ds}_ingest"] = "skipped"

        # Step 2: Bronze
        if not skip_bronze:
            print(f"\n[2/4] Bronze: {ds}")
            results[f"{ds}_bronze"] = run_bronze(ds)
        else:
            print(f"\n[2/4] Bronze: SKIPPED")
            results[f"{ds}_bronze"] = "skipped"

        # Step 3: Silver
        if not skip_silver:
            print(f"\n[3/4] Silver: {ds}")
            results[f"{ds}_silver"] = run_silver(ds)
        else:
            print(f"\n[3/4] Silver: SKIPPED")
            results[f"{ds}_silver"] = "skipped"

        # Step 4: Gold
        if not skip_gold:
            print(f"\n[4/4] Gold: {ds} (snapshot: {gold_version or 'auto'})")
            results[f"{ds}_gold"] = run_gold(ds, gold_version)
        else:
            print(f"\n[4/4] Gold: SKIPPED")
            results[f"{ds}_gold"] = "skipped"

    return results


def run_ingest(dataset: str) -> dict:
    """Run ingest for a dataset."""
    if dataset == "jsut":
        try:
            from modules.data_engineering.ingest.jsut import ingest_jsut
            result = ingest_jsut()
            return {"status": "success", "manifest": result.get("manifest_path")}
        except FileNotFoundError as e:
            print(f"  Ingest failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Ingest failed: {e}")
            return {"status": "error", "error": str(e)}
    elif dataset == "jvs":
        try:
            from modules.data_engineering.ingest.jvs import ingest_jvs
            result = ingest_jvs()
            return {"status": "success", "manifest": result.get("manifest_path")}
        except FileNotFoundError as e:
            print(f"  Ingest failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Ingest failed: {e}")
            return {"status": "error", "error": str(e)}
    else:
        print(f"  Ingest not implemented for {dataset}")
        return {"status": "not_implemented"}


def run_bronze(dataset: str) -> dict:
    """Run bronze for a dataset."""
    if dataset == "jsut":
        try:
            from modules.data_engineering.bronze.jsut import build_bronze_jsut
            result = build_bronze_jsut()
            return {
                "status": "success",
                "record_count": result.get("stats", {}).get("total_count"),
            }
        except FileNotFoundError as e:
            print(f"  Bronze failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Bronze failed: {e}")
            raise
    elif dataset == "jvs":
        try:
            from modules.data_engineering.bronze.jvs import build_bronze_jvs
            result = build_bronze_jvs()
            return {
                "status": "success",
                "record_count": result.get("stats", {}).get("total_count"),
            }
        except FileNotFoundError as e:
            print(f"  Bronze failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Bronze failed: {e}")
            raise
    else:
        print(f"  Bronze not implemented for {dataset}")
        return {"status": "not_implemented"}


def run_silver(dataset: str) -> dict:
    """Run silver for a dataset."""
    if dataset == "jsut":
        try:
            from modules.data_engineering.silver.jsut import build_silver_jsut
            result = build_silver_jsut()
            return {
                "status": "success",
                "record_count": result.get("stats", {}).get("record_count"),
            }
        except FileNotFoundError as e:
            print(f"  Silver failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Silver failed: {e}")
            raise
    elif dataset == "jvs":
        try:
            from modules.data_engineering.silver.jvs import build_silver_jvs
            result = build_silver_jvs()
            return {
                "status": "success",
                "record_count": result.get("stats", {}).get("record_count"),
            }
        except FileNotFoundError as e:
            print(f"  Silver failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Silver failed: {e}")
            raise
    else:
        print(f"  Silver not implemented for {dataset}")
        return {"status": "not_implemented"}


def run_gold(dataset: str, snapshot_id: Optional[str] = None) -> dict:
    """Run gold for a dataset."""
    if dataset == "jsut":
        try:
            from modules.data_engineering.gold.jsut import build_gold_jsut
            result = build_gold_jsut(snapshot_id=snapshot_id)
            return {
                "status": "success",
                "snapshot_id": result.get("snapshot_id"),
                "record_count": result.get("validation_stats", {}).get("total_count"),
                "manifest_path": result.get("manifest_path"),
            }
        except FileNotFoundError as e:
            print(f"  Gold failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Gold failed: {e}")
            raise
    elif dataset == "jvs":
        try:
            from modules.data_engineering.gold.jvs import build_gold_jvs
            result = build_gold_jvs(snapshot_id=snapshot_id)
            return {
                "status": "success",
                "snapshot_id": result.get("snapshot_id"),
                "record_count": result.get("validation_stats", {}).get("total_count"),
                "manifest_path": result.get("manifest_path"),
            }
        except FileNotFoundError as e:
            print(f"  Gold failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            print(f"  Gold failed: {e}")
            raise
    else:
        print(f"  Gold not implemented for {dataset}")
        return {"status": "not_implemented"}


def main():
    parser = argparse.ArgumentParser(description="Build dataset end-to-end")
    parser.add_argument(
        "--dataset",
        choices=["jsut", "jvs", "common_voice", "all"],
        default="jsut",
        help="Dataset to build",
    )
    parser.add_argument("--gold-version", help="Gold snapshot ID")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-bronze", action="store_true")
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")

    args = parser.parse_args()

    results = build_dataset(
        dataset=args.dataset,
        gold_version=args.gold_version,
        skip_ingest=args.skip_ingest,
        skip_bronze=args.skip_bronze,
        skip_silver=args.skip_silver,
        skip_gold=args.skip_gold,
    )

    print("\n" + "=" * 60)
    print("Build Results:")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, dict):
            status = value.get("status", "unknown")
            detail = value.get("record_count") or value.get("error") or ""
            print(f"  {key}: {status} {f'({detail})' if detail else ''}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
