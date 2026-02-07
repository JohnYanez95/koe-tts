"""
CLI for MLflow model registry operations.

Usage:
    python -m modules.registry.cli list-models
    python -m modules.registry.cli list-versions tts-ja-vits-jsut
    python -m modules.registry.cli promote tts-ja-vits-jsut --from-alias best --to-alias prod
"""

import argparse
from typing import Optional

from .config import configure_mlflow
from .models import (
    get_latest_version,
    get_model_version,
    list_model_versions,
    promote_to_prod,
    set_alias,
)


def cmd_list_models(args):
    """List all registered models."""
    import mlflow

    configure_mlflow()
    client = mlflow.tracking.MlflowClient()

    models = client.search_registered_models()
    if not models:
        print("No registered models found.")
        return

    print(f"{'Model Name':<30} {'Latest Version':<10} {'Aliases'}")
    print("-" * 60)
    for model in models:
        latest = get_latest_version(model.name)
        version = latest.version if latest else "-"
        aliases = ", ".join(model.aliases.keys()) if model.aliases else "-"
        print(f"{model.name:<30} {version:<10} {aliases}")


def cmd_list_versions(args):
    """List versions of a model."""
    configure_mlflow()
    versions = list_model_versions(args.model_name, max_results=args.limit)

    if not versions:
        print(f"No versions found for model: {args.model_name}")
        return

    print(f"Versions of {args.model_name}:")
    print(f"{'Version':<10} {'Aliases':<20} {'Created':<25} {'Description'}")
    print("-" * 80)
    for v in versions:
        aliases = ", ".join(v.aliases) if v.aliases else "-"
        created = v.creation_timestamp
        desc = (v.description or "-")[:30]
        print(f"{v.version:<10} {aliases:<20} {created:<25} {desc}")


def cmd_get_version(args):
    """Get details of a specific version."""
    configure_mlflow()

    if args.alias:
        version = get_model_version(args.model_name, alias=args.alias)
    else:
        version = get_model_version(args.model_name, version=args.version)

    print(f"Model: {version.name}")
    print(f"Version: {version.version}")
    print(f"Aliases: {', '.join(version.aliases) if version.aliases else '-'}")
    print(f"Source: {version.source}")
    print(f"Run ID: {version.run_id}")
    print(f"Created: {version.creation_timestamp}")
    if version.description:
        print(f"Description: {version.description}")


def cmd_set_alias(args):
    """Set an alias for a model version."""
    configure_mlflow()
    set_alias(args.model_name, args.version, args.alias)
    print(f"Set alias '{args.alias}' -> {args.model_name} v{args.version}")


def cmd_promote(args):
    """Promote a model version to production."""
    configure_mlflow()
    version = promote_to_prod(args.model_name, from_alias=args.from_alias)
    print(f"Promoted {args.model_name} v{version.version} to 'prod'")


def main():
    parser = argparse.ArgumentParser(
        description="MLflow model registry CLI",
        prog="python -m modules.registry.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-models
    subparsers.add_parser("list-models", help="List all registered models")

    # list-versions
    lv_parser = subparsers.add_parser("list-versions", help="List versions of a model")
    lv_parser.add_argument("model_name", help="Model name")
    lv_parser.add_argument("--limit", type=int, default=20, help="Max versions to show")

    # get-version
    gv_parser = subparsers.add_parser("get-version", help="Get version details")
    gv_parser.add_argument("model_name", help="Model name")
    gv_parser.add_argument("--version", type=int, help="Version number")
    gv_parser.add_argument("--alias", help="Alias name (alternative to version)")

    # set-alias
    sa_parser = subparsers.add_parser("set-alias", help="Set alias for a version")
    sa_parser.add_argument("model_name", help="Model name")
    sa_parser.add_argument("version", type=int, help="Version number")
    sa_parser.add_argument("alias", help="Alias name")

    # promote
    pr_parser = subparsers.add_parser("promote", help="Promote to production")
    pr_parser.add_argument("model_name", help="Model name")
    pr_parser.add_argument("--from-alias", default="best", help="Source alias (default: best)")

    args = parser.parse_args()

    if args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "list-versions":
        cmd_list_versions(args)
    elif args.command == "get-version":
        cmd_get_version(args)
    elif args.command == "set-alias":
        cmd_set_alias(args)
    elif args.command == "promote":
        cmd_promote(args)


if __name__ == "__main__":
    main()
