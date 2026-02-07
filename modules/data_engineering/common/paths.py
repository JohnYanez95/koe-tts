"""
Centralized path management for the koe-tts lakehouse.

All paths are derived from DATA_ROOT, which can be set via:
1. Environment variable: KOE_DATA_ROOT
2. Config file: configs/lakehouse/paths.yaml
3. Default: repo root (for local dev)

Path Contract:
    data/ingest/<dataset>/raw/       - Original downloads (zips/tars), checksums
    data/ingest/<dataset>/extracted/ - Extracted wav/flac + original sidecars
    data/assets/<dataset>/           - Non-regenerated assets (ground truth phonemes, speaker metadata, licenses)
    data/derived/<dataset>/          - Regenerated outputs (resampled, normalized)
    data/cache/<dataset>/<snapshot>/ - Cached training data (SSD-local)

    lake/bronze/<dataset>/           - Delta tables (raw ingested)
    lake/silver/<dataset>/           - Delta tables (cleaned, validated)
    lake/gold/<dataset>/             - Delta tables (train/val manifests)

    models/checkpoints/<run>/        - Training checkpoints
    models/exports/<run>/            - Exported models (ONNX, etc.)

    runs/data_engineering/           - DE pipeline artifacts
    runs/labeling/                   - Labeling batch artifacts
    runs/training/                   - Training run logs/metrics

Usage:
    from modules.data_engineering.common.paths import paths

    # Get paths
    paths.bronze / "jsut" / "utterances"
    paths.silver / "jsut" / "utterances"
    paths.gold / "jsut" / "train_manifest"

    # Dataset-specific
    paths.ingest_raw("jsut")      # -> data/ingest/jsut/raw/
    paths.ingest_extracted("jsut") # -> data/ingest/jsut/extracted/
    paths.derived("jsut")          # -> data/derived/jsut/
    paths.cache_snapshot("jsut", "20240115_v1")  # -> data/cache/jsut/20240115_v1/
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LakePaths:
    """
    Paths to lakehouse layers and artifacts.

    Two-tier storage model:
    - data_root (archive): G: drive, cold storage for completed runs
    - local_root (working): WSL local, hot storage for active training

    Training workflow:
    1. Clone checkpoint from runs_archive to runs (local)
    2. Train locally (fast I/O)
    3. Archive completed run from runs to runs_archive
    """

    data_root: Path
    local_root: Optional[Path] = None

    # Data paths (immutable ingest + assets + derived)
    data: Path = field(init=False)
    ingest: Path = field(init=False)
    assets: Path = field(init=False)
    derived: Path = field(init=False)

    # Lake paths (Delta tables)
    lake: Path = field(init=False)
    bronze: Path = field(init=False)
    silver: Path = field(init=False)
    gold: Path = field(init=False)

    # Model paths
    models: Path = field(init=False)
    checkpoints: Path = field(init=False)
    exports: Path = field(init=False)

    # Run paths - two-tier: local (active) + archive (completed)
    runs: Path = field(init=False)          # Local working dir (fast)
    runs_archive: Path = field(init=False)  # Archive on G: (cold storage)

    # Local cache (for training, may be on fast SSD)
    cache: Path = field(init=False)

    def __post_init__(self):
        """Derive all paths from data_root and local_root."""
        self.data_root = Path(self.data_root)
        if self.local_root is not None:
            self.local_root = Path(self.local_root)

        # Data (immutable inputs + assets + derived outputs)
        self.data = self.data_root / "data"
        self.ingest = self.data / "ingest"
        self.assets = self.data / "assets"
        self.derived = self.data / "derived"

        # Lake (Delta tables)
        self.lake = self.data_root / "lake"
        self.bronze = self.lake / "bronze"
        self.silver = self.lake / "silver"
        self.gold = self.lake / "gold"

        # Models
        self.models = self.data_root / "models"
        self.checkpoints = self.models / "checkpoints"
        self.exports = self.models / "exports"

        # Runs - two-tier storage
        # Archive: completed runs on G: drive (cold)
        self.runs_archive = self.data_root / "runs"
        # Local: active training on fast local storage
        if self.local_root is not None:
            self.runs = self.local_root / "runs"
        else:
            self.runs = self.runs_archive  # Fallback: same location

        # Local cache (defaults to data_root/data/cache, can override in config)
        self.cache = self.data / "cache"

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for path in [
            self.ingest, self.assets, self.derived,
            self.bronze, self.silver, self.gold,
            self.checkpoints, self.exports,
            self.runs, self.runs_archive, self.cache,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # === Ingest paths (raw downloads + extracted) ===

    def ingest_raw(self, dataset: str) -> Path:
        """Get path to raw dataset downloads (zips, tars, checksums)."""
        return self.ingest / dataset / "raw"

    def ingest_extracted(self, dataset: str) -> Path:
        """Get path to extracted dataset (wav/flac + sidecars)."""
        return self.ingest / dataset / "extracted"

    # Backwards compat aliases
    def dataset_raw(self, dataset: str, version: str = "v1") -> Path:
        """Deprecated: use ingest_raw() instead."""
        return self.ingest_raw(dataset) / version

    def dataset_extracted(self, dataset: str, version: str = "v1") -> Path:
        """Deprecated: use ingest_extracted() instead."""
        return self.ingest_extracted(dataset) / version

    # === Asset paths (non-regenerated: ground truth, metadata, licenses) ===

    def dataset_assets(self, dataset: str) -> Path:
        """Get path to dataset assets (ground truth phonemes, speaker metadata, licenses)."""
        return self.assets / dataset

    # === Derived paths (regenerated outputs) ===

    def derived_dataset(self, dataset: str) -> Path:
        """Get path to derived outputs for a dataset (resampled, normalized, etc.)."""
        return self.derived / dataset

    # === Cache paths (local training data) ===

    def cache_snapshot(self, dataset: str, snapshot_id: str) -> Path:
        """Get path to a cached training data snapshot."""
        return self.cache / dataset / snapshot_id

    def cache_latest(self, dataset: str) -> Path:
        """Get path to latest cache for a dataset (symlink or direct)."""
        return self.cache / dataset / "latest"

    # === Lake table paths ===

    def bronze_table(self, name: str) -> Path:
        """Get path to a bronze Delta table."""
        return self.bronze / name

    def silver_table(self, name: str) -> Path:
        """Get path to a silver Delta table."""
        return self.silver / name

    def gold_table(self, name: str) -> Path:
        """Get path to a gold Delta table."""
        return self.gold / name

    # === Run paths ===

    def run_dir(self, run_type: str, run_name: str) -> Path:
        """
        Get path to a run directory.

        Args:
            run_type: One of "data_engineering", "labeling", "training"
            run_name: Name/ID of the run
        """
        return self.runs / run_type / run_name

    def de_run(self, run_id: str) -> Path:
        """Get data engineering run directory."""
        return self.runs / "data_engineering" / run_id

    def labeling_run(self, batch_id: str) -> Path:
        """Get labeling batch directory."""
        return self.runs / "labeling" / batch_id

    def training_run(self, run_name: str) -> Path:
        """Get training run directory."""
        return self.runs / "training" / run_name

    # === Model paths ===

    def checkpoint_dir(self, run_name: str) -> Path:
        """Get checkpoint directory for a training run."""
        return self.checkpoints / run_name

    def export_dir(self, run_name: str) -> Path:
        """Get export directory for a training run."""
        return self.exports / run_name


@dataclass
class TrainRunPaths:
    """
    Paths for a training run.

    Splits artifacts between:
    - checkpoints: models/checkpoints/<run_name>/ (large, durable)
    - logs: runs/training/<run_name>/ (small metadata, ephemeral)
    """

    run_name: str
    lake_paths: LakePaths = field(default_factory=lambda: paths)

    # Checkpoint paths (large, under models/)
    checkpoints: Path = field(init=False)

    # Log/metadata paths (small, under runs/)
    logs: Path = field(init=False)
    samples: Path = field(init=False)
    config: Path = field(init=False)
    metrics: Path = field(init=False)
    cache_snapshot_ref: Path = field(init=False)

    def __post_init__(self):
        # Checkpoints go to models/ (large, durable)
        self.checkpoints = self.lake_paths.checkpoint_dir(self.run_name)

        # Logs/metadata go to runs/ (small, can be deleted)
        run_dir = self.lake_paths.training_run(self.run_name)
        self.logs = run_dir / "logs"
        self.samples = run_dir / "samples"
        self.config = run_dir / "config.yaml"
        self.metrics = run_dir / "metrics.jsonl"
        self.cache_snapshot_ref = run_dir / "cache_snapshot.json"

    def ensure_dirs(self) -> None:
        """Create run directories."""
        for path in [self.checkpoints, self.logs, self.samples]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints / "best.ckpt"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints / "last.ckpt"


def _load_config_paths() -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Load DATA_ROOT, LOCAL_ROOT, and cache path from configs/lakehouse/paths.yaml.

    Returns:
        Tuple of (data_root, local_root, cache_path), any can be None
    """
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "lakehouse" / "paths.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if config:
                data_root = None
                local_root = None
                cache_path = None

                if "data_root" in config:
                    root = os.path.expandvars(config["data_root"])
                    root = os.path.expanduser(root)
                    data_root = Path(root)

                if "local_root" in config:
                    local = os.path.expandvars(config["local_root"])
                    local = os.path.expanduser(local)
                    local_root = Path(local)

                if "cache_root" in config:
                    cache = os.path.expandvars(config["cache_root"])
                    cache = os.path.expanduser(cache)
                    cache_path = Path(cache)

                return data_root, local_root, cache_path
    return None, None, None


def _get_roots() -> tuple[Path, Optional[Path]]:
    """
    Determine DATA_ROOT and LOCAL_ROOT in order of precedence:
    1. KOE_DATA_ROOT / KOE_LOCAL_ROOT environment variables
    2. configs/lakehouse/paths.yaml
    3. Default to repo-relative (data_root only)

    Returns:
        Tuple of (data_root, local_root)
    """
    repo_root = Path(__file__).parent.parent.parent.parent

    # Load from config
    config_data_root, config_local_root, _ = _load_config_paths()

    # Data root: env > config > repo
    env_data_root = os.environ.get("KOE_DATA_ROOT")
    if env_data_root:
        data_root = Path(env_data_root)
    elif config_data_root:
        data_root = config_data_root
    else:
        data_root = repo_root

    # Local root: env > config > None (falls back to data_root in LakePaths)
    env_local_root = os.environ.get("KOE_LOCAL_ROOT")
    if env_local_root:
        local_root = Path(env_local_root)
    elif config_local_root:
        local_root = config_local_root
    else:
        local_root = None

    return data_root, local_root


# Singleton instance
_data_root, _local_root = _get_roots()
paths = LakePaths(_data_root, _local_root)


def get_paths(data_root: Optional[Path] = None) -> LakePaths:
    """
    Get LakePaths instance, optionally with custom data_root.

    Args:
        data_root: Override the default data root. If None, uses
                   env var / config / default.

    Returns:
        LakePaths instance with all derived paths.
    """
    if data_root is not None:
        return LakePaths(data_root)
    return paths


# =============================================================================
# Run Archive/Clone Helpers
# =============================================================================

import shutil


def clone_checkpoint_to_local(
    checkpoint_path: Path,
    run_id: Optional[str] = None,
) -> Path:
    """
    Clone a checkpoint from archive to local storage for training.

    If checkpoint is already on local storage, returns as-is.
    If checkpoint is on archive (G:), copies to local runs dir.

    Args:
        checkpoint_path: Path to checkpoint file (can be archive or local)
        run_id: Optional run ID to organize under (default: extracted from path)

    Returns:
        Path to checkpoint on local storage
    """
    checkpoint_path = Path(checkpoint_path)

    # Already local?
    if paths.runs != paths.runs_archive:
        try:
            checkpoint_path.relative_to(paths.runs)
            return checkpoint_path  # Already local
        except ValueError:
            pass  # Not under local runs

    # Extract run_id from path if not provided
    if run_id is None:
        # Path like: .../runs/<run_id>/checkpoints/<ckpt>.pt
        for parent in checkpoint_path.parents:
            if parent.parent.name == "runs" or (parent.parent.parent and parent.parent.parent.name == "runs"):
                run_id = parent.name
                break
        if run_id is None:
            run_id = "cloned"

    # Create local destination
    local_ckpt_dir = paths.runs / run_id / "checkpoints"
    local_ckpt_dir.mkdir(parents=True, exist_ok=True)
    local_ckpt = local_ckpt_dir / checkpoint_path.name

    # Copy if not exists or source is newer
    if not local_ckpt.exists() or checkpoint_path.stat().st_mtime > local_ckpt.stat().st_mtime:
        print(f"Cloning checkpoint to local: {checkpoint_path.name}")
        shutil.copy2(checkpoint_path, local_ckpt)

    return local_ckpt


def archive_run(
    run_id: str,
    delete_local: bool = True,
) -> Path:
    """
    Archive a completed run from local to archive storage.

    Moves (or copies) the entire run directory from local to archive.

    Args:
        run_id: Run ID to archive
        delete_local: If True, delete local copy after archiving

    Returns:
        Path to archived run
    """
    local_run = paths.runs / run_id
    archive_run = paths.runs_archive / run_id

    if not local_run.exists():
        raise FileNotFoundError(f"Local run not found: {local_run}")

    if paths.runs == paths.runs_archive:
        print(f"Local and archive are same location, skipping archive for {run_id}")
        return local_run

    # Archive destination
    archive_run.parent.mkdir(parents=True, exist_ok=True)

    if archive_run.exists():
        # Merge: copy new files, skip existing
        print(f"Merging {run_id} into existing archive...")
        for item in local_run.rglob("*"):
            if item.is_file():
                rel = item.relative_to(local_run)
                dest = archive_run / rel
                if not dest.exists() or item.stat().st_mtime > dest.stat().st_mtime:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
    else:
        # Fresh archive
        print(f"Archiving {run_id} to {archive_run}...")
        if delete_local:
            shutil.move(str(local_run), str(archive_run))
        else:
            shutil.copytree(local_run, archive_run)

    # Delete local if requested and we copied (not moved)
    if delete_local and local_run.exists():
        shutil.rmtree(local_run)
        print(f"Deleted local copy: {local_run}")

    return archive_run


def list_archived_runs() -> list[str]:
    """List run IDs in archive storage."""
    if not paths.runs_archive.exists():
        return []
    return sorted([
        p.name for p in paths.runs_archive.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])


def list_local_runs() -> list[str]:
    """List run IDs in local storage."""
    if not paths.runs.exists():
        return []
    return sorted([
        p.name for p in paths.runs.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])
