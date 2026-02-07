"""
Data layer for the segmentation labeling app.

Handles:
- Loading gold utterance manifests (JSONL)
- Loading auto-detected breakpoints from silver segment_breaks
- Enumerating pau tokens and mapping them to auto-breakpoints
- Stratifying utterances by pau count
- Session creation and persistence

Label schema v1 (per utterance):
    {
        "utterance_id": "...",
        "breaks": [
            {
                "pau_idx": 1,
                "token_position": 8,
                "ms_proposed": 630,
                "ms": 680,
                "delta_ms": 50,
                "use_break": true,
                "noise_zone_ms": 750  // optional: (b) marker for noise zone boundary
            }
        ],
        "trim_start_ms": 100,
        "trim_end_ms": 4500,
        "label_schema_version": 1,
        "heuristic_version": "pau_v1_adaptive",
        "heuristic_params_hash": "sha1:...",
        "sample_rate": 22050,
        "labeled_at": "...",
        "status": "labeled"
    }
"""

import json
import random
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from modules.data_engineering.common.paths import paths

LABEL_SCHEMA_VERSION = 1


@dataclass
class PauBreak:
    """A pau token with its labeling state and mapped position."""

    pau_idx: int  # 1-indexed pau number within this utterance
    token_position: int  # 0-indexed position in the phoneme token list
    ms: int | None = None  # current time position (user-adjusted or heuristic)
    ms_proposed: int | None = None  # original heuristic proposal
    use_break: bool = False  # user decision: split here or not
    noise_zone_ms: int | None = None  # (b) marker for noise zone boundary


@dataclass
class HeuristicMeta:
    """Metadata about the heuristic that produced auto-breakpoints."""

    version: str = "unknown"
    params_hash: str = "unknown"


@dataclass
class UtteranceItem:
    """A single utterance for labeling."""

    utterance_id: str
    text: str
    phonemes: str
    audio_relpath: str
    audio_abspath: str
    duration_sec: float
    speaker_id: str | None = None
    sample_rate: int = 22050
    pau_count: int = 0
    pau_breaks: list[PauBreak] = field(default_factory=list)
    trim_start_ms: int | None = None
    trim_end_ms: int | None = None
    status: str = "pending"  # "pending", "labeled", "skipped"


@dataclass
class Session:
    """A labeling session."""

    session_id: str
    dataset: str
    created_at: str
    batch_size: int
    stratum: int | None  # pau count stratum (0, 1, 2, 3+), None = all
    utterance_ids: list[str] = field(default_factory=list)
    heuristic_version: str = "unknown"
    heuristic_params_hash: str = "unknown"


def _sessions_root() -> Path:
    """Get the root directory for labeling sessions."""
    return paths.runs / "labeling"


def load_manifest(dataset: str) -> list[dict]:
    """
    Load the gold utterance manifest for a dataset.

    Reads the latest JSONL manifest from lake/gold/{dataset}/manifests/.
    """
    manifest_dir = paths.gold / dataset / "manifests"
    if not manifest_dir.exists():
        return []

    candidates = sorted(manifest_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return []

    train_manifest = manifest_dir / "train.jsonl"
    manifest_path = train_manifest if train_manifest.exists() else candidates[0]

    utterances = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterances.append(json.loads(line))

    return utterances


def load_auto_breakpoints(
    dataset: str,
    heuristic_params_hash: str | None = None,
) -> tuple[dict[str, list[int]], HeuristicMeta]:
    """
    Load auto-detected breakpoints.

    Priority:
    1. Heuristic JSONL cache (runs/labeling/heuristics/) — fast iteration
    2. Silver segment_breaks Delta table — production path

    Args:
        heuristic_params_hash: If set, load a specific heuristic run.
            None = most recent.

    Returns (mapping of utterance_id -> breakpoints_ms, heuristic metadata).
    """
    # Priority 1: heuristic cache (lightweight, no Spark/Delta)
    try:
        from modules.labeler.heuristic import load_heuristic_cache

        cache, method, params_hash = load_heuristic_cache(dataset, heuristic_params_hash)
        if cache:
            return cache, HeuristicMeta(version=method, params_hash=params_hash)
    except ImportError:
        pass

    # Priority 2: silver Delta table
    breaks_path = paths.silver / dataset / "segment_breaks"
    if not breaks_path.exists():
        return {}, HeuristicMeta()

    try:
        import deltalake

        dt = deltalake.DeltaTable(str(breaks_path))
        df = dt.to_pandas()
        result = {}
        meta = HeuristicMeta()
        for _, row in df.iterrows():
            uid = row.get("utterance_id", "")
            bps = row.get("breakpoints_ms", [])
            if uid and bps is not None:
                if isinstance(bps, str):
                    bps = json.loads(bps)
                result[uid] = list(bps) if bps else []
            if meta.version == "unknown":
                meta.version = row.get("method", "unknown")
                meta.params_hash = row.get("params_hash", "unknown")
        return result, meta
    except (ImportError, Exception):
        pass

    try:
        import pyarrow.parquet as pq

        parquet_files = list(breaks_path.glob("*.parquet"))
        if not parquet_files:
            return {}, HeuristicMeta()

        result = {}
        meta = HeuristicMeta()
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                uid = row.get("utterance_id", "")
                bps = row.get("breakpoints_ms", [])
                if uid and bps is not None:
                    if isinstance(bps, str):
                        bps = json.loads(bps)
                    result[uid] = list(bps) if bps else []
                if meta.version == "unknown":
                    meta.version = row.get("method", "unknown")
                    meta.params_hash = row.get("params_hash", "unknown")
        return result, meta
    except (ImportError, Exception):
        return {}, HeuristicMeta()


def load_auto_trim(
    dataset: str,
    trim_params_hash: str | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Load auto-trim predictions from the trim heuristic cache.

    Returns dict of utterance_id -> (trim_start_ms, trim_end_ms).
    Empty dict if no trim cache exists or cannot be read.
    """
    try:
        from modules.labeler.heuristic import load_trim_cache

        cache, method, params_hash = load_trim_cache(dataset, trim_params_hash)
        if cache:
            print(f"[data] Auto-trim loaded: {len(cache)} utterances, "
                  f"method={method}, hash={params_hash}")
        return cache
    except Exception as _e:
        print(f"[data] WARN: failed to load auto-trim for {dataset}: {repr(_e)}")
        return {}


def _enumerate_pau(phonemes: str) -> list[dict]:
    """
    Enumerate pau tokens in a phoneme string.

    Returns list of {pau_idx (1-indexed), token_position (0-indexed)}.
    """
    if not phonemes:
        return []
    tokens = phonemes.split()
    paus = []
    pau_num = 0
    for i, token in enumerate(tokens):
        if token == "pau":
            pau_num += 1
            paus.append({"pau_idx": pau_num, "token_position": i})
    return paus


def _count_pau(phonemes: str) -> int:
    """Count pau tokens in a phoneme string."""
    if not phonemes:
        return 0
    return sum(1 for token in phonemes.split() if token == "pau")


def _map_pau_to_breakpoints(
    pau_list: list[dict],
    auto_breakpoints_ms: list[int],
    num_tokens: int,
    duration_ms: int,
) -> list[PauBreak]:
    """
    Map enumerated pau tokens to time positions.

    Strategy:
    1. Compute expected_ms for each pau from its token position (uniform prior).
    2. Match auto-breakpoints to paus by nearest expected position (greedy,
       closest-first). This avoids misordering when breakpoint count != pau count.
    3. Unmatched paus keep their uniform prior position.

    This ensures every pau always gets an ms position, even without
    silver segment_breaks data, and markers stay in sequential order.
    """
    # Compute expected ms for each pau from token position
    expected: list[int | None] = []
    for pau in pau_list:
        if num_tokens > 0 and duration_ms > 0:
            expected.append(round(duration_ms * pau["token_position"] / num_tokens))
        else:
            expected.append(None)

    # Match breakpoints to paus by nearest expected position
    assignments: list[int | None] = [None] * len(pau_list)
    available_bps = sorted(auto_breakpoints_ms)

    if available_bps and any(e is not None for e in expected):
        # Build (distance, bp_idx, pau_idx) pairs and sort by distance
        pairs = []
        for bi, bp in enumerate(available_bps):
            for pi, exp in enumerate(expected):
                if exp is not None:
                    pairs.append((abs(bp - exp), bi, pi))
        pairs.sort()

        used_bps: set[int] = set()
        used_paus: set[int] = set()
        for _dist, bi, pi in pairs:
            if bi not in used_bps and pi not in used_paus:
                assignments[pi] = available_bps[bi]
                used_bps.add(bi)
                used_paus.add(pi)

    # Build result: use matched breakpoint or fall back to expected position
    result = []
    for i, pau in enumerate(pau_list):
        ms = assignments[i] if assignments[i] is not None else expected[i]

        result.append(
            PauBreak(
                pau_idx=pau["pau_idx"],
                token_position=pau["token_position"],
                ms=ms,
                ms_proposed=ms,  # heuristic proposal = initial mapping
                use_break=False,  # all start as not-yet-reviewed
            )
        )

    return result


def stratify_utterances(utterances: list[dict]) -> dict[int, list[dict]]:
    """
    Group utterances into strata by pau count.

    Strata: 0, 1, 2, 3 (3 = 3+)
    Within each stratum, ordered by duration ascending.
    """
    strata: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}

    for utt in utterances:
        pau_count = _count_pau(utt.get("phonemes", ""))
        stratum = min(pau_count, 3)
        strata[stratum].append(utt)

    for stratum in strata:
        strata[stratum].sort(key=lambda u: u.get("duration_sec", 0))

    return strata


def create_session(
    dataset: str,
    batch_size: int = 25,
    stratum: int | None = None,
    heuristic_only: bool = False,
    heuristic_params_hash: str | None = None,
) -> Session:
    """
    Create a new labeling session.

    Args:
        stratum: pau count stratum to filter by (None = random sample from all)
        heuristic_params_hash: Specific heuristic run to use. None = most recent.
    """
    session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    utterances = load_manifest(dataset)
    if not utterances:
        raise ValueError(f"No manifest found for dataset: {dataset}")

    # If heuristic_only, restrict to utterances with detected breakpoints
    if heuristic_only:
        auto_bps, _ = load_auto_breakpoints(dataset, heuristic_params_hash)
        heuristic_hit_ids = {uid for uid, bps in auto_bps.items() if bps}
        utterances = [u for u in utterances if u.get("utterance_id") in heuristic_hit_ids]
        if not utterances:
            raise ValueError(f"No utterances with heuristic breakpoints for dataset: {dataset}")

    # Filter by stratum if specified, otherwise use all
    if stratum is not None:
        strata = stratify_utterances(utterances)
        pool = strata.get(stratum, [])
        if not pool:
            stratum = max(strata, key=lambda k: len(strata[k]))
            pool = strata[stratum]
    else:
        pool = utterances

    assigned_ids = _get_all_assigned_ids(dataset)
    published_ids = get_published_utterance_ids(dataset)
    excluded_ids = assigned_ids | published_ids
    available = [u for u in pool if u.get("utterance_id") not in excluded_ids]
    if not available:
        available = pool

    # Random sample (deterministic per session for reproducibility)
    rng = random.Random(session_id)
    if len(available) > batch_size:
        batch = rng.sample(available, batch_size)
    else:
        batch = available
    utterance_ids = [u["utterance_id"] for u in batch]

    # Load heuristic metadata for this dataset
    _, heuristic_meta = load_auto_breakpoints(dataset, heuristic_params_hash)

    session = Session(
        session_id=session_id,
        dataset=dataset,
        created_at=datetime.now(UTC).isoformat(),
        batch_size=batch_size,
        stratum=stratum,
        utterance_ids=utterance_ids,
        heuristic_version=heuristic_meta.version,
        heuristic_params_hash=heuristic_meta.params_hash,
    )

    session_dir = _sessions_root() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    with open(session_dir / "session.json", "w") as f:
        json.dump(
            {
                "session_id": session.session_id,
                "dataset": session.dataset,
                "created_at": session.created_at,
                "batch_size": session.batch_size,
                "stratum": session.stratum,
                "utterance_ids": session.utterance_ids,
                "heuristic_version": session.heuristic_version,
                "heuristic_params_hash": session.heuristic_params_hash,
            },
            f,
            indent=2,
        )

    (session_dir / "labels.jsonl").touch()

    return session


def _get_all_assigned_ids(dataset: str) -> set[str]:
    """
    Get all utterance IDs that are assigned to any session for this dataset.

    Includes both labeled AND pending (not-yet-labeled) utterances in active
    sessions. This prevents double-assignment when creating new sessions.
    """
    assigned = set()
    sessions_dir = _sessions_root()
    if not sessions_dir.exists():
        return assigned

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue

        with open(session_file) as f:
            meta = json.load(f)

        if meta.get("dataset") != dataset:
            continue

        # Include ALL utterances assigned to this session
        for uid in meta.get("utterance_ids", []):
            assigned.add(uid)

    return assigned


def load_session(session_id: str) -> Session:
    """Load an existing session from disk."""
    session_dir = _sessions_root() / session_id
    session_file = session_dir / "session.json"

    if not session_file.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    with open(session_file) as f:
        meta = json.load(f)

    return Session(
        session_id=meta["session_id"],
        dataset=meta["dataset"],
        created_at=meta["created_at"],
        batch_size=meta["batch_size"],
        stratum=meta.get("stratum"),
        utterance_ids=meta["utterance_ids"],
        heuristic_version=meta.get("heuristic_version", "unknown"),
        heuristic_params_hash=meta.get("heuristic_params_hash", "unknown"),
    )


def get_batch(session_id: str) -> tuple[list[UtteranceItem], Session]:
    """
    Get the batch of utterances for a session.

    Enumerates pau tokens, maps to auto-breakpoints, merges saved labels.
    Trim precedence: saved trims > auto-trim predictions > None.
    Returns (items, session) so callers can access session metadata.
    """
    session = load_session(session_id)
    manifest = load_manifest(session.dataset)
    auto_bps, _ = load_auto_breakpoints(session.dataset)
    auto_trim_by_uid = load_auto_trim(session.dataset)

    manifest_by_id = {u["utterance_id"]: u for u in manifest}

    saved_labels = _load_saved_labels(session_id)
    label_statuses = _load_label_statuses(session_id)

    items = []
    for uid in session.utterance_ids:
        utt = manifest_by_id.get(uid)
        if utt is None:
            continue

        phonemes = utt.get("phonemes", "")
        pau_list = _enumerate_pau(phonemes)
        auto_bp_list = auto_bps.get(uid, [])
        num_tokens = len(phonemes.split()) if phonemes else 0
        duration_ms = round(utt.get("duration_sec", 0.0) * 1000)

        # Build pau_breaks from auto-mapping (falls back to uniform prior)
        pau_breaks = _map_pau_to_breakpoints(pau_list, auto_bp_list, num_tokens, duration_ms)

        # Resolve trims: saved > auto-trim > None
        trim_start: int | None = None
        trim_end: int | None = None
        auto_trim_window = auto_trim_by_uid.get(uid)
        if auto_trim_window is not None:
            trim_start, trim_end = auto_trim_window

        # Apply saved labels if present (overrides auto-trim)
        if uid in saved_labels:
            saved = saved_labels[uid]
            saved_breaks = saved.get("breaks", [])
            saved_trim_start = saved.get("trim_start_ms")
            saved_trim_end = saved.get("trim_end_ms")
            if saved_trim_start is not None:
                trim_start = saved_trim_start
            if saved_trim_end is not None:
                trim_end = saved_trim_end
            saved_by_idx = {b["pau_idx"]: b for b in saved_breaks}
            for pb in pau_breaks:
                if pb.pau_idx in saved_by_idx:
                    sb = saved_by_idx[pb.pau_idx]
                    pb.use_break = sb.get("use_break", True)
                    pb.ms = sb.get("ms", pb.ms)
                    pb.noise_zone_ms = sb.get("noise_zone_ms")
                    # Preserve ms_proposed from saved label if available,
                    # otherwise keep the one from auto-mapping
                    if "ms_proposed" in sb:
                        pb.ms_proposed = sb["ms_proposed"]
                else:
                    # pau_idx not in saved breaks = explicitly not a break
                    pb.use_break = False

        # Defensive clamp: prevent garbage trims from reaching the UI
        if trim_start is not None and trim_end is not None:
            trim_start = max(0, min(trim_start, duration_ms))
            trim_end = max(0, min(trim_end, duration_ms))
            if trim_end <= trim_start:
                trim_start, trim_end = None, None

        items.append(
            UtteranceItem(
                utterance_id=uid,
                text=utt.get("text", utt.get("text_norm", "")),
                phonemes=phonemes,
                audio_relpath=utt.get("audio_relpath", ""),
                audio_abspath=utt.get("audio_abspath", ""),
                duration_sec=utt.get("duration_sec", 0.0),
                speaker_id=utt.get("speaker_id"),
                sample_rate=utt.get("sample_rate", 22050),
                pau_count=len(pau_list),
                pau_breaks=pau_breaks,
                trim_start_ms=trim_start,
                trim_end_ms=trim_end,
                status=label_statuses.get(uid, "pending"),
            )
        )

    return items, session


def _load_saved_labels(session_id: str) -> dict[str, dict]:
    """Load saved labels from a session's labels.jsonl.

    Supports schema v1 (use_break, ms_proposed, delta_ms) and legacy formats.
    For each utterance, keeps only the latest label entry (last wins).

    Returns dict of utterance_id -> {breaks, trim_start_ms, trim_end_ms}.
    """
    labels_path = _sessions_root() / session_id / "labels.jsonl"
    if not labels_path.exists():
        return {}

    saved: dict[str, dict] = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = json.loads(line)
            uid = label["utterance_id"]
            if "breaks" in label:
                saved[uid] = {
                    "breaks": label["breaks"],
                    "trim_start_ms": label.get("trim_start_ms"),
                    "trim_end_ms": label.get("trim_end_ms"),
                }
            elif "breakpoints_ms" in label:
                # Legacy format: no pau mapping info
                saved[uid] = {"breaks": [], "trim_start_ms": None, "trim_end_ms": None}

    return saved


def _load_label_statuses(session_id: str) -> dict[str, str]:
    """Load per-utterance status from labels.jsonl (last entry wins)."""
    labels_path = _sessions_root() / session_id / "labels.jsonl"
    if not labels_path.exists():
        return {}

    statuses: dict[str, str] = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = json.loads(line)
            statuses[label["utterance_id"]] = label.get("status", "labeled")

    return statuses


def save_labels(
    session_id: str,
    utterance_id: str,
    breaks: list[dict],
    status: str = "labeled",
    heuristic_version: str = "unknown",
    heuristic_params_hash: str = "unknown",
    sample_rate: int = 22050,
    trim_start_ms: int | None = None,
    trim_end_ms: int | None = None,
) -> None:
    """
    Save pau break labels for an utterance (schema v1).

    All pau tokens are persisted — both use_break=true and use_break=false.
    This ensures negative labels are captured for active learning.

    Status values:
        "labeled" — user made break decisions
        "skipped" — user reviewed but couldn't confidently label

    Args:
        breaks: list of {
            "pau_idx": int,
            "token_position": int,
            "ms_proposed": int | None,
            "ms": int | None,
            "delta_ms": int | None,
            "use_break": bool,
            "noise_zone_ms": int | None,  # (b) marker for noise zone boundary
        }
    """
    labels_path = _sessions_root() / session_id / "labels.jsonl"

    label = {
        "utterance_id": utterance_id,
        "breaks": breaks,
        "trim_start_ms": trim_start_ms,
        "trim_end_ms": trim_end_ms,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "heuristic_version": heuristic_version,
        "heuristic_params_hash": heuristic_params_hash,
        "sample_rate": sample_rate,
        "labeled_at": datetime.now(UTC).isoformat(),
        "status": status,
    }

    with open(labels_path, "a") as f:
        f.write(json.dumps(label) + "\n")


def get_session_progress(session_id: str) -> dict:
    """Get progress stats for a session."""
    session = load_session(session_id)
    statuses = _load_label_statuses(session_id)

    total = len(session.utterance_ids)
    labeled = sum(1 for uid in session.utterance_ids if statuses.get(uid) == "labeled")
    skipped = sum(1 for uid in session.utterance_ids if statuses.get(uid) == "skipped")
    reviewed = labeled + skipped

    return {
        "session_id": session_id,
        "dataset": session.dataset,
        "created_at": session.created_at,
        "batch_size": session.batch_size,
        "stratum": session.stratum,
        "total": total,
        "labeled": labeled,
        "skipped": skipped,
        "remaining": total - reviewed,
        "pct": round(100 * reviewed / total, 1) if total > 0 else 0.0,
    }


def list_datasets() -> list[dict]:
    """List datasets that have gold manifests available."""
    gold_dir = paths.gold
    if not gold_dir.exists():
        return []

    datasets = []
    for ds_dir in sorted(gold_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        manifest_dir = ds_dir / "manifests"
        if not manifest_dir.exists():
            continue
        jsonl_files = list(manifest_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        datasets.append(
            {
                "name": ds_dir.name,
                "manifest_count": len(jsonl_files),
            }
        )

    return datasets


def list_sessions(dataset: str | None = None) -> list[dict]:
    """List existing labeling sessions."""
    sessions_dir = _sessions_root()
    if not sessions_dir.exists():
        return []

    # Collect published session IDs per dataset
    published_by_ds: dict[str, set[str]] = {}

    results = []
    for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue

        with open(session_file) as f:
            meta = json.load(f)

        if dataset and meta.get("dataset") != dataset:
            continue

        ds = meta.get("dataset", "")
        if ds not in published_by_ds:
            published_by_ds[ds] = get_published_session_ids(ds)

        progress = get_session_progress(meta["session_id"])
        progress["published"] = meta["session_id"] in published_by_ds[ds]
        results.append({**meta, **progress})

    return results


def delete_session(session_id: str) -> None:
    """Delete a labeling session and its data."""
    session_dir = _sessions_root() / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    shutil.rmtree(session_dir)


def delete_all_sessions(dataset: str) -> int:
    """Delete all labeling sessions for a dataset. Returns count deleted."""
    sessions_dir = _sessions_root()
    if not sessions_dir.exists():
        return 0

    deleted = 0
    for session_dir in list(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue
        with open(session_file) as f:
            meta = json.load(f)
        if meta.get("dataset") == dataset:
            shutil.rmtree(session_dir)
            deleted += 1

    return deleted


def _published_root(dataset: str) -> Path:
    """Get the root directory for published labels."""
    return paths.runs / "labeling" / "published" / dataset


def get_published_utterance_ids(dataset: str) -> set[str]:
    """Get all utterance IDs that have been published to the labels table."""
    labels_path = _published_root(dataset) / "labels.jsonl"
    if not labels_path.exists():
        return set()

    ids = set()
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = json.loads(line)
            ids.add(label["utterance_id"])
    return ids


def get_published_session_ids(dataset: str) -> set[str]:
    """Get session IDs that have been published."""
    meta_path = _published_root(dataset) / "manifest.json"
    if not meta_path.exists():
        return set()
    with open(meta_path) as f:
        manifest = json.load(f)
    return set(manifest.get("published_sessions", []))


def publish_session(session_id: str) -> dict:
    """
    Publish a session's labels to the persistent labels table.

    Copies all labeled/skipped entries from the session's labels.jsonl
    to the published labels table. Records the session as published.

    Returns summary stats.
    """
    session = load_session(session_id)
    dataset = session.dataset
    pub_dir = _published_root(dataset)
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Check not already published
    already_published = get_published_session_ids(dataset)
    if session_id in already_published:
        raise ValueError(f"Session already published: {session_id}")

    # Read session labels (only labeled/skipped, skip pending)
    labels_path = _sessions_root() / session_id / "labels.jsonl"
    if not labels_path.exists():
        raise ValueError(f"No labels found for session: {session_id}")

    # Dedupe: keep last entry per utterance (last wins)
    labels_by_uid: dict[str, str] = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = json.loads(line)
            status = label.get("status", "labeled")
            if status in ("labeled", "skipped"):
                # Add session provenance
                label["session_id"] = session_id
                label["published_at"] = datetime.now(UTC).isoformat()
                labels_by_uid[label["utterance_id"]] = json.dumps(label)

    if not labels_by_uid:
        raise ValueError(f"No labeled utterances in session: {session_id}")

    # Append to published labels table
    pub_labels_path = pub_dir / "labels.jsonl"
    with open(pub_labels_path, "a") as f:
        for line in labels_by_uid.values():
            f.write(line + "\n")

    # Update manifest
    manifest_path = pub_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"dataset": dataset, "published_sessions": [], "total_labels": 0}

    manifest["published_sessions"].append(session_id)
    manifest["total_labels"] += len(labels_by_uid)
    manifest["last_published_at"] = datetime.now(UTC).isoformat()

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "session_id": session_id,
        "dataset": dataset,
        "labels_published": len(labels_by_uid),
        "total_published": manifest["total_labels"],
    }
