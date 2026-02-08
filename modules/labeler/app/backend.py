"""
FastAPI backend for the segmentation labeling app.

Provides endpoints for:
- Listing datasets with gold manifests
- Creating/managing labeling sessions
- Serving batch items with pau break data
- Saving user labels (schema v1: use_break, ms_proposed, delta_ms)
- Serving audio files and SPA frontend
"""

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from modules.data_engineering.common.paths import paths

from . import data

# ============================================================================
# Security: Path validation
# ============================================================================

_SAFE_PATH_RE = re.compile(r"^[\w\-\.]+$")


def _validate_path_param(value: str, name: str = "parameter") -> str:
    """
    Validate a URL path parameter to prevent path traversal.

    Blocks ``..``, absolute paths, control characters, and anything
    that isn't alphanumeric / underscore / hyphen / dot.

    Raises HTTPException 400 if invalid.  Returns the value unchanged.
    """
    if ".." in value or value.startswith("/") or value.startswith("\\"):
        raise HTTPException(status_code=400, detail=f"Invalid {name}")
    if any(ord(c) < 32 for c in value):
        raise HTTPException(status_code=400, detail=f"Invalid {name}")
    if not _SAFE_PATH_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {name}")
    return value

# ============================================================================
# Models
# ============================================================================


class DatasetInfo(BaseModel):
    name: str
    manifest_count: int


class SessionCreate(BaseModel):
    dataset: str
    batch_size: int = 25
    stratum: int | None = None
    heuristic_only: bool = False
    heuristic_params_hash: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    dataset: str
    created_at: str
    batch_size: int
    stratum: int | None
    total: int
    labeled: int
    skipped: int = 0
    remaining: int
    pct: float
    published: bool = False


class PauBreakResponse(BaseModel):
    pau_idx: int
    token_position: int
    ms: int | None = None
    ms_proposed: int | None = None
    use_break: bool = False
    noise_zone_ms: int | None = None


class UtteranceResponse(BaseModel):
    utterance_id: str
    text: str
    phonemes: str
    audio_url: str
    duration_sec: float
    speaker_id: str | None = None
    sample_rate: int = 22050
    pau_count: int
    pau_breaks: list[PauBreakResponse]
    trim_start_ms: int | None = None
    trim_end_ms: int | None = None
    status: str = "pending"


class BatchResponse(BaseModel):
    session_id: str
    heuristic_version: str
    heuristic_params_hash: str
    items: list[UtteranceResponse]
    total: int
    labeled: int


class PauBreakSave(BaseModel):
    pau_idx: int
    token_position: int
    ms_proposed: int | None = None
    ms: int | None = None
    delta_ms: int | None = None
    use_break: bool = False
    noise_zone_ms: int | None = None


class LabelsSave(BaseModel):
    breaks: list[PauBreakSave]
    status: str = "labeled"  # "labeled" or "skipped"
    trim_start_ms: int | None = None
    trim_end_ms: int | None = None


class StratumInfo(BaseModel):
    stratum: int
    count: int
    heuristic_hits: int
    label: str


class HeuristicRunInfo(BaseModel):
    params_hash: str
    name: str
    method: str
    processed: int
    with_breaks: int
    elapsed_s: float
    created_at: str


class DatasetDetail(BaseModel):
    name: str
    total_utterances: int
    strata: list[StratumInfo]
    sessions: list[SessionInfo]
    heuristic_runs: list[HeuristicRunInfo]


# ============================================================================
# App Factory
# ============================================================================


def create_app() -> FastAPI:
    """Create the FastAPI application for the labeling app."""

    app = FastAPI(
        title="KOE Segmentation Labeler",
        description="Segmentation breakpoint labeling UI",
        version="0.2.0",
    )

    # CORS for local dev (localhost only)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3001", "http://127.0.0.1:3001", "http://localhost:8081", "http://127.0.0.1:8081"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store dataset for filtering (set by run_server)
    app.state.dataset_filter: str | None = None

    # ========================================================================
    # API Endpoints
    # ========================================================================

    @app.get("/api/datasets", response_model=list[DatasetInfo])
    async def list_datasets():
        """List datasets with gold manifests."""
        ds_list = data.list_datasets()
        filter_ds = app.state.dataset_filter
        if filter_ds:
            ds_list = [d for d in ds_list if d["name"] == filter_ds]
        return [DatasetInfo(**d) for d in ds_list]

    @app.get("/api/datasets/{dataset}", response_model=DatasetDetail)
    async def get_dataset_detail(dataset: str):
        """Get dataset detail with strata info."""
        dataset = _validate_path_param(dataset, "dataset")
        utterances = data.load_manifest(dataset)
        if not utterances:
            raise HTTPException(status_code=404, detail=f"No manifest for dataset: {dataset}")

        # Load heuristic hits for filtering info
        auto_bps, _ = data.load_auto_breakpoints(dataset)
        heuristic_hit_ids = {uid for uid, bps in auto_bps.items() if bps}

        strata = data.stratify_utterances(utterances)
        strata_info = []
        labels = {0: "no pau", 1: "1 pau", 2: "2 pau", 3: "3+ pau"}
        for s in sorted(strata.keys()):
            hits = sum(1 for u in strata[s] if u.get("utterance_id") in heuristic_hit_ids)
            strata_info.append(StratumInfo(stratum=s, count=len(strata[s]), heuristic_hits=hits, label=labels.get(s, f"{s} pau")))

        sessions = data.list_sessions(dataset)
        session_infos = [SessionInfo(**s) for s in sessions]

        # Load heuristic run info
        try:
            from modules.labeler.heuristic import list_heuristic_runs

            runs = list_heuristic_runs(dataset)
            heuristic_run_infos = [
                HeuristicRunInfo(
                    params_hash=r["params_hash"],
                    name=r.get("name", r["method"]),
                    method=r["method"],
                    processed=r["processed"],
                    with_breaks=r.get("with_breaks", 0),
                    elapsed_s=r["elapsed_s"],
                    created_at=r["created_at"],
                )
                for r in runs
            ]
        except ImportError:
            heuristic_run_infos = []

        return DatasetDetail(
            name=dataset,
            total_utterances=len(utterances),
            strata=strata_info,
            sessions=session_infos,
            heuristic_runs=heuristic_run_infos,
        )

    @app.post("/api/sessions", response_model=SessionInfo)
    async def create_session(body: SessionCreate):
        """Create a new labeling session."""
        _validate_path_param(body.dataset, "dataset")
        try:
            session = data.create_session(
                dataset=body.dataset,
                batch_size=body.batch_size,
                stratum=body.stratum,
                heuristic_only=body.heuristic_only,
                heuristic_params_hash=body.heuristic_params_hash,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

        progress = data.get_session_progress(session.session_id)
        return SessionInfo(**progress)

    @app.get("/api/sessions/{session_id}", response_model=SessionInfo)
    async def get_session(session_id: str):
        """Get session metadata and progress."""
        session_id = _validate_path_param(session_id, "session_id")
        try:
            progress = data.get_session_progress(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from None
        return SessionInfo(**progress)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a labeling session."""
        session_id = _validate_path_param(session_id, "session_id")
        try:
            data.delete_session(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from None
        return {"status": "deleted", "session_id": session_id}

    @app.post("/api/sessions/{session_id}/publish")
    async def publish_session(session_id: str):
        """Publish session labels to the persistent labels table."""
        session_id = _validate_path_param(session_id, "session_id")
        try:
            result = data.publish_session(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return result

    @app.delete("/api/datasets/{dataset}/sessions")
    async def delete_all_sessions(dataset: str):
        """Delete all labeling sessions for a dataset."""
        dataset = _validate_path_param(dataset, "dataset")
        count = data.delete_all_sessions(dataset)
        return {"status": "deleted", "dataset": dataset, "count": count}

    @app.get("/api/sessions/{session_id}/batch", response_model=BatchResponse)
    async def get_batch(session_id: str):
        """Get the full batch of utterances for a session."""
        session_id = _validate_path_param(session_id, "session_id")
        try:
            items, session = data.get_batch(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from None

        progress = data.get_session_progress(session_id)

        return BatchResponse(
            session_id=session_id,
            heuristic_version=session.heuristic_version,
            heuristic_params_hash=session.heuristic_params_hash,
            items=[_item_to_response(item) for item in items],
            total=progress["total"],
            labeled=progress["labeled"],
        )

    @app.get("/api/sessions/{session_id}/item/{idx}", response_model=UtteranceResponse)
    async def get_item(session_id: str, idx: int):
        """Get a single utterance from the batch."""
        session_id = _validate_path_param(session_id, "session_id")
        try:
            items, _ = data.get_batch(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from None

        if idx < 0 or idx >= len(items):
            raise HTTPException(status_code=404, detail=f"Item index out of range: {idx}")

        return _item_to_response(items[idx])

    @app.post("/api/sessions/{session_id}/item/{idx}/labels")
    async def save_labels(session_id: str, idx: int, body: LabelsSave):
        """Save pau break labels for an utterance (schema v1)."""
        session_id = _validate_path_param(session_id, "session_id")
        try:
            items, session = data.get_batch(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from None

        if idx < 0 or idx >= len(items):
            raise HTTPException(status_code=404, detail=f"Item index out of range: {idx}")

        item = items[idx]
        breaks = [
            {
                "pau_idx": b.pau_idx,
                "token_position": b.token_position,
                "ms_proposed": b.ms_proposed,
                "ms": b.ms,
                "delta_ms": b.delta_ms,
                "use_break": b.use_break,
                "noise_zone_ms": b.noise_zone_ms,
            }
            for b in body.breaks
        ]
        data.save_labels(
            session_id,
            item.utterance_id,
            breaks,
            status=body.status,
            heuristic_version=session.heuristic_version,
            heuristic_params_hash=session.heuristic_params_hash,
            sample_rate=item.sample_rate,
            trim_start_ms=body.trim_start_ms,
            trim_end_ms=body.trim_end_ms,
        )

        return {"status": "saved", "utterance_id": item.utterance_id}

    @app.get("/api/health")
    async def health():
        """Health check."""
        return {"status": "ok"}

    # ========================================================================
    # Audio serving
    # ========================================================================

    # Mount only the ingest directory for audio access (not all of paths.data)
    ingest_dir = paths.data / "ingest"
    if ingest_dir.exists():
        app.mount("/audio/ingest", StaticFiles(directory=str(ingest_dir)), name="audio")

    # ========================================================================
    # SPA serving
    # ========================================================================

    frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
    assets_dir = frontend_dist / "assets"
    index_html = frontend_dist / "index.html"

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_root():
        if index_html.exists():
            return FileResponse(str(index_html))
        raise HTTPException(
            status_code=503,
            detail="Labeler UI not built. Run: cd modules/labeler/app/frontend && npm install && npm run build",
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("audio/"):
            raise HTTPException(status_code=404, detail="Not Found")
        if index_html.exists():
            return FileResponse(str(index_html))
        raise HTTPException(
            status_code=503,
            detail="Labeler UI not built. Run: cd modules/labeler/app/frontend && npm install && npm run build",
        )

    return app


def _item_to_response(item: data.UtteranceItem) -> UtteranceResponse:
    """Convert an UtteranceItem to an API response."""
    return UtteranceResponse(
        utterance_id=item.utterance_id,
        text=item.text,
        phonemes=item.phonemes,
        audio_url=_make_audio_url(item.audio_relpath),
        duration_sec=item.duration_sec,
        speaker_id=item.speaker_id,
        sample_rate=item.sample_rate,
        pau_count=item.pau_count,
        pau_breaks=[
            PauBreakResponse(
                pau_idx=pb.pau_idx,
                token_position=pb.token_position,
                ms=pb.ms,
                ms_proposed=pb.ms_proposed,
                use_break=pb.use_break,
                noise_zone_ms=pb.noise_zone_ms,
            )
            for pb in item.pau_breaks
        ],
        trim_start_ms=item.trim_start_ms,
        trim_end_ms=item.trim_end_ms,
        status=item.status,
    )


def _make_audio_url(audio_relpath: str) -> str:
    """Convert a data-relative audio path to a URL served by the /audio mount."""
    return f"/audio/{audio_relpath}"


# ============================================================================
# Main
# ============================================================================


def run_server(
    host: str = "127.0.0.1",
    port: int = 8081,
    dataset: str | None = None,
) -> None:
    """Run the labeling server."""
    import uvicorn

    app = create_app()
    if dataset:
        app.state.dataset_filter = dataset

    print("Starting KOE Segmentation Labeler")
    print(f"  Dataset: {dataset or 'all'}")
    print(f"  URL: http://{host}:{port}")
    print(f"  API: http://{host}:{port}/api/datasets")
    print()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
