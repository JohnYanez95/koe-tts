"""
FastAPI backend for KOE training dashboard.

Provides endpoints for:
- Listing runs
- Run metadata and metrics
- GPU telemetry
- Eval artifacts
- SSE streaming for live metrics

Session-aware storage:
- Training now writes to train/sessions/<session_id>/{metrics,events}.jsonl
- Dashboard reads from the latest session by default
- Old runs without sessions fall back to legacy paths
"""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware

from modules.data_engineering.common.paths import paths
from modules.training.common.events import get_session_events_path, get_session_metrics_path

# ============================================================================
# Models
# ============================================================================

class RunSummary(BaseModel):
    """Summary of a training run."""
    run_id: str
    dataset: str
    stage: str
    step: int
    updated_at: str
    started_at: str  # Parsed from run_id datetime suffix
    status: str  # running, completed, stopped, emergency
    alarm_state: str  # healthy, unstable, d_dominant, g_collapse, stopped
    supports_control: bool  # Whether this run supports control plane
    has_metrics: bool  # Whether metrics are available
    checkpoint_count: int  # Number of checkpoint files


class RunMeta(BaseModel):
    """Full metadata for a run."""
    run_id: str
    config: dict
    run_md: str | None
    checkpoints: list[str]
    last_step: int
    last_timestamp: str | None
    created_at: str | None
    # Normalized fields for frontend convenience
    current_step: int
    last_updated_at: str | None
    alarm_state: str  # healthy, unstable, d_dominant, g_collapse
    stage: str  # core, gan, baseline, duration
    status: str  # running, stopped, completed, emergency, lost
    status_source: str  # "event" | "mtime_heuristic" | "legacy"
    status_reason: str | None  # "user_requested" | "thermal" | "mtime_timeout" | etc
    metrics_stale_seconds: int  # Threshold used for mtime heuristic
    has_metrics: bool  # Whether metrics.jsonl exists
    supports_control: bool  # Whether this run supports control plane


class MetricsResponse(BaseModel):
    """Paginated metrics response."""
    metrics: list[dict]
    total_lines: int
    cursor: int  # byte offset for next request
    reason: str | None = None  # "no_metrics" for legacy runs


class GpuInfo(BaseModel):
    """GPU status info."""
    index: int
    name: str
    temp_c: int
    util_pct: int
    mem_used_mb: int
    mem_total_mb: int
    power_w: int | None


class GpuResponse(BaseModel):
    """GPU telemetry response."""
    available: bool
    timestamp: str
    gpus: list[GpuInfo]


class ArtifactInfo(BaseModel):
    """Eval artifact info."""
    name: str
    path: str
    type: str  # multispeaker, eval, etc.
    updated_at: str


class ArtifactsResponse(BaseModel):
    """Artifacts listing response."""
    eval: list[ArtifactInfo]


class EventInfo(BaseModel):
    """Single event from events.jsonl."""
    ts: str
    event: str
    # Additional fields are event-specific and stored as extras


class EventsResponse(BaseModel):
    """Events listing response."""
    events: list[dict]  # Raw event dicts for flexibility
    total_lines: int
    cursor: int  # byte offset for next request


class ControlRequestBody(BaseModel):
    """Control request from dashboard."""
    action: str  # checkpoint, eval
    params: dict | None = None


class ControlResponse(BaseModel):
    """Control request response."""
    success: bool
    nonce: str
    message: str


class EvalRequestBody(BaseModel):
    """Eval request for runs (backend-triggered).

    Modes:
        - "multispeaker": Speaker separation eval (multi-speaker runs)
        - "teacher": VITS eval with training-comparable losses (mel, kl, dur)
        - "inference": VITS eval text-to-speech (no posterior, mel only)
    """
    seed: int = 42
    mode: str = "multispeaker"
    speakers: list[str] | None = None
    prompts_file: str | None = None
    tag: str | None = None
    checkpoint: str | None = None  # Specific checkpoint name (e.g. "step_025000.pt")
    force_rerun: bool = False  # Force re-run even if eval exists


class EvalResponse(BaseModel):
    """Eval execution response."""
    success: bool
    eval_id: str
    artifact_dir: str | None = None
    error: str | None = None
    cached: bool = False  # True if result was from existing eval


class CheckpointInfo(BaseModel):
    """Checkpoint with eval status."""
    name: str  # e.g. "step_025000.pt", "best.pt"
    step: int | None  # Extracted from name, None for best/final
    tag: str | None  # e.g. "emergency", "manual", None for periodic
    created_at: str  # ISO timestamp
    size_mb: float
    eval_status: str  # "none", "complete", "running"
    eval_dir: str | None  # Path to eval results if complete


class CheckpointsResponse(BaseModel):
    """List of checkpoints for a run."""
    checkpoints: list[CheckpointInfo]
    total: int


# ============================================================================
# Helpers
# ============================================================================


def parse_run_datetime(run_id: str) -> datetime | None:
    """Parse datetime from run_id suffix (format: *_YYYYMMDD_HHMMSS)."""
    match = re.search(r'_(\d{8})_(\d{6})$', run_id)
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def parse_run_md(run_path: Path) -> dict:
    """
    Parse RUN.md to extract step and val_loss from checkpoints table.

    Falls back to scanning checkpoints directory if RUN.md has no data.

    Precedence order:
    1. RUN.md checkpoints table (most semantically correct)
    2. checkpoints dir step_*.pt scan
    3. else step=0

    Returns dict with: step, val_loss, model_type, parameters, checkpoint_count
    """
    result = {
        "step": 0,
        "val_loss": None,
        "model_type": None,
        "parameters": None,
        "checkpoint_count": 0,
        "step_source": "none",  # For debugging: "run_md", "checkpoints_dir", "none"
    }

    # Count checkpoints first (always needed)
    ckpt_dir = run_path / "checkpoints"
    if ckpt_dir.exists():
        result["checkpoint_count"] = sum(
            1 for f in ckpt_dir.iterdir()
            if f.suffix == ".pt" and f.stem not in ("best",)  # Don't count best.pt as separate
        )

    run_md = run_path / "RUN.md"

    if run_md.exists():
        try:
            content = run_md.read_text()

            # Parse model type from "- **Type**: baseline_mel"
            type_match = re.search(r'\*\*Type\*\*:\s*(\S+)', content)
            if type_match:
                result["model_type"] = type_match.group(1)

            # Parse parameters from "- **Parameters**: 3,314,000"
            params_match = re.search(r'\*\*Parameters\*\*:\s*([\d,]+)', content)
            if params_match:
                result["parameters"] = int(params_match.group(1).replace(",", ""))

            # Parse checkpoints table for final step/loss
            # Format: | final.pt | 100 | inf | final |
            # Or: | step_045000.pt | 45000 | 3.2216 | periodic |
            checkpoint_pattern = re.compile(
                r'\|\s*(?:final\.pt|step_\d+\.pt|best\.pt)\s*\|\s*(\d+)\s*\|\s*([^\|]+)\s*\|'
            )

            best_step = 0
            best_loss = None
            for match in checkpoint_pattern.finditer(content):
                step = int(match.group(1))
                loss_str = match.group(2).strip()

                if step > best_step:
                    best_step = step
                    try:
                        loss = float(loss_str)
                        if loss != float('inf'):
                            best_loss = loss
                    except ValueError:
                        pass

            if best_step > 0:
                result["step"] = best_step
                result["val_loss"] = best_loss
                result["step_source"] = "run_md"

        except Exception:
            pass

    # Fallback: scan checkpoints directory for step_NNNNNN.pt or final.pt
    if result["step"] == 0 and ckpt_dir.exists():
        max_step = 0
        for ckpt_file in ckpt_dir.iterdir():
            if ckpt_file.suffix == ".pt":
                # Parse step from step_NNNNNN.pt
                step_match = re.match(r'step_(\d+)', ckpt_file.stem)
                if step_match:
                    max_step = max(max_step, int(step_match.group(1)))
                # final.pt typically has max_steps in config, use as marker
                elif ckpt_file.stem == "final":
                    max_step = max(max_step, 1)  # At least 1 to indicate completion

        if max_step > 0:
            result["step"] = max_step
            result["step_source"] = "checkpoints_dir"

    return result


def get_runs_dir() -> Path:
    """Get the runs directory from lakehouse config."""
    return paths.runs


def detect_control_support(run_path: Path) -> bool:
    """
    Detect if a run supports the control plane.

    Checks events.jsonl for run_started event with features.control_plane=true.
    Uses session-aware path (latest session if available, else legacy path).
    """
    events_path = get_session_events_path(run_path)
    if not events_path.exists():
        return False

    try:
        with open(events_path) as f:
            # Only check first ~10 lines (run_started should be near the top)
            for i, line in enumerate(f):
                if i >= 10:
                    break
                try:
                    event = json.loads(line.strip())
                    if event.get("event") == "run_started":
                        features = event.get("features", {})
                        return features.get("control_plane", False)
                except json.JSONDecodeError:
                    continue
        return False
    except Exception:
        return False


def discover_runs(runs_dir: Path) -> list[RunSummary]:
    """Discover all training runs, sorted by start time descending."""
    runs = []

    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        if run_path.name.startswith("."):
            continue

        # Check for metrics.jsonl to determine if it's a valid run
        # Use session-aware paths (latest session if available)
        metrics_path = get_session_metrics_path(run_path)
        events_path = get_session_events_path(run_path)
        config_path = run_path / "config.yaml"

        if not config_path.exists():
            continue

        try:
            # Load config
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Determine stage from run name or config
            run_id = run_path.name
            stage = "unknown"
            if "_gan_" in run_id:
                stage = "gan"
            elif "_core_" in run_id:
                stage = "core"
            elif "_baseline_" in run_id:
                stage = "baseline"
            elif "_duration_" in run_id:
                stage = "duration"

            # Get dataset
            dataset = config.get("data", {}).get("dataset", "unknown")

            # Parse started_at from run_id
            started_dt = parse_run_datetime(run_id)
            started_at = started_dt.isoformat() if started_dt else ""

            # Get last step and timestamp from metrics
            step = 0
            updated_at = ""
            status = "unknown"
            alarm_state = "unknown"

            # Check for metrics (GAN runs use train/metrics.jsonl)
            has_metrics = metrics_path.exists()

            # Always get checkpoint info from RUN.md/dir
            run_md_info = parse_run_md(run_path)
            checkpoint_count = run_md_info["checkpoint_count"]

            if has_metrics:
                mtime = datetime.fromtimestamp(
                    metrics_path.stat().st_mtime, tz=UTC
                )
                updated_at = mtime.isoformat()

                # Check if actively running (modified in last 60s)
                age_seconds = (datetime.now(UTC) - mtime).total_seconds()
                status = "running" if age_seconds < 60 else "stopped"

                # Read last line for step and alarm_state
                try:
                    with open(metrics_path, "rb") as f:
                        f.seek(0, 2)  # End
                        size = f.tell()
                        if size > 0:
                            # Read last 1KB
                            f.seek(max(0, size - 1024))
                            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                            if lines:
                                last = json.loads(lines[-1])
                                step = last.get("step", 0)
                                alarm_state = last.get("ctrl_controller_alarm", "healthy")
                                # Check for emergency stop
                                if last.get("ctrl_emergency_stop"):
                                    status = "emergency"
                except Exception:
                    pass

                # If we couldn't read alarm_state but have metrics, default to healthy
                if alarm_state == "unknown":
                    alarm_state = "healthy"

                # Check events.jsonl for terminal status (training_complete)
                if status == "stopped" and events_path.exists():
                    try:
                        with open(events_path, "rb") as f:
                            f.seek(0, 2)
                            size = f.tell()
                            if size > 0:
                                f.seek(max(0, size - 2048))
                                lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                                for line in reversed(lines):
                                    try:
                                        event = json.loads(line)
                                        if event.get("event") == "training_complete":
                                            evt_status = event.get("status", "success")
                                            if evt_status == "thermal_stop":
                                                status = "stopped"
                                            elif evt_status == "user_stopped":
                                                status = "stopped"
                                            elif evt_status == "emergency":
                                                status = "emergency"
                                            else:
                                                status = "completed"
                                            break
                                    except json.JSONDecodeError:
                                        continue
                    except Exception:
                        pass
            else:
                # Legacy runs without train/metrics.jsonl
                step = run_md_info["step"]

                # Use RUN.md mtime for updated_at, fall back to config
                run_md_path = run_path / "RUN.md"
                if run_md_path.exists():
                    mtime = datetime.fromtimestamp(
                        run_md_path.stat().st_mtime, tz=UTC
                    )
                else:
                    mtime = datetime.fromtimestamp(
                        config_path.stat().st_mtime, tz=UTC
                    )
                updated_at = mtime.isoformat()

                # Legacy runs are completed (no active training)
                status = "completed"

            # Check for control plane support (GAN stage with events.jsonl)
            supports_control = stage == "gan" and events_path.exists()

            runs.append(RunSummary(
                run_id=run_id,
                dataset=dataset,
                stage=stage,
                step=step,
                updated_at=updated_at,
                started_at=started_at,
                status=status,
                alarm_state=alarm_state or "healthy",
                supports_control=supports_control,
                has_metrics=has_metrics,
                checkpoint_count=checkpoint_count,
            ))

        except Exception:
            # Skip invalid runs
            continue

    # Sort by started_at descending (newest first)
    runs.sort(key=lambda r: r.started_at, reverse=True)
    return runs


def get_gpu_info() -> GpuResponse:
    """Get GPU telemetry via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
            except pynvml.NVMLError:
                power = None

            gpus.append(GpuInfo(
                index=i,
                name=name,
                temp_c=temp,
                util_pct=util.gpu,
                mem_used_mb=mem.used // (1024 * 1024),
                mem_total_mb=mem.total // (1024 * 1024),
                power_w=power,
            ))

        pynvml.nvmlShutdown()

        return GpuResponse(
            available=True,
            timestamp=datetime.now(UTC).isoformat(),
            gpus=gpus,
        )

    except Exception:
        return GpuResponse(
            available=False,
            timestamp=datetime.now(UTC).isoformat(),
            gpus=[],
        )


# ============================================================================
# App Factory
# ============================================================================

def create_app(runs_dir: Path | None = None) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="KOE Training Dashboard",
        description="Monitor training runs for koe-tts",
        version="0.1.0",
    )

    # CORS for local dev (localhost only - never use ["*"] in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3010",
            "http://127.0.0.1:3010",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache headers for artifact HTML files (prevents stale eval pages)
    class NoCacheHtmlMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            # Add no-cache headers for HTML files in /runs/
            if request.url.path.startswith("/runs/") and request.url.path.endswith(".html"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.add_middleware(NoCacheHtmlMiddleware)

    # Store runs_dir in app state
    app.state.runs_dir = runs_dir or get_runs_dir()

    # ========================================================================
    # Security: Path validation
    # ========================================================================

    def validate_run_id(run_id: str) -> str:
        """
        Validate run_id to prevent path traversal attacks.

        Raises HTTPException 400 if invalid.
        Returns the validated run_id if safe.
        """
        # Block path traversal attempts
        if ".." in run_id or run_id.startswith("/") or run_id.startswith("\\"):
            raise HTTPException(status_code=400, detail="Invalid run_id")
        # Block null bytes and other control characters
        if any(ord(c) < 32 for c in run_id):
            raise HTTPException(status_code=400, detail="Invalid run_id")
        # Ensure it's a simple directory name (alphanumeric, underscore, hyphen, dot)
        import re
        if not re.match(r"^[\w\-\.]+$", run_id):
            raise HTTPException(status_code=400, detail="Invalid run_id")
        return run_id

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/api/runs", response_model=list[RunSummary])
    async def list_runs():
        """List all training runs sorted by activity."""
        return discover_runs(app.state.runs_dir)

    @app.get("/api/runs/{run_id}/meta", response_model=RunMeta)
    async def get_run_meta(run_id: str):
        """Get full metadata for a run."""
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id

        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Load config
        config_path = run_path / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        # Load RUN.md
        run_md = None
        run_md_path = run_path / "RUN.md"
        if run_md_path.exists():
            with open(run_md_path) as f:
                run_md = f.read()

        # List checkpoints
        checkpoints = []
        ckpt_dir = run_path / "checkpoints"
        if ckpt_dir.exists():
            checkpoints = sorted([p.name for p in ckpt_dir.glob("*.pt")])

        # Get last step from metrics
        last_step = 0
        last_timestamp = None
        created_at = None
        alarm_state = "healthy"
        last_updated_at = None
        status = "unknown"
        status_source = "unknown"
        status_reason: str | None = None
        # Use session-aware paths (latest session if available)
        metrics_path = get_session_metrics_path(run_path)
        events_path = get_session_events_path(run_path)

        # Configurable stale threshold (could be env var later)
        METRICS_STALE_SECONDS = 60

        if metrics_path.exists():
            try:
                # Get file mtime for last_updated_at
                mtime = datetime.fromtimestamp(
                    metrics_path.stat().st_mtime, tz=UTC
                )
                last_updated_at = mtime.isoformat()

                # Check if actively running (modified recently)
                age_seconds = (datetime.now(UTC) - mtime).total_seconds()
                if age_seconds < METRICS_STALE_SECONDS:
                    status = "running"
                    status_source = "mtime_heuristic"
                else:
                    # Stale metrics - assume lost until we find terminal event
                    status = "lost"
                    status_source = "mtime_heuristic"
                    status_reason = "mtime_timeout"

                with open(metrics_path, "rb") as f:
                    # First line for created_at
                    first_line = f.readline().decode("utf-8", errors="ignore").strip()
                    if first_line:
                        first = json.loads(first_line)
                        created_at = first.get("ts")

                    # Last line for step/timestamp/alarm
                    f.seek(0, 2)
                    size = f.tell()
                    if size > 0:
                        f.seek(max(0, size - 1024))
                        lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                        if lines:
                            last = json.loads(lines[-1])
                            last_step = last.get("step", 0)
                            last_timestamp = last.get("ts")
                            # Get alarm state from controller fields
                            alarm_state = last.get("ctrl_controller_alarm", "healthy")
                            # Check for emergency stop in metrics
                            if last.get("ctrl_emergency_stop"):
                                status = "emergency"
                                status_source = "metrics"
                                status_reason = last.get("ctrl_emergency_reason", "emergency_stop")

                # Check events.jsonl for terminal status (overrides mtime heuristic)
                # But if run_started appears after training_complete, run was resumed
                if status in ("lost", "stopped") and events_path.exists():
                    try:
                        with open(events_path, "rb") as f:
                            f.seek(0, 2)
                            size = f.tell()
                            if size > 0:
                                f.seek(max(0, size - 2048))
                                lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                                for line in reversed(lines):
                                    try:
                                        event = json.loads(line)
                                        evt_type = event.get("event")
                                        # If run_started is more recent than training_complete,
                                        # the run was resumed and is active (not terminal)
                                        if evt_type == "run_started":
                                            break  # Run is active, keep mtime-based status
                                        if evt_type == "training_complete":
                                            evt_status = event.get("status", "success")
                                            status_source = "event"
                                            if evt_status == "thermal_stop":
                                                status = "stopped"
                                                status_reason = "thermal"
                                            elif evt_status == "user_stopped":
                                                status = "stopped"
                                                status_reason = "user_requested"
                                            elif evt_status in ("emergency", "emergency_stop"):
                                                status = "emergency"
                                                status_reason = event.get("reason", "emergency_stop")
                                            else:
                                                status = "completed"
                                                status_reason = None
                                            break
                                    except json.JSONDecodeError:
                                        continue
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            # Legacy runs without metrics
            status = "completed"
            status_source = "legacy"

        # Determine stage from run name or config
        stage = "unknown"
        if "_gan_" in run_id:
            stage = "gan"
        elif "_core_" in run_id:
            stage = "core"
        elif "_baseline_" in run_id:
            stage = "baseline"
        elif "_duration_" in run_id:
            stage = "duration"

        # Detect control plane support
        supports_control = detect_control_support(run_path)

        return RunMeta(
            run_id=run_id,
            config=config,
            run_md=run_md,
            checkpoints=checkpoints,
            last_step=last_step,
            last_timestamp=last_timestamp,
            created_at=created_at,
            current_step=last_step,
            last_updated_at=last_updated_at,
            alarm_state=alarm_state,
            stage=stage,
            status=status,
            status_source=status_source,
            status_reason=status_reason,
            metrics_stale_seconds=METRICS_STALE_SECONDS,
            has_metrics=metrics_path.exists(),
            supports_control=supports_control,
        )

    @app.get("/api/runs/{run_id}/metrics", response_model=MetricsResponse)
    async def get_metrics(
        run_id: str,
        after: int = Query(0, description="Byte offset to start from"),
        limit: int = Query(1000, description="Max lines to return"),
        tail: bool = Query(False, description="Return last N metrics instead of first"),
    ):
        """Get metrics from a run with pagination."""
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id
        metrics_path = get_session_metrics_path(run_path)

        # Legacy runs without metrics - return empty response with reason
        if not metrics_path.exists():
            return MetricsResponse(metrics=[], total_lines=0, cursor=0, reason="no_metrics")

        metrics = []
        total_lines = 0
        cursor = after

        try:
            if tail:
                # Read last N metrics (for recent data view)
                # Metrics lines are ~500-1000 bytes each, use 2MB buffer for safety
                with open(metrics_path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    read_size = min(size, 2 * 1024 * 1024)  # 2MB buffer
                    f.seek(max(0, size - read_size))
                    # Skip partial first line if we didn't start at beginning
                    if f.tell() > 0:
                        f.readline()
                    lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                    # Take last 'limit' lines
                    for line in lines[-limit:]:
                        try:
                            metrics.append(json.loads(line))
                            total_lines += 1
                        except json.JSONDecodeError:
                            pass
                    cursor = size
            else:
                # Forward pagination from offset
                with open(metrics_path, "rb") as f:
                    f.seek(after)
                    lines_read = 0

                    for line in f:
                        total_lines += 1
                        if lines_read < limit:
                            try:
                                metrics.append(json.loads(line.decode("utf-8", errors="ignore")))
                                lines_read += 1
                            except json.JSONDecodeError:
                                pass

                    cursor = f.tell()
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to read metrics") from None

        return MetricsResponse(
            metrics=metrics,
            total_lines=total_lines,
            cursor=cursor,
        )

    @app.get("/api/gpu", response_model=GpuResponse)
    async def get_gpu():
        """Get GPU telemetry."""
        return get_gpu_info()

    @app.get("/api/runs/{run_id}/artifacts", response_model=ArtifactsResponse)
    async def get_artifacts(run_id: str):
        """List eval artifacts for a run."""
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id
        eval_dir = run_path / "eval"

        artifacts = []

        if eval_dir.exists():
            for artifact_path in eval_dir.iterdir():
                if not artifact_path.is_dir():
                    continue

                # Determine type and path to serve
                artifact_type = "eval"
                if artifact_path.name.startswith("multispeaker"):
                    artifact_type = "multispeaker"

                # Check for index.html (HTML visualization)
                index_path = artifact_path / "index.html"
                # Check for metrics.json (raw eval results)
                metrics_path = artifact_path / "metrics.json"

                if index_path.exists():
                    serve_path = str(index_path.relative_to(app.state.runs_dir))
                elif metrics_path.exists():
                    # Raw eval without HTML - serve metrics.json
                    serve_path = str(metrics_path.relative_to(app.state.runs_dir))
                    artifact_type = "eval_metrics"
                else:
                    continue

                mtime = datetime.fromtimestamp(
                    artifact_path.stat().st_mtime, tz=UTC
                )

                artifacts.append(ArtifactInfo(
                    name=artifact_path.name,
                    path=serve_path,
                    type=artifact_type,
                    updated_at=mtime.isoformat(),
                ))

        # Sort by updated_at descending
        artifacts.sort(key=lambda a: a.updated_at, reverse=True)

        return ArtifactsResponse(eval=artifacts)

    @app.get("/api/runs/{run_id}/checkpoints", response_model=CheckpointsResponse)
    async def get_checkpoints(run_id: str):
        """List checkpoints with their eval status."""
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id
        ckpt_dir = run_path / "checkpoints"
        eval_dir = run_path / "eval"

        if not ckpt_dir.exists():
            return CheckpointsResponse(checkpoints=[], total=0)

        # Get existing eval directories to check status
        existing_evals: dict[str, str] = {}  # checkpoint_name -> eval_dir_name
        if eval_dir.exists():
            for eval_path in eval_dir.iterdir():
                if eval_path.is_dir():
                    # Parse eval dir name: eval_{checkpoint}_{mode}_n{samples}_s{seed}
                    # or multispeaker_{checkpoint}_s{seed}
                    name = eval_path.name
                    if name.startswith("eval_"):
                        # Extract checkpoint name from eval dir
                        parts = name.split("_")
                        if len(parts) >= 2:
                            # Could be "eval_best_..." or "eval_step_025000_..."
                            if parts[1] == "step" and len(parts) >= 3:
                                ckpt_name = f"step_{parts[2]}.pt"
                            elif parts[1] in ("best", "final"):
                                ckpt_name = f"{parts[1]}.pt"
                            else:
                                continue
                            existing_evals[ckpt_name] = eval_path.name
                    elif name.startswith("multispeaker_"):
                        parts = name.split("_")
                        if len(parts) >= 2:
                            if parts[1] == "step" and len(parts) >= 3:
                                ckpt_name = f"step_{parts[2]}.pt"
                            elif parts[1] in ("best", "final"):
                                ckpt_name = f"{parts[1]}.pt"
                            else:
                                continue
                            existing_evals[ckpt_name] = eval_path.name

        checkpoints = []
        for ckpt_path in ckpt_dir.iterdir():
            if not ckpt_path.name.endswith(".pt"):
                continue

            name = ckpt_path.name
            step: int | None = None
            tag: str | None = None

            # Parse step and tag from filename
            # Format: step_NNNNNN.pt or step_NNNNNN_tag.pt or best.pt or final.pt
            if name.startswith("step_"):
                parts = name[5:-3].split("_")  # Remove "step_" and ".pt"
                if parts:
                    try:
                        step = int(parts[0])
                    except ValueError:
                        pass
                    if len(parts) > 1:
                        tag = "_".join(parts[1:])

            # Get file stats
            stat = ckpt_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            size_mb = stat.st_size / (1024 * 1024)

            # Check eval status
            eval_status = "none"
            eval_dir_name = existing_evals.get(name)
            if eval_dir_name:
                eval_status = "complete"

            checkpoints.append(CheckpointInfo(
                name=name,
                step=step,
                tag=tag,
                created_at=created_at.isoformat(),
                size_mb=round(size_mb, 1),
                eval_status=eval_status,
                eval_dir=eval_dir_name,
            ))

        # Sort by step descending (best/final at top, then by step)
        def sort_key(c: CheckpointInfo) -> tuple:
            if c.name == "best.pt":
                return (0, 0)
            if c.name == "final.pt":
                return (1, 0)
            return (2, -(c.step or 0))

        checkpoints.sort(key=sort_key)

        return CheckpointsResponse(checkpoints=checkpoints, total=len(checkpoints))

    @app.get("/api/runs/{run_id}/events", response_model=EventsResponse)
    async def get_events(
        run_id: str,
        after: int = Query(0, description="Byte offset to start from"),
        limit: int = Query(100, description="Max events to return"),
        tail: bool = Query(False, description="Return last N events instead of first"),
    ):
        """Get events from a run with pagination."""
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id
        events_path = get_session_events_path(run_path)

        if not events_path.exists():
            # Return empty if no events yet (not an error)
            return EventsResponse(events=[], total_lines=0, cursor=0)

        events = []
        total_lines = 0
        cursor = after

        try:
            if tail:
                # Read last N events (for recent activity view)
                with open(events_path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    # Read last 64KB or whole file
                    read_size = min(size, 65536)
                    f.seek(max(0, size - read_size))
                    lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                    # Take last 'limit' lines
                    for line in lines[-limit:]:
                        try:
                            events.append(json.loads(line))
                            total_lines += 1
                        except json.JSONDecodeError:
                            pass
                    cursor = size
            else:
                # Forward pagination from offset
                with open(events_path, "rb") as f:
                    f.seek(after)
                    lines_read = 0

                    for line in f:
                        total_lines += 1
                        if lines_read < limit:
                            try:
                                events.append(json.loads(line.decode("utf-8", errors="ignore")))
                                lines_read += 1
                            except json.JSONDecodeError:
                                pass

                    cursor = f.tell()
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to read events") from None

        return EventsResponse(
            events=events,
            total_lines=total_lines,
            cursor=cursor,
        )

    @app.post("/api/runs/{run_id}/control", response_model=ControlResponse)
    async def send_control(run_id: str, body: ControlRequestBody):
        """
        Send a control request to a training run.

        Phase B1 (safe) actions:
        - checkpoint: Request manual checkpoint save
        - eval: Request evaluation run

        Training polls control.json and executes on next poll interval.
        """
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id

        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Validate action (Phase B1: safe actions only)
        valid_actions = {"checkpoint", "eval", "stop"}
        if body.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {body.action}. Valid actions: {valid_actions}",
            )

        # Import control writer
        from modules.training.common.control import write_control_request

        try:
            nonce = write_control_request(
                output_dir=run_path,
                action=body.action,
                params=body.params,
            )
            return ControlResponse(
                success=True,
                nonce=nonce,
                message=f"Control request queued: {body.action}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to write control command") from e

    @app.post("/api/runs/{run_id}/eval", response_model=EvalResponse)
    async def trigger_eval(run_id: str, body: EvalRequestBody):
        """
        Trigger eval for any run (including legacy runs without control plane).

        This runs eval as a backend operation, logging events to events.jsonl.
        Works for both running and completed training runs.
        """
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id

        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Check that checkpoints exist
        ckpt_dir = run_path / "checkpoints"
        eval_dir = run_path / "eval"
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pt")):
            raise HTTPException(
                status_code=400,
                detail="No checkpoints found for this run",
            )

        # Determine checkpoint to use
        def find_checkpoint(ckpt_dir: Path) -> str:
            """Find best available checkpoint: best.pt > final.pt > latest step."""
            if (ckpt_dir / "best.pt").exists():
                return "best.pt"
            if (ckpt_dir / "final.pt").exists():
                return "final.pt"
            # Fall back to latest step checkpoint
            step_ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
            if step_ckpts:
                return step_ckpts[-1].name
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

        if body.checkpoint:
            # Validate checkpoint name — must be a simple filename, no traversal
            ckpt_name = body.checkpoint
            if ".." in ckpt_name or "/" in ckpt_name or "\\" in ckpt_name:
                raise HTTPException(status_code=400, detail="Invalid checkpoint name")
            if any(ord(c) < 32 for c in ckpt_name):
                raise HTTPException(status_code=400, detail="Invalid checkpoint name")
            # Use specified checkpoint
            if not (ckpt_dir / ckpt_name).exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Checkpoint not found: {ckpt_name}",
                )
            checkpoint_name = ckpt_name
        else:
            checkpoint_name = find_checkpoint(ckpt_dir)

        # Check for existing eval (unless force_rerun)
        if not body.force_rerun and eval_dir.exists():
            # Build expected eval dir pattern
            ckpt_base = checkpoint_name.replace(".pt", "")
            # Check for matching eval dirs
            for eval_path in eval_dir.iterdir():
                if not eval_path.is_dir():
                    continue
                name = eval_path.name
                # Match patterns like eval_{ckpt}_{mode}_* or multispeaker_{ckpt}_*
                if body.mode == "multispeaker":
                    if name.startswith(f"multispeaker_{ckpt_base}_") and f"_s{body.seed}" in name:
                        # Found cached eval
                        return EvalResponse(
                            success=True,
                            eval_id=f"cached_{name}",
                            artifact_dir=str(eval_path),
                            cached=True,
                        )
                else:
                    if name.startswith(f"eval_{ckpt_base}_{body.mode}_") and f"_s{body.seed}" in name:
                        # Found cached eval
                        return EvalResponse(
                            success=True,
                            eval_id=f"cached_{name}",
                            artifact_dir=str(eval_path),
                            cached=True,
                        )

        # Generate eval_id
        from modules.training.common.control import generate_eval_id
        eval_id = generate_eval_id(body.tag or "backend")

        # Initialize or append to events.jsonl (session-aware)
        events_path = get_session_events_path(run_path)

        def log_event(event_type: str, **kwargs):
            """Append event to events.jsonl."""
            event = {
                "ts": datetime.now(UTC).isoformat(),
                "event": event_type,
                **kwargs,
            }
            with open(events_path, "a") as f:
                f.write(json.dumps(event) + "\n")

        # Log eval_started
        log_event(
            "eval_started",
            eval_id=eval_id,
            run_id=run_id,
            step=0,  # Unknown for legacy runs
            mode=body.mode,
            seed=body.seed,
            tag=body.tag,
            checkpoint=checkpoint_name,
            requested_by={"source": "backend"},
        )

        try:

            # Run eval synchronously (could be made async with background tasks)
            if body.mode in ("teacher", "inference"):
                # VITS evaluation with training-comparable losses
                from modules.training.common.events import EventLogger, get_latest_session
                from modules.training.pipelines.eval_vits import evaluate as eval_vits

                # Create EventLogger for this run's session
                session_dir = get_latest_session(run_path)
                session_id = session_dir.name if session_dir else None
                events_logger = EventLogger(run_path, session_id=session_id)

                result = eval_vits(
                    run_dir=run_path,
                    checkpoint_name=checkpoint_name,
                    mode=body.mode,
                    n_samples=20,
                    seed=body.seed,
                    write_mels=True,
                    write_audio=True,
                    write_target_audio=True,  # For A/B comparison
                    device="cpu",  # CPU to avoid competing with training for GPU
                    events=events_logger,
                )

                # Result already logged via EventLogger, just return
                return EvalResponse(
                    success=True,
                    eval_id=result.get("eval_id", eval_id),
                    artifact_dir=result["eval_dir"],
                )

            else:
                # Multispeaker evaluation (speaker separation)
                from modules.training.pipelines.synthesize import eval_multispeaker

                # Validate prompts_file if provided — prevent arbitrary file read
                prompts_file = body.prompts_file
                if prompts_file:
                    pf = Path(prompts_file).resolve()
                    allowed_roots = [app.state.runs_dir.resolve(), Path.cwd().resolve()]
                    if not any(pf.is_relative_to(root) for root in allowed_roots):
                        raise HTTPException(
                            status_code=400,
                            detail="prompts_file must be within the project directory",
                        )

                result = eval_multispeaker(
                    run_id=run_id,
                    checkpoint=None,  # Use default (best.pt or final.pt)
                    speakers=body.speakers,
                    prompts_file=prompts_file,
                    seed=body.seed,
                    device="cpu",  # CPU to avoid competing with training for GPU
                )

                # Log eval_complete
                log_event(
                    "eval_complete",
                    eval_id=eval_id,
                    run_id=run_id,
                    step=result.get("manifest", {}).get("run", {}).get("step", 0),
                    success=True,
                    artifact_dir=result["output_dir"],
                    summary={
                        "n_speakers": result["n_speakers"],
                        "n_prompts": result["n_prompts"],
                        "mean_inter_speaker_distance": result["manifest"]["separation_metrics"]["mean_inter_speaker_distance"],
                        "valid_outputs": f"{sum(s['n_valid'] for s in result['manifest']['per_speaker_summary'].values())}/{result['n_speakers'] * result['n_prompts']}",
                    },
                    requested_by={"source": "backend"},
                )

                return EvalResponse(
                    success=True,
                    eval_id=eval_id,
                    artifact_dir=result["output_dir"],
                )

        except Exception as e:
            # Log full error internally for debugging
            log_event(
                "eval_failed",
                eval_id=eval_id,
                run_id=run_id,
                step=0,
                success=False,
                error=str(e),
                requested_by={"source": "backend"},
            )

            # Return sanitized error to client (no internal paths)
            return EvalResponse(
                success=False,
                eval_id=eval_id,
                error=f"Eval failed: {type(e).__name__}",
            )

    @app.get("/api/runs/{run_id}/stream")
    async def stream_metrics(run_id: str):
        """SSE stream of new metrics as they're written."""
        run_id = validate_run_id(run_id)
        run_path = app.state.runs_dir / run_id
        metrics_path = get_session_metrics_path(run_path)

        # Legacy runs without metrics - return empty stream that signals completion
        if not metrics_path.exists():
            async def empty_generator():
                yield {"event": "complete", "data": json.dumps({"reason": "no_metrics"})}
            return EventSourceResponse(empty_generator())

        async def event_generator():
            """Tail metrics.jsonl and yield new lines."""
            with open(metrics_path, "rb") as f:
                # Start at end
                f.seek(0, 2)

                while True:
                    line = f.readline()
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8", errors="ignore"))
                            yield {
                                "event": "metrics",
                                "data": json.dumps(data),
                            }
                        except json.JSONDecodeError:
                            pass
                    else:
                        # No new data, wait
                        await asyncio.sleep(1.0)

        return EventSourceResponse(event_generator())

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}

    # ========================================================================
    # Serve bundled dashboard UI (SPA)
    # ========================================================================
    # Must come AFTER all /api/* routes to avoid swallowing them

    frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
    assets_dir = frontend_dist / "assets"
    index_html = frontend_dist / "index.html"

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Mount runs directory for serving artifact files (eval HTML, audio, etc.)
    # html=False to prevent directory listing of the entire runs tree
    if runs_dir.exists():
        app.mount("/runs", StaticFiles(directory=str(runs_dir), html=False), name="runs")

    @app.get("/")
    async def serve_root():
        """Serve the dashboard UI root."""
        if index_html.exists():
            return FileResponse(str(index_html))
        raise HTTPException(
            status_code=503,
            detail="Dashboard UI not built. Run: make dashboard-build",
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA for client-side routing (non-API paths)."""
        # Let /api/* and /runs/* 404 normally (handled by their own mounts/routes)
        if full_path.startswith("api/") or full_path.startswith("runs/"):
            raise HTTPException(status_code=404, detail="Not Found")
        if index_html.exists():
            return FileResponse(str(index_html))
        raise HTTPException(
            status_code=503,
            detail="Dashboard UI not built. Run: make dashboard-build",
        )

    return app


# ============================================================================
# Main
# ============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8080, runs_dir: Path | None = None):
    """Run the dashboard server."""
    import uvicorn

    app = create_app(runs_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
