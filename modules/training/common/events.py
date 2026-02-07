"""
Training event logger for dashboard timeline.

Writes append-only events to events.jsonl for:
- run_started
- resume_from
- checkpoint_saved
- alarm_state_change
- escalation_level_change
- exception

Session-based storage:
- Each training invocation creates a new session under train/sessions/<session_id>/
- Dashboard reads from the latest session to avoid confusion on resume
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def generate_session_id() -> str:
    """Generate a timestamp-based session ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_latest_session(run_dir: Path) -> Optional[Path]:
    """Find the latest session directory by name (lexicographic = chronological)."""
    sessions_dir = run_dir / "train" / "sessions"
    if not sessions_dir.exists():
        return None
    sessions = sorted(sessions_dir.iterdir(), reverse=True)
    return sessions[0] if sessions else None


def get_session_metrics_path(run_dir: Path, session_id: Optional[str] = None) -> Path:
    """Get metrics path for a session (or latest if session_id is None)."""
    if session_id:
        return run_dir / "train" / "sessions" / session_id / "metrics.jsonl"
    latest = get_latest_session(run_dir)
    if latest:
        return latest / "metrics.jsonl"
    # Fallback to legacy path
    return run_dir / "train" / "metrics.jsonl"


def get_session_events_path(run_dir: Path, session_id: Optional[str] = None) -> Path:
    """Get events path for a session (or latest if session_id is None)."""
    if session_id:
        return run_dir / "train" / "sessions" / session_id / "events.jsonl"
    latest = get_latest_session(run_dir)
    if latest:
        return latest / "events.jsonl"
    # Fallback to legacy path
    return run_dir / "events.jsonl"


class EventLogger:
    """
    Thread-safe event logger for training runs.

    Usage:
        events = EventLogger(output_dir)  # Creates new session automatically
        events.log("run_started", stage="gan", max_steps=25000)
        events.log("checkpoint_saved", path="checkpoints/step_015000.pt", step=15000)

    Session storage:
        - Creates train/sessions/<session_id>/ for each training invocation
        - Writes events.jsonl and metrics.jsonl to session directory
        - Dashboard reads from latest session to avoid confusion on resume
    """

    def __init__(self, output_dir: Path, session_id: Optional[str] = None):
        """
        Initialize event logger with session-based storage.

        Args:
            output_dir: Run output directory
            session_id: Optional session ID (auto-generated if not provided)
        """
        self.output_dir = Path(output_dir)
        self.session_id = session_id or generate_session_id()

        # Session directory under train/sessions/<session_id>/
        self.session_dir = self.output_dir / "train" / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Events and metrics in session directory
        self.events_file = self.session_dir / "events.jsonl"
        self.metrics_file = self.session_dir / "metrics.jsonl"

        self._lock = threading.Lock()

        # Also ensure legacy directories exist for backwards compat
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        """
        Log an event with payload.

        Args:
            event: Event type (run_started, checkpoint_saved, etc.)
            **payload: Event-specific data
        """
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }

        with self._lock:
            with open(self.events_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    def write_metrics(self, metrics: dict) -> None:
        """
        Write metrics to the session's metrics.jsonl file.

        Args:
            metrics: Dictionary of metrics to write (must include 'step')
        """
        with self._lock:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    def run_started(
        self,
        stage: str,
        dataset: str,
        max_steps: int,
        num_speakers: int,
        batch_size: int,
        resume_from: Optional[str] = None,
        features: Optional[dict] = None,
    ) -> None:
        """Log training run start with feature flags."""
        # Default features for capability detection
        if features is None:
            features = {"control_plane": True}

        self.log(
            "run_started",
            stage=stage,
            dataset=dataset,
            max_steps=max_steps,
            num_speakers=num_speakers,
            batch_size=batch_size,
            features=features,
        )

        if resume_from:
            self.log("resume_from", checkpoint=resume_from)

    def checkpoint_saved(
        self,
        path: str,
        step: int,
        val_loss: float,
        tag: str = "periodic",
        is_best: bool = False,
        mel_loss: Optional[float] = None,
        alarm_state: Optional[str] = None,
    ) -> None:
        """Log checkpoint save event with training context."""
        payload = {
            "path": path,
            "step": step,
            "val_loss": round(val_loss, 4),
            "tag": tag,
            "is_best": is_best,
        }
        if mel_loss is not None:
            payload["mel_loss"] = round(mel_loss, 4)
        if alarm_state is not None:
            payload["alarm_state"] = alarm_state
        self.log("checkpoint_saved", **payload)

    def alarm_state_change(
        self,
        previous: str,
        current: str,
        step: int,
        reason: Optional[str] = None,
    ) -> None:
        """Log controller alarm state change."""
        self.log(
            "alarm_state_change",
            previous=previous,
            current=current,
            step=step,
            reason=reason,
        )

    def escalation_level_change(
        self,
        previous_level: int,
        current_level: int,
        step: int,
        reason: Optional[str] = None,
    ) -> None:
        """Log controller escalation level change (L0→L1→L2→L3→Emergency)."""
        self.log(
            "escalation_level_change",
            previous_level=previous_level,
            current_level=current_level,
            step=step,
            reason=reason,
        )

    def exception(
        self,
        error_type: str,
        message: str,
        step: Optional[int] = None,
        traceback_path: Optional[str] = None,
    ) -> None:
        """Log training exception."""
        self.log(
            "exception",
            error_type=error_type,
            message=message,
            step=step,
            traceback_path=traceback_path,
        )

    def training_complete(
        self,
        step: int,
        best_val_loss: float,
        final_val_loss: float,
        total_seconds: float,
        status: str = "success",  # "success" | "user_stopped" | "thermal_stop"
        reason: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Log training completion."""
        payload = {
            "step": step,
            "best_val_loss": round(best_val_loss, 4),
            "final_val_loss": round(final_val_loss, 4),
            "total_seconds": round(total_seconds, 1),
            "status": status,
        }
        if reason:
            payload["reason"] = reason
        if checkpoint_path:
            payload["checkpoint_path"] = checkpoint_path
        self.log("training_complete", **payload)

    def eval_started(
        self,
        eval_id: str,
        run_id: str,
        step: int,
        mode: str,
        seed: int,
        tag: Optional[str] = None,
        nonce: Optional[str] = None,
    ) -> None:
        """Log eval execution start."""
        self.log(
            "eval_started",
            eval_id=eval_id,
            run_id=run_id,
            step=step,
            mode=mode,
            seed=seed,
            tag=tag,
            requested_by={"nonce": nonce, "source": "control_plane"} if nonce else None,
        )

    def eval_complete(
        self,
        eval_id: str,
        run_id: str,
        step: int,
        artifact_dir: str,
        summary: dict,
        nonce: Optional[str] = None,
        losses: Optional[dict] = None,
    ) -> None:
        """Log eval completion with results.

        Args:
            losses: Optional dict with training-comparable losses
                    {"mel_loss": float, "kl_loss": float, "dur_loss": float}
                    Used by dashboard to display eval markers on loss charts.
        """
        self.log(
            "eval_complete",
            eval_id=eval_id,
            run_id=run_id,
            step=step,
            success=True,
            artifact_dir=artifact_dir,
            summary=summary,
            losses=losses,
            requested_by={"nonce": nonce, "source": "control_plane"} if nonce else None,
        )

    def eval_failed(
        self,
        eval_id: str,
        run_id: str,
        step: int,
        error: str,
        nonce: Optional[str] = None,
    ) -> None:
        """Log eval failure."""
        self.log(
            "eval_failed",
            eval_id=eval_id,
            run_id=run_id,
            step=step,
            success=False,
            error=error,
            requested_by={"nonce": nonce, "source": "control_plane"} if nonce else None,
        )


class NullEventLogger:
    """No-op event logger for when events are disabled."""

    def log(self, event: str, **payload: Any) -> None:
        pass

    def run_started(self, **kwargs) -> None:
        pass

    def checkpoint_saved(self, **kwargs) -> None:
        pass

    def alarm_state_change(self, **kwargs) -> None:
        pass

    def exception(self, **kwargs) -> None:
        pass

    def training_complete(self, **kwargs) -> None:
        pass

    def eval_started(self, **kwargs) -> None:
        pass

    def eval_complete(self, **kwargs) -> None:
        pass

    def eval_failed(self, **kwargs) -> None:
        pass
