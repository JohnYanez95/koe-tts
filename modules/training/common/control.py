"""
Training control plane for dashboard integration.

Phase B1: Safe request-only controls (checkpoint, eval)
Phase B2: Dangerous controls (pause, stop) - implemented later

Control flow:
1. Dashboard writes control.json via POST /api/runs/{id}/control
2. Training polls control.json every N steps
3. Training validates nonce (idempotency) and executes
4. Training writes ack event and clears control.json
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class ControlRequest:
    """Parsed control request."""
    nonce: str
    requested_at: str
    action: str
    params: dict

    @classmethod
    def from_dict(cls, data: dict) -> "ControlRequest":
        return cls(
            nonce=data["nonce"],
            requested_at=data["requested_at"],
            action=data["action"],
            params=data.get("params", {}),
        )


class ControlPlane:
    """
    Training control plane for receiving commands from dashboard.

    Usage:
        control = ControlPlane(output_dir, events_logger)

        # In training loop, every N steps:
        request = control.poll()
        if request:
            if request.action == "checkpoint":
                # Save checkpoint
                control.ack(request, success=True, result={"path": "..."})
            elif request.action == "eval":
                # Queue eval for after step
                control.ack(request, success=True)
    """

    # SAFE actions: idempotent, bounded, non-destructive
    # - checkpoint: saves state, no side effects
    # - eval: read-only analysis, bounded runtime
    # - stop: checkpoints first, then clean exit (bounded, preserves state)
    SAFE_ACTIONS = {"checkpoint", "eval", "stop"}

    # DANGEROUS actions: require explicit opt-in (not yet implemented)
    # - pause: unbounded state freeze, complex resume semantics
    DANGEROUS_ACTIONS = {"pause"}

    def __init__(
        self,
        output_dir: Path,
        events_logger: Optional[Any] = None,
        poll_every_steps: int = 25,
        poll_every_seconds: float = 30.0,
    ):
        """
        Initialize control plane.

        Args:
            output_dir: Run output directory
            events_logger: EventLogger for ack events
            poll_every_steps: Poll every N steps (step-based cadence)
            poll_every_seconds: Poll if N seconds elapsed since last poll (time-based cadence)
        """
        self.output_dir = Path(output_dir)
        self.control_file = self.output_dir / "control.json"
        self.events = events_logger
        self.poll_every_steps = poll_every_steps
        self.poll_every_seconds = poll_every_seconds
        self._seen_nonces: set[str] = set()
        self._last_poll_time: float = time.time()

    def should_poll(self, step: int) -> bool:
        """
        Check if we should poll this step.

        Returns True if either:
        - step % poll_every_steps == 0 (step-based cadence)
        - time since last poll >= poll_every_seconds (time-based cadence)

        This dual cadence ensures control requests are never missed,
        even on short runs or slow training steps.
        """
        # Step-based cadence
        if step % self.poll_every_steps == 0:
            return True

        # Time-based cadence
        if time.time() - self._last_poll_time >= self.poll_every_seconds:
            return True

        return False

    def mark_polled(self) -> None:
        """Mark that a poll was performed (updates time tracking)."""
        self._last_poll_time = time.time()

    def poll(self) -> Optional[ControlRequest]:
        """
        Poll for control request.

        Returns:
            ControlRequest if valid request found, None otherwise
        """
        # Update poll time tracking (for time-based cadence)
        self._last_poll_time = time.time()

        if not self.control_file.exists():
            return None

        try:
            with open(self.control_file) as f:
                data = json.load(f)

            # Validate required fields
            if not all(k in data for k in ["nonce", "requested_at", "action"]):
                self._log_rejected("missing_fields", data)
                self._clear_control()
                return None

            request = ControlRequest.from_dict(data)

            # Check nonce (idempotency)
            if request.nonce in self._seen_nonces:
                # Already processed, silently ignore
                return None

            # Validate action
            if request.action not in self.SAFE_ACTIONS:
                if request.action in self.DANGEROUS_ACTIONS:
                    self._log_rejected("dangerous_action_not_enabled", data)
                else:
                    self._log_rejected("unknown_action", data)
                self._clear_control()
                return None

            return request

        except json.JSONDecodeError:
            self._log_rejected("invalid_json", {})
            self._clear_control()
            return None
        except Exception as e:
            self._log_rejected("parse_error", {"error": str(e)})
            self._clear_control()
            return None

    def ack(
        self,
        request: ControlRequest,
        success: bool,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Acknowledge a control request.

        Args:
            request: The request being acknowledged
            success: Whether the action succeeded
            result: Optional result data
            error: Optional error message if failed
        """
        # Mark nonce as seen
        self._seen_nonces.add(request.nonce)

        # Log ack event
        if self.events:
            self.events.log(
                "control_ack",
                nonce=request.nonce,
                action=request.action,
                success=success,
                result=result,
                error=error,
            )

        # Clear control file
        self._clear_control()

    def _clear_control(self) -> None:
        """Remove control file after processing."""
        try:
            if self.control_file.exists():
                self.control_file.unlink()
        except Exception:
            pass

    def _log_rejected(self, reason: str, data: dict) -> None:
        """Log a rejected control request."""
        if self.events:
            self.events.log(
                "control_rejected",
                reason=reason,
                data=data,
            )

    def drain(self, handler: Callable[["ControlRequest"], None]) -> bool:
        """
        Final drain: poll once and execute any pending request.

        Call this immediately before logging training_complete to ensure
        no control requests are missed (e.g., user clicked right before training ended).

        Args:
            handler: Callback to handle the request. Receives ControlRequest,
                     should call self.ack() when done.

        Returns:
            True if a request was found and handled, False otherwise.

        Example:
            def handle_final_request(req):
                if req.action == "checkpoint":
                    save_checkpoint(...)
                    control.ack(req, success=True, result={...})
                else:
                    control.ack(req, success=False, error="Training complete, cannot execute")

            control.drain(handle_final_request)
        """
        request = self.poll()
        if request:
            handler(request)
            return True
        return False


def generate_eval_id(tag: Optional[str] = None) -> str:
    """
    Generate a unique eval ID for tracking.

    Format: eval_YYYYMMDD_HHMMSS[_tag]

    Args:
        tag: Optional tag to append

    Returns:
        eval_id string
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"eval_{ts}_{tag}"
    return f"eval_{ts}"


def write_control_request(
    output_dir: Path,
    action: str,
    params: Optional[dict] = None,
) -> str:
    """
    Write a control request atomically.

    Args:
        output_dir: Run output directory
        action: Action to request (checkpoint, eval)
        params: Action parameters

    Returns:
        nonce: The nonce for this request (for tracking)
    """
    import uuid

    nonce = uuid.uuid4().hex[:8]
    request = {
        "nonce": nonce,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "params": params or {},
    }

    control_file = Path(output_dir) / "control.json"

    # Atomic write: write to temp file, then rename
    fd, temp_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=".control_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(request, f, indent=2)
        os.rename(temp_path, control_file)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise

    return nonce
