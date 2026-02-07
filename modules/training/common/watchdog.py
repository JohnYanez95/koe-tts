"""
Training safety watchdog.

Monitors GPU temperature and triggers emergency shutdown if overheating.
Designed to protect hardware during long/overnight training runs.

Usage:
    watchdog = ThermalWatchdog(events_logger=events)

    # In training loop (time-based, not step-based):
    if watchdog.should_check():
        action = watchdog.check()
        if action == "warn":
            # Log warning, continue training
            pass
        elif action == "stop":
            # Save checkpoint, log event, exit
            save_checkpoint(...)
            break
"""

import time
from dataclasses import dataclass
from typing import Literal, Optional, Any

from .gpu import get_gpu_stats, GpuStats


@dataclass
class ThermalWatchdogConfig:
    """Configuration for thermal watchdog."""
    # Temperature thresholds (Celsius)
    warn_temp: int = 80  # Log warning above this
    stop_temp: int = 86  # Trigger shutdown above this

    # Grace period: must exceed stop_temp for this long before shutdown
    grace_seconds: float = 60.0

    # Polling interval
    check_interval_seconds: float = 10.0

    # GPU device index
    device_idx: int = 0


ThermalAction = Literal["ok", "warn", "stop"]


class ThermalWatchdog:
    """
    GPU thermal watchdog for training safety.

    Monitors temperature and triggers emergency shutdown if overheating persists.
    Uses grace period to avoid shutdown on brief spikes.
    """

    def __init__(
        self,
        config: Optional[ThermalWatchdogConfig] = None,
        events_logger: Optional[Any] = None,
    ):
        """
        Initialize watchdog.

        Args:
            config: Watchdog configuration
            events_logger: EventLogger for logging thermal events
        """
        self.config = config or ThermalWatchdogConfig()
        self.events = events_logger

        # State tracking
        self._last_check_time: float = 0.0
        self._overheat_start_time: Optional[float] = None
        self._warned: bool = False
        self._last_temp: Optional[int] = None

    def should_check(self) -> bool:
        """Check if enough time has passed since last check."""
        return (time.time() - self._last_check_time) >= self.config.check_interval_seconds

    def check(self) -> ThermalAction:
        """
        Check GPU temperature and return recommended action.

        Returns:
            "ok": Temperature normal
            "warn": Above warn_temp, log warning but continue
            "stop": Above stop_temp for grace_seconds, emergency shutdown

        Side effects:
            - Updates internal state
            - Logs events via events_logger if provided
        """
        self._last_check_time = time.time()

        stats = get_gpu_stats(self.config.device_idx)
        if stats is None:
            # Can't read GPU stats, assume OK
            return "ok"

        temp = stats.temp_c
        self._last_temp = temp

        # Check for shutdown condition (exceeds stop_temp for grace period)
        if temp >= self.config.stop_temp:
            if self._overheat_start_time is None:
                # Start grace period
                self._overheat_start_time = time.time()
                self._log_event("thermal_warning", temp=temp, message="Overheat detected, starting grace period")

            elapsed = time.time() - self._overheat_start_time
            if elapsed >= self.config.grace_seconds:
                self._log_event(
                    "thermal_shutdown_requested",
                    temp=temp,
                    grace_elapsed=elapsed,
                    message=f"GPU at {temp}°C for {elapsed:.0f}s, requesting emergency shutdown"
                )
                return "stop"
            else:
                # Still in grace period
                return "warn"

        else:
            # Below stop_temp, reset grace period
            if self._overheat_start_time is not None:
                self._log_event("thermal_recovered", temp=temp, message="Temperature returned to safe levels")
                self._overheat_start_time = None

        # Check for warning condition
        if temp >= self.config.warn_temp:
            if not self._warned:
                self._log_event("thermal_warning", temp=temp, message=f"GPU temperature high: {temp}°C")
                self._warned = True
            return "warn"
        else:
            self._warned = False
            return "ok"

    def _log_event(self, event_type: str, **kwargs) -> None:
        """Log thermal event if logger available."""
        if self.events:
            self.events.log(event_type, **kwargs)
        else:
            # Fallback to print
            print(f"[ThermalWatchdog] {event_type}: {kwargs}")

    @property
    def last_temp(self) -> Optional[int]:
        """Last recorded temperature."""
        return self._last_temp

    @property
    def is_overheating(self) -> bool:
        """True if currently in overheat state (grace period started)."""
        return self._overheat_start_time is not None

    @property
    def overheat_duration(self) -> float:
        """Seconds spent overheating (0 if not overheating)."""
        if self._overheat_start_time is None:
            return 0.0
        return time.time() - self._overheat_start_time
