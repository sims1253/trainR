"""Tool telemetry for tracking per-tool impact on outcomes.

This module provides telemetry collection for tool usage during evaluations:
- ToolCallEvent: tracks individual tool invocations
- ToolErrorEvent: tracks tool-related errors
- TelemetryCollector: aggregates telemetry per run

Event Taxonomy (low-cardinality):
- experiment_id: unique identifier for the experiment run
- profile_fingerprint: hash of the skill/support profile
- mode: support mode (none, system_only, agents_only, system_plus_agents)
- tool_profile_version: version of the tool configuration
- outcome: success/failure/timeout
- latency: total time in milliseconds
- cost: token-based cost (optional)

Usage:
    from bench.eval.telemetry import TelemetryCollector

    collector = TelemetryCollector()
    collector.record_call("read_file", duration_ms=150, success=True)
    collector.record_error("write_file", "PERMISSION_DENIED", "Cannot write to /etc")

    # Get aggregated metrics
    metrics = collector.get_metrics()
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ToolErrorType(str, Enum):
    """Normalized error types for tool failures (low-cardinality)."""

    TIMEOUT = "TIMEOUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK_ERROR = "NETWORK_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN = "UNKNOWN"


class OutcomeType(str, Enum):
    """Outcome types for telemetry events."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class ToolCallEvent:
    """Event representing a single tool invocation.

    Attributes:
        tool_name: Name of the tool invoked (e.g., "read_file", "run_test")
        timestamp: ISO timestamp when the call started
        duration_ms: Duration of the call in milliseconds
        success: Whether the call succeeded
        error_type: Error type if failed, None otherwise
    """

    tool_name: str
    timestamp: str
    duration_ms: float
    success: bool
    error_type: ToolErrorType | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_type": self.error_type.value if self.error_type else None,
        }


@dataclass
class ToolErrorEvent:
    """Event representing a tool-related error.

    Attributes:
        tool_name: Name of the tool that encountered the error
        error_type: Normalized error type (low-cardinality)
        error_message: Human-readable error message (for debugging)
        timestamp: ISO timestamp when the error occurred
    """

    tool_name: str
    error_type: ToolErrorType
    error_message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolMetrics:
    """Aggregated metrics for a single tool.

    Attributes:
        call_count: Total number of calls
        error_count: Total number of errors
        total_time_ms: Total time spent in this tool (milliseconds)
        success_count: Number of successful calls
        avg_time_ms: Average time per call
    """

    call_count: int = 0
    error_count: int = 0
    total_time_ms: float = 0.0
    success_count: int = 0

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / self.call_count

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "success_count": self.success_count,
            "avg_time_ms": round(self.avg_time_ms, 2),
            "success_rate": round(self.success_rate, 4),
        }


class TelemetryCollector:
    """Collects and aggregates telemetry events per evaluation run.

    This class provides a context manager for timing tool calls and
    aggregates results into per-tool metrics.

    Example:
        collector = TelemetryCollector()

        # Using context manager for timing
        with collector.time_call("read_file") as ctx:
            content = read_file(path)
            ctx.success()

        # Or manual recording
        collector.record_call("write_file", duration_ms=50, success=True)

        # Get final metrics
        metrics = collector.get_metrics()
    """

    def __init__(
        self,
        experiment_id: str = "",
        profile_fingerprint: str = "",
        mode: str = "",
        tool_profile_version: str = "v1",
    ):
        """Initialize the telemetry collector.

        Args:
            experiment_id: Unique identifier for this experiment
            profile_fingerprint: Hash of the skill/support profile
            mode: Support mode being used
            tool_profile_version: Version of the tool configuration
        """
        self.experiment_id = experiment_id
        self.profile_fingerprint = profile_fingerprint
        self.mode = mode
        self.tool_profile_version = tool_profile_version

        self._events: list[ToolCallEvent | ToolErrorEvent] = []
        self._tool_metrics: dict[str, ToolMetrics] = {}
        self._start_time: float | None = None

    @property
    def outcome(self) -> OutcomeType:
        """Determine overall outcome from collected events."""
        if not self._events:
            return OutcomeType.SUCCESS

        # Check for any successful calls
        call_events = [e for e in self._events if isinstance(e, ToolCallEvent)]
        if not call_events:
            return OutcomeType.SUCCESS

        # If all calls succeeded, outcome is SUCCESS
        if all(e.success for e in call_events):
            return OutcomeType.SUCCESS

        # If any timeout, outcome is TIMEOUT
        if any(
            e.error_type == ToolErrorType.TIMEOUT
            for e in call_events
            if isinstance(e, ToolCallEvent) and e.error_type
        ):
            return OutcomeType.TIMEOUT

        # Otherwise, outcome is FAILURE
        return OutcomeType.FAILURE

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all tool calls in milliseconds."""
        return sum(m.total_time_ms for m in self._tool_metrics.values())

    def record_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error_type: ToolErrorType | None = None,
        timestamp: str | None = None,
    ) -> ToolCallEvent:
        """Record a tool call event.

        Args:
            tool_name: Name of the tool
            duration_ms: Duration in milliseconds
            success: Whether the call succeeded
            error_type: Error type if failed
            timestamp: Optional timestamp (defaults to now)

        Returns:
            The created ToolCallEvent
        """
        event = ToolCallEvent(
            tool_name=tool_name,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
        )
        self._events.append(event)
        self._update_metrics(event)
        return event

    def record_error(
        self,
        tool_name: str,
        error_type: ToolErrorType,
        error_message: str,
    ) -> ToolErrorEvent:
        """Record a tool error event.

        Args:
            tool_name: Name of the tool that encountered the error
            error_type: Normalized error type
            error_message: Human-readable error message

        Returns:
            The created ToolErrorEvent
        """
        event = ToolErrorEvent(
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
        )
        self._events.append(event)

        # Update error count for this tool
        if tool_name not in self._tool_metrics:
            self._tool_metrics[tool_name] = ToolMetrics()
        self._tool_metrics[tool_name].error_count += 1

        return event

    def time_call(self, tool_name: str) -> "ToolCallContext":
        """Create a context manager for timing a tool call.

        Args:
            tool_name: Name of the tool being called

        Returns:
            A context manager that records timing on exit

        Example:
            with collector.time_call("read_file") as ctx:
                result = read_file(path)
                ctx.success()
        """
        return ToolCallContext(self, tool_name)

    def _update_metrics(self, event: ToolCallEvent) -> None:
        """Update aggregated metrics for a tool call event."""
        tool_name = event.tool_name
        if tool_name not in self._tool_metrics:
            self._tool_metrics[tool_name] = ToolMetrics()

        metrics = self._tool_metrics[tool_name]
        metrics.call_count += 1
        metrics.total_time_ms += event.duration_ms

        if event.success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics for all tools.

        Returns:
            Dict with per-tool metrics and summary statistics
        """
        return {
            "experiment_id": self.experiment_id,
            "profile_fingerprint": self.profile_fingerprint,
            "mode": self.mode,
            "tool_profile_version": self.tool_profile_version,
            "outcome": self.outcome.value,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tool_call_counts": {
                name: metrics.call_count for name, metrics in self._tool_metrics.items()
            },
            "tool_errors": {
                name: metrics.error_count
                for name, metrics in self._tool_metrics.items()
                if metrics.error_count > 0
            },
            "tool_total_time_ms": {
                name: round(metrics.total_time_ms, 2)
                for name, metrics in self._tool_metrics.items()
            },
            "tools": {name: metrics.to_dict() for name, metrics in self._tool_metrics.items()},
        }

    def get_result_fields(self) -> dict[str, Any]:
        """Get fields suitable for inclusion in ResultV1.

        This returns a flat dict with the three required telemetry fields
        for the ResultV1 schema:
        - tool_errors: dict of tool_name -> error count
        - tool_call_counts: dict of tool_name -> call count
        - tool_total_time_ms: dict of tool_name -> total time

        Returns:
            Dict with telemetry fields for ResultV1
        """
        return {
            "tool_errors": {
                name: metrics.error_count
                for name, metrics in self._tool_metrics.items()
                if metrics.error_count > 0
            },
            "tool_call_counts": {
                name: metrics.call_count for name, metrics in self._tool_metrics.items()
            },
            "tool_total_time_ms": {
                name: round(metrics.total_time_ms, 2)
                for name, metrics in self._tool_metrics.items()
            },
        }

    def get_events(self) -> list[dict[str, Any]]:
        """Get all recorded events as dictionaries.

        Returns:
            List of event dictionaries
        """
        return [e.to_dict() for e in self._events]

    def reset(self) -> None:
        """Reset the collector for reuse."""
        self._events.clear()
        self._tool_metrics.clear()
        self._start_time = None


class ToolCallContext:
    """Context manager for timing a tool call.

    Automatically records timing when exiting the context.
    Call success() or failure() to set the result status.
    """

    def __init__(self, collector: TelemetryCollector, tool_name: str):
        self.collector = collector
        self.tool_name = tool_name
        self._start_time: float | None = None
        self._success: bool | None = None
        self._error_type: ToolErrorType | None = None

    def __enter__(self) -> "ToolCallContext":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record the call on exit."""
        end_time = time.perf_counter()
        duration_ms = (end_time - self._start_time) * 1000 if self._start_time else 0

        # If exception occurred, mark as failure
        if exc_type is not None:
            self._success = False
            self._error_type = ToolErrorType.INTERNAL_ERROR

        # Default to success if not explicitly set
        if self._success is None:
            self._success = True

        self.collector.record_call(
            tool_name=self.tool_name,
            duration_ms=duration_ms,
            success=self._success,
            error_type=self._error_type,
        )

    def success(self) -> None:
        """Mark the call as successful."""
        self._success = True

    def failure(self, error_type: ToolErrorType = ToolErrorType.UNKNOWN) -> None:
        """Mark the call as failed.

        Args:
            error_type: The type of error that occurred
        """
        self._success = False
        self._error_type = error_type


def classify_error_type(error_message: str) -> ToolErrorType:
    """Classify an error message into a normalized error type.

    This function maps free-form error messages to low-cardinality
    error types for consistent telemetry.

    Args:
        error_message: The error message to classify

    Returns:
        A normalized ToolErrorType
    """
    error_lower = error_message.lower()

    if "timeout" in error_lower or "timed out" in error_lower:
        return ToolErrorType.TIMEOUT
    elif (
        "permission" in error_lower
        or "access denied" in error_lower
        or "unauthorized" in error_lower
    ):
        return ToolErrorType.PERMISSION_DENIED
    elif "not found" in error_lower or "does not exist" in error_lower or "no such" in error_lower:
        return ToolErrorType.NOT_FOUND
    elif "invalid" in error_lower or "malformed" in error_lower or "bad request" in error_lower:
        return ToolErrorType.INVALID_INPUT
    elif "rate limit" in error_lower or "too many" in error_lower or "throttl" in error_lower:
        return ToolErrorType.RATE_LIMIT
    elif "network" in error_lower or "connection" in error_lower or "dns" in error_lower:
        return ToolErrorType.NETWORK_ERROR
    elif "memory" in error_lower or "resource" in error_lower or "quota" in error_lower:
        return ToolErrorType.RESOURCE_EXHAUSTED
    else:
        return ToolErrorType.UNKNOWN
