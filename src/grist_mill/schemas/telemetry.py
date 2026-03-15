"""Telemetry schema and collector for the grist-mill framework.

Provides:
- TokenUsage: Tracks prompt/completion/total token counts (defaults to 0)
- LatencyBreakdown: Tracks per-phase durations (setup, execution, teardown, total)
- ToolCallMetrics: Tracks tool invocation counts and durations
- TelemetrySchema: Versioned, forward-compatible telemetry container
- TelemetryCollector: Context-manager-based phase tracking with accumulation

Design decisions:
- TokenUsage defaults to 0 (not None) because agents that don't report tokens
  should still produce valid telemetry, not null checks everywhere.
- Phase durations accumulate: calling track_phase("execution") twice adds the
  durations together. This supports multi-turn agents that have multiple
  execution phases within a single task run.
- Schema is versioned ("V1") and uses Pydantic v2's `extra="ignore"` for
  forward compatibility: loading telemetry from a newer version won't error.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Valid phase names for track_phase()
VALID_PHASES: frozenset[str] = frozenset({"setup", "execution", "teardown"})


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token usage metrics for a task execution.

    All fields default to 0 so that telemetry is always well-formed even
    when the agent adapter does not report token counts.
    """

    model_config = ConfigDict(
        frozen=False,
    )

    prompt: int = Field(
        default=0,
        ge=0,
        description="Number of prompt (input) tokens consumed.",
    )
    completion: int = Field(
        default=0,
        ge=0,
        description="Number of completion (output) tokens generated.",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total tokens (prompt + completion). Set explicitly or computed.",
    )


# ---------------------------------------------------------------------------
# LatencyBreakdown
# ---------------------------------------------------------------------------


class LatencyBreakdown(BaseModel):
    """Per-phase latency breakdown in seconds."""

    model_config = ConfigDict(
        frozen=False,
    )

    setup_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on environment setup (seconds).",
    )
    execution_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on agent execution (seconds).",
    )
    teardown_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on environment teardown (seconds).",
    )
    total_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock time (seconds). Sum of phases.",
    )


# ---------------------------------------------------------------------------
# ToolCallMetrics
# ---------------------------------------------------------------------------


class ToolCallMetrics(BaseModel):
    """Tool invocation metrics for a task execution."""

    model_config = ConfigDict(
        frozen=False,
    )

    total_calls: int = Field(
        default=0,
        ge=0,
        description="Total number of tool calls made.",
    )
    successful_calls: int = Field(
        default=0,
        ge=0,
        description="Number of successful tool calls.",
    )
    failed_calls: int = Field(
        default=0,
        ge=0,
        description="Number of failed tool calls.",
    )
    by_tool: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-tool breakdown of call counts and statuses.",
    )
    total_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total duration of all tool calls in milliseconds.",
    )


# ---------------------------------------------------------------------------
# TelemetrySchema
# ---------------------------------------------------------------------------


class TelemetrySchema(BaseModel):
    """Versioned telemetry container for a single task execution.

    Uses ``extra="ignore"`` so that unknown fields from newer schema versions
    are silently discarded, ensuring forward compatibility.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=False,
    )

    version: str = Field(
        default="V1",
        description="Schema version for forward compatibility.",
    )
    tokens: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token usage metrics.",
    )
    latency: LatencyBreakdown = Field(
        default_factory=LatencyBreakdown,
        description="Per-phase latency breakdown.",
    )
    tool_calls: ToolCallMetrics = Field(
        default_factory=ToolCallMetrics,
        description="Tool invocation metrics.",
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        description="Estimated cost in USD. None when no pricing data is available.",
    )
    raw_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw event audit trail from adapter-specific events.",
    )


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TelemetryCollector:
    """Collector that builds a ``TelemetrySchema`` over the course of a run.

    Usage::

        collector = TelemetryCollector()
        with collector.track_phase("setup"):
            prepare_environment(...)
        with collector.track_phase("execution"):
            agent.run(task, config)
        with collector.track_phase("teardown"):
            cleanup(...)
        collector.record_tokens(prompt=100, completion=50)
        telemetry = collector.build()
    """

    def __init__(self) -> None:
        self._telemetry = TelemetrySchema()

    @property
    def telemetry(self) -> TelemetrySchema:
        """Access the live telemetry object being built."""
        return self._telemetry

    def record_tokens(self, prompt: int, completion: int) -> None:
        """Accumulate token usage from an adapter report.

        Args:
            prompt: Number of prompt tokens in this chunk.
            completion: Number of completion tokens in this chunk.
        """
        self._telemetry.tokens.prompt += prompt
        self._telemetry.tokens.completion += completion
        self._telemetry.tokens.total += prompt + completion

    def record_tool_call(
        self,
        tool_name: str,
        *,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a single tool invocation.

        Args:
            tool_name: Name of the tool that was invoked.
            success: Whether the invocation succeeded.
            duration_ms: Duration of the invocation in milliseconds.
        """
        self._telemetry.tool_calls.total_calls += 1
        self._telemetry.tool_calls.total_duration_ms += duration_ms

        if success:
            self._telemetry.tool_calls.successful_calls += 1
        else:
            self._telemetry.tool_calls.failed_calls += 1

        # Per-tool breakdown
        if tool_name not in self._telemetry.tool_calls.by_tool:
            self._telemetry.tool_calls.by_tool[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
            }
        entry = self._telemetry.tool_calls.by_tool[tool_name]
        entry["calls"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1

    def record_raw_event(self, event: dict[str, Any]) -> None:
        """Append a raw event to the audit trail.

        Args:
            event: Arbitrary dict representing an adapter-specific event.
        """
        self._telemetry.raw_events.append(event)

    def set_estimated_cost(self, cost_usd: float) -> None:
        """Set the estimated cost for this run.

        Args:
            cost_usd: Estimated cost in USD.
        """
        self._telemetry.estimated_cost_usd = cost_usd

    @contextmanager
    def track_phase(self, phase_name: str) -> Generator[None, None, None]:
        """Context manager that records the duration of a phase.

        Phase durations **accumulate**: calling ``track_phase("execution")``
        multiple times adds the durations together.

        Args:
            phase_name: One of ``"setup"``, ``"execution"``, ``"teardown"``.

        Raises:
            ValueError: If *phase_name* is not a recognised phase.

        Yields:
            None
        """
        if phase_name not in VALID_PHASES:
            msg = f"Unknown phase: {phase_name!r}. Must be one of {sorted(VALID_PHASES)}."
            raise ValueError(msg)

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start

            if phase_name == "setup":
                self._telemetry.latency.setup_s += elapsed
            elif phase_name == "execution":
                self._telemetry.latency.execution_s += elapsed
            elif phase_name == "teardown":
                self._telemetry.latency.teardown_s += elapsed

            # Recompute total as the sum of all phases
            self._telemetry.latency.total_s = (
                self._telemetry.latency.setup_s
                + self._telemetry.latency.execution_s
                + self._telemetry.latency.teardown_s
            )

            logger.debug(
                "Phase %s completed in %.4fs (total now %.4fs)",
                phase_name,
                elapsed,
                self._telemetry.latency.total_s,
            )

    def build(self) -> TelemetrySchema:
        """Return a deep copy of the current telemetry.

        The returned object is independent of subsequent collector mutations.

        Returns:
            A ``TelemetrySchema`` snapshot.
        """
        return self._telemetry.model_copy(deep=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LatencyBreakdown",
    "TelemetryCollector",
    "TelemetrySchema",
    "TokenUsage",
    "ToolCallMetrics",
]
