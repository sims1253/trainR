"""Telemetry collection utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from .schema import (
    LatencyBreakdown,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)


class TelemetryCollector:
    """Collects telemetry during execution."""

    def __init__(self) -> None:
        """Initialize the collector."""
        self._start_time: float | None = None
        self._tokens = TokenUsage()
        self._tools = ToolCallMetrics()
        self._turns = 0
        self._raw_events: list[dict] = []
        self._latency = LatencyBreakdown()

    def start(self) -> None:
        """Start collection."""
        self._start_time = time.time()

    def record_tokens(
        self,
        prompt: int,
        completion: int,
        cache_read: int | None = None,
        cache_write: int | None = None,
    ) -> None:
        """Record token usage.

        Args:
            prompt: Number of prompt tokens
            completion: Number of completion tokens
            cache_read: Tokens read from cache (optional)
            cache_write: Tokens written to cache (optional)
        """
        self._tokens.prompt += prompt
        self._tokens.completion += completion
        self._tokens.total += prompt + completion
        if cache_read is not None:
            self._tokens.cache_read = (self._tokens.cache_read or 0) + cache_read
        if cache_write is not None:
            self._tokens.cache_write = (self._tokens.cache_write or 0) + cache_write

    def record_tool_call(self, tool: str, success: bool, duration_ms: float = 0.0) -> None:
        """Record a tool call.

        Args:
            tool: Name of the tool called
            success: Whether the call succeeded
            duration_ms: Duration of the call in milliseconds
        """
        self._tools.total_calls += 1
        if success:
            self._tools.successful_calls += 1
        else:
            self._tools.failed_calls += 1
            self._tools.errors[tool] = self._tools.errors.get(tool, 0) + 1
        self._tools.by_tool[tool] = self._tools.by_tool.get(tool, 0) + 1
        self._tools.total_duration_ms += duration_ms

    def record_turn(self) -> None:
        """Record a turn/iteration."""
        self._turns += 1

    def record_raw_event(self, event: dict) -> None:
        """Record a raw adapter event for audit.

        Args:
            event: Raw event dictionary from the adapter
        """
        self._raw_events.append(event)

    def record_first_token_time(self, first_token_s: float) -> None:
        """Record time to first token.

        Args:
            first_token_s: Time in seconds until first token
        """
        self._latency.first_token_s = first_token_s

    def record_setup_time(self, setup_s: float) -> None:
        """Record setup phase time.

        Args:
            setup_s: Setup time in seconds
        """
        self._latency.setup_s = setup_s

    def record_execution_time(self, execution_s: float) -> None:
        """Record execution phase time.

        Args:
            execution_s: Execution time in seconds
        """
        self._latency.execution_s = execution_s

    def record_teardown_time(self, teardown_s: float) -> None:
        """Record teardown phase time.

        Args:
            teardown_s: Teardown time in seconds
        """
        self._latency.teardown_s = teardown_s

    @contextmanager
    def track_phase(self, phase: str) -> Generator[None, None, None]:
        """Context manager to track a phase duration.

        Args:
            phase: Phase name ('setup', 'execution', 'teardown')

        Yields:
            None
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if phase == "setup":
                self._latency.setup_s = (self._latency.setup_s or 0) + duration
            elif phase == "execution":
                self._latency.execution_s = (self._latency.execution_s or 0) + duration
            elif phase == "teardown":
                self._latency.teardown_s = (self._latency.teardown_s or 0) + duration

    def collect(
        self,
        provider: str | None = None,
        model: str | None = None,
        harness: str | None = None,
    ) -> TelemetrySchema:
        """Collect final telemetry.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model identifier
            harness: Harness name

        Returns:
            TelemetrySchema with collected metrics
        """
        total_time = time.time() - self._start_time if self._start_time else 0
        self._latency.total_s = total_time

        return TelemetrySchema(
            tokens=self._tokens,
            turns=self._turns,
            tools=self._tools,
            latency=self._latency,
            raw_events=self._raw_events,
            provider=provider,
            model=model,
            harness=harness,
        )


def collect_telemetry(
    provider: str | None = None,
    model: str | None = None,
    harness: str | None = None,
) -> TelemetryCollector:
    """Create and start a new telemetry collector.

    Args:
        provider: Provider name
        model: Model identifier
        harness: Harness name

    Returns:
        Started TelemetryCollector instance
    """
    collector = TelemetryCollector()
    collector.start()
    return collector
