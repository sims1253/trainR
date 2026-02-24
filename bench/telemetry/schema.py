"""Canonical telemetry schema definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TelemetryVersion(str, Enum):
    """Version of the telemetry schema."""

    V1 = "v1"


class TokenUsage(BaseModel):
    """Token usage metrics."""

    prompt: int = 0
    completion: int = 0
    total: int = 0
    cache_read: int | None = None  # Null if unknown
    cache_write: int | None = None  # Null if unknown


class ToolCallMetrics(BaseModel):
    """Tool call metrics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    by_tool: dict[str, int] = Field(default_factory=dict)  # tool_name -> count
    errors: dict[str, int] = Field(default_factory=dict)  # tool_name -> error_count


class LatencyBreakdown(BaseModel):
    """Latency breakdown by phase."""

    total_s: float = 0.0
    setup_s: float | None = None
    execution_s: float | None = None
    teardown_s: float | None = None
    first_token_s: float | None = None  # Time to first token


class TelemetrySchema(BaseModel):
    """Canonical telemetry schema for all harnesses."""

    version: TelemetryVersion = TelemetryVersion.V1

    # Token usage
    tokens: TokenUsage = Field(default_factory=TokenUsage)

    # Turn/iteration counts
    turns: int = 0
    max_turns: int | None = None

    # Tool calls
    tools: ToolCallMetrics = Field(default_factory=ToolCallMetrics)

    # Latency
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

    # Cost estimation (nullable - requires pricing table)
    estimated_cost_usd: float | None = None

    # Raw adapter events for audit
    raw_events: list[dict[str, Any]] = Field(default_factory=list)

    # Provider/harness metadata
    provider: str | None = None
    model: str | None = None
    harness: str | None = None
