"""Tests for telemetry schema and collector.

Covers:
- VAL-TELEM-01: Telemetry captures token usage per run (defaults to 0)
- VAL-TELEM-04: Telemetry schema is versioned and forward-compatible
- VAL-TELEM-05: Telemetry collector supports context-manager phase tracking
"""

from __future__ import annotations

import json
import time

import pytest
from pydantic import ValidationError

from grist_mill.schemas.telemetry import (
    LatencyBreakdown,
    TelemetryCollector,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)

# ===========================================================================
# TokenUsage
# ===========================================================================


class TestTokenUsage:
    """Tests for the TokenUsage model."""

    def test_defaults_to_zero(self) -> None:
        """VAL-TELEM-01: TokenUsage defaults to 0 when adapter doesn't report."""
        tokens = TokenUsage()
        assert tokens.prompt == 0
        assert tokens.completion == 0
        assert tokens.total == 0

    def test_explicit_values(self) -> None:
        """TokenUsage accepts explicit values."""
        tokens = TokenUsage(prompt=100, completion=50, total=150)
        assert tokens.prompt == 100
        assert tokens.completion == 50
        assert tokens.total == 150

    def test_serialization(self) -> None:
        """TokenUsage serializes to JSON correctly."""
        tokens = TokenUsage(prompt=10, completion=20, total=30)
        data = tokens.model_dump()
        assert data == {"prompt": 10, "completion": 20, "total": 30}

    def test_total_is_computed_when_zero(self) -> None:
        """When total=0 and prompt/completion are set, total should stay 0
        (explicit setting takes precedence)."""
        tokens = TokenUsage(prompt=10, completion=20, total=0)
        assert tokens.total == 0

    def test_negative_values_rejected(self) -> None:
        """TokenUsage fields must be non-negative."""
        with pytest.raises(ValidationError, match="prompt"):
            TokenUsage(prompt=-1)
        with pytest.raises(ValidationError, match="completion"):
            TokenUsage(completion=-1)
        with pytest.raises(ValidationError, match="total"):
            TokenUsage(total=-1)


# ===========================================================================
# LatencyBreakdown
# ===========================================================================


class TestLatencyBreakdown:
    """Tests for the LatencyBreakdown model."""

    def test_defaults_to_zero(self) -> None:
        """LatencyBreakdown defaults to 0 for all phases."""
        latency = LatencyBreakdown()
        assert latency.setup_s == 0.0
        assert latency.execution_s == 0.0
        assert latency.teardown_s == 0.0
        assert latency.total_s == 0.0

    def test_explicit_values(self) -> None:
        """LatencyBreakdown accepts explicit values."""
        latency = LatencyBreakdown(setup_s=0.1, execution_s=2.5, teardown_s=0.05, total_s=2.65)
        assert latency.setup_s == 0.1
        assert latency.execution_s == 2.5
        assert latency.teardown_s == 0.05
        assert latency.total_s == 2.65

    def test_serialization(self) -> None:
        """LatencyBreakdown serializes to JSON correctly."""
        latency = LatencyBreakdown(setup_s=1.0, execution_s=2.0, teardown_s=0.5, total_s=3.5)
        data = latency.model_dump()
        assert data["setup_s"] == 1.0
        assert data["execution_s"] == 2.0
        assert data["teardown_s"] == 0.5
        assert data["total_s"] == 3.5

    def test_negative_values_rejected(self) -> None:
        """LatencyBreakdown fields must be non-negative."""
        with pytest.raises(ValidationError):
            LatencyBreakdown(setup_s=-1.0)


# ===========================================================================
# ToolCallMetrics
# ===========================================================================


class TestToolCallMetrics:
    """Tests for the ToolCallMetrics model."""

    def test_defaults(self) -> None:
        """ToolCallMetrics has sensible defaults."""
        metrics = ToolCallMetrics()
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.by_tool == {}
        assert metrics.total_duration_ms == 0.0

    def test_explicit_values(self) -> None:
        """ToolCallMetrics accepts explicit values."""
        metrics = ToolCallMetrics(
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            by_tool={"bash": {"calls": 5, "successes": 4, "failures": 1}},
            total_duration_ms=1500.0,
        )
        assert metrics.total_calls == 10
        assert metrics.successful_calls == 8
        assert metrics.failed_calls == 2
        assert metrics.by_tool["bash"]["calls"] == 5
        assert metrics.total_duration_ms == 1500.0

    def test_serialization(self) -> None:
        """ToolCallMetrics serializes to JSON correctly."""
        metrics = ToolCallMetrics(total_calls=3, by_tool={"grep": {"calls": 3}})
        data = metrics.model_dump()
        assert data["total_calls"] == 3
        assert data["by_tool"]["grep"]["calls"] == 3


# ===========================================================================
# TelemetrySchema
# ===========================================================================


class TestTelemetrySchema:
    """Tests for the TelemetrySchema model."""

    def test_version_field_present(self) -> None:
        """VAL-TELEM-04: Telemetry schema includes a version field."""
        telemetry = TelemetrySchema()
        assert telemetry.version == "V1"

    def test_serializes_to_json_with_version(self) -> None:
        """TelemetrySchema serializes to JSON with version field."""
        telemetry = TelemetrySchema()
        data = json.loads(telemetry.model_dump_json())
        assert data["version"] == "V1"

    def test_token_usage_defaults_to_zero(self) -> None:
        """VAL-TELEM-01: TokenUsage defaults to 0 (not null)."""
        telemetry = TelemetrySchema()
        assert telemetry.tokens.prompt == 0
        assert telemetry.tokens.completion == 0
        assert telemetry.tokens.total == 0

    def test_estimated_cost_default_is_none(self) -> None:
        """estimated_cost_usd is None when no pricing data provided."""
        telemetry = TelemetrySchema()
        assert telemetry.estimated_cost_usd is None

    def test_estimated_cost_can_be_set(self) -> None:
        """estimated_cost_usd can be set explicitly."""
        telemetry = TelemetrySchema(estimated_cost_usd=0.005)
        assert telemetry.estimated_cost_usd == 0.005

    def test_raw_events_default_is_empty_list(self) -> None:
        """raw_events defaults to an empty list."""
        telemetry = TelemetrySchema()
        assert telemetry.raw_events == []

    def test_raw_events_can_be_set(self) -> None:
        """raw_events can store adapter-specific events."""
        events = [{"type": "api_call", "latency_ms": 250}]
        telemetry = TelemetrySchema(raw_events=events)
        assert len(telemetry.raw_events) == 1
        assert telemetry.raw_events[0]["type"] == "api_call"

    def test_forward_compatibility_unknown_fields_ignored(self) -> None:
        """VAL-TELEM-04: Loading V1 telemetry with extra fields does not error.

        This is achieved by Pydantic v2's default behavior of ignoring
        extra fields when `extra='ignore'` is set.
        """
        base = TelemetrySchema()
        data = json.loads(base.model_dump_json())
        # Add a future-version field
        data["future_field"] = "some_value"
        data["another_future_field"] = {"nested": True}
        # Should not raise
        loaded = TelemetrySchema.model_validate(data)
        assert loaded.version == "V1"
        assert loaded.estimated_cost_usd is None

    def test_forward_compat_with_extra_numeric_fields(self) -> None:
        """Forward compatibility works with numeric extra fields too."""
        base = TelemetrySchema()
        data = json.loads(base.model_dump_json())
        data["cache_hit_rate"] = 0.85
        data["prompt_caching_tokens"] = 500
        loaded = TelemetrySchema.model_validate(data)
        assert loaded.version == "V1"

    def test_json_round_trip(self) -> None:
        """TelemetrySchema round-trips through JSON without data loss."""
        original = TelemetrySchema(
            tokens=TokenUsage(prompt=100, completion=50, total=150),
            latency=LatencyBreakdown(setup_s=0.1, execution_s=2.0, teardown_s=0.05, total_s=2.15),
            tool_calls=ToolCallMetrics(
                total_calls=5,
                successful_calls=4,
                failed_calls=1,
                by_tool={"bash": {"calls": 5, "successes": 4, "failures": 1}},
                total_duration_ms=1000.0,
            ),
            estimated_cost_usd=0.01,
            raw_events=[{"type": "tool_call", "name": "bash"}],
        )
        json_str = original.model_dump_json()
        loaded = TelemetrySchema.model_validate_json(json_str)
        assert loaded == original

    def test_latency_default(self) -> None:
        """TelemetrySchema has a default LatencyBreakdown."""
        telemetry = TelemetrySchema()
        assert telemetry.latency.setup_s == 0.0
        assert telemetry.latency.execution_s == 0.0

    def test_tool_calls_default(self) -> None:
        """TelemetrySchema has a default ToolCallMetrics."""
        telemetry = TelemetrySchema()
        assert telemetry.tool_calls.total_calls == 0


# ===========================================================================
# TelemetryCollector
# ===========================================================================


class TestTelemetryCollector:
    """Tests for the TelemetryCollector."""

    def test_initial_state(self) -> None:
        """Collector starts with a fresh TelemetrySchema."""
        collector = TelemetryCollector()
        assert collector.telemetry.version == "V1"
        assert collector.telemetry.tokens.prompt == 0
        assert collector.telemetry.latency.setup_s == 0.0
        assert collector.telemetry.tool_calls.total_calls == 0

    def test_track_phase_records_duration(self) -> None:
        """VAL-TELEM-05: track_phase() context manager records duration."""
        collector = TelemetryCollector()
        with collector.track_phase("execution"):
            time.sleep(0.05)
        # The execution phase should have some positive duration
        assert collector.telemetry.latency.execution_s >= 0.04

    def test_track_phase_setup(self) -> None:
        """track_phase("setup") records setup duration."""
        collector = TelemetryCollector()
        with collector.track_phase("setup"):
            time.sleep(0.03)
        assert collector.telemetry.latency.setup_s >= 0.02

    def test_track_phase_teardown(self) -> None:
        """track_phase("teardown") records teardown duration."""
        collector = TelemetryCollector()
        with collector.track_phase("teardown"):
            time.sleep(0.03)
        assert collector.telemetry.latency.teardown_s >= 0.02

    def test_phase_accumulation(self) -> None:
        """VAL-TELEM-05: Phase tracking accumulates across nested phases.

        Calling the same phase twice should ADD durations, not overwrite.
        """
        collector = TelemetryCollector()
        with collector.track_phase("execution"):
            time.sleep(0.02)
        first_duration = collector.telemetry.latency.execution_s

        with collector.track_phase("execution"):
            time.sleep(0.02)

        # Total should be approximately double the first duration
        assert collector.telemetry.latency.execution_s >= first_duration * 1.5

    def test_multiple_phases(self) -> None:
        """Multiple different phases record independently."""
        collector = TelemetryCollector()
        with collector.track_phase("setup"):
            time.sleep(0.02)
        with collector.track_phase("execution"):
            time.sleep(0.03)
        with collector.track_phase("teardown"):
            time.sleep(0.01)

        assert collector.telemetry.latency.setup_s >= 0.01
        assert collector.telemetry.latency.execution_s >= 0.02
        assert collector.telemetry.latency.teardown_s >= 0.005

    def test_total_s_is_updated(self) -> None:
        """total_s should be the sum of setup + execution + teardown."""
        collector = TelemetryCollector()
        with collector.track_phase("setup"):
            time.sleep(0.01)
        with collector.track_phase("execution"):
            time.sleep(0.01)
        with collector.track_phase("teardown"):
            time.sleep(0.01)

        expected_total = (
            collector.telemetry.latency.setup_s
            + collector.telemetry.latency.execution_s
            + collector.telemetry.latency.teardown_s
        )
        assert abs(collector.telemetry.latency.total_s - expected_total) < 0.05

    def test_track_phase_invalid_phase_raises(self) -> None:
        """track_phase with an invalid phase name raises ValueError."""
        collector = TelemetryCollector()
        with (
            pytest.raises(ValueError, match="Unknown phase"),
            collector.track_phase("invalid_phase"),
        ):
            pass

    def test_record_tokens(self) -> None:
        """Collector can record token usage."""
        collector = TelemetryCollector()
        collector.record_tokens(prompt=100, completion=50)
        assert collector.telemetry.tokens.prompt == 100
        assert collector.telemetry.tokens.completion == 50
        assert collector.telemetry.tokens.total == 150

    def test_record_tokens_accumulates(self) -> None:
        """Token recording accumulates across calls."""
        collector = TelemetryCollector()
        collector.record_tokens(prompt=50, completion=25)
        collector.record_tokens(prompt=50, completion=25)
        assert collector.telemetry.tokens.prompt == 100
        assert collector.telemetry.tokens.completion == 50
        assert collector.telemetry.tokens.total == 150

    def test_record_tool_call(self) -> None:
        """Collector can record individual tool calls."""
        collector = TelemetryCollector()
        collector.record_tool_call("bash", success=True, duration_ms=100.0)
        assert collector.telemetry.tool_calls.total_calls == 1
        assert collector.telemetry.tool_calls.successful_calls == 1
        assert collector.telemetry.tool_calls.failed_calls == 0
        assert collector.telemetry.tool_calls.by_tool["bash"]["calls"] == 1
        assert collector.telemetry.tool_calls.by_tool["bash"]["successes"] == 1
        assert collector.telemetry.tool_calls.total_duration_ms == 100.0

    def test_record_tool_call_failure(self) -> None:
        """Collector records failed tool calls."""
        collector = TelemetryCollector()
        collector.record_tool_call("grep", success=False, duration_ms=50.0)
        assert collector.telemetry.tool_calls.total_calls == 1
        assert collector.telemetry.tool_calls.successful_calls == 0
        assert collector.telemetry.tool_calls.failed_calls == 1
        assert collector.telemetry.tool_calls.by_tool["grep"]["failures"] == 1

    def test_record_raw_event(self) -> None:
        """Collector can record raw events for audit trail."""
        collector = TelemetryCollector()
        collector.record_raw_event({"type": "api_call", "latency_ms": 200})
        assert len(collector.telemetry.raw_events) == 1
        assert collector.telemetry.raw_events[0]["type"] == "api_call"

    def test_record_raw_events_multiple(self) -> None:
        """Collector accumulates raw events."""
        collector = TelemetryCollector()
        collector.record_raw_event({"type": "turn_1"})
        collector.record_raw_event({"type": "turn_2"})
        assert len(collector.telemetry.raw_events) == 2

    def test_set_estimated_cost(self) -> None:
        """Collector can set estimated cost."""
        collector = TelemetryCollector()
        collector.set_estimated_cost(0.005)
        assert collector.telemetry.estimated_cost_usd == 0.005

    def test_build_returns_telemetry(self) -> None:
        """build() returns the current telemetry snapshot."""
        collector = TelemetryCollector()
        collector.record_tokens(prompt=10, completion=5)
        result = collector.build()
        assert isinstance(result, TelemetrySchema)
        assert result.tokens.prompt == 10
        assert result.version == "V1"

    def test_build_returns_copy(self) -> None:
        """build() returns a copy, not a reference to internal state."""
        collector = TelemetryCollector()
        result1 = collector.build()
        collector.record_tokens(prompt=100, completion=50)
        result2 = collector.build()
        assert result1.tokens.prompt == 0
        assert result2.tokens.prompt == 100

    def test_phase_uses_start_time(self) -> None:
        """track_phase uses time.perf_counter for precision."""
        collector = TelemetryCollector()
        with collector.track_phase("setup"):
            time.sleep(0.01)
        assert collector.telemetry.latency.setup_s > 0
