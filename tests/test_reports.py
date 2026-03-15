"""Tests for the reporting module.

Covers:
- VAL-REPORT-01: Experiment comparison (per-task delta, aggregate pass-rate delta, significance)
- VAL-REPORT-02: Telemetry aggregation (per-model, per-tool, per-experiment summaries)
- VAL-REPORT-03: Per-tool performance breakdown
- VAL-REPORT-04: Cross-experiment telemetry rollup
- VAL-REPORT-05: Error taxonomy breakdown
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from grist_mill.schemas import (
    ErrorCategory,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.telemetry import (
    LatencyBreakdown,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_telemetry(
    *,
    prompt: int = 100,
    completion: int = 50,
    total_s: float = 5.0,
    setup_s: float = 1.0,
    execution_s: float = 3.0,
    teardown_s: float = 1.0,
    total_calls: int = 3,
    successful_calls: int = 2,
    failed_calls: int = 1,
    by_tool: dict[str, dict[str, Any]] | None = None,
    cost: float | None = 0.01,
) -> TelemetrySchema:
    """Create a TelemetrySchema with sensible defaults."""
    if by_tool is None:
        by_tool = {
            "file_read": {"calls": 2, "successes": 2, "failures": 0},
            "shell_exec": {"calls": 1, "successes": 0, "failures": 1},
        }
    return TelemetrySchema(
        version="V1",
        tokens=TokenUsage(prompt=prompt, completion=completion, total=prompt + completion),
        latency=LatencyBreakdown(
            setup_s=setup_s,
            execution_s=execution_s,
            teardown_s=teardown_s,
            total_s=total_s,
        ),
        tool_calls=ToolCallMetrics(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            by_tool=by_tool,
            total_duration_ms=total_calls * 100.0,
        ),
        estimated_cost_usd=cost,
    )


def _make_result(
    *,
    task_id: str = "task-1",
    status: TaskStatus = TaskStatus.SUCCESS,
    score: float = 1.0,
    error_category: ErrorCategory | None = None,
    model: str = "gpt-4",
    provider: str = "openrouter",
    telemetry: TelemetrySchema | None = None,
) -> TaskResult:
    """Create a TaskResult with extra metadata stored in telemetry raw_events."""
    tel = telemetry or _make_telemetry()
    # Store model/provider info in a way we can retrieve
    return TaskResult(
        task_id=task_id,
        status=status,
        score=score,
        error_category=error_category,
        telemetry=tel,
    )


def _make_experiment_results(
    model: str = "gpt-4",
    provider: str = "openrouter",
    n_tasks: int = 10,
    pass_rate: float = 0.7,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create a list of result dicts for an experiment.

    Each dict has: task_id, status, score, error_category, model, provider, telemetry.
    """
    results: list[dict[str, Any]] = []
    rng = _seeded_rng(seed)
    n_pass = int(n_tasks * pass_rate)
    for i in range(n_tasks):
        passed = i < n_pass
        status = TaskStatus.SUCCESS if passed else TaskStatus.FAILURE
        error_cat = None if passed else ErrorCategory.TEST_FAILURE
        score = 1.0 if passed else 0.0
        results.append(
            {
                "task_id": f"{model}-task-{i}",
                "status": status,
                "score": score,
                "error_category": error_cat,
                "model": model,
                "provider": provider,
                "timestamp": datetime(2025, 1, 1 + i % 5, tzinfo=timezone.utc),
                "telemetry": _make_telemetry(
                    prompt=100 + i * 10,
                    completion=50 + i * 5,
                    total_s=5.0 + rng.random(),
                    cost=0.01 + rng.random() * 0.01,
                ),
            }
        )
    return results


def _seeded_rng(seed: int = 42) -> Any:
    """Create a simple deterministic RNG."""
    import random

    rng = random.Random(seed)
    return rng


# ===========================================================================
# VAL-REPORT-01: Experiment Comparison
# ===========================================================================


class TestExperimentComparison:
    """Tests for comparing two experiments."""

    def test_per_task_delta(self) -> None:
        """Per-task comparison shows score delta between experiments."""
        from grist_mill.reports.comparison import compare_experiments

        exp_a = _make_experiment_results(model="gpt-4", pass_rate=0.6, seed=42, n_tasks=10)
        exp_b = _make_experiment_results(model="gpt-4", pass_rate=0.8, seed=42, n_tasks=10)

        comparison = compare_experiments(exp_a, exp_b)

        # Should have per-task deltas
        assert "per_task" in comparison
        per_task = comparison["per_task"]
        assert len(per_task) == 10
        for entry in per_task:
            assert "task_id" in entry
            assert "delta_score" in entry
            assert "status_a" in entry
            assert "status_b" in entry

    def test_aggregate_pass_rate_delta(self) -> None:
        """Aggregate comparison shows pass-rate delta with confidence interval."""
        from grist_mill.reports.comparison import compare_experiments

        exp_a = _make_experiment_results(model="gpt-4", pass_rate=0.6, seed=42, n_tasks=10)
        exp_b = _make_experiment_results(model="gpt-4", pass_rate=0.8, seed=42, n_tasks=10)

        comparison = compare_experiments(exp_a, exp_b)

        assert "aggregate" in comparison
        agg = comparison["aggregate"]
        assert "pass_rate_a" in agg
        assert "pass_rate_b" in agg
        assert "delta_pass_rate" in agg
        assert "confidence_interval" in agg
        ci = agg["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        # Pass rate B should be higher
        assert agg["pass_rate_b"] > agg["pass_rate_a"]
        assert agg["delta_pass_rate"] > 0

    def test_identical_experiments_zero_delta(self) -> None:
        """Identical experiments produce zero delta."""
        from grist_mill.reports.comparison import compare_experiments

        results = _make_experiment_results(model="gpt-4", pass_rate=0.7, seed=42, n_tasks=10)
        comparison = compare_experiments(results, results)

        agg = comparison["aggregate"]
        assert abs(agg["delta_pass_rate"]) < 1e-9

    def test_significance_with_large_delta(self) -> None:
        """Large delta should produce statistical significance indication."""
        from grist_mill.reports.comparison import compare_experiments

        # All pass vs all fail
        n = 20
        exp_a = [
            {
                "task_id": f"task-{i}",
                "status": TaskStatus.FAILURE,
                "score": 0.0,
                "error_category": ErrorCategory.TEST_FAILURE,
                "model": "gpt-4",
                "provider": "openrouter",
                "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "telemetry": _make_telemetry(),
            }
            for i in range(n)
        ]
        exp_b = [
            {
                "task_id": f"task-{i}",
                "status": TaskStatus.SUCCESS,
                "score": 1.0,
                "error_category": None,
                "model": "gpt-4",
                "provider": "openrouter",
                "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "telemetry": _make_telemetry(),
            }
            for i in range(n)
        ]

        comparison = compare_experiments(exp_a, exp_b)
        agg = comparison["aggregate"]
        # 100% delta should be significant (p < 0.05)
        assert agg["is_significant"] is True

    def test_different_task_sets_raises(self) -> None:
        """Comparing experiments with different task sets should raise ValueError."""
        from grist_mill.reports.comparison import compare_experiments

        exp_a = _make_experiment_results(model="gpt-4", n_tasks=5, seed=1)
        exp_b = _make_experiment_results(model="gpt-4", n_tasks=3, seed=2)

        with pytest.raises(ValueError, match="same task set"):
            compare_experiments(exp_a, exp_b)


# ===========================================================================
# VAL-REPORT-02: Telemetry Aggregation (per-model, per-tool, per-experiment)
# ===========================================================================


class TestTelemetryAggregation:
    """Tests for aggregating telemetry data."""

    def test_per_model_summary(self) -> None:
        """Aggregation produces per-model summaries with all metrics."""
        from grist_mill.reports.aggregation import aggregate_telemetry

        results_a = _make_experiment_results(model="gpt-4", pass_rate=0.8, n_tasks=10, seed=42)
        results_b = _make_experiment_results(model="claude-3", pass_rate=0.6, n_tasks=10, seed=99)
        all_results = results_a + results_b

        summaries = aggregate_telemetry(all_results, group_by="model")

        assert len(summaries) == 2
        model_names = {s["group"] for s in summaries}
        assert model_names == {"gpt-4", "claude-3"}

        for summary in summaries:
            assert "pass_rate" in summary
            assert "pass_rate_std" in summary
            assert "total_cost_usd" in summary
            assert "median_latency_s" in summary
            assert "mean_latency_s" in summary
            assert "total_tasks" in summary
            assert "passed_tasks" in summary
            assert "failed_tasks" in summary
            assert "total_tokens" in summary

    def test_per_experiment_summary(self) -> None:
        """Aggregation groups by experiment name."""
        from grist_mill.reports.aggregation import aggregate_telemetry

        results = []
        for name in ["exp-a", "exp-b"]:
            results.extend(
                _make_experiment_results(model="gpt-4", pass_rate=0.7, n_tasks=5, seed=hash(name))
            )
            # Tag the experiment name on results
            for r in results[-5:]:
                r["experiment"] = name

        summaries = aggregate_telemetry(results, group_by="experiment")
        assert len(summaries) == 2

    def test_empty_results(self) -> None:
        """Empty input produces empty summary list."""
        from grist_mill.reports.aggregation import aggregate_telemetry

        summaries = aggregate_telemetry([], group_by="model")
        assert summaries == []

    def test_single_result(self) -> None:
        """Single result produces summary with zero std."""
        from grist_mill.reports.aggregation import aggregate_telemetry

        results = _make_experiment_results(model="gpt-4", n_tasks=1, seed=42)
        summaries = aggregate_telemetry(results, group_by="model")

        assert len(summaries) == 1
        assert summaries[0]["pass_rate_std"] == 0.0
        assert summaries[0]["total_tasks"] == 1

    def test_cost_summation(self) -> None:
        """Total cost is sum of individual estimated costs (rounded to 6 decimals)."""
        from grist_mill.reports.aggregation import aggregate_telemetry

        results = _make_experiment_results(model="gpt-4", n_tasks=5, seed=42)
        summaries = aggregate_telemetry(results, group_by="model")

        expected_cost = sum(r["telemetry"].estimated_cost_usd or 0.0 for r in results)
        # Aggregation rounds to 6 decimal places
        assert abs(summaries[0]["total_cost_usd"] - round(expected_cost, 6)) < 1e-9


# ===========================================================================
# VAL-REPORT-03: Per-tool performance breakdown
# ===========================================================================


class TestToolPerformanceBreakdown:
    """Tests for per-tool performance tables."""

    def test_per_tool_table(self) -> None:
        """Tool breakdown produces success/error rates and avg latency per tool."""
        from grist_mill.reports.tools import tool_performance_breakdown

        results = _make_experiment_results(model="gpt-4", n_tasks=5, seed=42)
        breakdown = tool_performance_breakdown(results)

        assert isinstance(breakdown, list)
        assert len(breakdown) > 0
        for entry in breakdown:
            assert "tool_name" in entry
            assert "total_calls" in entry
            assert "successful_calls" in entry
            assert "failed_calls" in entry
            assert "success_rate" in entry
            assert "error_rate" in entry
            assert "avg_duration_ms" in entry
            # Verify rates are between 0 and 1
            assert 0.0 <= entry["success_rate"] <= 1.0
            assert 0.0 <= entry["error_rate"] <= 1.0

    def test_empty_results_empty_table(self) -> None:
        """Empty input produces empty table."""
        from grist_mill.reports.tools import tool_performance_breakdown

        breakdown = tool_performance_breakdown([])
        assert breakdown == []

    def test_tools_from_multiple_results(self) -> None:
        """Multiple results aggregate tool metrics correctly."""
        from grist_mill.reports.tools import tool_performance_breakdown

        results = _make_experiment_results(model="gpt-4", n_tasks=10, seed=42)
        breakdown = tool_performance_breakdown(results)

        # file_read should have all successes
        file_read = next((e for e in breakdown if e["tool_name"] == "file_read"), None)
        assert file_read is not None
        assert file_read["total_calls"] == 10 * 2  # 2 calls per result
        assert file_read["success_rate"] == 1.0


# ===========================================================================
# VAL-REPORT-04: Cross-experiment rollup
# ===========================================================================


class TestCrossExperimentRollup:
    """Tests for rollup across multiple experiments."""

    def test_rollup_one_row_per_experiment(self) -> None:
        """Rollup across experiments produces one row per config."""
        from grist_mill.reports.rollup import cross_experiment_rollup

        experiments: dict[str, list[dict[str, Any]]] = {}
        for name in ["exp-a", "exp-b", "exp-c"]:
            experiments[name] = _make_experiment_results(
                model="gpt-4", n_tasks=5, seed=hash(name) % 1000
            )

        rollup = cross_experiment_rollup(experiments)

        assert len(rollup) == 3
        exp_names = {r["experiment"] for r in rollup}
        assert exp_names == {"exp-a", "exp-b", "exp-c"}

        for row in rollup:
            assert "experiment" in row
            assert "total_tasks" in row
            assert "pass_rate" in row
            assert "total_cost_usd" in row
            assert "mean_latency_s" in row

    def test_empty_experiments(self) -> None:
        """Empty experiment dict produces empty rollup."""
        from grist_mill.reports.rollup import cross_experiment_rollup

        rollup = cross_experiment_rollup({})
        assert rollup == []

    def test_single_experiment(self) -> None:
        """Single experiment produces single-row rollup."""
        from grist_mill.reports.rollup import cross_experiment_rollup

        experiments = {"exp-1": _make_experiment_results(n_tasks=3, seed=42)}
        rollup = cross_experiment_rollup(experiments)

        assert len(rollup) == 1
        assert rollup[0]["experiment"] == "exp-1"
        assert rollup[0]["total_tasks"] == 3


# ===========================================================================
# VAL-REPORT-05: Error taxonomy breakdown
# ===========================================================================


class TestErrorTaxonomyBreakdown:
    """Tests for error taxonomy breakdown."""

    def test_error_categories_counted(self) -> None:
        """Error breakdown groups and counts by error category."""
        from grist_mill.reports.errors import error_taxonomy_breakdown

        results: list[dict[str, Any]] = []
        for i in range(3):
            results.append(
                {
                    "task_id": f"task-{i}",
                    "status": TaskStatus.FAILURE,
                    "score": 0.0,
                    "error_category": ErrorCategory.TEST_FAILURE,
                    "model": "gpt-4",
                    "provider": "openrouter",
                    "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "telemetry": _make_telemetry(),
                }
            )
        for i in range(2):
            results.append(
                {
                    "task_id": f"task-{i + 3}",
                    "status": TaskStatus.ERROR,
                    "score": 0.0,
                    "error_category": ErrorCategory.API_ERROR,
                    "model": "gpt-4",
                    "provider": "openrouter",
                    "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "telemetry": _make_telemetry(),
                }
            )
        # Add a success
        results.append(
            {
                "task_id": "task-5",
                "status": TaskStatus.SUCCESS,
                "score": 1.0,
                "error_category": None,
                "model": "gpt-4",
                "provider": "openrouter",
                "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "telemetry": _make_telemetry(),
            }
        )

        breakdown = error_taxonomy_breakdown(results)

        assert isinstance(breakdown, list)
        # Should have entries for TEST_FAILURE and API_ERROR
        categories = {e["error_category"] for e in breakdown}
        assert "TEST_FAILURE" in categories
        assert "API_ERROR" in categories

        test_failure_entry = next(e for e in breakdown if e["error_category"] == "TEST_FAILURE")
        assert test_failure_entry["count"] == 3
        assert test_failure_entry["percentage"] > 0

    def test_all_success_no_errors(self) -> None:
        """All-success experiment produces empty error breakdown."""
        from grist_mill.reports.errors import error_taxonomy_breakdown

        results = _make_experiment_results(model="gpt-4", pass_rate=1.0, n_tasks=5, seed=42)
        breakdown = error_taxonomy_breakdown(results)

        assert breakdown == []

    def test_empty_results(self) -> None:
        """Empty input produces empty breakdown."""
        from grist_mill.reports.errors import error_taxonomy_breakdown

        breakdown = error_taxonomy_breakdown([])
        assert breakdown == []

    def test_percentage_sums_to_100(self) -> None:
        """Percentages sum to 100% for error tasks."""
        from grist_mill.reports.errors import error_taxonomy_breakdown

        results: list[dict[str, Any]] = []
        categories = [
            ErrorCategory.TEST_FAILURE,
            ErrorCategory.API_ERROR,
            ErrorCategory.SYNTAX_ERROR,
        ]
        for i, cat in enumerate(categories):
            for _ in range(i + 1):
                results.append(
                    {
                        "task_id": f"task-{len(results)}",
                        "status": TaskStatus.FAILURE,
                        "score": 0.0,
                        "error_category": cat,
                        "model": "gpt-4",
                        "provider": "openrouter",
                        "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                        "telemetry": _make_telemetry(),
                    }
                )

        breakdown = error_taxonomy_breakdown(results)
        total_pct = sum(e["percentage"] for e in breakdown)
        assert abs(total_pct - 100.0) < 1e-6


# ===========================================================================
# Filtering helpers
# ===========================================================================


class TestFiltering:
    """Tests for filtering result sets."""

    def test_filter_by_model(self) -> None:
        """Filtering by model returns only matching results."""
        from grist_mill.reports.filtering import filter_results

        results = _make_experiment_results(model="gpt-4", n_tasks=5, seed=42)
        results.extend(_make_experiment_results(model="claude-3", n_tasks=3, seed=99))

        filtered = filter_results(results, model="gpt-4")
        assert len(filtered) == 5

    def test_filter_by_tool(self) -> None:
        """Filtering by tool returns only results that used that tool."""
        from grist_mill.reports.filtering import filter_results

        results = _make_experiment_results(model="gpt-4", n_tasks=5, seed=42)
        filtered = filter_results(results, tool="file_read")
        # All results use file_read tool
        assert len(filtered) == 5

    def test_filter_by_date_range(self) -> None:
        """Filtering by date range returns only results in range."""
        from grist_mill.reports.filtering import filter_results

        results = _make_experiment_results(model="gpt-4", n_tasks=10, seed=42)
        # Results have timestamps from 2025-01-01 to 2025-01-05
        from datetime import date

        filtered = filter_results(
            results,
            date_range=(date(2025, 1, 2), date(2025, 1, 3)),
        )
        assert len(filtered) > 0
        assert len(filtered) < 10

    def test_filter_empty_results(self) -> None:
        """Filtering empty list returns empty list."""
        from grist_mill.reports.filtering import filter_results

        filtered = filter_results([], model="gpt-4")
        assert filtered == []

    def test_filter_no_match(self) -> None:
        """Filter that matches nothing returns empty list."""
        from grist_mill.reports.filtering import filter_results

        results = _make_experiment_results(model="gpt-4", n_tasks=5, seed=42)
        filtered = filter_results(results, model="nonexistent")
        assert filtered == []
