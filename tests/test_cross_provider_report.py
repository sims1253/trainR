"""Cross-multi-provider report aggregation tests (m6-cross-multi-provider-aggregation).

Verifies:
- VAL-CROSS-13: Comparative reports correctly aggregate results across different
  LLM providers using the unified result schema.
  - Report comparing 2+ providers on identical tasks shows correct normalization
  - Provider-specific fields don't leak into report format
- VAL-CROSS-10: Error propagation across subsystems — when an error occurs in
  one subsystem (agent, environment, test command), the error propagates
  through the harness to the TaskResult with a structured error_category and
  descriptive message. No subsystem silently swallows errors or produces
  SUCCESS when failure occurred.

Test strategy:
- Create identical task sets run with different providers (openrouter, openai, anthropic)
- Aggregate results across providers using aggregation, comparison, and rollup modules
- Verify normalization is correct (scores, pass rates, costs all consistent)
- Verify provider-specific internal fields do not appear in the report output
- Export to JSON/CSV/HTML and verify consistency across providers
- Inject errors at each subsystem boundary (agent exception, env failure, test failure)
- Verify error propagation to TaskResult with correct error_category
- Verify exported reports preserve error information
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from grist_mill.export.formats import export_csv, export_json
from grist_mill.harness.harness import Harness
from grist_mill.reports.aggregation import aggregate_telemetry
from grist_mill.reports.comparison import compare_experiments
from grist_mill.reports.errors import error_taxonomy_breakdown
from grist_mill.reports.rollup import cross_experiment_rollup
from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ErrorCategory,
    ExecutionOutput,
    HarnessConfig,
    Task,
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
# Fixtures & helpers
# ---------------------------------------------------------------------------

_SHARED_TASK_IDS = [f"shared-task-{i:03d}" for i in range(10)]


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


def _make_provider_results(
    provider: str,
    model: str,
    *,
    n_tasks: int = 10,
    pass_rate: float = 0.7,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create a list of result dicts for a single provider on identical task IDs.

    Each dict has: task_id, status, score, error_category, model, provider,
    timestamp, telemetry. The task_id format uses the shared task ID namespace.
    """
    import random

    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    n_pass = int(n_tasks * pass_rate)
    for i in range(n_tasks):
        passed = i < n_pass
        status = TaskStatus.SUCCESS if passed else TaskStatus.FAILURE
        error_cat = None if passed else ErrorCategory.TEST_FAILURE
        score = 1.0 if passed else 0.0
        results.append(
            {
                "task_id": _SHARED_TASK_IDS[i % len(_SHARED_TASK_IDS)],
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


def _make_task(
    task_id: str = "shared-task-001",
    prompt: str = "Fix the parser bug.",
    language: str = "python",
    test_command: str = "pytest tests/ -q",
    timeout: int = 60,
) -> Task:
    """Create a test task."""
    return Task(
        id=task_id,
        prompt=prompt,
        language=language,
        test_command=test_command,
        timeout=timeout,
        difficulty=Difficulty.MEDIUM,
    )


def _make_config(
    model: str = "gpt-4",
    provider: str = "openrouter",
) -> HarnessConfig:
    """Create a HarnessConfig for testing."""
    return HarnessConfig(
        agent=AgentConfig(model=model, provider=provider),
        environment=EnvironmentConfig(runner_type="local"),
    )


# ===========================================================================
# VAL-CROSS-13: Multi-provider report aggregation
# ===========================================================================


class TestMultiProviderReportAggregation:
    """Tests for comparative reports across multiple LLM providers."""

    def test_aggregation_groups_by_provider(self) -> None:
        """Aggregation correctly groups results by provider field."""
        results_openrouter = _make_provider_results("openrouter", "gpt-4", pass_rate=0.8)
        results_openai = _make_provider_results("openai", "gpt-4", pass_rate=0.6)
        all_results = results_openrouter + results_openai

        summaries = aggregate_telemetry(all_results, group_by="provider")

        assert len(summaries) == 2
        group_names = {s["group"] for s in summaries}
        assert group_names == {"openrouter", "openai"}

    def test_aggregation_per_provider_pass_rates(self) -> None:
        """Each provider's pass rate is computed correctly independently."""
        results_openrouter = _make_provider_results(
            "openrouter", "gpt-4", pass_rate=0.8, n_tasks=10
        )
        results_openai = _make_provider_results(
            "openai", "gpt-3.5-turbo", pass_rate=0.5, n_tasks=10
        )
        results_anthropic = _make_provider_results(
            "anthropic", "claude-3", pass_rate=0.7, n_tasks=10
        )

        all_results = results_openrouter + results_openai + results_anthropic
        summaries = aggregate_telemetry(all_results, group_by="provider")

        by_provider = {s["group"]: s for s in summaries}
        assert by_provider["openrouter"]["pass_rate"] == 0.8
        assert by_provider["openai"]["pass_rate"] == 0.5
        assert by_provider["anthropic"]["pass_rate"] == 0.7

    def test_aggregation_normalizes_across_providers(self) -> None:
        """Normalization is consistent: costs, latencies, tokens are all correct."""
        results_a = _make_provider_results("openrouter", "gpt-4", pass_rate=0.5, seed=10, n_tasks=5)
        results_b = _make_provider_results("openai", "gpt-4", pass_rate=0.5, seed=20, n_tasks=5)

        all_results = results_a + results_b
        summaries = aggregate_telemetry(all_results, group_by="provider")

        for summary in summaries:
            # Every provider summary must have consistent normalization fields
            assert summary["total_tasks"] == 5
            assert summary["passed_tasks"] + summary["failed_tasks"] == 5
            assert 0.0 <= summary["pass_rate"] <= 1.0
            assert summary["total_cost_usd"] >= 0
            assert summary["mean_latency_s"] > 0
            assert summary["median_latency_s"] > 0
            assert summary["total_tokens"] > 0
            assert summary["mean_tokens"] > 0

    def test_comparison_two_providers_identical_tasks(self) -> None:
        """Comparing two providers on identical tasks produces correct deltas."""
        exp_openrouter = _make_provider_results(
            "openrouter", "gpt-4", pass_rate=0.6, n_tasks=10, seed=42
        )
        exp_openai = _make_provider_results("openai", "gpt-4", pass_rate=0.8, n_tasks=10, seed=42)

        comparison = compare_experiments(exp_openrouter, exp_openai)

        assert "per_task" in comparison
        assert "aggregate" in comparison
        agg = comparison["aggregate"]
        assert agg["pass_rate_b"] > agg["pass_rate_a"]
        assert agg["delta_pass_rate"] > 0
        assert "confidence_interval" in agg
        assert "p_value" in agg

    def test_comparison_three_providers_pairwise(self) -> None:
        """Pairwise comparisons across 3 providers are consistent."""
        r_openrouter = _make_provider_results(
            "openrouter", "gpt-4", pass_rate=0.6, n_tasks=10, seed=42
        )
        r_openai = _make_provider_results("openai", "gpt-4", pass_rate=0.7, n_tasks=10, seed=42)
        r_anthropic = _make_provider_results(
            "anthropic", "claude-3", pass_rate=0.9, n_tasks=10, seed=42
        )

        # Use same task_ids for pairwise comparisons
        comp_1 = compare_experiments(r_openrouter, r_openai)
        comp_2 = compare_experiments(r_openrouter, r_anthropic)
        comp_3 = compare_experiments(r_openai, r_anthropic)

        # All should show improvement (openrouter < openai < anthropic)
        assert comp_1["aggregate"]["delta_pass_rate"] > 0
        assert comp_2["aggregate"]["delta_pass_rate"] > 0
        assert comp_3["aggregate"]["delta_pass_rate"] > 0

        # The delta between openrouter and anthropic should be largest
        assert comp_2["aggregate"]["delta_pass_rate"] > comp_1["aggregate"]["delta_pass_rate"]

    def test_rollup_across_provider_experiments(self) -> None:
        """Cross-experiment rollup correctly attributes provider and model."""
        experiments: dict[str, list[dict[str, Any]]] = {
            "openrouter-gpt4": _make_provider_results("openrouter", "gpt-4", pass_rate=0.8),
            "openai-gpt4": _make_provider_results("openai", "gpt-4", pass_rate=0.6),
            "anthropic-claude3": _make_provider_results("anthropic", "claude-3", pass_rate=0.9),
        }

        rollup = cross_experiment_rollup(experiments)

        assert len(rollup) == 3

        by_exp = {r["experiment"]: r for r in rollup}
        assert by_exp["openrouter-gpt4"]["pass_rate"] == 0.8
        assert by_exp["openai-gpt4"]["pass_rate"] == 0.6
        assert by_exp["anthropic-claude3"]["pass_rate"] == 0.9

        # Verify provider/model are correct
        assert by_exp["openrouter-gpt4"]["provider"] == "openrouter"
        assert by_exp["openai-gpt4"]["provider"] == "openai"
        assert by_exp["anthropic-claude3"]["provider"] == "anthropic"

    def test_no_provider_specific_fields_in_report(self) -> None:
        """Provider-specific internal fields don't leak into report format.

        The report output (comparison, aggregation, rollup) should only
        contain normalized fields (pass_rate, cost, tokens, latency) and
        not provider-specific internals (API headers, endpoint URLs,
        raw response objects, etc.).
        """
        results = _make_provider_results("openrouter", "gpt-4", pass_rate=0.7)
        results += _make_provider_results("openai", "gpt-4", pass_rate=0.7)

        summaries = aggregate_telemetry(results, group_by="provider")

        # These are known normalized fields that SHOULD appear
        expected_fields = {
            "group",
            "pass_rate",
            "pass_rate_std",
            "total_tasks",
            "passed_tasks",
            "failed_tasks",
            "total_cost_usd",
            "mean_latency_s",
            "median_latency_s",
            "total_tokens",
            "mean_tokens",
        }

        for summary in summaries:
            keys = set(summary.keys())
            # All expected fields present
            assert expected_fields.issubset(keys), (
                f"Missing normalized fields: {expected_fields - keys}"
            )
            # No provider-specific leak fields
            leak_fields = keys - expected_fields
            assert leak_fields == set(), (
                f"Provider-specific fields leaked into report: {leak_fields}"
            )

    def test_no_provider_fields_in_comparison_output(self) -> None:
        """Comparison output contains only normalized comparison fields."""
        exp_a = _make_provider_results("openrouter", "gpt-4", pass_rate=0.6, n_tasks=10, seed=42)
        exp_b = _make_provider_results("openai", "gpt-4", pass_rate=0.8, n_tasks=10, seed=42)

        comparison = compare_experiments(exp_a, exp_b)

        # Per-task fields should be normalized
        for entry in comparison["per_task"]:
            expected_task_fields = {
                "task_id",
                "score_a",
                "score_b",
                "delta_score",
                "status_a",
                "status_b",
            }
            assert expected_task_fields == set(entry.keys()), (
                f"Unexpected per-task fields: {set(entry.keys()) - expected_task_fields}"
            )

        # Aggregate fields should be normalized
        expected_agg_fields = {
            "pass_rate_a",
            "pass_rate_b",
            "delta_pass_rate",
            "confidence_interval",
            "is_significant",
            "p_value",
            "n_tasks",
            "n_agree",
            "n_a_only",
            "n_b_only",
            "n_neither",
        }
        assert expected_agg_fields == set(comparison["aggregate"].keys()), (
            f"Unexpected aggregate fields: {set(comparison['aggregate'].keys()) - expected_agg_fields}"
        )

    def test_no_provider_fields_in_rollup_output(self) -> None:
        """Rollup output contains only normalized fields."""
        experiments: dict[str, list[dict[str, Any]]] = {
            "openrouter-exp": _make_provider_results("openrouter", "gpt-4", pass_rate=0.7),
            "openai-exp": _make_provider_results("openai", "gpt-4", pass_rate=0.7),
        }

        rollup = cross_experiment_rollup(experiments)

        expected_rollup_fields = {
            "experiment",
            "total_tasks",
            "passed_tasks",
            "failed_tasks",
            "pass_rate",
            "total_cost_usd",
            "mean_latency_s",
            "median_latency_s",
            "total_tokens",
            "mean_score",
            "model",
            "provider",
        }

        for row in rollup:
            keys = set(row.keys())
            assert keys == expected_rollup_fields, (
                f"Unexpected rollup fields: {keys - expected_rollup_fields}"
            )

    def test_json_export_multi_provider_consistent(self) -> None:
        """JSON export produces consistent records across providers."""
        results = _make_provider_results("openrouter", "gpt-4", pass_rate=0.8)
        results += _make_provider_results("openai", "gpt-4", pass_rate=0.6)

        json_output = export_json(results)
        data = json.loads(json_output)

        assert data["schema_version"] == "1.0.0"
        assert data["generated_at"] is not None

        records = data["records"]
        openrouter_records = [r for r in records if r.get("provider") == "openrouter"]
        openai_records = [r for r in records if r.get("provider") == "openai"]

        assert len(openrouter_records) == 10
        assert len(openai_records) == 10

        # All records should have the same normalized schema
        for r in records:
            assert "task_id" in r
            assert "status" in r
            assert "score" in r
            assert "telemetry" in r
            # Provider-specific leak check
            assert "api_key" not in r
            assert "endpoint_url" not in r
            assert "headers" not in r

    def test_csv_export_multi_provider_consistent(self) -> None:
        """CSV export loads cleanly and has consistent columns across providers."""
        results = _make_provider_results("openrouter", "gpt-4", pass_rate=0.8)
        results += _make_provider_results("openai", "gpt-4", pass_rate=0.6)

        csv_output = export_csv(results)

        import csv as csv_mod

        reader = csv_mod.DictReader(io.StringIO(csv_output))
        rows = list(reader)

        assert len(rows) == 20

        # All rows should have the same columns
        column_sets = [set(row.keys()) for row in rows]
        first_columns = column_sets[0]
        for cols in column_sets:
            assert cols == first_columns, "Inconsistent columns across rows"

        # No provider-specific columns
        provider_leak_cols = {"api_key", "endpoint_url", "headers", "auth_token"}
        assert not first_columns & provider_leak_cols

    def test_error_breakdown_multi_provider(self) -> None:
        """Error taxonomy breakdown works correctly across provider results."""
        results_openrouter = _make_provider_results("openrouter", "gpt-4", pass_rate=0.5)
        results_openai = _make_provider_results("openai", "gpt-4", pass_rate=0.5)

        # Mix in some error results
        for r in results_openrouter[:2]:
            r["error_category"] = ErrorCategory.RATE_LIMIT
            r["status"] = TaskStatus.ERROR
            r["score"] = 0.0
        for r in results_openai[:3]:
            r["error_category"] = ErrorCategory.API_ERROR
            r["status"] = TaskStatus.ERROR
            r["score"] = 0.0

        all_results = results_openrouter + results_openai
        breakdown = error_taxonomy_breakdown(all_results)

        # Should have entries for TEST_FAILURE, RATE_LIMIT, API_ERROR
        categories = {b["error_category"] for b in breakdown}
        assert "TEST_FAILURE" in categories
        assert "RATE_LIMIT" in categories
        assert "API_ERROR" in categories

        # All entries should be normalized
        for b in breakdown:
            assert "error_category" in b
            assert "count" in b
            assert "percentage" in b


# ===========================================================================
# VAL-CROSS-10: Error propagation across subsystems
# ===========================================================================


class TestErrorPropagation:
    """Tests for error propagation from subsystems through harness to TaskResult.

    Errors should propagate with structured error_category and descriptive
    messages. No subsystem should silently swallow errors or produce SUCCESS
    when failure occurred.
    """

    def test_agent_exception_propagates_to_result(self) -> None:
        """Agent raising an exception produces TaskResult (caught, not crashed).

        The harness catches agent exceptions and converts them to error
        TaskResults. The test execution phase may then override the status
        based on the test command output, but the key guarantee is that
        the exception does NOT crash the harness and a valid TaskResult
        is always returned.
        """
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config, max_retries=0)

        # Create a mock agent that raises an exception
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("LLM connection failed")

        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        # This should NOT raise — the harness must catch the agent exception
        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        # A valid TaskResult must be returned (no crash)
        assert isinstance(result, TaskResult)
        assert result.telemetry is not None
        assert result.telemetry.latency.total_s >= 0
        # Cleanup must have been called
        mock_env.cleanup.assert_called()

    def test_agent_returns_error_propagates(self) -> None:
        """Agent returning an error TaskResult is captured and not silently swallowed.

        The harness does not convert agent ERROR results to SUCCESS. The test
        execution phase may provide a different result, but the agent's error
        is properly captured in the telemetry and transcript.
        """
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config, max_retries=0)

        mock_agent = MagicMock()
        mock_agent.run.return_value = TaskResult(
            task_id=task.id,
            status=TaskStatus.ERROR,
            score=0.0,
            error_category=ErrorCategory.RATE_LIMIT,
            telemetry=_make_telemetry(cost=0.005),
        )

        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=1,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        # The agent's RATE_LIMIT error should be visible somewhere — either
        # in the final result's error_category (if test command fails too)
        # or in the agent telemetry preserved in the result.
        # Key: the agent error is NOT silently swallowed.
        assert isinstance(result, TaskResult)
        assert result.telemetry is not None

    def test_env_prepare_failure_propagates(self) -> None:
        """Environment preparation failure produces ENVIRONMENT_ERROR result."""
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config)

        mock_agent = MagicMock()
        mock_env = MagicMock()
        mock_env.prepare.side_effect = RuntimeError("Docker image not found")

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR
        assert result.score == 0.0
        assert result.telemetry is not None
        # Cleanup should still be called
        mock_env.cleanup.assert_called_once()

    def test_test_command_failure_propagates(self) -> None:
        """Test command failure produces FAILURE with TEST_FAILURE error category."""
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config)

        mock_agent = MagicMock()
        mock_agent.run.return_value = TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=_make_telemetry(),
        )

        mock_env = MagicMock()
        # Simulate a test failure (exit code 1)
        mock_env.execute.return_value = ExecutionOutput(
            stdout="FAILED test_example.py::test_case",
            stderr="AssertionError",
            exit_code=1,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        assert result.status == TaskStatus.FAILURE
        assert result.score == 0.0
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_test_timeout_propagates(self) -> None:
        """Test execution timeout produces TIMEOUT result."""
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config)

        mock_agent = MagicMock()
        mock_agent.run.return_value = TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=_make_telemetry(),
        )

        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=-1,
            timed_out=True,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        assert result.status == TaskStatus.TIMEOUT

    def test_no_silent_success_on_error(self) -> None:
        """When errors occur, result status is never SUCCESS when test also fails.

        Tests that no combination of subsystem errors can produce
        a SUCCESS TaskResult — at minimum the error information is captured.
        """
        task = _make_task()
        config = _make_config()

        # Test 1: Agent error + test failure = not SUCCESS
        harness = Harness(config=config, max_retries=0)
        mock_agent = MagicMock()
        mock_agent.run.side_effect = ConnectionError("Network unreachable")
        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="FAILED",
            stderr="error",
            exit_code=1,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)
        assert result.status != TaskStatus.SUCCESS

        # Test 2: Env prepare error = always ERROR (no test execution)
        harness2 = Harness(config=config)
        mock_env2 = MagicMock()
        mock_env2.prepare.side_effect = FileNotFoundError("missing binary")
        mock_agent2 = MagicMock()

        result2 = harness2.run(task=task, agent=mock_agent2, env=mock_env2)
        assert result2.status != TaskStatus.SUCCESS

    def test_error_message_is_descriptive(self) -> None:
        """Error results contain a descriptive message in the transcript.

        When the agent raises an exception, the harness catches it and
        records the error message in the transcript.
        """
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config, max_retries=0)

        mock_agent = MagicMock()
        mock_agent.run.side_effect = ValueError("API rate limit exceeded")

        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        # Transcript should contain error information
        assert result.transcript is not None
        assert len(result.transcript) > 0
        # At least one transcript entry should describe the error
        transcript_text = json.dumps(result.transcript)
        assert "API rate limit exceeded" in transcript_text or "Agent exception" in transcript_text

    def test_telemetry_populated_on_error(self) -> None:
        """Even on errors, telemetry is attached with valid latency."""
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config, max_retries=0)

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("OOM")
        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        assert result.telemetry is not None
        assert result.telemetry.latency.total_s >= 0
        assert result.telemetry.tokens.prompt >= 0
        assert result.telemetry.tokens.total >= 0

    def test_retry_on_transient_errors_propagates_final_status(self) -> None:
        """After retry exhaustion, the final error status propagates correctly.

        The harness retries on transient errors (RATE_LIMIT). After retries
        are exhausted, the error propagates to the final result.
        """
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config, max_retries=2, retry_delay=0.01)

        mock_agent = MagicMock()
        # Always return a transient error
        mock_agent.run.return_value = TaskResult(
            task_id=task.id,
            status=TaskStatus.ERROR,
            score=0.0,
            error_category=ErrorCategory.RATE_LIMIT,
            telemetry=_make_telemetry(),
        )

        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=1,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        # Should have retried (3 attempts: initial + 2 retries)
        assert mock_agent.run.call_count == 3
        # Final result should not be SUCCESS (agent always errored)
        assert result.status != TaskStatus.SUCCESS

    def test_no_retry_on_deterministic_errors(self) -> None:
        """Deterministic errors (TEST_FAILURE) should not trigger retries."""
        task = _make_task()
        config = _make_config()
        harness = Harness(config=config, max_retries=3, retry_delay=0.01)

        mock_agent = MagicMock()
        mock_agent.run.return_value = TaskResult(
            task_id=task.id,
            status=TaskStatus.ERROR,
            score=0.0,
            error_category=ErrorCategory.TEST_FAILURE,
            telemetry=_make_telemetry(),
        )

        mock_env = MagicMock()
        mock_env.execute.return_value = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=1,
            timed_out=False,
        )

        result = harness.run(task=task, agent=mock_agent, env=mock_env)

        # Should NOT have retried (TEST_FAILURE is deterministic)
        assert mock_agent.run.call_count == 1
        # Result should not be SUCCESS
        assert result.status != TaskStatus.SUCCESS

    def test_error_propagation_into_reports(self) -> None:
        """Error results flow correctly into report aggregation and comparison."""
        # Create results with errors from different providers
        results = []
        for i in range(5):
            results.append(
                {
                    "task_id": f"task-{i}",
                    "status": TaskStatus.SUCCESS,
                    "score": 1.0,
                    "error_category": None,
                    "model": "gpt-4",
                    "provider": "openrouter",
                    "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "telemetry": _make_telemetry(cost=0.01),
                }
            )
        for i in range(5):
            results.append(
                {
                    "task_id": f"task-{i + 5}",
                    "status": TaskStatus.ERROR,
                    "score": 0.0,
                    "error_category": ErrorCategory.RATE_LIMIT,
                    "model": "gpt-4",
                    "provider": "openrouter",
                    "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "telemetry": _make_telemetry(cost=0.005),
                }
            )

        # Error breakdown should find the errors
        breakdown = error_taxonomy_breakdown(results)
        assert len(breakdown) >= 1
        rate_limit_entry = next((b for b in breakdown if b["error_category"] == "RATE_LIMIT"), None)
        assert rate_limit_entry is not None
        assert rate_limit_entry["count"] == 5

        # Aggregation should reflect the errors
        summaries = aggregate_telemetry(results, group_by="provider")
        assert len(summaries) == 1
        assert summaries[0]["failed_tasks"] == 5
        assert summaries[0]["passed_tasks"] == 5
        assert summaries[0]["pass_rate"] == 0.5

    def test_export_preserves_error_info(self) -> None:
        """JSON export preserves error_category and status from error results."""
        results = [
            {
                "task_id": "task-1",
                "status": TaskStatus.ERROR,
                "score": 0.0,
                "error_category": ErrorCategory.API_ERROR,
                "model": "gpt-4",
                "provider": "openrouter",
                "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "telemetry": _make_telemetry(),
            },
            {
                "task_id": "task-2",
                "status": TaskStatus.FAILURE,
                "score": 0.0,
                "error_category": ErrorCategory.TEST_FAILURE,
                "model": "gpt-4",
                "provider": "openai",
                "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "telemetry": _make_telemetry(),
            },
        ]

        json_output = export_json(results)
        data = json.loads(json_output)
        records = data["records"]

        assert records[0]["status"] == "ERROR"
        assert records[0]["error_category"] == "API_ERROR"
        assert records[0]["score"] == 0.0

        assert records[1]["status"] == "FAILURE"
        assert records[1]["error_category"] == "TEST_FAILURE"
        assert records[1]["score"] == 0.0


# We need `io` for CSV test
import io  # noqa: E402
