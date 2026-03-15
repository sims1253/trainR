"""Tests for the GEPA evaluator adapter.

Validates:
- VAL-GEPA-01: Evaluator adapter captures actionable side information
- VAL-GEPA-02: Side information is passed to reflection model
- VAL-GEPA-03: Multi-objective evaluation produces composite scores
- VAL-GEPA-06: Evaluator adapter is pluggable
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from grist_mill.optimization.evaluator_adapter import (
    CostAdjustedObjective,
    DifficultyWeightedObjective,
    EvaluatorAdapterConfig,
    GepaEvaluatorAdapter,
    ObjectiveFunction,
    PassRateObjective,
    create_evaluator_adapter,
    load_custom_evaluator,
)
from grist_mill.schemas import (
    ErrorCategory,
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
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "task-1",
    difficulty: str = "EASY",
    timeout: int = 30,
) -> Task:
    """Create a simple test task."""
    return Task(
        id=task_id,
        prompt="Fix the failing test",
        language="python",
        test_command="pytest test_foo.py",
        timeout=timeout,
        difficulty=difficulty,
    )


def _make_success_result(task_id: str = "task-1") -> TaskResult:
    """Create a successful TaskResult with telemetry."""
    telemetry = TelemetrySchema(
        tokens=TokenUsage(prompt=100, completion=50, total=150),
        latency=LatencyBreakdown(setup_s=0.5, execution_s=2.0, teardown_s=0.3, total_s=2.8),
        tool_calls=ToolCallMetrics(total_calls=3, successful_calls=3, failed_calls=0),
    )
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        score=1.0,
        telemetry=telemetry,
    )


def _make_failure_result(task_id: str = "task-1") -> TaskResult:
    """Create a failing TaskResult with telemetry and error info."""
    telemetry = TelemetrySchema(
        tokens=TokenUsage(prompt=200, completion=100, total=300),
        latency=LatencyBreakdown(setup_s=0.5, execution_s=5.0, teardown_s=0.3, total_s=5.8),
        tool_calls=ToolCallMetrics(
            total_calls=5,
            successful_calls=3,
            failed_calls=2,
            by_tool={"bash": {"calls": 3, "successes": 2, "failures": 1}},
        ),
    )
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.FAILURE,
        score=0.0,
        error_category=ErrorCategory.TEST_FAILURE,
        telemetry=telemetry,
        transcript=[
            {"message": "Test failed: assert 1 == 2"},
            {"stdout": "FAILED test_foo.py::test_bar"},
        ],
    )


def _make_timeout_result(task_id: str = "task-1") -> TaskResult:
    """Create a timeout TaskResult."""
    telemetry = TelemetrySchema(
        tokens=TokenUsage(prompt=50, completion=0, total=50),
        latency=LatencyBreakdown(setup_s=0.5, execution_s=30.0, teardown_s=0.3, total_s=30.8),
        tool_calls=ToolCallMetrics(total_calls=1, successful_calls=1, failed_calls=0),
    )
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.TIMEOUT,
        score=0.0,
        telemetry=telemetry,
    )


def _make_error_result(task_id: str = "task-1") -> TaskResult:
    """Create an error TaskResult with API error."""
    telemetry = TelemetrySchema(
        tokens=TokenUsage(prompt=0, completion=0, total=0),
        latency=LatencyBreakdown(setup_s=0.1, execution_s=0.0, teardown_s=0.1, total_s=0.2),
        tool_calls=ToolCallMetrics(total_calls=0, successful_calls=0, failed_calls=0),
    )
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.ERROR,
        score=0.0,
        error_category=ErrorCategory.API_ERROR,
        telemetry=telemetry,
        transcript=[{"message": "Rate limit exceeded"}],
    )


def _make_harness_config() -> HarnessConfig:
    """Create a simple HarnessConfig."""
    from grist_mill.schemas import AgentConfig, EnvironmentConfig

    return HarnessConfig(
        agent=AgentConfig(model="gpt-4", provider="openai"),
        environment=EnvironmentConfig(runner_type="local"),
    )


# ---------------------------------------------------------------------------
# Test Objective Functions
# ---------------------------------------------------------------------------


class TestPassRateObjective:
    """Tests for PassRateObjective."""

    def test_all_pass(self) -> None:
        """All tasks passing gives score 1.0."""
        results = [_make_success_result("t1"), _make_success_result("t2")]
        tasks = [_make_task("t1"), _make_task("t2")]
        obj = PassRateObjective()
        score = obj.compute(results, tasks)
        assert score == pytest.approx(1.0)

    def test_half_pass(self) -> None:
        """Half tasks passing gives score 0.5."""
        results = [_make_success_result("t1"), _make_failure_result("t2")]
        tasks = [_make_task("t1"), _make_task("t2")]
        obj = PassRateObjective()
        score = obj.compute(results, tasks)
        assert score == pytest.approx(0.5)

    def test_none_pass(self) -> None:
        """No tasks passing gives score 0.0."""
        results = [_make_failure_result("t1"), _make_failure_result("t2")]
        tasks = [_make_task("t1"), _make_task("t2")]
        obj = PassRateObjective()
        score = obj.compute(results, tasks)
        assert score == pytest.approx(0.0)

    def test_empty_results(self) -> None:
        """Empty results list gives score 0.0."""
        obj = PassRateObjective()
        score = obj.compute([], [])
        assert score == pytest.approx(0.0)

    def test_single_success(self) -> None:
        """Single successful task gives score 1.0."""
        results = [_make_success_result("t1")]
        tasks = [_make_task("t1")]
        obj = PassRateObjective()
        score = obj.compute(results, tasks)
        assert score == pytest.approx(1.0)

    def test_deterministic(self) -> None:
        """Same inputs always produce same score."""
        results = [_make_success_result("t1"), _make_failure_result("t2")]
        tasks = [_make_task("t1"), _make_task("t2")]
        obj = PassRateObjective()
        s1 = obj.compute(results, tasks)
        s2 = obj.compute(results, tasks)
        assert s1 == s2


class TestCostAdjustedObjective:
    """Tests for CostAdjustedObjective."""

    def test_cheap_success(self) -> None:
        """Success with low cost gets high score."""
        results = [_make_success_result("t1")]
        tasks = [_make_task("t1")]
        obj = CostAdjustedObjective(cost_per_token=0.0001)
        score = obj.compute(results, tasks)
        assert 0.0 < score <= 1.0

    def test_expensive_success_lower_score(self) -> None:
        """Success with high cost gets lower score than cheap success."""
        cheap_telemetry = TelemetrySchema(
            tokens=TokenUsage(prompt=10, completion=5, total=15),
            latency=LatencyBreakdown(setup_s=0.1, execution_s=0.5, teardown_s=0.1, total_s=0.7),
            tool_calls=ToolCallMetrics(total_calls=1, successful_calls=1, failed_calls=0),
        )
        expensive_telemetry = TelemetrySchema(
            tokens=TokenUsage(prompt=10000, completion=5000, total=15000),
            latency=LatencyBreakdown(setup_s=0.1, execution_s=50.0, teardown_s=0.1, total_s=50.2),
            tool_calls=ToolCallMetrics(total_calls=100, successful_calls=100, failed_calls=0),
        )
        cheap_result = TaskResult(
            task_id="t1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=cheap_telemetry,
        )
        expensive_result = TaskResult(
            task_id="t1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=expensive_telemetry,
        )
        tasks = [_make_task("t1")]
        obj = CostAdjustedObjective(cost_per_token=0.001)
        cheap_score = obj.compute([cheap_result], tasks)
        expensive_score = obj.compute([expensive_result], tasks)
        assert expensive_score < cheap_score

    def test_failure_zero(self) -> None:
        """Failed task always gets 0.0 regardless of cost."""
        results = [_make_failure_result("t1")]
        tasks = [_make_task("t1")]
        obj = CostAdjustedObjective(cost_per_token=0.001)
        score = obj.compute(results, tasks)
        assert score == pytest.approx(0.0)

    def test_deterministic(self) -> None:
        """Same inputs produce same score."""
        results = [_make_success_result("t1")]
        tasks = [_make_task("t1")]
        obj = CostAdjustedObjective(cost_per_token=0.001)
        s1 = obj.compute(results, tasks)
        s2 = obj.compute(results, tasks)
        assert s1 == s2


class TestDifficultyWeightedObjective:
    """Tests for DifficultyWeightedObjective."""

    def test_hard_success_weighs_more(self) -> None:
        """Solving a HARD task contributes more than solving an EASY task."""
        hard_result = _make_success_result("t1")
        easy_result = _make_success_result("t2")
        hard_task = _make_task("t1", difficulty="HARD")
        easy_task = _make_task("t2", difficulty="EASY")

        obj = DifficultyWeightedObjective()

        # Score when hard succeeds and easy fails
        score_hard_only = obj.compute(
            [hard_result, _make_failure_result("t2")], [hard_task, easy_task]
        )
        # Score when easy succeeds and hard fails
        score_easy_only = obj.compute(
            [_make_failure_result("t1"), easy_result], [hard_task, easy_task]
        )

        assert score_hard_only > score_easy_only

    def test_all_difficulties_pass(self) -> None:
        """All tasks passing gives score 1.0 regardless of difficulty."""
        results = [
            _make_success_result("t1"),
            _make_success_result("t2"),
            _make_success_result("t3"),
        ]
        tasks = [
            _make_task("t1", difficulty="EASY"),
            _make_task("t2", difficulty="MEDIUM"),
            _make_task("t3", difficulty="HARD"),
        ]
        obj = DifficultyWeightedObjective()
        score = obj.compute(results, tasks)
        assert score == pytest.approx(1.0)

    def test_none_pass(self) -> None:
        """No tasks passing gives 0.0."""
        results = [_make_failure_result("t1")]
        tasks = [_make_task("t1", difficulty="HARD")]
        obj = DifficultyWeightedObjective()
        score = obj.compute(results, tasks)
        assert score == pytest.approx(0.0)

    def test_deterministic(self) -> None:
        """Same inputs produce same score."""
        results = [_make_success_result("t1"), _make_failure_result("t2")]
        tasks = [_make_task("t1", difficulty="EASY"), _make_task("t2", difficulty="HARD")]
        obj = DifficultyWeightedObjective()
        s1 = obj.compute(results, tasks)
        s2 = obj.compute(results, tasks)
        assert s1 == s2


class TestObjectiveFactory:
    """Tests for objective function creation from config."""

    def test_create_pass_rate(self) -> None:
        """Pass-rate objective created from config string."""
        obj = ObjectiveFunction.create("pass-rate")
        assert isinstance(obj, PassRateObjective)

    def test_create_cost_adjusted(self) -> None:
        """Cost-adjusted objective created from config string."""
        obj = ObjectiveFunction.create("cost-adjusted", cost_per_token=0.001)
        assert isinstance(obj, CostAdjustedObjective)
        assert obj.cost_per_token == 0.001

    def test_create_difficulty_weighted(self) -> None:
        """Difficulty-weighted objective created from config string."""
        obj = ObjectiveFunction.create("difficulty-weighted")
        assert isinstance(obj, DifficultyWeightedObjective)

    def test_invalid_objective_raises(self) -> None:
        """Invalid objective name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown objective"):
            ObjectiveFunction.create("nonexistent")

    def test_objectives_produce_distinct_scores(self) -> None:
        """Different objectives produce different scores for the same results."""
        results = [
            _make_success_result("t1"),
            _make_failure_result("t2"),
            _make_success_result("t3"),
        ]
        tasks = [
            _make_task("t1", difficulty="HARD"),
            _make_task("t2", difficulty="MEDIUM"),
            _make_task("t3", difficulty="EASY"),
        ]

        pass_rate = ObjectiveFunction.create("pass-rate").compute(results, tasks)
        cost_adj = ObjectiveFunction.create("cost-adjusted", cost_per_token=0.001).compute(
            results, tasks
        )
        diff_weighted = ObjectiveFunction.create("difficulty-weighted").compute(results, tasks)

        # All should be in [0, 1]
        assert 0.0 <= pass_rate <= 1.0
        assert 0.0 <= cost_adj <= 1.0
        assert 0.0 <= diff_weighted <= 1.0

        # Cost-adjusted should be different from pass-rate (due to cost penalty)
        assert cost_adj != pass_rate


# ---------------------------------------------------------------------------
# Test GepaEvaluatorAdapter
# ---------------------------------------------------------------------------


class TestGepaEvaluatorAdapter:
    """Tests for the main GepaEvaluatorAdapter class."""

    def _make_adapter(
        self,
        objective: str = "pass-rate",
        trace_enabled: bool = False,
    ) -> GepaEvaluatorAdapter:
        """Create an adapter with a mock harness."""
        mock_harness = MagicMock()
        config = EvaluatorAdapterConfig(
            objective=objective,
            trace_enabled=trace_enabled,
        )
        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=config,
        )
        return adapter

    def test_returns_tuple(self) -> None:
        """Adapter returns (float, dict) tuple."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        score, side_info = adapter(candidate="skill prompt", example=task)

        assert isinstance(score, float)
        assert isinstance(side_info, dict)

    def test_success_returns_score_1_0(self) -> None:
        """Successful task returns score 1.0."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        score, _side_info = adapter(candidate="skill prompt", example=task)
        assert score == pytest.approx(1.0)

    def test_failure_returns_score_0_0(self) -> None:
        """Failed task returns score 0.0."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_failure_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        score, _side_info = adapter(candidate="skill prompt", example=task)
        assert score == pytest.approx(0.0)

    # --- VAL-GEPA-01: Side information contains traces, errors, timing, token usage ---

    def test_side_info_has_task_id(self) -> None:
        """Side info contains task_id."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_failure_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert side_info["task_id"] == "task-1"

    def test_side_info_has_status(self) -> None:
        """Side info contains task status."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_failure_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert side_info["status"] == "FAILURE"

    def test_side_info_has_traces(self) -> None:
        """Side info contains execution traces (transcript)."""
        mock_harness = MagicMock()
        result = _make_failure_result("task-1")
        mock_harness.run.return_value = result

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert "traces" in side_info
        assert isinstance(side_info["traces"], list)

    def test_side_info_has_errors(self) -> None:
        """Side info contains error info for failed tasks."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_failure_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert "errors" in side_info
        assert side_info["errors"] is not None

    def test_side_info_has_duration_s(self) -> None:
        """Side info contains execution duration in seconds."""
        mock_harness = MagicMock()
        result = _make_success_result("task-1")
        mock_harness.run.return_value = result

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert "duration_s" in side_info
        assert side_info["duration_s"] > 0

    def test_side_info_has_token_usage(self) -> None:
        """Side info contains token usage."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert "token_usage" in side_info
        assert "prompt" in side_info["token_usage"]
        assert "completion" in side_info["token_usage"]
        assert "total" in side_info["token_usage"]

    def test_side_info_has_error_category(self) -> None:
        """Side info contains error category when present."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_failure_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert side_info["error_category"] == "TEST_FAILURE"

    def test_side_info_for_timeout(self) -> None:
        """Side info correctly captures timeout results."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_timeout_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert side_info["status"] == "TIMEOUT"
        assert side_info["error_category"] is None
        assert side_info["duration_s"] >= 0.0

    def test_side_info_for_error(self) -> None:
        """Side info correctly captures error results."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_error_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert side_info["status"] == "ERROR"
        assert side_info["error_category"] == "API_ERROR"

    def test_side_info_null_telemetry(self) -> None:
        """Side info handles null telemetry gracefully."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = TaskResult(
            task_id="task-1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=None,
        )

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill prompt", example=task)
        assert side_info["token_usage"]["total"] == 0
        # duration_s is wall-clock time (always > 0 due to harness call)
        assert side_info["duration_s"] >= 0.0

    # --- VAL-GEPA-03: Configurable objectives produce distinct scores ---

    def test_pass_rate_objective(self) -> None:
        """Pass-rate objective correctly computes pass rate."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        score, _ = adapter(candidate="skill prompt", example=task)
        assert score == pytest.approx(1.0)

    def test_cost_adjusted_objective(self) -> None:
        """Cost-adjusted objective penalizes high cost."""
        # Create results with different costs
        cheap_result = TaskResult(
            task_id="task-1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=TelemetrySchema(
                tokens=TokenUsage(prompt=10, completion=5, total=15),
                latency=LatencyBreakdown(
                    setup_s=0.1,
                    execution_s=0.5,
                    teardown_s=0.1,
                    total_s=0.7,
                ),
            ),
        )
        expensive_result = TaskResult(
            task_id="task-1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=TelemetrySchema(
                tokens=TokenUsage(prompt=10000, completion=5000, total=15000),
                latency=LatencyBreakdown(
                    setup_s=0.1,
                    execution_s=50.0,
                    teardown_s=0.1,
                    total_s=50.2,
                ),
            ),
        )
        task = _make_task()

        # Cheap evaluation
        mock_harness_cheap = MagicMock()
        mock_harness_cheap.run.return_value = cheap_result
        adapter_cheap = GepaEvaluatorAdapter(
            harness=mock_harness_cheap,
            config=EvaluatorAdapterConfig(
                objective="cost-adjusted",
                cost_per_token=0.001,
            ),
        )
        cheap_score, _ = adapter_cheap(candidate="skill", example=task)

        # Expensive evaluation
        mock_harness_expensive = MagicMock()
        mock_harness_expensive.run.return_value = expensive_result
        adapter_expensive = GepaEvaluatorAdapter(
            harness=mock_harness_expensive,
            config=EvaluatorAdapterConfig(
                objective="cost-adjusted",
                cost_per_token=0.001,
            ),
        )
        expensive_score, _ = adapter_expensive(candidate="skill", example=task)

        assert expensive_score < cheap_score
        assert cheap_score <= 1.0
        assert expensive_score > 0.0

    def test_difficulty_weighted_objective(self) -> None:
        """Difficulty-weighted objective correctly weights by difficulty."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="difficulty-weighted"),
        )
        hard_task = _make_task("task-1", difficulty="HARD")

        score, _ = adapter(candidate="skill", example=hard_task)
        # Hard task success should get higher than 1/N for N=1 tasks
        assert score > 0.0

    def test_objectives_deterministic(self) -> None:
        """All objectives produce deterministic scores."""
        mock_harness = MagicMock()
        result = _make_success_result("task-1")
        mock_harness.run.return_value = result
        task = _make_task()

        for objective_name in ["pass-rate", "cost-adjusted", "difficulty-weighted"]:
            adapter = GepaEvaluatorAdapter(
                harness=mock_harness,
                config=EvaluatorAdapterConfig(
                    objective=objective_name,
                    cost_per_token=0.001,
                ),
            )
            scores = set()
            for _ in range(5):
                s, _ = adapter(candidate="skill", example=task)
                scores.add(s)
            assert len(scores) == 1, f"Objective {objective_name} is not deterministic"

    # --- VAL-GEPA-02: Side info forwarded to reflection model ---

    def test_side_info_has_difficulty(self) -> None:
        """Side info contains task difficulty for reflection."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task("task-1", difficulty="HARD")

        _score, side_info = adapter(candidate="skill", example=task)
        assert side_info["difficulty"] == "HARD"

    def test_side_info_forwards_error_messages(self) -> None:
        """Side info forwards error messages from transcript for reflection."""
        mock_harness = MagicMock()
        result = _make_failure_result("task-1")
        mock_harness.run.return_value = result

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill", example=task)
        # Error messages should be extractable for reflection
        errors = side_info.get("errors", [])
        assert isinstance(errors, list)
        # At minimum, we should have some error info
        assert side_info.get("error_category") == "TEST_FAILURE"

    def test_side_info_forwards_trace_events(self) -> None:
        """Side info contains trace events when trace_enabled=True."""
        mock_harness = MagicMock()
        telemetry = TelemetrySchema(
            tokens=TokenUsage(prompt=10, completion=5, total=15),
            latency=LatencyBreakdown(setup_s=0.1, execution_s=0.5, teardown_s=0.1, total_s=0.7),
            tool_calls=ToolCallMetrics(total_calls=1, successful_calls=1, failed_calls=0),
            raw_events=[
                {"phase": "prepare", "status": "completed"},
                {"phase": "agent_run", "status": "completed"},
            ],
        )
        mock_harness.run.return_value = TaskResult(
            task_id="task-1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=telemetry,
        )

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate", trace_enabled=True),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill", example=task)
        assert "raw_events" in side_info
        assert len(side_info["raw_events"]) == 2

    # --- VAL-GEPA-06: Custom evaluator adapter via config ---

    def test_custom_evaluator_called(self) -> None:
        """Custom evaluator function is called correctly."""
        calls: list[tuple[Any, Any]] = []

        def custom_eval(
            candidate: str, example: Any, **kwargs: Any
        ) -> tuple[float, dict[str, Any]]:
            calls.append((candidate, example))
            return 0.75, {"custom": True, "task_id": getattr(example, "id", "unknown")}

        adapter = load_custom_evaluator(custom_eval)
        task = _make_task()

        score, side_info = adapter(candidate="test prompt", example=task)
        assert score == pytest.approx(0.75)
        assert side_info["custom"] is True
        assert side_info["task_id"] == "task-1"
        assert len(calls) == 1

    def test_load_custom_evaluator_wraps_callable(self) -> None:
        """load_custom_evaluator wraps a plain callable into adapter protocol."""

        def my_eval(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict[str, Any]]:
            return 0.5, {"source": "custom"}

        adapter = load_custom_evaluator(my_eval)
        task = _make_task()

        score, side_info = adapter(candidate="test", example=task)
        assert score == pytest.approx(0.5)
        assert side_info["source"] == "custom"

    def test_create_evaluator_adapter_factory(self) -> None:
        """Factory function creates adapter from config."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        config = EvaluatorAdapterConfig(
            objective="pass-rate",
            trace_enabled=False,
        )
        adapter = create_evaluator_adapter(harness=mock_harness, config=config)
        assert isinstance(adapter, GepaEvaluatorAdapter)

    def test_create_evaluator_adapter_custom_objective_params(self) -> None:
        """Factory passes objective parameters correctly."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        config = EvaluatorAdapterConfig(
            objective="cost-adjusted",
            cost_per_token=0.002,
            trace_enabled=False,
        )
        adapter = create_evaluator_adapter(harness=mock_harness, config=config)
        assert isinstance(adapter, GepaEvaluatorAdapter)
        assert adapter._objective_function.cost_per_token == 0.002  # type: ignore[attr-defined]

    # --- Edge cases ---

    def test_harness_raises_exception(self) -> None:
        """Adapter handles harness exceptions gracefully."""
        mock_harness = MagicMock()
        mock_harness.run.side_effect = RuntimeError("Docker not available")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        score, side_info = adapter(candidate="skill", example=task)
        assert score == pytest.approx(0.0)
        assert side_info["status"] == "ERROR"
        # Error info is prefixed with "Harness exception: "
        assert any("Docker not available" in e for e in side_info["errors"])

    def test_side_info_has_tool_calls(self) -> None:
        """Side info contains tool call metrics when available."""
        mock_harness = MagicMock()
        mock_harness.run.return_value = _make_success_result("task-1")

        adapter = GepaEvaluatorAdapter(
            harness=mock_harness,
            config=EvaluatorAdapterConfig(objective="pass-rate"),
        )
        task = _make_task()

        _score, side_info = adapter(candidate="skill", example=task)
        assert "tool_calls" in side_info
        assert side_info["tool_calls"]["total_calls"] == 3
        assert side_info["tool_calls"]["successful_calls"] == 3


# ---------------------------------------------------------------------------
# Test EvaluatorAdapterConfig
# ---------------------------------------------------------------------------


class TestEvaluatorAdapterConfig:
    """Tests for the configuration model."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = EvaluatorAdapterConfig()
        assert config.objective == "pass-rate"
        assert config.cost_per_token == 0.0
        assert config.trace_enabled is False
        assert config.difficulty_weights == {"EASY": 1.0, "MEDIUM": 2.0, "HARD": 3.0}

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = EvaluatorAdapterConfig(
            objective="cost-adjusted",
            cost_per_token=0.001,
            trace_enabled=True,
        )
        assert config.objective == "cost-adjusted"
        assert config.cost_per_token == 0.001
        assert config.trace_enabled is True

    def test_custom_difficulty_weights(self) -> None:
        """Config accepts custom difficulty weights."""
        config = EvaluatorAdapterConfig(
            difficulty_weights={"EASY": 1.0, "MEDIUM": 3.0, "HARD": 10.0},
        )
        assert config.difficulty_weights["HARD"] == 10.0
