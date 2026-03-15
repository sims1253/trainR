"""Tests for the optimization runtime.

Validates:
- VAL-OPT-01: optimize command runs end-to-end
- VAL-OPT-02: Budget enforcement triggers graceful termination
- VAL-OPT-03: Timeout budget halts within tolerance
- VAL-OPT-04: No-improvement stop condition detects stagnation
- VAL-OPT-05: Checkpointing persists full optimization state
- VAL-OPT-06: Resume from checkpoint restores exact state
- VAL-OPT-07: Evolved artifact is exportable
- VAL-OPT-08: Optimization supports multiple target types
- VAL-GEPA-04: Holdout evaluation prevents overfitting
- VAL-GEPA-05: Pareto-front accumulation prevents regression
"""

from __future__ import annotations

import json
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from grist_mill.schemas import (
    Task,
    TaskResult,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "task-1",
    difficulty: str = "EASY",
    split: str = "train",
    timeout: int = 30,
) -> Task:
    """Create a simple test task."""
    return Task(
        id=task_id,
        prompt=f"Fix test {task_id}",
        language="python",
        test_command=f"pytest {task_id}.py",
        timeout=timeout,
        difficulty=difficulty,
    )


def _make_success_result(task_id: str = "task-1") -> TaskResult:
    """Create a successful TaskResult."""
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        score=1.0,
    )


def _make_failure_result(task_id: str = "task-1") -> TaskResult:
    """Create a failing TaskResult."""
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.FAILURE,
        score=0.0,
    )


def _make_train_tasks(n: int = 3) -> list[Task]:
    """Create a list of train tasks."""
    return [_make_task(f"train-{i}", split="train") for i in range(n)]


def _make_holdout_tasks(n: int = 2) -> list[Task]:
    """Create a list of holdout tasks."""
    return [_make_task(f"holdout-{i}", split="holdout") for i in range(n)]


def _simple_evaluator(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict[str, Any]]:
    """A simple mock evaluator that returns 0.5 always."""
    return 0.5, {"task_id": getattr(example, "id", "unknown"), "score": 0.5}


# ============================================================================
# Test Budget Conditions (VAL-OPT-02, VAL-OPT-03, VAL-OPT-04)
# ============================================================================


class TestBudgetConditions:
    """Tests for composable stop conditions: max_calls, timeout, no_improvement."""

    def test_max_calls_stop_condition(self) -> None:
        """max_calls budget stops the loop after the configured number of calls."""
        from grist_mill.optimization.runtime import BudgetConfig, StopCondition

        budget = BudgetConfig(max_calls=5)
        stop = StopCondition(budget=budget)

        # Calls 1-4 should not trigger stop
        for i in range(1, 5):
            assert not stop.should_stop(call_count=i, elapsed_s=0.0, best_score=0.0)
        # The 5th call should trigger stop (call_count >= max_calls)
        assert stop.should_stop(call_count=5, elapsed_s=0.0, best_score=0.0)
        reason = stop.termination_reason(call_count=5, elapsed_s=0.0, best_score=0.0)
        assert reason == "max_calls"

    def test_timeout_stop_condition(self) -> None:
        """timeout budget stops the loop when elapsed time exceeds threshold."""
        from grist_mill.optimization.runtime import BudgetConfig, StopCondition

        budget = BudgetConfig(timeout_s=1.0)
        stop = StopCondition(budget=budget)

        assert not stop.should_stop(call_count=1, elapsed_s=0.5, best_score=0.5)
        assert stop.should_stop(call_count=2, elapsed_s=1.5, best_score=0.5)
        reason = stop.termination_reason(call_count=2, elapsed_s=1.5, best_score=0.5)
        assert reason == "timeout"

    def test_no_improvement_stop_condition(self) -> None:
        """no_improvement patience stops after configured consecutive iterations."""
        from grist_mill.optimization.runtime import BudgetConfig, StopCondition

        budget = BudgetConfig(no_improvement_patience=3)
        stop = StopCondition(budget=budget)

        # Score improves each time — no stop (use update_and_check for no_improvement)
        assert not stop.update_and_check(call_count=1, elapsed_s=0.0, best_score=0.5)
        assert not stop.update_and_check(call_count=2, elapsed_s=0.0, best_score=0.6)
        assert not stop.update_and_check(call_count=3, elapsed_s=0.0, best_score=0.7)

        # Now score stays the same for 3 consecutive iterations
        assert not stop.update_and_check(call_count=4, elapsed_s=0.0, best_score=0.7)
        assert not stop.update_and_check(call_count=5, elapsed_s=0.0, best_score=0.7)
        # 3rd consecutive with same score — should stop
        assert stop.update_and_check(call_count=6, elapsed_s=0.0, best_score=0.7)
        reason = stop.termination_reason(call_count=6, elapsed_s=0.0, best_score=0.7)
        assert reason == "no_improvement"

    def test_budget_conditions_are_composable(self) -> None:
        """All three conditions can be active simultaneously."""
        from grist_mill.optimization.runtime import BudgetConfig, StopCondition

        budget = BudgetConfig(max_calls=100, timeout_s=60.0, no_improvement_patience=10)
        stop = StopCondition(budget=budget)

        # Timeout triggers first
        assert stop.should_stop(call_count=5, elapsed_s=61.0, best_score=0.5)
        reason = stop.termination_reason(call_count=5, elapsed_s=61.0, best_score=0.5)
        assert reason == "timeout"

    def test_termination_reports_which_condition_triggered(self) -> None:
        """Termination reason identifies the specific condition."""
        from grist_mill.optimization.runtime import BudgetConfig, StopCondition

        # max_calls triggers
        budget = BudgetConfig(max_calls=3)
        stop = StopCondition(budget=budget)
        assert stop.should_stop(call_count=4, elapsed_s=0.0, best_score=0.5)
        assert stop.termination_reason(call_count=4, elapsed_s=0.0, best_score=0.5) == "max_calls"

        # no_improvement triggers
        budget2 = BudgetConfig(no_improvement_patience=2)
        stop2 = StopCondition(budget=budget2)
        stop2.update_and_check(call_count=1, elapsed_s=0.0, best_score=0.5)
        stop2.update_and_check(call_count=2, elapsed_s=0.0, best_score=0.5)
        assert stop2.update_and_check(call_count=3, elapsed_s=0.0, best_score=0.5)
        assert (
            stop2.termination_reason(call_count=3, elapsed_s=0.0, best_score=0.5)
            == "no_improvement"
        )

    def test_budget_config_defaults(self) -> None:
        """BudgetConfig has sensible defaults (no limits by default)."""
        from grist_mill.optimization.runtime import BudgetConfig

        budget = BudgetConfig()
        assert budget.max_calls is None
        assert budget.timeout_s is None
        assert budget.no_improvement_patience is None

        # With no limits, never stops
        from grist_mill.optimization.runtime import StopCondition

        stop = StopCondition(budget=budget)
        assert not stop.should_stop(call_count=100000, elapsed_s=999999.0, best_score=0.0)
        assert (
            stop.termination_reason(call_count=100000, elapsed_s=999999.0, best_score=0.0) is None
        )


class TestTimeoutTolerance:
    """VAL-OPT-03: Timeout terminates within configurable tolerance (+/- 10%)."""

    def test_timeout_within_tolerance(self) -> None:
        """Timeout halts within 10% tolerance of configured value."""
        from grist_mill.optimization.runtime import BudgetConfig, StopCondition

        timeout_s = 2.0
        budget = BudgetConfig(timeout_s=timeout_s)
        stop = StopCondition(budget=budget)

        start = time.monotonic()
        # Simulate a loop checking every 50ms
        call_count = 0
        while True:
            call_count += 1
            elapsed = time.monotonic() - start
            if stop.should_stop(call_count=call_count, elapsed_s=elapsed, best_score=0.5):
                break
            if elapsed > timeout_s * 1.2:  # Safety: don't loop forever
                break
            time.sleep(0.05)

        elapsed = time.monotonic() - start
        # Should be within +/- 10% (allow extra 50ms for sleep granularity)
        assert elapsed < timeout_s * 1.1 + 0.1


# ============================================================================
# Test Checkpointing (VAL-OPT-05, VAL-OPT-06)
# ============================================================================


class TestCheckpointing:
    """Tests for checkpoint persistence and resume."""

    def test_checkpoint_persists_full_state(self) -> None:
        """Checkpoint contains candidate pool, Pareto front, iteration count, budget counters."""
        from grist_mill.optimization.runtime import (
            CheckpointState,
            OptimizationCheckpoint,
        )

        candidates = [
            {"content": "skill v1", "score": 0.5, "iteration": 0},
            {"content": "skill v2", "score": 0.7, "iteration": 1},
        ]
        pareto_front = [{"content": "skill v2", "score": 0.7, "iteration": 1}]

        state = CheckpointState(
            iteration=2,
            call_count=5,
            best_score=0.7,
            candidates=candidates,
            pareto_front=pareto_front,
            best_candidate={"content": "skill v2", "score": 0.7},
            trajectory=[
                {"iteration": 0, "score": 0.5, "candidate": "skill v1"},
                {"iteration": 1, "score": 0.7, "candidate": "skill v2"},
            ],
            target_type="skill",
            seed_content="initial skill",
            elapsed_s=12.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = OptimizationCheckpoint(path=path)
            cp.save(state)

            # Verify file exists and has all fields
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["iteration"] == 2
            assert loaded["call_count"] == 5
            assert loaded["best_score"] == 0.7
            assert len(loaded["candidates"]) == 2
            assert len(loaded["pareto_front"]) == 1
            assert loaded["target_type"] == "skill"
            assert loaded["elapsed_s"] == 12.5
            assert len(loaded["trajectory"]) == 2

    def test_checkpoint_valid_after_sigterm(self) -> None:
        """Checkpoint is valid JSON after writing."""
        from grist_mill.optimization.runtime import (
            CheckpointState,
            OptimizationCheckpoint,
        )

        state = CheckpointState(
            iteration=3,
            call_count=10,
            best_score=0.8,
            candidates=[{"content": "best", "score": 0.8, "iteration": 2}],
            pareto_front=[{"content": "best", "score": 0.8, "iteration": 2}],
            best_candidate={"content": "best", "score": 0.8},
            trajectory=[{"iteration": 2, "score": 0.8, "candidate": "best"}],
            target_type="system_prompt",
            seed_content="initial prompt",
            elapsed_s=45.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = OptimizationCheckpoint(path=path)
            cp.save(state)

            # Parse as valid JSON
            data = json.loads(path.read_text())
            assert isinstance(data, dict)
            # All required fields present
            required_fields = {
                "iteration",
                "call_count",
                "best_score",
                "candidates",
                "pareto_front",
                "best_candidate",
                "trajectory",
                "target_type",
                "seed_content",
                "elapsed_s",
            }
            assert required_fields.issubset(set(data.keys()))

    def test_resume_from_checkpoint_restores_state(self) -> None:
        """--resume reconstructs state so iteration numbering and budget counters continue."""
        from grist_mill.optimization.runtime import (
            CheckpointState,
            OptimizationCheckpoint,
        )

        original_state = CheckpointState(
            iteration=5,
            call_count=12,
            best_score=0.85,
            candidates=[
                {"content": "v1", "score": 0.5, "iteration": 0},
                {"content": "v2", "score": 0.7, "iteration": 2},
                {"content": "v3", "score": 0.85, "iteration": 4},
            ],
            pareto_front=[{"content": "v3", "score": 0.85, "iteration": 4}],
            best_candidate={"content": "v3", "score": 0.85},
            trajectory=[
                {"iteration": 0, "score": 0.5, "candidate": "v1"},
                {"iteration": 2, "score": 0.7, "candidate": "v2"},
                {"iteration": 4, "score": 0.85, "candidate": "v3"},
            ],
            target_type="tool_policy",
            seed_content="initial policy",
            elapsed_s=67.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = OptimizationCheckpoint(path=path)
            cp.save(state=original_state)

            # Resume
            resumed = cp.load()
            assert resumed is not None
            assert resumed.iteration == 5
            assert resumed.call_count == 12
            assert resumed.best_score == 0.85
            assert len(resumed.candidates) == 3
            assert len(resumed.pareto_front) == 1
            assert resumed.target_type == "tool_policy"
            assert resumed.elapsed_s == 67.3

    def test_resume_no_duplicate_iterations(self) -> None:
        """Resumed run starts from iteration N+1, not from 0."""
        from grist_mill.optimization.runtime import (
            CheckpointState,
            OptimizationCheckpoint,
        )

        state = CheckpointState(
            iteration=3,
            call_count=7,
            best_score=0.6,
            candidates=[{"content": "v1", "score": 0.6, "iteration": 2}],
            pareto_front=[{"content": "v1", "score": 0.6, "iteration": 2}],
            best_candidate={"content": "v1", "score": 0.6},
            trajectory=[{"iteration": 2, "score": 0.6, "candidate": "v1"}],
            target_type="skill",
            seed_content="seed",
            elapsed_s=10.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = OptimizationCheckpoint(path=path)
            cp.save(state=state)

            resumed = cp.load()
            # The next iteration should be 3 (resumed.iteration), not 0
            assert resumed is not None
            assert resumed.iteration == 3

    def test_checkpoint_load_nonexistent_returns_none(self) -> None:
        """Loading a nonexistent checkpoint returns None."""
        from grist_mill.optimization.runtime import OptimizationCheckpoint

        cp = OptimizationCheckpoint(path=Path("/nonexistent/checkpoint.json"))
        assert cp.load() is None


# ============================================================================
# Test Pareto Front (VAL-GEPA-05)
# ============================================================================


class TestParetoFront:
    """Tests for Pareto front management."""

    def test_pareto_front_accepts_improvement(self) -> None:
        """Pareto front accepts candidates that improve the score."""
        from grist_mill.optimization.runtime import ParetoFront

        front = ParetoFront()
        front.update("candidate-1", {"task-a": 0.5})
        front.update("candidate-2", {"task-a": 0.8})

        assert len(front.front) == 1
        assert "candidate-2" in front.front

    def test_pareto_front_rejects_regression(self) -> None:
        """Pareto front prevents regression on previously-solved tasks."""
        from grist_mill.optimization.runtime import ParetoFront

        front = ParetoFront()
        # First candidate solves task-a with score 0.8
        front.update("candidate-1", {"task-a": 0.8, "task-b": 0.5})

        # Second candidate solves task-a worse but task-b better — multi-objective
        # The front should keep both since neither dominates the other
        front.update("candidate-2", {"task-a": 0.5, "task-b": 0.9})
        assert len(front.front) == 2

        # Third candidate is dominated by both — should be rejected
        front.update("candidate-3", {"task-a": 0.4, "task-b": 0.4})
        assert len(front.front) == 2

    def test_pareto_front_prevents_regression_single_objective(self) -> None:
        """With single objective, Pareto front only keeps the best."""
        from grist_mill.optimization.runtime import ParetoFront

        front = ParetoFront()
        front.update("c1", {"score": 0.7})
        front.update("c2", {"score": 0.9})
        front.update("c3", {"score": 0.8})

        # Only c2 should remain (highest score)
        assert len(front.front) == 1
        assert "c2" in front.front

    def test_pareto_front_get_best(self) -> None:
        """get_best returns the candidate with the highest average score."""
        from grist_mill.optimization.runtime import ParetoFront

        front = ParetoFront()
        front.update("c1", {"t1": 0.5, "t2": 0.5})  # avg 0.5
        front.update("c2", {"t1": 0.9, "t2": 0.3})  # avg 0.6 — nondominated
        front.update("c3", {"t1": 0.3, "t2": 0.9})  # avg 0.6 — nondominated

        best = front.get_best()
        assert best is not None
        assert best in ("c2", "c3")

    def test_pareto_front_empty(self) -> None:
        """Empty Pareto front returns None for get_best."""
        from grist_mill.optimization.runtime import ParetoFront

        front = ParetoFront()
        assert front.get_best() is None


# ============================================================================
# Test Target Types (VAL-OPT-08)
# ============================================================================


class TestTargetTypes:
    """Tests for multiple target types: skill, system_prompt, tool_policy."""

    def test_skill_target_serialization(self) -> None:
        """Skill target serializes and deserializes correctly."""
        from grist_mill.optimization.runtime import TargetType, serialize_target

        target = TargetType(
            type="skill",
            content="You are a helpful assistant that writes tests.",
            metadata={"skill_name": "test-writer"},
        )
        data = serialize_target(target)
        assert data["type"] == "skill"
        assert "content" in data
        assert data["metadata"]["skill_name"] == "test-writer"

    def test_system_prompt_target_serialization(self) -> None:
        """System prompt target serializes and deserializes correctly."""
        from grist_mill.optimization.runtime import TargetType, serialize_target

        target = TargetType(
            type="system_prompt",
            content="Write clean, well-tested code.",
            metadata={"model": "gpt-4"},
        )
        data = serialize_target(target)
        assert data["type"] == "system_prompt"
        assert data["content"] == "Write clean, well-tested code."

    def test_tool_policy_target_serialization(self) -> None:
        """Tool policy target serializes and deserializes correctly."""
        from grist_mill.optimization.runtime import TargetType, serialize_target

        target = TargetType(
            type="tool_policy",
            content='{"allow": ["bash", "read_file"], "deny": ["rm"]}',
            metadata={"policy_version": "1.0"},
        )
        data = serialize_target(target)
        assert data["type"] == "tool_policy"
        assert "allow" in data["content"]

    def test_target_type_validation(self) -> None:
        """Target type must be one of the known types."""
        from grist_mill.optimization.runtime import TargetType

        # Valid types
        for t in ["skill", "system_prompt", "tool_policy"]:
            target = TargetType(type=t, content="test")
            assert target.type == t

        # Invalid type should raise
        with pytest.raises(ValueError, match="Unknown target type"):
            TargetType(type="invalid_type", content="test")

    def test_target_config_from_dict(self) -> None:
        """TargetType can be created from a config dict."""
        from grist_mill.optimization.runtime import TargetConfig

        config = TargetConfig(
            type="skill",
            content="Write tests for R packages.",
            metadata={"skill_name": "r-tester"},
        )
        assert config.type == "skill"
        assert config.content == "Write tests for R packages."


# ============================================================================
# Test Optimization Config (OptimizationConfig)
# ============================================================================


class TestOptimizationConfig:
    """Tests for the optimization configuration model."""

    def test_config_from_yaml(self) -> None:
        """OptimizationConfig loads from YAML file."""
        from grist_mill.optimization.runtime import OptimizationConfig

        config_dict = {
            "target": {
                "type": "skill",
                "content": "Write good tests.",
                "metadata": {"skill_name": "test-writer"},
            },
            "budget": {
                "max_calls": 10,
                "timeout_s": 120.0,
                "no_improvement_patience": 5,
            },
            "evaluator": {
                "objective": "pass-rate",
                "trace_enabled": False,
            },
            "output_dir": "/tmp/opt-results",
            "checkpoint_path": "/tmp/opt-results/checkpoint.json",
            "train_split": "train",
            "holdout_split": "holdout",
        }
        config = OptimizationConfig(**config_dict)
        assert config.target.type == "skill"
        assert config.budget.max_calls == 10
        assert config.budget.timeout_s == 120.0
        assert config.budget.no_improvement_patience == 5
        assert config.evaluator.objective == "pass-rate"

    def test_config_defaults(self) -> None:
        """OptimizationConfig has sensible defaults."""
        from grist_mill.optimization.runtime import OptimizationConfig, TargetConfig

        target = TargetConfig(type="skill", content="test")
        config = OptimizationConfig(target=target)
        assert config.budget.max_calls is None
        assert config.budget.timeout_s is None
        assert config.evaluator.objective == "pass-rate"


# ============================================================================
# Test Best Candidate Export (VAL-OPT-07)
# ============================================================================


class TestBestCandidateExport:
    """Tests for exporting the best candidate."""

    def test_export_best_candidate_as_json(self) -> None:
        """Best candidate is exportable as a deployable JSON artifact."""
        from grist_mill.optimization.runtime import export_best_candidate

        best = {
            "content": "You are a test-writing expert.",
            "score": 0.9,
            "iteration": 5,
            "target_type": "skill",
            "metadata": {"skill_name": "test-writer"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_candidate.json"
            export_best_candidate(best, path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["content"] == "You are a test-writing expert."
            assert data["score"] == 0.9
            assert data["target_type"] == "skill"
            assert "exported_at" in data

    def test_export_includes_schema_version(self) -> None:
        """Export includes schema_version for forward compatibility."""
        from grist_mill.optimization.runtime import export_best_candidate

        best = {"content": "test", "score": 0.5, "iteration": 0, "target_type": "skill"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_candidate.json"
            export_best_candidate(best, path)
            data = json.loads(path.read_text())
            assert "schema_version" in data


# ============================================================================
# Test Holdout Evaluation (VAL-GEPA-04)
# ============================================================================


class TestHoldoutEvaluation:
    """Tests for holdout evaluation preventing overfitting."""

    def test_holdout_tasks_are_disjoint(self) -> None:
        """Holdout task set is disjoint from training tasks."""
        from grist_mill.optimization.runtime import split_tasks

        tasks = _make_train_tasks(5) + _make_holdout_tasks(2)
        train, holdout = split_tasks(tasks, train_split="train", holdout_split="holdout")

        train_ids = {t.id for t in train}
        holdout_ids = {t.id for t in holdout}
        assert len(train_ids & holdout_ids) == 0
        assert len(train) == 5
        assert len(holdout) == 2

    def test_holdout_metrics_are_present(self) -> None:
        """Holdout evaluation produces metrics separate from training."""
        from grist_mill.optimization.runtime import (
            OptimizationResult,
        )

        result = OptimizationResult(
            best_candidate={"content": "best", "score": 0.8},
            train_score=0.8,
            holdout_score=0.7,
            iteration_count=10,
            call_count=10,
            termination_reason="max_calls",
            trajectory=[],
            pareto_front=[],
        )

        assert result.holdout_score is not None
        assert result.holdout_score < result.train_score  # typical overfitting scenario

    def test_no_holdout_when_no_holdout_tasks(self) -> None:
        """When no holdout tasks, holdout_score is None."""
        from grist_mill.optimization.runtime import (
            OptimizationResult,
        )

        result = OptimizationResult(
            best_candidate={"content": "best", "score": 0.8},
            train_score=0.8,
            holdout_score=None,
            iteration_count=5,
            call_count=5,
            termination_reason="max_calls",
            trajectory=[],
            pareto_front=[],
        )
        assert result.holdout_score is None


# ============================================================================
# Test Optimization Runner End-to-End (VAL-OPT-01)
# ============================================================================


class TestOptimizationRunner:
    """End-to-end tests for the optimization runner."""

    def test_run_with_max_calls_budget(self) -> None:
        """Optimization runs and stops when max_calls budget is exhausted."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        call_count = 0

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            nonlocal call_count
            call_count += 1
            return 0.5, {"task_id": example.id, "score": 0.5}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial skill"),
            budget=BudgetConfig(max_calls=3),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
            )
            result = runner.run()

            assert result.termination_reason == "max_calls"
            assert call_count <= 3  # budget enforced
            assert result.train_score >= 0.0
            assert result.best_candidate is not None

    def test_run_produces_trajectory(self) -> None:
        """Optimization run produces a trajectory log."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
            )
            result = runner.run()

            assert len(result.trajectory) > 0
            # Each trajectory entry has required fields
            for entry in result.trajectory:
                assert "iteration" in entry
                assert "score" in entry

    def test_run_saves_checkpoint(self) -> None:
        """Optimization run saves a checkpoint file."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
            )
            runner.run()

            assert checkpoint_path.exists()
            data = json.loads(checkpoint_path.read_text())
            assert "iteration" in data
            assert "candidates" in data

    def test_run_saves_best_candidate(self) -> None:
        """Optimization run saves best_candidate.json."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.8, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
            )
            runner.run()

            best_path = Path(tmpdir) / "best_candidate.json"
            assert best_path.exists()
            data = json.loads(best_path.read_text())
            assert data["score"] > 0.0

    def test_run_saves_trajectory_jsonl(self) -> None:
        """Optimization run saves trajectory.jsonl."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.6, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
            )
            runner.run()

            traj_path = Path(tmpdir) / "trajectory.jsonl"
            assert traj_path.exists()
            lines = traj_path.read_text().strip().split("\n")
            assert len(lines) > 0
            for line in lines:
                entry = json.loads(line)
                assert "iteration" in entry
                assert "score" in entry

    def test_resume_from_checkpoint(self) -> None:
        """Resuming from checkpoint continues without duplicating iterations."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        # First run
        checkpoint_path = Path(tempfile.mktemp(suffix=".json"))

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # First run with max_calls=2
            config_first = OptimizationConfig(
                target=TargetConfig(type="skill", content="initial"),
                budget=BudgetConfig(max_calls=2),
            )
            runner1 = OptimizationRunner(
                config=config_first,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
            )
            result1 = runner1.run()
            first_iteration = result1.iteration_count

            assert checkpoint_path.exists()

            # Resume with max_calls=5 (total budget)
            config_resume = OptimizationConfig(
                target=TargetConfig(type="skill", content="initial"),
                budget=BudgetConfig(max_calls=5),
            )
            runner2 = OptimizationRunner(
                config=config_resume,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
                resume=True,
            )
            result2 = runner2.run()

            # Resumed run should have more iterations than first run
            assert result2.iteration_count > first_iteration

    def test_no_improvement_stops_loop(self) -> None:
        """No-improvement patience stops the optimization loop."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)

        # Always return the same score — no improvement
        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(no_improvement_patience=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
            )
            result = runner.run()

            assert result.termination_reason == "no_improvement"

    def test_holdout_evaluation_prevents_overfitting(self) -> None:
        """Final candidates are evaluated on holdout set."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)
        holdout_tasks = _make_holdout_tasks(2)

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.8, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                holdout_tasks=holdout_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
            )
            result = runner.run()

            assert result.holdout_score is not None

    def test_multiple_target_types(self) -> None:
        """Optimization supports skill, system_prompt, and tool_policy targets."""
        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(1)

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.7, {"task_id": example.id}

        for target_type in ["skill", "system_prompt", "tool_policy"]:
            config = OptimizationConfig(
                target=TargetConfig(type=target_type, content=f"test {target_type}"),
                budget=BudgetConfig(max_calls=1),
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                runner = OptimizationRunner(
                    config=config,
                    train_tasks=train_tasks,
                    evaluate_fn=mock_evaluate,
                    propose_fn=MockProposer(),
                    output_dir=Path(tmpdir),
                )
                result = runner.run()

                assert result.termination_reason == "max_calls"
                assert result.best_candidate["target_type"] == target_type


# ============================================================================
# Test SIGTERM Handling
# ============================================================================


class TestGracefulTermination:
    """Tests for graceful termination on SIGTERM."""

    def test_sigterm_writes_checkpoint(self) -> None:
        """SIGTERM triggers checkpoint write."""
        import os

        from grist_mill.optimization.runtime import (
            BudgetConfig,
            MockProposer,
            OptimizationConfig,
            OptimizationRunner,
            TargetConfig,
        )

        train_tasks = _make_train_tasks(2)
        checkpoint_written = False

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            # Send SIGTERM on first evaluation
            nonlocal checkpoint_written
            if not checkpoint_written:
                checkpoint_written = True
                os.kill(os.getpid(), signal.SIGTERM)
            return 0.5, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=10),  # High limit; SIGTERM should trigger first
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                propose_fn=MockProposer(),
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
            )

            # Catch SIGTERM at the runner level
            runner.run()

            # Should have terminated and written checkpoint
            assert checkpoint_path.exists()
            data = json.loads(checkpoint_path.read_text())
            assert "iteration" in data
            assert "candidates" in data
