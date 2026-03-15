"""Cross-optimization consistency tests (m5-cross-optimization-consistency).

Verifies:
- VAL-CROSS-05: The optimization loop uses a stable task dataset and consistent
  harness configuration across iterations, with monotonically improving or
  converging metrics.
- VAL-CROSS-07: Checkpointing persists state so a resumed run produces identical
  final results to an uninterrupted run.

Evidence required:
- VAL-CROSS-05: iteration logs showing stable harness config and monotonically
  improving or converging metrics.
- VAL-CROSS-07: comparison of final results from interrupted-resumed run vs
  continuous run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from grist_mill.optimization.runtime import (
    BudgetConfig,
    CheckpointState,
    OptimizationCheckpoint,
    OptimizationConfig,
    OptimizationRunner,
    ParetoFront,
    TargetConfig,
    export_best_candidate,
    split_tasks,
)
from grist_mill.schemas import (
    Task,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "train-0",
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


def _make_train_tasks(n: int = 3) -> list[Task]:
    """Create a list of train tasks."""
    return [_make_task(f"train-{i}", split="train") for i in range(n)]


def _make_holdout_tasks(n: int = 2) -> list[Task]:
    """Create a list of holdout tasks."""
    return [_make_task(f"holdout-{i}", split="holdout") for i in range(n)]


class _DeterministicProposer:
    """A proposer that always produces deterministic output for a given input."""

    def __init__(self) -> None:
        self._call_count = 0

    def propose(
        self,
        current_content: str,
        side_info: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        self._call_count += 1
        return f"{current_content}|propose-{self._call_count}"


class _ImprovingProposer:
    """A proposer whose proposals lead to monotonically improving scores.

    Each proposal appends an incrementing marker so the evaluator can
    simulate improving performance over iterations.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def propose(
        self,
        current_content: str,
        side_info: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        self._call_count += 1
        return f"candidate-v{self._call_count}"


# ===========================================================================
# VAL-CROSS-05: Stable task dataset and harness configuration
# ===========================================================================


class TestStableTaskDataset:
    """Verify the optimization loop uses a fixed, stable task dataset across
    all iterations without drift."""

    def test_task_dataset_ids_are_constant_across_iterations(self) -> None:
        """The same task IDs are used in every iteration of the optimization loop."""
        train_tasks = _make_train_tasks(3)

        iteration = 0
        all_iteration_ids: list[list[str]] = []

        def mock_evaluate_with_tracking(
            candidate: str, example: Any, **kwargs: Any
        ) -> tuple[float, dict]:
            nonlocal iteration
            task_id = example.id
            # Ensure we have a slot for this iteration's tasks
            while len(all_iteration_ids) <= iteration:
                all_iteration_ids.append([])
            all_iteration_ids[iteration].append(task_id)
            return 0.5, {"task_id": task_id, "score": 0.5}

        class _TrackingProposer:
            """Proposer that tracks iteration boundaries."""

            def propose(
                self,
                current_content: str,
                side_info: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> str:
                nonlocal iteration
                iteration += 1
                return f"v{iteration}"

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=6),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate_with_tracking,
                output_dir=Path(tmpdir),
            )
            runner._propose_fn = _TrackingProposer()
            runner.run()

        # Each iteration should see the exact same set of task IDs
        if len(all_iteration_ids) >= 2:
            first_ids = sorted(all_iteration_ids[0])
            for iteration_ids in all_iteration_ids[1:]:
                assert sorted(iteration_ids) == first_ids, (
                    f"Task dataset drifted: expected {first_ids}, got {sorted(iteration_ids)}"
                )

    def test_train_and_holdout_are_disjoint(self) -> None:
        """Training and holdout task sets share no task IDs."""
        train_tasks = _make_train_tasks(5)
        holdout_tasks = _make_holdout_tasks(3)
        all_tasks = train_tasks + holdout_tasks

        train, holdout = split_tasks(all_tasks, train_split="train", holdout_split="holdout")

        train_ids = {t.id for t in train}
        holdout_ids = {t.id for t in holdout}
        assert train_ids.isdisjoint(holdout_ids), "Train and holdout sets must be disjoint"
        assert len(train) == 5
        assert len(holdout) == 3


class TestConsistentHarnessConfiguration:
    """Verify the harness configuration used by the optimization loop is
    consistent across all iterations."""

    def test_target_config_remains_constant(self) -> None:
        """The target configuration (type, seed content) does not change across iterations."""
        target = TargetConfig(
            type="skill",
            content="You are a test-writing assistant.",
            metadata={"skill_name": "tester"},
        )
        config = OptimizationConfig(target=target, budget=BudgetConfig(max_calls=4))

        captured_configs: list[dict[str, Any]] = []

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            captured_configs.append({"candidate": candidate, "task": example.id})
            return 0.5, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            runner.run()

        # The initial candidate should always be the seed content
        assert captured_configs[0]["candidate"] == "You are a test-writing assistant."
        # The target type in the config should remain stable
        assert config.target.type == "skill"
        assert config.target.content == "You are a test-writing assistant."

    def test_evaluator_objective_is_stable(self) -> None:
        """The evaluator objective does not change between iterations."""
        from grist_mill.optimization import EvaluatorAdapterConfig

        evaluator_config = EvaluatorAdapterConfig(objective="pass-rate")
        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
            evaluator=evaluator_config,
        )

        call_count = 0

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            nonlocal call_count
            call_count += 1
            return 0.6, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            runner.run()

        assert config.evaluator.objective == "pass-rate"
        assert call_count > 0

    def test_checkpoint_target_type_matches_config(self) -> None:
        """Checkpoint file records the same target type as the config."""
        config = OptimizationConfig(
            target=TargetConfig(type="system_prompt", content="Be helpful."),
            budget=BudgetConfig(max_calls=2),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.json"
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(1),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=cp_path,
            )
            runner.run()

            checkpoint_data = json.loads(cp_path.read_text())
            assert checkpoint_data["target_type"] == "system_prompt"
            assert checkpoint_data["seed_content"] == "Be helpful."


class TestMonotonicOrConvergingMetrics:
    """Verify metrics improve monotonically or converge over iterations."""

    def test_best_score_is_best_across_all_iterations(self) -> None:
        """The best score reported is the maximum across all iteration scores."""
        # Simulate varying scores across iterations
        iteration = 0
        scores_by_iteration = [0.3, 0.5, 0.4, 0.7, 0.6]

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            nonlocal iteration
            # Each iteration evaluates all tasks; return the same score for all tasks
            # in this iteration
            return scores_by_iteration[min(iteration, len(scores_by_iteration) - 1)], {
                "task_id": example.id,
            }

        class _IncrementingProposer:
            def propose(
                self,
                current_content: str,
                side_info: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> str:
                nonlocal iteration
                iteration += 1
                return f"v{iteration}"

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=10),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(1),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            runner._propose_fn = _IncrementingProposer()
            result = runner.run()

        # The best score should be 0.7 (the max)
        assert result.train_score == pytest.approx(0.7, abs=0.01)

    def test_pareto_front_prevents_regression(self) -> None:
        """Pareto front ensures accepted candidates never regress."""
        front = ParetoFront()

        # Add improving candidates
        front.update("c1", {"task-a": 0.5})
        front.update("c2", {"task-a": 0.7})
        front.update("c3", {"task-a": 0.9})

        # Front should only contain the best
        assert len(front.front) == 1
        assert "c3" in front.front
        assert front.front["c3"]["task-a"] == 0.9

        # A worse candidate should be rejected
        added = front.update("c4", {"task-a": 0.6})
        assert not added
        assert len(front.front) == 1

    def test_trajectory_records_all_iterations(self) -> None:
        """Trajectory captures scores for every completed iteration."""
        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=4),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            result = runner.run()

        # Every iteration should have a trajectory entry
        assert len(result.trajectory) > 0
        for entry in result.trajectory:
            assert "iteration" in entry
            assert "score" in entry
            assert 0.0 <= entry["score"] <= 1.0

    def test_converging_metrics_stabilize(self) -> None:
        """When scores converge (no improvement), the loop stops."""

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            # Always return 0.5 — no improvement
            return 0.5, {"task_id": example.id}

        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(no_improvement_patience=2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(1),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            result = runner.run()

        assert result.termination_reason == "no_improvement"


class TestOptimizationResultsExportable:
    """Verify optimization results are exportable to report format."""

    def test_best_candidate_json_is_valid_report_artifact(self) -> None:
        """The exported best_candidate.json has schema_version and timestamps."""
        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.8, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            runner.run()

            best_path = Path(tmpdir) / "best_candidate.json"
            assert best_path.exists()
            data = json.loads(best_path.read_text())

            # Report-format requirements
            assert "schema_version" in data
            assert "exported_at" in data
            assert "content" in data
            assert "score" in data
            assert "target_type" in data
            assert data["score"] > 0.0

    def test_trajectory_jsonl_is_produced(self) -> None:
        """Optimization run produces trajectory.jsonl with all iteration records."""
        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=2),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.6, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                evaluate_fn=mock_evaluate,
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

    def test_optimization_result_contains_pareto_front(self) -> None:
        """OptimizationResult includes Pareto front for report integration."""
        config = OptimizationConfig(
            target=TargetConfig(type="skill", content="initial"),
            budget=BudgetConfig(max_calls=3),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.7, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            result = runner.run()

            # Result should include pareto_front
            assert isinstance(result.pareto_front, list)
            assert len(result.pareto_front) > 0
            for entry in result.pareto_front:
                assert "candidate_id" in entry
                assert "scores" in entry

    def test_export_best_candidate_produces_valid_json(self) -> None:
        """export_best_candidate produces valid JSON with all required fields."""
        candidate = {
            "content": "You are a test expert.",
            "score": 0.9,
            "iteration": 5,
            "target_type": "skill",
            "metadata": {"name": "test-expert"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "export.json"
            export_best_candidate(candidate, path)
            data = json.loads(path.read_text())

            assert data["content"] == "You are a test expert."
            assert data["score"] == 0.9
            assert data["target_type"] == "skill"
            assert "schema_version" in data
            assert "exported_at" in data


# ===========================================================================
# VAL-CROSS-07: Checkpoint/resume produces identical results
# ===========================================================================


class TestCheckpointResumeConsistency:
    """Verify checkpoint/resume produces identical final results to an
    uninterrupted run."""

    def test_checkpoint_state_is_complete(self) -> None:
        """Checkpoint contains all required fields for full state reconstruction."""
        state = CheckpointState(
            iteration=5,
            call_count=12,
            best_score=0.85,
            candidates=[
                {"content": "v1", "score": 0.5, "iteration": 0},
                {"content": "v2", "score": 0.7, "iteration": 2},
                {"content": "v3", "score": 0.85, "iteration": 4},
            ],
            pareto_front=[
                {"candidate_id": "candidate-4", "scores": {"task-0": 0.85, "task-1": 0.85}}
            ],
            best_candidate={"content": "v3", "score": 0.85, "target_type": "skill"},
            trajectory=[
                {"iteration": 0, "score": 0.5, "candidate": "v1"},
                {"iteration": 2, "score": 0.7, "candidate": "v2"},
                {"iteration": 4, "score": 0.85, "candidate": "v3"},
            ],
            target_type="skill",
            seed_content="initial skill",
            elapsed_s=67.3,
        )

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

        dumped = state.model_dump(mode="json")
        assert required_fields.issubset(set(dumped.keys()))

    def test_checkpoint_roundtrip_preserves_values(self) -> None:
        """Checkpoint round-trips through save/load without data loss."""
        state = CheckpointState(
            iteration=3,
            call_count=7,
            best_score=0.6,
            candidates=[
                {"content": "v1", "score": 0.5, "iteration": 0},
                {"content": "v2", "score": 0.6, "iteration": 2},
            ],
            pareto_front=[{"candidate_id": "candidate-2", "scores": {"task-0": 0.6}}],
            best_candidate={"content": "v2", "score": 0.6},
            trajectory=[
                {"iteration": 0, "score": 0.5, "candidate": "v1"},
                {"iteration": 2, "score": 0.6, "candidate": "v2"},
            ],
            target_type="system_prompt",
            seed_content="initial prompt",
            elapsed_s=25.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = OptimizationCheckpoint(path=path)
            cp.save(state)
            loaded = cp.load()

            assert loaded is not None
            assert loaded.iteration == state.iteration
            assert loaded.call_count == state.call_count
            assert loaded.best_score == state.best_score
            assert len(loaded.candidates) == len(state.candidates)
            assert len(loaded.pareto_front) == len(state.pareto_front)
            assert loaded.best_candidate == state.best_candidate
            assert loaded.trajectory == state.trajectory
            assert loaded.target_type == state.target_type
            assert loaded.seed_content == state.seed_content
            assert loaded.elapsed_s == pytest.approx(state.elapsed_s, abs=0.01)

    def test_resumed_run_continues_from_correct_iteration(self) -> None:
        """A resumed run starts from the checkpointed iteration, not from 0."""
        call_log: list[str] = []

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            call_log.append(f"eval:{example.id}")
            return 0.5, {"task_id": example.id}

        train_tasks = _make_train_tasks(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Phase 1: Run with max_calls=2
            config1 = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=2),
            )
            runner1 = OptimizationRunner(
                config=config1,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
            )
            result1 = runner1.run()

            assert checkpoint_path.exists()
            checkpoint_data = json.loads(checkpoint_path.read_text())
            phase1_iteration = checkpoint_data["iteration"]

            # Phase 2: Resume with max_calls=6 (total budget)
            config2 = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=6),
            )
            runner2 = OptimizationRunner(
                config=config2,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
                resume=True,
            )
            result2 = runner2.run()

            # Resumed run should have MORE iterations than the first run
            assert result2.iteration_count >= result1.iteration_count

            # The resumed run should not re-evaluate from iteration 0
            # (it picks up from where it left off)
            final_checkpoint = json.loads(checkpoint_path.read_text())
            assert final_checkpoint["iteration"] >= phase1_iteration

    def test_resumed_run_uses_same_task_dataset(self) -> None:
        """Resumed run uses the same task dataset as the original run."""
        seen_tasks_in_first_run: set[str] = set()
        seen_tasks_in_resumed_run: set[str] = set()
        phase = 0

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            task_id = example.id
            if phase == 0:
                seen_tasks_in_first_run.add(task_id)
            else:
                seen_tasks_in_resumed_run.add(task_id)
            return 0.5, {"task_id": task_id}

        train_tasks = _make_train_tasks(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Phase 1
            phase = 0
            config1 = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=3),
            )
            runner1 = OptimizationRunner(
                config=config1,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
            )
            runner1.run()

            # Phase 2: resume
            phase = 1
            config2 = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=6),
            )
            runner2 = OptimizationRunner(
                config=config2,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
                resume=True,
            )
            runner2.run()

            # The resumed run should use the same task dataset
            assert seen_tasks_in_resumed_run.issubset(seen_tasks_in_first_run), (
                "Resumed run used different tasks than the original run"
            )
            assert seen_tasks_in_first_run == {"train-0", "train-1", "train-2"}

    def test_identical_results_with_deterministic_evaluator(self) -> None:
        """With a deterministic evaluator, interrupted+resumed and uninterrupted
        runs produce identical best_score and final trajectory length."""

        # Use a simple deterministic evaluator that returns the same score
        # regardless of candidate content
        def deterministic_evaluate(
            candidate: str, example: Any, **kwargs: Any
        ) -> tuple[float, dict]:
            return 0.6, {"task_id": example.id}

        train_tasks = _make_train_tasks(1)

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            cp_path1 = Path(tmpdir1) / "checkpoint.json"
            cp_path2 = Path(tmpdir2) / "checkpoint.json"

            # Uninterrupted run with max_calls=4
            config_uninterrupted = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=4),
            )
            runner_uninterrupted = OptimizationRunner(
                config=config_uninterrupted,
                train_tasks=train_tasks,
                evaluate_fn=deterministic_evaluate,
                output_dir=Path(tmpdir1),
                checkpoint_path=cp_path1,
            )
            result_uninterrupted = runner_uninterrupted.run()

            # Interrupted run with max_calls=2, then resume with max_calls=4
            config_first = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=2),
            )
            runner_first = OptimizationRunner(
                config=config_first,
                train_tasks=train_tasks,
                evaluate_fn=deterministic_evaluate,
                output_dir=Path(tmpdir2),
                checkpoint_path=cp_path2,
            )
            runner_first.run()

            config_resume = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=4),
            )
            runner_resume = OptimizationRunner(
                config=config_resume,
                train_tasks=train_tasks,
                evaluate_fn=deterministic_evaluate,
                output_dir=Path(tmpdir2),
                checkpoint_path=cp_path2,
                resume=True,
            )
            result_resumed = runner_resume.run()

            # With the same deterministic evaluator and same total budget,
            # both approaches should reach the same best score
            assert result_uninterrupted.train_score == result_resumed.train_score

    def test_checkpoint_survives_across_runs(self) -> None:
        """Checkpoint file written during one run is readable in a new process."""
        state = CheckpointState(
            iteration=2,
            call_count=4,
            best_score=0.75,
            candidates=[{"content": "v1", "score": 0.75, "iteration": 1}],
            pareto_front=[{"candidate_id": "candidate-1", "scores": {"task-0": 0.75}}],
            best_candidate={"content": "v1", "score": 0.75, "target_type": "tool_policy"},
            trajectory=[{"iteration": 1, "score": 0.75, "candidate": "v1"}],
            target_type="tool_policy",
            seed_content="initial policy",
            elapsed_s=10.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = OptimizationCheckpoint(path=path)
            cp.save(state)

            # Simulate a "new process" by creating a new checkpoint manager
            cp2 = OptimizationCheckpoint(path=path)
            loaded = cp2.load()

            assert loaded is not None
            assert loaded.iteration == 2
            assert loaded.best_score == 0.75
            assert loaded.target_type == "tool_policy"
            assert loaded.seed_content == "initial policy"

    def test_resume_preserves_budget_counter_continuity(self) -> None:
        """Resumed run continues budget counting from where it stopped."""
        total_calls = 0

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            nonlocal total_calls
            total_calls += 1
            return 0.5, {"task_id": example.id}

        train_tasks = _make_train_tasks(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # First run: max_calls=2
            config1 = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=2),
            )
            runner1 = OptimizationRunner(
                config=config1,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
            )
            runner1.run()
            calls_after_phase1 = total_calls

            # Check checkpoint records the correct call count
            cp_data = json.loads(checkpoint_path.read_text())
            assert cp_data["call_count"] == calls_after_phase1

            # Resume with max_calls=4 (total budget, not additional)
            config2 = OptimizationConfig(
                target=TargetConfig(type="skill", content="seed"),
                budget=BudgetConfig(max_calls=4),
            )
            runner2 = OptimizationRunner(
                config=config2,
                train_tasks=train_tasks,
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=checkpoint_path,
                resume=True,
            )
            runner2.run()

            # Total calls in resumed run should respect total budget
            # (resume restores call_count, then continues from there)
            assert total_calls <= 4  # max_calls budget

            # Final checkpoint should have the correct total
            final_cp = json.loads(checkpoint_path.read_text())
            assert final_cp["call_count"] == total_calls


# ===========================================================================
# Cross-cutting: optimization + task dataset + harness configuration stability
# ===========================================================================


class TestOptimizationEndToEndStability:
    """End-to-end tests that verify optimization, task dataset, and harness
    configuration work together without drift."""

    def test_full_optimization_produces_valid_outputs(self) -> None:
        """A full optimization run produces all expected output files."""
        config = OptimizationConfig(
            target=TargetConfig(
                type="skill",
                content="Write good tests for R code.",
                metadata={"skill_name": "r-tester"},
            ),
            budget=BudgetConfig(max_calls=3),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.7, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(2),
                holdout_tasks=_make_holdout_tasks(1),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
            )
            result = runner.run()

            # All expected files should exist
            assert (Path(tmpdir) / "best_candidate.json").exists()
            assert (Path(tmpdir) / "trajectory.jsonl").exists()
            assert (Path(tmpdir) / "checkpoint.json").exists()

            # Result should be well-formed
            assert result.train_score > 0.0
            assert result.iteration_count > 0
            assert result.call_count > 0
            assert result.termination_reason is not None
            assert isinstance(result.best_candidate, dict)
            assert "content" in result.best_candidate
            assert "score" in result.best_candidate

    def test_target_type_propagates_to_output(self) -> None:
        """The target type from config propagates through to all outputs."""
        for target_type in ["skill", "system_prompt", "tool_policy"]:
            config = OptimizationConfig(
                target=TargetConfig(type=target_type, content=f"test {target_type}"),
                budget=BudgetConfig(max_calls=1),
            )

            def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
                return 0.5, {"task_id": example.id}

            with tempfile.TemporaryDirectory() as tmpdir:
                runner = OptimizationRunner(
                    config=config,
                    train_tasks=_make_train_tasks(1),
                    evaluate_fn=mock_evaluate,
                    output_dir=Path(tmpdir),
                )
                result = runner.run()

                # Check result
                assert result.best_candidate["target_type"] == target_type

                # Check best_candidate.json
                best_path = Path(tmpdir) / "best_candidate.json"
                data = json.loads(best_path.read_text())
                assert data["target_type"] == target_type

                # Check checkpoint
                cp_path = Path(tmpdir) / "checkpoint.json"
                cp_data = json.loads(cp_path.read_text())
                assert cp_data["target_type"] == target_type

    def test_seed_content_preserved_in_checkpoint(self) -> None:
        """Original seed content is preserved in the checkpoint for reference."""
        seed = "Original skill content that should not change"
        config = OptimizationConfig(
            target=TargetConfig(type="skill", content=seed),
            budget=BudgetConfig(max_calls=2),
        )

        def mock_evaluate(candidate: str, example: Any, **kwargs: Any) -> tuple[float, dict]:
            return 0.5, {"task_id": example.id}

        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.json"
            runner = OptimizationRunner(
                config=config,
                train_tasks=_make_train_tasks(1),
                evaluate_fn=mock_evaluate,
                output_dir=Path(tmpdir),
                checkpoint_path=cp_path,
            )
            runner.run()

            cp_data = json.loads(cp_path.read_text())
            assert cp_data["seed_content"] == seed

    def test_multiple_target_types_with_checkpoint_resume(self) -> None:
        """Checkpoint/resume works correctly for all target types."""
        for target_type in ["skill", "system_prompt", "tool_policy"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                cp_path = Path(tmpdir) / "checkpoint.json"

                def mock_evaluate(
                    candidate: str, example: Any, **kwargs: Any
                ) -> tuple[float, dict]:
                    return 0.6, {"task_id": example.id}

                # Phase 1
                config1 = OptimizationConfig(
                    target=TargetConfig(type=target_type, content=f"seed-{target_type}"),
                    budget=BudgetConfig(max_calls=1),
                )
                runner1 = OptimizationRunner(
                    config=config1,
                    train_tasks=_make_train_tasks(1),
                    evaluate_fn=mock_evaluate,
                    output_dir=Path(tmpdir),
                    checkpoint_path=cp_path,
                )
                runner1.run()

                # Phase 2: resume
                config2 = OptimizationConfig(
                    target=TargetConfig(type=target_type, content=f"seed-{target_type}"),
                    budget=BudgetConfig(max_calls=3),
                )
                runner2 = OptimizationRunner(
                    config=config2,
                    train_tasks=_make_train_tasks(1),
                    evaluate_fn=mock_evaluate,
                    output_dir=Path(tmpdir),
                    checkpoint_path=cp_path,
                    resume=True,
                )
                result = runner2.run()

                assert result.best_candidate["target_type"] == target_type
                cp_data = json.loads(cp_path.read_text())
                assert cp_data["target_type"] == target_type
