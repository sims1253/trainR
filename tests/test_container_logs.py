"""Tests for container log artifact persistence."""

from __future__ import annotations

from bench.experiments import ExperimentConfig
from bench.experiments.matrix import ExperimentRun, ModelSpec, TaskSpec
from bench.experiments.runner import ExperimentRunner
from bench.harness import HarnessResult


def test_convert_harness_result_saves_container_log_artifact(tmp_path) -> None:
    """When enabled, container stdout/stderr should be persisted per run."""
    config = ExperimentConfig.model_validate(
        {
            "name": "container-log-test",
            "output": {"dir": str(tmp_path)},
            "execution": {"save_container_logs": True},
        }
    )
    runner = ExperimentRunner(config)
    runner.output_dir = tmp_path

    run = ExperimentRun(
        run_index=0,
        task=TaskSpec(task_id="task-1", task_path="tasks/dev/task-1.json"),
        model=ModelSpec(name="test-model", litellm_model="openrouter/test-model"),
        repeat_index=0,
        fingerprint="abc123",
    )

    harness_result = HarnessResult(
        task_id="task-1",
        run_id="run-1",
        success=False,
        output="container stdout",
        error_message="container stderr",
    )

    result = runner._convert_harness_result(harness_result, run, latency=1.0)
    rel_log_path = result.metadata.get("container_log_path")

    assert isinstance(rel_log_path, str)
    log_path = tmp_path / rel_log_path
    assert log_path.exists()
    content = log_path.read_text()
    assert "=== STDOUT ===" in content
    assert "container stdout" in content
    assert "=== STDERR/ERROR ===" in content
    assert "container stderr" in content
