"""Tests for Kaggle task extraction and non-binary score propagation."""

from __future__ import annotations

import json

from rich.console import Console

from bench.experiments import ExperimentConfig
from bench.experiments.matrix import ExperimentRun, ModelSpec, TaskSpec
from bench.experiments.runner import ExperimentRunner
from bench.harness import HarnessResult, TokenUsage
from bench.schema.v1 import ResultV1, TokenUsageV1
from scripts.run_experiment import ExperimentTUI


def _make_run(task_path) -> ExperimentRun:
    return ExperimentRun(
        run_index=0,
        task=TaskSpec(task_id="task-1", task_path=str(task_path)),
        model=ModelSpec(name="test-model", litellm_model="openrouter/test-model"),
        repeat_index=0,
        fingerprint="fp-1",
    )


def test_build_harness_request_reads_problem_statement_fields(tmp_path) -> None:
    """Kaggle-style task schema should populate prompt/context correctly."""
    task_path = tmp_path / "kaggle_task.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "kaggle_titanic_test",
                "problem_statement": {
                    "instruction": "Build a Titanic survival model.",
                    "context": "Use the standard train/test CSV format.",
                },
                "grading": {"grading_method": "metric", "metric": "accuracy"},
            }
        )
    )

    runner = ExperimentRunner(ExperimentConfig(name="kaggle-request-test"))
    request = runner._build_harness_request(_make_run(task_path))

    assert request.prompt == "Build a Titanic survival model."
    assert request.metadata["context"] == "Use the standard train/test CSV format."
    assert request.metadata["grading"] == {"grading_method": "metric", "metric": "accuracy"}


def test_convert_harness_result_preserves_non_binary_score_and_tier(tmp_path) -> None:
    """Harness-provided score metadata should not be collapsed to pass/fail."""
    runner = ExperimentRunner(ExperimentConfig(name="kaggle-score-test"))
    run = _make_run(tmp_path / "task.json")

    harness_result = HarnessResult(
        task_id="task-1",
        run_id="run-1",
        success=False,
        token_usage=TokenUsage(prompt=10, completion=5, total=15),
        metadata={"score": 0.73, "tier": "silver"},
    )

    result = runner._convert_harness_result(harness_result, run, latency=1.0)

    assert result.passed is False
    assert result.score == 0.73
    assert result.metadata["harness_score_source"] == "metadata.score"
    assert result.metadata["harness_score_tier"] == "silver"


def test_record_result_progress_event_includes_score_and_runtime(tmp_path) -> None:
    """Progress events should include score/runtime for richer TUI metrics."""
    events: list[dict] = []
    runner = ExperimentRunner(
        ExperimentConfig(name="progress-payload-test"), progress_callback=events.append
    )
    results_file = tmp_path / "results.jsonl"

    result = ResultV1(
        result_id="fp-1",
        task_id="task-1",
        model="test-model",
        profile_id="no_skill",
        passed=True,
        score=0.82,
        latency_s=1.2,
        execution_time=2.4,
        token_usage=TokenUsageV1(total=123),
    )

    runner._record_result(result, results_file)

    event = events[-1]
    assert event["type"] == "result"
    assert event["score"] == 0.82
    assert event["runtime_s"] == 2.4


def test_tui_shows_score_and_runtime_columns() -> None:
    """TUI table should render score/runtime aggregates."""
    console = Console(record=True, width=180)
    tui = ExperimentTUI(console)
    tui.callback(
        {
            "type": "start",
            "models": ["test-model"],
            "task_count": 1,
            "total_runs": 1,
            "model_totals": {"test-model": 1},
        }
    )
    tui.callback(
        {
            "type": "result",
            "model": "test-model",
            "task_id": "task-1",
            "passed": True,
            "score": 0.81,
            "tokens": 50,
            "latency_s": 1.1,
            "runtime_s": 2.5,
        }
    )

    table = tui.build_table()
    console.print(table)
    rendered = console.export_text()

    assert "Score" in rendered
    assert "Time" in rendered
    assert "0.81" in rendered
    assert "2.5s" in rendered
