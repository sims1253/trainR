"""Tests for per-provider parallel execution limits."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from bench.experiments import ExperimentConfig
from bench.experiments.matrix import ExperimentMatrix, ExperimentRun, ModelSpec, TaskSpec
from bench.experiments.runner import ExperimentRunner
from bench.schema.v1 import ResultV1, TokenUsageV1


def _success_result(run: ExperimentRun) -> ResultV1:
    """Create a minimal successful ResultV1 for mocked run execution."""
    return ResultV1(
        result_id=run.fingerprint,
        task_id=run.task.task_id,
        model=run.model.name,
        profile_id="no_skill",
        passed=True,
        score=1.0,
        error_category=None,
        error_message=None,
        latency_s=0.01,
        execution_time=0.01,
        token_usage=TokenUsageV1(),
        test_results=[],
        repeat_index=run.repeat_index,
        metadata={},
        telemetry=None,
        tool_call_counts={},
        tool_errors={},
        tool_total_time_ms={},
    )


def _make_run(run_index: int, provider: str) -> ExperimentRun:
    """Create a matrix run with a provider-tagged model."""
    return ExperimentRun(
        run_index=run_index,
        task=TaskSpec(task_id=f"task-{run_index}", task_path=f"tasks/dev/task-{run_index}.json"),
        model=ModelSpec(
            name=f"{provider}-model-{run_index}",
            litellm_model=f"{provider}/model-{run_index}",
            provider=provider,
        ),
        repeat_index=0,
        fingerprint=f"fp-{run_index}",
    )


def test_parallel_execution_respects_provider_caps(tmp_path) -> None:
    """Provider-specific limits should apply on top of global parallel_workers."""
    config = ExperimentConfig.model_validate(
        {
            "name": "provider-cap-test",
            "output": {"dir": str(tmp_path)},
            "execution": {
                "parallel_workers": 4,
                "provider_parallel_limits": {"openrouter": 1, "zai": 2},
            },
        }
    )
    runner = ExperimentRunner(config)
    runner._memory_saving = False

    runs: list[ExperimentRun] = []
    run_index = 0
    for provider, count in [("openrouter", 6), ("zai", 6), ("opencode", 4)]:
        for _ in range(count):
            runs.append(_make_run(run_index, provider))
            run_index += 1

    runner.matrix = ExperimentMatrix(config=config, runs=runs)
    results_file = tmp_path / "results.jsonl"

    lock = threading.Lock()
    active_total = 0
    max_total = 0
    active_by_provider: defaultdict[str, int] = defaultdict(int)
    max_by_provider: defaultdict[str, int] = defaultdict(int)

    def fake_execute_run(run: ExperimentRun) -> ResultV1:
        nonlocal active_total, max_total
        provider = runner._resolve_run_provider(run)

        with lock:
            active_total += 1
            max_total = max(max_total, active_total)
            active_by_provider[provider] += 1
            max_by_provider[provider] = max(max_by_provider[provider], active_by_provider[provider])

        time.sleep(0.03)

        with lock:
            active_by_provider[provider] -= 1
            active_total -= 1

        return _success_result(run)

    # Bind a test double at the instance level to observe concurrent execution.
    runner._execute_run = fake_execute_run  # type: ignore[method-assign]

    runner._run_parallel(results_file)

    assert len(runner.results) == len(runs)
    assert max_total <= 4
    assert max_by_provider["openrouter"] <= 1
    assert max_by_provider["zai"] <= 2
    assert max_by_provider["opencode"] >= 1


def test_provider_limits_are_capped_by_global_workers() -> None:
    """Per-provider limits above global workers should be clamped."""
    config = ExperimentConfig.model_validate(
        {
            "name": "provider-cap-clamp-test",
            "execution": {
                "parallel_workers": 2,
                "provider_parallel_limits": {"openrouter": 5, "zai": 3},
            },
        }
    )
    runner = ExperimentRunner(config)
    assert runner._get_provider_parallel_limits(worker_limit=2) == {"openrouter": 2, "zai": 2}


def test_provider_min_interval_throttles_run_starts(tmp_path) -> None:
    """Provider pacing should enforce minimum spacing between run starts."""
    config = ExperimentConfig.model_validate(
        {
            "name": "provider-pacing-test",
            "output": {"dir": str(tmp_path)},
            "execution": {
                "parallel_workers": 3,
                "provider_min_interval_s": {"openrouter": 0.05},
            },
        }
    )
    runner = ExperimentRunner(config)
    runs = [_make_run(i, "openrouter") for i in range(6)]
    runner.matrix = ExperimentMatrix(config=config, runs=runs)
    results_file = tmp_path / "results.jsonl"

    start_times: list[float] = []
    lock = threading.Lock()

    def fake_execute_run(run: ExperimentRun) -> ResultV1:
        with lock:
            start_times.append(time.monotonic())
        time.sleep(0.005)
        return _success_result(run)

    runner._execute_run = fake_execute_run  # type: ignore[method-assign]
    runner._run_parallel(results_file)

    assert len(start_times) == len(runs)
    ordered = sorted(start_times)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    assert all(gap >= 0.045 for gap in gaps)


def test_provider_rps_cap_sets_effective_interval() -> None:
    """RPS caps should translate to minimum spacing and dominate smaller intervals."""
    config = ExperimentConfig.model_validate(
        {
            "name": "provider-rps-interval-test",
            "execution": {
                "provider_min_interval_s": {"openrouter": 0.1},
                "provider_max_requests_per_second": {"openrouter": 2.0},
            },
        }
    )
    runner = ExperimentRunner(config)
    assert runner._get_provider_min_interval_s("openrouter") == 0.5
    assert runner._get_provider_min_interval_s("unknown") == 0.0


def test_extract_retry_after_seconds_from_error_message() -> None:
    """Runner should parse Retry-After style hints in provider errors."""
    runner = ExperimentRunner(ExperimentConfig(name="retry-after-parse"))
    assert runner._extract_retry_after_seconds("429 retry-after: 90") == 90.0
    assert runner._extract_retry_after_seconds("Too many requests, try again in 2 minutes") == 120.0


def test_interleave_runs_by_model_round_robin() -> None:
    """Runs should be scheduled in model round-robin order."""
    config = ExperimentConfig.model_validate(
        {
            "name": "model-round-robin-test",
            "models": {"names": ["model-a", "model-b"]},
        }
    )
    runner = ExperimentRunner(config)

    runs = [
        ExperimentRun(
            run_index=0,
            task=TaskSpec(task_id="t1", task_path="tasks/dev/task-1.json"),
            model=ModelSpec(
                name="model-a", litellm_model="openrouter/model-a", provider="openrouter"
            ),
            repeat_index=0,
            fingerprint="fp-0",
        ),
        ExperimentRun(
            run_index=1,
            task=TaskSpec(task_id="t2", task_path="tasks/dev/task-2.json"),
            model=ModelSpec(
                name="model-a", litellm_model="openrouter/model-a", provider="openrouter"
            ),
            repeat_index=0,
            fingerprint="fp-1",
        ),
        ExperimentRun(
            run_index=2,
            task=TaskSpec(task_id="t1", task_path="tasks/dev/task-1.json"),
            model=ModelSpec(
                name="model-b", litellm_model="openrouter/model-b", provider="openrouter"
            ),
            repeat_index=0,
            fingerprint="fp-2",
        ),
        ExperimentRun(
            run_index=3,
            task=TaskSpec(task_id="t2", task_path="tasks/dev/task-2.json"),
            model=ModelSpec(
                name="model-b", litellm_model="openrouter/model-b", provider="openrouter"
            ),
            repeat_index=0,
            fingerprint="fp-3",
        ),
    ]
    runner.matrix = ExperimentMatrix(
        config=config,
        models=[
            ModelSpec(name="model-a", litellm_model="openrouter/model-a", provider="openrouter"),
            ModelSpec(name="model-b", litellm_model="openrouter/model-b", provider="openrouter"),
        ],
        runs=runs,
    )

    interleaved = runner._interleave_runs_by_model(runs)
    assert [r.model.name for r in interleaved] == ["model-a", "model-b", "model-a", "model-b"]
