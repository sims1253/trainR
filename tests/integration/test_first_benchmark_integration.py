"""Integration coverage for the first benchmark configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from bench.experiments import load_experiment_config
from bench.experiments.runner import ExperimentRunner
from bench.provider import run_preflight
from bench.runner import run

FIRST_BENCHMARK_CONFIG = Path("configs/experiments/first_benchmark.yaml")
FIRST_BENCHMARK_MODELS = ["gpt-oss-120b", "glm-5-free", "glm-5-plan"]
REQUIRED_PROVIDER_ENV_VARS = ("OPENROUTER_API_KEY", "OPENCODE_API_KEY", "Z_AI_API_KEY")


def _missing_provider_env_vars() -> list[str]:
    """Return required provider env vars that are currently missing."""
    return [name for name in REQUIRED_PROVIDER_ENV_VARS if not os.environ.get(name)]


def _load_first_benchmark_config(tmp_path: Path, run_name: str) -> Any:
    """Load first benchmark config and redirect output to a temp dir."""
    config = load_experiment_config(FIRST_BENCHMARK_CONFIG)
    config.name = run_name
    config.output.dir = str(tmp_path)
    return config


def test_first_benchmark_dry_run_matrix_contract(tmp_path: Path) -> None:
    """First benchmark must stay pinned to 1 task x 3 models."""
    config = _load_first_benchmark_config(tmp_path, "first_benchmark_dry_run_contract")

    manifest = run(config, dry_run=True)

    assert manifest.models == FIRST_BENCHMARK_MODELS
    assert manifest.task_count == 1
    assert manifest.config.get("total_runs") == 3
    assert manifest.config.get("repeats") == 1


@pytest.mark.integration_provider
def test_first_benchmark_preflight_with_real_provider_keys() -> None:
    """First benchmark preflight should validate all 3 provider credentials."""
    missing = _missing_provider_env_vars()
    if missing:
        pytest.skip(
            f"Missing provider credentials for integration_provider test: {', '.join(missing)}"
        )

    preflight = run_preflight(FIRST_BENCHMARK_MODELS, strict=True)
    assert preflight.is_valid, f"Preflight failed: {preflight.errors}"


@pytest.mark.integration_provider
def test_first_benchmark_setup_preserves_provider_model_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner setup should preserve expected provider/model routing for all 3 models."""
    missing = _missing_provider_env_vars()
    if missing:
        pytest.skip(
            f"Missing provider credentials for integration_provider test: {', '.join(missing)}"
        )

    class DummyHarness:
        def __init__(self, _config: Any):
            self._config = _config

        def validate_environment(self) -> tuple[bool, list[str]]:
            return (True, [])

    monkeypatch.setattr(
        "bench.experiments.runner.HarnessRegistry.get",
        lambda _harness_name, harness_config: DummyHarness(harness_config),
    )

    config = _load_first_benchmark_config(tmp_path, "first_benchmark_setup_contract")
    runner = ExperimentRunner(config)
    output_dir = runner.setup()

    assert output_dir.exists()
    assert runner.matrix is not None
    assert len(runner.matrix.runs) == 3
    assert runner.manifest is not None
    assert runner.manifest.metadata.get("providers") == {
        "gpt-oss-120b": "openrouter",
        "glm-5-free": "opencode",
        "glm-5-plan": "zai_coding_plan",
    }

    request_models_by_name: dict[str, str] = {}
    for matrix_run in runner.matrix.runs:
        request = runner._build_harness_request(matrix_run)
        request_models_by_name[matrix_run.model.name] = request.metadata["model"]

    assert request_models_by_name == {
        "gpt-oss-120b": "openrouter/openai/gpt-oss-120b:free",
        "glm-5-free": "openai/glm-5-free",
        "glm-5-plan": "openai/glm-5",
    }
