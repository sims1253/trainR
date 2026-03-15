"""Behavior tests for ExperimentConfig execution details."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bench.experiments.config import ExperimentConfig


def test_generate_run_id_is_stable_within_instance() -> None:
    """Auto-generated run_id should not change across repeated calls."""
    config = ExperimentConfig(name="stable-run-id")

    run_id_first = config.generate_run_id()
    run_id_second = config.generate_run_id()
    output_dir = config.get_output_dir()

    assert run_id_first == run_id_second
    assert output_dir.name == run_id_first


def test_generate_run_id_respects_explicit_output_run_id() -> None:
    """Explicit output.run_id should be returned as-is."""
    config = ExperimentConfig.model_validate(
        {
            "name": "explicit-run-id",
            "output": {"run_id": "my-fixed-run-id"},
        }
    )

    assert config.generate_run_id() == "my-fixed-run-id"
    assert config.get_output_dir().name == "my-fixed-run-id"


def test_execution_config_rejects_unregistered_harness() -> None:
    """Execution harness validation should reject names not in registry."""
    with pytest.raises(ValidationError, match="not registered"):
        ExperimentConfig.model_validate(
            {"name": "invalid-harness", "execution": {"harness": "pi_sdk"}}
        )


def test_execution_config_accepts_provider_parallel_limits() -> None:
    """Provider parallel limits should normalize provider names."""
    config = ExperimentConfig.model_validate(
        {
            "name": "provider-limits-valid",
            "execution": {"provider_parallel_limits": {"OpenRouter": 1, "zai": 2}},
        }
    )
    assert config.execution.provider_parallel_limits == {"openrouter": 1, "zai": 2}


def test_execution_config_rejects_invalid_provider_parallel_limits() -> None:
    """Provider limits must be positive integers."""
    with pytest.raises(ValidationError, match="must be >= 1"):
        ExperimentConfig.model_validate(
            {
                "name": "provider-limits-invalid",
                "execution": {"provider_parallel_limits": {"openrouter": 0}},
            }
        )


def test_execution_config_accepts_provider_pacing_limits() -> None:
    """Provider pacing maps should normalize provider names and numeric values."""
    config = ExperimentConfig.model_validate(
        {
            "name": "provider-pacing-valid",
            "execution": {
                "provider_min_interval_s": {"OpenRouter": 0.75},
                "provider_max_requests_per_second": {"ZAI": 0.5},
            },
        }
    )
    assert config.execution.provider_min_interval_s == {"openrouter": 0.75}
    assert config.execution.provider_max_requests_per_second == {"zai": 0.5}


def test_execution_config_rejects_invalid_provider_pacing_limits() -> None:
    """Provider pacing maps must use positive numeric values."""
    with pytest.raises(ValidationError, match="must be > 0"):
        ExperimentConfig.model_validate(
            {
                "name": "provider-pacing-invalid",
                "execution": {"provider_min_interval_s": {"openrouter": 0}},
            }
        )
