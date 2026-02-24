"""Benchmark infrastructure package.

This package provides:
- Schema definitions (TaskV1, ResultV1, ManifestV1, ProfileV1)
- Experiment runner (ExperimentConfig, ExperimentRunner)
- Canonical execution API (bench.runner.run)
- Validation utilities

The canonical entry point for benchmark execution is bench.runner.run():
    from bench import run
    manifest = run("configs/experiments/smoke.yaml")
"""

from bench.runner import run
from bench.schema.v1 import (
    ManifestV1,
    ProfileV1,
    ResultV1,
    TaskV1,
    load_json_schema,
    validate_manifest,
    validate_profile,
    validate_result,
    validate_task,
)

__all__ = [
    # Schema types
    "ManifestV1",
    "ProfileV1",
    "ResultV1",
    "TaskV1",
    # Schema utilities
    "load_json_schema",
    # Canonical execution API
    "run",
    "validate_manifest",
    "validate_profile",
    "validate_result",
    "validate_task",
]


def get_experiment_runner():
    """Get the ExperimentRunner class (lazy import)."""
    from bench.experiments import ExperimentRunner

    return ExperimentRunner


def get_experiment_config():
    """Get the ExperimentConfig class (lazy import)."""
    from bench.experiments import ExperimentConfig

    return ExperimentConfig
