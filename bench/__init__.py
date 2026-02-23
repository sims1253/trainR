"""Benchmark infrastructure package.

This package provides:
- Schema definitions (TaskV1, ResultV1, ManifestV1, ProfileV1)
- Experiment runner (ExperimentConfig, ExperimentRunner)
- Validation utilities
"""

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

# Lazy import for experiments to avoid circular imports
__all__ = [
    "ManifestV1",
    "ProfileV1",
    "ResultV1",
    # Schema
    "TaskV1",
    "load_json_schema",
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
