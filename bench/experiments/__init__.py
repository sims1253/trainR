"""Unified experiment runner for benchmark evaluation.

This module provides the canonical entry point for running experiments:
- ExperimentConfig: Configuration schema
- ExperimentRunner: Main execution engine
- ExperimentMatrix: Pre-computed experiment plan

Usage:
    from bench.experiments import ExperimentConfig, run_experiment

    config = ExperimentConfig.from_yaml("configs/experiments/smoke.yaml")
    manifest = run_experiment(config)
"""

from bench.experiments.config import (
    DeterminismConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelsConfig,
    OutputConfig,
    RetryConfig,
    RetryStrategy,
    SkillConfig,
    TaskSelectionMode,
    TasksConfig,
    load_experiment_config,
)
from bench.experiments.matrix import (
    ExperimentMatrix,
    ExperimentRun,
    ModelSpec,
    TaskSpec,
    generate_matrix,
)
from bench.experiments.runner import ExperimentRunner, run_experiment

__all__ = [
    # Config
    "ExperimentConfig",
    "TasksConfig",
    "ModelsConfig",
    "SkillConfig",
    "ExecutionConfig",
    "RetryConfig",
    "RetryStrategy",
    "OutputConfig",
    "DeterminismConfig",
    "TaskSelectionMode",
    "load_experiment_config",
    # Matrix
    "ExperimentMatrix",
    "ExperimentRun",
    "TaskSpec",
    "ModelSpec",
    "generate_matrix",
    # Runner
    "ExperimentRunner",
    "run_experiment",
]
