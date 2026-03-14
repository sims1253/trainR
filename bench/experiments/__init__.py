"""Unified experiment runner for benchmark evaluation.

This module provides configuration and matrix generation for experiments.
For execution, use the canonical runner API:

    from bench.runner import run

    config = ExperimentConfig.from_yaml("configs/experiments/smoke.yaml")
    manifest = run(config)

Internal APIs (not for direct use):
    - ExperimentRunner: Internal execution engine
    - run_experiment: Internal convenience function
"""

from bench.experiments.config import (
    DeterminismConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelsConfig,
    OutputConfig,
    PairingConfig,
    RetryConfig,
    RetryStrategy,
    SkillConfig,
    TasksConfig,
    TaskSelectionMode,
    load_experiment_config,
)
from bench.experiments.matrix import (
    ExperimentMatrix,
    ExperimentRun,
    ModelSpec,
    SupportSpec,
    TaskSpec,
    ToolABMatrix,
    ToolABPair,
    ToolABRun,
    ToolSpec,
    generate_matrix,
    generate_tool_ab_matrix,
)
from bench.experiments.runner import ExperimentRunner, run_experiment

# Internal: Use bench.runner.run() as the canonical API
__all__ = [
    "DeterminismConfig",
    "ExecutionConfig",
    "ExperimentConfig",
    "ExperimentMatrix",
    "ExperimentRun",
    "ExperimentRunner",
    "ModelSpec",
    "ModelsConfig",
    "OutputConfig",
    "PairingConfig",
    "RetryConfig",
    "RetryStrategy",
    "SkillConfig",
    "SupportSpec",
    "TaskSelectionMode",
    "TaskSpec",
    "TasksConfig",
    "ToolABMatrix",
    "ToolABPair",
    "ToolABRun",
    "ToolSpec",
    "generate_matrix",
    "generate_tool_ab_matrix",
    "load_experiment_config",
    "run_experiment",
]
