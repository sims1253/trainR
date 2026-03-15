"""Optimization module for grist-mill.

Provides:
- GEPA evaluator adapter for wrapping the evaluation harness as a
  GEPA-compatible evaluator function (score, side_info_dict).
- Optimization runtime with budget management, checkpointing, resume,
  Pareto front, and multiple target types.
- CLI 'grist-mill optimize' subcommand.

Validates:
- VAL-GEPA-01 through VAL-GEPA-06 (evaluator adapter)
- VAL-OPT-01 through VAL-OPT-08 (optimization runtime)
- VAL-GEPA-04, VAL-GEPA-05 (holdout and Pareto front)
"""

from __future__ import annotations

from grist_mill.optimization.evaluator_adapter import (
    CostAdjustedObjective,
    DifficultyWeightedObjective,
    EvaluatorAdapterConfig,
    GepaEvaluatorAdapter,
    ObjectiveFunction,
    PassRateObjective,
    create_evaluator_adapter,
    load_custom_evaluator,
)
from grist_mill.optimization.runtime import (
    BaseProposer,
    BudgetConfig,
    CheckpointState,
    MockProposer,
    OptimizationCheckpoint,
    OptimizationConfig,
    OptimizationResult,
    OptimizationRunner,
    ParetoFront,
    StopCondition,
    TargetConfig,
    TargetType,
    export_best_candidate,
    serialize_target,
    split_tasks,
)

__all__ = [
    "BaseProposer",
    "BudgetConfig",
    "CheckpointState",
    "CostAdjustedObjective",
    "DifficultyWeightedObjective",
    "EvaluatorAdapterConfig",
    "GepaEvaluatorAdapter",
    "MockProposer",
    "ObjectiveFunction",
    "OptimizationCheckpoint",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationRunner",
    "ParetoFront",
    "PassRateObjective",
    "StopCondition",
    "TargetConfig",
    "TargetType",
    "create_evaluator_adapter",
    "export_best_candidate",
    "load_custom_evaluator",
    "serialize_target",
    "split_tasks",
]
