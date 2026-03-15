"""Optimization module for grist-mill.

Provides GEPA evaluator adapter for wrapping the evaluation harness as a
GEPA-compatible evaluator function. The adapter returns (score, side_info_dict)
per the GEPA contract, with side info capturing execution traces, errors,
timing, and token usage.

Supports configurable objectives (pass-rate, cost-adjusted, difficulty-weighted)
with deterministic scoring.

Validates:
- VAL-GEPA-01: Evaluator adapter captures actionable side information
- VAL-GEPA-02: Side information is passed to reflection model
- VAL-GEPA-03: Multi-objective evaluation produces composite scores
- VAL-GEPA-06: Evaluator adapter is pluggable
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

__all__ = [
    "CostAdjustedObjective",
    "DifficultyWeightedObjective",
    "EvaluatorAdapterConfig",
    "GepaEvaluatorAdapter",
    "ObjectiveFunction",
    "PassRateObjective",
    "create_evaluator_adapter",
    "load_custom_evaluator",
]
