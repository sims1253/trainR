"""Optimization package for benchmark configuration.

This package provides a generalized optimization framework that can optimize
various components of the benchmark system:

- Skills: Text content for guiding agents
- System prompts: Base prompts and injection strategies
- Tool policies: Tool selection and prioritization

The key abstraction is the OptimizableTarget interface, which defines a
contract for any component that can be optimized.

Optimization Contract:
----------------------
Each target must define:
- Objective metric: How to measure success (e.g., pass rate, score)
- Constraints: What must remain valid (e.g., valid markdown, non-empty)
- Tie-breaks: How to choose between equal scores (e.g., shorter text)
- Stop rule: When to halt optimization (e.g., max iterations, convergence)
"""

from bench.optimize.runtime import (
    BudgetConfig,
    BudgetExceededError,
    BudgetUsage,
    InterruptHandler,
    OptimizationRun,
    OptimizationState,
    StopReason,
    TrajectoryEntry,
    check_budget,
    has_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from bench.optimize.targets.base import (
    CandidateType,
    OptimizableTarget,
    ParamSpace,
    ParamType,
    TargetFingerprint,
)
from bench.optimize.gepa_adapter import (
    BatchEvaluator,
    EvaluationResult,
    GEPASkillEvaluator,
    ObjectiveConfig,
    ObjectiveType,
    create_gepa_evaluator,
)
from bench.optimize.targets.skill import SkillCandidate, SkillTarget
from bench.optimize.targets.system_prompt import SystemPromptCandidate, SystemPromptTarget
from bench.optimize.targets.tool_policy import ToolPolicyCandidate, ToolPolicyTarget

__all__ = [
    # GEPA adapter
    "GEPASkillEvaluator",
    "BatchEvaluator",
    "ObjectiveType",
    "ObjectiveConfig",
    "EvaluationResult",
    "create_gepa_evaluator",
    # Base interface
    "OptimizableTarget",
    "ParamSpace",
    "ParamType",
    "CandidateType",
    "TargetFingerprint",
    # Concrete targets
    "SkillTarget",
    "SkillCandidate",
    "SystemPromptTarget",
    "SystemPromptCandidate",
    "ToolPolicyTarget",
    "ToolPolicyCandidate",
    # Runtime
    "BudgetConfig",
    "BudgetUsage",
    "BudgetExceededError",
    "StopReason",
    "TrajectoryEntry",
    "OptimizationState",
    "OptimizationRun",
    "InterruptHandler",
    "check_budget",
    "save_checkpoint",
    "load_checkpoint",
    "has_checkpoint",
]
