"""Optimizable targets for benchmark optimization.

Targets define what can be optimized in the benchmark system.
Each target implements the OptimizableTarget interface, which
specifies how to serialize, deserialize, and apply candidates.
"""

from bench.optimize.targets.base import (
    CandidateType,
    OptimizableTarget,
    ParamSpace,
    ParamType,
    TargetFingerprint,
)
from bench.optimize.targets.skill import SkillCandidate, SkillTarget
from bench.optimize.targets.system_prompt import SystemPromptCandidate, SystemPromptTarget
from bench.optimize.targets.tool_policy import ToolPolicyCandidate, ToolPolicyTarget

__all__ = [
    # Base interface
    "OptimizableTarget",
    "ParamSpace",
    "ParamType",
    "CandidateType",
    "TargetFingerprint",
    # Skill target
    "SkillTarget",
    "SkillCandidate",
    # System prompt target
    "SystemPromptTarget",
    "SystemPromptCandidate",
    # Tool policy target
    "ToolPolicyTarget",
    "ToolPolicyCandidate",
]
