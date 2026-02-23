"""Evaluation modules for benchmark framework."""

from bench.eval.prompt_builder import (
    PromptBuilder,
    build_prompt_from_profile,
    compose_system_prompt,
)
from bench.eval.skill_policy import (
    HeuristicPolicy,
    KeywordMatchPolicy,
    PolicyType,
    SelectionRationale,
    SelectionResult,
    SkillMetadata,
    SkillSelectionPolicy,
    discover_skills,
    load_policy_from_config,
    load_policy_from_yaml,
)
from bench.eval.telemetry import (
    OutcomeType,
    TelemetryCollector,
    ToolCallContext,
    ToolCallEvent,
    ToolErrorEvent,
    ToolErrorType,
    ToolMetrics,
    classify_error_type,
)
from bench.eval.tool_registry import (
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
    ToolValidationError,
    ToolVersionNotFoundError,
    get_registry,
    reset_registry,
)

__all__ = [
    # Skill policy
    "HeuristicPolicy",
    "KeywordMatchPolicy",
    "OutcomeType",
    "PolicyType",
    # Prompt builder
    "PromptBuilder",
    "SelectionRationale",
    "SelectionResult",
    "SkillMetadata",
    "SkillSelectionPolicy",
    # Telemetry
    "TelemetryCollector",
    "ToolCallContext",
    "ToolCallEvent",
    "ToolErrorEvent",
    "ToolErrorType",
    "ToolMetrics",
    "ToolNotFoundError",
    # Tool registry
    "ToolRegistry",
    "ToolRegistryError",
    "ToolValidationError",
    "ToolVersionNotFoundError",
    "build_prompt_from_profile",
    "classify_error_type",
    "compose_system_prompt",
    "discover_skills",
    "get_registry",
    "load_policy_from_config",
    "load_policy_from_yaml",
    "reset_registry",
]
