"""Profile modules for benchmark configuration.

This package provides:
- SupportProfile: Configuration for skill/agent support modes
- ToolProfile: Tool/sandbox configuration profiles
"""

from bench.profiles.support import (
    DEFAULT_PROFILES,
    AgentConfig,
    ComposedSupportArtifact,
    SelectionMetadata,
    SkillReference,
    SupportFingerprint,
    SupportMode,
    SupportProfile,
    SystemPromptConfig,
    load_support_profile,
)
from bench.profiles.tools import (
    ToolConfig,
    ToolProfile,
    ToolVersion,
    load_tool_profile,
    validate_tool_config,
)

__all__ = [
    "DEFAULT_PROFILES",
    "AgentConfig",
    "ComposedSupportArtifact",
    "SelectionMetadata",
    "SkillReference",
    "SupportFingerprint",
    "SupportMode",
    # Support profiles
    "SupportProfile",
    "SystemPromptConfig",
    # Tool profiles
    "ToolConfig",
    "ToolProfile",
    "ToolVersion",
    "load_support_profile",
    "load_tool_profile",
    "validate_tool_config",
]
