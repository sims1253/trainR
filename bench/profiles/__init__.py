"""Profile modules for benchmark configuration.

This package provides:
- SupportProfile: Configuration for skill/agent support modes
- ToolProfile: Tool/sandbox configuration profiles
"""

from bench.profiles.support import (
    AgentConfig,
    ComposedSupportArtifact,
    DEFAULT_PROFILES,
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
    # Support profiles
    "SupportProfile",
    "SupportMode",
    "SupportFingerprint",
    "SkillReference",
    "AgentConfig",
    "SystemPromptConfig",
    "ComposedSupportArtifact",
    "load_support_profile",
    "DEFAULT_PROFILES",
    # Tool profiles
    "ToolConfig",
    "ToolProfile",
    "ToolVersion",
    "load_tool_profile",
    "validate_tool_config",
]
