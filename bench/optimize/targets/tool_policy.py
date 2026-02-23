"""Tool policy optimization target.

This target optimizes the tool selection and availability policy -
which tools are enabled, their priorities, and how they are configured.
The optimization can explore:
- Which tools to enable/disable
- Tool priority ordering
- Tool-specific configuration parameters

Target Isolation:
----------------
ToolPolicyTarget ONLY modifies tool configuration.
It does NOT modify:
- Skill content/references
- System prompt configuration
- Agent configuration
- Any other profile dimension
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from bench.experiments.config import ExperimentConfig

from bench.optimize.targets.base import (
    OptimizableTarget,
    ParamSpace,
    ParamSpec,
    ParamType,
    TargetFingerprint,
)


@dataclass
class ToolConfig:
    """Configuration for a single tool."""

    enabled: bool = True
    """Whether the tool is available."""

    priority: int = 0
    """Priority for tool ordering (higher = preferred)."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Tool-specific configuration parameters."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "priority": self.priority,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolConfig":
        """Deserialize from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0),
            parameters=data.get("parameters", {}),
        )


@dataclass
class ToolPolicyCandidate:
    """Candidate value for tool policy optimization.

    A tool policy candidate represents a potential configuration
    for tool availability, priorities, and parameters.
    """

    tools: dict[str, ToolConfig] = field(default_factory=dict)
    """Tool configurations keyed by tool ID."""

    default_enabled: bool = True
    """Whether unknown tools are enabled by default."""

    max_concurrent_tools: int | None = None
    """Maximum number of tools that can be used concurrently."""

    tool_timeout: int | None = None
    """Default timeout for tool operations in seconds."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional configuration metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tools": {tool_id: config.to_dict() for tool_id, config in self.tools.items()},
            "default_enabled": self.default_enabled,
            "max_concurrent_tools": self.max_concurrent_tools,
            "tool_timeout": self.tool_timeout,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPolicyCandidate":
        """Deserialize from dictionary."""
        tools = {
            tool_id: ToolConfig.from_dict(config_data)
            for tool_id, config_data in data.get("tools", {}).items()
        }

        return cls(
            tools=tools,
            default_enabled=data.get("default_enabled", True),
            max_concurrent_tools=data.get("max_concurrent_tools"),
            tool_timeout=data.get("tool_timeout"),
            metadata=data.get("metadata", {}),
        )

    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tool IDs sorted by priority."""
        enabled = [
            (tool_id, config.priority) for tool_id, config in self.tools.items() if config.enabled
        ]
        # Sort by priority descending
        enabled.sort(key=lambda x: -x[1])
        return [tool_id for tool_id, _ in enabled]

    def is_tool_enabled(self, tool_id: str) -> bool:
        """Check if a specific tool is enabled."""
        if tool_id in self.tools:
            return self.tools[tool_id].enabled
        return self.default_enabled

    def get_tool_priority(self, tool_id: str) -> int:
        """Get priority for a specific tool."""
        if tool_id in self.tools:
            return self.tools[tool_id].priority
        return 0

    def compute_hash(self) -> str:
        """Compute hash for change detection."""
        import json

        data = self.to_dict()
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()[:16]


class ToolPolicyTarget(OptimizableTarget[ToolPolicyCandidate]):
    """
    Target for optimizing tool selection policy.

    This target allows optimization of which tools are available,
    their priorities, and their configurations.

    Target Isolation:
    ----------------
    When applying a candidate, this target ONLY modifies:
    - Tool availability and configuration stored in config.settings

    It preserves ALL other configuration:
    - Skill content/references
    - System prompt configuration
    - Agent configuration
    - Task selection
    - Execution settings

    Note: Tool configuration is stored in config.settings['tool_policy']
    and picked up by the tool registry during evaluation.
    """

    # Common tool IDs for reference
    COMMON_TOOLS: ClassVar[list[str]] = [
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "web_fetch",
        "notebook",
    ]

    def __init__(
        self,
        available_tools: list[str] | None = None,
        min_enabled_tools: int = 1,
        max_enabled_tools: int | None = None,
        max_priority: int = 100,
        max_concurrent_range: tuple[int, int] = (1, 10),
        timeout_range: tuple[int, int] = (10, 600),
    ) -> None:
        """Initialize the tool policy target.

        Args:
            available_tools: List of available tool IDs (None = use COMMON_TOOLS)
            min_enabled_tools: Minimum number of tools that must be enabled
            max_enabled_tools: Maximum number of tools that can be enabled
            max_priority: Maximum priority value
            max_concurrent_range: (min, max) for max_concurrent_tools
            timeout_range: (min, max) for tool_timeout
        """
        self.available_tools = available_tools or self.COMMON_TOOLS
        self.min_enabled_tools = min_enabled_tools
        self.max_enabled_tools = max_enabled_tools or len(self.available_tools)
        self.max_priority = max_priority
        self.max_concurrent_range = max_concurrent_range
        self.timeout_range = timeout_range

        # Compute config hash for fingerprint
        config_str = f"{available_tools}:{min_enabled_tools}:{max_enabled_tools}:{max_priority}"
        self._config_hash = sha256(config_str.encode()).hexdigest()[:16]

    @property
    def target_type(self) -> str:
        """Get target type identifier."""
        return "tool_policy"

    def fingerprint(self) -> TargetFingerprint:
        """Get unique fingerprint for this target."""
        return TargetFingerprint(
            target_type=self.target_type,
            config_hash=self._config_hash,
            version="1.0",
        )

    def serialize_candidate(self, candidate: ToolPolicyCandidate) -> dict[str, Any]:
        """Serialize a tool policy candidate for storage.

        Args:
            candidate: Tool policy candidate to serialize

        Returns:
            Dictionary representation
        """
        return candidate.to_dict()

    def deserialize_candidate(self, data: dict[str, Any]) -> ToolPolicyCandidate:
        """Deserialize a tool policy candidate from storage.

        Args:
            data: Dictionary from serialize_candidate

        Returns:
            ToolPolicyCandidate instance

        Raises:
            ValueError: If data is invalid
        """
        return ToolPolicyCandidate.from_dict(data)

    def apply_candidate_to_config(
        self,
        config: "ExperimentConfig",
        candidate: ToolPolicyCandidate,
    ) -> "ExperimentConfig":
        """Apply a tool policy candidate to an experiment configuration.

        CRITICAL: This method ONLY modifies tool policy settings.
        All other configuration remains unchanged.

        Args:
            config: Base experiment configuration
            candidate: Tool policy candidate to apply

        Returns:
            New ExperimentConfig with tool policy applied
        """
        from bench.experiments.config import ExperimentConfig

        # Create a deep copy of the config
        config_dict = config.model_dump(mode="json")

        # ONLY modify settings.tool_policy - preserve everything else
        if "settings" not in config_dict:
            config_dict["settings"] = {}

        # Store tool policy configuration
        config_dict["settings"]["tool_policy"] = candidate.to_dict()

        return ExperimentConfig.from_dict(config_dict)

    def get_current_value(self, config: "ExperimentConfig") -> ToolPolicyCandidate:
        """Extract current tool policy from config.

        Args:
            config: Experiment configuration

        Returns:
            ToolPolicyCandidate with current settings
        """
        settings = config.settings or {}
        tp_settings = settings.get("tool_policy", {})

        if tp_settings:
            return ToolPolicyCandidate.from_dict(tp_settings)

        # Return default with all tools enabled
        return self.create_default_candidate()

    def create_default_candidate(self) -> ToolPolicyCandidate:
        """Create a default candidate with all tools enabled.

        Returns:
            ToolPolicyCandidate with default settings
        """
        tools = {tool_id: ToolConfig(enabled=True, priority=0) for tool_id in self.available_tools}
        return ToolPolicyCandidate(
            tools=tools,
            default_enabled=True,
        )

    def get_param_space(self) -> ParamSpace:
        """Define the searchable parameter space for tool policy optimization.

        The parameter space includes:
        - Per-tool enabled/disabled flags
        - Per-tool priority values
        - Global settings (default_enabled, max_concurrent, timeout)
        """
        space = ParamSpace()

        # Per-tool configuration
        for tool_id in self.available_tools:
            # Enabled flag
            space.add_param(
                ParamSpec(
                    name=f"tool_{tool_id}_enabled",
                    param_type=ParamType.BOOLEAN,
                    description=f"Whether {tool_id} tool is enabled",
                    default=True,
                    required=True,
                )
            )

            # Priority
            space.add_param(
                ParamSpec(
                    name=f"tool_{tool_id}_priority",
                    param_type=ParamType.INTEGER,
                    description=f"Priority for {tool_id} tool",
                    default=0,
                    min_value=0,
                    max_value=self.max_priority,
                    required=True,
                )
            )

        # Global settings
        space.add_param(
            ParamSpec(
                name="default_enabled",
                param_type=ParamType.BOOLEAN,
                description="Whether unknown tools are enabled by default",
                default=True,
                required=True,
            )
        )

        space.add_param(
            ParamSpec(
                name="max_concurrent_tools",
                param_type=ParamType.INTEGER,
                description="Maximum concurrent tools (None = unlimited)",
                default=None,
                min_value=self.max_concurrent_range[0],
                max_value=self.max_concurrent_range[1],
                required=False,
            )
        )

        space.add_param(
            ParamSpec(
                name="tool_timeout",
                param_type=ParamType.INTEGER,
                description="Default tool timeout in seconds",
                default=None,
                min_value=self.timeout_range[0],
                max_value=self.timeout_range[1],
                required=False,
            )
        )

        # Add global constraints
        space.constraints = [
            f"At least {self.min_enabled_tools} tool(s) must be enabled",
            f"At most {self.max_enabled_tools} tool(s) can be enabled",
            "Priority values must be in [0, max_priority]",
        ]

        return space

    def validate_candidate(self, candidate: ToolPolicyCandidate) -> tuple[bool, list[str]]:
        """Validate a tool policy candidate.

        Args:
            candidate: Candidate to validate

        Returns:
            Tuple of (is_valid, error messages)
        """
        errors: list[str] = []

        # Check enabled tool count
        enabled_count = len(candidate.get_enabled_tools())
        if enabled_count < self.min_enabled_tools:
            errors.append(f"Too few enabled tools: {enabled_count} < {self.min_enabled_tools}")
        if enabled_count > self.max_enabled_tools:
            errors.append(f"Too many enabled tools: {enabled_count} > {self.max_enabled_tools}")

        # Check priority values
        for tool_id, config in candidate.tools.items():
            if config.priority < 0:
                errors.append(f"Priority for {tool_id} cannot be negative")
            if config.priority > self.max_priority:
                errors.append(
                    f"Priority for {tool_id} exceeds max: {config.priority} > {self.max_priority}"
                )

        # Check concurrent tools setting
        if candidate.max_concurrent_tools is not None:
            if candidate.max_concurrent_tools < self.max_concurrent_range[0]:
                errors.append("max_concurrent_tools below minimum")
            if candidate.max_concurrent_tools > self.max_concurrent_range[1]:
                errors.append("max_concurrent_tools above maximum")

        # Check timeout setting
        if candidate.tool_timeout is not None:
            if candidate.tool_timeout < self.timeout_range[0]:
                errors.append("tool_timeout below minimum")
            if candidate.tool_timeout > self.timeout_range[1]:
                errors.append("tool_timeout above maximum")

        # Check for unknown tools
        for tool_id in candidate.tools:
            if tool_id not in self.available_tools:
                errors.append(f"Unknown tool: {tool_id}")

        return len(errors) == 0, errors

    def candidate_from_flat_dict(self, data: dict[str, Any]) -> ToolPolicyCandidate:
        """Create a candidate from a flat dictionary (e.g., from optimizer).

        This converts flat parameter names like 'tool_bash_enabled' to
        the nested ToolPolicyCandidate structure.

        Args:
            data: Flat dictionary with parameter values

        Returns:
            ToolPolicyCandidate instance
        """
        tools: dict[str, ToolConfig] = {}

        for tool_id in self.available_tools:
            enabled_key = f"tool_{tool_id}_enabled"
            priority_key = f"tool_{tool_id}_priority"

            enabled = data.get(enabled_key, True)
            priority = data.get(priority_key, 0)

            tools[tool_id] = ToolConfig(
                enabled=enabled,
                priority=priority,
            )

        return ToolPolicyCandidate(
            tools=tools,
            default_enabled=data.get("default_enabled", True),
            max_concurrent_tools=data.get("max_concurrent_tools"),
            tool_timeout=data.get("tool_timeout"),
        )


def create_minimal_tool_policy(essential_tools: list[str] | None = None) -> ToolPolicyCandidate:
    """Create a minimal tool policy with only essential tools enabled.

    Args:
        essential_tools: Tools to enable (default: bash, read, write)

    Returns:
        ToolPolicyCandidate with minimal tools enabled
    """
    if essential_tools is None:
        essential_tools = ["bash", "read", "write"]

    tools = {
        tool_id: ToolConfig(enabled=tool_id in essential_tools, priority=0)
        for tool_id in ToolPolicyTarget.COMMON_TOOLS
    }

    return ToolPolicyCandidate(
        tools=tools,
        default_enabled=False,
    )


def create_maximal_tool_policy() -> ToolPolicyCandidate:
    """Create a maximal tool policy with all tools enabled.

    Returns:
        ToolPolicyCandidate with all common tools enabled
    """
    tools = {
        tool_id: ToolConfig(enabled=True, priority=0) for tool_id in ToolPolicyTarget.COMMON_TOOLS
    }

    return ToolPolicyCandidate(
        tools=tools,
        default_enabled=True,
    )
