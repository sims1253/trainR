"""System prompt optimization target.

This target optimizes the system prompt configuration - how prompts
are composed and injected into the agent context. The optimization
can explore:
- Base prompt content
- Injection point (before_context, after_context, replace)
- Whether to include skill metadata

Target Isolation:
----------------
SystemPromptTarget ONLY modifies system prompt configuration.
It does NOT modify:
- Skill content/references
- Tool availability
- Agent configuration
- Any other profile dimension
"""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench.experiments.config import ExperimentConfig

from bench.optimize.targets.base import (
    OptimizableTarget,
    ParamSpace,
    ParamSpec,
    ParamType,
    TargetFingerprint,
)


class InjectionPoint(str, Enum):
    """Where to inject skill content in the system prompt."""

    BEFORE_CONTEXT = "before_context"
    """Inject skill content before the task context."""

    AFTER_CONTEXT = "after_context"
    """Inject skill content after the task context."""

    REPLACE = "replace"
    """Replace the base prompt with skill content."""


@dataclass
class SystemPromptCandidate:
    """Candidate value for system prompt optimization.

    A system prompt candidate represents a potential configuration
    for how system prompts are composed and injected.
    """

    enabled: bool = True
    """Whether system prompt modification is enabled."""

    base_prompt: str | None = None
    """Base system prompt text."""

    injection_point: InjectionPoint = InjectionPoint.AFTER_CONTEXT
    """Where to inject skill content."""

    include_skill_metadata: bool = True
    """Whether to include skill names/headers in prompt."""

    prelude: str | None = None
    """Text to prepend to all prompts."""

    postlude: str | None = None
    """Text to append to all prompts."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional configuration metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "base_prompt": self.base_prompt,
            "injection_point": self.injection_point.value,
            "include_skill_metadata": self.include_skill_metadata,
            "prelude": self.prelude,
            "postlude": self.postlude,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemPromptCandidate":
        """Deserialize from dictionary."""
        injection_point = data.get("injection_point", "after_context")
        if isinstance(injection_point, str):
            injection_point = InjectionPoint(injection_point)

        return cls(
            enabled=data.get("enabled", True),
            base_prompt=data.get("base_prompt"),
            injection_point=injection_point,
            include_skill_metadata=data.get("include_skill_metadata", True),
            prelude=data.get("prelude"),
            postlude=data.get("postlude"),
            metadata=data.get("metadata", {}),
        )

    def compute_hash(self) -> str:
        """Compute hash for change detection."""
        import json

        data = self.to_dict()
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()[:16]


class SystemPromptTarget(OptimizableTarget[SystemPromptCandidate]):
    """
    Target for optimizing system prompt configuration.

    This target allows optimization of how system prompts are composed
    and injected into the agent context.

    Target Isolation:
    ----------------
    When applying a candidate, this target ONLY modifies:
    - support profile's system_prompt configuration

    It preserves ALL other configuration:
    - Skill content/references
    - Agent configuration
    - Tool configuration
    - All other experiment configuration

    Note: System prompt configuration is stored in SupportProfile,
    not directly in ExperimentConfig. This target requires the
    experiment to use a support profile that can be modified.
    """

    def __init__(
        self,
        max_base_prompt_length: int = 10000,
        max_prelude_length: int = 1000,
        max_postlude_length: int = 1000,
        allowed_injection_points: list[InjectionPoint] | None = None,
    ) -> None:
        """Initialize the system prompt target.

        Args:
            max_base_prompt_length: Maximum base prompt length
            max_prelude_length: Maximum prelude length
            max_postlude_length: Maximum postlude length
            allowed_injection_points: Restrict injection points (None = all)
        """
        self.max_base_prompt_length = max_base_prompt_length
        self.max_prelude_length = max_prelude_length
        self.max_postlude_length = max_postlude_length
        self.allowed_injection_points = allowed_injection_points or list(InjectionPoint)

        # Compute config hash for fingerprint
        config_str = f"{max_base_prompt_length}:{max_prelude_length}:{max_postlude_length}:{allowed_injection_points}"
        self._config_hash = sha256(config_str.encode()).hexdigest()[:16]

    @property
    def target_type(self) -> str:
        """Get target type identifier."""
        return "system_prompt"

    def fingerprint(self) -> TargetFingerprint:
        """Get unique fingerprint for this target."""
        return TargetFingerprint(
            target_type=self.target_type,
            config_hash=self._config_hash,
            version="1.0",
        )

    def serialize_candidate(self, candidate: SystemPromptCandidate) -> dict[str, Any]:
        """Serialize a system prompt candidate for storage.

        Args:
            candidate: System prompt candidate to serialize

        Returns:
            Dictionary representation
        """
        return candidate.to_dict()

    def deserialize_candidate(self, data: dict[str, Any]) -> SystemPromptCandidate:
        """Deserialize a system prompt candidate from storage.

        Args:
            data: Dictionary from serialize_candidate

        Returns:
            SystemPromptCandidate instance

        Raises:
            ValueError: If data is invalid
        """
        return SystemPromptCandidate.from_dict(data)

    def apply_candidate_to_config(
        self,
        config: "ExperimentConfig",
        candidate: SystemPromptCandidate,
    ) -> "ExperimentConfig":
        """Apply a system prompt candidate to an experiment configuration.

        CRITICAL: This method ONLY modifies the support profile settings
        stored in the config's settings. It does NOT modify skill content
        or other profile dimensions.

        Note: System prompt configuration is applied via the support profile
        which is referenced in the experiment's settings. This target
        stores the system prompt candidate in config.settings['system_prompt'].

        Args:
            config: Base experiment configuration
            candidate: System prompt candidate to apply

        Returns:
            New ExperimentConfig with system prompt settings applied
        """
        from bench.experiments.config import ExperimentConfig

        # Create a deep copy of the config
        config_dict = config.model_dump(mode="json")

        # ONLY modify settings.system_prompt - preserve everything else
        if "settings" not in config_dict:
            config_dict["settings"] = {}

        # Store system prompt configuration
        # This will be picked up by the support profile during composition
        config_dict["settings"]["system_prompt"] = {
            "enabled": candidate.enabled,
            "base_prompt": candidate.base_prompt,
            "injection_point": candidate.injection_point.value,
            "include_skill_metadata": candidate.include_skill_metadata,
            "prelude": candidate.prelude,
            "postlude": candidate.postlude,
        }

        return ExperimentConfig.from_dict(config_dict)

    def get_current_value(self, config: "ExperimentConfig") -> SystemPromptCandidate:
        """Extract current system prompt configuration from config.

        Args:
            config: Experiment configuration

        Returns:
            SystemPromptCandidate with current settings
        """
        settings = config.settings or {}
        sp_settings = settings.get("system_prompt", {})

        if sp_settings:
            return SystemPromptCandidate.from_dict(sp_settings)

        # Return default if not configured
        return SystemPromptCandidate()

    def get_param_space(self) -> ParamSpace:
        """Define the searchable parameter space for system prompt optimization.

        The parameter space includes:
        - enabled: Whether to modify system prompt
        - base_prompt: Base prompt text
        - injection_point: Where to inject skill content
        - include_skill_metadata: Include skill headers
        - prelude/postlude: Wrapping text
        """
        space = ParamSpace()

        # Enabled flag
        space.add_param(
            ParamSpec(
                name="enabled",
                param_type=ParamType.BOOLEAN,
                description="Whether system prompt modification is enabled",
                default=True,
                required=True,
            )
        )

        # Base prompt
        space.add_param(
            ParamSpec(
                name="base_prompt",
                param_type=ParamType.TEXT,
                description="Base system prompt text",
                required=False,
                max_value=self.max_base_prompt_length,
            )
        )

        # Injection point
        space.add_param(
            ParamSpec(
                name="injection_point",
                param_type=ParamType.CATEGORICAL,
                description="Where to inject skill content",
                choices=[ip.value for ip in self.allowed_injection_points],
                default=InjectionPoint.AFTER_CONTEXT.value,
                required=True,
            )
        )

        # Include skill metadata
        space.add_param(
            ParamSpec(
                name="include_skill_metadata",
                param_type=ParamType.BOOLEAN,
                description="Include skill names/headers in prompt",
                default=True,
                required=True,
            )
        )

        # Prelude
        space.add_param(
            ParamSpec(
                name="prelude",
                param_type=ParamType.TEXT,
                description="Text to prepend to all prompts",
                required=False,
                max_value=self.max_prelude_length,
            )
        )

        # Postlude
        space.add_param(
            ParamSpec(
                name="postlude",
                param_type=ParamType.TEXT,
                description="Text to append to all prompts",
                required=False,
                max_value=self.max_postlude_length,
            )
        )

        # Add global constraints
        space.constraints = [
            "If enabled=False, other settings have no effect",
            "base_prompt length must be <= max_base_prompt_length",
        ]

        return space

    def validate_candidate(self, candidate: SystemPromptCandidate) -> tuple[bool, list[str]]:
        """Validate a system prompt candidate.

        Args:
            candidate: Candidate to validate

        Returns:
            Tuple of (is_valid, error messages)
        """
        errors: list[str] = []

        # Check injection point is allowed
        if candidate.injection_point not in self.allowed_injection_points:
            errors.append(f"Injection point '{candidate.injection_point.value}' not allowed")

        # Check base prompt length
        if candidate.base_prompt and len(candidate.base_prompt) > self.max_base_prompt_length:
            errors.append(
                f"Base prompt too long: {len(candidate.base_prompt)} > {self.max_base_prompt_length}"
            )

        # Check prelude length
        if candidate.prelude and len(candidate.prelude) > self.max_prelude_length:
            errors.append(f"Prelude too long: {len(candidate.prelude)} > {self.max_prelude_length}")

        # Check postlude length
        if candidate.postlude and len(candidate.postlude) > self.max_postlude_length:
            errors.append(
                f"Postlude too long: {len(candidate.postlude)} > {self.max_postlude_length}"
            )

        return len(errors) == 0, errors


def create_default_system_prompt_candidate() -> SystemPromptCandidate:
    """Create a default system prompt candidate.

    Returns:
        SystemPromptCandidate with sensible defaults
    """
    return SystemPromptCandidate(
        enabled=True,
        base_prompt=None,
        injection_point=InjectionPoint.AFTER_CONTEXT,
        include_skill_metadata=True,
    )
