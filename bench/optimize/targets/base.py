"""Base interface for optimizable targets.

This module defines the OptimizableTarget interface - the core abstraction
for any component that can be optimized in the benchmark system.

Optimization Contract:
----------------------
Each target operates within a contract that defines:

1. Objective Metric: What to optimize (e.g., pass_rate, avg_score)
2. Constraints: What must remain valid (e.g., non-empty, parseable)
3. Tie-breaks: How to choose between equal scores (e.g., shorter, simpler)
4. Stop Rule: When to halt optimization (e.g., max_iterations, convergence)

Target Isolation:
----------------
Each target mutates ONLY its intended profile dimension:
- SkillTarget: Only modifies skill text in support profile
- SystemPromptTarget: Only modifies system prompt config
- ToolPolicyTarget: Only modifies tool selection/availability

This isolation ensures optimization in one dimension doesn't accidentally
affect others.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from bench.experiments.config import ExperimentConfig


class ParamType(str, Enum):
    """Types of parameters in the search space."""

    TEXT = "text"
    """Free-form text content (e.g., skill markdown)."""

    CATEGORICAL = "categorical"
    """One of a fixed set of options (e.g., injection_point)."""

    BOOLEAN = "boolean"
    """True/False value."""

    INTEGER = "integer"
    """Integer with optional bounds."""

    FLOAT = "float"
    """Float with optional bounds."""

    LIST = "list"
    """List of items (e.g., enabled tools)."""


@dataclass
class ParamSpec:
    """Specification for a single parameter in the search space."""

    name: str
    """Parameter name."""

    param_type: ParamType
    """Type of parameter."""

    description: str = ""
    """Human-readable description."""

    # Type-specific constraints
    choices: list[str] | None = None
    """For CATEGORICAL: valid choices."""

    min_value: int | float | None = None
    """For INTEGER/FLOAT: minimum value."""

    max_value: int | float | None = None
    """For INTEGER/FLOAT: maximum value."""

    default: Any = None
    """Default value for the parameter."""

    item_type: ParamType | None = None
    """For LIST: type of list items."""

    required: bool = True
    """Whether parameter must have a value."""

    mutable: bool = True
    """Whether this parameter can be modified during optimization."""

    def validate_value(self, value: Any) -> bool:
        """Validate a value against this specification.

        Args:
            value: Value to validate

        Returns:
            True if value is valid for this parameter
        """
        if value is None:
            return not self.required

        if self.param_type == ParamType.TEXT:
            return isinstance(value, str)

        elif self.param_type == ParamType.CATEGORICAL:
            return value in (self.choices or [])

        elif self.param_type == ParamType.BOOLEAN:
            return isinstance(value, bool)

        elif self.param_type == ParamType.INTEGER:
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            return not (self.max_value is not None and value > self.max_value)

        elif self.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            return not (self.max_value is not None and value > self.max_value)

        elif self.param_type == ParamType.LIST:
            return isinstance(value, list)

        return False


@dataclass
class ParamSpace:
    """Definition of the searchable parameter space for a target.

    The parameter space defines all dimensions that can be explored
    during optimization. Each dimension has constraints that define
    valid values.
    """

    params: dict[str, ParamSpec] = field(default_factory=dict)
    """Parameter specifications keyed by parameter name."""

    constraints: list[str] = field(default_factory=list)
    """Global constraints that apply across parameters."""

    def get_param(self, name: str) -> ParamSpec | None:
        """Get a parameter specification by name."""
        return self.params.get(name)

    def add_param(self, spec: ParamSpec) -> "ParamSpace":
        """Add a parameter specification.

        Args:
            spec: Parameter specification to add

        Returns:
            Self for chaining
        """
        self.params[spec.name] = spec
        return self

    def validate_candidate(self, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a candidate against the parameter space.

        Args:
            candidate: Dictionary of parameter values

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        # Check all required params are present
        for name, spec in self.params.items():
            if spec.required and name not in candidate:
                errors.append(f"Missing required parameter: {name}")

        # Validate each provided value
        for name, value in candidate.items():
            if name not in self.params:
                errors.append(f"Unknown parameter: {name}")
                continue

            spec = self.params[name]
            if not spec.validate_value(value):
                errors.append(f"Invalid value for {name}: {value!r}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "params": {
                name: {
                    "name": spec.name,
                    "param_type": spec.param_type.value,
                    "description": spec.description,
                    "choices": spec.choices,
                    "min_value": spec.min_value,
                    "max_value": spec.max_value,
                    "default": spec.default,
                    "item_type": spec.item_type.value if spec.item_type else None,
                    "required": spec.required,
                    "mutable": spec.mutable,
                }
                for name, spec in self.params.items()
            },
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParamSpace":
        """Deserialize from dictionary."""
        params = {}
        for name, spec_data in data.get("params", {}).items():
            spec = ParamSpec(
                name=spec_data["name"],
                param_type=ParamType(spec_data["param_type"]),
                description=spec_data.get("description", ""),
                choices=spec_data.get("choices"),
                min_value=spec_data.get("min_value"),
                max_value=spec_data.get("max_value"),
                default=spec_data.get("default"),
                item_type=ParamType(spec_data["item_type"]) if spec_data.get("item_type") else None,
                required=spec_data.get("required", True),
                mutable=spec_data.get("mutable", True),
            )
            params[name] = spec

        return cls(
            params=params,
            constraints=data.get("constraints", []),
        )


class CandidateType(Protocol):
    """Protocol for candidate values.

    Candidates must be serializable to/from dictionaries for storage
    and transfer between optimization iterations.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateType":
        """Deserialize candidate from dictionary."""
        ...


@dataclass(frozen=True)
class TargetFingerprint:
    """Unique identifier for an optimization target.

    The fingerprint captures the target type and any configuration
    that affects what constitutes a valid candidate. Two targets
    with the same fingerprint can share optimization results.
    """

    target_type: str
    """Type of target (e.g., 'skill', 'system_prompt', 'tool_policy')."""

    config_hash: str
    """Hash of target-specific configuration."""

    version: str = "1.0"
    """Version of the target interface."""

    def to_string(self) -> str:
        """Get a compact string representation."""
        return f"{self.target_type}:{self.config_hash}:{self.version}"

    @classmethod
    def from_string(cls, s: str) -> "TargetFingerprint":
        """Parse from string representation."""
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid fingerprint string: {s}")
        return cls(
            target_type=parts[0],
            config_hash=parts[1],
            version=parts[2],
        )


# Type variable for candidate types
CandidateT = TypeVar("CandidateT")


class OptimizableTarget(ABC, Generic[CandidateT]):
    """
    Interface for any component that can be optimized.

    This is the core abstraction that allows the optimization framework
    to work with different types of optimizable components (skills,
    system prompts, tool policies) through a unified interface.

    Target Isolation Principle:
    ---------------------------
    Each target implementation MUST ensure that applying a candidate
    only mutates its intended profile dimension. For example:

    - SkillTarget: Only modifies skill.content or skill.path
    - SystemPromptTarget: Only modifies system_prompt config
    - ToolPolicyTarget: Only modifies tool availability/priorities

    Implementation Requirements:
    ---------------------------
    1. Implement all abstract methods
    2. Ensure serialize/deserialize are inverse operations
    3. Ensure apply_candidate_to_config only modifies intended dimension
    4. Provide meaningful fingerprints for caching/sharing
    """

    @property
    @abstractmethod
    def target_type(self) -> str:
        """Get the type identifier for this target.

        Returns:
            String identifier (e.g., 'skill', 'system_prompt', 'tool_policy')
        """
        pass

    @abstractmethod
    def fingerprint(self) -> TargetFingerprint:
        """Get a unique fingerprint for this target.

        The fingerprint should capture any configuration that affects
        what constitutes a valid candidate or how candidates are evaluated.

        Returns:
            TargetFingerprint uniquely identifying this target
        """
        pass

    @abstractmethod
    def serialize_candidate(self, candidate: CandidateT) -> dict[str, Any]:
        """Serialize a candidate for storage.

        Args:
            candidate: Candidate value to serialize

        Returns:
            Dictionary representation suitable for JSON storage
        """
        pass

    @abstractmethod
    def deserialize_candidate(self, data: dict[str, Any]) -> CandidateT:
        """Deserialize a candidate from storage.

        Args:
            data: Dictionary representation from serialize_candidate

        Returns:
            Candidate value

        Raises:
            ValueError: If data cannot be deserialized
        """
        pass

    @abstractmethod
    def apply_candidate_to_config(
        self,
        config: "ExperimentConfig",
        candidate: CandidateT,
    ) -> "ExperimentConfig":
        """Apply a candidate to an experiment configuration.

        CRITICAL: This method MUST only modify the intended profile dimension.
        All other aspects of the config must remain unchanged.

        Args:
            config: Base experiment configuration
            candidate: Candidate value to apply

        Returns:
            New ExperimentConfig with candidate applied
        """
        pass

    @abstractmethod
    def get_current_value(self, config: "ExperimentConfig") -> CandidateT:
        """Get the current baseline value from a configuration.

        Args:
            config: Experiment configuration to extract from

        Returns:
            Current candidate value from the config
        """
        pass

    @abstractmethod
    def get_param_space(self) -> ParamSpace:
        """Define the searchable parameter space.

        Returns:
            ParamSpace describing all searchable dimensions
        """
        pass

    def validate_candidate(self, candidate: CandidateT) -> tuple[bool, list[str]]:
        """Validate a candidate against constraints.

        Args:
            candidate: Candidate to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        # Default implementation serializes and validates against param space
        data = self.serialize_candidate(candidate)
        return self.get_param_space().validate_candidate(data)

    def compute_candidate_hash(self, candidate: CandidateT) -> str:
        """Compute a hash for a candidate value.

        Args:
            candidate: Candidate to hash

        Returns:
            SHA256 hash of the candidate (first 16 chars)
        """
        import json

        data = self.serialize_candidate(candidate)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()[:16]


# Import ExperimentConfig for type hints (avoid circular import)
def _get_experiment_config_class() -> type:
    """Lazily import ExperimentConfig to avoid circular imports."""
    from bench.experiments.config import ExperimentConfig

    return ExperimentConfig


# Update the type hints at runtime
# This allows the type hints to work without circular imports at module load
__all__ = [
    "CandidateT",
    "CandidateType",
    "OptimizableTarget",
    "ParamSpace",
    "ParamSpec",
    "ParamType",
    "TargetFingerprint",
]
