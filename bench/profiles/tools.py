"""Tool profile schemas for benchmark tool configurations.

Tool profiles define:
- Tool name, version, and description
- Tool configuration parameters
- Enabled/disabled flags
- Variant definitions (patch_v1, patch_v2, etc.)

Profiles are versioned and fingerprinted for reproducibility.
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ToolVersion(str, Enum):
    """Standard tool version identifiers."""

    V1 = "v1"
    PATCH_V1 = "patch_v1"
    PATCH_V2 = "patch_v2"
    PATCH_V3 = "patch_v3"
    CUSTOM = "custom"


class ToolConfig(BaseModel):
    """Configuration parameters for a tool.

    This is a flexible configuration object that can hold
    any tool-specific parameters.
    """

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration parameters",
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the tool",
    )
    mounts: list[str] = Field(
        default_factory=list,
        description="Volume mounts for containerized tools",
    )
    command: str | None = Field(
        default=None,
        description="Override command to run the tool",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Additional command-line arguments",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)


class ToolProfile(BaseModel):
    """
    Canonical tool profile schema for benchmark configuration.

    Tool profiles define how tools are configured and executed
    within the benchmark framework. Each tool has:
    - A unique name and version
    - Configuration parameters
    - Enable/disable flags
    - A fingerprint for change detection
    """

    # Schema versioning
    schema_version: str = Field(
        default="1.0",
        description="Schema version",
    )

    # Tool identification
    tool_id: str = Field(
        description="Unique identifier for the tool (e.g., 'r-cli', 'python-pytest')",
    )
    name: str = Field(
        description="Human-readable tool name",
    )
    version: ToolVersion | str = Field(
        default=ToolVersion.V1,
        description="Tool version identifier",
    )
    description: str = Field(
        default="",
        description="Tool description",
    )

    # Configuration
    config: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Tool configuration parameters",
    )

    # State
    enabled: bool = Field(
        default=True,
        description="Whether the tool is enabled",
    )

    # Variant tracking
    variant: str | None = Field(
        default=None,
        description="Variant name (e.g., 'strict', 'lenient')",
    )
    base_version: str | None = Field(
        default=None,
        description="Base version this variant derives from",
    )

    # Metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the profile was created",
    )
    updated_at: str | None = Field(
        default=None,
        description="When the profile was last updated",
    )

    # Fingerprint cache
    _fingerprint: str | None = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Ensure schema version is valid."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported schema version: {v}. Expected 1.x")
        return v

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> ToolVersion | str:
        """Convert version string to enum or keep as custom string."""
        if isinstance(v, ToolVersion):
            return v
        if isinstance(v, str):
            try:
                return ToolVersion(v)
            except ValueError:
                # Custom version string
                return v
        raise ValueError(f"Invalid version: {v}")

    @model_validator(mode="after")
    def compute_fingerprint(self) -> "ToolProfile":
        """Compute and cache the fingerprint."""
        self._fingerprint = self._compute_fingerprint()
        return self

    def _compute_fingerprint(self) -> str:
        """
        Compute a fingerprint hash of the tool definition.

        The fingerprint changes when:
        - Tool name changes
        - Tool version changes
        - Tool configuration changes

        Returns:
            SHA256 hash of the tool definition
        """
        # Create a canonical representation for hashing
        fingerprint_data = {
            "tool_id": self.tool_id,
            "name": self.name,
            "version": self.version.value
            if isinstance(self.version, ToolVersion)
            else self.version,
            "config": self.config.to_dict(),
            "enabled": self.enabled,
            "variant": self.variant,
        }

        # Sort keys for consistent ordering
        canonical = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        """Get the fingerprint hash of this tool profile."""
        if self._fingerprint is None:
            self._fingerprint = self._compute_fingerprint()
        return self._fingerprint

    def get_version_string(self) -> str:
        """Get the version as a string."""
        if isinstance(self.version, ToolVersion):
            return self.version.value
        return self.version

    def get_full_id(self) -> str:
        """Get the full tool ID including version.

        Format: {tool_id}@{version}[:{variant}]
        """
        base = f"{self.tool_id}@{self.get_version_string()}"
        if self.variant:
            return f"{base}:{self.variant}"
        return base

    def is_variant_of(self, version: str) -> bool:
        """Check if this profile is a variant of the given version."""
        if self.base_version:
            return self.base_version == version
        return self.get_version_string() == version

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = self.model_dump(exclude_none=True)
        data["fingerprint"] = self.fingerprint
        return data

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ToolProfile":
        """Load a tool profile from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            ToolProfile instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValidationError: If the YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tool profile not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolProfile":
        """Create a tool profile from a dictionary.

        Args:
            data: Dictionary with tool profile data

        Returns:
            ToolProfile instance
        """
        return cls.model_validate(data)

    def create_variant(
        self,
        variant_name: str,
        config_overrides: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> "ToolProfile":
        """Create a variant of this tool profile.

        Args:
            variant_name: Name for the variant
            config_overrides: Configuration overrides
            description: Optional description override

        Returns:
            New ToolProfile instance as a variant
        """
        # Start with current config
        new_config = self.config.model_copy(deep=True)

        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
                else:
                    new_config.parameters[key] = value

        return ToolProfile(
            tool_id=self.tool_id,
            name=self.name,
            version=self.version,
            description=description or f"{self.name} - {variant_name} variant",
            config=new_config,
            enabled=self.enabled,
            variant=variant_name,
            base_version=self.get_version_string(),
            tags=self.tags.copy(),
        )


def validate_tool_config(data: dict[str, Any]) -> ToolConfig:
    """
    Validate tool configuration data.

    Args:
        data: Raw tool configuration data

    Returns:
        Validated ToolConfig instance

    Raises:
        ValidationError: If data doesn't conform to schema
    """
    return ToolConfig.model_validate(data)


def load_tool_profile(path: str | Path) -> ToolProfile:
    """
    Load and validate a tool profile from a file.

    Args:
        path: Path to the tool profile file (YAML)

    Returns:
        Validated ToolProfile instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If profile is invalid
    """
    return ToolProfile.from_yaml(path)


def create_default_tool_profile(tool_id: str, name: str) -> ToolProfile:
    """
    Create a default tool profile with sensible defaults.

    Args:
        tool_id: Unique tool identifier
        name: Human-readable tool name

    Returns:
        ToolProfile with default configuration
    """
    return ToolProfile(
        tool_id=tool_id,
        name=name,
        version=ToolVersion.V1,
        description=f"Default configuration for {name}",
        config=ToolConfig(),
        enabled=True,
        tags=["default"],
    )
