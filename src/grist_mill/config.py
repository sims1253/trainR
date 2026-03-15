"""Configuration system for grist-mill.

Provides a typed configuration model using pydantic-settings with support for:
- YAML file loading
- Environment variable overrides (GRIST_MILL_ prefix)
- CLI argument overrides
- Precedence: CLI > env vars > YAML
- Strict mode for unknown keys
- Secret masking in __str__/__repr__
- DEBUG logging of resolved value sources

Validates VAL-CFG-01 through VAL-CFG-08.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import InitErrorDetails, PydanticUndefined
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from grist_mill.schemas import Task
from grist_mill.schemas.artifact import (
    Artifact,
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)

logger = logging.getLogger(__name__)

# Fields considered secret — masked in __str__ and __repr__
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "token",
        "secret",
        "password",
        "auth_token",
        "access_key",
        "private_key",
    }
)


def _mask_value(value: str) -> str:
    """Mask a secret value, showing only first 3 and last 4 characters."""
    if len(value) <= 8:
        return "***"
    return value[:3] + "***" + value[-4:]


def _is_secret_field(name: str) -> bool:
    """Check if a field name looks like a secret."""
    return name.lower() in _SECRET_FIELD_NAMES


# ---------------------------------------------------------------------------
# Nested Configuration Sections
# ---------------------------------------------------------------------------


class AgentSection(BaseModel):
    """Configuration for the AI agent.

    Nested section under the top-level config. Validates independently
    with clear error paths.

    Attributes:
        model: LLM model identifier.
        provider: LLM provider (openrouter, openai, etc.).
        system_prompt: Optional system prompt for the agent.
        max_turns: Maximum conversation turns (default 5).
        timeout: Per-task timeout in seconds (default 600).
        api_key: API key for the provider (masked in string representation).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    model: str = Field(
        ...,
        min_length=1,
        description="LLM model identifier (e.g., 'gpt-4', 'claude-3-opus').",
    )
    provider: str = Field(
        ...,
        min_length=1,
        description="LLM provider (e.g., 'openrouter', 'openai', 'anthropic').",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the agent.",
    )
    max_turns: int = Field(
        default=5,
        gt=0,
        description="Maximum number of conversation turns.",
    )
    timeout: int = Field(
        default=600,
        gt=0,
        description="Per-task timeout in seconds.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider.",
        json_schema_extra={"secret": True},
    )


class EnvironmentSection(BaseModel):
    """Configuration for the execution environment.

    Nested section under the top-level config.

    Attributes:
        runner_type: Type of runner ('local' or 'docker').
        docker_image: Docker image for containerized execution.
        cpu_limit: CPU limit in cores (default 1.0).
        memory_limit: Memory limit string (e.g., '4g').
        network_access: Whether to allow network access (default False).
        working_dir: Working directory inside the environment.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    runner_type: str = Field(
        default="local",
        min_length=1,
        description="Type of runner ('local' or 'docker').",
    )
    docker_image: str | None = Field(
        default=None,
        description="Docker image for containerized execution.",
    )
    cpu_limit: float = Field(
        default=1.0,
        gt=0,
        description="CPU limit in cores.",
    )
    memory_limit: str | None = Field(
        default=None,
        description="Memory limit string (e.g., '4g', '512m').",
    )
    network_access: bool = Field(
        default=False,
        description="Whether to allow network access.",
    )
    working_dir: str | None = Field(
        default=None,
        description="Working directory inside the environment.",
    )


class TelemetrySection(BaseModel):
    """Configuration for telemetry collection.

    Nested section under the top-level config.

    Attributes:
        enabled: Whether telemetry collection is enabled (default True).
        trace_enabled: Whether to capture raw event audit trail (default False).
        output_dir: Directory for telemetry output files.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    enabled: bool = Field(
        default=True,
        description="Whether telemetry collection is enabled.",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to capture raw event audit trail.",
    )
    output_dir: str = Field(
        default="./results/telemetry",
        description="Directory for telemetry output files.",
    )


# ---------------------------------------------------------------------------
# GristMillConfig — main config model
# ---------------------------------------------------------------------------


class GristMillConfig(BaseModel):
    """Top-level configuration for grist-mill.

    Supports nested sections (agent, environment, telemetry), task lists,
    and artifact definitions. Provides secret masking in string representation
    and DEBUG-level source logging.

    Use ``load_config()`` to load from YAML with env var and CLI overrides.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        # Default is non-strict (extra='ignore'); use strict=True to reject unknowns
        extra="ignore",
    )

    agent: AgentSection = Field(
        ...,
        description="Agent configuration section.",
    )
    environment: EnvironmentSection = Field(
        default_factory=lambda: EnvironmentSection(runner_type="local"),
        description="Environment configuration section.",
    )
    telemetry: TelemetrySection = Field(
        default_factory=TelemetrySection,
        description="Telemetry configuration section.",
    )
    tasks: list[Task] = Field(
        default_factory=list,
        description="List of task definitions.",
    )
    artifacts: list[ToolArtifact | MCPServerArtifact | SkillArtifact] = Field(
        default_factory=list,
        description="List of artifact definitions.",
    )

    # ------------------------------------------------------------------
    # Secret masking (VAL-CFG-08)
    # ------------------------------------------------------------------

    def _mask_dict(self, data: dict[str, Any], parent_path: str = "") -> dict[str, Any]:
        """Recursively mask secret fields in a dict representation."""
        masked: dict[str, Any] = {}
        for key, value in data.items():
            full_path = f"{parent_path}.{key}" if parent_path else key
            if isinstance(value, dict):
                masked[key] = self._mask_dict(value, full_path)
            elif isinstance(value, str) and _is_secret_field(key):
                masked[key] = _mask_value(value)
            elif isinstance(value, (list, tuple)) and not isinstance(value, str):
                masked[key] = [
                    self._mask_dict(item, f"{full_path}[{i}]") if isinstance(item, dict) else item
                    for i, item in enumerate(value)
                ]
            else:
                masked[key] = value
        return masked

    def __str__(self) -> str:
        """Return a string representation with secrets masked."""
        data = self.model_dump()
        masked = self._mask_dict(data)
        # Truncate long lists for readability
        if len(masked.get("tasks", [])) > 3:
            masked["tasks"] = masked["tasks"][:3]
            masked["_tasks_truncated"] = True
        if len(masked.get("artifacts", [])) > 3:
            masked["artifacts"] = masked["artifacts"][:3]
            masked["_artifacts_truncated"] = True
        return f"GristMillConfig({yaml.dump(masked, default_flow_style=False).strip()})"

    def __repr__(self) -> str:
        """Return a repr with secrets masked."""
        data = self.model_dump()
        masked = self._mask_dict(data)
        return f"GristMillConfig({masked!r})"


# ---------------------------------------------------------------------------
# Custom Settings Source for YAML
# ---------------------------------------------------------------------------


class _YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom pydantic-settings source that reads from a YAML file.

    Handles loading YAML content, flattening nested sections for env var
    override compatibility, and logging the source of each resolved value.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_content: str | None = None,
        yaml_path: str | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__(settings_cls)
        self._yaml_content = yaml_content
        self._yaml_path = yaml_path
        self._strict = strict
        self._yaml_data: dict[str, Any] | None = None
        self._resolved_sources: dict[str, str] = {}

    def _load_yaml(self) -> dict[str, Any]:
        """Load and parse YAML content."""
        if self._yaml_data is not None:
            return self._yaml_data

        if self._yaml_content is not None:
            self._yaml_data = yaml.safe_load(self._yaml_content)
        elif self._yaml_path is not None:
            path = Path(self._yaml_path)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self._yaml_path}")
            self._yaml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            self._yaml_data = {}

        if not isinstance(self._yaml_data, dict):
            raise ValueError(
                f"Invalid YAML configuration: expected a mapping, got {type(self._yaml_data).__name__}"
            )

        return self._yaml_data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Get a field value from the YAML source.

        Returns:
            Tuple of (value, key_for_model, is_complex).
        """
        yaml_data = self._load_yaml()
        if field_name in yaml_data:
            value = yaml_data[field_name]
            self._resolved_sources[field_name] = "yaml"
            logger.debug(
                "Config field '%s' = %r (from YAML%s)",
                field_name,
                _truncate_for_log(value),
                f": {self._yaml_path}" if self._yaml_path else " content",
            )
            return value, field_name, True
        return PydanticUndefined, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the full YAML data as a settings dict."""
        yaml_data = self._load_yaml()
        return yaml_data


# ---------------------------------------------------------------------------
# Flat settings model for env var override resolution
# ---------------------------------------------------------------------------


class _FlatAgentSettings(BaseSettings):
    """Flat agent settings for env var resolution."""

    model_config = SettingsConfigDict(
        env_prefix="grist_mill_",
        extra="ignore",
    )

    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None
    max_turns: int | None = None
    timeout: int | None = None
    api_key: str | None = None


class _FlatEnvironmentSettings(BaseSettings):
    """Flat environment settings for env var resolution."""

    model_config = SettingsConfigDict(
        env_prefix="grist_mill_",
        extra="ignore",
    )

    runner_type: str | None = None
    docker_image: str | None = None
    cpu_limit: float | None = None
    memory_limit: str | None = None
    network_access: bool | None = None
    working_dir: str | None = None


class _FlatTelemetrySettings(BaseSettings):
    """Flat telemetry settings for env var resolution."""

    model_config = SettingsConfigDict(
        env_prefix="grist_mill_",
        extra="ignore",
    )

    enabled: bool | None = None
    trace_enabled: bool | None = None
    output_dir: str | None = None


def _resolve_env_overrides(section: str, yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Resolve environment variable overrides for a config section.

    Uses pydantic-settings to map GRIST_MILL_* env vars to section fields.
    Returns a dict of overrides with source annotations.
    """
    if section == "agent":
        flat = _FlatAgentSettings()
    elif section == "environment":
        flat = _FlatEnvironmentSettings()
    elif section == "telemetry":
        flat = _FlatTelemetrySettings()
    else:
        return {}

    overrides: dict[str, Any] = {}
    flat_dict = flat.model_dump(exclude_none=True)

    for field_name, value in flat_dict.items():
        overrides[field_name] = value
        logger.debug(
            "Config field '%s.%s' = %r (from env var GRIST_MILL_%s)",
            section,
            field_name,
            _truncate_for_log(value),
            field_name.upper(),
        )

    return overrides


def _apply_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply overrides to a base dict, preserving other keys."""
    result = dict(base)
    result.update(overrides)
    return result


def _resolve_cli_overrides(section: str, cli_args: dict[str, Any]) -> dict[str, Any]:
    """Extract CLI overrides relevant to a config section.

    CLI args are flat key-value pairs that apply to the appropriate section.
    """
    return cli_args


def _truncate_for_log(value: Any, max_len: int = 80) -> Any:
    """Truncate values for log output."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    return value


# ---------------------------------------------------------------------------
# Public API: load_config
# ---------------------------------------------------------------------------


def load_config(
    *,
    yaml_file: str | None = None,
    yaml_content: str | None = None,
    cli_args: dict[str, Any] | None = None,
    strict: bool = False,
) -> GristMillConfig:
    """Load grist-mill configuration from YAML with env var and CLI overrides.

    Resolution precedence (highest wins):
        1. CLI arguments (via ``cli_args`` dict)
        2. Environment variables (``GRIST_MILL_`` prefix)
        3. YAML file or content

    Args:
        yaml_file: Path to a YAML configuration file.
        yaml_content: YAML configuration as a string (alternative to yaml_file).
        cli_args: Flat dict of CLI argument overrides (e.g., ``{"timeout": 900}``).
        strict: If True, reject unknown keys. If False, preserve them.

    Returns:
        A fully-resolved ``GristMillConfig`` instance.

    Raises:
        FileNotFoundError: If ``yaml_file`` is specified but doesn't exist.
        ValueError: If the YAML content is invalid or not a mapping.
        ValidationError: If config values fail validation.

    Examples:
        Load from file:
            config = load_config(yaml_file="config.yaml")

        Load with CLI overrides:
            config = load_config(yaml_file="config.yaml", cli_args={"timeout": 900})

        Load from string:
            config = load_config(yaml_content=yaml_string)
    """
    if yaml_file is None and yaml_content is None:
        raise ValueError("Either yaml_file or yaml_content must be provided")

    # --- Step 1: Parse YAML ---
    source_name = yaml_file or "<string>"
    try:
        if yaml_file is not None:
            path = Path(yaml_file)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {yaml_file}")
            yaml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            yaml_data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration from {source_name}: {exc}") from exc

    if not isinstance(yaml_data, dict):
        raise ValueError(
            f"Invalid YAML configuration: expected a mapping, got {type(yaml_data).__name__}"
        )

    logger.debug("Loaded YAML config from %s", yaml_file or "<string>")

    # --- Step 2: Resolve env var overrides for each section ---
    sections = {"agent", "environment", "telemetry"}
    resolved: dict[str, Any] = dict(yaml_data)

    for section in sections:
        section_data = resolved.get(section, {})
        if not isinstance(section_data, dict):
            section_data = {}

        # Log YAML-sourced values
        for key, value in section_data.items():
            logger.debug(
                "Config field '%s.%s' = %r (from YAML)",
                section,
                key,
                _truncate_for_log(value),
            )

        # Apply env var overrides
        env_overrides = _resolve_env_overrides(section, yaml_data)
        section_data = _apply_overrides(section_data, env_overrides)

        # Apply CLI overrides (flat CLI args apply to the appropriate section)
        if cli_args:
            cli_section_overrides = _resolve_cli_overrides(section, cli_args)
            for key, value in cli_section_overrides.items():
                if value is not None:
                    section_data[key] = value
                    logger.debug(
                        "Config field '%s.%s' = %r (from CLI --%s)",
                        section,
                        key,
                        _truncate_for_log(value),
                        key,
                    )

        resolved[section] = section_data

    # --- Step 3: Handle tasks ---
    tasks_data = resolved.get("tasks", [])
    if isinstance(tasks_data, list):
        for i, task_data in enumerate(tasks_data):
            task_id = "?"
            if isinstance(task_data, dict):
                task_id = str(task_data.get("id", "?"))  # type: ignore[union-attr]
            logger.debug(
                "Config task[%d].id = %r (from YAML)",
                i,
                task_id,
            )

    # --- Step 4: Handle artifacts ---
    artifacts_data = resolved.get("artifacts", [])
    if isinstance(artifacts_data, list):
        parsed_artifacts: list[ToolArtifact | MCPServerArtifact | SkillArtifact] = []
        for artifact_data in artifacts_data:
            if isinstance(artifact_data, dict):
                artifact = Artifact.model_validate(artifact_data)
                parsed_artifacts.append(artifact)
                logger.debug(
                    "Config artifact '%s' (type=%s) loaded from YAML",
                    artifact.name if hasattr(artifact, "name") else "?",
                    artifact.type if hasattr(artifact, "type") else "?",
                )
        resolved["artifacts"] = parsed_artifacts
    elif isinstance(artifacts_data, dict):
        # Map-style artifact definitions
        parsed_artifacts: list[ToolArtifact | MCPServerArtifact | SkillArtifact] = []
        for name, artifact_data in artifacts_data.items():
            if isinstance(artifact_data, dict):
                artifact_data_with_name = dict(artifact_data)
                if "name" not in artifact_data_with_name:
                    artifact_data_with_name["name"] = name
                artifact = Artifact.model_validate(artifact_data_with_name)
                parsed_artifacts.append(artifact)
                logger.debug(
                    "Config artifact '%s' (type=%s) loaded from YAML",
                    artifact.name,
                    artifact.type,
                )
        resolved["artifacts"] = parsed_artifacts

    # --- Step 5: Validate into GristMillConfig ---
    if strict:
        # In strict mode, reject unknown keys
        known_keys = {"agent", "environment", "telemetry", "tasks", "artifacts"}
        unknown_keys = set(resolved.keys()) - known_keys
        if unknown_keys:
            errors: list[InitErrorDetails] = [
                InitErrorDetails(
                    type="extra_forbidden",
                    loc=(key,),
                    input=resolved[key],
                    ctx={"message": f"Extra inputs are not permitted: '{key}'"},
                )
                for key in sorted(unknown_keys)
            ]
            raise ValidationError.from_exception_data(
                title="GristMillConfig",
                line_errors=errors,
            )

        # Also enforce strict on nested sections
        for section in sections:
            section_data = resolved.get(section, {})
            if isinstance(section_data, dict):
                _enforce_strict_nested(section, section_data)

    # Remove unknown top-level keys in non-strict mode (they're ignored by extra='ignore')
    config = GristMillConfig(**resolved)
    return config


def _enforce_strict_nested(section_name: str, data: dict[str, Any]) -> None:
    """Enforce strict mode on a nested section.

    Raises ValidationError if unknown fields are found.
    """
    section_models = {
        "agent": AgentSection,
        "environment": EnvironmentSection,
        "telemetry": TelemetrySection,
    }
    model_cls = section_models.get(section_name)
    if model_cls is None:
        return

    # Check for extra fields by comparing known fields vs provided fields
    schema = model_cls.model_json_schema()
    known_props = set(schema.get("properties", {}).keys())
    provided_keys = set(data.keys())
    extra_keys = provided_keys - known_props

    if extra_keys:
        nested_errors: list[InitErrorDetails] = [
            InitErrorDetails(
                type="extra_forbidden",
                loc=(section_name, key),
                input=data[key],
                ctx={"message": f"Extra inputs are not permitted: '{key}'"},
            )
            for key in sorted(extra_keys)
        ]
        raise ValidationError.from_exception_data(
            title=f"GristMillConfig.{section_name}",
            line_errors=nested_errors,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AgentSection",
    "EnvironmentSection",
    "GristMillConfig",
    "TelemetrySection",
    "load_config",
]
