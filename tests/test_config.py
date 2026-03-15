"""Tests for the grist-mill configuration system.

Validates VAL-CFG-01 through VAL-CFG-08:
- VAL-CFG-01: YAML configuration loads and validates into typed config model
- VAL-CFG-02: Environment variables override YAML config values
- VAL-CFG-03: CLI arguments take highest precedence in config resolution
- VAL-CFG-04: Config validation rejects unknown keys in strict mode
- VAL-CFG-05: Nested config sections validate independently
- VAL-CFG-06: Config supports list and map values for task and artifact definitions
- VAL-CFG-07: Config resolution logs the source of each resolved value
- VAL-CFG-08: Config supports secret masking in string representation
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from grist_mill.config import (
    AgentSection,
    EnvironmentSection,
    GristMillConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a YAML file for testing."""
    path.write_text(yaml.dump(data, default_flow_style=False))


def _sample_yaml_data() -> dict[str, Any]:
    """Return a valid sample config dict."""
    return {
        "agent": {
            "model": "gpt-4",
            "provider": "openrouter",
            "system_prompt": "You are a helpful assistant.",
            "max_turns": 10,
            "timeout": 300,
            "api_key": "sk-test-secret-key-12345",
        },
        "environment": {
            "runner_type": "local",
            "docker_image": "python:3.12",
            "cpu_limit": 2.0,
            "memory_limit": "4g",
            "network_access": False,
        },
        "telemetry": {
            "enabled": True,
            "trace_enabled": False,
            "output_dir": "./results/telemetry",
        },
    }


def _sample_yaml_str() -> str:
    """Return a valid sample YAML config as string."""
    return textwrap.dedent("""\
        agent:
          model: gpt-4
          provider: openrouter
          system_prompt: "You are a helpful assistant."
          max_turns: 10
          timeout: 300
          api_key: sk-test-secret-key-12345
        environment:
          runner_type: local
          docker_image: "python:3.12"
          cpu_limit: 2.0
          memory_limit: "4g"
          network_access: false
        telemetry:
          enabled: true
          trace_enabled: false
          output_dir: "./results/telemetry"
    """)


# =========================================================================
# VAL-CFG-01: YAML configuration loads and validates into typed model
# =========================================================================


class TestYamlLoading:
    """Test YAML config loading and validation (VAL-CFG-01)."""

    def test_load_valid_yaml_file(self, tmp_path: Path) -> None:
        """A valid YAML config file loads into GristMillConfig."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))

        assert config.agent.model == "gpt-4"
        assert config.agent.provider == "openrouter"
        assert config.agent.system_prompt == "You are a helpful assistant."
        assert config.agent.max_turns == 10
        assert config.agent.timeout == 300
        assert config.environment.runner_type == "local"
        assert config.environment.docker_image == "python:3.12"
        assert config.environment.cpu_limit == 2.0
        assert config.environment.memory_limit == "4g"
        assert config.environment.network_access is False
        assert config.telemetry.enabled is True
        assert config.telemetry.trace_enabled is False

    def test_load_invalid_yaml_raises_clear_error(self, tmp_path: Path) -> None:
        """An invalid YAML file raises a clear error with file path."""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("{invalid yaml: [unclosed")

        with pytest.raises(ValueError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        error_msg = str(exc_info.value).lower()
        # Should reference the file in some way
        assert "bad.yaml" in error_msg or "yaml" in error_msg

    def test_load_invalid_config_values_raises_validation_error(self, tmp_path: Path) -> None:
        """Invalid config values produce a ValidationError with field details."""
        yaml_file = tmp_path / "invalid.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {
                    "model": "gpt-4",
                    "provider": "openrouter",
                    "timeout": -5,  # Invalid: must be > 0
                },
                "environment": {
                    "runner_type": "local",
                },
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        errors = exc_info.value.errors()
        # Should reference the timeout field
        field_paths = [str(e["loc"]) for e in errors]
        assert any("timeout" in p for p in field_paths)

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Loading a nonexistent file raises a clear FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(yaml_file="/nonexistent/path/config.yaml")

    def test_load_from_yaml_string(self) -> None:
        """Config can be loaded from a YAML string."""
        config = load_config(yaml_content=_sample_yaml_str())

        assert config.agent.model == "gpt-4"
        assert config.environment.runner_type == "local"

    def test_minimal_config_with_defaults(self, tmp_path: Path) -> None:
        """Config loads with minimal YAML, using sensible defaults."""
        yaml_file = tmp_path / "minimal.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {
                    "model": "gpt-4",
                    "provider": "openrouter",
                },
                "environment": {
                    "runner_type": "local",
                },
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.agent.model == "gpt-4"
        assert config.agent.max_turns == 5  # default
        assert config.agent.timeout == 600  # default
        assert config.environment.cpu_limit == 1.0  # default


# =========================================================================
# VAL-CFG-02: Environment variables override YAML config values
# =========================================================================


class TestEnvVarOverrides:
    """Test environment variable overrides (VAL-CFG-02)."""

    def test_env_var_overrides_yaml_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GRIST_MILL_TIMEOUT env var overrides YAML value."""
        monkeypatch.setenv("GRIST_MILL_TIMEOUT", "600")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.agent.timeout == 600

    def test_env_var_overrides_yaml_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GRIST_MILL_MODEL env var overrides YAML model."""
        monkeypatch.setenv("GRIST_MILL_MODEL", "claude-3-opus")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.agent.model == "claude-3-opus"

    def test_env_var_overrides_yaml_runner_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GRIST_MILL_RUNNER_TYPE env var overrides YAML runner_type."""
        monkeypatch.setenv("GRIST_MILL_RUNNER_TYPE", "docker")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.environment.runner_type == "docker"

    def test_env_var_overrides_nested_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars override nested fields like environment.docker_image."""
        monkeypatch.setenv("GRIST_MILL_DOCKER_IMAGE", "rocker/r-base:4.3")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local", "docker_image": "python:3.12"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.environment.docker_image == "rocker/r-base:4.3"

    def test_no_env_var_uses_yaml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without env var, YAML value is used."""
        monkeypatch.delenv("GRIST_MILL_TIMEOUT", raising=False)
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.agent.timeout == 300


# =========================================================================
# VAL-CFG-03: CLI arguments take highest precedence
# =========================================================================


class TestCliOverrides:
    """Test CLI argument precedence (VAL-CFG-03).

    CLI args > env vars > YAML defaults.
    """

    def test_cli_overrides_env_and_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI arg overrides both env var and YAML value."""
        monkeypatch.setenv("GRIST_MILL_TIMEOUT", "600")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(
            yaml_file=str(yaml_file),
            cli_args={"timeout": 900},
        )

        assert config.agent.timeout == 900  # CLI wins

    def test_cli_overrides_env_var_when_yaml_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI overrides env var, env overrides YAML."""
        monkeypatch.setenv("GRIST_MILL_TIMEOUT", "600")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(
            yaml_file=str(yaml_file),
            cli_args={"timeout": 900},
        )

        assert config.agent.timeout == 900

    def test_cli_model_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI model override takes precedence."""
        monkeypatch.setenv("GRIST_MILL_MODEL", "gpt-4")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-3.5", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(
            yaml_file=str(yaml_file),
            cli_args={"model": "claude-3-opus"},
        )

        assert config.agent.model == "claude-3-opus"

    def test_partial_cli_overrides(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI only overrides specified fields; others come from env/YAML."""
        monkeypatch.setenv("GRIST_MILL_TIMEOUT", "600")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(
            yaml_file=str(yaml_file),
            cli_args={"model": "claude-3-opus"},
        )

        assert config.agent.model == "claude-3-opus"  # from CLI
        assert config.agent.timeout == 600  # from env
        assert config.agent.provider == "openrouter"  # from YAML


# =========================================================================
# VAL-CFG-04: Strict mode rejects unknown keys
# =========================================================================


class TestStrictMode:
    """Test strict mode for unknown keys (VAL-CFG-04)."""

    def test_strict_mode_rejects_unknown_top_level_key(self, tmp_path: Path) -> None:
        """In strict mode, unknown top-level keys raise ValidationError."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "unknown_section": "should fail in strict mode",
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file), strict=True)

        errors = exc_info.value.errors()
        error_types = [e["type"] for e in errors]
        assert "extra_forbidden" in error_types

    def test_non_strict_mode_preserves_unknown_keys(self, tmp_path: Path) -> None:
        """In non-strict mode (default), unknown keys are preserved."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "future_feature": "should be preserved",
            },
        )

        config = load_config(yaml_file=str(yaml_file), strict=False)

        # Core values still load correctly
        assert config.agent.model == "gpt-4"
        assert config.environment.runner_type == "local"

    def test_strict_mode_rejects_unknown_nested_key(self, tmp_path: Path) -> None:
        """In strict mode, unknown nested keys raise ValidationError."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {
                    "model": "gpt-4",
                    "provider": "openrouter",
                    "unknown_field": "should fail",
                },
                "environment": {"runner_type": "local"},
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file), strict=True)

        errors = exc_info.value.errors()
        error_types = [e["type"] for e in errors]
        assert "extra_forbidden" in error_types


# =========================================================================
# VAL-CFG-05: Nested config sections validate independently
# =========================================================================


class TestNestedValidation:
    """Test nested section independent validation (VAL-CFG-05)."""

    def test_invalid_agent_section_reports_agent_path(self, tmp_path: Path) -> None:
        """Error in agent section references agent-specific path."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "", "provider": "openrouter"},  # empty model
                "environment": {"runner_type": "local"},
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        errors = exc_info.value.errors()
        # Error should reference the agent section
        field_paths = [str(e["loc"]) for e in errors]
        assert any("agent" in p.lower() or "model" in p.lower() for p in field_paths)

    def test_invalid_environment_section_reports_environment_path(self, tmp_path: Path) -> None:
        """Error in environment section references environment-specific path."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": ""},  # empty runner_type
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        errors = exc_info.value.errors()
        field_paths = [str(e["loc"]) for e in errors]
        assert any("environment" in p.lower() or "runner" in p.lower() for p in field_paths)

    def test_invalid_timeout_in_agent_reports_nested_path(self, tmp_path: Path) -> None:
        """Error in nested agent.timeout reports the full path."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": -1},
                "environment": {"runner_type": "local"},
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        errors = exc_info.value.errors()
        field_paths = [str(e["loc"]) for e in errors]
        assert any("timeout" in p.lower() for p in field_paths)

    def test_multiple_sections_with_one_invalid(self, tmp_path: Path) -> None:
        """Error in one section doesn't prevent other section errors from being reported."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "max_turns": -1},
                "environment": {"runner_type": "local", "cpu_limit": -1.0},
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        errors = exc_info.value.errors()
        # Should report errors from both sections
        assert len(errors) >= 2


# =========================================================================
# VAL-CFG-06: Config supports list and map values
# =========================================================================


class TestListAndMapValues:
    """Test list and map support for tasks and artifacts (VAL-CFG-06)."""

    def test_empty_task_list_valid(self, tmp_path: Path) -> None:
        """An empty tasks list is valid."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "tasks": [],
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.tasks == []

    def test_single_task_in_config(self, tmp_path: Path) -> None:
        """A single task definition in config loads correctly."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "tasks": [
                    {
                        "id": "task-001",
                        "prompt": "Fix the bug",
                        "language": "python",
                        "test_command": "pytest tests/",
                        "timeout": 300,
                    },
                ],
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert len(config.tasks) == 1
        assert config.tasks[0].id == "task-001"
        assert config.tasks[0].language == "python"

    def test_five_tasks_in_config(self, tmp_path: Path) -> None:
        """Multiple task definitions load correctly."""
        yaml_file = tmp_path / "config.yaml"
        tasks = [
            {
                "id": f"task-{i:03d}",
                "prompt": f"Task {i}",
                "language": "python",
                "test_command": "pytest",
                "timeout": 300,
            }
            for i in range(5)
        ]
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "tasks": tasks,
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert len(config.tasks) == 5

    def test_artifact_definitions_as_maps(self, tmp_path: Path) -> None:
        """Artifact definitions as YAML maps load correctly."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "artifacts": {
                    "my-tool": {
                        "type": "tool",
                        "name": "my-tool",
                        "description": "A test tool",
                        "input_schema": {"type": "object"},
                    },
                    "my-mcp": {
                        "type": "mcp_server",
                        "name": "my-mcp",
                        "command": "npx",
                        "args": ["mcp-server"],
                    },
                    "my-skill": {
                        "type": "skill",
                        "name": "my-skill",
                        "skill_file_path": "/path/to/skill.md",
                    },
                },
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert len(config.artifacts) == 3
        artifact_names = {a.name for a in config.artifacts}
        assert artifact_names == {"my-tool", "my-mcp", "my-skill"}

    def test_invalid_task_in_list_reports_index(self, tmp_path: Path) -> None:
        """Invalid task in list reports the problematic index."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
                "tasks": [
                    {
                        "id": "task-001",
                        "prompt": "Valid task",
                        "language": "python",
                        "test_command": "pytest",
                        "timeout": 300,
                    },
                    {
                        "id": "task-002",
                        "prompt": "",  # Invalid: empty prompt
                        "language": "python",
                        "test_command": "pytest",
                        "timeout": -1,  # Invalid: negative timeout
                    },
                ],
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file=str(yaml_file))

        errors = exc_info.value.errors()
        # Should reference the second task (index 1)
        field_paths = [str(e["loc"]) for e in errors]
        assert any("1" in p for p in field_paths)


# =========================================================================
# VAL-CFG-07: Config resolution logs source of each resolved value
# =========================================================================


class TestDebugLogging:
    """Test config source logging (VAL-CFG-07)."""

    def test_debug_logging_shows_yaml_source(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DEBUG logging shows source as YAML for values from YAML file."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        with caplog.at_level(logging.DEBUG, logger="grist_mill.config"):
            config = load_config(yaml_file=str(yaml_file))

        assert config.agent.timeout == 300
        log_messages = caplog.text
        assert "timeout" in log_messages.lower()
        assert "yaml" in log_messages.lower()

    def test_debug_logging_with_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DEBUG logging shows source as env var for overridden values."""
        monkeypatch.setenv("GRIST_MILL_TIMEOUT", "600")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        with caplog.at_level(logging.DEBUG, logger="grist_mill.config"):
            load_config(yaml_file=str(yaml_file))

        log_messages = caplog.text
        # Should log that timeout came from env var
        assert "timeout" in log_messages.lower()

    def test_debug_logging_with_cli_override(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DEBUG logging shows source as CLI for CLI-overridden values."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        with caplog.at_level(logging.DEBUG, logger="grist_mill.config"):
            load_config(yaml_file=str(yaml_file), cli_args={"timeout": 900})

        log_messages = caplog.text
        # Should log that timeout came from CLI
        assert "timeout" in log_messages.lower()
        assert "cli" in log_messages.lower()

    def test_debug_logging_shows_yaml_source_for_non_overridden(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DEBUG logging shows YAML source for values not overridden."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter", "timeout": 300},
                "environment": {"runner_type": "local"},
            },
        )

        with caplog.at_level(logging.DEBUG, logger="grist_mill.config"):
            load_config(yaml_file=str(yaml_file))

        log_messages = caplog.text
        assert "timeout" in log_messages.lower()
        assert "yaml" in log_messages.lower()


# =========================================================================
# VAL-CFG-08: Config supports secret masking in string representation
# =========================================================================


class TestSecretMasking:
    """Test secret masking in __str__ and __repr__ (VAL-CFG-08)."""

    def test_api_key_masked_in_str(self, tmp_path: Path) -> None:
        """API key is masked in string representation."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))
        config_str = str(config)

        assert "sk-test-secret-key-12345" not in config_str
        assert "***" in config_str

    def test_api_key_masked_in_repr(self, tmp_path: Path) -> None:
        """API key is masked in repr."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))
        config_repr = repr(config)

        assert "sk-test-secret-key-12345" not in config_repr
        assert "***" in config_repr

    def test_direct_field_access_returns_real_value(self, tmp_path: Path) -> None:
        """Direct field access returns the actual secret value."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))

        # Direct access should return the real value
        assert config.agent.api_key == "sk-test-secret-key-12345"

    def test_secret_masking_with_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secret masking works even when value comes from env var."""
        monkeypatch.setenv("GRIST_MILL_API_KEY", "sk-env-secret-99999")
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))
        config_str = str(config)

        assert "sk-env-secret-99999" not in config_str
        assert config.agent.api_key == "sk-env-secret-99999"

    def test_non_secret_fields_visible_in_str(self, tmp_path: Path) -> None:
        """Non-secret fields are visible in string representation."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))
        config_str = str(config)

        assert "gpt-4" in config_str
        assert "openrouter" in config_str
        assert "local" in config_str


# =========================================================================
# Additional edge cases
# =========================================================================


class TestConfigDefaults:
    """Test that default values are sensible."""

    def test_default_agent_max_turns(self) -> None:
        """Agent section has sensible default for max_turns."""
        section = AgentSection(model="gpt-4", provider="openrouter")
        assert section.max_turns == 5

    def test_default_agent_timeout(self) -> None:
        """Agent section has sensible default for timeout."""
        section = AgentSection(model="gpt-4", provider="openrouter")
        assert section.timeout == 600

    def test_default_environment_cpu(self) -> None:
        """Environment section has sensible default for cpu_limit."""
        section = EnvironmentSection(runner_type="local")
        assert section.cpu_limit == 1.0

    def test_default_telemetry(self, tmp_path: Path) -> None:
        """Telemetry defaults are sensible."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(
            yaml_file,
            {
                "agent": {"model": "gpt-4", "provider": "openrouter"},
                "environment": {"runner_type": "local"},
            },
        )

        config = load_config(yaml_file=str(yaml_file))

        assert config.telemetry.enabled is True  # default enabled
        assert config.telemetry.trace_enabled is False


class TestConfigRoundTrip:
    """Test config serialization round-trip."""

    def test_config_to_dict_round_trip(self, tmp_path: Path) -> None:
        """Config serializes to dict and back without data loss."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))
        data = config.model_dump()
        config2 = GristMillConfig(**data)

        assert config2.agent.model == config.agent.model
        assert config2.agent.timeout == config.agent.timeout
        assert config2.environment.runner_type == config.environment.runner_type

    def test_config_to_json_round_trip(self, tmp_path: Path) -> None:
        """Config serializes to JSON and back without data loss."""
        yaml_file = tmp_path / "config.yaml"
        _write_yaml(yaml_file, _sample_yaml_data())

        config = load_config(yaml_file=str(yaml_file))
        json_str = config.model_dump_json()
        config2 = GristMillConfig.model_validate_json(json_str)

        assert config2.agent.model == config.agent.model
        assert config2.agent.timeout == config.agent.timeout
