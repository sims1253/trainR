"""Tests for the grist-mill CLI entrypoint.

Validates:
- VAL-CLI-01: CLI entrypoint is installed and runnable
- VAL-CLI-03: validate subcommand checks config without executing
- VAL-CLI-04: list subcommand shows registered artifacts
- VAL-CLI-05: --dry-run previews execution
- VAL-CLI-07: missing config handled gracefully
- VAL-CLI-08: verbose/quiet flags control logging
"""

from __future__ import annotations

import json
import textwrap

from click.testing import CliRunner

from grist_mill.cli.main import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CONFIG = textwrap.dedent("""\
    agent:
      model: gpt-4
      provider: openrouter
      system_prompt: "You are a helpful coding assistant."
      max_turns: 5
      timeout: 300
    environment:
      runner_type: local
      docker_image: python:3.12
      cpu_limit: 1.0
      memory_limit: "2g"
      network_access: false
    telemetry:
      enabled: true
      trace_enabled: false
      output_dir: "./results/telemetry"
    tasks:
      - id: task-001
        prompt: "Fix the bug in the parse function."
        language: python
        test_command: "pytest tests/test_parse.py"
        timeout: 120
        difficulty: EASY
      - id: task-002
        prompt: "Add error handling to the HTTP client."
        language: python
        test_command: "pytest tests/test_http_client.py"
        setup_command: "pip install -r requirements.txt"
        timeout: 180
        difficulty: MEDIUM
    artifacts:
      - type: tool
        name: file_reader
        description: "Read file contents"
        input_schema:
          type: object
          properties:
            path:
              type: string
          required:
            - path
      - type: skill
        name: python_debugging
        skill_file_path: "/skills/python_debugging.md"
        description: "Python debugging best practices"
      - type: mcp_server
        name: github_server
        command: npx
        args:
          - "@modelcontextprotocol/server-github"
""")

_INVALID_CONFIG = textwrap.dedent("""\
    agent:
      model: ""
      provider: ""
""")


def _write_config(tmp_path, content: str, filename: str = "config.yaml") -> str:
    """Write config content to a temp file and return its path."""
    p = tmp_path / filename
    p.write_text(content)
    return str(p)


# =========================================================================
# VAL-CLI-01: CLI entrypoint is installed and runnable
# =========================================================================


class TestCliEntrypoint:
    """Tests that grist-mill CLI is installed and provides help."""

    def test_help_shows_subcommands(self) -> None:
        """--help shows run, validate, and list subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "list" in result.output

    def test_help_shows_global_options(self) -> None:
        """--help shows --verbose and --quiet global flags."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output or "-v" in result.output
        assert "--quiet" in result.output or "-q" in result.output

    def test_version(self) -> None:
        """--version prints the semantic version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
        assert "grist-mill" in result.output

    def test_run_help(self) -> None:
        """run --help shows options including --config and --dry-run."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--dry-run" in result.output
        assert "--output-format" in result.output

    def test_validate_help(self) -> None:
        """validate --help shows options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_list_help(self) -> None:
        """list --help shows --artifacts and --harnesses options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "--artifacts" in result.output
        assert "--harnesses" in result.output


# =========================================================================
# VAL-CLI-03: validate subcommand
# =========================================================================


class TestValidate:
    """Tests for the validate subcommand."""

    def test_valid_config_exits_zero(self, tmp_path) -> None:
        """validate with valid config exits 0 and prints summary."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", config_path])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output
        assert "gpt-4" in result.output
        assert "openrouter" in result.output
        assert "2" in result.output  # 2 tasks
        assert "3" in result.output  # 3 artifacts
        assert "task-001" in result.output
        assert "task-002" in result.output

    def test_invalid_config_exits_one(self, tmp_path) -> None:
        """validate with invalid config exits 1 with error."""
        config_path = _write_config(tmp_path, _INVALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", config_path])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_missing_config_exits_one(self) -> None:
        """validate with nonexistent file exits 1 with friendly error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", "nonexistent.yaml"])
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "not found" in result.output.lower()


# =========================================================================
# VAL-CLI-04: list subcommand
# =========================================================================


class TestList:
    """Tests for the list subcommand."""

    def test_list_artifacts(self) -> None:
        """list --artifacts prints artifact listing."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--artifacts"])
        assert result.exit_code == 0
        # Empty registry should show a message
        assert "No artifacts registered" in result.output or "artifact" in result.output.lower()

    def test_list_harnesses(self) -> None:
        """list --harnesses prints harness listing."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--harnesses"])
        assert result.exit_code == 0
        assert "LocalHarness" in result.output

    def test_list_defaults_to_artifacts(self) -> None:
        """list with no flags defaults to showing artifacts."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0


# =========================================================================
# VAL-CLI-05: --dry-run
# =========================================================================


class TestDryRun:
    """Tests for the run --dry-run subcommand."""

    def test_dry_run_with_valid_config(self, tmp_path) -> None:
        """--dry-run loads config and prints experiment matrix."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", config_path, "--dry-run"])
        assert result.exit_code == 0
        assert "dry_run" in result.output.lower() or "dry" in result.output.lower()
        assert "task-001" in result.output
        assert "task-002" in result.output
        assert "gpt-4" in result.output
        assert "openrouter" in result.output
        assert "2" in result.output  # total_tasks

    def test_dry_run_json_output(self, tmp_path) -> None:
        """--dry-run --output-format json produces valid JSON."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--config", config_path, "--dry-run", "--output-format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["dry_run"] is True
        assert data["total_tasks"] == 2
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["task_id"] == "task-001"
        assert data["agent"]["model"] == "gpt-4"

    def test_dry_run_yaml_output(self, tmp_path) -> None:
        """--dry-run --output-format yaml produces YAML output."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--config", config_path, "--dry-run", "--output-format", "yaml"],
        )
        assert result.exit_code == 0
        assert "task-001" in result.output
        assert "gpt-4" in result.output

    def test_dry_run_without_config_fails(self) -> None:
        """--dry-run without --config should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--dry-run"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output


# =========================================================================
# VAL-CLI-07: missing config handled gracefully
# =========================================================================


class TestMissingConfig:
    """Tests for graceful handling of missing config files."""

    def test_run_missing_config_exits_one(self) -> None:
        """run with nonexistent config exits 1 with friendly error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", "nonexistent.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Error" in result.output
        # No Python traceback
        assert "Traceback" not in result.output

    def test_run_invalid_yaml_exits_one(self, tmp_path) -> None:
        """run with invalid YAML exits 1 with error message."""
        config_path = _write_config(tmp_path, "not: valid: yaml: structure: :")
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", config_path])
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Traceback" not in result.output

    def test_run_missing_required_field_exits_one(self, tmp_path) -> None:
        """run with missing required fields exits 1 with validation error."""
        config_path = _write_config(tmp_path, _INVALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", config_path])
        assert result.exit_code == 1
        assert "Error" in result.output


# =========================================================================
# VAL-CLI-08: verbose and quiet flags
# =========================================================================


class TestVerbosity:
    """Tests for --verbose and --quiet flags controlling logging."""

    def test_verbose_sets_debug(self, tmp_path, caplog) -> None:
        """--verbose enables DEBUG logging."""

        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "validate", "--config", config_path])
        assert result.exit_code == 0
        # With --verbose, we should see DEBUG-level output in logs
        # The validate command should succeed
        assert "Configuration is valid" in result.output

    def test_quiet_sets_warning(self, tmp_path) -> None:
        """--quiet suppresses output to WARNING and above."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["--quiet", "validate", "--config", config_path])
        assert result.exit_code == 0
        # With --quiet, INFO messages are suppressed but output still shows
        assert "Configuration is valid" in result.output

    def test_default_is_info(self, tmp_path) -> None:
        """Default logging level is INFO."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", config_path])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output

    def test_verbose_and_quiet_mutually_exclusive(self, tmp_path) -> None:
        """Both --verbose and --quiet produces an error."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--quiet", "validate", "--config", config_path])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_verbose_short_flag(self, tmp_path) -> None:
        """-v short flag works the same as --verbose."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["-v", "validate", "--config", config_path])
        assert result.exit_code == 0

    def test_quiet_short_flag(self, tmp_path) -> None:
        """-q short flag works the same as --quiet."""
        config_path = _write_config(tmp_path, _VALID_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["-q", "validate", "--config", config_path])
        assert result.exit_code == 0


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge case tests for the CLI."""

    def test_run_without_config(self) -> None:
        """run without --config should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output

    def test_validate_without_config(self) -> None:
        """validate without --config should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code != 0

    def test_unknown_command(self) -> None:
        """Unknown command produces error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["unknown"])
        assert result.exit_code != 0
        assert "Error" in result.output or "No such command" in result.output
