"""CLI 'run' subcommand.

Executes benchmark evaluations from a configuration file. Supports
``--dry-run`` to preview the experiment matrix without running tasks.
Supports ``--output-format json/yaml`` for structured output.

For end-to-end execution, uses a stub agent that runs the test command
directly through the environment (since the LLM-backed agent is not yet
implemented in M2). This validates the full evaluation loop:
task -> environment preparation -> test execution -> result capture.

Validates:
- VAL-CLI-02: run subcommand executes from config
- VAL-CLI-05: --dry-run previews execution
- VAL-CLI-06: structured output format option
- VAL-CLI-07: missing config handled gracefully
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import yaml

from grist_mill.config import GristMillConfig, load_config
from grist_mill.harness import run_experiment
from grist_mill.harness.result_parser import ResultParser
from grist_mill.schemas import (
    AgentConfig,
    EnvironmentConfig,
    HarnessConfig,
    Manifest,
    Task,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class _StubAgent:
    """Stub agent that executes the test command directly.

    Used for end-to-end testing of the evaluation harness before the
    LLM-backed agent is implemented (M3). This agent simply invokes
    the task's test_command through the environment and returns the
    parsed result.

    In a real benchmark, the LLM agent would:
    1. Read the task prompt
    2. Write/modify code
    3. Return a TaskResult

    The stub agent skips step 2 and goes directly to testing.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._result_parser = ResultParser()

    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        """Execute the task's test command directly."""
        output = self._env.execute(task.test_command, timeout=float(task.timeout))
        return self._result_parser.parse(
            output,
            task_id=task.id,
            language=task.language,
        )


@click.command()
@click.option(
    "--config",
    "-c",
    "config_file",
    required=True,
    help="Path to the configuration YAML file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview what would run without executing tasks.",
)
@click.option(
    "--output-format",
    "-o",
    "output_format",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    default=None,
    help="Output format for results (json or yaml).",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    default=None,
    help="Write results manifest to this file instead of stdout.",
)
@click.pass_context
def run(
    ctx: click.Context,
    config_file: str,
    dry_run: bool,
    output_format: str | None,
    output_file: str | None,
) -> None:
    """Run benchmark evaluations from a configuration file.

    Loads the configuration, resolves the experiment matrix (tasks x agents x environments),
    and either executes the tasks or previews them with --dry-run.

    Examples:

        \b
        grist-mill run --config experiment.yaml

        \b
        grist-mill run --config experiment.yaml --dry-run

        \b
        grist-mill run --config experiment.yaml --output-format json

        \b
        grist-mill run --config experiment.yaml --output results.json
    """
    # --- Load configuration ---
    try:
        config = load_config(yaml_file=config_file)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(
            f"Error loading configuration from '{config_file}': {exc}",
            err=True,
        )
        sys.exit(1)

    # --- Resolve experiment matrix ---
    matrix = _build_experiment_matrix(config)
    logger.debug("Experiment matrix: %d entries", len(matrix))

    if dry_run:
        _print_dry_run(matrix, config, output_format or "yaml")
        return

    # --- Full execution ---
    if not config.tasks:
        click.echo("No tasks defined in configuration.", err=True)
        sys.exit(1)

    results = _execute_tasks(config)

    # --- Build and output manifest ---
    manifest = _build_manifest(config, results)
    _output_results(manifest, output_format or "yaml", output_file)


def _execute_tasks(config: GristMillConfig) -> list[TaskResult]:
    """Execute all tasks from the configuration through the harness.

    Args:
        config: The loaded configuration.

    Returns:
        A list of TaskResult objects.
    """
    from grist_mill.environments.local_runner import LocalRunner

    # Create the environment based on config
    if config.environment.runner_type == "docker":
        try:
            from grist_mill.environments.docker_runner import DockerRunner

            kwargs: dict[str, Any] = {}
            if config.environment.docker_image:
                kwargs["docker_image"] = config.environment.docker_image
            if config.environment.memory_limit:
                kwargs["memory_limit"] = config.environment.memory_limit
            if config.environment.cpu_limit:
                kwargs["cpu_limit"] = config.environment.cpu_limit
            env: Any = DockerRunner(**kwargs)
        except Exception as exc:
            click.echo(
                f"Warning: Docker runner unavailable ({exc}), falling back to local runner.",
                err=True,
            )
            env = LocalRunner()
    else:
        env = LocalRunner()

    # Build HarnessConfig from GristMillConfig
    harness_config = HarnessConfig(
        agent=AgentConfig(
            model=config.agent.model,
            provider=config.agent.provider,
            system_prompt=config.agent.system_prompt,
        ),
        environment=EnvironmentConfig(
            runner_type=config.environment.runner_type,
            docker_image=config.environment.docker_image,
            resource_limits=None,
        ),
        artifact_bindings=[],
    )

    # Create stub agent (uses environment to run test commands directly)
    agent = _StubAgent(env)

    # Run all tasks
    click.echo(f"Running {len(config.tasks)} task(s)...")
    results = run_experiment(
        tasks=config.tasks,
        config=harness_config,
        agent=agent,
        env=env,
        max_retries=3,
        retry_delay=1.0,
        trace_enabled=config.telemetry.trace_enabled,
    )

    # Print summary
    passed = sum(1 for r in results if r.status == TaskStatus.SUCCESS)
    failed = sum(1 for r in results if r.status in (TaskStatus.FAILURE, TaskStatus.ERROR))
    timed_out = sum(1 for r in results if r.status == TaskStatus.TIMEOUT)
    click.echo(
        f"Completed: {passed} passed, {failed} failed, {timed_out} timed out "
        f"(total: {len(results)})"
    )

    return results


def _build_manifest(config: GristMillConfig, results: list[TaskResult]) -> Manifest:
    """Build a Manifest from configuration and results.

    Args:
        config: The loaded configuration.
        results: The task results.

    Returns:
        A Manifest with results metadata.
    """
    return Manifest(
        name="grist-mill-evaluation",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        tasks=config.tasks,
    )


def _output_results(
    manifest: Manifest,
    output_format: str,
    output_file: str | None,
) -> None:
    """Output results in the specified format.

    Args:
        manifest: The results manifest.
        output_format: 'json' or 'yaml'.
        output_file: Optional file path to write results to.
    """
    manifest_data = _manifest_to_output(manifest)

    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            path.write_text(json.dumps(manifest_data, indent=2, default=str), encoding="utf-8")
        else:
            path.write_text(
                yaml.dump(manifest_data, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        click.echo(f"Results written to {output_file}")
    else:
        if output_format == "json":
            click.echo(json.dumps(manifest_data, indent=2, default=str))
        else:
            click.echo(yaml.dump(manifest_data, default_flow_style=False, sort_keys=False))


def _manifest_to_output(manifest: Manifest) -> dict[str, Any]:
    """Convert a Manifest to a serializable dict for output.

    Args:
        manifest: The results manifest.

    Returns:
        A dict suitable for JSON/YAML serialization.
    """
    return {
        "name": manifest.name,
        "version": manifest.version,
        "timestamp": manifest.timestamp.isoformat() if manifest.timestamp else None,
        "schema_version": "V1",
        "total_tasks": len(manifest.tasks),
        "tasks": [
            {
                "id": task.id,
                "language": task.language,
                "difficulty": task.difficulty.value,
                "test_command": task.test_command,
            }
            for task in manifest.tasks
        ],
    }


def _build_experiment_matrix(config: GristMillConfig) -> list[dict[str, object]]:
    """Build the experiment matrix from the configuration.

    Returns a list of dicts, each describing one experiment entry:
    task_id, language, test_command, agent model/provider, environment runner_type.

    Args:
        config: The loaded GristMillConfig.

    Returns:
        A list of experiment matrix entries.
    """
    matrix: list[dict[str, object]] = []
    for task in config.tasks:
        entry: dict[str, object] = {
            "task_id": task.id,
            "language": task.language,
            "test_command": task.test_command,
            "timeout": task.timeout,
            "difficulty": task.difficulty.value,
            "agent": {
                "model": config.agent.model,
                "provider": config.agent.provider,
            },
            "environment": {
                "runner_type": config.environment.runner_type,
                "docker_image": config.environment.docker_image,
            },
        }
        if task.setup_command:
            entry["setup_command"] = task.setup_command
        if config.environment.docker_image:
            entry["environment"]["docker_image"] = config.environment.docker_image
        matrix.append(entry)
    return matrix


def _print_dry_run(
    matrix: list[dict[str, object]],
    config: GristMillConfig,
    output_format: str,
) -> None:
    """Print the dry-run experiment matrix.

    Args:
        matrix: The experiment matrix to display.
        config: The loaded configuration.
        output_format: 'json' or 'yaml'.
    """
    output_data: dict[str, object] = {
        "dry_run": True,
        "config_file": "<provided>",
        "agent": {
            "model": config.agent.model,
            "provider": config.agent.provider,
            "max_turns": config.agent.max_turns,
            "timeout": config.agent.timeout,
        },
        "environment": {
            "runner_type": config.environment.runner_type,
            "docker_image": config.environment.docker_image,
        },
        "tasks": matrix,
        "total_tasks": len(matrix),
    }

    if output_format == "json":
        click.echo(json.dumps(output_data, indent=2, default=str))
    else:
        click.echo(yaml.dump(output_data, default_flow_style=False, sort_keys=False))
