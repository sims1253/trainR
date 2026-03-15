"""CLI 'run' subcommand.

Executes benchmark evaluations from a configuration file. Supports
``--dry-run`` to preview the experiment matrix without running tasks.

Validates:
- VAL-CLI-02: run subcommand executes from config
- VAL-CLI-05: --dry-run previews execution
- VAL-CLI-07: missing config handled gracefully
"""

from __future__ import annotations

import json
import logging
import sys

import click
import yaml

from grist_mill.config import GristMillConfig, load_config

logger = logging.getLogger(__name__)


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
@click.pass_context
def run(ctx: click.Context, config_file: str, dry_run: bool, output_format: str | None) -> None:
    """Run benchmark evaluations from a configuration file.

    Loads the configuration, resolves the experiment matrix (tasks x agents x environments),
    and either executes the tasks or previews them with --dry-run.

    Examples:

        \b
        grist-mill run --config experiment.yaml

        \b
        grist-mill run --config experiment.yaml --dry-run

        \b
        grist-mill run --config experiment.yaml --dry-run --output-format json
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

    # Full execution is a stub for now (M2+ will implement actual execution)
    click.echo(
        "Full execution is not yet implemented. Use --dry-run to preview the experiment matrix.",
        err=True,
    )
    sys.exit(1)


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
