"""CLI 'validate' subcommand.

Validates a configuration file without executing any tasks.

Validates:
- VAL-CLI-03: validate checks config without executing
- VAL-CLI-07: missing config handled gracefully
"""

from __future__ import annotations

import sys

import click

from grist_mill.config import load_config


@click.command()
@click.option(
    "--config",
    "-c",
    "config_file",
    required=True,
    help="Path to the configuration YAML file to validate.",
)
def validate(config_file: str) -> None:
    """Validate a configuration file without executing tasks.

    Checks that the YAML is well-formed, all required fields are present,
    referenced artifacts exist, and all values pass validation rules.

    Exits with code 0 if valid, code 1 if invalid.

    Examples:

        \b
        grist-mill validate --config experiment.yaml
    """
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

    # Report validation results
    n_tasks = len(config.tasks)
    n_artifacts = len(config.artifacts)
    click.echo(f"Configuration is valid: {config_file}")
    click.echo(f"  Agent:      {config.agent.model} ({config.agent.provider})")
    click.echo(f"  Environment: {config.environment.runner_type}", nl=False)
    if config.environment.docker_image:
        click.echo(f" (image: {config.environment.docker_image})")
    else:
        click.echo()
    click.echo(f"  Tasks:      {n_tasks}")
    click.echo(f"  Artifacts:  {n_artifacts}")
    if n_tasks > 0:
        for task in config.tasks:
            click.echo(f"    - {task.id} [{task.language}] {task.difficulty.value}")
    if n_artifacts > 0:
        for artifact in config.artifacts:
            click.echo(f"    - {artifact.name} ({artifact.type})")
