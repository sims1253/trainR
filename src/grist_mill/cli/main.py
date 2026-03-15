"""CLI entrypoint for grist-mill.

Provides the main CLI group and global flags (``--verbose``, ``--quiet``).
Subcommands are defined in separate modules and attached here.

Validates:
- VAL-CLI-01: CLI entrypoint is installed and runnable
- VAL-CLI-02: run subcommand executes from config (stub with --dry-run)
- VAL-CLI-03: validate subcommand checks config without executing
- VAL-CLI-04: list subcommand shows registered artifacts
- VAL-CLI-05: --dry-run previews execution
- VAL-CLI-07: missing config handled gracefully
- VAL-CLI-08: verbose/quiet flags control logging
"""

from __future__ import annotations

import logging
import sys

import click

from grist_mill import __version__


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags.

    Args:
        verbose: If True, set level to DEBUG.
        quiet: If True, set level to WARNING.
        Default (neither): INFO.

    Raises:
        click.BadParameter: If both --verbose and --quiet are set.
    """
    if verbose and quiet:
        raise click.BadParameter("--verbose and --quiet are mutually exclusive.")

    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(name)s: %(message)s",
        stream=sys.stderr,
    )


@click.group()
@click.version_option(version=__version__, prog_name="grist-mill")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress output; only show WARNING and above.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """grist-mill: Language-agnostic benchmarking framework for autonomous coding agents.

    Use grist-mill to define, run, and analyze benchmark evaluations of
    autonomous coding agents across their full toolchain.
    """
    ctx.ensure_object(dict)
    _setup_logging(verbose, quiet)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


def _register_subcommands() -> None:
    """Attach subcommand modules to the CLI group."""
    from grist_mill.cli.export_cmd import export
    from grist_mill.cli.list_cmd import list_cmd
    from grist_mill.cli.optimize_cmd import optimize
    from grist_mill.cli.report_cmd import report
    from grist_mill.cli.run_cmd import run
    from grist_mill.cli.tasks_cmd import tasks
    from grist_mill.cli.validate_cmd import validate

    cli.add_command(run, name="run")
    cli.add_command(validate, name="validate")
    cli.add_command(list_cmd, name="list")
    cli.add_command(tasks, name="tasks")
    cli.add_command(optimize, name="optimize")
    cli.add_command(report, name="report")
    cli.add_command(export, name="export")


# Register subcommands on import
_register_subcommands()

if __name__ == "__main__":
    cli()
