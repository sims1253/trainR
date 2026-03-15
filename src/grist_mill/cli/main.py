"""CLI entrypoint for grist-mill."""

from __future__ import annotations

import click

from grist_mill import __version__


@click.group()
@click.version_option(version=__version__, prog_name="grist-mill")
def cli() -> None:
    """grist-mill: Language-agnostic benchmarking framework for autonomous coding agents."""


@cli.command()
def version() -> None:
    """Print the current version and exit."""
    click.echo(f"grist-mill {__version__}")


if __name__ == "__main__":
    cli()
