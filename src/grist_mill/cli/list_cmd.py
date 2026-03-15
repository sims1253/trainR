"""CLI 'list' subcommand.

Lists registered artifacts and harness implementations.

Validates:
- VAL-CLI-04: list shows registered artifacts and harnesses
"""

from __future__ import annotations

import click

from grist_mill.registry import ArtifactRegistry


@click.command("list")
@click.option(
    "--artifacts",
    "show_artifacts",
    is_flag=True,
    default=False,
    help="List registered artifacts with type and name.",
)
@click.option(
    "--harnesses",
    "show_harnesses",
    is_flag=True,
    default=False,
    help="List registered harness implementations.",
)
def list_cmd(show_artifacts: bool, show_harnesses: bool) -> None:
    """List registered artifacts and harness implementations.

    If no specific option is given, defaults to showing artifacts.

    Examples:

        \b
        grist-mill list --artifacts

        \b
        grist-mill list --harnesses
    """
    # Default to showing artifacts if no option is specified
    if not show_artifacts and not show_harnesses:
        show_artifacts = True

    if show_artifacts:
        _list_artifacts()

    if show_harnesses:
        _list_harnesses()


def _list_artifacts() -> None:
    """Print all registered artifacts with type and name."""
    registry = ArtifactRegistry()

    # Register artifacts from any config if available — for now show empty
    # The list command shows what's currently in the default registry
    artifacts = registry.list_artifacts()

    if not artifacts:
        click.echo("No artifacts registered.")
        click.echo(
            "Artifacts can be registered via configuration files or programmatically. "
            "Use 'grist-mill validate --config <file>' to check artifact definitions."
        )
        return

    # Group by type
    by_type: dict[str, list[str]] = {}
    for artifact in artifacts:
        by_type.setdefault(artifact.type, []).append(artifact.name)

    for artifact_type, names in sorted(by_type.items()):
        click.echo(f"[{artifact_type}]")
        for name in sorted(names):
            click.echo(f"  - {name}")

    click.echo(f"\nTotal: {len(artifacts)} artifact(s)")


def _list_harnesses() -> None:
    """Print registered harness implementations."""
    click.echo("[harnesses]")
    click.echo("  - LocalHarness (local process runner)")
    click.echo(
        "\nHarness implementations can be extended by subclassing "
        "grist_mill.interfaces.BaseHarness."
    )
