"""CLI subcommand: grist-mill tasks generate.

Provides the 'tasks' command group with the 'generate' subcommand for
end-to-end task synthesis from a source repository.

Validates:
- VAL-SYNTH-05: CLI 'grist-mill tasks generate --repo <path>' produces output dataset
- VAL-PKG-02: CLI provides rich help and subcommand discovery
"""

from __future__ import annotations

import logging
import sys

import click

logger = logging.getLogger(__name__)


@click.group()
def tasks() -> None:
    """Task synthesis commands.

    Generate, manage, and validate benchmark tasks from source code.
    """
    pass


@tasks.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=False),
    help="Path to the source repository to generate tasks from.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output directory for the generated dataset (default: <repo>/generated_tasks).",
)
@click.option(
    "--max-mutations-per-type",
    default=3,
    type=int,
    help="Maximum mutations per mutation type per file (default: 3).",
)
@click.option(
    "--max-tasks-per-file",
    default=5,
    type=int,
    help="Maximum tasks to generate per source file (default: 5).",
)
@click.option(
    "--timeout",
    default=300,
    type=int,
    help="Default timeout in seconds for generated tasks (default: 300).",
)
@click.option(
    "--no-difficulty",
    is_flag=True,
    default=False,
    help="Skip difficulty estimation for generated tasks.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress progress output.",
)
def generate(
    repo: str,
    output: str | None,
    max_mutations_per_type: int,
    max_tasks_per_file: int,
    timeout: int,
    no_difficulty: bool,
    quiet: bool,
) -> None:
    """Generate benchmark tasks from a source repository.

    Runs the end-to-end task synthesis pipeline: discovers source files,
    analyzes code with AST parsing, applies mutations to induce controlled
    failures, generates task descriptions, applies quality gates, and
    produces a validated task dataset.

    The generated dataset includes tasks with language declarations,
    dependency information, and execution environment requirements.

    Example:

        grist-mill tasks generate --repo /path/to/my-project

        grist-mill tasks generate --repo /path/to/my-project --output ./tasks_out

        grist-mill tasks generate --repo /path/to/my-project --max-mutations-per-type 5
    """
    from pathlib import Path

    from rich.console import Console

    from grist_mill.tasks.pipeline import PipelineConfig, TaskPipeline

    console = Console()

    # Validate repo path
    repo_path = Path(repo)
    if not repo_path.exists():
        console.print(
            f"[red]Error:[/red] Repository path does not exist: {repo}",
        )
        sys.exit(1)

    if not repo_path.is_dir():
        console.print(
            f"[red]Error:[/red] Repository path is not a directory: {repo}",
        )
        sys.exit(1)

    # Set output directory
    output_path = repo_path / "generated_tasks" if output is None else Path(output)

    # Build pipeline config
    config = PipelineConfig(
        max_mutations_per_type=max_mutations_per_type,
        max_tasks_per_file=max_tasks_per_file,
        timeout=timeout,
        estimate_difficulty=not no_difficulty,
    )

    # Run pipeline
    if not quiet:
        console.print(
            "[bold]Task Synthesis Pipeline[/bold]",
        )
        console.print(f"  Repository: {repo_path}")
        console.print(f"  Output: {output_path}")
        console.print(f"  Max mutations/type: {max_mutations_per_type}")
        console.print(f"  Max tasks/file: {max_tasks_per_file}")
        console.print()

    pipeline = TaskPipeline(config=config)
    result = pipeline.run(repo_path, output_dir=output_path)

    # Print results
    if not quiet:
        console.print("[bold green]Pipeline completed![/bold green]")
        console.print(f"  Source files processed: {result.total_source_files}")
        console.print(f"  Mutations attempted: {result.total_mutations_attempted}")
        console.print(f"  Mutations succeeded: {result.total_mutations_succeeded}")
        console.print(f"  Tasks filtered by quality gate: {result.tasks_filtered_by_quality}")
        console.print(f"  Tasks generated: [bold]{result.dataset.task_count}[/bold]")
        console.print(f"  Output: {output_path / 'tasks.json'}")

        # Language breakdown
        if result.dataset.task_count > 0:
            from collections import Counter

            langs = Counter(task.language for task in result.dataset.list_tasks())
            console.print()
            console.print("  Language breakdown:")
            for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
                console.print(f"    {lang}: {count}")
    else:
        # Quiet mode: just print the count
        print(result.dataset.task_count)


__all__ = ["generate", "tasks"]
