#!/usr/bin/env python
"""Generate testing tasks from R packages.

This script generates testing tasks from R package source code using the
task_generator module. It clones packages from r-lib, extracts test patterns,
and generates tasks suitable for training/testing code generation models.

Usage:
    uv run python scripts/generate_tasks.py --package cli --output ./tasks --num-tasks 15
    uv run python scripts/generate_tasks.py --package cli --output ./tasks --dry-run
    uv run python scripts/generate_tasks.py --package cli --output ./tasks --verbose
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from task_generator import TaskGenerator, TaskQualityGate, TestingTask

console = Console()
logger = logging.getLogger(__name__)


class TaskGenerationError(Exception):
    """Custom exception for task generation errors."""

    pass


class PackageCloneError(Exception):
    """Custom exception for package cloning errors."""

    pass


def clone_package(package_name: str, packages_dir: Path) -> Path:
    """Clone an R package from r-lib if not already present.

    Args:
        package_name: Name of the R package to clone.
        packages_dir: Directory to clone packages into.

    Returns:
        Path to the cloned package.

    Raises:
        PackageCloneError: If cloning fails.
    """
    package_path = packages_dir / package_name

    if package_path.exists():
        # Verify it's a valid git repo
        git_dir = package_path / ".git"
        if git_dir.exists():
            console.print(
                f"[green]✓[/green] Package {package_name} already exists at {package_path}"
            )
            return package_path
        else:
            console.print("[yellow]![/yellow] Directory exists but not a git repo, re-cloning...")
            import shutil

            shutil.rmtree(package_path)

    packages_dir.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/r-lib/{package_name}.git"

    console.print(f"[blue]Cloning {package_name} from {repo_url}...[/blue]")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(package_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise PackageCloneError(f"Git clone failed: {error_msg}")

        console.print(f"[green]✓[/green] Successfully cloned {package_name}")
        return package_path

    except subprocess.TimeoutExpired as e:
        raise PackageCloneError("Git clone timed out after 120 seconds") from e
    except FileNotFoundError as e:
        raise PackageCloneError("Git not found. Please install git.") from e


def validate_package_structure(package_path: Path) -> bool:
    """Validate that the package has required R package structure.

    Args:
        package_path: Path to the package.

    Returns:
        True if valid R package structure.
    """
    required_files = ["DESCRIPTION"]
    required_dirs = ["R"]

    for f in required_files:
        if not (package_path / f).exists():
            logger.warning(f"Missing required file: {f}")
            return False

    for d in required_dirs:
        if not (package_path / d).is_dir():
            logger.warning(f"Missing required directory: {d}")
            return False

    return True


def run_quality_gate(
    tasks: list[TestingTask], quality_gate: TaskQualityGate, verbose: bool = False
) -> tuple[list[TestingTask], list[dict[str, Any]]]:
    """Run quality gate on tasks and return valid ones.

    Args:
        tasks: List of tasks to validate.
        quality_gate: Quality gate instance.
        verbose: Whether to show verbose output.

    Returns:
        Tuple of (valid_tasks, rejected_tasks_with_reasons).
    """
    valid_tasks = []
    rejected_tasks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=verbose,
    ) as progress:
        task = progress.add_task("Running quality checks...", total=len(tasks))

        for t in tasks:
            metrics = quality_gate.validate(t)

            # Quality score is set by the quality gate's composite scoring
            t.quality_score = metrics.composite_score

            if metrics.is_valid:
                valid_tasks.append(t)

                if verbose:
                    console.print(
                        f"[green]✓[/green] {t.task_id}: quality_score={t.quality_score:.2f}"
                    )
            else:
                rejected_tasks.append(
                    {
                        "task_id": t.task_id,
                        "instruction_preview": t.instruction[:50] + "...",
                        "reasons": metrics.issues,
                        "metrics": metrics.to_dict(),
                    }
                )

                if verbose:
                    console.print(f"[red]✗[/red] {t.task_id}: {', '.join(metrics.issues)}")

            progress.advance(task)

    return valid_tasks, rejected_tasks


def print_statistics(
    tasks: list[TestingTask], rejected: list[dict[str, Any]], generator: TaskGenerator
) -> None:
    """Print statistics about generated tasks.

    Args:
        tasks: List of valid tasks.
        rejected: List of rejected tasks with reasons.
        generator: TaskGenerator instance.
    """
    stats = generator.get_statistics(tasks)

    # Summary panel
    total_generated = len(tasks) + len(rejected)
    pass_rate = (len(tasks) / total_generated * 100) if total_generated > 0 else 0

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Task Generation Summary[/bold]\n\n"
            f"Total generated: {total_generated}\n"
            f"Passed quality gate: [green]{len(tasks)}[/green] ({pass_rate:.1f}%)\n"
            f"Rejected: [red]{len(rejected)}[/red]",
            title="Summary",
            border_style="blue",
        )
    )

    # Statistics table
    table = Table(title="Task Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Total valid tasks", str(stats.get("total", 0)))

    # By split
    by_split = stats.get("by_split", {})
    if by_split:
        table.add_row("── By Split ──", "")
        for split, count in sorted(by_split.items()):
            table.add_row(f"  {split}", str(count))

    # By difficulty
    by_difficulty = stats.get("by_difficulty", {})
    if by_difficulty:
        table.add_row("── By Difficulty ──", "")
        for diff, count in sorted(by_difficulty.items()):
            color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(diff, "white")
            table.add_row(f"  [{color}]{diff}[/{color}]", str(count))

    # By test type
    by_test_type = stats.get("by_test_type", {})
    if by_test_type:
        table.add_row("── By Test Type ──", "")
        for test_type, count in sorted(by_test_type.items()):
            table.add_row(f"  {test_type}", str(count))

    console.print(table)

    # Rejection reasons summary
    if rejected:
        console.print()
        rejection_table = Table(
            title="Rejection Reasons", show_header=True, header_style="bold red"
        )
        rejection_table.add_column("Reason", style="yellow")
        rejection_table.add_column("Count", justify="right")

        # Count rejection reasons
        reason_counts: dict[str, int] = {}
        for r in rejected:
            for reason in r.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            rejection_table.add_row(reason, str(count))

        console.print(rejection_table)


def main() -> int:
    """Main entry point for task generation."""
    parser = argparse.ArgumentParser(
        description="Generate testing tasks from R packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate 15 tasks from cli package:
    uv run python scripts/generate_tasks.py --package cli --num-tasks 15

  Dry run without saving:
    uv run python scripts/generate_tasks.py --package cli --dry-run

  Verbose output:
    uv run python scripts/generate_tasks.py --package cli -v
""",
    )
    parser.add_argument(
        "--package", default="cli", help="Package name to generate tasks from (default: cli)"
    )
    parser.add_argument(
        "--output",
        default="./tasks",
        help="Output directory for generated tasks (default: ./tasks)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=15,
        help="Number of tasks to generate (default: 15, MVP v1.1 scope: 10-20)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        metavar=("TRAIN", "DEV", "HELD_OUT"),
        help="Split ratio for train/dev/held_out (default: 0.6 0.2 0.2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't save tasks, just show what would be generated"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--skip-clone", action="store_true", help="Skip cloning (use existing package)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate split ratio
    if sum(args.split_ratio) != 1.0:
        console.print("[red]Error: Split ratios must sum to 1.0[/red]")
        return 1

    # Validate num_tasks
    if args.num_tasks < 1 or args.num_tasks > 100:
        console.print("[yellow]Warning: num_tasks outside recommended range (1-100)[/yellow]")

    # Paths
    project_root = Path(__file__).parent.parent
    packages_dir = project_root / "packages"
    output_dir = Path(args.output)

    # Make output_dir absolute if relative
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    console.print(
        Panel.fit(
            f"[bold]Task Generation Configuration[/bold]\n\n"
            f"Package: {args.package}\n"
            f"Output: {output_dir}\n"
            f"Num tasks: {args.num_tasks}\n"
            f"Split ratio: {args.split_ratio[0]:.0%} / {args.split_ratio[1]:.0%} / {args.split_ratio[2]:.0%}\n"
            f"Dry run: {args.dry_run}",
            title="Configuration",
            border_style="cyan",
        )
    )

    try:
        # Clone or locate package
        if args.skip_clone:
            package_path = packages_dir / args.package
            if not package_path.exists():
                console.print(f"[red]Error: Package not found at {package_path}[/red]")
                console.print("Run without --skip-clone to clone the package first.")
                return 1
        else:
            try:
                package_path = clone_package(args.package, packages_dir)
            except PackageCloneError as e:
                console.print(f"[red]Error cloning package: {e}[/red]")
                return 1

        # Validate package structure
        if not validate_package_structure(package_path):
            console.print(
                "[yellow]Warning: Package may not have valid R package structure[/yellow]"
            )

        # Initialize generator
        console.print("\n[blue]Initializing task generator...[/blue]")
        quality_gate = TaskQualityGate(
            min_instruction_length=50,
            min_context_length=20,
            require_reference_test=True,
            validate_r_syntax=True,
        )
        generator = TaskGenerator(
            output_dir=output_dir,
            quality_gate=quality_gate,
        )

        # Generate tasks
        console.print(f"[blue]Generating tasks from {package_path}...[/blue]")

        # Note: TaskGenerator.generate_from_package already runs quality_gate.filter_valid
        # But we want more detailed statistics, so we'll do it separately
        all_tasks = generator.generate_from_package(
            package_path=package_path,
            num_tasks=args.num_tasks,
            split_ratio=tuple(args.split_ratio),
        )

        if not all_tasks:
            console.print("[yellow]No tasks were generated. Check if package has tests.[/yellow]")
            return 0

        # Run quality gate again for detailed statistics (generator already filters)
        # We do this to get rejection details
        valid_tasks, rejected = run_quality_gate(all_tasks, quality_gate, args.verbose)

        # Save or dry run
        if args.dry_run:
            console.print(f"\n[yellow]Dry run - not saving {len(valid_tasks)} tasks[/yellow]")
            for task in valid_tasks[:10]:  # Show first 10
                console.print(f"  - {task.task_id} ({task.split}): {task.instruction[:60]}...")
            if len(valid_tasks) > 10:
                console.print(f"  ... and {len(valid_tasks) - 10} more")
        else:
            # Create output directories
            for split in ["train", "dev", "held_out"]:
                (output_dir / split).mkdir(parents=True, exist_ok=True)

            # Save tasks
            console.print(f"\n[blue]Saving tasks to {output_dir}...[/blue]")
            saved_paths = []
            for task in valid_tasks:
                path = generator.save_task(task)
                saved_paths.append(path)
                if args.verbose:
                    console.print(f"  [green]✓[/green] Saved: {path.relative_to(project_root)}")

            console.print(f"[green]✓[/green] Saved {len(saved_paths)} tasks")

        # Print statistics
        print_statistics(valid_tasks, rejected, generator)

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
