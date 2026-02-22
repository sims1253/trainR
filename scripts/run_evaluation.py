#!/usr/bin/env python
"""Run evaluation of a skill on a testing task."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from evaluation import DockerPiRunnerConfig, EvaluationSandbox
from task_generator.models import TestingTask

console = Console()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a skill on a testing task")
    parser.add_argument("--task", required=True, help="Path to task JSON file")
    parser.add_argument("--skill", required=True, help="Path to skill markdown file")
    parser.add_argument(
        "--package-dir",
        default=None,
        help="Path to R package directory (defaults to packages/{source_package})",
    )
    parser.add_argument(
        "--docker-image",
        default="posit-gskill-eval:latest",
        help="Docker image to use for evaluation",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for evaluation (default: 600)",
    )
    parser.add_argument(
        "--save-result",
        default=None,
        help="Path to save evaluation result as JSON",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Load task
    task_path = Path(args.task)
    if not task_path.exists():
        console.print(f"[red]Task file not found: {task_path}[/red]")
        sys.exit(1)

    task_data = json.loads(task_path.read_text())
    task = TestingTask.from_dict(task_data)

    # Load skill
    skill_path = Path(args.skill)
    if not skill_path.exists():
        console.print(f"[red]Skill file not found: {skill_path}[/red]")
        sys.exit(1)

    skill_prompt = skill_path.read_text()

    # Check API key is in environment
    if not os.environ.get("Z_AI_API_KEY"):
        console.print(
            "[red]Z_AI_API_KEY not set in environment. "
            "Set it in your shell config (e.g. ~/.config/fish/conf.d/secrets.fish).[/red]"
        )
        sys.exit(1)

    # Build runner config
    runner_config = DockerPiRunnerConfig(
        docker_image=args.docker_image,
        timeout=args.timeout,
    )

    # Determine package directory
    package_dir = None
    if args.package_dir:
        package_dir = Path(args.package_dir)
        if not package_dir.exists():
            console.print(f"[red]Package directory not found: {package_dir}[/red]")
            sys.exit(1)

    # Show configuration
    config_table = Table(show_header=False, box=None)
    config_table.add_column("key", style="dim")
    config_table.add_column("value")
    config_table.add_row("Task ID:", task.task_id)
    config_table.add_row("Package:", task.source_package)
    config_table.add_row("Difficulty:", str(task.difficulty))
    config_table.add_row("Type:", task.test_type)
    config_table.add_row("Docker Image:", args.docker_image)
    config_table.add_row("Timeout:", f"{args.timeout}s")
    if package_dir:
        config_table.add_row("Package Dir:", str(package_dir))

    console.print(Panel(config_table, title="Evaluation Configuration", border_style="blue"))

    # Show task instruction
    if args.verbose:
        console.print("\n[bold]Task Instruction:[/bold]")
        console.print(Panel(task.instruction, border_style="dim"))

    # Run evaluation
    console.print("\n[blue]Running evaluation in Docker...[/blue]")

    sandbox = EvaluationSandbox(runner_config=runner_config)
    result = sandbox.evaluate_task(task, skill_prompt, package_dir=package_dir)

    # Show results
    if result.success:
        console.print("\n[green bold]PASSED[/green bold]")
    else:
        console.print("\n[red bold]FAILED[/red bold]")
        if result.failure_category:
            console.print(f"[red]Failure: {result.failure_category}[/red]")
        if result.error_message:
            console.print(f"[red]Error: {result.error_message}[/red]")

    # Show test results
    if result.test_results:
        console.print("\n[bold]Test Results:[/bold]")
        test_table = Table(show_header=True, header_style="bold")
        test_table.add_column("Test", style="dim")
        test_table.add_column("Status")
        test_table.add_column("Message")

        for test in result.test_results:
            status = "[green]PASS[/green]" if test.passed else "[red]FAIL[/red]"
            message = test.message[:80] if test.message else ""
            test_table.add_row(test.name, status, message)

        console.print(test_table)

    # Show generated code
    if result.generated_code and args.verbose:
        console.print("\n[dim]Generated Code:[/dim]")
        syntax = Syntax(result.generated_code, "r", theme="monokai", line_numbers=True)
        console.print(syntax)

    console.print(f"\n[dim]Execution time: {result.execution_time:.2f}s[/dim]")

    # Save result if requested
    if args.save_result:
        result_path = Path(args.save_result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"[dim]Result saved to: {result_path}[/dim]")

    # Return exit code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
