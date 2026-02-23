"""Command implementations for the CLI."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_optimize(args: Any) -> None:
    """Run skill optimization using GEPA."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Import here to avoid circular imports
    from optimization import optimize_skill
    from task_generator import TaskGenerator

    # Load seed skill
    seed_path = Path(args.seed_skill)
    if not seed_path.exists():
        console.print(f"[red]Seed skill not found: {seed_path}[/red]")
        sys.exit(1)
    seed_skill = seed_path.read_text()
    seed_skill_name = seed_path.stem

    # Load tasks
    tasks_dir = Path(args.tasks_dir)
    generator = TaskGenerator(tasks_dir)

    train_tasks = generator.load_all_tasks(split="train")
    val_tasks = generator.load_all_tasks(split="dev")

    if not train_tasks:
        console.print("[red]No training tasks found[/red]")
        sys.exit(1)
    if not val_tasks:
        console.print("[yellow]No validation tasks found, using train tasks[/yellow]")
        val_tasks = train_tasks

    # Show configuration
    config_table = f"""Seed Skill: {seed_skill_name}
Train Tasks: {len(train_tasks)}
Val Tasks: {len(val_tasks)}
Max Metric Calls: {args.max_metric_calls}"""

    console.print(Panel(config_table, title="Optimization Configuration", border_style="blue"))

    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run optimization
    console.print("\n[blue]Starting optimization...[/blue]")

    try:
        result = optimize_skill(
            seed_skill=seed_skill,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            docker_image="posit-gskill-eval:latest",
            max_metric_calls=args.max_metric_calls,
            run_dir=str(run_dir),
        )

        console.print("\n[green bold]Optimization complete![/green bold]")

        # Handle both string and dict candidates
        best_skill = result.best_candidate
        if isinstance(best_skill, dict):
            best_skill = best_skill.get("skill", str(best_skill))

        best_score = max(result.val_aggregate_scores) if result.val_aggregate_scores else 0.0
        console.print(f"\nBest candidate score: {best_score:.2%}")

        # Save best skill
        best_skill_path = run_dir / "best_skill.md"
        best_skill_path.write_text(best_skill)
        console.print(f"Best skill saved to: {best_skill_path}")

    except Exception as e:
        console.print(f"\n[red]Optimization failed: {e}[/red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_generate(args: Any) -> None:
    """Generate tasks from patterns."""
    from task_generator import TaskGenerator

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TaskGenerator(output_dir)
    console.print(f"[green]Task generator initialized at {output_dir}[/green]")
    console.print(
        "[yellow]Use task_generator module directly for full generation capabilities[/yellow]"
    )


def run_evaluate(args: Any) -> None:
    """Evaluate a skill against tasks."""
    from evaluation.test_runner import TestRunner
    from task_generator import TaskGenerator

    skill_path = Path(args.skill)
    if not skill_path.exists():
        console.print(f"[red]Skill file not found: {skill_path}[/red]")
        sys.exit(1)

    skill_content = skill_path.read_text()
    tasks_dir = Path(args.tasks_dir)

    generator = TaskGenerator(tasks_dir)
    tasks = generator.load_all_tasks()

    if not tasks:
        console.print(f"[red]No tasks found in {tasks_dir}[/red]")
        sys.exit(1)

    console.print(f"[blue]Evaluating skill against {len(tasks)} tasks...[/blue]")

    # Run evaluation
    runner = TestRunner()
    results = runner.evaluate_skill(skill_content, tasks)

    console.print(f"\n[green]Evaluation complete![/green]")
    console.print(f"Results: {json.dumps(results, indent=2)}")
