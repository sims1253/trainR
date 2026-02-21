#!/usr/bin/env python
"""Run skill optimization using GEPA."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from optimization import optimize_skill
from task_generator import TaskGenerator

console = Console()
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for optimization CLI."""
    parser = argparse.ArgumentParser(description="Optimize R testing skill with GEPA")
    parser.add_argument(
        "--seed-skill",
        default="skills/testing-r-packages-orig.md",
        help="Path to seed skill markdown",
    )
    parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task JSONs",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for optimization results",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=50,
        help="Maximum number of evaluations",
    )
    parser.add_argument(
        "--reflection-lm",
        default=None,
        help="LiteLLM model for reflection (default: LLM_MODEL_REFLECTION env or openai/glm-5)",
    )
    parser.add_argument(
        "--docker-image",
        default="posit-gskill-eval:latest",
        help="Docker image for evaluation",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load seed skill
    seed_path = Path(args.seed_skill)
    if not seed_path.exists():
        console.print(f"[red]Seed skill not found: {seed_path}[/red]")
        sys.exit(1)
    seed_skill = seed_path.read_text()

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
    config_table = f"""Seed Skill: {args.seed_skill}
Train Tasks: {len(train_tasks)}
Val Tasks: {len(val_tasks)}
Max Metric Calls: {args.max_metric_calls}
Reflection LM: {args.reflection_lm}
Docker Image: {args.docker_image}"""

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
            docker_image=args.docker_image,
            max_metric_calls=args.max_metric_calls,
            reflection_lm=args.reflection_lm,
            run_dir=str(run_dir),
        )

        # Show results
        console.print("\n[green bold]Optimization complete![/green bold]")

        # Handle both string and dict candidates
        best_skill = result.best_candidate
        if isinstance(best_skill, dict):
            best_skill = best_skill.get("skill", str(best_skill))

        # Get best score
        best_score = max(result.val_aggregate_scores) if result.val_aggregate_scores else 0.0
        console.print(f"\nBest candidate score: {best_score:.2%}")

        # Save best skill
        best_skill_path = run_dir / "best_skill.md"
        best_skill_path.write_text(best_skill)
        console.print(f"Best skill saved to: {best_skill_path}")

        # Show skill preview
        if args.verbose:
            console.print("\n[bold]Best Skill Preview:[/bold]")
            syntax = Syntax(best_skill[:2000], "markdown", theme="monokai")
            console.print(syntax)

        # Save full result
        result_path = run_dir / "result.json"
        result_data = {
            "best_candidate": best_skill,
            "best_score": best_score,
            "total_metric_calls": result.total_metric_calls,
            "num_candidates": len(result.candidates),
            "config": {
                "seed_skill": str(args.seed_skill),
                "train_tasks": len(train_tasks),
                "val_tasks": len(val_tasks),
                "max_metric_calls": args.max_metric_calls,
                "reflection_lm": args.reflection_lm,
            },
        }
        result_path.write_text(json.dumps(result_data, indent=2))
        console.print(f"Full result saved to: {result_path}")

    except Exception as e:
        console.print(f"\n[red]Optimization failed: {e}[/red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
