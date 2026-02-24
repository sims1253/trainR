"""Command implementations for the CLI."""

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

    TaskGenerator(output_dir)
    console.print(f"[green]Task generator initialized at {output_dir}[/green]")
    console.print(
        "[yellow]Use task_generator module directly for full generation capabilities[/yellow]"
    )


def run_evaluate(args: Any) -> None:
    """Evaluate a skill against tasks using the canonical runner API."""
    import bench.runner
    from bench.experiments import (
        ExecutionConfig,
        ExperimentConfig,
        ModelsConfig,
        OutputConfig,
        SkillConfig,
        TasksConfig,
        TaskSelectionMode,
    )
    from config import get_llm_config

    skill_path = Path(args.skill)
    if not skill_path.exists():
        console.print(f"[red]Skill file not found: {skill_path}[/red]")
        sys.exit(1)

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        console.print(f"[red]Tasks directory not found: {tasks_dir}[/red]")
        sys.exit(1)

    # Resolve default model from llm config
    llm_config = get_llm_config()
    default_model = llm_config.get_default_model("task_agent") or "gpt-oss-120b"

    # Build ExperimentConfig for this evaluation
    config = ExperimentConfig(
        name=f"evaluate_{skill_path.stem}",
        description=f"Evaluation run for {skill_path.name}",
        tasks=TasksConfig(
            selection=TaskSelectionMode.ALL,
            dir=str(tasks_dir),
        ),
        models=ModelsConfig(
            names=[default_model],
        ),
        skill=SkillConfig(
            path=str(skill_path),
        ),
        execution=ExecutionConfig(
            timeout=600,
            docker_image="posit-gskill-eval:latest",
            repeats=1,
            parallel_workers=1,
            save_trajectories=True,
        ),
        output=OutputConfig(
            dir="results/evaluate",
            save_intermediate=True,
        ),
    )

    console.print(f"[blue]Evaluating skill: {skill_path.name}[/blue]")
    console.print(f"[blue]Tasks directory: {tasks_dir}[/blue]")
    console.print(f"[blue]Model: {default_model}[/blue]")
    console.print()

    try:
        # Run evaluation via canonical runner
        manifest = bench.runner.run(config)

        # Display results
        console.print("\n[green]Evaluation complete![/green]")

        # Summary
        console.print(
            Panel(
                f"Tasks: {manifest.summary.completed}/{manifest.summary.total_tasks}\n"
                f"Passed: {manifest.summary.passed}\n"
                f"Failed: {manifest.summary.failed}\n"
                f"Pass Rate: {manifest.summary.pass_rate:.1%}\n"
                f"Avg Score: {manifest.summary.avg_score:.2f}\n"
                f"Avg Latency: {manifest.summary.avg_latency_s:.1f}s",
                title="Summary",
                border_style="green",
            )
        )

        # Per-task results
        if manifest.results:
            console.print("\n[bold]Per-Task Results:[/bold]")
            for result in manifest.results:
                status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
                score_str = f" (score: {result.score:.2f})" if result.score else ""
                console.print(f"  {result.task_id}: {status}{score_str}")

        # Output location
        if manifest.results_path:
            console.print(f"\n[dim]Results saved to: {manifest.results_path}[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Invalid configuration: {e}[/red]")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)
