#!/usr/bin/env python
"""Run optimization with resume/budget/safety controls.

This script provides a robust optimization runner that supports:
- Resume from interrupted runs
- Budget caps (iterations, time, tokens, cost)
- Graceful interrupt handling (SIGINT/SIGTERM)
- Automatic checkpoint persistence

Usage:
    # Start new optimization
    uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml

    # Resume interrupted run
    uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml --resume

    # Run with budget limits
    uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml \\
        --max-iterations 100 --max-time 3600 --max-cost 10.0
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from bench.optimize import (
    BudgetConfig,
    OptimizationRun,
    OptimizationState,
    StopReason,
)

console = Console()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run optimization with resume/budget/safety controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new optimization
  uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml

  # Resume from a specific run directory
  uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml --resume

  # Run with budget limits
  uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml \\
      --max-iterations 100 --max-time 3600 --max-cost 10.0
        """,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to optimization config YAML",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (default: auto-generated or from config)",
    )

    # Resume control
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force start fresh even if checkpoint exists",
    )

    # Budget caps
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum optimization iterations",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Maximum wall-clock time in seconds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum total token usage",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Maximum cost in dollars",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N iterations (default: 5)",
    )

    # Legacy/compatibility arguments (for backward compatibility)
    parser.add_argument(
        "--seed-skill",
        default=None,
        help="Path to seed skill markdown (legacy)",
    )
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help="Directory containing task JSONs (legacy)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for optimization results (legacy)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Maximum number of evaluations (legacy, maps to max-iterations)",
    )
    parser.add_argument(
        "--reflection-lm",
        default=None,
        help="Model for GEPA reflection (uses inference gateway)",
    )
    parser.add_argument(
        "--model",
        default="glm-4.5",
        help="Model for task execution (default: glm-4.5)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models for multi-model optimization",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Start from empty skill (baseline optimization)",
    )
    parser.add_argument(
        "--aggregation",
        choices=["min", "mean", "weighted"],
        default="min",
        help="Score aggregation for multi-model (default: min)",
    )
    parser.add_argument(
        "--parallel",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Enable parallel evaluation (default: true)",
    )
    parser.add_argument(
        "--docker-image",
        default="posit-gskill-eval:latest",
        help="Docker image for evaluation",
    )

    # Simulation mode (for testing without Docker)
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (no Docker, fake evaluations)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load optimization configuration from YAML file.

    Args:
        config_path: Path to config YAML

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    return config


def merge_args_with_config(args: argparse.Namespace, config: dict) -> dict:
    """Merge command line arguments with config file.

    CLI arguments take precedence over config file values.

    Args:
        args: Parsed command line arguments
        config: Configuration from file

    Returns:
        Merged configuration
    """
    merged = dict(config)

    # Budget config - CLI takes precedence
    budget = merged.get("budget", {})
    if args.max_iterations is not None:
        budget["max_iterations"] = args.max_iterations
    elif args.max_metric_calls is not None:
        # Legacy compatibility
        budget["max_iterations"] = args.max_metric_calls
    if args.max_time is not None:
        budget["max_time_seconds"] = args.max_time
    if args.max_tokens is not None:
        budget["max_tokens"] = args.max_tokens
    if args.max_cost is not None:
        budget["max_cost"] = args.max_cost
    if args.checkpoint_interval is not None:
        budget["checkpoint_interval"] = args.checkpoint_interval
    merged["budget"] = budget

    # Run settings - CLI takes precedence
    if args.run_dir is not None:
        merged["run_dir"] = args.run_dir
    if args.seed_skill is not None:
        merged["seed_skill"] = args.seed_skill
    if args.tasks_dir is not None:
        merged["tasks_dir"] = args.tasks_dir
    if args.output_dir is not None:
        merged["output_dir"] = args.output_dir
    if args.reflection_lm is not None:
        merged["reflection_lm"] = args.reflection_lm
    if args.model is not None:
        merged["model"] = args.model
    if args.models is not None:
        merged["models"] = [m.strip() for m in args.models.split(",")]
    if args.docker_image is not None:
        merged["docker_image"] = args.docker_image

    # Boolean flags
    if args.no_skill:
        merged["no_skill"] = True
    merged["parallel"] = args.parallel
    merged["aggregation"] = args.aggregation

    return merged


def display_config_summary(config: dict, state: OptimizationState | None = None) -> None:
    """Display a summary of the optimization configuration.

    Args:
        config: Configuration dictionary
        state: Current optimization state (if resuming)
    """
    budget = config.get("budget", {})

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="dim")
    table.add_column("Value")

    # Target info
    target = config.get("target", {})
    table.add_row("Target Type", target.get("type", "skill"))
    if target.get("skill_name"):
        table.add_row("Skill Name", target["skill_name"])

    # Budget limits
    if budget.get("max_iterations"):
        table.add_row("Max Iterations", str(budget["max_iterations"]))
    if budget.get("max_time_seconds"):
        table.add_row("Max Time", f"{budget['max_time_seconds']}s")
    if budget.get("max_tokens"):
        table.add_row("Max Tokens", f"{budget['max_tokens']:,}")
    if budget.get("max_cost"):
        table.add_row("Max Cost", f"${budget['max_cost']:.2f}")
    table.add_row("Checkpoint Interval", f"every {budget.get('checkpoint_interval', 5)} iterations")

    # Model config
    models = config.get("models", [config.get("model", "default")])
    table.add_row("Models", ", ".join(models) if isinstance(models, list) else models)
    table.add_row("Aggregation", config.get("aggregation", "min"))

    # Resume info
    if state:
        table.add_row("", "")
        table.add_row("[bold]Resuming From[/bold]", "")
        table.add_row("Previous Iterations", str(state.iteration))
        table.add_row("Best Score", f"{state.best_score:.2%}")
        table.add_row("Elapsed Time", f"{state.budget_usage.elapsed_seconds:.1f}s")

    console.print(Panel(table, title="Optimization Configuration", border_style="blue"))


def display_results(state: OptimizationState) -> None:
    """Display final optimization results.

    Args:
        state: Final optimization state
    """
    console.print()

    # Summary panel
    summary = f"""[bold]Run ID:[/bold] {state.run_id}
[bold]Stop Reason:[/bold] {state.stop_reason.value if state.stop_reason else "unknown"}
[bold]Total Iterations:[/bold] {state.iteration}
[bold]Best Score:[/bold] {state.best_score:.2%}
[bold]Best Iteration:[/bold] {state.best_iteration}
[bold]Elapsed Time:[/bold] {state.budget_usage.elapsed_seconds:.1f}s
[bold]Total Tokens:[/bold] {state.budget_usage.total_tokens:,}
[bold]Total Cost:[/bold] ${state.budget_usage.total_cost:.2f}"""

    if state.error_message:
        summary += f"\n[red bold]Error:[/red bold] {state.error_message}"

    console.print(Panel(summary, title="Optimization Results", border_style="green"))

    # Trajectory summary
    if state.trajectory:
        console.print("\n[bold]Score Progress:[/bold]")

        # Show score progression (every 10th iteration or all if few)
        step = max(1, len(state.trajectory) // 10)
        for entry in state.trajectory[::step]:
            marker = "[green]*[/green]" if entry.is_best else " "
            console.print(
                f"  {marker} Iter {entry.iteration}: {entry.score:.2%}"
                + (" [dim](best)[/dim]" if entry.is_best else "")
            )


def create_run_directory(config: dict, args: argparse.Namespace) -> Path:
    """Create or get the run directory.

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Path to run directory
    """
    # Determine base output directory
    output_dir = config.get("output_dir", "results/optimization")

    # Generate run ID
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = config.get("target", {}).get("skill_name", "optimize")
        run_id = f"{target_name}_{timestamp}"
        run_dir = Path(output_dir) / run_id

    return run_dir


def run_optimization(config: dict, args: argparse.Namespace) -> tuple[OptimizationState, Path]:
    """Run the optimization with all controls.

    Args:
        config: Merged configuration
        args: Command line arguments

    Returns:
        Tuple of (Final optimization state, Run directory path)
    """
    # Create/get run directory
    run_dir = create_run_directory(config, args)

    # Build budget config
    budget_dict = config.get("budget", {})
    budget_config = BudgetConfig(
        max_iterations=budget_dict.get("max_iterations"),
        max_time_seconds=budget_dict.get("max_time_seconds"),
        max_tokens=budget_dict.get("max_tokens"),
        max_cost=budget_dict.get("max_cost"),
        checkpoint_interval=budget_dict.get("checkpoint_interval", 5),
    )

    # Determine target info
    target_config = config.get("target", {})
    target_type = target_config.get("type", "skill")
    target_fingerprint = f"{target_type}:{target_config.get('skill_name', 'default')}:1.0"

    # Create optimization run
    run = OptimizationRun(
        run_dir=run_dir,
        budget_config=budget_config,
        target_type=target_type,
        target_fingerprint=target_fingerprint,
    )

    # Start or resume
    if args.resume and not args.force and run.can_resume():
        console.print(f"[yellow]Resuming from {run_dir}[/yellow]")
        state = run.resume()
    else:
        if args.force and run.can_resume():
            console.print("[yellow]Force starting fresh (ignoring existing checkpoint)[/yellow]")
            # Backup existing checkpoint
            checkpoint_path = run_dir / "checkpoint.json"
            if checkpoint_path.exists():
                backup_path = (
                    run_dir / f"checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                shutil.copy(checkpoint_path, backup_path)
                console.print(f"[dim]Backed up checkpoint to {backup_path}[/dim]")
        state = run.start(metadata={"config": config})

    # Display configuration
    display_config_summary(config, state if args.resume else None)

    # Enter the run context (installs signal handlers)
    with run:
        optimize_skill_fn = None
        # Import optimization function (deferred to allow for missing dependencies)
        try:
            from optimization import optimize_skill
            from task_generator import TaskGenerator

            optimize_skill_fn = optimize_skill
            task_generator_cls = TaskGenerator
        except ImportError as e:
            console.print(f"[red]Failed to import optimization module: {e}[/red]")
            console.print("[yellow]Running in simulation mode (no actual optimization)[/yellow]")
            task_generator_cls = None

        # Load seed skill
        if config.get("no_skill"):
            seed_skill = ""
        else:
            seed_path = config.get("seed_skill")
            if seed_path and Path(seed_path).exists():
                seed_skill = Path(seed_path).read_text()
            else:
                seed_skill = ""

        # Load tasks
        tasks_dir = config.get("tasks_dir", "tasks")
        if task_generator_cls and Path(tasks_dir).exists():
            generator = task_generator_cls(tasks_dir)
            train_tasks = generator.load_all_tasks(split="train")
            val_tasks = generator.load_all_tasks(split="dev")
            if not val_tasks:
                val_tasks = train_tasks
        else:
            train_tasks = []
            val_tasks = []

        # Run optimization loop
        if not args.simulation and optimize_skill_fn and train_tasks:
            console.print("\n[blue]Starting optimization...[/blue]")

            # Create progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                disable=args.quiet,
            ) as progress:
                if budget_config.max_iterations:
                    task = progress.add_task(
                        "Optimizing",
                        total=budget_config.max_iterations,
                        completed=state.iteration,
                    )
                else:
                    task = progress.add_task("Optimizing", total=None)

                try:
                    # Run actual optimization
                    result = optimize_skill_fn(
                        seed_skill=seed_skill,
                        train_tasks=train_tasks,
                        val_tasks=val_tasks,
                        docker_image=config.get("docker_image", "posit-gskill-eval:latest"),
                        max_metric_calls=budget_config.max_iterations or 50,
                        reflection_lm=config.get("reflection_lm"),
                        model=config.get("model", "glm-4.5"),
                        models=config.get("models"),
                        aggregation=config.get("aggregation", "min"),
                        parallel_eval=config.get("parallel", True),
                        run_dir=str(run_dir),
                    )

                    # Record results into state
                    if hasattr(result, "candidates") and result.candidates:
                        for i, candidate in enumerate(result.candidates):
                            if isinstance(candidate, dict):
                                candidate_dict = candidate
                                candidate_str = candidate.get("skill", str(candidate))
                            else:
                                candidate_str = str(candidate)
                                candidate_dict = {"skill": candidate_str}

                            score = 0.0
                            if (
                                hasattr(result, "val_aggregate_scores")
                                and result.val_aggregate_scores
                            ) and i < len(result.val_aggregate_scores):
                                score = result.val_aggregate_scores[i]

                            # Compute hash
                            from hashlib import sha256

                            candidate_hash = sha256(candidate_str.encode()).hexdigest()[:16]

                            run.record(
                                candidate=candidate_dict,
                                candidate_hash=candidate_hash,
                                score=score,
                            )
                            progress.update(task, advance=1)

                            if run.should_checkpoint():
                                run.save_checkpoint()

                    # Get best candidate
                    if hasattr(result, "best_candidate"):
                        best = result.best_candidate
                        if isinstance(best, dict):
                            best_skill = best.get("skill", str(best))
                        else:
                            best_skill = str(best)

                        # Save best skill
                        best_skill_path = run_dir / "best_skill.md"
                        best_skill_path.write_text(best_skill)
                        console.print(f"\n[green]Best skill saved to: {best_skill_path}[/green]")

                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted by user[/yellow]")
                    run.save_checkpoint()
                except Exception as e:
                    logger.exception("Optimization failed")
                    state.mark_error(str(e))
        else:
            # Simulation mode
            console.print("\n[yellow]Running in simulation mode...[/yellow]")

            import time

            iterations = budget_config.max_iterations or 10
            for i in range(state.iteration, iterations):
                if run.is_complete():
                    break

                # Simulate iteration
                time.sleep(0.1)

                # Record fake result
                score = 0.5 + (i * 0.03) + (hash(i) % 10) / 100
                run.record(
                    candidate={"iteration": i, "content": f"Simulated candidate {i}"},
                    candidate_hash=f"sim_{i:04d}",
                    score=score,
                    tokens_used=1000,
                    cost=0.01,
                )

                if run.should_checkpoint():
                    run.save_checkpoint()

        # Finalize
        stop_reason = run.get_stop_reason()
        state = run.finalize(stop_reason)

    return state, run_dir


def save_final_results(state: OptimizationState, run_dir: Path) -> None:
    """Save final results to the run directory.

    Args:
        state: Final optimization state
        run_dir: Run directory
    """
    # Save detailed state
    state_path = run_dir / "optimization_state.json"
    with open(state_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    console.print(f"[dim]State saved to: {state_path}[/dim]")

    # Save trajectory as CSV for analysis
    if state.trajectory:
        import csv

        trajectory_path = run_dir / "trajectory.csv"
        with open(trajectory_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "score",
                    "is_best",
                    "tokens_used",
                    "cost",
                    "candidate_hash",
                ],
            )
            writer.writeheader()
            for entry in state.trajectory:
                writer.writerow(
                    {
                        "iteration": entry.iteration,
                        "score": entry.score,
                        "is_best": entry.is_best,
                        "tokens_used": entry.tokens_used,
                        "cost": entry.cost,
                        "candidate_hash": entry.candidate_hash,
                    }
                )
        console.print(f"[dim]Trajectory saved to: {trajectory_path}[/dim]")

    # Save best candidate
    if state.best_candidate:
        best_path = run_dir / "best_candidate.json"
        with open(best_path, "w") as f:
            json.dump(state.best_candidate, f, indent=2)
        console.print(f"[dim]Best candidate saved to: {best_path}[/dim]")

        # If it's a skill, also save as markdown
        if state.target_type == "skill" and "content" in state.best_candidate:
            skill_path = run_dir / "best_skill.md"
            skill_path.write_text(state.best_candidate["content"])
            console.print(f"[dim]Best skill saved to: {skill_path}[/dim]")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load and merge config
        config = load_config(args.config)
        config = merge_args_with_config(args, config)

        # Run optimization
        state, run_dir = run_optimization(config, args)

        # Display results
        display_results(state)

        # Save final results
        save_final_results(state, run_dir)

        # Return appropriate exit code
        if state.stop_reason == StopReason.ERROR:
            return 1
        elif state.stop_reason == StopReason.INTERRUPTED:
            return 130  # Standard exit code for SIGINT
        return 0

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
