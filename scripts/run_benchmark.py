#!/usr/bin/env python
"""Run benchmark: evaluate task set across multiple models."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from benchmark.schema import BenchmarkResult, BenchmarkRun
from evaluation import DockerPiRunnerConfig, EvaluationSandbox
from task_generator import TaskGenerator

console = Console()
logger = logging.getLogger(__name__)


def load_model_config(config_path: Path) -> dict[str, Any]:
    """Load model configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_completed_pairs(results_dir: Path) -> set[tuple[str, str]]:
    """Get already-completed (model, task_id) pairs for resumption."""
    completed = set()
    for result_file in results_dir.glob("*.json"):
        try:
            data = json.loads(result_file.read_text())
            for r in data.get("results", []):
                completed.add((r["model"], r["task_id"]))
        except Exception:
            continue
    return completed


def run_benchmark(
    config: dict,
    tasks_dir: Path,
    skill_path: Path,
    output_dir: Path,
    resume: bool = True,
) -> BenchmarkRun:
    """Run benchmark across all models and tasks."""
    models = config["models"]
    settings = config.get("settings", {})

    timeout = settings.get("timeout", 600)
    docker_image = settings.get("docker_image", "posit-gskill-eval:latest")
    task_splits = settings.get("task_splits", ["dev", "held_out"])
    repeats = settings.get("repeats", 1)
    save_trajectories = settings.get("save_trajectories", True)

    # Load tasks
    generator = TaskGenerator(tasks_dir)
    tasks = []
    for split in task_splits:
        tasks.extend(generator.load_all_tasks(split=split))

    if not tasks:
        console.print("[red]No tasks found for specified splits[/red]")
        sys.exit(1)

    # Load skill
    skill_prompt = skill_path.read_text()

    # Setup results
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory directory
    traj_dir = results_dir / "trajectories"
    if save_trajectories:
        traj_dir.mkdir(parents=True, exist_ok=True)

    # Check for completed pairs (resumption)
    completed = get_completed_pairs(output_dir) if resume else set()
    if completed:
        console.print(f"[dim]Resuming: {len(completed)} task-model pairs already complete[/dim]")

    run = BenchmarkRun(
        run_id=run_id,
        models=[m["name"] for m in models],
        task_count=len(tasks),
        skill_version=str(skill_path),
        config=config.get("settings", {}),
    )

    total_evals = len(models) * len(tasks) * repeats
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        eval_task = progress.add_task("Benchmarking...", total=total_evals)

        for model_config in models:
            model_name = model_config["name"]
            model_env = model_config.get("env", {})

            console.print(f"\n[bold blue]Model: {model_name}[/bold blue]")

            # Set model-specific environment
            env_backup = {}
            for key, value in model_env.items():
                env_backup[key] = os.environ.get(key)
                os.environ[key] = value

            runner_config = DockerPiRunnerConfig(
                docker_image=docker_image,
                timeout=timeout,
            )
            sandbox = EvaluationSandbox(runner_config=runner_config)

            for task in tasks:
                for repeat_idx in range(repeats):
                    pair = (model_name, task.task_id)
                    if pair in completed:
                        skipped += 1
                        progress.advance(eval_task)
                        continue

                    progress.update(
                        eval_task,
                        description=f"{model_name} / {task.task_id}",
                    )

                    start = time.time()
                    result = sandbox.evaluate_task(task, skill_prompt)
                    latency = time.time() - start

                    # Save trajectory if enabled
                    traj_path = None
                    if save_trajectories and result.generated_code:
                        traj_file = traj_dir / model_name / f"{task.task_id}.txt"
                        traj_file.parent.mkdir(parents=True, exist_ok=True)
                        traj_file.write_text(result.generated_code)
                        traj_path = str(traj_file.relative_to(results_dir))

                    bench_result = BenchmarkResult(
                        task_id=task.task_id,
                        model=model_name,
                        passed=result.success,
                        score=result.score,
                        latency_s=latency,
                        error_category=str(result.failure_category)
                        if result.failure_category
                        else None,
                        error_message=result.error_message,
                        token_usage=result.token_usage or {},
                        test_results=[
                            {"name": t.name, "passed": t.passed, "message": t.message}
                            for t in result.test_results
                        ],
                        trajectory_path=traj_path,
                        repeat_index=repeat_idx,
                    )

                    run.add_result(bench_result)
                    progress.advance(eval_task)

            # Restore environment
            for key, value in env_backup.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    # Save results
    run.save(results_dir / "benchmark_results.json")
    console.print(f"\n[green]Results saved to {results_dir}/benchmark_results.json[/green]")

    if skipped:
        console.print(f"[dim]Skipped {skipped} already-completed evaluations[/dim]")

    return run


def print_summary(run: BenchmarkRun) -> None:
    """Print a summary table of benchmark results."""
    table = Table(title="Benchmark Results", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Tasks", justify="right")

    for model in run.models:
        model_results = [r for r in run.results if r.model == model]
        pass_rate = run.pass_rate(model)
        avg_lat = run.avg_latency(model)

        color = "green" if pass_rate > 0.7 else "yellow" if pass_rate > 0.3 else "red"
        table.add_row(
            model,
            f"[{color}]{pass_rate:.1%}[/{color}]",
            f"{avg_lat:.1f}s",
            str(len(model_results)),
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark across models")
    parser.add_argument(
        "--config",
        default="configs/benchmark_models.yaml",
        help="Path to benchmark model config",
    )
    parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task JSONs",
    )
    parser.add_argument(
        "--skill",
        default="skills/testing-r-packages-orig.md",
        help="Path to skill file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already-completed task-model pairs",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Check API key
    if not os.environ.get("Z_AI_API_KEY"):
        console.print("[red]Z_AI_API_KEY not set in environment[/red]")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        sys.exit(1)

    config = load_model_config(config_path)

    output_dir = Path(
        args.output_dir or config.get("settings", {}).get("results_dir", "results/benchmarks")
    )

    console.print(
        Panel(
            f"Models: {', '.join(m['name'] for m in config['models'])}\n"
            f"Tasks dir: {args.tasks_dir}\n"
            f"Skill: {args.skill}\n"
            f"Output: {output_dir}",
            title="Benchmark Configuration",
            border_style="blue",
        )
    )

    run = run_benchmark(
        config=config,
        tasks_dir=Path(args.tasks_dir),
        skill_path=Path(args.skill),
        output_dir=output_dir,
        resume=not args.no_resume,
    )

    print_summary(run)


if __name__ == "__main__":
    main()
