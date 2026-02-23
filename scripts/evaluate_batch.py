"""Run batch evaluations with parallel execution and YAML configuration."""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from config import get_llm_config
from evaluation.config import EvaluationConfig
from evaluation.pi_runner import DockerPiRunner, DockerPiRunnerConfig
from task_generator import TaskGenerator

console = Console()
logger = logging.getLogger(__name__)


def get_required_api_key_for_model(model: str) -> tuple[str | None, str | None]:
    """Get the required API key environment variable for a model.

    Args:
        model: Model name (either a short name from llm.yaml or full LiteLLM path)

    Returns:
        Tuple of (env_var_name, env_var_value) or (None, None) if no key required.
    """
    llm_config = get_llm_config()

    # Try to resolve as a model name from llm.yaml
    try:
        model_cfg = llm_config.get_model_config(model)
        env_var = model_cfg.get("api_key_env")
        if env_var:
            return env_var, os.environ.get(env_var)
    except ValueError:
        pass

    # Fallback: infer from LiteLLM prefix in model string
    provider_key_mapping = {
        "openrouter/": "OPENROUTER_API_KEY",
        "opencode/": "OPENCODE_API_KEY",
        "zai/": "Z_AI_API_KEY",
        "openai/": "OPENAI_API_KEY",
        "anthropic/": "ANTHROPIC_API_KEY",
        "gemini/": "GEMINI_API_KEY",
    }

    for prefix, key_name in provider_key_mapping.items():
        if model.startswith(prefix):
            return key_name, os.environ.get(key_name)

    # No provider prefix - assume no key required or will be handled by LiteLLM
    return None, None


def load_config(config_path: Path, cli_overrides: dict) -> EvaluationConfig:
    """Load config from YAML and apply CLI overrides."""
    config = EvaluationConfig.from_yaml(config_path)

    # Apply CLI overrides
    if cli_overrides.get("workers") is not None:
        config.workers.count = cli_overrides["workers"]
    if cli_overrides.get("model") is not None:
        config.model.task = cli_overrides["model"]
    if cli_overrides.get("no_skill"):
        config.skill.no_skill = True
        config.skill.path = None
    if cli_overrides.get("splits") is not None:
        config.tasks.splits = cli_overrides["splits"]

    return config


def run_batch_evaluation(config: EvaluationConfig, cli_args: dict | None = None) -> dict:
    """Run batch evaluation with the given configuration."""
    # Load tasks
    generator = TaskGenerator(Path(config.tasks.dir))
    tasks = []
    for split in config.tasks.splits:
        tasks.extend(generator.load_all_tasks(split=split))

    if not tasks:
        console.print("[red]No tasks found for specified splits[/red]")
        sys.exit(1)

    # Load skill content
    if config.skill.no_skill:
        skill_content = ""  # Empty = no skill baseline
        skill_name = "no_skill"
    elif config.skill.path:
        skill_path = Path(config.skill.path)
        if not skill_path.exists():
            console.print(f"[red]Skill file not found: {skill_path}[/red]")
            sys.exit(1)
        skill_content = skill_path.read_text()
        skill_name = skill_path.stem
    else:
        skill_content = ""
        skill_name = "no_skill"

    # Build task list with package directories
    packages_dir = Path("packages")
    task_items = []
    for task in tasks:
        package_dir = packages_dir / task.source_package
        if not package_dir.exists():
            console.print(f"[yellow]Warning: Package not found: {package_dir}[/yellow]")
            continue
        task_items.append(
            {
                "task": task,
                "package_dir": package_dir,
            }
        )

    # Apply max-tasks limit if specified
    if cli_args and cli_args.get("max_tasks"):
        task_items = task_items[: cli_args["max_tasks"]]
        console.print(f"[yellow]Limiting to {len(task_items)} tasks (--max-tasks)[/yellow]")

    if not task_items:
        console.print("[red]No valid tasks with existing packages[/red]")
        sys.exit(1)

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trajectories directory
    traj_dir = None
    if config.output.save_trajectories:
        traj_dir = output_dir / f"trajectories_{timestamp}"
        traj_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    console.print(
        f"\n[blue]Starting batch evaluation: {len(task_items)} tasks, {config.workers.count} workers[/blue]"
    )
    console.print(f"[dim]Model: {config.model.task}, Skill: {skill_name}[/dim]")

    runner_config = DockerPiRunnerConfig(
        model=config.model.task,
        timeout=config.execution.timeout,
        docker_image=config.execution.docker_image,
    )
    runner = DockerPiRunner(runner_config)

    results = []
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        eval_task = progress.add_task("Evaluating...", total=len(task_items))

        def evaluate_single(task_item: dict) -> dict:
            """Run evaluation for a single task."""
            task = task_item["task"]
            package_dir = task_item["package_dir"]

            result = runner.run_evaluation(
                skill_content=skill_content,
                task_instruction=task.instruction,
                task_context=task.context,
                package_dir=package_dir,
                model=config.model.task,
            )

            return {
                "task": task,
                "package_dir": package_dir,
                "result": result,
            }

        # Run evaluations in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=config.workers.count) as executor:
            futures = {executor.submit(evaluate_single, item): item for item in task_items}

            for future in as_completed(futures):
                progress.advance(eval_task)
                try:
                    eval_result = future.result()
                    results.append(eval_result)
                except Exception as e:
                    item = futures[future]
                    task = item["task"]
                    results.append(
                        {
                            "task": task,
                            "package_dir": item["package_dir"],
                            "result": {
                                "success": False,
                                "score": 0.0,
                                "output": "",
                                "error": str(e),
                                "execution_time": 0,
                            },
                        }
                    )

    total_time = time.time() - start_time

    # Sort results by task_id for consistent output
    results.sort(key=lambda x: x["task"].task_id)

    # Aggregate results
    passed = 0
    failed = 0
    output_results = []

    for eval_result in results:
        task = eval_result["task"]
        result = eval_result["result"]
        success = result.get("success", False)

        if success:
            passed += 1
        else:
            failed += 1

        result_entry = {
            "task_id": task.task_id,
            "success": success,
            "score": result.get("score", 1.0 if success else 0.0),
            "error": result.get("error"),
            "latency_s": result.get("execution_time", 0),
            "source_package": task.source_package,
            "difficulty": str(task.difficulty),
            "split": task.split,
        }

        # Save trajectory if enabled
        if config.output.save_trajectories and traj_dir and result.get("output"):
            traj_file = traj_dir / f"{task.task_id}.txt"
            traj_file.write_text(result["output"])

        output_results.append(result_entry)

    # Build final output
    output = {
        "timestamp": timestamp,
        "config": {
            "model": config.model.task,
            "skill": skill_name,
            "workers": config.workers.count,
            "splits": config.tasks.splits,
            "timeout": config.execution.timeout,
        },
        "summary": {
            "total": len(task_items),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(task_items) if task_items else 0,
            "total_time_s": total_time,
        },
        "results": output_results,
    }

    return output


def print_summary(output: dict) -> None:
    """Print evaluation summary table."""
    summary = output["summary"]

    table = Table(title="Batch Evaluation Results", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    pass_rate = summary["pass_rate"]
    color = "green" if pass_rate > 0.7 else "yellow" if pass_rate > 0.3 else "red"

    table.add_row("Model", output["config"]["model"])
    table.add_row("Skill", output["config"]["skill"])
    table.add_row("Workers", str(output["config"]["workers"]))
    table.add_row("Total Tasks", str(summary["total"]))
    table.add_row("Passed", f"[green]{summary['passed']}[/green]")
    table.add_row("Failed", f"[red]{summary['failed']}[/red]")
    table.add_row("Pass Rate", f"[{color}]{pass_rate:.1%}[/{color}]")
    table.add_row("Total Time", f"{summary['total_time_s']:.1f}s")

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch evaluations with parallel execution")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Override number of parallel workers",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Override task execution model",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Run without skill (baseline)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Override task splits (comma-separated)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to run (for testing)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Override output file path",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Load config first to determine which model(s) will be used
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    cli_overrides = {
        "workers": args.workers,
        "model": args.model,
        "no_skill": args.no_skill,
        "splits": args.splits.split(",") if args.splits else None,
        "max_tasks": args.max_tasks,
    }

    config = load_config(config_path, cli_overrides)

    # Provider-aware API key validation
    # Only require API key for the model that will actually be used
    models_to_check = config.model.get_models()
    missing_keys = []
    for model in models_to_check:
        env_var, api_key = get_required_api_key_for_model(model)
        if env_var and not api_key:
            missing_keys.append((model, env_var))

    if missing_keys:
        for model, key_name in missing_keys:
            console.print(
                f"[red]{key_name} not set in environment (required for model: {model})[/red]"
            )
        console.print("[dim]Set the required environment variables for the selected models[/dim]")
        sys.exit(1)

    # Show configuration
    console.print(
        Panel(
            f"Config: {args.config}\n"
            f"Model: {config.model.task}\n"
            f"Skill: {'no_skill' if config.skill.no_skill else config.skill.path}\n"
            f"Workers: {config.workers.count}\n"
            f"Splits: {', '.join(config.tasks.splits)}",
            title="Evaluation Configuration",
            border_style="blue",
        )
    )

    # Run evaluation
    output = run_batch_evaluation(config, cli_overrides)

    # Save results
    output_path = Path(args.output) if args.output else None
    if not output_path:
        output_dir = Path(config.output.dir)
        timestamp = output["timestamp"]
        model = config.model.task
        skill = config.get_skill_name()
        filename = f"eval_{model}_{skill}_{timestamp}.json"
        output_path = output_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    console.print(f"\n[green]Results saved to: {output_path}[/green]")

    # Print summary
    print_summary(output)


if __name__ == "__main__":
    main()
