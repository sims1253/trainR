#!/usr/bin/env python
"""Run benchmark: evaluate task set across multiple models.

Quick test example:
    uv run python scripts/run_benchmark.py --task tasks/mined/tidyverse_readr_1615.json --worker-model glm-5-free --debug
"""

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
from config import get_llm_config
from evaluation import DockerPiRunnerConfig, EvaluationSandbox
from task_generator import TaskGenerator

console = Console()
logger = logging.getLogger(__name__)


def resolve_model(model_name: str) -> dict:
    """Resolve a model name to its full configuration from llm.yaml."""
    llm_config = get_llm_config()
    try:
        cfg = llm_config.get_model_config(model_name)
        api_key = llm_config.get_model_api_key(model_name)
        env_var = cfg.get("api_key_env")
        return {
            "name": model_name,
            "task_model": llm_config.get_litellm_model(model_name),
            "reflection_model": llm_config.get_litellm_model(model_name),
            "env": {env_var: api_key} if env_var and api_key else {},
            "capabilities": {
                k: v
                for k, v in cfg.items()
                if k not in ["name", "id", "provider", "base_url", "api_key_env", "litellm_prefix"]
            },
            "enabled": True,
        }
    except ValueError:
        # Model not in config - try as raw model ID
        return {
            "name": model_name,
            "task_model": model_name,
            "reflection_model": model_name,
            "env": {},
            "capabilities": {},
            "enabled": True,
        }


class TraceLogger:
    """Logs all LLM request/response traces for debugging."""

    def __init__(self, trace_dir: Path, enabled: bool = True):
        self.trace_dir = trace_dir
        self.enabled = enabled
        if enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def log(self, model: str, task_id: str, request: dict, response: dict):
        """Log a request/response trace."""
        if not self.enabled:
            return

        trace_file = self.trace_dir / f"{task_id}_{model}_{datetime.now().strftime('%H%M%S')}.json"
        trace_file.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "task_id": task_id,
                    "request": request,
                    "response": response,
                },
                indent=2,
                default=str,
            )
        )


def load_model_config(config_path: Path) -> dict[str, Any]:
    """Load model configuration from YAML.

    Handles both old format (models list at root) and new unified format
    (with models, tasks, judge, skill, settings sections).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # If this is the new unified format, models is already a key
    if "models" in config:
        return config

    # Old format: models are at the root level
    # Wrap in expected structure for backward compatibility
    return {
        "models": config if isinstance(config, list) else [config],
        "settings": {},
        "tasks": {"selection": "splits", "splits": ["dev", "held_out"]},
    }


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


def load_tasks(config: dict, tasks_dir: Path) -> list:
    """Load tasks based on config selection mode."""
    generator = TaskGenerator(tasks_dir)
    task_config = config.get("tasks", {})
    selection = task_config.get("selection", "splits")

    if selection == "all":
        return generator.load_all_tasks()

    elif selection == "splits":
        splits = task_config.get("splits", ["dev", "held_out"])
        tasks = []
        for split in splits:
            tasks.extend(generator.load_all_tasks(split=split))
        return tasks

    elif selection == "categories":
        # Load all tasks and filter by categories
        all_tasks = generator.load_all_tasks()
        categories = task_config.get("categories", {})
        task_types = set(categories.get("task_types", []))
        difficulties = set(categories.get("difficulties", []))

        filtered = []
        for task in all_tasks:
            # Check task_type if specified
            if task_types:
                task_type = getattr(task, "task_type", None) or getattr(
                    getattr(task, "metadata", {}), "task_type", None
                )
                if task_type and task_type not in task_types:
                    continue

            # Check difficulty if specified
            if difficulties:
                difficulty = getattr(task, "difficulty", None) or getattr(
                    getattr(task, "metadata", {}), "difficulty", None
                )
                if difficulty and difficulty not in difficulties:
                    continue

            filtered.append(task)
        return filtered

    elif selection == "files":
        # Load specific task files
        tasks = []
        for file_path in task_config.get("files", []):
            path = Path(file_path)
            if not path.is_absolute() and not str(path).startswith(str(tasks_dir)):
                path = tasks_dir / path
            if path.exists():
                data = json.loads(path.read_text())
                # Derive source_package from source.repo if not present
                if "source_package" not in data and "source" in data:
                    source = data["source"]
                    if isinstance(source, dict):
                        repo = source.get("repo", "")
                    else:
                        repo = getattr(source, "repo", "")
                    # tidyverse/readr -> readr
                    source_package = repo.split("/")[-1] if "/" in str(repo) else str(repo)
                    data["source_package"] = source_package
                # Create task object with ALL fields from the JSON (SWE-bench fields included)
                task = type(
                    "Task",
                    (),
                    {
                        "task_id": data.get("task_id", path.stem),
                        "split": data.get("split", "custom"),
                        **data.get("task", {}),
                        **data.get("metadata", {}),
                        # Include all raw fields for SWE-bench evaluation
                        "test_patch": data.get("test_patch", ""),
                        "patch": data.get("patch", ""),
                        "tests": data.get("tests", {}),
                        "source": data.get("source", {}),
                        **data,
                    },
                )()
                tasks.append(task)
        return tasks

    elif selection == "task_id":
        # Load single task by ID
        task_id = task_config.get("task_id")
        if not task_id:
            return []
        # Search for task file by ID
        for task_file in tasks_dir.glob("**/*.json"):
            data = json.loads(task_file.read_text())
            if data.get("task_id") == task_id:
                task = type(
                    "Task",
                    (),
                    {
                        "task_id": data.get("task_id", task_file.stem),
                        "split": data.get("split", "custom"),
                        **data.get("task", {}),
                        **data.get("metadata", {}),
                        # Include all raw fields for SWE-bench evaluation
                        "test_patch": data.get("test_patch", ""),
                        "patch": data.get("patch", ""),
                        "tests": data.get("tests", {}),
                        "source": data.get("source", {}),
                        **data,
                    },
                )()
                return [task]
        return []

    return []


def run_benchmark(
    config: dict,
    tasks_dir: Path,
    skill_path: Path | None,
    output_dir: Path,
    resume: bool = True,
    trace_logger: TraceLogger | None = None,
) -> BenchmarkRun:
    """Run benchmark across all models and tasks."""
    models = config["models"]
    settings = config.get("settings", {})

    timeout = settings.get("timeout", 600)
    docker_image = settings.get("docker_image", "posit-gskill-eval:latest")
    repeats = settings.get("repeats", 1)
    save_trajectories = settings.get("save_trajectories", True)

    # Load tasks using the new task selection system
    tasks = load_tasks(config, tasks_dir)

    if not tasks:
        console.print("[red]No tasks found for specified selection[/red]")
        sys.exit(1)

    # Load skill (optional - null means no skill, use raw task instruction)
    skill_prompt = skill_path.read_text() if skill_path else None

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

    # Filter to enabled models only
    enabled_models = [m for m in models if m.get("enabled", True)]

    run = BenchmarkRun(
        run_id=run_id,
        models=[m["name"] for m in enabled_models],
        task_count=len(tasks),
        skill_version=str(skill_path) if skill_path else "none",
        config=config.get("settings", {}),
    )

    total_evals = len(enabled_models) * len(tasks) * repeats
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        eval_task = progress.add_task("Benchmarking...", total=total_evals)

        for model_config in enabled_models:
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

            # Resolve model to get the actual LiteLLM model string
            litellm_model = model_config.get("task_model", model_name)
            console.print(f"  Using model: {litellm_model}")

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
                    result = sandbox.evaluate_task(task, skill_prompt, model=model_name)
                    latency = time.time() - start

                    # Log trace if enabled
                    if trace_logger:
                        trace_logger.log(
                            model=model_name,
                            task_id=task.task_id,
                            request={"skill": str(skill_path), "task_id": task.task_id},
                            response={
                                "success": result.success,
                                "score": result.score,
                                "error": result.error_message,
                            },
                        )

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
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cache R/W", justify="right")
    table.add_column("Tasks", justify="right")

    for model in run.models:
        model_results = [r for r in run.results if r.model == model]
        pass_rate = run.pass_rate(model)
        avg_lat = run.avg_latency(model)

        # Aggregate token usage for this model
        total_input = sum(r.token_usage.get("input_tokens", 0) for r in model_results)
        total_output = sum(r.token_usage.get("output_tokens", 0) for r in model_results)
        total_cache_read = sum(r.token_usage.get("cache_read_tokens", 0) for r in model_results)
        total_cache_write = sum(r.token_usage.get("cache_write_tokens", 0) for r in model_results)

        color = "green" if pass_rate > 0.7 else "yellow" if pass_rate > 0.3 else "red"
        table.add_row(
            model,
            f"[{color}]{pass_rate:.1%}[/{color}]",
            f"{avg_lat:.1f}s",
            f"{total_input:,}",
            f"{total_output:,}",
            f"{total_cache_read:,}/{total_cache_write:,}",
            str(len(model_results)),
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark across models")
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark config",
    )
    parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task JSONs",
    )
    parser.add_argument(
        "--skill",
        default=None,
        help="Path to skill file (default: from config)",
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

    # Quick test options
    parser.add_argument(
        "--task",
        help="Run single task by task_id or file path (for quick testing)",
    )
    parser.add_argument(
        "--judge-model",
        help="Override judge model (for quick testing)",
    )
    parser.add_argument(
        "--worker-model",
        help="Run only this model (for quick testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full trace logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose or args.debug else logging.INFO,
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

    # Resolve model names (strings) to full model configurations from llm.yaml
    model_names = config.get("models", [])
    resolved_models = []
    for model_entry in model_names:
        if isinstance(model_entry, str):
            # It's a model name - resolve from llm.yaml
            resolved_models.append(resolve_model(model_entry))
        else:
            # It's already a full config dict - keep as-is
            resolved_models.append(model_entry)
    config["models"] = resolved_models

    # Handle --task option (single task for testing)
    if args.task:
        # Override task selection to use single task
        task_path = Path(args.task)
        if task_path.exists():
            config["tasks"] = {"selection": "files", "files": [str(task_path)]}
        else:
            # Treat as task_id, find the file
            config["tasks"] = {"selection": "task_id", "task_id": args.task}

    # Handle --worker-model option (single model for testing)
    if args.worker_model:
        # Try to find this model in the resolved config first
        matching_models = [
            m
            for m in config.get("models", [])
            if m.get("name") == args.worker_model or m.get("task_model") == args.worker_model
        ]

        if matching_models:
            config["models"] = matching_models
            for m in config["models"]:
                m["enabled"] = True
        else:
            # Try to resolve from llm.yaml
            try:
                config["models"] = [resolve_model(args.worker_model)]
            except ValueError:
                # Fallback - create a temporary config for unknown model
                task_model = args.worker_model
                if task_model.startswith("opencode/"):
                    env = {"OPENCODE_API_KEY": os.environ.get("OPENCODE_API_KEY", "")}
                elif task_model.startswith("openrouter/"):
                    env = {"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")}
                else:
                    env = {}

                config["models"] = [
                    {
                        "name": args.worker_model.replace("/", "-"),
                        "task_model": task_model,
                        "reflection_model": task_model,
                        "env": env,
                        "enabled": True,
                    }
                ]

    # Handle --judge-model option
    if args.judge_model:
        if "judge" not in config:
            config["judge"] = {}
        config["judge"]["mode"] = "single"
        config["judge"]["single"] = args.judge_model  # Now just a string

    # Handle --debug option
    if args.debug:
        if "settings" not in config:
            config["settings"] = {}
        config["settings"]["log_level"] = "DEBUG"
        config["settings"]["save_all_traces"] = True

    # Resolve judge model names to litellm strings
    judge_config = config.get("judge", {})
    if judge_config.get("mode") == "single":
        single_judge = judge_config.get("single")
        if isinstance(single_judge, str):
            # It's a model name - resolve from llm.yaml
            get_llm_config()
            try:
                judge_model_cfg = resolve_model(single_judge)
                config["judge"]["single"] = {
                    "model": judge_model_cfg["task_model"],
                    "env": judge_model_cfg["env"],
                }
            except ValueError:
                # Not found - use as-is
                config["judge"]["single"] = {"model": single_judge, "env": {}}
    elif judge_config.get("mode") == "ensemble":
        # Resolve ensemble judges
        ensemble = judge_config.get("ensemble", {})
        judge_names = ensemble.get("judges", [])
        resolved_judges = []
        get_llm_config()
        for judge_name in judge_names:
            if isinstance(judge_name, str):
                try:
                    judge_model_cfg = resolve_model(judge_name)
                    resolved_judges.append(
                        {
                            "name": judge_name,
                            "model": judge_model_cfg["task_model"],
                            "env": judge_model_cfg["env"],
                        }
                    )
                except ValueError:
                    resolved_judges.append(
                        {
                            "name": judge_name,
                            "model": judge_name,
                            "env": {},
                        }
                    )
            else:
                resolved_judges.append(judge_name)
        config["judge"]["ensemble"]["judges"] = resolved_judges

    # Use skill from config if not specified via CLI (null means no skill)
    skill_config = args.skill if args.skill else config.get("skill")
    skill_path = Path(skill_config) if skill_config else None

    output_dir = Path(
        args.output_dir or config.get("settings", {}).get("results_dir", "results/benchmarks")
    )

    # Setup trace logger if enabled
    trace_logger = None
    if config.get("settings", {}).get("save_all_traces", False):
        trace_dir = output_dir / "traces"
        trace_logger = TraceLogger(trace_dir, enabled=True)

    # Filter to enabled models for display
    enabled_models = [m for m in config.get("models", []) if m.get("enabled", True)]

    console.print(
        Panel(
            f"Models: {', '.join(m['name'] for m in enabled_models)}\n"
            f"Tasks dir: {args.tasks_dir}\n"
            f"Skill: {skill_path if skill_path else '(none)'}\n"
            f"Output: {output_dir}",
            title="Benchmark Configuration",
            border_style="blue",
        )
    )

    run = run_benchmark(
        config=config,
        tasks_dir=Path(args.tasks_dir),
        skill_path=skill_path,
        output_dir=output_dir,
        resume=not args.no_resume,
        trace_logger=trace_logger,
    )

    print_summary(run)


if __name__ == "__main__":
    main()
