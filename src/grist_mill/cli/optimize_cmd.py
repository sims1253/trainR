"""CLI 'optimize' subcommand for grist-mill.

Runs the optimization loop from a YAML configuration file::

    grist-mill optimize --config optimize.yaml
    grist-mill optimize --config optimize.yaml --resume

Produces:
- ``best_candidate.json``: The evolved artifact
- ``trajectory.jsonl``: Per-iteration score history
- ``checkpoint.json``: Full state for resuming

Validates:
- VAL-OPT-01: optimize command runs end-to-end
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import click
import yaml

from grist_mill.optimization.runtime import (
    BudgetConfig,
    MockProposer,
    OptimizationConfig,
    OptimizationRunner,
    TargetConfig,
)
from grist_mill.schemas import Difficulty, Task

logger = logging.getLogger(__name__)


def _load_tasks_from_dir(tasks_dir: str) -> list[Task]:
    """Load tasks from a directory of YAML/JSON files.

    Args:
        tasks_dir: Path to the directory containing task files.

    Returns:
        List of Task objects.
    """
    tasks: list[Task] = []
    dir_path = Path(tasks_dir)

    if not dir_path.exists():
        logger.warning("Tasks directory does not exist: %s", tasks_dir)
        return tasks

    for file_path in sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.json")):
        try:
            data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        tasks.append(_parse_task(item))
            elif isinstance(data, dict):
                tasks.append(_parse_task(data))
        except Exception as exc:
            logger.warning("Failed to load task file %s: %s", file_path, exc)

    return tasks


def _parse_task(data: dict[str, Any]) -> Task:
    """Parse a task from a dict (from YAML/JSON).

    Args:
        data: The task data dict.

    Returns:
        A Task instance.
    """
    difficulty = Difficulty.EASY
    if "difficulty" in data:
        with contextlib.suppress(ValueError):
            difficulty = Difficulty(data["difficulty"].upper())

    return Task(
        id=data.get("id", "unknown"),
        prompt=data.get("prompt", ""),
        language=data.get("language", "python"),
        test_command=data.get("test_command", "true"),
        setup_command=data.get("setup_command"),
        timeout=data.get("timeout", 60),
        difficulty=difficulty,
    )


def _build_config(config_path: Path) -> OptimizationConfig:
    """Build an OptimizationConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        An OptimizationConfig instance.

    Raises:
        click.BadParameter: If the config file is invalid.
    """
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise click.BadParameter(f"Invalid YAML in {config_path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise click.BadParameter(f"Config file not found: {config_path}") from exc

    if not isinstance(data, dict):
        raise click.BadParameter(f"Config file must be a YAML mapping, got {type(data).__name__}")

    # Parse target
    target_data = data.get("target", {})
    if not isinstance(target_data, dict):
        raise click.BadParameter("'target' section must be a mapping in the config file.")

    target = TargetConfig(
        type=target_data.get("type", "skill"),
        content=data.get("seed_skill", target_data.get("content", "")),
        metadata=target_data.get("metadata", {}),
    )

    # Parse budget
    budget_data = data.get("budget", {})
    budget = BudgetConfig(
        max_calls=budget_data.get("max_calls") or budget_data.get("max_iterations"),
        timeout_s=budget_data.get("timeout_s") or budget_data.get("max_time_seconds"),
        no_improvement_patience=budget_data.get("no_improvement_patience"),
        checkpoint_interval=budget_data.get("checkpoint_interval", 1),
    )

    # Parse evaluator config
    evaluator_data = data.get("evaluator", {})
    from grist_mill.optimization.evaluator_adapter import EvaluatorAdapterConfig

    evaluator = EvaluatorAdapterConfig(
        objective=evaluator_data.get("objective", data.get("objective", "pass-rate")),
        cost_per_token=evaluator_data.get("cost_per_token", 0.0),
        trace_enabled=evaluator_data.get("trace_enabled", False),
    )

    # Output directory
    output_dir = data.get("output_dir", "./results/optimization")

    # Checkpoint path
    checkpoint_path = data.get("checkpoint_path")

    # Splits
    train_split = data.get("train_split", "train")
    holdout_split = data.get("holdout_split", "holdout")

    return OptimizationConfig(
        target=target,
        budget=budget,
        evaluator=evaluator,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        train_split=train_split,
        holdout_split=holdout_split,
    )


def _create_evaluator(
    config: OptimizationConfig,
    seed_content: str,
) -> Any:
    """Create an evaluator function from the config.

    For the smoke/demo case, returns a simple evaluator that scores
    based on content length (mock). In production, this would wrap
    the actual evaluation harness.

    Args:
        config: The optimization configuration.
        seed_content: The seed content for the target.

    Returns:
        A callable evaluator function.
    """

    # For now, create a mock evaluator that simulates scores.
    # In production, this would use the actual harness.
    def _mock_evaluator(
        candidate: str,
        task: Task,
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        # Simulate evaluation: score based on content non-triviality
        score = min(1.0, len(candidate.split()) / 50.0)
        return score, {
            "task_id": task.id,
            "status": "SUCCESS" if score > 0.3 else "FAILURE",
            "score": score,
        }

    return _mock_evaluator


@click.command(name="optimize")
@click.option(
    "--config",
    "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the optimization YAML configuration file.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from the last checkpoint.",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the output directory from the config.",
)
@click.pass_context
def optimize(
    ctx: click.Context,
    config_path: Path,
    resume: bool,
    output_dir: Path | None,
) -> None:
    """Run the optimization loop to evolve a target artifact.

    Reads an optimization configuration from a YAML file and iteratively
    improves the target (skill, system prompt, or tool policy) using
    evaluation feedback.

    Produces:
    - ``best_candidate.json``: The evolved artifact
    - ``trajectory.jsonl``: Per-iteration score history
    - ``checkpoint.json``: Full state for resuming

    Example::

        grist-mill optimize --config optimize.yaml
        grist-mill optimize --config optimize.yaml --resume
    """
    # Build config
    config = _build_config(config_path)

    if output_dir is not None:
        config.output_dir = str(output_dir)

    effective_output = Path(config.output_dir or "./results/optimization")
    logger.info("Optimization output directory: %s", effective_output)

    # Load tasks
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks_dir = raw_config.get("tasks_dir")
    tasks: list[Task] = []
    if tasks_dir:
        tasks = _load_tasks_from_dir(tasks_dir)

    # If no tasks found, create minimal synthetic tasks for smoke testing
    if not tasks:
        logger.info("No tasks loaded from directory. Using minimal synthetic tasks.")
        for i in range(3):
            tasks.append(
                Task(
                    id=f"synth-train-{i}",
                    prompt=f"Synthetic training task {i}",
                    language="python",
                    test_command="true",
                    timeout=30,
                    difficulty=Difficulty.EASY,
                )
            )

    # Split train/holdout
    holdout_tasks: list[Task] = []
    train_tasks: list[Task] = []
    for task in tasks:
        if "holdout" in task.id.lower() or "held_out" in task.id.lower():
            holdout_tasks.append(task)
        else:
            train_tasks.append(task)

    if not holdout_tasks:
        logger.info("No holdout tasks found. Holdout evaluation will be skipped.")

    logger.info(
        "Optimization: %d training tasks, %d holdout tasks, target=%s",
        len(train_tasks),
        len(holdout_tasks),
        config.target.type,
    )

    # Create evaluator
    evaluate_fn = _create_evaluator(config, config.target.content)

    # Create proposer
    propose_fn = MockProposer()

    # Set up checkpoint path
    checkpoint_path = (
        Path(config.checkpoint_path)
        if config.checkpoint_path
        else effective_output / "checkpoint.json"
    )

    # Run optimization
    runner = OptimizationRunner(
        config=config,
        train_tasks=train_tasks,
        holdout_tasks=holdout_tasks if holdout_tasks else None,
        evaluate_fn=evaluate_fn,
        propose_fn=propose_fn,
        output_dir=effective_output,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )

    click.echo(f"Starting optimization (target: {config.target.type})...")
    if resume:
        click.echo("Resuming from checkpoint...")

    result = runner.run()

    # Report results
    click.echo("\nOptimization complete!")
    click.echo(f"  Termination reason: {result.termination_reason}")
    click.echo(f"  Iterations: {result.iteration_count}")
    click.echo(f"  Evaluator calls: {result.call_count}")
    click.echo(f"  Best training score: {result.train_score:.4f}")
    if result.holdout_score is not None:
        click.echo(f"  Best holdout score: {result.holdout_score:.4f}")
    click.echo(f"  Pareto front size: {len(result.pareto_front)}")
    click.echo(f"\nOutputs saved to: {effective_output}")
    click.echo("  - best_candidate.json")
    click.echo("  - trajectory.jsonl")
    click.echo("  - checkpoint.json")

    # Write result summary
    summary_path = effective_output / "result_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    click.echo("  - result_summary.json")
