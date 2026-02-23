#!/usr/bin/env python
"""Unified experiment runner CLI.

This is the canonical entry point for running experiments.
It produces deterministic outputs and proper artifact structure.

Usage:
    uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml

Output artifacts (in output_dir/run_id/):
    - results.jsonl: Individual task results (one JSON per line)
    - summary.json: Aggregated statistics
    - manifest.json: Run metadata, fingerprints, and configuration snapshot
    - matrix.json: Pre-computed experiment matrix
    - trajectories/: Generated code outputs (if enabled)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from bench.experiments import ExperimentConfig, ExperimentRunner, load_experiment_config
from bench.schema.v1 import ManifestV1

console = Console()
logger = logging.getLogger(__name__)


def validate_config(config_path: Path) -> ExperimentConfig:
    """Validate and load experiment configuration."""
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    try:
        config = load_experiment_config(config_path)
        return config
    except Exception as e:
        console.print(f"[red]Invalid configuration: {e}[/red]")
        sys.exit(1)


def print_config_summary(config: ExperimentConfig) -> None:
    """Print a summary of the experiment configuration."""
    skill_name = config.skill.get_name()
    models = ", ".join(config.models.names) if config.models.names else "(none)"

    console.print(
        Panel(
            f"Name: {config.name}\n"
            f"Models: {models}\n"
            f"Tasks: {config.tasks.selection.value}\n"
            f"Skill: {skill_name}\n"
            f"Repeats: {config.execution.repeats}\n"
            f"Timeout: {config.execution.timeout}s\n"
            f"Workers: {config.execution.parallel_workers}\n"
            f"Seed: {config.determinism.seed or 'none'}",
            title="Experiment Configuration",
            border_style="blue",
        )
    )


def print_results_summary(manifest: ManifestV1) -> None:
    """Print a summary table of results."""
    table = Table(title="Experiment Results", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Tokens", justify="right")

    for model_summary in manifest.model_summaries:
        pass_rate = model_summary.pass_rate
        color = "green" if pass_rate > 0.7 else "yellow" if pass_rate > 0.3 else "red"

        table.add_row(
            model_summary.model,
            f"[{color}]{pass_rate:.1%}[/{color}]",
            str(model_summary.passed),
            str(model_summary.total),
            f"{model_summary.avg_latency_s:.1f}s",
            f"{model_summary.total_tokens:,}",
        )

    console.print(table)

    # Overall summary
    console.print(
        Panel(
            f"Total: {manifest.summary.completed}/{manifest.summary.total_tasks} tasks\n"
            f"Pass Rate: {manifest.summary.pass_rate:.1%}\n"
            f"Avg Score: {manifest.summary.avg_score:.2f}\n"
            f"Avg Latency: {manifest.summary.avg_latency_s:.1f}s\n"
            f"Total Tokens: {manifest.summary.total_tokens:,}\n"
            f"Duration: {manifest.duration_s:.1f}s"
            if manifest.duration_s
            else "Duration: N/A",
            title="Summary",
            border_style="green" if manifest.summary.pass_rate > 0.5 else "red",
        )
    )


def run_with_progress(runner: ExperimentRunner) -> ManifestV1:
    """Run experiment with progress display."""
    # Setup
    output_dir = runner.setup()
    matrix = runner.matrix

    if not matrix:
        raise RuntimeError("Experiment matrix is empty")

    console.print(f"\n[blue]Starting experiment: {len(matrix)} total runs[/blue]")
    console.print(f"[dim]Output directory: {output_dir}[/dim]")

    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running...", total=len(matrix.runs))

        # Run with custom progress callback
        manifest = run_with_callbacks(
            runner,
            on_result=lambda r: progress.advance(task),
        )

    return manifest


def run_with_callbacks(runner: ExperimentRunner, on_result=None) -> ManifestV1:
    """Run experiment with callbacks for progress updates."""
    import time


    if not runner.matrix or not runner.output_dir or not runner.manifest:
        raise RuntimeError("Experiment not set up")

    start_time = time.time()
    results_file = runner.output_dir / "results.jsonl"

    # Execute runs
    for run in runner.matrix.runs:
        try:
            result = runner._execute_run(run)
        except Exception as e:
            logger.error(f"Run {run.run_index} failed: {e}")
            result = runner._create_error_result(run, str(e))

        # Record result
        runner.results.append(result)
        runner.manifest.add_result(result)

        # Append to results.jsonl
        with open(results_file, "a") as f:
            f.write(json.dumps(result.model_dump(mode="json")) + "\n")

        # Progress callback
        if on_result:
            on_result(result)

    # Finalize
    from datetime import datetime, timezone

    end_time = datetime.now(timezone.utc)
    runner.manifest.finalize(end_time)
    runner.manifest.results_path = str(results_file)

    runner._save_manifest()
    runner._save_summary()

    total_time = time.time() - start_time
    console.print(f"\n[green]Experiment complete in {total_time:.1f}s[/green]")

    return runner.manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments with unified configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run smoke test
    uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml

    # Run with custom output directory
    uv run python scripts/run_experiment.py --config my_config.yaml --output-dir results/my_run

    # Run with specific seed for reproducibility
    uv run python scripts/run_experiment.py --config my_config.yaml --seed 42

    # Validate config without running
    uv run python scripts/run_experiment.py --config my_config.yaml --validate
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Override random seed for reproducibility",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Override number of parallel workers",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration without running",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment matrix without running",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Load and validate config
    config_path = Path(args.config)
    config = validate_config(config_path)

    # Apply CLI overrides
    if args.output_dir:
        config.output.dir = args.output_dir
    if args.seed is not None:
        config.determinism.seed = args.seed
    if args.workers is not None:
        config.execution.parallel_workers = args.workers

    # Print config summary
    print_config_summary(config)

    # Validate only
    if args.validate:
        console.print("[green]Configuration is valid[/green]")
        sys.exit(0)

    # Dry run - show matrix without executing
    if args.dry_run:
        from bench.experiments import generate_matrix

        matrix = generate_matrix(config)
        console.print("\n[blue]Experiment Matrix[/blue]")
        console.print(f"  Tasks: {len(matrix.tasks)}")
        console.print(f"  Models: {len(matrix.models)}")
        console.print(f"  Repeats: {config.execution.repeats}")
        console.print(f"  Total runs: {len(matrix.runs)}")

        if matrix.tasks:
            console.print(
                f"\n[dim]Task IDs: {[t.task_id for t in matrix.tasks[:5]]}{'...' if len(matrix.tasks) > 5 else ''}[/dim]"
            )
        if matrix.models:
            console.print(f"[dim]Models: {[m.name for m in matrix.models]}[/dim]")

        sys.exit(0)

    # Run experiment
    try:
        runner = ExperimentRunner(config)
        manifest = run_with_progress(runner)

        # Print results
        print_results_summary(manifest)

        # Print output location
        output_dir = runner.output_dir
        console.print("\n[dim]Output artifacts:[/dim]")
        console.print(f"[dim]  - {output_dir}/manifest.json[/dim]")
        console.print(f"[dim]  - {output_dir}/results.jsonl[/dim]")
        console.print(f"[dim]  - {output_dir}/summary.json[/dim]")
        console.print(f"[dim]  - {output_dir}/matrix.json[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Experiment failed: {e}[/red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
