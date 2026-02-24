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
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import bench.runner
from bench.experiments import ExperimentConfig, load_experiment_config
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

    # Print config summary showing base config
    print_config_summary(config)

    # Common override kwargs passed to bench.runner.run() for all modes.
    # The runner applies these via _apply_overrides on a copy of the config,
    # ensuring the original is not mutated and all code paths are consistent.
    run_kwargs: dict[str, Any] = {}
    if args.output_dir:
        run_kwargs["output_dir"] = args.output_dir
    if args.seed is not None:
        run_kwargs["seed"] = args.seed
    if args.workers is not None:
        run_kwargs["workers"] = args.workers

    # Validate only - delegate to runner with validate_only=True
    if args.validate:
        bench.runner.run(config, validate_only=True, **run_kwargs)
        console.print("[green]Configuration is valid[/green]")
        return

    # Dry run - delegate to runner with dry_run=True
    if args.dry_run:
        manifest = bench.runner.run(config, dry_run=True, **run_kwargs)
        console.print("\n[blue]Experiment Matrix[/blue]")
        console.print(f"  Tasks: {manifest.task_count}")
        console.print(f"  Models: {len(manifest.models)}")
        console.print(f"  Total runs: {manifest.config.get('total_runs', 'N/A')}")
        if manifest.config.get("task_ids"):
            console.print(f"\n[dim]Task IDs: {manifest.config['task_ids'][:5]}...[/dim]")
        console.print(f"[dim]Models: {manifest.models}[/dim]")
        return

    # Run experiment - delegate to canonical bench.runner.run()
    try:
        manifest = bench.runner.run(
            config,
            **run_kwargs,
        )

        # Print results
        print_results_summary(manifest)

        # Print output location (derive from results_path)
        output_dir = Path(manifest.results_path).parent if manifest.results_path else None
        if output_dir:
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
