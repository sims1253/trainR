#!/usr/bin/env python3
"""Real-time experiment progress monitor."""

import contextlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table


def find_latest_experiment(results_dir: Path) -> Path | None:
    """Find the most recent experiment directory."""
    exp_dirs = sorted(
        results_dir.glob("*_benchmark_*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return exp_dirs[0] if exp_dirs else None


def load_manifest(exp_dir: Path) -> dict:
    """Load experiment manifest."""
    manifest_path = exp_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def load_results(exp_dir: Path) -> list[dict]:
    """Load all results from results.jsonl."""
    results = []
    results_path = exp_dir / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text().strip().split("\n"):
            if line:
                results.append(json.loads(line))
    return results


def compute_stats(results: list[dict], total_tasks: int) -> dict:
    """Compute per-model statistics."""
    stats = defaultdict(
        lambda: {
            "completed": 0,
            "passed": 0,
            "failed": 0,
            "tokens": 0,
            "latency": 0.0,
            "running": False,
        }
    )

    for r in results:
        model = r.get("model", "unknown")
        stats[model]["completed"] += 1
        if r.get("passed"):
            stats[model]["passed"] += 1
        else:
            stats[model]["failed"] += 1
        stats[model]["tokens"] += r.get("token_usage", {}).get("total", 0)
        stats[model]["latency"] += r.get("latency_s", 0)

    return dict(stats)


def format_tokens(tokens: int) -> str:
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.0f}K"
    return str(tokens)


def format_latency(seconds: float) -> str:
    """Format latency in seconds."""
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def build_table(stats: dict, models: list[str], total_tasks: int, console: Console) -> Table:
    """Build the results table."""
    table = Table(title="Experiment Progress", show_header=True, header_style="bold")

    table.add_column("Model", style="cyan", width=20)
    table.add_column("Progress", width=20)
    table.add_column("Pass", width=10)
    table.add_column("Rate", width=8)
    table.add_column("Tokens", width=10)
    table.add_column("Avg Lat", width=10)
    table.add_column("Status", width=10)

    total_completed = 0
    total_passed = 0
    total_tokens = 0
    total_latency = 0.0

    for model in models:
        s = stats.get(model, {"completed": 0, "passed": 0, "tokens": 0, "latency": 0})
        completed = s["completed"]
        passed = s["passed"]
        tokens = s["tokens"]
        latency = s["latency"]

        total_completed += completed
        total_passed += passed
        total_tokens += tokens
        total_latency += latency

        # Progress bar
        pct = completed / total_tasks if total_tasks > 0 else 0
        bar_width = 15
        filled = int(bar_width * pct)
        bar = f"[green]{'█' * filled}[/green]{'░' * (bar_width - filled)} {int(pct * 100):3d}%"

        # Pass rate
        rate = f"{100 * passed / completed:.0f}%" if completed > 0 else "-"

        # Avg latency
        avg_lat = format_latency(latency / completed) if completed > 0 else "-"

        # Status
        status = "[green]Done[/green]" if completed >= total_tasks else "[yellow]Running[/yellow]"

        # Truncate model name
        model_display = model[-18:] if len(model) > 18 else model

        table.add_row(
            model_display,
            bar,
            f"{passed}/{completed}",
            rate,
            format_tokens(tokens),
            avg_lat,
            status,
        )

    # Summary row
    overall_rate = f"{100 * total_passed / total_completed:.1f}%" if total_completed > 0 else "-"
    overall_avg_lat = (
        format_latency(total_latency / total_completed) if total_completed > 0 else "-"
    )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_completed}/{len(models) * total_tasks}[/bold]",
        f"[bold]{total_passed}[/bold]",
        f"[bold]{overall_rate}[/bold]",
        f"[bold]{format_tokens(total_tokens)}[/bold]",
        f"[bold]{overall_avg_lat}[/bold]",
        "",
    )

    return table


def main():
    console = Console()
    results_dir = Path("results/experiments")

    # Find experiment
    exp_dir = find_latest_experiment(results_dir)
    if not exp_dir:
        console.print("[red]No experiment found[/red]")
        return 1

    console.print(f"[bold]Watching:[/bold] {exp_dir.name}")

    # Load manifest
    manifest = load_manifest(exp_dir)
    models = manifest.get("models", [])
    total_tasks = manifest.get("task_count", 44)

    if not models:
        console.print("[red]No models in manifest[/red]")
        return 1

    console.print(f"[bold]Models:[/bold] {len(models)}, [bold]Tasks each:[/bold] {total_tasks}")
    console.print("[bold]Press Ctrl+C to exit[/bold]\n")

    last_count = 0

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            results = load_results(exp_dir)

            if len(results) != last_count:
                stats = compute_stats(results, total_tasks)
                table = build_table(stats, models, total_tasks, console)
                live.update(table)
                last_count = len(results)

            time.sleep(0.5)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
