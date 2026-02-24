#!/usr/bin/env python
"""Generate benchmark summary from multiple benchmark_results.json files."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from bench.schema.v1 import ResultV1, ManifestV1

console = Console()


def load_all_results(base_path: Path) -> list[ResultV1]:
    """Load all benchmark results from all benchmark_results.json files."""
    results = []
    json_files = list(base_path.rglob("benchmark_results.json"))

    for json_file in json_files:
        try:
            manifest = ManifestV1.load(str(json_file))
            results.extend(manifest.results)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {json_file}: {e}[/yellow]")

    return results


def clean_error_category(category) -> str:
    """Clean up error category name."""
    if category is None:
        return "PASSED"
    # Handle both enum and string, remove any prefix
    cat_str = category.value if hasattr(category, "value") else str(category)
    return cat_str.replace("FailureCategory.", "")


def get_pass_rate_color(rate: float) -> str:
    """Get color for pass rate."""
    if rate >= 0.8:
        return "green"
    elif rate >= 0.5:
        return "yellow"
    return "red"


def generate_summary(results: list[ResultV1]) -> None:
    """Generate and display the benchmark summary."""
    if not results:
        console.print("[red]No results found![/red]")
        return

    console.print("\n[bold cyan]Benchmark Summary[/bold cyan]\n")

    results_by_model: dict[str, list[ResultV1]] = defaultdict(list)
    for r in results:
        results_by_model[r.model].append(r)

    console.rule("[bold]Model Performance[/bold]")
    model_table = Table(show_header=True, header_style="bold")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Tasks", justify="right")
    model_table.add_column("Passed", justify="right")
    model_table.add_column("Pass Rate", justify="right")
    model_table.add_column("Avg Latency", justify="right")

    for model in sorted(results_by_model.keys()):
        model_results = results_by_model[model]
        passed = sum(1 for r in model_results if r.passed)
        total = len(model_results)
        rate = passed / total if total > 0 else 0
        avg_lat = sum(r.latency_s for r in model_results) / total if total > 0 else 0

        color = get_pass_rate_color(rate)
        model_table.add_row(
            model,
            str(total),
            str(passed),
            f"[{color}]{rate:.1%}[/{color}]",
            f"{avg_lat:.1f}s",
        )

    console.print(model_table)

    failures = [r for r in results if not r.passed]
    if failures:
        console.print()
        console.rule("[bold]Error Distribution (All Tasks)[/bold]")
        error_counts: dict[str, int] = defaultdict(int)
        for r in failures:
            cat = clean_error_category(r.error_category)
            error_counts[cat] += 1

        error_table = Table(show_header=True, header_style="bold")
        error_table.add_column("Error Category", style="red")
        error_table.add_column("Count", justify="right")
        error_table.add_column("Percentage", justify="right")

        for cat, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            pct = count / len(failures) * 100
            error_table.add_row(cat, str(count), f"{pct:.1f}%")

        console.print(error_table)

        console.print()
        console.rule("[bold]Error Profile by Model[/bold]")

        all_categories = sorted({clean_error_category(r.error_category) for r in failures})
        models = sorted(results_by_model.keys())

        profile_table = Table(show_header=True, header_style="bold")
        profile_table.add_column("Error Category", style="cyan")
        for model in models:
            profile_table.add_column(model, justify="center")

        for cat in all_categories:
            row = [cat]
            for model in models:
                model_failures = [r for r in results_by_model[model] if not r.passed]
                cat_count = sum(
                    1 for r in model_failures if clean_error_category(r.error_category) == cat
                )
                total_failures = len(model_failures)
                if cat_count > 0:
                    pct = cat_count / total_failures * 100 if total_failures > 0 else 0
                    row.append(f"{cat_count} ({pct:.0f}%)")
                else:
                    row.append("-")
            profile_table.add_row(*row)

        console.print(profile_table)

    real_packages = [r for r in results if not r.task_id.startswith("task-")]
    if real_packages:
        console.print()
        console.rule("[bold]Package Performance (Real Packages)[/bold]")

        package_results: dict[str, list[ResultV1]] = defaultdict(list)
        for r in real_packages:
            package_name = r.task_id.rsplit("_", 1)[0]
            package_results[package_name].append(r)

        pkg_table = Table(show_header=True, header_style="bold")
        pkg_table.add_column("Package", style="cyan")
        pkg_table.add_column("Passed/Total", justify="right")
        pkg_table.add_column("Pass Rate", justify="right")

        package_stats = []
        for pkg_name, pkg_res in package_results.items():
            passed = sum(1 for r in pkg_res if r.passed)
            total = len(pkg_res)
            rate = passed / total if total > 0 else 0
            package_stats.append((pkg_name, passed, total, rate))

        package_stats.sort(key=lambda x: (-x[3], -x[2]))

        for pkg_name, passed, total, rate in package_stats:
            color = get_pass_rate_color(rate)
            pkg_table.add_row(
                pkg_name,
                f"{passed}/{total}",
                f"[{color}]{rate:.1%}[/{color}]",
            )

        console.print(pkg_table)

    console.print()
    console.rule("[bold]Additional Insights[/bold]")

    timeout_incomplete: dict[str, dict[str, int]] = defaultdict(
        lambda: {"TIMEOUT": 0, "INCOMPLETE_SOLUTION": 0}
    )
    for r in failures:
        cat = clean_error_category(r.error_category)
        if cat in ("TIMEOUT", "INCOMPLETE_SOLUTION"):
            timeout_incomplete[r.model][cat] += 1

    if any(v["TIMEOUT"] or v["INCOMPLETE_SOLUTION"] for v in timeout_incomplete.values()):
        insight_table = Table(show_header=True, header_style="bold")
        insight_table.add_column("Model", style="cyan")
        insight_table.add_column("Timeout", justify="right")
        insight_table.add_column("Incomplete", justify="right")

        for model in sorted(timeout_incomplete.keys()):
            stats = timeout_incomplete[model]
            insight_table.add_row(
                model,
                str(stats["TIMEOUT"]),
                str(stats["INCOMPLETE_SOLUTION"]),
            )

        console.print("\n[dim]Timeout vs Incomplete Breakdown[/dim]")
        console.print(insight_table)

    task_results: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in results:
        task_results[r.task_id][r.model] = r.passed

    disagreements = []
    for task_id, model_results in task_results.items():
        if len(model_results) > 1:
            outcomes = set(model_results.values())
            if len(outcomes) > 1:
                disagreements.append(task_id)

    if disagreements:
        console.print(f"\n[dim]Tasks with Model Disagreement ({len(disagreements)} tasks)[/dim]")
        disag_table = Table(show_header=True, header_style="bold")
        disag_table.add_column("Task ID", style="cyan")
        for model in sorted(results_by_model.keys()):
            disag_table.add_column(model, justify="center")

        for task_id in sorted(disagreements):
            row = [task_id]
            for model in sorted(results_by_model.keys()):
                if model in task_results[task_id]:
                    outcome = task_results[task_id][model]
                    row.append("[green]PASS[/green]" if outcome else "[red]FAIL[/red]")
                else:
                    row.append("-")
            disag_table.add_row(*row)

        console.print(disag_table)
    else:
        console.print("\n[dim]No tasks with model disagreement[/dim]")

    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark summary from results")
    parser.add_argument(
        "path",
        nargs="?",
        default="results/benchmarks/full_all_tasks",
        help="Path to directory containing benchmark results",
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        console.print(f"[red]Path not found: {base_path}[/red]")
        sys.exit(1)

    results = load_all_results(base_path)
    generate_summary(results)


if __name__ == "__main__":
    main()
