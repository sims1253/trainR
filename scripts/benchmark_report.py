#!/usr/bin/env python
"""Generate benchmark report from results."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from benchmark.schema import BenchmarkRun

console = Console()


def generate_report(results_path: Path, output_path: Path | None = None) -> str:
    """Generate a markdown benchmark report."""
    run = BenchmarkRun.load(results_path)

    lines = []
    lines.append(f"# Benchmark Report: {run.run_id}")
    lines.append("")
    lines.append(f"- **Date:** {run.timestamp}")
    lines.append(f"- **Git SHA:** {run.git_sha}")
    lines.append(f"- **Skill:** {run.skill_version}")
    lines.append(f"- **Tasks:** {run.task_count}")
    lines.append("")

    # Overall results table
    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Model | Pass Rate | Avg Latency (s) | Tasks Evaluated |")
    lines.append("|-------|-----------|-----------------|-----------------|")

    for model in run.models:
        model_results = [r for r in run.results if r.model == model]
        pass_rate = run.pass_rate(model)
        avg_lat = run.avg_latency(model)
        lines.append(f"| {model} | {pass_rate:.1%} | {avg_lat:.1f} | {len(model_results)} |")

    lines.append("")

    # Failure analysis per model
    lines.append("## Failure Analysis")
    lines.append("")

    for model in run.models:
        model_results = [r for r in run.results if r.model == model]
        failures = [r for r in model_results if not r.passed]

        if not failures:
            lines.append(f"### {model}: No failures")
            lines.append("")
            continue

        lines.append(f"### {model}: {len(failures)} failures")
        lines.append("")

        # Count error categories
        categories = Counter(r.error_category or "unknown" for r in failures)
        lines.append("| Error Category | Count |")
        lines.append("|----------------|-------|")
        for cat, count in categories.most_common():
            lines.append(f"| {cat} | {count} |")
        lines.append("")

    # Per-task comparison (if multiple models)
    if len(run.models) > 1:
        lines.append("## Per-Task Comparison")
        lines.append("")

        # Get unique task IDs
        task_ids = sorted({r.task_id for r in run.results})

        header = "| Task ID |" + " | ".join(run.models) + " |"
        separator = "|---------|" + " | ".join(["---"] * len(run.models)) + " |"
        lines.append(header)
        lines.append(separator)

        for tid in task_ids:
            row = f"| {tid} |"
            for model in run.models:
                result = next(
                    (r for r in run.results if r.task_id == tid and r.model == model),
                    None,
                )
                if result is None:
                    row += " - |"
                elif result.passed:
                    row += " PASS |"
                else:
                    row += " FAIL |"
            lines.append(row)

        lines.append("")

    report = "\n".join(lines)

    # Print to console
    console.print(report)

    # Save to file if requested
    if output_path:
        output_path.write_text(report)
        console.print(f"\n[dim]Report saved to {output_path}[/dim]")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "results",
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    generate_report(results_path, output_path)


if __name__ == "__main__":
    main()
