#!/usr/bin/env python
"""Generate benchmark report from results."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from bench.schema.v1 import ManifestV1

console = Console()


def generate_report(results_path: Path, output_path: Path | None = None) -> str:
    """Generate a markdown benchmark report."""
    manifest = ManifestV1.load(str(results_path))

    lines = []
    lines.append(f"# Benchmark Report: {manifest.run_id}")
    lines.append("")
    lines.append(f"- **Date:** {manifest.timestamp}")
    lines.append(f"- **Git SHA:** {manifest.git_sha}")
    lines.append(f"- **Skill:** {manifest.skill_version}")
    lines.append(f"- **Tasks:** {manifest.task_count}")
    lines.append("")

    # Overall results table
    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Model | Pass Rate | Avg Latency (s) | Tasks Evaluated |")
    lines.append("|-------|-----------|-----------------|-----------------|")

    for model in manifest.models:
        model_results = [r for r in manifest.results if r.model == model]
        passed = sum(1 for r in model_results if r.passed)
        total = len(model_results)
        pass_rate = passed / total if total > 0 else 0.0
        avg_lat = sum(r.latency_s for r in model_results) / total if total > 0 else 0.0
        lines.append(f"| {model} | {pass_rate:.1%} | {avg_lat:.1f} | {len(model_results)} |")

    lines.append("")

    # Failure analysis per model
    lines.append("## Failure Analysis")
    lines.append("")

    for model in manifest.models:
        model_results = [r for r in manifest.results if r.model == model]
        failures = [r for r in model_results if not r.passed]

        if not failures:
            lines.append(f"### {model}: No failures")
            lines.append("")
            continue

        lines.append(f"### {model}: {len(failures)} failures")
        lines.append("")

        # Count error categories - handle both enum and string
        def get_error_cat(r):
            if r.error_category is None:
                return "unknown"
            return (
                r.error_category.value
                if hasattr(r.error_category, "value")
                else str(r.error_category)
            )

        categories = Counter(get_error_cat(r) for r in failures)
        lines.append("| Error Category | Count |")
        lines.append("|----------------|-------|")
        for cat, count in categories.most_common():
            lines.append(f"| {cat} | {count} |")
        lines.append("")

    # Per-task comparison (if multiple models)
    if len(manifest.models) > 1:
        lines.append("## Per-Task Comparison")
        lines.append("")

        # Get unique task IDs
        task_ids = sorted({r.task_id for r in manifest.results})

        header = "| Task ID |" + " | ".join(manifest.models) + " |"
        separator = "|---------|" + " | ".join(["---"] * len(manifest.models)) + " |"
        lines.append(header)
        lines.append(separator)

        for tid in task_ids:
            row = f"| {tid} |"
            for model in manifest.models:
                result = next(
                    (r for r in manifest.results if r.task_id == tid and r.model == model),
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
