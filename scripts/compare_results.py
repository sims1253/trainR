#!/usr/bin/env python3
"""Compare evaluation results across models and skill conditions."""

import argparse
import glob
import json

from rich.console import Console
from rich.table import Table


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--dir", default="results/baselines", help="Directory with results")
    parser.add_argument("--output", help="Output file for markdown table")
    args = parser.parse_args()

    console = Console()

    # Load all results
    results = []
    for f in sorted(glob.glob(f"{args.dir}/eval_*.json")):
        if "_test" in f:
            continue
        with open(f) as fp:
            data = json.load(fp)
            results.append(
                {
                    "file": f,
                    "model": data["config"]["model"],
                    "skill": data["config"]["skill"],
                    "pass_rate": data["summary"]["pass_rate"],
                    "total": data["summary"]["total"],
                    "passed": data["summary"]["passed"],
                    "failed": data["summary"]["failed"],
                    "time_s": data["summary"]["total_time_s"],
                }
            )

    if not results:
        console.print("[red]No results found[/red]")
        return

    # Create comparison table
    table = Table(title="Baseline Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Skill", style="magenta")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Passed/Total", justify="right")
    table.add_column("Time (m)", justify="right")

    for r in sorted(results, key=lambda x: (x["model"], x["skill"])):
        model_short = r["model"].split("/")[-1].replace(":free", "")
        skill_short = r["skill"] if r["skill"] != "no_skill" else "none"
        table.add_row(
            model_short,
            skill_short,
            f"{r['pass_rate']:.1%}",
            f"{r['passed']}/{r['total']}",
            f"{r['time_s'] / 60:.1f}",
        )

    console.print(table)

    # Generate markdown output
    if args.output:
        md = "| Model | Skill | Pass Rate | Passed/Total | Time (m) |\n"
        md += "|-------|-------|-----------|--------------|----------|\n"
        for r in sorted(results, key=lambda x: (x["model"], x["skill"])):
            model_short = r["model"].split("/")[-1].replace(":free", "")
            skill_short = r["skill"] if r["skill"] != "no_skill" else "none"
            md += f"| {model_short} | {skill_short} | {r['pass_rate']:.1%} | {r['passed']}/{r['total']} | {r['time_s'] / 60:.1f} |\n"

        with open(args.output, "w") as f:
            f.write(md)
        console.print(f"[green]Saved markdown to {args.output}[/green]")


if __name__ == "__main__":
    main()
