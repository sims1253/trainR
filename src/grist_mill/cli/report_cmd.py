"""CLI 'report' subcommand for grist-mill.

Produces analysis reports from experiment results:
- Comparison of two experiments (per-task and aggregate deltas)
- Telemetry aggregation (per-model, per-tool, per-experiment)
- Per-tool performance breakdown
- Cross-experiment rollup
- Error taxonomy breakdown

Validates:
- VAL-REPORT-01 through VAL-REPORT-05
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import yaml

logger = logging.getLogger(__name__)


def _load_results(path: Path) -> list[dict[str, Any]]:
    """Load results from a JSON or JSONL file.

    Args:
        path: Path to the results file.

    Returns:
        List of result dicts.

    Raises:
        click.BadParameter: If the file cannot be read or parsed.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise click.BadParameter(f"Results file not found: {path}") from exc
    except Exception as exc:
        raise click.BadParameter(f"Cannot read results file: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        results: list[dict[str, Any]] = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                results.append(json.loads(line))
        return results
    elif suffix == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return data
        # May be a manifest with records
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        return [data]
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
        if isinstance(data, list):
            return data
        return [data]
    else:
        raise click.BadParameter(f"Unsupported file format: {suffix}. Use .json, .jsonl, or .yaml.")


def _load_experiments(dir_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load experiments from a directory of result files.

    Each file becomes an experiment named after its stem.

    Args:
        dir_path: Path to directory containing result files.

    Returns:
        Dict mapping experiment name to list of results.
    """
    experiments: dict[str, list[dict[str, Any]]] = {}
    if not dir_path.exists():
        raise click.BadParameter(f"Experiments directory not found: {dir_path}")

    for file_path in sorted(dir_path.glob("*")):
        if file_path.is_file() and file_path.suffix.lower() in (".json", ".jsonl", ".yaml", ".yml"):
            name = file_path.stem
            experiments[name] = _load_results(file_path)
            logger.info(
                "Loaded experiment '%s': %d results from %s",
                name,
                len(experiments[name]),
                file_path,
            )

    if not experiments:
        raise click.BadParameter(f"No result files found in: {dir_path}")

    return experiments


@click.command(name="report")
@click.option(
    "--type",
    "report_type",
    type=click.Choice(
        ["comparison", "aggregation", "tools", "rollup", "errors"],
        case_sensitive=False,
    ),
    required=True,
    help="Type of report to generate.",
)
@click.option(
    "--results",
    "results_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to results file (JSON/JSONL/YAML). Required for comparison, aggregation, tools, errors.",
)
@click.option(
    "--compare-with",
    "compare_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to second results file for comparison (required for 'comparison' type).",
)
@click.option(
    "--experiments-dir",
    "experiments_dir",
    type=click.Path(exists=True, path_type=Path),
    help="Path to directory of experiment result files (required for 'rollup').",
)
@click.option(
    "--group-by",
    "group_by",
    type=click.Choice(["model", "provider", "experiment"], case_sensitive=False),
    default="model",
    help="Group by field for aggregation (default: model).",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write report to this file (default: stdout).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    default="json",
    help="Output format (default: json).",
)
@click.pass_context
def report(
    ctx: click.Context,
    report_type: str,
    results_file: Path | None,
    compare_file: Path | None,
    experiments_dir: Path | None,
    group_by: str,
    output_file: Path | None,
    output_format: str,
) -> None:
    """Generate analysis reports from experiment results.

    Supported report types:
    - comparison: Compare two experiments (per-task + aggregate deltas)
    - aggregation: Telemetry aggregation grouped by model/provider
    - tools: Per-tool performance breakdown
    - rollup: Cross-experiment rollup (one row per experiment)
    - errors: Error taxonomy breakdown

    Examples:

    \b
        grist-mill report --type comparison --results exp-a.json --compare-with exp-b.json

    \b
        grist-mill report --type aggregation --results results.json --group-by model

    \b
        grist-mill report --type rollup --experiments-dir results/

    \b
        grist-mill report --type errors --results results.json
    """
    report_data: dict[str, Any] = {}

    if report_type == "comparison":
        if results_file is None or compare_file is None:
            raise click.UsageError("Comparison requires --results and --compare-with.")
        report_data = _generate_comparison(results_file, compare_file)

    elif report_type == "aggregation":
        if results_file is None:
            raise click.UsageError("Aggregation requires --results.")
        report_data = _generate_aggregation(results_file, group_by)

    elif report_type == "tools":
        if results_file is None:
            raise click.UsageError("Tools report requires --results.")
        report_data = _generate_tools_report(results_file)

    elif report_type == "rollup":
        if experiments_dir is None:
            raise click.UsageError("Rollup requires --experiments-dir.")
        report_data = _generate_rollup(experiments_dir)

    elif report_type == "errors":
        if results_file is None:
            raise click.UsageError("Error report requires --results.")
        report_data = _generate_errors_report(results_file)

    # Output
    _output_report(report_data, output_format, output_file)


def _generate_comparison(
    results_a: Path,
    results_b: Path,
) -> dict[str, Any]:
    """Generate a comparison report between two experiments."""
    from grist_mill.reports.comparison import compare_experiments

    exp_a = _load_results(results_a)
    exp_b = _load_results(results_b)

    click.echo(
        f"Comparing experiments: {results_a} ({len(exp_a)} tasks) vs "
        f"{results_b} ({len(exp_b)} tasks)",
        err=True,
    )

    return compare_experiments(exp_a, exp_b)


def _generate_aggregation(
    results_file: Path,
    group_by: str,
) -> dict[str, Any]:
    """Generate an aggregation report."""
    from grist_mill.reports.aggregation import aggregate_telemetry

    results = _load_results(results_file)
    click.echo(
        f"Aggregating {len(results)} results by '{group_by}'",
        err=True,
    )

    summaries = aggregate_telemetry(results, group_by=group_by)
    return {
        "report_type": "aggregation",
        "group_by": group_by,
        "total_results": len(results),
        "groups": summaries,
    }


def _generate_tools_report(results_file: Path) -> dict[str, Any]:
    """Generate a per-tool performance report."""
    from grist_mill.reports.tools import tool_performance_breakdown

    results = _load_results(results_file)
    click.echo(
        f"Generating tool breakdown for {len(results)} results",
        err=True,
    )

    breakdown = tool_performance_breakdown(results)
    return {
        "report_type": "tools",
        "total_results": len(results),
        "tools": breakdown,
    }


def _generate_rollup(experiments_dir: Path) -> dict[str, Any]:
    """Generate a cross-experiment rollup."""
    from grist_mill.reports.rollup import cross_experiment_rollup

    experiments = _load_experiments(experiments_dir)
    click.echo(
        f"Generating rollup for {len(experiments)} experiments",
        err=True,
    )

    rollup = cross_experiment_rollup(experiments)
    return {
        "report_type": "rollup",
        "total_experiments": len(experiments),
        "experiments": rollup,
    }


def _generate_errors_report(results_file: Path) -> dict[str, Any]:
    """Generate an error taxonomy breakdown."""
    from grist_mill.reports.errors import error_taxonomy_breakdown

    results = _load_results(results_file)
    click.echo(
        f"Generating error breakdown for {len(results)} results",
        err=True,
    )

    breakdown = error_taxonomy_breakdown(results)
    n_errors = sum(e["count"] for e in breakdown)
    return {
        "report_type": "errors",
        "total_results": len(results),
        "total_errors": n_errors,
        "error_categories": breakdown,
    }


def _output_report(
    data: dict[str, Any],
    fmt: str,
    output_file: Path | None,
) -> None:
    """Output a report in the specified format.

    Args:
        data: The report data.
        fmt: Output format ('json' or 'yaml').
        output_file: Optional file to write to.
    """
    if fmt == "json":
        content = json.dumps(data, indent=2, default=str)
    else:
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        click.echo(f"Report written to {output_file}", err=True)
    else:
        click.echo(content)


__all__ = ["report"]
