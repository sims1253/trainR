"""CLI 'export' subcommand for grist-mill.

Exports experiment results to JSON, CSV, or HTML formats.
Supports filtering by model, tool, and date range.

Validates:
- VAL-EXPORT-01: JSON export with schema_version and generated_at
- VAL-EXPORT-02: CSV export loads cleanly with pandas
- VAL-EXPORT-03: HTML export is self-contained
- VAL-EXPORT-04: Export supports filtering by model, tool, date
- VAL-EXPORT-05: Multi-format export is consistent
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


def _load_results(path: Path) -> list[dict[str, Any]]:
    """Load results from a JSON, JSONL, or YAML file.

    Args:
        path: Path to the results file.

    Returns:
        List of result dicts.

    Raises:
        click.BadParameter: If the file cannot be read or parsed.
    """
    import yaml

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


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string.

    Returns:
        A date object.

    Raises:
        click.BadParameter: If the format is invalid.
    """
    try:
        return date.fromisoformat(date_str)
    except ValueError as exc:
        raise click.BadParameter(f"Invalid date format: {date_str!r}. Use YYYY-MM-DD.") from exc


@click.command(name="export")
@click.option(
    "--results",
    "results_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to results file (JSON/JSONL/YAML).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "html"], case_sensitive=False),
    default="json",
    help="Output format (default: json).",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write export to this file (default: stdout).",
)
@click.option(
    "--model",
    default=None,
    help="Filter results by model name.",
)
@click.option(
    "--tool",
    default=None,
    help="Filter results to those that used this tool.",
)
@click.option(
    "--date-start",
    "date_start",
    type=str,
    default=None,
    help="Start date for filtering (YYYY-MM-DD, inclusive).",
)
@click.option(
    "--date-end",
    "date_end",
    type=str,
    default=None,
    help="End date for filtering (YYYY-MM-DD, inclusive).",
)
@click.pass_context
def export(
    ctx: click.Context,
    results_file: Path,
    output_format: str,
    output_file: Path | None,
    model: str | None,
    tool: str | None,
    date_start: str | None,
    date_end: str | None,
) -> None:
    """Export experiment results to JSON, CSV, or HTML.

    Reads results from a file and exports them in the specified format.
    Supports filtering by model, tool, and date range.

    Examples:

    \b
        grist-mill export --results results.json --format json

    \b
        grist-mill export --results results.json --format csv --model gpt-4

    \b
        grist-mill export --results results.json --format html --output report.html

    \b
        grist-mill export --results results.json --format json --date-start 2025-01-01 --date-end 2025-01-31
    """
    # Load results
    results = _load_results(results_file)
    click.echo(
        f"Loaded {len(results)} results from {results_file}",
        err=True,
    )

    # Parse date range
    date_range: tuple[date, date] | None = None
    if date_start is not None and date_end is not None:
        start = _parse_date(date_start)
        end = _parse_date(date_end)
        if start > end:
            raise click.BadParameter(f"Date start ({date_start}) is after date end ({date_end}).")
        date_range = (start, end)
    elif date_start is not None or date_end is not None:
        raise click.UsageError("Both --date-start and --date-end must be provided together.")

    # Apply filters via export functions
    from grist_mill.export.formats import export_csv, export_html, export_json

    if output_format == "json":
        content = export_json(
            results,
            model=model,
            tool=tool,
            date_range=date_range,
        )
    elif output_format == "csv":
        content = export_csv(
            results,
            model=model,
            tool=tool,
            date_range=date_range,
        )
    elif output_format == "html":
        content = export_html(
            results,
            model=model,
            tool=tool,
            date_range=date_range,
        )
    else:
        raise click.BadParameter(f"Unsupported format: {output_format}")

    # Output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        click.echo(f"Export written to {output_file}", err=True)
    else:
        click.echo(content)


__all__ = ["export"]
