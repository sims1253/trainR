"""Export formats for grist-mill results.

Implements JSON, CSV, and HTML export with consistent data output.

Validates:
- VAL-EXPORT-01: JSON export with schema_version and generated_at
- VAL-EXPORT-02: CSV export loads cleanly with pandas
- VAL-EXPORT-03: HTML export is self-contained
- VAL-EXPORT-04: Export supports filtering by model, tool, date
- VAL-EXPORT-05: Multi-format export is consistent
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from grist_mill.reports.filtering import filter_results

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_filters(
    results: list[dict[str, Any]],
    model: str | None,
    tool: str | None,
    date_range: tuple[date, date] | None,
) -> list[dict[str, Any]]:
    """Apply optional filters to results.

    Args:
        results: Raw result dicts.
        model: Optional model filter.
        tool: Optional tool filter.
        date_range: Optional date range filter.

    Returns:
        Filtered results.
    """
    return filter_results(
        results,
        model=model,
        tool=tool,
        date_range=date_range,
    )


def _serialize_result(r: dict[str, Any]) -> dict[str, Any]:
    """Serialize a single result dict for export.

    Converts TelemetrySchema objects to dicts, datetime objects to ISO strings,
    and Enum values to their string representation.

    Args:
        r: A result dict.

    Returns:
        A JSON-serializable dict.
    """
    serialized: dict[str, Any] = {}

    for key, value in r.items():
        if key == "telemetry":
            serialized[key] = _serialize_telemetry(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif hasattr(value, "value"):
            serialized[key] = value.value
        else:
            serialized[key] = value

    return serialized


def _serialize_telemetry(telemetry: Any) -> dict[str, Any]:
    """Serialize a TelemetrySchema object or dict.

    Args:
        telemetry: TelemetrySchema or dict.

    Returns:
        A dict suitable for JSON serialization.
    """
    if telemetry is None:
        return {}
    if hasattr(telemetry, "model_dump"):
        return telemetry.model_dump(mode="json")
    if isinstance(telemetry, dict):
        return telemetry
    return {}


def _compute_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics for the exported records.

    Args:
        records: Serialized result dicts.

    Returns:
        A summary dict with total_tasks, pass_rate, total_cost, etc.
    """
    if not records:
        return {
            "total_tasks": 0,
            "pass_rate": 0.0,
            "total_cost_usd": 0.0,
            "mean_latency_s": 0.0,
            "total_tokens": 0,
        }

    n = len(records)
    scores = [r.get("score", 0.0) for r in records]
    passed = sum(1 for s in scores if s >= 1.0)

    costs: list[float] = []
    latencies: list[float] = []
    tokens: list[int] = []

    for r in records:
        tel = r.get("telemetry", {})
        if isinstance(tel, dict):
            costs.append(tel.get("estimated_cost_usd") or 0.0)
            lat_data = tel.get("latency", {})
            if isinstance(lat_data, dict):
                latencies.append(lat_data.get("total_s", 0.0))
            tok_data = tel.get("tokens", {})
            if isinstance(tok_data, dict):
                tokens.append(tok_data.get("total", 0))

    return {
        "total_tasks": n,
        "pass_rate": round(passed / n, 4) if n > 0 else 0.0,
        "total_cost_usd": round(sum(costs), 6),
        "mean_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "total_tokens": sum(tokens),
    }


# ---------------------------------------------------------------------------
# JSON Export (VAL-EXPORT-01)
# ---------------------------------------------------------------------------


def export_json(
    results: list[dict[str, Any]],
    *,
    model: str | None = None,
    tool: str | None = None,
    date_range: tuple[date, date] | None = None,
) -> str:
    """Export results to JSON format.

    Produces a self-describing JSON document with ``schema_version``
    and ``generated_at`` fields.

    Args:
        results: List of result dicts.
        model: Optional model filter.
        tool: Optional tool filter.
        date_range: Optional (start, end) date range filter.

    Returns:
        A JSON string.
    """
    filtered = _apply_filters(results, model, tool, date_range)
    records = [_serialize_result(r) for r in filtered]
    summary = _compute_summary(records)

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "records": records,
    }

    return json.dumps(output, indent=2, default=str)


# ---------------------------------------------------------------------------
# CSV Export (VAL-EXPORT-02)
# ---------------------------------------------------------------------------

# Columns included in CSV output — must be flat (no nested structures)
_CSV_COLUMNS = [
    "task_id",
    "status",
    "score",
    "error_category",
    "model",
    "provider",
    "timestamp",
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "latency_total_s",
    "latency_setup_s",
    "latency_execution_s",
    "latency_teardown_s",
    "tool_calls_total",
    "tool_calls_successful",
    "tool_calls_failed",
    "estimated_cost_usd",
]


def export_csv(
    results: list[dict[str, Any]],
    *,
    model: str | None = None,
    tool: str | None = None,
    date_range: tuple[date, date] | None = None,
) -> str:
    """Export results to CSV format.

    Produces a pandas-compatible CSV with flat headers.

    Args:
        results: List of result dicts.
        model: Optional model filter.
        tool: Optional tool filter.
        date_range: Optional (start, end) date range filter.

    Returns:
        A CSV string.
    """
    filtered = _apply_filters(results, model, tool, date_range)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_COLUMNS)

    for r in filtered:
        tel = r.get("telemetry")
        row = _result_to_csv_row(r, tel)
        writer.writerow(row)

    return buf.getvalue()


def _result_to_csv_row(
    r: dict[str, Any],
    telemetry: Any,
) -> list[str]:
    """Convert a result dict to a flat CSV row.

    Args:
        r: Result dict.
        telemetry: Telemetry object or dict.

    Returns:
        List of string values matching _CSV_COLUMNS.
    """
    # Status
    status = r.get("status")
    status_str = status.value if hasattr(status, "value") else str(status)

    # Error category
    error_cat = r.get("error_category")
    error_cat_str = (
        error_cat.value if hasattr(error_cat, "value") else str(error_cat) if error_cat else ""
    )

    # Timestamp
    ts = r.get("timestamp")
    ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts) if ts else ""

    # Telemetry fields
    if hasattr(telemetry, "tokens"):
        tokens_prompt = telemetry.tokens.prompt
        tokens_completion = telemetry.tokens.completion
        tokens_total = telemetry.tokens.total
    elif isinstance(telemetry, dict):
        tok = telemetry.get("tokens", {})
        tokens_prompt = tok.get("prompt", 0) if isinstance(tok, dict) else 0
        tokens_completion = tok.get("completion", 0) if isinstance(tok, dict) else 0
        tokens_total = tok.get("total", 0) if isinstance(tok, dict) else 0
    else:
        tokens_prompt = tokens_completion = tokens_total = 0

    if hasattr(telemetry, "latency"):
        lat_total = telemetry.latency.total_s
        lat_setup = telemetry.latency.setup_s
        lat_exec = telemetry.latency.execution_s
        lat_teardown = telemetry.latency.teardown_s
    elif isinstance(telemetry, dict):
        lat = telemetry.get("latency", {})
        lat_total = lat.get("total_s", 0.0) if isinstance(lat, dict) else 0.0
        lat_setup = lat.get("setup_s", 0.0) if isinstance(lat, dict) else 0.0
        lat_exec = lat.get("execution_s", 0.0) if isinstance(lat, dict) else 0.0
        lat_teardown = lat.get("teardown_s", 0.0) if isinstance(lat, dict) else 0.0
    else:
        lat_total = lat_setup = lat_exec = lat_teardown = 0.0

    if hasattr(telemetry, "tool_calls"):
        tc_total = telemetry.tool_calls.total_calls
        tc_success = telemetry.tool_calls.successful_calls
        tc_failed = telemetry.tool_calls.failed_calls
    elif isinstance(telemetry, dict):
        tc = telemetry.get("tool_calls", {})
        tc_total = tc.get("total_calls", 0) if isinstance(tc, dict) else 0
        tc_success = tc.get("successful_calls", 0) if isinstance(tc, dict) else 0
        tc_failed = tc.get("failed_calls", 0) if isinstance(tc, dict) else 0
    else:
        tc_total = tc_success = tc_failed = 0

    if hasattr(telemetry, "estimated_cost_usd"):
        cost = telemetry.estimated_cost_usd or ""
    elif isinstance(telemetry, dict):
        cost = telemetry.get("estimated_cost_usd") or ""
    else:
        cost = ""

    return [
        r.get("task_id", ""),
        status_str,
        str(r.get("score", "")),
        error_cat_str,
        r.get("model", ""),
        r.get("provider", ""),
        ts_str,
        str(tokens_prompt),
        str(tokens_completion),
        str(tokens_total),
        str(lat_total),
        str(lat_setup),
        str(lat_exec),
        str(lat_teardown),
        str(tc_total),
        str(tc_success),
        str(tc_failed),
        str(cost),
    ]


# ---------------------------------------------------------------------------
# HTML Export (VAL-EXPORT-03)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="schema_version" content="{schema_version}">
<meta name="generated_at" content="{generated_at}">
<title>grist-mill Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2rem; background: #f8f9fa; color: #212529; }}
  h1 {{ color: #0d6efd; margin-bottom: 0.5rem; }}
  .meta {{ color: #6c757d; font-size: 0.9rem; margin-bottom: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden; }}
  th {{ background: #0d6efd; color: white; padding: 0.75rem 1rem; text-align: left;
       font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #dee2e6; font-size: 0.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #f1f3f5; }}
  .success {{ color: #198754; font-weight: 600; }}
  .failure {{ color: #dc3545; font-weight: 600; }}
  .summary {{ background: white; padding: 1.5rem; border-radius: 8px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                  gap: 1rem; }}
  .summary-item {{ text-align: center; }}
  .summary-value {{ font-size: 1.5rem; font-weight: 700; color: #0d6efd; }}
  .summary-label {{ font-size: 0.8rem; color: #6c757d; text-transform: uppercase;
                  letter-spacing: 0.5px; }}
</style>
</head>
<body>
<h1>grist-mill Report</h1>
<div class="meta">
  <span>Schema version: {schema_version}</span> &middot;
  <span>Generated: {generated_at}</span> &middot;
  <span>{total_tasks} task(s)</span>
</div>
<div class="summary">
  <div class="summary-grid">
    <div class="summary-item">
      <div class="summary-value">{pass_rate_display}</div>
      <div class="summary-label">Pass Rate</div>
    </div>
    <div class="summary-item">
      <div class="summary-value">${total_cost_display}</div>
      <div class="summary-label">Total Cost</div>
    </div>
    <div class="summary-item">
      <div class="summary-value">{mean_latency_display}s</div>
      <div class="summary-label">Mean Latency</div>
    </div>
    <div class="summary-item">
      <div class="summary-value">{total_tokens_display}</div>
      <div class="summary-label">Total Tokens</div>
    </div>
  </div>
</div>
<table>
<thead>
<tr>
{header_row}
</tr>
</thead>
<tbody>
{data_rows}
</tbody>
</table>
</body>
</html>
"""


def export_html(
    results: list[dict[str, Any]],
    *,
    model: str | None = None,
    tool: str | None = None,
    date_range: tuple[date, date] | None = None,
) -> str:
    """Export results to a self-contained HTML document.

    The HTML document has embedded CSS, no external dependencies,
    and renders all data in a styled table.

    Args:
        results: List of result dicts.
        model: Optional model filter.
        tool: Optional tool filter.
        date_range: Optional (start, end) date range filter.

    Returns:
        A complete HTML document string.
    """
    filtered = _apply_filters(results, model, tool, date_range)
    records = [_serialize_result(r) for r in filtered]
    summary = _compute_summary(records)

    # Build header row
    headers = ["Task ID", "Status", "Score", "Model", "Error", "Latency (s)", "Cost ($)"]
    header_row = "".join(f"<th>{h}</th>" for h in headers)

    # Build data rows
    data_rows: list[str] = []
    for r in records:
        status = r.get("status", "")
        status_class = (
            "success"
            if status == "SUCCESS"
            else "failure"
            if status in ("FAILURE", "ERROR", "TIMEOUT")
            else ""
        )

        error_cat = r.get("error_category") or ""
        tel = r.get("telemetry", {})
        if isinstance(tel, dict):
            lat = tel.get("latency", {})
            latency = f"{lat.get('total_s', 0):.2f}" if isinstance(lat, dict) else "0.00"
            cost = tel.get("estimated_cost_usd")
            cost_str = f"{cost:.4f}" if cost is not None else "—"
        else:
            latency = "0.00"
            cost_str = "—"

        data_rows.append(
            f"<tr>"
            f"<td>{_html_escape(r.get('task_id', ''))}</td>"
            f'<td class="{status_class}">{_html_escape(status)}</td>'
            f"<td>{r.get('score', '')}</td>"
            f"<td>{_html_escape(r.get('model', ''))}</td>"
            f"<td>{_html_escape(error_cat)}</td>"
            f"<td>{latency}</td>"
            f"<td>{cost_str}</td>"
            f"</tr>"
        )

    # Summary display values
    pass_rate_display = f"{summary['pass_rate']:.1%}"
    total_cost_display = f"{summary['total_cost_usd']:.4f}"
    mean_latency_display = f"{summary['mean_latency_s']:.2f}"
    total_tokens_display = f"{summary['total_tokens']:,}"

    html = _HTML_TEMPLATE.format(
        schema_version=_html_escape(SCHEMA_VERSION),
        generated_at=_html_escape(datetime.now(timezone.utc).isoformat()),
        total_tasks=len(records),
        pass_rate_display=pass_rate_display,
        total_cost_display=total_cost_display,
        mean_latency_display=mean_latency_display,
        total_tokens_display=total_tokens_display,
        header_row=header_row,
        data_rows="\n".join(data_rows),
    )

    return html


def _html_escape(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Raw text string.

    Returns:
        HTML-safe string.
    """
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


__all__ = ["export_csv", "export_html", "export_json"]
