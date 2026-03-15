"""Per-tool performance breakdown.

Produces per-tool performance tables with success/error rates and
average latency for each tool used across results.

Validates:
- VAL-REPORT-03: Per-tool performance breakdown
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def tool_performance_breakdown(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-tool performance metrics across results.

    Args:
        results: List of result dicts. Each must have a ``telemetry``
            field with ``tool_calls.by_tool`` breakdown.

    Returns:
        A list of per-tool dicts, each containing:
        - ``tool_name``: Name of the tool.
        - ``total_calls``: Total number of invocations.
        - ``successful_calls``: Number of successful invocations.
        - ``failed_calls``: Number of failed invocations.
        - ``success_rate``: Fraction of successful calls (0.0-1.0).
        - ``error_rate``: Fraction of failed calls (0.0-1.0).
        - ``avg_duration_ms``: Average duration per call (from total).

        Sorted by total_calls descending.
    """
    if not results:
        return []

    # Aggregate tool metrics across all results
    tool_totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"calls": 0, "successes": 0, "failures": 0, "total_duration_ms": 0.0}
    )

    for r in results:
        telemetry = r.get("telemetry")
        if telemetry is None:
            continue

        by_tool = _get_by_tool(telemetry)
        total_duration = _get_total_tool_duration(telemetry)

        n_tools = len(by_tool)
        avg_duration_per_tool = total_duration / n_tools if n_tools > 0 else 0.0

        for tool_name, metrics in by_tool.items():
            entry = tool_totals[tool_name]
            entry["calls"] += metrics.get("calls", 0)
            entry["successes"] += metrics.get("successes", 0)
            entry["failures"] += metrics.get("failures", 0)
            entry["total_duration_ms"] += avg_duration_per_tool

    # Build breakdown
    breakdown: list[dict[str, Any]] = []
    for tool_name, totals in sorted(tool_totals.items(), key=lambda x: x[1]["calls"], reverse=True):
        calls = totals["calls"]
        successes = totals["successes"]
        failures = totals["failures"]

        breakdown.append(
            {
                "tool_name": tool_name,
                "total_calls": calls,
                "successful_calls": successes,
                "failed_calls": failures,
                "success_rate": successes / calls if calls > 0 else 0.0,
                "error_rate": failures / calls if calls > 0 else 0.0,
                "avg_duration_ms": (
                    round(totals["total_duration_ms"] / calls, 2) if calls > 0 else 0.0
                ),
            }
        )

    logger.debug(
        "Tool breakdown: %d tools across %d results",
        len(breakdown),
        len(results),
    )
    return breakdown


def _get_by_tool(telemetry: Any) -> dict[str, dict[str, int]]:
    """Extract the per-tool breakdown from telemetry.

    Args:
        telemetry: A TelemetrySchema object or dict.

    Returns:
        Dict mapping tool_name -> {calls, successes, failures}.
    """
    if hasattr(telemetry, "tool_calls"):
        return dict(telemetry.tool_calls.by_tool)
    if isinstance(telemetry, dict):
        tool_calls = telemetry.get("tool_calls", {})
        if isinstance(tool_calls, dict):
            return tool_calls.get("by_tool", {})
    return {}


def _get_total_tool_duration(telemetry: Any) -> float:
    """Get total tool call duration from telemetry.

    Args:
        telemetry: A TelemetrySchema object or dict.

    Returns:
        Total duration in milliseconds.
    """
    if hasattr(telemetry, "tool_calls"):
        return telemetry.tool_calls.total_duration_ms
    if isinstance(telemetry, dict):
        tool_calls = telemetry.get("tool_calls", {})
        if isinstance(tool_calls, dict):
            return tool_calls.get("total_duration_ms", 0.0)
    return 0.0


__all__ = ["tool_performance_breakdown"]
