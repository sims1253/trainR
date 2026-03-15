"""Filtering utilities for report and export results.

Supports filtering by model, tool, and date range. Used by both
the reporting module and the export module.

Validates:
- VAL-EXPORT-04: Export supports filtering by model, tool, date
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def filter_results(
    results: list[dict[str, Any]],
    *,
    model: str | None = None,
    tool: str | None = None,
    date_range: tuple[date, date] | None = None,
) -> list[dict[str, Any]]:
    """Filter a list of result dicts by model, tool, and/or date range.

    Args:
        results: List of result dicts, each containing at least
            ``model``, ``timestamp``, and ``telemetry`` fields.
        model: If provided, only return results where ``model`` matches.
        tool: If provided, only return results where the tool appears
            in ``telemetry.tool_calls.by_tool``.
        date_range: If provided, a ``(start_date, end_date)`` tuple.
            Only return results whose ``timestamp`` falls within
            ``[start_date, end_date]`` inclusive.

    Returns:
        A filtered list of result dicts.
    """
    filtered = results

    if model is not None:
        filtered = [r for r in filtered if r.get("model") == model]

    if tool is not None:
        filtered = [r for r in filtered if _result_uses_tool(r, tool)]

    if date_range is not None:
        start_date, end_date = date_range
        start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
        # End date is inclusive, so go to the end of that day
        end_dt = datetime(
            end_date.year,
            end_date.month,
            end_date.day,
            23,
            59,
            59,
            999999,
            tzinfo=timezone.utc,
        )
        filtered = [r for r in filtered if _result_in_date_range(r, start_dt, end_dt)]

    logger.debug(
        "Filtered %d results down to %d (model=%s, tool=%s, date_range=%s)",
        len(results),
        len(filtered),
        model,
        tool,
        date_range,
    )
    return filtered


def _result_uses_tool(result: dict[str, Any], tool_name: str) -> bool:
    """Check if a result used a specific tool.

    Args:
        result: A result dict with a ``telemetry`` field.
        tool_name: Name of the tool to check.

    Returns:
        ``True`` if the tool appears in the telemetry's ``by_tool`` dict.
    """
    telemetry = result.get("telemetry")
    if telemetry is None:
        return False

    # Handle both dict and TelemetrySchema
    if hasattr(telemetry, "tool_calls"):
        return tool_name in telemetry.tool_calls.by_tool
    if isinstance(telemetry, dict):
        tool_calls = telemetry.get("tool_calls", {})
        by_tool = tool_calls.get("by_tool", {})
        if isinstance(by_tool, dict):
            return tool_name in by_tool
    return False


def _result_in_date_range(
    result: dict[str, Any],
    start: datetime,
    end: datetime,
) -> bool:
    """Check if a result's timestamp falls within a range.

    Args:
        result: A result dict with a ``timestamp`` field.
        start: Start of the date range (inclusive).
        end: End of the date range (inclusive).

    Returns:
        ``True`` if the timestamp is within the range.
    """
    ts = result.get("timestamp")
    if ts is None:
        return False

    if isinstance(ts, datetime):
        return start <= ts <= end

    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            return start <= dt <= end
        except (ValueError, TypeError):
            return False

    return False


__all__ = ["filter_results"]
