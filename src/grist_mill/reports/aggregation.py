"""Telemetry aggregation: per-model, per-tool, per-experiment summaries.

Collapses per-task results into grouped summaries with mean pass-rate,
standard deviation, median latency, total cost, and token counts.

Validates:
- VAL-REPORT-02: Telemetry aggregation produces per-model summaries
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def aggregate_telemetry(
    results: list[dict[str, Any]],
    group_by: str = "model",
) -> list[dict[str, Any]]:
    """Aggregate telemetry across results, grouped by a specified field.

    Args:
        results: List of result dicts. Each must have at least:
            - ``score`` (float): Task score (0.0-1.0)
            - ``telemetry``: TelemetrySchema or dict with ``tokens``,
              ``latency``, ``estimated_cost_usd`` fields.
            - ``group_by`` field: The field to group by (e.g., ``model``).
        group_by: Field name to group results by.
            Common values: ``"model"``, ``"experiment"``, ``"provider"``.

    Returns:
        A list of summary dicts, one per group. Each contains:
        - ``group``: The group key value.
        - ``pass_rate``: Fraction of tasks with score >= 1.0.
        - ``pass_rate_std``: Standard deviation (0.0 for single-task groups).
        - ``total_tasks``: Number of tasks in the group.
        - ``passed_tasks``: Number of tasks with score >= 1.0.
        - ``failed_tasks``: Number of tasks with score < 1.0.
        - ``total_cost_usd``: Sum of estimated costs.
        - ``mean_latency_s``: Mean total latency.
        - ``median_latency_s``: Median total latency.
        - ``total_tokens``: Sum of total token counts.
        - ``mean_tokens``: Mean total token count per task.
    """
    if not results:
        return []

    # Group results
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        key = r.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    summaries: list[dict[str, Any]] = []
    for group_key, group_results in sorted(groups.items()):
        n = len(group_results)
        scores = [r["score"] for r in group_results]

        # Pass rate: fraction with score >= 1.0
        passed = sum(1 for s in scores if s >= 1.0)
        failed = n - passed
        pass_rate = passed / n if n > 0 else 0.0

        # Standard deviation of pass indicators (binary)
        if n > 1:
            pass_indicators = [1.0 if s >= 1.0 else 0.0 for s in scores]
            pass_rate_std = statistics.stdev(pass_indicators)
        else:
            pass_rate_std = 0.0

        # Latencies
        latencies = _extract_latencies(group_results)
        mean_latency = statistics.mean(latencies) if latencies else 0.0
        median_latency = statistics.median(latencies) if latencies else 0.0

        # Costs
        total_cost = sum(_get_cost(r) for r in group_results)

        # Tokens
        total_tokens = sum(_get_total_tokens(r) for r in group_results)
        mean_tokens = total_tokens / n if n > 0 else 0.0

        summary: dict[str, Any] = {
            "group": group_key,
            "pass_rate": pass_rate,
            "pass_rate_std": pass_rate_std,
            "total_tasks": n,
            "passed_tasks": passed,
            "failed_tasks": failed,
            "total_cost_usd": round(total_cost, 6),
            "mean_latency_s": round(mean_latency, 4),
            "median_latency_s": round(median_latency, 4),
            "total_tokens": total_tokens,
            "mean_tokens": round(mean_tokens, 1),
        }

        summaries.append(summary)

    logger.debug(
        "Aggregated %d results into %d groups (by '%s')",
        len(results),
        len(summaries),
        group_by,
    )
    return summaries


def _extract_latencies(results: list[dict[str, Any]]) -> list[float]:
    """Extract total latency values from results.

    Args:
        results: List of result dicts with telemetry.

    Returns:
        List of total latency values in seconds.
    """
    latencies: list[float] = []
    for r in results:
        telemetry = r.get("telemetry")
        if telemetry is None:
            continue
        if hasattr(telemetry, "latency"):
            latencies.append(telemetry.latency.total_s)
        elif isinstance(telemetry, dict):
            lat_data = telemetry.get("latency", {})
            if isinstance(lat_data, dict):
                latencies.append(lat_data.get("total_s", 0.0))
    return latencies


def _get_cost(result: dict[str, Any]) -> float:
    """Get the estimated cost from a result.

    Args:
        result: A result dict with telemetry.

    Returns:
        Estimated cost in USD, or 0.0 if not available.
    """
    telemetry = result.get("telemetry")
    if telemetry is None:
        return 0.0
    if hasattr(telemetry, "estimated_cost_usd"):
        return telemetry.estimated_cost_usd or 0.0
    if isinstance(telemetry, dict):
        return telemetry.get("estimated_cost_usd") or 0.0
    return 0.0


def _get_total_tokens(result: dict[str, Any]) -> int:
    """Get total token count from a result.

    Args:
        result: A result dict with telemetry.

    Returns:
        Total token count, or 0 if not available.
    """
    telemetry = result.get("telemetry")
    if telemetry is None:
        return 0
    if hasattr(telemetry, "tokens"):
        return telemetry.tokens.total
    if isinstance(telemetry, dict):
        tok_data = telemetry.get("tokens", {})
        if isinstance(tok_data, dict):
            return tok_data.get("total", 0)
    return 0


__all__ = ["aggregate_telemetry"]
