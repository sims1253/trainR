"""Cross-experiment telemetry rollup.

Produces a summary table with one row per experiment configuration,
aggregating results from multiple experiment runs.

Validates:
- VAL-REPORT-04: Cross-experiment telemetry rollup
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def cross_experiment_rollup(
    experiments: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Produce a rollup summary across multiple experiments.

    Args:
        experiments: Dict mapping experiment name to list of result dicts.
            Each result dict should have ``score``, ``telemetry`` fields.

    Returns:
        A list of summary dicts, one per experiment, each containing:
        - ``experiment``: The experiment name.
        - ``total_tasks``: Number of tasks.
        - ``passed_tasks``: Number of tasks with score >= 1.0.
        - ``failed_tasks``: Number of tasks with score < 1.0.
        - ``pass_rate``: Fraction of tasks that passed.
        - ``total_cost_usd``: Sum of estimated costs.
        - ``mean_latency_s``: Mean total latency.
        - ``median_latency_s``: Median total latency.
        - ``total_tokens``: Sum of total token counts.
        - ``mean_score``: Mean score across tasks.
        - ``model``: The model used (if consistent across results).
        - ``provider``: The provider used (if consistent across results).

        Sorted by experiment name.
    """
    if not experiments:
        return []

    rollup: list[dict[str, Any]] = []

    for exp_name in sorted(experiments.keys()):
        results = experiments[exp_name]
        if not results:
            continue

        n = len(results)
        scores = [r["score"] for r in results]
        passed = sum(1 for s in scores if s >= 1.0)
        failed = n - passed
        pass_rate = passed / n if n > 0 else 0.0

        # Latencies
        latencies = _extract_latencies(results)
        mean_latency = statistics.mean(latencies) if latencies else 0.0
        median_latency = statistics.median(latencies) if latencies else 0.0

        # Cost
        total_cost = sum(_get_cost(r) for r in results)

        # Tokens
        total_tokens = sum(_get_total_tokens(r) for r in results)

        # Determine consistent model/provider
        models = {r.get("model") for r in results}
        providers = {r.get("provider") for r in results}

        row: dict[str, Any] = {
            "experiment": exp_name,
            "total_tasks": n,
            "passed_tasks": passed,
            "failed_tasks": failed,
            "pass_rate": pass_rate,
            "total_cost_usd": round(total_cost, 6),
            "mean_latency_s": round(mean_latency, 4),
            "median_latency_s": round(median_latency, 4),
            "total_tokens": total_tokens,
            "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "model": next(iter(models)) if len(models) == 1 else "mixed",
            "provider": next(iter(providers)) if len(providers) == 1 else "mixed",
        }

        rollup.append(row)

    logger.debug(
        "Cross-experiment rollup: %d experiments",
        len(rollup),
    )
    return rollup


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


__all__ = ["cross_experiment_rollup"]
