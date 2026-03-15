"""Error taxonomy breakdown.

Groups and counts errors by category from the error taxonomy.
Produces a breakdown of which error categories occurred and their
frequency, useful for identifying systematic issues.

Validates:
- VAL-REPORT-05: Error taxonomy breakdown
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def error_taxonomy_breakdown(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Produce an error category breakdown from results.

    Only includes results that have a non-null ``error_category``.
    Results with ``status`` SUCCESS or ``error_category`` of None
    are excluded from the breakdown.

    Args:
        results: List of result dicts. Each must have ``error_category``
            and ``status`` fields.

    Returns:
        A list of error category dicts, sorted by count descending,
        each containing:
        - ``error_category``: The error category name (string).
        - ``count``: Number of results with this category.
        - ``percentage``: Percentage of error results with this category
          (sums to 100% across all categories).
    """
    # Count errors by category
    category_counts: dict[str, int] = {}
    for r in results:
        cat = r.get("error_category")
        if cat is None:
            continue
        cat_str = cat.value if hasattr(cat, "value") else str(cat)
        category_counts[cat_str] = category_counts.get(cat_str, 0) + 1

    if not category_counts:
        return []

    total_errors = sum(category_counts.values())

    # Build breakdown sorted by count descending
    breakdown: list[dict[str, Any]] = []
    for cat_name in sorted(
        category_counts.keys(),
        key=lambda k: category_counts[k],
        reverse=True,
    ):
        count = category_counts[cat_name]
        pct = (count / total_errors * 100.0) if total_errors > 0 else 0.0
        breakdown.append(
            {
                "error_category": cat_name,
                "count": count,
                "percentage": round(pct, 2),
            }
        )

    logger.debug(
        "Error breakdown: %d categories, %d total errors",
        len(breakdown),
        total_errors,
    )
    return breakdown


__all__ = ["error_taxonomy_breakdown"]
