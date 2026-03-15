"""Reporting module for grist-mill.

Provides:
- comparison: Experiment comparison with per-task deltas and significance testing
- aggregation: Telemetry aggregation (per-model, per-tool, per-experiment)
- tools: Per-tool performance breakdown
- rollup: Cross-experiment rollup
- errors: Error taxonomy breakdown
- filtering: Result filtering by model, tool, date range
"""

from __future__ import annotations

from grist_mill.reports.aggregation import aggregate_telemetry
from grist_mill.reports.comparison import compare_experiments
from grist_mill.reports.errors import error_taxonomy_breakdown
from grist_mill.reports.filtering import filter_results
from grist_mill.reports.rollup import cross_experiment_rollup
from grist_mill.reports.tools import tool_performance_breakdown

__all__ = [
    "aggregate_telemetry",
    "compare_experiments",
    "cross_experiment_rollup",
    "error_taxonomy_breakdown",
    "filter_results",
    "tool_performance_breakdown",
]
