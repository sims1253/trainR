"""Reports module for experiment analysis.

This module provides tools for analyzing experiment results:
- Tool A/B comparison analysis
- Support profile pairing (WP-04B)
- Paired delta computation
- Statistical significance testing (bootstrap)
- Report generation
"""

from bench.reports.paired import (
    # Tool A/B pairing
    BootstrapStats,
    PairedToolReport,
    ToolPairDelta,
    analyze_tool_ab_matrix,
    bootstrap_significance_test,
    compute_paired_deltas,
    generate_paired_tool_report,
    save_paired_report,
    # Support profile pairing (WP-04B)
    SupportPairDelta,
    SupportPairReport,
    compute_support_pair_deltas,
    emit_support_pair_deltas,
    emit_support_pair_report,
    generate_support_pair_report,
    load_paired_support_results,
    process_support_pair_experiment,
)

# Backward compatibility aliases
PairedDelta = ToolPairDelta
PairedDeltaReport = PairedToolReport
emit_paired_deltas = save_paired_report


def load_paired_results(path):
    """Load paired results from file (backward compatibility)."""
    import json
    from pathlib import Path

    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data


__all__ = [
    # Tool A/B pairing (primary)
    "ToolPairDelta",
    "BootstrapStats",
    "PairedToolReport",
    "bootstrap_significance_test",
    "compute_paired_deltas",
    "generate_paired_tool_report",
    "analyze_tool_ab_matrix",
    "save_paired_report",
    # Support profile pairing (WP-04B)
    "SupportPairDelta",
    "SupportPairReport",
    "compute_support_pair_deltas",
    "emit_support_pair_deltas",
    "emit_support_pair_report",
    "generate_support_pair_report",
    "load_paired_support_results",
    "process_support_pair_experiment",
    # Backward compatibility
    "PairedDelta",
    "PairedDeltaReport",
    "emit_paired_deltas",
    "load_paired_results",
]
