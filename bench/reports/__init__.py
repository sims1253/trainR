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
    # Support profile pairing (WP-04B)
    SupportPairDelta,
    SupportPairReport,
    ToolPairDelta,
    analyze_tool_ab_matrix,
    bootstrap_significance_test,
    compute_paired_deltas,
    compute_support_pair_deltas,
    emit_support_pair_deltas,
    emit_support_pair_report,
    generate_paired_tool_report,
    generate_support_pair_report,
    load_paired_support_results,
    process_support_pair_experiment,
    save_paired_report,
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
    "BootstrapStats",
    # Backward compatibility
    "PairedDelta",
    "PairedDeltaReport",
    "PairedToolReport",
    # Support profile pairing (WP-04B)
    "SupportPairDelta",
    "SupportPairReport",
    # Tool A/B pairing (primary)
    "ToolPairDelta",
    "analyze_tool_ab_matrix",
    "bootstrap_significance_test",
    "compute_paired_deltas",
    "compute_support_pair_deltas",
    "emit_paired_deltas",
    "emit_support_pair_deltas",
    "emit_support_pair_report",
    "generate_paired_tool_report",
    "generate_support_pair_report",
    "load_paired_results",
    "load_paired_support_results",
    "process_support_pair_experiment",
    "save_paired_report",
]
