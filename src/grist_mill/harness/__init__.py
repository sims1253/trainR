"""harness module — evaluation harness components.

Provides:
- Harness: Reference evaluation harness wiring task -> env -> agent -> test -> result
- ResultParser: Converts raw test output to structured TaskResult
- run_experiment: Convenience function for running multiple tasks
"""

from __future__ import annotations

from grist_mill.harness.harness import Harness, run_experiment
from grist_mill.harness.result_parser import ResultParser

__all__ = [
    "Harness",
    "ResultParser",
    "run_experiment",
]
