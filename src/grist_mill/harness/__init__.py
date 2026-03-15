"""harness module — evaluation harness components.

Provides:
- Harness: Reference evaluation harness wiring task -> env -> agent -> test -> result
- ResultParser: Converts raw test output to structured TaskResult
- run_experiment: Convenience function for running multiple tasks
- InterruptibleExperiment: SIGTERM-safe multi-task runner with partial manifest
"""

from __future__ import annotations

from grist_mill.harness.harness import Harness, InterruptibleExperiment, run_experiment
from grist_mill.harness.result_parser import ResultParser

__all__ = [
    "Harness",
    "InterruptibleExperiment",
    "ResultParser",
    "run_experiment",
]
