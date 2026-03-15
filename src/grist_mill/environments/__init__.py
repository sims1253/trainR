"""Environments module — execution environment implementations.

Provides runners for executing commands in different environments:
- LocalRunner: Host subprocess execution (fast dev iteration)
"""

from grist_mill.environments.local_runner import LocalRunner

__all__ = [
    "LocalRunner",
]
