"""Shared pytest fixtures for grist-mill tests."""

from __future__ import annotations

# Tests that import from application modules (bench/, evaluation/) rather than
# the grist-mill package. These are collected-ignored from the package test suite
# but can be run directly when needed (e.g., uv run pytest tests/test_sandbox_docker.py).
collect_ignore = [
    "test_experiment_config_behavior.py",
    "test_experiment_kaggle_scoring.py",
    "test_provider_parallel_limits.py",
    "test_sandbox_docker.py",
    "test_config_import.py",
]
