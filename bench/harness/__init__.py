"""Harness abstraction layer for agent execution.

This module provides a pluggable interface for different agent execution
backends (Pi SDK, Pi CLI, Codex, Claude Code, etc.).

Canonical usage:
    from bench.harness import HarnessRegistry, PiDockerHarness

    harness = HarnessRegistry.get("pi_docker", config)
    result = harness.execute(request)
"""

from .base import (
    HarnessRequest,
    HarnessResult,
    HarnessConfig,
    TokenUsage,
    TestResult,
    ErrorCategory,
    AgentHarness,
)
from .registry import HarnessRegistry, HarnessFactory, register_harness
from .adapters import PiDockerHarness

__all__ = [
    # Base types
    "HarnessRequest",
    "HarnessResult",
    "HarnessConfig",
    "TokenUsage",
    "TestResult",
    "ErrorCategory",
    # Protocol
    "AgentHarness",
    # Registry
    "HarnessRegistry",
    "HarnessFactory",
    "register_harness",
    # Adapters
    "PiDockerHarness",
]
