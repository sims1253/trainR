"""Harness abstraction layer for agent execution.

This module provides a pluggable interface for different agent execution
backends (Pi SDK, Pi CLI, Codex, Claude Code, etc.).

Canonical usage:
    from bench.harness import HarnessRegistry, PiDockerHarness

    harness = HarnessRegistry.get("pi_docker", config)
    result = harness.execute(request)
"""

from .adapters import (
    ClaudeCliHarness,
    CliHarnessBase,
    CodexCliHarness,
    GeminiCliHarness,
    PiDockerHarness,
)
from .base import (
    AgentHarness,
    ErrorCategory,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    TestResult,
    TokenUsage,
)
from .registry import HarnessFactory, HarnessRegistry, register_harness

__all__ = [
    # Protocol
    "AgentHarness",
    "ClaudeCliHarness",
    "CliHarnessBase",
    "CodexCliHarness",
    "ErrorCategory",
    "GeminiCliHarness",
    "HarnessConfig",
    "HarnessFactory",
    # Registry
    "HarnessRegistry",
    # Base types
    "HarnessRequest",
    "HarnessResult",
    # Adapters
    "PiDockerHarness",
    "TestResult",
    "TokenUsage",
    "register_harness",
]
