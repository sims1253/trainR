"""Data models for evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    """Classification of evaluation failures."""

    # Configuration/Environment errors (blocking issues, not test failures)
    CONFIG_ERROR = "CONFIG_ERROR"  # Misconfiguration, missing required config
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"  # Missing env vars, API keys
    PACKAGE_NOT_FOUND = "PACKAGE_NOT_FOUND"  # R package not available

    # Runtime errors
    TIMEOUT = "TIMEOUT"  # Operation timed out

    # Code generation errors
    SYNTAX_ERROR = "SYNTAX_ERROR"  # Generated code has syntax errors
    MISSING_IMPORT = "MISSING_IMPORT"  # Required library not loaded

    # Test failures (expected behavior, not infrastructure errors)
    TEST_FAILURE = "TEST_FAILURE"  # Tests failed (expected behavior)
    WRONG_ASSERTION = "WRONG_ASSERTION"  # Wrong test assertion used
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"  # Snapshot test mismatch
    INCOMPLETE_SOLUTION = "INCOMPLETE_SOLUTION"  # Partial/incomplete code
    OVERLY_COMPLEX = "OVERLY_COMPLEX"  # Solution too complex
    WRONG_FIXTURE_USAGE = "WRONG_FIXTURE_USAGE"  # Incorrect test fixture use


@dataclass
class TestResult:
    """Result of running a single test."""

    name: str
    passed: bool
    message: str = ""
    execution_time: float = 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating a task with a skill."""

    task_id: str
    success: bool
    score: float  # 0.0 to 1.0
    generated_code: str | None = None
    test_results: list[TestResult] = field(default_factory=list)
    error_message: str | None = None
    failure_category: FailureCategory | None = None
    execution_time: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    # Telemetry fields
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_errors: dict[str, int] = field(default_factory=dict)
    tool_total_time_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "score": self.score,
            "generated_code": self.generated_code,
            "test_results": [
                {"name": t.name, "passed": t.passed, "message": t.message}
                for t in self.test_results
            ],
            "error_message": self.error_message,
            "failure_category": str(self.failure_category) if self.failure_category else None,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "tool_call_counts": self.tool_call_counts,
            "tool_errors": self.tool_errors,
            "tool_total_time_ms": self.tool_total_time_ms,
        }


@dataclass
class TrajectoryRecord:
    """Record of an agent's trajectory for GEPA reflection."""

    task_id: str
    skill_version: str
    input_prompt: str
    generated_output: str
    evaluation_result: EvaluationResult
    feedback: str = ""
    improvement_suggestion: str = ""
