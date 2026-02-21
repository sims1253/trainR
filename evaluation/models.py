"""Data models for evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    TIMEOUT = "TIMEOUT"
    MISSING_IMPORT = "MISSING_IMPORT"
    WRONG_ASSERTION = "WRONG_ASSERTION"
    INCOMPLETE_SOLUTION = "INCOMPLETE_SOLUTION"
    OVERLY_COMPLEX = "OVERLY_COMPLEX"
    WRONG_FIXTURE_USAGE = "WRONG_FIXTURE_USAGE"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"


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
