"""Result schemas for benchmark evaluation results.

Captures:
- Task evaluation results (pass/fail, scores, errors)
- Test execution details
- Token usage and latency metrics
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ErrorCategoryV1(str, Enum):
    """Error categories for failed evaluations."""

    SYNTAX_ERROR = "SYNTAX_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    TIMEOUT = "TIMEOUT"
    MISSING_IMPORT = "MISSING_IMPORT"
    WRONG_ASSERTION = "WRONG_ASSERTION"
    INCOMPLETE_SOLUTION = "INCOMPLETE_SOLUTION"
    OVERLY_COMPLEX = "OVERLY_COMPLEX"
    WRONG_FIXTURE_USAGE = "WRONG_FIXTURE_USAGE"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"
    LLM_ERROR = "LLM_ERROR"
    SANDBOX_ERROR = "SANDBOX_ERROR"
    UNKNOWN = "UNKNOWN"


class CaseResultV1(BaseModel):
    """Result of running a single test case."""

    name: str = Field(description="Test name")
    passed: bool = Field(description="Whether the test passed")
    message: str = Field(default="", description="Test output or error message")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")


class TokenUsageV1(BaseModel):
    """Token usage metrics."""

    prompt: int = Field(default=0, description="Prompt tokens")
    completion: int = Field(default=0, description="Completion tokens")
    total: int = Field(default=0, description="Total tokens")

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Auto-calculate total if not provided
        if self.total == 0 and (self.prompt > 0 or self.completion > 0):
            self.total = self.prompt + self.completion


class ResultV1(BaseModel):
    """
    Canonical result schema for benchmark evaluation.

    This schema captures the outcome of evaluating a single task
    with a specific model/configuration.
    """

    # Schema versioning
    schema_version: str = Field(default="1.0", description="Schema version")

    # Core identification
    result_id: str = Field(default="", description="Unique identifier for this result")
    task_id: str = Field(description="ID of the evaluated task")
    model: str = Field(description="Model used for evaluation")
    profile_id: str = Field(default="default", description="Profile used for evaluation")

    # Outcome
    passed: bool = Field(description="Whether the evaluation passed")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score from 0.0 to 1.0")

    # Error information
    error_category: ErrorCategoryV1 | None = Field(
        default=None,
        description="Category of error if failed",
    )
    error_message: str | None = Field(default=None, description="Error message if failed")

    # Metrics
    latency_s: float = Field(default=0.0, ge=0.0, description="Total latency in seconds")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    token_usage: TokenUsageV1 = Field(
        default_factory=TokenUsageV1,
        description="Token usage metrics",
    )

    # Test details
    test_results: list[CaseResultV1] = Field(
        default_factory=list,
        description="Individual test results",
    )

    # Output artifacts
    generated_code: str | None = Field(default=None, description="Generated code output")
    trajectory_path: str | None = Field(default=None, description="Path to trajectory file")

    # Repetition tracking
    repeat_index: int = Field(default=0, ge=0, description="Repetition index (0-based)")

    # Timestamps
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the result was recorded",
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Telemetry fields (tool-level metrics)
    tool_call_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Number of calls per tool (tool_name -> count)",
    )
    tool_errors: dict[str, int] = Field(
        default_factory=dict,
        description="Error count per tool (tool_name -> error_count)",
    )
    tool_total_time_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Total time per tool in milliseconds (tool_name -> ms)",
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Ensure schema version is valid."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported schema version: {v}. Expected 1.x")
        return v

    @field_validator("score", mode="after")
    @classmethod
    def validate_score_consistency(cls, v: float, info: Any) -> float:
        """Ensure score is consistent with passed status."""
        # If passed, score should be high; if failed, score should be low
        # This is a soft validation - we allow any valid score
        return v

    def to_json_schema(self) -> dict[str, Any]:
        """Export this model's JSON schema."""
        return ResultV1.model_json_schema()

    @property
    def test_count(self) -> int:
        """Total number of tests."""
        return len(self.test_results)

    @property
    def passed_count(self) -> int:
        """Number of passed tests."""
        return sum(1 for t in self.test_results if t.passed)

    @property
    def failed_count(self) -> int:
        """Number of failed tests."""
        return self.test_count - self.passed_count

    @property
    def pass_rate(self) -> float:
        """Test pass rate as fraction."""
        if self.test_count == 0:
            return 0.0
        return self.passed_count / self.test_count

    @classmethod
    def from_legacy_benchmark_result(cls, data: dict[str, Any]) -> "ResultV1":
        """
        Convert from legacy BenchmarkResult format to ResultV1.

        Args:
            data: Dictionary from BenchmarkResult.to_dict()

        Returns:
            ResultV1 instance
        """
        # Map error categories
        error_category_map = {
            "SYNTAX_ERROR": ErrorCategoryV1.SYNTAX_ERROR,
            "TEST_FAILURE": ErrorCategoryV1.TEST_FAILURE,
            "TIMEOUT": ErrorCategoryV1.TIMEOUT,
            "MISSING_IMPORT": ErrorCategoryV1.MISSING_IMPORT,
            "WRONG_ASSERTION": ErrorCategoryV1.WRONG_ASSERTION,
            "INCOMPLETE_SOLUTION": ErrorCategoryV1.INCOMPLETE_SOLUTION,
            "OVERLY_COMPLEX": ErrorCategoryV1.OVERLY_COMPLEX,
            "WRONG_FIXTURE_USAGE": ErrorCategoryV1.WRONG_FIXTURE_USAGE,
            "SNAPSHOT_MISMATCH": ErrorCategoryV1.SNAPSHOT_MISMATCH,
        }

        legacy_error = data.get("error_category")
        error_category = None
        if legacy_error:
            error_category = error_category_map.get(legacy_error, ErrorCategoryV1.UNKNOWN)

        # Convert token usage
        token_data = data.get("token_usage", {})
        token_usage = TokenUsageV1(
            prompt=token_data.get("prompt", 0),
            completion=token_data.get("completion", 0),
            total=token_data.get("total", 0),
        )

        # Convert test results
        test_results = [
            CaseResultV1(
                name=t.get("name", ""),
                passed=t.get("passed", False),
                message=t.get("message", ""),
                execution_time=t.get("execution_time", 0.0),
            )
            for t in data.get("test_results", [])
        ]

        return cls(
            task_id=data.get("task_id", ""),
            model=data.get("model", ""),
            passed=data.get("passed", False),
            score=data.get("score", 0.0),
            error_category=error_category,
            error_message=data.get("error_message"),
            latency_s=data.get("latency_s", 0.0),
            token_usage=token_usage,
            test_results=test_results,
            trajectory_path=data.get("trajectory_path"),
            repeat_index=data.get("repeat_index", 0),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def from_legacy_evaluation_result(cls, data: dict[str, Any], model: str = "") -> "ResultV1":
        """
        Convert from legacy EvaluationResult format to ResultV1.

        Args:
            data: Dictionary from EvaluationResult.to_dict()
            model: Model name (not in legacy format)

        Returns:
            ResultV1 instance
        """
        # Map failure categories
        failure_category_map = {
            "SYNTAX_ERROR": ErrorCategoryV1.SYNTAX_ERROR,
            "TEST_FAILURE": ErrorCategoryV1.TEST_FAILURE,
            "TIMEOUT": ErrorCategoryV1.TIMEOUT,
            "MISSING_IMPORT": ErrorCategoryV1.MISSING_IMPORT,
            "WRONG_ASSERTION": ErrorCategoryV1.WRONG_ASSERTION,
            "INCOMPLETE_SOLUTION": ErrorCategoryV1.INCOMPLETE_SOLUTION,
            "OVERLY_COMPLEX": ErrorCategoryV1.OVERLY_COMPLEX,
            "WRONG_FIXTURE_USAGE": ErrorCategoryV1.WRONG_FIXTURE_USAGE,
            "SNAPSHOT_MISMATCH": ErrorCategoryV1.SNAPSHOT_MISMATCH,
        }

        legacy_failure = data.get("failure_category")
        error_category = None
        if legacy_failure:
            error_category = failure_category_map.get(legacy_failure, ErrorCategoryV1.UNKNOWN)

        # Convert token usage
        token_data = data.get("token_usage", {})
        token_usage = TokenUsageV1(
            prompt=token_data.get("prompt", 0),
            completion=token_data.get("completion", 0),
            total=token_data.get("total", 0),
        )

        # Convert test results
        test_results = [
            CaseResultV1(
                name=t.get("name", ""),
                passed=t.get("passed", False),
                message=t.get("message", ""),
                execution_time=t.get("execution_time", 0.0),
            )
            for t in data.get("test_results", [])
        ]

        return cls(
            task_id=data.get("task_id", ""),
            model=model,
            passed=data.get("success", False),
            score=data.get("score", 0.0),
            error_category=error_category,
            error_message=data.get("error_message"),
            latency_s=data.get("execution_time", 0.0),
            execution_time=data.get("execution_time", 0.0),
            token_usage=token_usage,
            test_results=test_results,
            generated_code=data.get("generated_code"),
        )


def validate_result(data: dict[str, Any]) -> ResultV1:
    """
    Validate result data and return a ResultV1 instance.

    Args:
        data: Raw result data dictionary

    Returns:
        Validated ResultV1 instance

    Raises:
        ValidationError: If data doesn't conform to ResultV1 schema
    """
    return ResultV1.model_validate(data)
