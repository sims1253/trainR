"""Data models for task generation."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TestPattern(str, Enum):
    """testthat testing patterns."""

    EXPECT_EQUAL = "expect_equal"
    EXPECT_SNAPSHOT = "expect_snapshot"
    EXPECT_ERROR = "expect_error"
    EXPECT_WARNING = "expect_warning"
    EXPECT_MESSAGE = "expect_message"
    EXPECT_TRUE = "expect_true"
    EXPECT_FALSE = "expect_false"
    EXPECT_TYPE = "expect_type"
    EXPECT_S3_CLASS = "expect_s3_class"
    DESCRIBE_IT = "describe_it"
    WITH_FIXTURE = "with_fixture"
    LOCAL_MOCKED_BINDINGS = "local_mocked_bindings"
    TEST_THAT = "test_that"
    SKIP_IF = "skip_if"

    def __str__(self) -> str:
        return self.value


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    def __str__(self) -> str:
        return self.value


@dataclass
class ExtractedPattern:
    """A testing pattern extracted from source code."""

    pattern_type: TestPattern
    source_file: str
    line_number: int
    code_snippet: str
    function_name: str | None = None
    context_before: str = ""
    context_after: str = ""
    expectations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": str(self.pattern_type),
            "source_file": self.source_file,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "function_name": self.function_name,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "expectations": self.expectations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedPattern":
        """Create from dictionary."""
        return cls(
            pattern_type=TestPattern(data["pattern_type"]),
            source_file=data["source_file"],
            line_number=data["line_number"],
            code_snippet=data["code_snippet"],
            function_name=data.get("function_name"),
            context_before=data.get("context_before", ""),
            context_after=data.get("context_after", ""),
            expectations=data.get("expectations", []),
        )


@dataclass
class TestingTask:
    """A synthetic testing task for R packages."""

    task_id: str
    source_package: str
    source_file: str
    difficulty: Difficulty
    instruction: str
    context: str  # Code context (function to test)
    reference_test: str  # Ground truth test
    test_type: str  # unit, snapshot, integration
    patterns: list[TestPattern]
    dependencies: list[str]
    split: str  # train, dev, held_out
    function_name: str | None = None
    quality_score: float = 0.0
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_package": self.source_package,
            "source_file": self.source_file,
            "difficulty": str(self.difficulty),
            "instruction": self.instruction,
            "context": self.context,
            "reference_test": self.reference_test,
            "test_type": self.test_type,
            "patterns": [str(p) for p in self.patterns],
            "dependencies": self.dependencies,
            "split": self.split,
            "function_name": self.function_name,
            "quality_score": self.quality_score,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestingTask":
        data = data.copy()  # Don't modify the original
        data["difficulty"] = Difficulty(data["difficulty"])
        data["patterns"] = [TestPattern(p) for p in data["patterns"]]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
