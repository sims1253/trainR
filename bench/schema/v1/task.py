"""Task schema for benchmark evaluation tasks.

Supports task types:
- swe: Software engineering tasks (bug fixes, features)
- test_gen: Test generation tasks
- skill: Skill evaluation tasks
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskTypeV1(str, Enum):
    """Canonical task types for benchmark evaluation."""

    SWE = "swe"  # Software engineering (bug fixes, features)
    TEST_GEN = "test_gen"  # Test generation
    SKILL = "skill"  # Skill evaluation tasks


class DifficultyV1(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskSourceV1(BaseModel):
    """Source information for a task."""

    type: str = Field(default="github_pr", description="Source type (github_pr, synthetic, manual)")
    repo: str | None = Field(default=None, description="Repository name (owner/repo)")
    pr_number: int | None = Field(default=None, description="Pull request number")
    pr_url: str | None = Field(default=None, description="URL to the pull request")
    merged_at: str | None = Field(default=None, description="When the PR was merged (ISO format)")
    issue_number: int | None = Field(default=None, description="Linked issue number")
    issue_url: str | None = Field(default=None, description="URL to the linked issue")


class TaskTestsV1(BaseModel):
    """Test specifications for task validation."""

    fail_to_pass: list[str] = Field(
        default_factory=list,
        description="Tests that should fail before fix, pass after",
    )
    pass_to_pass: list[str] = Field(
        default_factory=list,
        description="Tests that should always pass",
    )
    test_files: list[str] = Field(
        default_factory=list,
        description="Test file paths to run",
    )


class TaskSolutionV1(BaseModel):
    """Solution hints and reference information."""

    key_changes: list[str] = Field(default_factory=list, description="Key changes needed")
    potential_pitfalls: list[str] = Field(
        default_factory=list, description="Common mistakes to avoid"
    )
    reference_diff: str | None = Field(default=None, description="Reference solution diff")


class TaskFilesV1(BaseModel):
    """Files related to the task."""

    changed: list[str] = Field(default_factory=list, description="Files that were changed")
    source_files: list[str] = Field(default_factory=list, description="Source files to modify")
    test_changes: str | None = Field(default=None, description="Test file changes")


class TaskV1(BaseModel):
    """
    Canonical task schema for benchmark evaluation.

    This schema represents the canonical format for all task types in the benchmark
    system. It supports SWE tasks, test generation, and skill evaluation.
    """

    # Schema versioning
    schema_version: str = Field(default="1.0", description="Schema version")

    # Core identification
    task_id: str = Field(description="Unique identifier for the task")
    task_type: TaskTypeV1 = Field(description="Type of task")
    difficulty: DifficultyV1 = Field(default=DifficultyV1.MEDIUM, description="Difficulty level")

    # Task content
    instruction: str = Field(description="Task instruction for the agent")
    context: str = Field(default="", description="Code context to help understand the task")
    hints: list[str] = Field(default_factory=list, description="Optional hints for solving")

    # Package/source info
    source_package: str = Field(default="", description="Source package name")
    source_file: str | None = Field(default=None, description="Primary source file")

    # Test specifications
    tests: TaskTestsV1 = Field(default_factory=TaskTestsV1, description="Test specifications")

    # Solution information (optional, for evaluation)
    solution: TaskSolutionV1 | None = Field(default=None, description="Solution hints")

    # Source provenance
    source: TaskSourceV1 | None = Field(default=None, description="Source information")

    # Related files
    files: TaskFilesV1 = Field(default_factory=TaskFilesV1, description="Related files")

    # Metadata
    quality_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Quality rating 0-10")
    split: str = Field(default="dev", description="Dataset split (train, dev, held_out, mined)")
    dependencies: list[str] = Field(default_factory=list, description="Package dependencies")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Task constraints")

    # Timestamps
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the task was created",
    )
    updated_at: str | None = Field(default=None, description="When the task was last updated")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Ensure schema version is valid."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported schema version: {v}. Expected 1.x")
        return v

    def to_json_schema(self) -> dict[str, Any]:
        """Export this model's JSON schema."""
        return TaskV1.model_json_schema()

    @classmethod
    def from_legacy_mined_task(cls, data: dict[str, Any]) -> "TaskV1":
        """
        Convert from legacy MinedTask format to TaskV1.

        Args:
            data: Dictionary from MinedTask.model_dump() or similar

        Returns:
            TaskV1 instance
        """
        # Map legacy task types
        task_type_map = {
            "bug_fix": TaskTypeV1.SWE,
            "feature_impl": TaskTypeV1.SWE,
            "test_writing": TaskTypeV1.TEST_GEN,
            "refactor": TaskTypeV1.SWE,
            "documentation": TaskTypeV1.SWE,
            "performance": TaskTypeV1.SWE,
            "security": TaskTypeV1.SWE,
            "dependency": TaskTypeV1.SWE,
        }

        # Map difficulty
        difficulty_map = {
            "easy": DifficultyV1.EASY,
            "medium": DifficultyV1.MEDIUM,
            "hard": DifficultyV1.HARD,
        }

        # Extract source info
        source_data = data.get("source", {})
        source = (
            TaskSourceV1(
                type=source_data.get("type", "github_pr"),
                repo=source_data.get("repo"),
                pr_number=source_data.get("pr_number"),
                pr_url=source_data.get("pr_url"),
                merged_at=source_data.get("merged_at"),
                issue_number=source_data.get("issue_number"),
                issue_url=source_data.get("issue_url"),
            )
            if source_data
            else None
        )

        # Extract tests
        tests_data = data.get("tests", {})
        tests = TaskTestsV1(
            fail_to_pass=tests_data.get("fail_to_pass", []),
            pass_to_pass=tests_data.get("pass_to_pass", []),
            test_files=tests_data.get("test_files", []),
        )

        # Extract solution
        solution_data = data.get("solution", {})
        solution = (
            TaskSolutionV1(
                key_changes=solution_data.get("key_changes", []),
                potential_pitfalls=solution_data.get("potential_pitfalls", []),
                reference_diff=solution_data.get("reference_diff"),
            )
            if solution_data
            else None
        )

        # Extract files
        files_data = data.get("files", {})
        files = TaskFilesV1(
            changed=files_data.get("changed", []),
            source_files=files_data.get("source_files", []),
            test_changes=files_data.get("test_changes"),
        )

        # Extract task definition
        task_data = data.get("task", {})
        metadata = data.get("metadata", {})

        legacy_task_type = metadata.get("task_type", "bug_fix")
        task_type = task_type_map.get(legacy_task_type, TaskTypeV1.SWE)

        legacy_difficulty = metadata.get("difficulty", "medium")
        difficulty = difficulty_map.get(legacy_difficulty, DifficultyV1.MEDIUM)

        return cls(
            task_id=data.get("task_id", ""),
            task_type=task_type,
            difficulty=difficulty,
            instruction=task_data.get("instruction", ""),
            context=task_data.get("context", ""),
            hints=task_data.get("hints", []),
            source_package=data.get(
                "source_package", source.repo.split("/")[-1] if source and source.repo else ""
            ),
            tests=tests,
            solution=solution,
            source=source,
            files=files,
            quality_score=float(metadata.get("quality_score", 0)),
            split="mined",
            created_at=metadata.get("mined_at", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def from_legacy_testing_task(cls, data: dict[str, Any]) -> "TaskV1":
        """
        Convert from legacy TestingTask format to TaskV1.

        Args:
            data: Dictionary from TestingTask.to_dict() or similar

        Returns:
            TaskV1 instance
        """
        # Map difficulty
        difficulty_map = {
            "easy": DifficultyV1.EASY,
            "medium": DifficultyV1.MEDIUM,
            "hard": DifficultyV1.HARD,
        }

        legacy_difficulty = data.get("difficulty", "medium")
        difficulty = difficulty_map.get(str(legacy_difficulty).lower(), DifficultyV1.MEDIUM)

        source_file = data.get("source_file")
        source_file_value = str(source_file) if source_file else None

        return cls(
            task_id=data.get("task_id", ""),
            task_type=TaskTypeV1.TEST_GEN,
            difficulty=difficulty,
            instruction=data.get("instruction", ""),
            context=data.get("context", ""),
            source_package=data.get("source_package", ""),
            source_file=source_file_value,
            tests=TaskTestsV1(
                fail_to_pass=[],
                pass_to_pass=[],
                test_files=data.get("dependencies", []),
            ),
            files=TaskFilesV1(
                changed=[source_file_value] if source_file_value else [],
                source_files=[source_file_value] if source_file_value else [],
            ),
            quality_score=float(data.get("quality_score", 0)),
            split=data.get("split", "dev"),
            dependencies=data.get("dependencies", []),
            constraints=data.get("constraints", {}),
        )


def validate_task(data: dict[str, Any]) -> TaskV1:
    """
    Validate task data and return a TaskV1 instance.

    Args:
        data: Raw task data dictionary

    Returns:
        Validated TaskV1 instance

    Raises:
        ValidationError: If data doesn't conform to TaskV1 schema
    """
    return TaskV1.model_validate(data)
