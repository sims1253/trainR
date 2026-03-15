"""Pydantic v2 schemas for the grist-mill framework.

Core data models:
- Task: A benchmark task definition
- TaskResult: Outcome of running a task
- Manifest: Collection of tasks with metadata
- HarnessConfig: Binding of agent, environment, and artifacts
- ExecutionOutput: Raw output from command execution
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grist_mill.schemas.artifact import (
    Artifact,
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Status of a task execution."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


class Difficulty(str, Enum):
    """Difficulty level of a task."""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class ErrorCategory(str, Enum):
    """Categorization of execution errors."""

    TEST_FAILURE = "TEST_FAILURE"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    API_ERROR = "API_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    MAX_TURNS_EXCEEDED = "MAX_TURNS_EXCEEDED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """A benchmark task definition.

    Represents a single task to be evaluated, including the prompt,
    expected test command, and optional constraints.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        strict=False,
    )

    id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this task.",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="Natural language description of the task.",
    )
    language: str = Field(
        ...,
        min_length=1,
        description="Target programming language (e.g., 'python', 'r', 'typescript').",
    )
    test_command: str = Field(
        ...,
        min_length=1,
        description="Command to run to verify the solution.",
    )
    setup_command: str | None = Field(
        default=None,
        description="Optional command to prepare the environment before execution.",
    )
    timeout: int = Field(
        ...,
        gt=0,
        description="Maximum execution time in seconds (must be positive).",
    )
    difficulty: Difficulty = Field(
        default=Difficulty.EASY,
        description="Estimated difficulty of the task.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints on the task (e.g., 'no-network', 'no-filesystem').",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required dependencies for the task environment.",
    )


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


class TaskResult(BaseModel):
    """Outcome of running a task through an agent.

    Captures the status, score, and optional metadata about the execution.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    task_id: str = Field(
        ...,
        min_length=1,
        description="ID of the task this result corresponds to.",
    )
    status: TaskStatus = Field(
        ...,
        description="Final status of the task execution.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 and 1.0 indicating task completion.",
    )
    error_category: ErrorCategory | None = Field(
        default=None,
        description="Category of error if the task did not succeed.",
    )
    telemetry: Any | None = Field(
        default=None,
        description="Telemetry data from the execution (populated by telemetry schema).",
    )
    transcript: list[dict[str, Any]] | None = Field(
        default=None,
        description="Full conversation transcript from the agent.",
    )


# ---------------------------------------------------------------------------
# ExecutionOutput
# ---------------------------------------------------------------------------


class ExecutionOutput(BaseModel):
    """Raw output from a command execution.

    Captures stdout, stderr, exit code, and whether the execution timed out.
    """

    stdout: str = Field(
        default="",
        description="Standard output from the execution.",
    )
    stderr: str = Field(
        default="",
        description="Standard error from the execution.",
    )
    exit_code: int = Field(
        default=0,
        description="Exit code of the process.",
    )
    timed_out: bool = Field(
        default=False,
        description="Whether the execution was killed due to timeout.",
    )


# ---------------------------------------------------------------------------
# HarnessConfig
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Configuration for an AI agent.

    Defines the model, provider, and optional system prompt for the agent.
    """

    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier (e.g., 'gpt-4', 'claude-3-opus').",
    )
    provider: str = Field(
        ...,
        min_length=1,
        description="LLM provider (e.g., 'openrouter', 'openai', 'anthropic').",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the agent.",
    )


class EnvironmentConfig(BaseModel):
    """Configuration for the execution environment.

    Defines the runner type, Docker image, and resource limits.
    """

    runner_type: str = Field(
        ...,
        min_length=1,
        description="Type of runner ('local' or 'docker').",
    )
    docker_image: str | None = Field(
        default=None,
        description="Docker image to use for containerized execution.",
    )
    resource_limits: dict[str, Any] | None = Field(
        default=None,
        description="Resource limits (cpu, memory) for the environment.",
    )


class HarnessConfig(BaseModel):
    """Binding of agent, environment, and artifacts for task execution.

    Defines the complete configuration for running a task through the harness.
    """

    agent: AgentConfig = Field(
        ...,
        description="Agent configuration.",
    )
    environment: EnvironmentConfig = Field(
        ...,
        description="Environment configuration.",
    )
    artifact_bindings: list[str] = Field(
        default_factory=list,
        description="List of artifact names to bind into the execution context.",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class Manifest(BaseModel):
    """Collection of tasks with metadata.

    Aggregates a benchmark's task list along with global metadata (name,
    version, timestamp). Enforces unique task IDs within the manifest.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        description="Name of the benchmark.",
    )
    version: str = Field(
        ...,
        min_length=1,
        description="Version of the benchmark.",
    )
    timestamp: datetime | None = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the manifest was created.",
    )
    tasks: list[Task] = Field(
        default_factory=list,
        description="List of tasks in this benchmark.",
    )

    @model_validator(mode="after")
    def _validate_unique_task_ids(self) -> Manifest:
        """Ensure all task IDs are unique within the manifest."""
        seen_ids: set[str] = set()
        for task in self.tasks:
            if task.id in seen_ids:
                msg = (
                    f"Duplicate task ID found: '{task.id}'. "
                    f"Each task in a manifest must have a unique ID."
                )
                raise ValueError(msg)
            seen_ids.add(task.id)
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AgentConfig",
    "Artifact",
    "Difficulty",
    "EnvironmentConfig",
    "ErrorCategory",
    "ExecutionOutput",
    "HarnessConfig",
    "MCPServerArtifact",
    "Manifest",
    "SkillArtifact",
    "Task",
    "TaskResult",
    "TaskStatus",
    "ToolArtifact",
]
