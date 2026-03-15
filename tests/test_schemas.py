"""Tests for grist-mill core schemas: Task, TaskResult, Manifest, HarnessConfig, ExecutionOutput."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ErrorCategory,
    ExecutionOutput,
    HarnessConfig,
    Manifest,
    Task,
    TaskResult,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_task_data() -> dict[str, Any]:
    """Return a valid task data dictionary."""
    return {
        "id": "task-001",
        "prompt": "Fix the failing test in the calculator module.",
        "language": "python",
        "test_command": "pytest tests/test_calc.py -x",
        "setup_command": "pip install -e .",
        "timeout": 300,
        "difficulty": Difficulty.MEDIUM,
        "constraints": ["no-network"],
        "dependencies": ["numpy>=1.20"],
    }


@pytest.fixture
def valid_task(valid_task_data: dict[str, Any]) -> Task:
    """Return a valid Task instance."""
    return Task(**valid_task_data)


# ===========================================================================
# Task Model Tests
# ===========================================================================


class TestTaskConstruction:
    """VAL-FOUND-01: Task model validates required fields and rejects invalid input."""

    def test_valid_construction(self, valid_task_data: dict[str, Any]) -> None:
        """A valid task with all required fields should construct successfully."""
        task = Task(**valid_task_data)
        assert task.id == "task-001"
        assert task.prompt == "Fix the failing test in the calculator module."
        assert task.language == "python"
        assert task.test_command == "pytest tests/test_calc.py -x"
        assert task.timeout == 300

    def test_required_fields(self) -> None:
        """Missing required fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            Task(id="t1", prompt="test", language="python", test_command="pytest")
        # No timeout should fail if it's required
        # Let's test missing truly required fields
        with pytest.raises(ValidationError):
            Task()  # type: ignore[call-arg]

    def test_rejects_negative_timeout(self, valid_task_data: dict[str, Any]) -> None:
        """Negative timeout should raise ValidationError with helpful message."""
        data = {**valid_task_data, "timeout": -10}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error = str(exc_info.value)
        assert "timeout" in error.lower()
        assert "greater than 0" in error.lower() or "positive" in error.lower()

    def test_rejects_zero_timeout(self, valid_task_data: dict[str, Any]) -> None:
        """Zero timeout should raise ValidationError."""
        data = {**valid_task_data, "timeout": 0}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error = str(exc_info.value)
        assert "timeout" in error.lower()

    def test_rejects_empty_prompt(self, valid_task_data: dict[str, Any]) -> None:
        """Empty prompt should raise ValidationError."""
        data = {**valid_task_data, "prompt": ""}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error = str(exc_info.value)
        assert "prompt" in error.lower()

    def test_rejects_empty_id(self, valid_task_data: dict[str, Any]) -> None:
        """Empty task ID should raise ValidationError."""
        data = {**valid_task_data, "id": ""}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error = str(exc_info.value)
        assert "id" in error.lower()

    def test_optional_fields_have_defaults(self, valid_task_data: dict[str, Any]) -> None:
        """Optional fields (setup_command, difficulty, constraints, dependencies) should have defaults."""
        minimal = {
            "id": "task-002",
            "prompt": "Do something.",
            "language": "python",
            "test_command": "pytest",
            "timeout": 60,
        }
        task = Task(**minimal)
        assert task.setup_command is None
        assert task.difficulty == Difficulty.EASY  # default
        assert task.constraints == []
        assert task.dependencies == []

    def test_json_round_trip(self, valid_task: Task) -> None:
        """Task should serialize to JSON and deserialize back identically."""
        json_str = valid_task.model_dump_json()
        restored = Task.model_validate_json(json_str)
        assert restored == valid_task

    def test_dict_round_trip(self, valid_task: Task) -> None:
        """Task should dump to dict and load back identically."""
        data = valid_task.model_dump()
        restored = Task.model_validate(data)
        assert restored == valid_task


# ===========================================================================
# TaskResult Model Tests
# ===========================================================================


class TestTaskResult:
    """VAL-FOUND-02: TaskResult constrains score to [0.0, 1.0] and status to closed enum."""

    def test_valid_success_result(self) -> None:
        """A valid success result should construct successfully."""
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.SUCCESS,
            score=1.0,
        )
        assert result.task_id == "task-001"
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0

    def test_valid_failure_result(self) -> None:
        """A valid failure result with partial score should construct successfully."""
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.FAILURE,
            score=0.5,
            error_category=ErrorCategory.TEST_FAILURE,
        )
        assert result.status == TaskStatus.FAILURE
        assert result.score == 0.5

    def test_score_at_boundaries(self) -> None:
        """Score at 0.0 and 1.0 should be valid."""
        result_zero = TaskResult(task_id="t1", status=TaskStatus.FAILURE, score=0.0)
        result_one = TaskResult(task_id="t2", status=TaskStatus.SUCCESS, score=1.0)
        assert result_zero.score == 0.0
        assert result_one.score == 1.0

    def test_rejects_score_above_one(self) -> None:
        """Score > 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskResult(task_id="t1", status=TaskStatus.SUCCESS, score=1.5)
        error = str(exc_info.value)
        assert "score" in error.lower()

    def test_rejects_score_below_zero(self) -> None:
        """Score < 0.0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskResult(task_id="t1", status=TaskStatus.FAILURE, score=-0.1)
        error = str(exc_info.value)
        assert "score" in error.lower()

    def test_rejects_invalid_status_string(self) -> None:
        """An arbitrary status string should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskResult(task_id="t1", status="almost", score=0.5)
        error = str(exc_info.value)
        assert "status" in error.lower()

    def test_all_status_values_valid(self) -> None:
        """All enum values should be accepted."""
        for status in TaskStatus:
            result = TaskResult(task_id="t1", status=status, score=0.5)
            assert result.status == status

    def test_optional_fields(self) -> None:
        """error_category, telemetry, and transcript should default to None."""
        result = TaskResult(task_id="t1", status=TaskStatus.SKIPPED, score=0.0)
        assert result.error_category is None
        assert result.telemetry is None
        assert result.transcript is None

    def test_json_round_trip(self) -> None:
        """TaskResult should serialize/deserialize losslessly through JSON."""
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.SUCCESS,
            score=0.85,
            error_category=ErrorCategory.UNKNOWN,
            transcript=[{"role": "user", "content": "fix it"}],
        )
        json_str = result.model_dump_json()
        restored = TaskResult.model_validate_json(json_str)
        assert restored == result


# ===========================================================================
# Manifest Model Tests
# ===========================================================================


class TestManifest:
    """VAL-FOUND-03: Manifest aggregates tasks with unique ID validation."""

    @pytest.fixture
    def tasks(self) -> list[Task]:
        return [
            Task(
                id=f"task-{i:03d}",
                prompt=f"Task {i}",
                language="python",
                test_command="pytest",
                timeout=60,
            )
            for i in range(3)
        ]

    def test_valid_manifest(self, tasks: list[Task]) -> None:
        """A manifest with unique task IDs should construct successfully."""
        manifest = Manifest(
            name="test-benchmark",
            version="1.0.0",
            tasks=tasks,
        )
        assert manifest.name == "test-benchmark"
        assert manifest.version == "1.0.0"
        assert len(manifest.tasks) == 3

    def test_timestamp_auto_generated(self, tasks: list[Task]) -> None:
        """Timestamp should be auto-generated if not provided."""
        manifest = Manifest(name="test", version="1.0.0", tasks=tasks)
        assert manifest.timestamp is not None
        assert isinstance(manifest.timestamp, datetime)

    def test_rejects_duplicate_task_ids(self, tasks: list[Task]) -> None:
        """Duplicate task IDs should raise ValidationError."""
        duplicate_tasks = [tasks[0], tasks[0]]  # same task twice
        with pytest.raises(ValidationError) as exc_info:
            Manifest(
                name="test-benchmark",
                version="1.0.0",
                tasks=duplicate_tasks,
            )
        error = str(exc_info.value)
        assert "duplicate" in error.lower() or "unique" in error.lower()

    def test_deterministic_json_serialization(self, tasks: list[Task]) -> None:
        """Manifest should serialize to deterministic JSON."""
        manifest = Manifest(
            name="test-benchmark",
            version="1.0.0",
            tasks=tasks,
        )
        json_str = manifest.model_dump_json()
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["name"] == "test-benchmark"
        assert len(data["tasks"]) == 3

    def test_manifest_json_round_trip(self, tasks: list[Task]) -> None:
        """Manifest should round-trip through JSON losslessly."""
        manifest = Manifest(
            name="test-benchmark",
            version="1.0.0",
            tasks=tasks,
        )
        json_str = manifest.model_dump_json()
        restored = Manifest.model_validate_json(json_str)
        assert restored.name == manifest.name
        assert restored.version == manifest.version
        assert [t.id for t in restored.tasks] == [t.id for t in manifest.tasks]

    def test_empty_tasks_list(self) -> None:
        """A manifest with empty task list should be valid."""
        manifest = Manifest(name="empty", version="1.0.0", tasks=[])
        assert len(manifest.tasks) == 0


# ===========================================================================
# HarnessConfig Model Tests
# ===========================================================================


class TestHarnessConfig:
    """VAL-FOUND-05: HarnessConfig models agent-environment-artifact binding."""

    def test_valid_config(self) -> None:
        """A valid HarnessConfig should construct successfully."""
        config = HarnessConfig(
            agent=AgentConfig(
                model="gpt-4",
                provider="openrouter",
                system_prompt="You are a helpful assistant.",
            ),
            environment=EnvironmentConfig(
                runner_type="local",
                docker_image="python:3.12",
            ),
            artifact_bindings=["tool-1", "skill-2"],
        )
        assert config.agent.model == "gpt-4"
        assert config.environment.runner_type == "local"
        assert config.artifact_bindings == ["tool-1", "skill-2"]

    def test_optional_artifact_bindings(self) -> None:
        """artifact_bindings should default to empty list."""
        config = HarnessConfig(
            agent=AgentConfig(model="gpt-4", provider="openrouter"),
            environment=EnvironmentConfig(runner_type="local"),
        )
        assert config.artifact_bindings == []

    def test_sensible_defaults(self) -> None:
        """AgentConfig and EnvironmentConfig should have sensible defaults."""
        agent = AgentConfig(model="gpt-4", provider="openrouter")
        assert agent.system_prompt is None

        env = EnvironmentConfig(runner_type="local")
        assert env.docker_image is None

    def test_json_round_trip(self) -> None:
        """HarnessConfig should round-trip through JSON."""
        config = HarnessConfig(
            agent=AgentConfig(
                model="gpt-4",
                provider="openrouter",
                system_prompt="Hello",
            ),
            environment=EnvironmentConfig(
                runner_type="docker",
                docker_image="python:3.12",
            ),
            artifact_bindings=["tool-1"],
        )
        json_str = config.model_dump_json()
        restored = HarnessConfig.model_validate_json(json_str)
        assert restored == config


# ===========================================================================
# ExecutionOutput Model Tests
# ===========================================================================


class TestExecutionOutput:
    """ExecutionOutput should capture stdout, stderr, exit_code, and timed_out."""

    def test_valid_output(self) -> None:
        """A valid ExecutionOutput should construct successfully."""
        output = ExecutionOutput(
            stdout="hello world",
            stderr="",
            exit_code=0,
            timed_out=False,
        )
        assert output.stdout == "hello world"
        assert output.exit_code == 0
        assert output.timed_out is False

    def test_timeout_output(self) -> None:
        """A timed-out execution should capture partial output."""
        output = ExecutionOutput(
            stdout="partial output",
            stderr="killed after timeout",
            exit_code=-1,
            timed_out=True,
        )
        assert output.timed_out is True
        assert output.exit_code == -1

    def test_json_round_trip(self) -> None:
        """ExecutionOutput should round-trip through JSON."""
        output = ExecutionOutput(
            stdout="some output",
            stderr="some errors",
            exit_code=1,
            timed_out=False,
        )
        json_str = output.model_dump_json()
        restored = ExecutionOutput.model_validate_json(json_str)
        assert restored == output


# ===========================================================================
# VAL-FOUND-12: Helpful Error Messages
# ===========================================================================


class TestErrorMessages:
    """VAL-FOUND-12: Validation errors include field name, expected constraint, and actual value."""

    def test_task_error_includes_field_name(self, valid_task_data: dict[str, Any]) -> None:
        """Error message should include the field name."""
        data = {**valid_task_data, "timeout": -5}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error_str = str(exc_info.value)
        assert "timeout" in error_str.lower()

    def test_task_error_includes_constraint_hint(self, valid_task_data: dict[str, Any]) -> None:
        """Error message should include constraint hint (e.g., greater than)."""
        data = {**valid_task_data, "timeout": -5}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error_str = str(exc_info.value)
        # Pydantic v2 should mention the constraint
        assert "greater" in error_str.lower() or "positive" in error_str.lower()

    def test_task_error_includes_actual_value(self, valid_task_data: dict[str, Any]) -> None:
        """Error message should include the actual value that was provided."""
        data = {**valid_task_data, "timeout": -5}
        with pytest.raises(ValidationError) as exc_info:
            Task(**data)
        error_str = str(exc_info.value)
        assert "-5" in error_str

    def test_result_score_error_includes_value(self) -> None:
        """Score error should show the actual value."""
        with pytest.raises(ValidationError) as exc_info:
            TaskResult(task_id="t1", status=TaskStatus.SUCCESS, score=2.0)
        error_str = str(exc_info.value)
        assert "score" in error_str.lower()
        assert "2" in error_str

    def test_result_status_error_includes_value(self) -> None:
        """Status error should show the actual value."""
        with pytest.raises(ValidationError) as exc_info:
            TaskResult(task_id="t1", status="almost", score=0.5)
        error_str = str(exc_info.value)
        assert "status" in error_str.lower()
        assert "almost" in error_str

    def test_manifest_duplicate_error_includes_task_id(self) -> None:
        """Duplicate ID error should include the problematic task ID."""
        t = Task(id="dup-1", prompt="p", language="python", test_command="pytest", timeout=60)
        with pytest.raises(ValidationError) as exc_info:
            Manifest(name="test", version="1.0.0", tasks=[t, t])
        error_str = str(exc_info.value)
        assert "dup-1" in error_str


# ===========================================================================
# Enum / Constant Tests
# ===========================================================================


class TestEnums:
    """Verify enums have expected values."""

    def test_task_status_values(self) -> None:
        assert TaskStatus.SUCCESS.value == "SUCCESS"
        assert TaskStatus.FAILURE.value == "FAILURE"
        assert TaskStatus.ERROR.value == "ERROR"
        assert TaskStatus.TIMEOUT.value == "TIMEOUT"
        assert TaskStatus.SKIPPED.value == "SKIPPED"

    def test_difficulty_values(self) -> None:
        assert Difficulty.EASY.value == "EASY"
        assert Difficulty.MEDIUM.value == "MEDIUM"
        assert Difficulty.HARD.value == "HARD"

    def test_error_category_values(self) -> None:
        assert ErrorCategory.TEST_FAILURE.value == "TEST_FAILURE"
        assert ErrorCategory.SYNTAX_ERROR.value == "SYNTAX_ERROR"
        assert ErrorCategory.ENVIRONMENT_ERROR.value == "ENVIRONMENT_ERROR"
        assert ErrorCategory.NETWORK_ERROR.value == "NETWORK_ERROR"
        assert ErrorCategory.API_ERROR.value == "API_ERROR"
        assert ErrorCategory.RATE_LIMIT.value == "RATE_LIMIT"
        assert ErrorCategory.MAX_TURNS_EXCEEDED.value == "MAX_TURNS_EXCEEDED"
        assert ErrorCategory.UNKNOWN.value == "UNKNOWN"


# ===========================================================================
# VAL-FOUND-11: Schemas integrate for round-trip through JSON
# ===========================================================================


class TestAllModelsJsonRoundTrip:
    """All models should round-trip through JSON serialization losslessly."""

    def test_task_round_trip(self, valid_task: Task) -> None:
        json_str = valid_task.model_dump_json()
        restored = Task.model_validate_json(json_str)
        assert restored == valid_task

    def test_task_result_round_trip(self) -> None:
        result = TaskResult(
            task_id="t1",
            status=TaskStatus.ERROR,
            score=0.0,
            error_category=ErrorCategory.SYNTAX_ERROR,
        )
        json_str = result.model_dump_json()
        restored = TaskResult.model_validate_json(json_str)
        assert restored == result

    def test_execution_output_round_trip(self) -> None:
        output = ExecutionOutput(stdout="out", stderr="err", exit_code=1, timed_out=False)
        json_str = output.model_dump_json()
        restored = ExecutionOutput.model_validate_json(json_str)
        assert restored == output

    def test_manifest_round_trip(self) -> None:
        tasks = [
            Task(id=f"t{i}", prompt=f"p{i}", language="python", test_command="pytest", timeout=60)
            for i in range(2)
        ]
        manifest = Manifest(name="bench", version="1.0.0", tasks=tasks)
        json_str = manifest.model_dump_json()
        restored = Manifest.model_validate_json(json_str)
        assert restored.name == manifest.name
        assert restored.version == manifest.version
        assert [t.id for t in restored.tasks] == [t.id for t in manifest.tasks]

    def test_harness_config_round_trip(self) -> None:
        config = HarnessConfig(
            agent=AgentConfig(model="gpt-4", provider="openrouter"),
            environment=EnvironmentConfig(runner_type="local"),
        )
        json_str = config.model_dump_json()
        restored = HarnessConfig.model_validate_json(json_str)
        assert restored == config
