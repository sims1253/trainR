"""Tests for canonical schema v1 models."""

import pytest
from pydantic import ValidationError

from bench.schema.v1 import (
    DifficultyV1,
    ErrorCategoryV1,
    JudgeConfigV1,
    JudgeModeV1,
    ManifestV1,
    ModelConfigV1,
    ProfileTypeV1,
    ProfileV1,
    ResultV1,
    SkillConfigV1,
    TaskTestsV1,
    TaskTypeV1,
    TaskV1,
    TokenUsageV1,
    VotingStrategyV1,
    adapt_from_legacy_result,
    adapt_from_legacy_task,
    load_json_schema,
    validate_result,
    validate_task,
)
from bench.schema.v1.results import CaseResultV1


class TestTaskV1:
    """Tests for TaskV1 schema."""

    def test_create_swe_task(self) -> None:
        """Test creating an SWE task."""
        task = TaskV1(
            task_id="test-swe-001",
            task_type=TaskTypeV1.SWE,
            instruction="Fix the bug in the filter function",
        )
        assert task.task_id == "test-swe-001"
        assert task.task_type == TaskTypeV1.SWE
        assert task.schema_version == "1.0"

    def test_create_test_gen_task(self) -> None:
        """Test creating a test generation task."""
        task = TaskV1(
            task_id="test-gen-001",
            task_type=TaskTypeV1.TEST_GEN,
            instruction="Write tests for the sum function",
            source_package="dplyr",
        )
        assert task.task_type == TaskTypeV1.TEST_GEN
        assert task.source_package == "dplyr"

    def test_create_skill_task(self) -> None:
        """Test creating a skill evaluation task."""
        task = TaskV1(
            task_id="test-skill-001",
            task_type=TaskTypeV1.SKILL,
            instruction="Evaluate the skill",
        )
        assert task.task_type == TaskTypeV1.SKILL

    def test_task_with_tests(self) -> None:
        """Test task with test specifications."""
        task = TaskV1(
            task_id="test-001",
            task_type=TaskTypeV1.SWE,
            instruction="Fix the bug",
            tests=TaskTestsV1(
                fail_to_pass=["test_one", "test_two"],
                pass_to_pass=["test_three"],
            ),
        )
        assert len(task.tests.fail_to_pass) == 2
        assert len(task.tests.pass_to_pass) == 1

    def test_task_difficulty_validation(self) -> None:
        """Test difficulty levels."""
        for difficulty in [DifficultyV1.EASY, DifficultyV1.MEDIUM, DifficultyV1.HARD]:
            task = TaskV1(
                task_id="test-001",
                task_type=TaskTypeV1.SWE,
                instruction="Test",
                difficulty=difficulty,
            )
            assert task.difficulty == difficulty

    def test_task_quality_score_range(self) -> None:
        """Test quality score validation."""
        # Valid scores
        for score in [0.0, 5.0, 10.0]:
            task = TaskV1(
                task_id="test-001",
                task_type=TaskTypeV1.SWE,
                instruction="Test",
                quality_score=score,
            )
            assert task.quality_score == score

        # Invalid scores should raise
        with pytest.raises(ValidationError):
            TaskV1(
                task_id="test-001",
                task_type=TaskTypeV1.SWE,
                instruction="Test",
                quality_score=11.0,
            )

    def test_validate_task_function(self) -> None:
        """Test validate_task helper."""
        data = {
            "task_id": "test-001",
            "task_type": "swe",
            "instruction": "Fix the bug",
        }
        task = validate_task(data)
        assert task.task_id == "test-001"
        assert task.task_type == TaskTypeV1.SWE

    def test_from_legacy_mined_task(self) -> None:
        """Test conversion from legacy MinedTask format."""
        legacy = {
            "task_id": "mined-001",
            "source_package": "dplyr",
            "source": {
                "type": "github_pr",
                "repo": "tidyverse/dplyr",
                "pr_number": 123,
            },
            "task": {
                "instruction": "Fix the filter bug",
                "context": "Some context",
            },
            "metadata": {
                "task_type": "bug_fix",
                "difficulty": "medium",
                "quality_score": 8,
            },
            "tests": {
                "fail_to_pass": ["test1"],
                "pass_to_pass": ["test2"],
            },
        }
        task = TaskV1.from_legacy_mined_task(legacy)
        assert task.task_id == "mined-001"
        assert task.task_type == TaskTypeV1.SWE
        assert task.difficulty == DifficultyV1.MEDIUM
        assert task.quality_score == 8.0

    def test_from_legacy_testing_task(self) -> None:
        """Test conversion from legacy TestingTask format."""
        legacy = {
            "task_id": "test-gen-001",
            "source_package": "readr",
            "source_file": "R/read.R",
            "difficulty": "easy",
            "instruction": "Write tests for read_csv",
            "context": "read_csv function code",
            "split": "dev",
            "quality_score": 7.5,
        }
        task = TaskV1.from_legacy_testing_task(legacy)
        assert task.task_id == "test-gen-001"
        assert task.task_type == TaskTypeV1.TEST_GEN
        assert task.difficulty == DifficultyV1.EASY


class TestProfileV1:
    """Tests for ProfileV1 schema."""

    def test_create_basic_profile(self) -> None:
        """Test creating a basic profile."""
        profile = ProfileV1(
            profile_id="test-profile",
            profile_type=ProfileTypeV1.AGENT,
        )
        assert profile.profile_id == "test-profile"
        assert profile.profile_type == ProfileTypeV1.AGENT

    def test_profile_with_model(self) -> None:
        """Test profile with model configuration."""
        profile = ProfileV1(
            profile_id="test-profile",
            model=ModelConfigV1(name="glm-5"),
        )
        assert profile.model is not None
        assert profile.model.name == "glm-5"

    def test_profile_with_skill(self) -> None:
        """Test profile with skill configuration."""
        profile = ProfileV1(
            profile_id="test-profile",
            skill=SkillConfigV1(path="skills/testing.md"),
        )
        assert profile.skill.path == "skills/testing.md"
        assert profile.skill.is_active()

    def test_profile_no_skill_mode(self) -> None:
        """Test profile with no_skill mode."""
        profile = ProfileV1(
            profile_id="test-profile",
            skill=SkillConfigV1(no_skill=True),
        )
        assert profile.skill.no_skill
        assert not profile.skill.is_active()

    def test_profile_with_judge(self) -> None:
        """Test profile with judge configuration."""
        profile = ProfileV1(
            profile_id="test-profile",
            judge=JudgeConfigV1(
                mode=JudgeModeV1.ENSEMBLE,
                ensemble_judges=["glm-5", "gpt-4"],
                voting=VotingStrategyV1.MAJORITY,
            ),
        )
        assert profile.judge is not None
        assert profile.judge.mode == JudgeModeV1.ENSEMBLE
        assert len(profile.judge.ensemble_judges) == 2

    def test_from_evaluation_config(self) -> None:
        """Test conversion from legacy EvaluationConfig."""
        legacy = {
            "model": {
                "task": "glm-5",
                "reflection": "glm-5",
            },
            "skill": {
                "path": "skills/testing.md",
                "no_skill": False,
            },
            "execution": {
                "timeout": 600,
                "docker_image": "posit-gskill-eval:latest",
                "repeats": 1,
            },
            "workers": {
                "count": 5,
            },
        }
        profile = ProfileV1.from_evaluation_config(legacy, "eval-profile")
        assert profile.profile_id == "eval-profile"
        assert profile.model is not None
        assert profile.model.name == "glm-5"


class TestResultV1:
    """Tests for ResultV1 schema."""

    def test_create_basic_result(self) -> None:
        """Test creating a basic result."""
        result = ResultV1(
            task_id="test-001",
            model="glm-5",
            passed=True,
            score=0.95,
        )
        assert result.task_id == "test-001"
        assert result.model == "glm-5"
        assert result.passed
        assert result.score == 0.95

    def test_result_with_error(self) -> None:
        """Test result with error information."""
        result = ResultV1(
            task_id="test-001",
            model="glm-5",
            passed=False,
            score=0.0,
            error_category=ErrorCategoryV1.TEST_FAILURE,
            error_message="Test assertion failed",
        )
        assert result.error_category == ErrorCategoryV1.TEST_FAILURE
        assert result.error_message == "Test assertion failed"

    def test_result_with_tests(self) -> None:
        """Test result with test results."""
        result = ResultV1(
            task_id="test-001",
            model="glm-5",
            passed=True,
            score=1.0,
            test_results=[
                CaseResultV1(name="test_one", passed=True),
                CaseResultV1(name="test_two", passed=True),
                CaseResultV1(name="test_three", passed=False, message="Failed"),
            ],
        )
        assert result.test_count == 3
        assert result.passed_count == 2
        assert result.failed_count == 1
        assert result.pass_rate == pytest.approx(2 / 3)

    def test_result_with_token_usage(self) -> None:
        """Test result with token usage."""
        result = ResultV1(
            task_id="test-001",
            model="glm-5",
            passed=True,
            token_usage=TokenUsageV1(prompt=100, completion=50, total=150),
        )
        assert result.token_usage.prompt == 100
        assert result.token_usage.total == 150

    def test_from_legacy_benchmark_result(self) -> None:
        """Test conversion from legacy BenchmarkResult."""
        legacy = {
            "task_id": "test-001",
            "model": "glm-5",
            "passed": True,
            "score": 0.9,
            "latency_s": 2.5,
            "error_category": None,
            "token_usage": {"prompt": 100, "completion": 50},
            "test_results": [
                {"name": "test1", "passed": True, "message": ""},
            ],
        }
        result = ResultV1.from_legacy_benchmark_result(legacy)
        assert result.task_id == "test-001"
        assert result.model == "glm-5"
        assert result.passed
        assert result.latency_s == 2.5

    def test_validate_result_function(self) -> None:
        """Test validate_result helper."""
        data = {
            "task_id": "test-001",
            "model": "glm-5",
            "passed": True,
            "score": 0.95,
        }
        result = validate_result(data)
        assert result.task_id == "test-001"


class TestManifestV1:
    """Tests for ManifestV1 schema."""

    def test_create_basic_manifest(self) -> None:
        """Test creating a basic manifest."""
        manifest = ManifestV1(
            run_id="run-001",
            models=["glm-5", "gpt-4"],
            task_count=10,
        )
        assert manifest.run_id == "run-001"
        assert len(manifest.models) == 2
        assert manifest.task_count == 10

    def test_manifest_add_result(self) -> None:
        """Test adding results to manifest."""
        manifest = ManifestV1(
            run_id="run-001",
            models=["glm-5"],
            task_count=2,
        )

        result1 = ResultV1(task_id="task-1", model="glm-5", passed=True, score=1.0)
        result2 = ResultV1(task_id="task-2", model="glm-5", passed=False, score=0.0)

        manifest.add_result(result1)
        manifest.add_result(result2)

        assert len(manifest.results) == 2
        assert manifest.summary.completed == 2
        assert manifest.summary.passed == 1

    def test_manifest_model_summaries(self) -> None:
        """Test per-model summaries."""
        manifest = ManifestV1(
            run_id="run-001",
            models=["glm-5", "gpt-4"],
            task_count=2,
        )

        manifest.add_result(ResultV1(task_id="task-1", model="glm-5", passed=True, score=1.0))
        manifest.add_result(ResultV1(task_id="task-2", model="gpt-4", passed=False, score=0.5))

        assert len(manifest.model_summaries) == 2

        glm_summary = next(s for s in manifest.model_summaries if s.model == "glm-5")
        assert glm_summary.passed == 1
        assert glm_summary.total == 1

    def test_manifest_finalize(self) -> None:
        """Test manifest finalization."""
        manifest = ManifestV1(
            run_id="run-001",
            models=["glm-5"],
            task_count=1,
        )

        manifest.add_result(ResultV1(task_id="task-1", model="glm-5", passed=True, score=1.0))
        manifest.finalize()

        assert manifest.end_timestamp is not None
        assert manifest.duration_s is not None

    def test_create_new_with_env_detection(self) -> None:
        """Test creating manifest with environment detection."""
        manifest = ManifestV1.create_new(
            run_id="run-001",
            models=["glm-5"],
            task_count=10,
        )

        assert manifest.run_id == "run-001"
        assert manifest.environment_fingerprint is not None
        assert manifest.environment_fingerprint.python_version != ""


class TestJSONSchemas:
    """Tests for JSON schema functionality."""

    def test_load_task_schema(self) -> None:
        """Test loading task JSON schema."""
        schema = load_json_schema("task")
        assert schema["title"] == "TaskV1"
        assert "task_id" in schema["properties"]
        assert "task_type" in schema["properties"]

    def test_load_profile_schema(self) -> None:
        """Test loading profile JSON schema."""
        schema = load_json_schema("profile")
        assert schema["title"] == "ProfileV1"
        assert "profile_id" in schema["properties"]

    def test_load_result_schema(self) -> None:
        """Test loading result JSON schema."""
        schema = load_json_schema("result")
        assert schema["title"] == "ResultV1"
        assert "task_id" in schema["properties"]
        assert "passed" in schema["properties"]

    def test_load_manifest_schema(self) -> None:
        """Test loading manifest JSON schema."""
        schema = load_json_schema("manifest")
        assert schema["title"] == "ManifestV1"
        assert "run_id" in schema["properties"]

    def test_invalid_schema_type(self) -> None:
        """Test loading invalid schema type."""
        with pytest.raises(ValueError):
            load_json_schema("invalid")


class TestAdapters:
    """Tests for legacy format adapters."""

    def test_adapt_legacy_result_benchmark(self) -> None:
        """Test adapting legacy benchmark result."""
        legacy = {
            "task_id": "test-001",
            "model": "glm-5",
            "passed": True,
            "score": 0.9,
        }
        result = adapt_from_legacy_result(legacy, source="benchmark")
        assert result.task_id == "test-001"

    def test_adapt_legacy_result_evaluation(self) -> None:
        """Test adapting legacy evaluation result."""
        legacy = {
            "task_id": "test-001",
            "success": True,
            "score": 0.9,
        }
        result = adapt_from_legacy_result(legacy, source="evaluation")
        assert result.task_id == "test-001"
        assert result.passed

    def test_adapt_legacy_task_mined(self) -> None:
        """Test adapting legacy mined task."""
        legacy = {
            "task_id": "mined-001",
            "task": {"instruction": "Test"},
            "metadata": {"task_type": "bug_fix", "difficulty": "medium"},
        }
        task = adapt_from_legacy_task(legacy, source="mined")
        assert task.task_id == "mined-001"
        assert task.task_type == TaskTypeV1.SWE

    def test_adapt_legacy_task_testing(self) -> None:
        """Test adapting legacy testing task."""
        legacy = {
            "task_id": "test-gen-001",
            "difficulty": "easy",
            "instruction": "Write tests",
        }
        task = adapt_from_legacy_task(legacy, source="testing")
        assert task.task_id == "test-gen-001"
        assert task.task_type == TaskTypeV1.TEST_GEN
