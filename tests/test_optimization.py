"""Integration tests for GEPA optimization adapter."""

import os
from unittest.mock import MagicMock, patch

import pytest

from evaluation.models import FailureCategory
from optimization.adapter import SkillEvaluator, optimize_skill
from task_generator.models import Difficulty, TestingTask, TestPattern


def _make_task(
    task_id: str = "task-test001",
    function_name: str | None = "cli_alert",
) -> TestingTask:
    """Create a minimal TestingTask for testing."""
    return TestingTask(
        task_id=task_id,
        source_package="cli",
        source_file="R/cli.R",
        difficulty=Difficulty.EASY,
        instruction="Write tests for cli_alert()",
        context="Package: cli\nFunction: cli_alert",
        reference_test='test_that("cli_alert works", { expect_no_error(cli_alert("hi")) })',
        test_type="unit",
        patterns=[TestPattern.TEST_THAT, TestPattern.EXPECT_EQUAL],
        dependencies=["testthat", "cli"],
        split="train",
        function_name=function_name,
        quality_score=0.7,
    )


class TestSkillEvaluator:
    """Tests for SkillEvaluator."""

    @patch("evaluation.sandbox.DockerPiRunner")
    def test_init_without_api_key_param(self, mock_runner):
        """SkillEvaluator no longer requires api_key."""
        mock_runner.return_value = MagicMock()
        evaluator = SkillEvaluator(docker_image="test:latest", timeout=60)
        assert evaluator.sandbox is not None

    @patch("evaluation.sandbox.DockerPiRunner")
    def test_handles_missing_function_name(self, mock_runner):
        """Tasks without function_name should not crash."""
        mock_runner.return_value = MagicMock()
        evaluator = SkillEvaluator()
        task = _make_task(function_name=None)

        # Mock the sandbox to avoid Docker calls
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.failure_category = None
        mock_result.error_message = "mock error"
        mock_result.test_results = []
        mock_result.generated_code = ""

        with patch.object(evaluator.sandbox, "evaluate_task", return_value=mock_result):
            score, info = evaluator(candidate="# test skill", example=task)

        assert score == 0.0
        assert info["function_name"] == "unknown"

    @patch("evaluation.sandbox.DockerPiRunner")
    def test_handles_task_with_function_name(self, mock_runner):
        """Tasks with function_name should use it."""
        mock_runner.return_value = MagicMock()
        evaluator = SkillEvaluator()
        task = _make_task(function_name="cli_alert")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.failure_category = None
        mock_result.error_message = None
        mock_result.test_results = []
        mock_result.generated_code = "# test code"

        with patch.object(evaluator.sandbox, "evaluate_task", return_value=mock_result):
            score, info = evaluator(candidate="# test skill", example=task)

        assert score == 1.0
        assert info["function_name"] == "cli_alert"

    @patch("evaluation.sandbox.DockerPiRunner")
    def test_side_info_on_failure(self, mock_runner):
        """Failed evaluations should include error details in side_info."""
        mock_runner.return_value = MagicMock()
        evaluator = SkillEvaluator()
        task = _make_task()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.failure_category = "SYNTAX_ERROR"
        mock_result.error_message = "unexpected token"
        mock_result.test_results = [MagicMock(name="test1", passed=False, message="syntax error")]
        mock_result.generated_code = "bad code"

        with patch.object(evaluator.sandbox, "evaluate_task", return_value=mock_result):
            score, info = evaluator(candidate="# skill", example=task)

        assert score == 0.0
        assert info["failure_category"] == "SYNTAX_ERROR"
        assert info["error_message"] == "unexpected token"


class TestOptimizeSkill:
    """Tests for optimize_skill function."""

    @patch("optimization.adapter.oa.optimize_anything")
    @patch("evaluation.sandbox.DockerPiRunner")
    def test_raises_without_api_key(self, mock_runner, mock_oa):
        """Should raise ValueError when required API key is not set.

        Note: The key required depends on the reflection model from config.
        This test verifies the general behavior of requiring API keys.
        """
        mock_runner.return_value = MagicMock()
        env = os.environ.copy()
        # Clear all known API keys
        for key in [
            "OPENROUTER_API_KEY",
            "Z_AI_API_KEY",
            "ZAI_API_KEY",
            "OPENCODE_API_KEY",
            "OPENCODE_API_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ]:
            env.pop(key, None)
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="_API_KEY"),
        ):
            optimize_skill(
                seed_skill="# test",
                train_tasks=[_make_task()],
                val_tasks=[_make_task()],
            )

    @patch("evaluation.sandbox.DockerPiRunner")
    def test_uses_config_reflection_model(self, mock_runner):
        """Should read reflection model from config file by default."""
        mock_runner.return_value = MagicMock()
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
            patch("optimization.adapter.oa.optimize_anything") as mock_oa,
        ):
            mock_result = MagicMock()
            mock_oa.return_value = mock_result

            optimize_skill(
                seed_skill="# test",
                train_tasks=[_make_task()],
                val_tasks=[_make_task()],
                max_metric_calls=1,
            )

            # Verify optimize_anything was called
            assert mock_oa.called

    @patch("evaluation.sandbox.DockerPiRunner")
    def test_reflection_lm_override(self, mock_runner):
        """Should use explicit reflection_lm parameter over config."""
        mock_runner.return_value = MagicMock()
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
            patch("optimization.adapter.oa.optimize_anything") as mock_oa,
        ):
            mock_result = MagicMock()
            mock_oa.return_value = mock_result

            optimize_skill(
                seed_skill="# test",
                train_tasks=[_make_task()],
                val_tasks=[_make_task()],
                max_metric_calls=1,
                reflection_lm="custom-model-override",
            )

            # Verify the config passed to optimize_anything uses the override
            call_kwargs = mock_oa.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config.reflection.reflection_lm == "custom-model-override"


class TestFailureClassification:
    """Tests for error classification in evaluation."""

    def test_environment_error_category_exists(self):
        """Test ENVIRONMENT_ERROR is a valid FailureCategory."""
        assert FailureCategory.ENVIRONMENT_ERROR.value == "ENVIRONMENT_ERROR"

    def test_config_error_category_exists(self):
        """Test CONFIG_ERROR is a valid FailureCategory."""
        assert FailureCategory.CONFIG_ERROR.value == "CONFIG_ERROR"

    def test_package_not_found_category_exists(self):
        """Test PACKAGE_NOT_FOUND is a valid FailureCategory."""
        assert FailureCategory.PACKAGE_NOT_FOUND.value == "PACKAGE_NOT_FOUND"

    def test_timeout_category_exists(self):
        """Test TIMEOUT is a valid FailureCategory."""
        assert FailureCategory.TIMEOUT.value == "TIMEOUT"

    def test_test_failure_category_exists(self):
        """Test TEST_FAILURE is a valid FailureCategory."""
        assert FailureCategory.TEST_FAILURE.value == "TEST_FAILURE"
