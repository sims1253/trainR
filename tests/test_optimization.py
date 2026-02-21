"""Integration tests for GEPA optimization adapter."""

import os
from unittest.mock import MagicMock, patch

import pytest

from optimization.adapter import SkillEvaluator, optimize_skill
from task_generator.models import Difficulty, TestingTask, TestPattern


def _make_task(
    task_id: str = "task-test001", function_name: str | None = "cli_alert",
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

    def test_init_without_api_key_param(self):
        """SkillEvaluator no longer requires api_key."""
        evaluator = SkillEvaluator(docker_image="test:latest", timeout=60)
        assert evaluator.sandbox is not None

    def test_handles_missing_function_name(self):
        """Tasks without function_name should not crash."""
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

    def test_handles_task_with_function_name(self):
        """Tasks with function_name should use it."""
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

    def test_side_info_on_failure(self):
        """Failed evaluations should include error details in side_info."""
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

    def test_raises_without_api_key(self):
        """Should raise ValueError when Z_AI_API_KEY is not set."""
        env = os.environ.copy()
        env.pop("Z_AI_API_KEY", None)
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="Z_AI_API_KEY"),
        ):
            optimize_skill(
                seed_skill="# test",
                train_tasks=[_make_task()],
                val_tasks=[_make_task()],
            )

    def test_reads_reflection_lm_from_env(self):
        """Should read reflection model from LLM_MODEL_REFLECTION env."""
        with (
            patch.dict(
                os.environ, {"Z_AI_API_KEY": "test", "LLM_MODEL_REFLECTION": "openai/custom-model"}
            ),
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

            # Verify the config passed to optimize_anything uses the env model
            call_kwargs = mock_oa.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config.reflection.reflection_lm == "openai/custom-model"
