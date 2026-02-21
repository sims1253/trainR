"""GEPA adapter for skill optimization."""

import logging
import os
from pathlib import Path
from typing import Any

import gepa.optimize_anything as oa
from gepa.core.result import GEPAResult
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig

from evaluation import EvaluationSandbox, TestRunnerConfig
from task_generator.models import TestingTask

logger = logging.getLogger(__name__)


class SkillEvaluator:
    """Evaluator for R testing skills using Docker + cc-mirror.

    This is the main evaluator that GEPA calls repeatedly during optimization.
    """

    def __init__(
        self,
        docker_image: str = "posit-gskill-eval:latest",
        timeout: int = 600,
        packages_dir: Path = Path("packages"),
    ) -> None:
        """Initialize the skill evaluator.

        Args:
            docker_image: Docker image to use for evaluation.
            timeout: Timeout per evaluation in seconds.
            packages_dir: Directory containing R packages.
        """
        self.packages_dir = packages_dir

        config = TestRunnerConfig(
            docker_image=docker_image,
            timeout=timeout,
        )
        self.sandbox = EvaluationSandbox(runner_config=config)

    def __call__(
        self,
        candidate: str,
        example: TestingTask,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a skill candidate on a single task.

        This is the signature GEPA expects for generalization mode.

        Args:
            candidate: The skill markdown text to evaluate.
            example: A TestingTask from the dataset.

        Returns:
            Tuple of (score, side_info) where:
            - score: 1.0 for pass, 0.0 for fail
            - side_info: Dict with diagnostic information for GEPA reflection
        """
        # Log the task being evaluated
        func_name = getattr(example, "function_name", None) or "unknown"
        oa.log(f"Evaluating skill on task: {example.task_id}")
        oa.log(f"Package: {example.source_package}")
        oa.log(f"Function: {func_name}")
        oa.log(f"Difficulty: {example.difficulty}")

        # Determine package path
        package_dir = self.packages_dir / example.source_package

        # Run evaluation
        result = self.sandbox.evaluate_task(
            task=example,
            skill_prompt=candidate,
            package_dir=package_dir,
        )

        # Build side info for GEPA reflection
        side_info: dict[str, Any] = {
            "task_id": example.task_id,
            "function_name": func_name,
            "difficulty": str(example.difficulty),
            "test_type": getattr(example, "test_type", "unknown"),
            "success": result.success,
        }

        # Add error information if failed
        if not result.success:
            if result.failure_category:
                side_info["failure_category"] = str(result.failure_category)
            if result.error_message:
                side_info["error_message"] = result.error_message

            # Add test failure details
            failed_tests = [
                {"name": t.name, "message": t.message} for t in result.test_results if not t.passed
            ]
            if failed_tests:
                side_info["failed_tests"] = failed_tests

        # Add generated code for debugging
        if result.generated_code:
            # Truncate if too long
            code = result.generated_code
            if len(code) > 2000:
                code = code[:2000] + "\n... [truncated]"
            side_info["generated_code"] = code

        # Log result
        oa.log(f"Result: {'PASS' if result.success else 'FAIL'}")
        if not result.success and result.failure_category:
            oa.log(f"Failure category: {result.failure_category}")

        # Return score (1.0 for pass, 0.0 for fail)
        score = 1.0 if result.success else 0.0
        return score, side_info


def optimize_skill(
    seed_skill: str,
    train_tasks: list[TestingTask],
    val_tasks: list[TestingTask],
    docker_image: str = "posit-gskill-eval:latest",
    timeout: int = 600,
    max_metric_calls: int = 100,
    reflection_lm: str | None = None,
    run_dir: str | None = None,
) -> GEPAResult:
    """Optimize a testing skill using GEPA.

    Args:
        seed_skill: Initial skill markdown to start optimization.
        train_tasks: Tasks to train on.
        val_tasks: Tasks to validate generalization.
        docker_image: Docker image for evaluation.
        timeout: Timeout per evaluation in seconds.
        max_metric_calls: Maximum number of evaluations.
        reflection_lm: LiteLLM model string for reflection (reads from
            LLM_MODEL_REFLECTION env, falls back to openai/glm-5).
        run_dir: Directory to save optimization state.

    Returns:
        GEPAResult with best_candidate and optimization history.
    """
    if not os.environ.get("Z_AI_API_KEY"):
        raise ValueError("Z_AI_API_KEY not set in environment.")

    if reflection_lm is None:
        reflection_lm = os.environ.get("LLM_MODEL_REFLECTION", "openai/glm-5")

    # Create evaluator
    evaluator = SkillEvaluator(
        docker_image=docker_image,
        timeout=timeout,
    )

    # Configure GEPA
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_metric_calls,
            run_dir=run_dir,
            parallel=False,  # Docker evaluation should be sequential
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,  # Evaluate on 3 tasks per reflection
        ),
    )

    # Define optimization objective
    objective = """Generate a skill that helps coding agents write effective testthat tests for R packages.

The skill should guide agents to:
1. Use testthat 3+ patterns (test_that, describe/it, local_edition)
2. Choose appropriate expectations (expect_equal, expect_snapshot, etc.)
3. Handle edge cases and error conditions
4. Follow tidyverse testing conventions

The skill will be used by a Claude Code agent that can explore the R package source code."""

    background = """Context: Testing R packages, specifically packages from the tidyverse ecosystem like cli.

Key patterns to encourage:
- Use describe/it blocks for organized tests
- Use local_edition(3) for testthat 3.x features
- Use expect_snapshot() for complex outputs
- Use local_mocked_bindings() for mocking
- Test both success and error cases

Common pitfalls to avoid:
- Don't assume functions exist without checking
- Don't forget to load required packages
- Don't write overly complex tests
- Don't ignore test file organization"""

    # Run optimization
    logger.info(
        f"Starting GEPA optimization with {len(train_tasks)} train, {len(val_tasks)} val tasks"
    )
    logger.info(f"Max metric calls: {max_metric_calls}")

    result = oa.optimize_anything(
        seed_candidate=seed_skill,
        evaluator=evaluator,
        dataset=train_tasks,
        valset=val_tasks,
        objective=objective,
        background=background,
        config=config,
    )

    return result
