"""GEPA adapter for skill optimization."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import gepa.optimize_anything as oa
from gepa.core.result import GEPAResult
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig

from config import get_llm_config
from evaluation import DockerPiRunnerConfig, EvaluationSandbox
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
        model: str = "glm-4.5",
    ) -> None:
        """Initialize the skill evaluator.

        Args:
            docker_image: Docker image to use for evaluation.
            timeout: Timeout per evaluation in seconds.
            packages_dir: Directory containing R packages.
            model: Model to use for task execution.
        """
        self.packages_dir = packages_dir
        self.model = model

        config = DockerPiRunnerConfig(
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
            model=self.model,
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


class MultiModelSkillEvaluator:
    """Evaluator that tests skill on multiple models for robustness.

    Evaluates the skill candidate on ALL models and aggregates scores.
    This ensures the optimized skill generalizes across different models.
    """

    def __init__(
        self,
        models: list[str],
        docker_image: str = "posit-gskill-eval:latest",
        timeout: int = 600,
        packages_dir: Path = Path("packages"),
        aggregation: Literal["min", "mean", "weighted"] = "min",
        parallel: bool = True,
    ) -> None:
        """Initialize multi-model evaluator.

        Args:
            models: List of models to evaluate on.
            docker_image: Docker image to use.
            timeout: Timeout per evaluation.
            packages_dir: Directory containing R packages.
            aggregation: How to aggregate scores across models.
                - "min": Use minimum score (optimize worst-case)
                - "mean": Use average score
                - "weighted": Weight by model capability
            parallel: Whether to evaluate models in parallel.
        """
        self.models = models
        self.packages_dir = packages_dir
        self.aggregation = aggregation
        self.parallel = parallel

        # Create evaluator for each model
        self.evaluators = {
            model: SkillEvaluator(
                docker_image=docker_image,
                timeout=timeout,
                packages_dir=packages_dir,
                model=model,
            )
            for model in models
        }

        # Weights for weighted aggregation (weaker models weighted higher)
        # glm-4.5 is weakest, so weight it highest
        self.model_weights = {
            "glm-4.5": 0.5,
            "glm-4.6": 0.3,
            "glm-4.7": 0.2,
        }

    def __call__(
        self,
        candidate: str,
        example: TestingTask,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate skill on all models and aggregate.

        Args:
            candidate: The skill markdown text to evaluate.
            example: A TestingTask from the dataset.

        Returns:
            Tuple of (aggregated_score, side_info) where side_info contains
            per-model breakdowns.
        """
        func_name = getattr(example, "function_name", None) or "unknown"
        oa.log(f"Multi-model eval on task: {example.task_id}")
        oa.log(f"Models: {', '.join(self.models)}")

        # Evaluate on all models
        per_model_scores: dict[str, float] = {}
        per_model_info: dict[str, dict[str, Any]] = {}

        if self.parallel and len(self.models) > 1:
            # Parallel evaluation
            def eval_model(model: str) -> tuple[str, float, dict]:
                score, info = self.evaluators[model](candidate, example)
                return model, score, info

            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = {executor.submit(eval_model, m): m for m in self.models}
                for future in as_completed(futures):
                    model, score, info = future.result()
                    per_model_scores[model] = score
                    per_model_info[model] = info
        else:
            # Sequential evaluation
            for model in self.models:
                score, info = self.evaluators[model](candidate, example)
                per_model_scores[model] = score
                per_model_info[model] = info

        # Aggregate scores
        if self.aggregation == "min":
            agg_score = min(per_model_scores.values())
        elif self.aggregation == "mean":
            agg_score = sum(per_model_scores.values()) / len(per_model_scores)
        elif self.aggregation == "weighted":
            total_weight = 0.0
            weighted_sum = 0.0
            for model, score in per_model_scores.items():
                weight = self.model_weights.get(model, 0.33)
                weighted_sum += score * weight
                total_weight += weight
            agg_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            agg_score = min(per_model_scores.values())

        # Build combined side info
        # Use the info from the worst-performing model for reflection
        worst_model = min(per_model_scores.keys(), key=lambda k: per_model_scores[k])

        side_info: dict[str, Any] = {
            "task_id": example.task_id,
            "function_name": func_name,
            "aggregated_score": agg_score,
            "aggregation_method": self.aggregation,
            "per_model_scores": per_model_scores,
            "worst_model": worst_model,
            # Include detailed info from worst model for reflection
            "worst_model_success": per_model_info[worst_model].get("success", False),
        }

        # Add failure info from worst model if it failed
        if not per_model_info[worst_model].get("success", True):
            for key in ["failure_category", "error_message", "failed_tests", "generated_code"]:
                if key in per_model_info[worst_model]:
                    side_info[key] = per_model_info[worst_model][key]

        # Log summary
        scores_str = ", ".join(f"{m}:{s:.0%}" for m, s in per_model_scores.items())
        oa.log(f"Per-model: [{scores_str}] -> agg={agg_score:.0%}")

        return agg_score, side_info


def optimize_skill(
    seed_skill: str,
    train_tasks: list[TestingTask],
    val_tasks: list[TestingTask],
    docker_image: str = "posit-gskill-eval:latest",
    timeout: int = 600,
    max_metric_calls: int = 100,
    reflection_lm: str | None = None,
    model: str = "glm-4.5",
    models: list[str] | None = None,  # NEW: multi-model optimization
    aggregation: Literal["min", "mean", "weighted"] = "min",  # NEW
    parallel_eval: bool = True,  # NEW: parallel task evaluation
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
        reflection_lm: LiteLLM model string for reflection. Defaults to config.
        model: Single model for task execution (ignored if models is set).
        models: List of models for multi-model optimization.
        aggregation: How to aggregate scores across models ("min", "mean", "weighted").
        parallel_eval: Whether to enable parallel evaluation in GEPA.
        run_dir: Directory to save optimization state.

    Returns:
        GEPAResult with best_candidate and optimization history.
    """
    config = get_llm_config()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not set in environment.")

    if reflection_lm is None:
        reflection_lm = config.reflection

    # Create evaluator(s)
    if models and len(models) > 1:
        logger.info(f"Multi-model optimization with models: {models}")
        logger.info(f"Aggregation method: {aggregation}")
        evaluator = MultiModelSkillEvaluator(
            models=models,
            docker_image=docker_image,
            timeout=timeout,
            aggregation=aggregation,
            parallel=parallel_eval,
        )
    else:
        single_model = models[0] if models else model
        logger.info(f"Single-model optimization with: {single_model}")
        evaluator = SkillEvaluator(
            docker_image=docker_image,
            timeout=timeout,
            model=single_model,
        )

    # Configure GEPA
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_metric_calls,
            run_dir=run_dir,
            parallel=parallel_eval,  # Enable parallel task evaluation
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
    model_info = f"models={models}" if models else f"model={model}"
    logger.info(
        f"Starting GEPA optimization with {len(train_tasks)} train, {len(val_tasks)} val tasks"
    )
    logger.info(f"Models: {model_info}, Max metric calls: {max_metric_calls}")
    logger.info(f"Parallel evaluation: {parallel_eval}")

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
