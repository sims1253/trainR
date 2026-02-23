"""GEPA adapter for optimize-anything integration.

This module provides the adapter layer between GEPA's optimize-anything API
and the benchmark experiment runner. It supports multiple objective functions
for optimizing skills and other target types.

Objective Functions:
-------------------
- pass-rate: Maximize simple pass rate (default)
- cost-adjusted: Maximize pass rate with token cost penalty
- weighted-by-difficulty: Weight results by task difficulty

Usage:
------
    from bench.optimize.gepa_adapter import GEPASkillEvaluator, ObjectiveType

    evaluator = GEPASkillEvaluator(
        base_config=experiment_config,
        objective=ObjectiveType.PASS_RATE,
    )

    score, info = evaluator(candidate_skill, task)
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from bench.experiments.config import ExperimentConfig
from bench.experiments.runner import ExperimentRunner
from bench.optimize.targets.base import OptimizableTarget
from bench.optimize.targets.skill import SkillCandidate, SkillTarget
from bench.schema.v1 import ResultV1

logger = logging.getLogger(__name__)


class ObjectiveType(str, Enum):
    """Available optimization objectives."""

    PASS_RATE = "pass-rate"
    """Maximize simple pass rate."""

    COST_ADJUSTED = "cost-adjusted"
    """Maximize pass rate with token cost penalty."""

    WEIGHTED_BY_DIFFICULTY = "weighted-by-difficulty"
    """Weight results by task difficulty."""


@dataclass
class ObjectiveConfig:
    """Configuration for optimization objectives."""

    objective: ObjectiveType = ObjectiveType.PASS_RATE
    """The optimization objective to use."""

    cost_penalty_factor: float = 0.0001
    """Penalty per 1000 tokens for cost-adjusted objective."""

    difficulty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "easy": 1.0,
            "medium": 1.5,
            "hard": 2.0,
        }
    )
    """Weights for different difficulty levels."""


@dataclass
class EvaluationResult:
    """Result from evaluating a candidate."""

    score: float
    """The computed score based on objective."""

    pass_rate: float
    """Raw pass rate."""

    total_tasks: int
    """Total tasks evaluated."""

    passed_tasks: int
    """Number of passed tasks."""

    total_tokens: int
    """Total tokens used."""

    avg_latency_s: float
    """Average latency in seconds."""

    per_task_results: list[dict[str, Any]] = field(default_factory=list)
    """Detailed per-task results."""

    errors: list[str] = field(default_factory=list)
    """Any errors encountered."""


class GEPASkillEvaluator:
    """
    Evaluator for skill optimization using the benchmark experiment runner.

    This evaluator wraps the ExperimentRunner to evaluate skill candidates
    on a set of tasks. It supports multiple objective functions.

    Usage with GEPA:
    ---------------
    ```python
    import gepa.optimize_anything as oa

    evaluator = GEPASkillEvaluator(
        base_config=config,
        objective=ObjectiveType.PASS_RATE,
    )

    result = oa.optimize_anything(
        seed_candidate=seed_skill,
        evaluator=evaluator,
        dataset=train_tasks,
        valset=val_tasks,
        objective="Improve skill pass rate",
        config=gepa_config,
    )
    ```
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        objective: ObjectiveType | ObjectiveConfig = ObjectiveType.PASS_RATE,
        target: OptimizableTarget | None = None,
        output_dir: Path | None = None,
        cleanup_after_eval: bool = True,
    ) -> None:
        """Initialize the skill evaluator.

        Args:
            base_config: Base experiment configuration to clone for each evaluation.
            objective: Objective type or full objective configuration.
            target: OptimizableTarget instance (defaults to SkillTarget).
            output_dir: Directory for evaluation outputs (temp if None).
            cleanup_after_eval: Whether to clean up temp directories after evaluation.
        """
        self.base_config = base_config
        self.objective = (
            ObjectiveConfig(objective=objective)
            if isinstance(objective, ObjectiveType)
            else objective
        )
        self.target = target or SkillTarget()
        self.output_dir = output_dir
        self.cleanup_after_eval = cleanup_after_eval

        # Track evaluations for trajectory
        self.evaluation_count = 0
        self.evaluation_history: list[dict[str, Any]] = []

    def __call__(
        self,
        candidate: str | SkillCandidate,
        task: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a skill candidate on a single task.

        This is the signature GEPA expects for generalization mode.

        Args:
            candidate: The skill to evaluate (string or SkillCandidate).
            task: A task object from the dataset.

        Returns:
            Tuple of (score, side_info) where:
            - score: Computed score based on objective (0.0 to 1.0)
            - side_info: Dict with diagnostic information for GEPA reflection
        """
        # Convert string to SkillCandidate if needed
        if isinstance(candidate, str):
            candidate = SkillCandidate(content=candidate)

        self.evaluation_count += 1

        # Create a modified config with this candidate applied
        config = self.target.apply_candidate_to_config(self.base_config, candidate)

        # Create a task-specific config
        task_config = self._create_task_config(config, task)

        try:
            # Run the experiment
            runner = ExperimentRunner(task_config)
            runner.setup()

            # Suppress detailed logging during evaluation
            old_level = logging.getLogger("bench.experiments.runner").level
            logging.getLogger("bench.experiments.runner").setLevel(logging.WARNING)

            try:
                manifest = runner.run()
            finally:
                logging.getLogger("bench.experiments.runner").setLevel(old_level)

            # Compute result
            eval_result = self._compute_score(runner.results)

            # Build side info for GEPA reflection
            side_info = self._build_side_info(eval_result, task, candidate)

            # Record in history
            self.evaluation_history.append(
                {
                    "evaluation_id": self.evaluation_count,
                    "task_id": getattr(task, "task_id", "unknown"),
                    "score": eval_result.score,
                    "pass_rate": eval_result.pass_rate,
                    "candidate_hash": self.target.compute_candidate_hash(candidate),
                }
            )

            return eval_result.score, side_info

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            side_info = {
                "task_id": getattr(task, "task_id", "unknown"),
                "error": str(e),
                "success": False,
            }
            return 0.0, side_info

    def _create_task_config(
        self,
        config: ExperimentConfig,
        task: Any,
    ) -> ExperimentConfig:
        """Create a config for evaluating a single task.

        Args:
            config: Base configuration.
            task: Task to create config for.

        Returns:
            Modified ExperimentConfig for single task evaluation.
        """
        config_dict = config.model_dump(mode="json")

        # Override task selection to run just this task
        config_dict["tasks"] = {
            "selection": "task_id",
            "task_id": getattr(task, "task_id", None),
            "dir": config.tasks.dir,
        }

        # Reduce output verbosity
        config_dict["output"]["save_intermediate"] = False
        config_dict["execution"]["save_trajectories"] = False

        # Use temp output dir if not specified
        if self.output_dir is None:
            import tempfile

            config_dict["output"]["dir"] = tempfile.mkdtemp(prefix="gepa_eval_")
        else:
            config_dict["output"]["dir"] = str(self.output_dir / f"eval_{self.evaluation_count}")

        return ExperimentConfig.from_dict(config_dict)

    def _compute_score(self, results: list[ResultV1]) -> EvaluationResult:
        """Compute the objective score from results.

        Args:
            results: List of evaluation results.

        Returns:
            EvaluationResult with computed score.
        """
        if not results:
            return EvaluationResult(
                score=0.0,
                pass_rate=0.0,
                total_tasks=0,
                passed_tasks=0,
                total_tokens=0,
                avg_latency_s=0.0,
            )

        total_tasks = len(results)
        passed_tasks = sum(1 for r in results if r.passed)
        pass_rate = passed_tasks / total_tasks if total_tasks > 0 else 0.0

        total_tokens = sum(r.token_usage.total for r in results)
        avg_latency = sum(r.latency_s for r in results) / total_tasks if total_tasks > 0 else 0.0

        # Compute score based on objective
        if self.objective.objective == ObjectiveType.PASS_RATE:
            score = pass_rate

        elif self.objective.objective == ObjectiveType.COST_ADJUSTED:
            # Penalize by token usage
            token_penalty = (total_tokens / 1000) * self.objective.cost_penalty_factor
            score = pass_rate - token_penalty

        elif self.objective.objective == ObjectiveType.WEIGHTED_BY_DIFFICULTY:
            # Weight by difficulty
            weighted_sum = 0.0
            weight_total = 0.0
            for r in results:
                # Get difficulty from metadata if available
                difficulty = "medium"  # default
                if hasattr(r, "metadata") and isinstance(r.metadata, dict):
                    difficulty = r.metadata.get("difficulty", "medium")

                weight = self.objective.difficulty_weights.get(difficulty, 1.0)
                weighted_sum += (1.0 if r.passed else 0.0) * weight
                weight_total += weight

            score = weighted_sum / weight_total if weight_total > 0 else 0.0

        else:
            score = pass_rate

        # Build per-task results
        per_task_results = []
        for r in results:
            per_task_results.append(
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "score": r.score,
                    "latency_s": r.latency_s,
                    "tokens": r.token_usage.total,
                    "error_category": r.error_category.value if r.error_category else None,
                }
            )

        return EvaluationResult(
            score=max(0.0, score),  # Ensure non-negative
            pass_rate=pass_rate,
            total_tasks=total_tasks,
            passed_tasks=passed_tasks,
            total_tokens=total_tokens,
            avg_latency_s=avg_latency,
            per_task_results=per_task_results,
        )

    def _build_side_info(
        self,
        eval_result: EvaluationResult,
        task: Any,
        candidate: SkillCandidate,
    ) -> dict[str, Any]:
        """Build side info for GEPA reflection.

        Args:
            eval_result: Evaluation result.
            task: Task that was evaluated.
            candidate: Skill candidate that was evaluated.

        Returns:
            Dictionary with diagnostic information.
        """
        side_info: dict[str, Any] = {
            "task_id": getattr(task, "task_id", "unknown"),
            "source_package": getattr(task, "source_package", "unknown"),
            "function_name": getattr(task, "function_name", None),
            "difficulty": str(getattr(task, "difficulty", "unknown")),
            "success": eval_result.pass_rate > 0,
            "score": eval_result.score,
            "pass_rate": eval_result.pass_rate,
            "total_tokens": eval_result.total_tokens,
            "avg_latency_s": eval_result.avg_latency_s,
            "candidate_hash": self.target.compute_candidate_hash(candidate),
        }

        # Add failure information if any
        if eval_result.pass_rate < 1.0:
            failed_tasks = [t for t in eval_result.per_task_results if not t["passed"]]
            if failed_tasks:
                side_info["failed_tasks"] = failed_tasks[:3]  # Limit to 3 for reflection

                # Add error details
                for ft in failed_tasks[:3]:
                    if ft.get("error_category"):
                        side_info["error_category"] = ft["error_category"]

        # Add candidate preview
        content_preview = candidate.get_full_content()
        if len(content_preview) > 1000:
            content_preview = content_preview[:1000] + "\n... [truncated]"
        side_info["candidate_preview"] = content_preview

        return side_info


class BatchEvaluator:
    """
    Evaluator that runs candidates on multiple tasks in batch.

    This is useful for evaluating on a full training/validation set
    rather than individual tasks.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        objective: ObjectiveType | ObjectiveConfig = ObjectiveType.PASS_RATE,
        target: OptimizableTarget | None = None,
        parallel: bool = False,
    ) -> None:
        """Initialize batch evaluator.

        Args:
            base_config: Base experiment configuration.
            objective: Optimization objective.
            target: OptimizableTarget instance.
            parallel: Whether to evaluate tasks in parallel.
        """
        self.base_config = base_config
        self.objective = (
            ObjectiveConfig(objective=objective)
            if isinstance(objective, ObjectiveType)
            else objective
        )
        self.target = target or SkillTarget()
        self.parallel = parallel

    def evaluate_batch(
        self,
        candidate: str | SkillCandidate,
        tasks: list[Any],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate on a batch of tasks.

        Args:
            candidate: Skill to evaluate.
            tasks: List of tasks to evaluate on.

        Returns:
            Tuple of (average_score, aggregated_side_info).
        """
        if isinstance(candidate, str):
            candidate = SkillCandidate(content=candidate)

        evaluator = GEPASkillEvaluator(
            base_config=self.base_config,
            objective=self.objective,
            target=self.target,
        )

        scores = []
        all_side_info = {}

        for task in tasks:
            score, side_info = evaluator(candidate, task)
            scores.append(score)
            all_side_info[getattr(task, "task_id", f"task_{len(scores)}")] = side_info

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Aggregate summary
        aggregated = {
            "num_tasks": len(tasks),
            "avg_score": avg_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "scores": scores,
        }

        return avg_score, aggregated


def create_gepa_evaluator(
    config: ExperimentConfig,
    objective: str = "pass-rate",
    **kwargs: Any,
) -> GEPASkillEvaluator:
    """Factory function to create a GEPA evaluator.

    Args:
        config: Experiment configuration.
        objective: Objective type string.
        **kwargs: Additional arguments for GEPASkillEvaluator.

    Returns:
        Configured GEPASkillEvaluator instance.
    """
    objective_type = ObjectiveType(objective)
    return GEPASkillEvaluator(
        base_config=config,
        objective=objective_type,
        **kwargs,
    )
