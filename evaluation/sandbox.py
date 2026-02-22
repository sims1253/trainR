"""Evaluation sandbox using Docker + cc-mirror."""

import logging
import os
import time
from pathlib import Path
from typing import Any

from .models import EvaluationResult, FailureCategory, TestResult, TrajectoryRecord
from .pi_runner import DockerPiRunner, DockerPiRunnerConfig

logger = logging.getLogger(__name__)


class EvaluationSandbox:
    """Orchestrates task evaluation using Docker + cc-mirror."""

    def __init__(
        self,
        runner_config: DockerPiRunnerConfig | None = None,
    ):
        if runner_config is None:
            runner_config = DockerPiRunnerConfig()
        self.runner = DockerPiRunner(runner_config)

    def evaluate_task(
        self,
        task: Any,  # TestingTask from task_generator
        skill_prompt: str | None,
        package_dir: Path | None = None,
        model: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single task with the given skill.

        Args:
            task: The testing task to evaluate.
            skill_prompt: The skill prompt to use (None for no skill, raw task instruction).
            package_dir: Path to the R package (defaults to packages/{source_package}).
            model: Model to use for task execution (default: from llm.yaml task_agent).

        Returns:
            EvaluationResult with score and details.
        """
        # Resolve model from llm.yaml config
        if model is None:
            from config import get_llm_config

            model = get_llm_config().task_agent
        start_time = time.time()

        if not os.environ.get("OPENROUTER_API_KEY"):
            return EvaluationResult(
                task_id=task.task_id,
                success=False,
                score=0.0,
                error_message="OPENROUTER_API_KEY not set in environment",
                failure_category=FailureCategory.SYNTAX_ERROR,
                execution_time=time.time() - start_time,
            )

        # Determine package path
        if package_dir is None:
            # Try source_package first, then derive from source.repo
            source_package = getattr(task, "source_package", None)
            if source_package is None:
                # Try to get from source.repo (format: "owner/repo")
                source = getattr(task, "source", {})
                if isinstance(source, dict):
                    repo = source.get("repo", "")
                else:
                    repo = getattr(source, "repo", "")
                source_package = repo.split("/")[-1] if "/" in str(repo) else str(repo)

            package_dir = Path(f"packages/{source_package}")

        if not package_dir.exists():
            return EvaluationResult(
                task_id=task.task_id,
                success=False,
                score=0.0,
                error_message=f"Package directory not found: {package_dir}",
                failure_category=FailureCategory.SYNTAX_ERROR,
                execution_time=time.time() - start_time,
            )

        # Run evaluation in Docker
        result = self.runner.run_evaluation(
            skill_content=skill_prompt,
            task_instruction=task.instruction,
            task_context=task.context,
            package_dir=package_dir,
            model=model,
            task=task,  # Pass task for repo/commit info
        )

        # Parse results
        test_results = self._parse_test_results(result)
        success = result.get("success", False)

        failure_category = None
        if not success:
            failure_category = self._classify_failure(result)

        return EvaluationResult(
            task_id=task.task_id,
            success=success,
            score=1.0 if success else 0.0,
            generated_code=result.get("output", ""),
            test_results=test_results,
            failure_category=failure_category,
            execution_time=time.time() - start_time,
            token_usage={},  # cc-mirror doesn't return token counts easily
        )

    def _parse_test_results(self, result: dict) -> list[TestResult]:
        """Parse test results from Docker output."""
        test_results = []

        # DockerPiRunner format: test_results is a dict with 'passed', 'failed', 'total', 'tests' list
        tr = result.get("test_results", {})
        if isinstance(tr, dict):
            for test in tr.get("tests", []):
                test_results.append(
                    TestResult(
                        name=test.get("name", "unknown"),
                        passed=test.get("passed", False),
                        message=test.get("message", ""),
                    )
                )

        # Fallback: create single result from success
        if not test_results:
            test_results.append(
                TestResult(
                    name="evaluation",
                    passed=result.get("success", False),
                    message=result.get("error", result.get("output", "")),
                )
            )

        return test_results

    def _classify_failure(self, result: dict) -> FailureCategory:
        """Classify failure type from result."""
        error = result.get("error", "") or result.get("output", "")
        error_lower = error.lower()

        if "timeout" in error_lower:
            return FailureCategory.TIMEOUT
        elif "syntax" in error_lower or "parse" in error_lower:
            return FailureCategory.SYNTAX_ERROR
        elif "could not find function" in error_lower:
            return FailureCategory.MISSING_IMPORT
        elif "snapshot" in error_lower:
            return FailureCategory.SNAPSHOT_MISMATCH
        elif "failure" in error_lower or "expect" in error_lower:
            return FailureCategory.TEST_FAILURE
        else:
            return FailureCategory.INCOMPLETE_SOLUTION

    def build_trajectory(
        self,
        task: Any,  # TestingTask from task_generator
        skill_prompt: str,
        skill_version: str,
        result: EvaluationResult,
    ) -> TrajectoryRecord:
        """Build a trajectory record for GEPA reflection."""
        feedback = self._generate_feedback(task, result)
        suggestion = self._generate_suggestion(task, result)

        return TrajectoryRecord(
            task_id=task.task_id,
            skill_version=skill_version,
            input_prompt=task.instruction,
            generated_output=result.generated_code or "",
            evaluation_result=result,
            feedback=feedback,
            improvement_suggestion=suggestion,
        )

    def _generate_feedback(self, task: Any, result: EvaluationResult) -> str:
        """Generate feedback on the evaluation result."""
        if result.success:
            return f"Test passed successfully for task {task.task_id}"

        parts = [f"Test failed for task {task.task_id}"]

        if result.failure_category:
            parts.append(f"Failure type: {result.failure_category}")

        if result.error_message:
            parts.append(f"Error: {result.error_message}")

        for test in result.test_results:
            if not test.passed:
                parts.append(f"Failed test: {test.name} - {test.message}")

        return "\n".join(parts)

    def _generate_suggestion(self, task: Any, result: EvaluationResult) -> str:
        """Generate improvement suggestion for the skill."""
        if result.success:
            return ""

        suggestions = {
            FailureCategory.SYNTAX_ERROR: "Strengthen code examples in the skill to show correct R syntax",
            FailureCategory.TEST_FAILURE: "Improve guidance on assertion selection and test structure",
            FailureCategory.TIMEOUT: "Encourage simpler, more focused test approaches",
            FailureCategory.MISSING_IMPORT: "Emphasize the importance of loading required libraries",
            FailureCategory.SNAPSHOT_MISMATCH: "Provide better examples of snapshot testing patterns",
            FailureCategory.INCOMPLETE_SOLUTION: "Guide toward more comprehensive test coverage",
        }

        failure_cat = result.failure_category
        if failure_cat is None:
            return "Review and improve the skill guidance"
        return suggestions.get(failure_cat, "Review and improve the skill guidance")
