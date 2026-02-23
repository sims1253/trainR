"""Evaluation sandbox using Docker + cc-mirror."""

import logging
import os
import time
from pathlib import Path
from typing import Any

from config import get_llm_config

from bench.eval.telemetry import (
    TelemetryCollector,
    ToolErrorType,
    classify_error_type,
)

from .models import EvaluationResult, FailureCategory, TestResult, TrajectoryRecord
from .pi_runner import DockerPiRunner, DockerPiRunnerConfig

logger = logging.getLogger(__name__)


def get_required_api_key(model: str) -> tuple[str | None, str | None]:
    """Get the required API key environment variable for a model.

    Args:
        model: Model name (either a short name from llm.yaml or full LiteLLM path)

    Returns:
        Tuple of (env_var_name, env_var_value) or (None, None) if no key required.
    """
    llm_config = get_llm_config()

    # Try to resolve as a model name from llm.yaml
    try:
        model_cfg = llm_config.get_model_config(model)
        env_var = model_cfg.get("api_key_env")
        if env_var:
            return env_var, os.environ.get(env_var)
    except ValueError:
        pass

    # Fallback: infer from LiteLLM prefix in model string
    provider_key_mapping = {
        "openrouter/": "OPENROUTER_API_KEY",
        "opencode/": "OPENCODE_API_KEY",
        "zai/": "Z_AI_API_KEY",
        "openai/": "OPENAI_API_KEY",
        "anthropic/": "ANTHROPIC_API_KEY",
        "gemini/": "GEMINI_API_KEY",
    }

    for prefix, key_name in provider_key_mapping.items():
        if model.startswith(prefix):
            return key_name, os.environ.get(key_name)

    # No provider prefix - assume no key required or will be handled by LiteLLM
    return None, None


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
        # Initialize telemetry collector
        telemetry = TelemetryCollector()

        # Resolve model from llm.yaml config
        if model is None:
            model = get_llm_config().task_agent
        start_time = time.time()

        # Provider-aware API key check
        env_var, api_key = get_required_api_key(model)
        if env_var and not api_key:
            telemetry.record_error(
                "config_check",
                ToolErrorType.PERMISSION_DENIED,
                f"{env_var} not set in environment",
            )
            telemetry_fields = telemetry.get_result_fields()
            return EvaluationResult(
                task_id=task.task_id,
                success=False,
                score=0.0,
                error_message=f"{env_var} not set in environment (required for model: {model})",
                failure_category=FailureCategory.ENVIRONMENT_ERROR,
                execution_time=time.time() - start_time,
                tool_call_counts=telemetry_fields["tool_call_counts"],
                tool_errors=telemetry_fields["tool_errors"],
                tool_total_time_ms=telemetry_fields["tool_total_time_ms"],
            )

        # Track package resolution
        with telemetry.time_call("package_resolution") as ctx:
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
                ctx.failure(ToolErrorType.NOT_FOUND)
                telemetry_fields = telemetry.get_result_fields()
                return EvaluationResult(
                    task_id=task.task_id,
                    success=False,
                    score=0.0,
                    error_message=f"Package directory not found: {package_dir}",
                    failure_category=FailureCategory.PACKAGE_NOT_FOUND,
                    execution_time=time.time() - start_time,
                    tool_call_counts=telemetry_fields["tool_call_counts"],
                    tool_errors=telemetry_fields["tool_errors"],
                    tool_total_time_ms=telemetry_fields["tool_total_time_ms"],
                )
            ctx.success()

        # Run evaluation in Docker with telemetry
        with telemetry.time_call("docker_evaluation") as ctx:
            result = self.runner.run_evaluation(
                skill_content=skill_prompt,
                task_instruction=task.instruction,
                task_context=task.context,
                package_dir=package_dir,
                model=model,
                task=task,  # Pass task for repo/commit info
            )

            # Check for errors in the result
            if result.get("error"):
                error_msg = result.get("error", "")
                error_type = classify_error_type(error_msg)
                ctx.failure(error_type)
            else:
                ctx.success()

        # Track test parsing
        with telemetry.time_call("test_parsing") as ctx:
            test_results = self._parse_test_results(result)
            ctx.success()

        success = result.get("success", False)

        failure_category = None
        if not success:
            failure_category = self._classify_failure(result)

        # Get telemetry fields
        telemetry_fields = telemetry.get_result_fields()

        return EvaluationResult(
            task_id=task.task_id,
            success=success,
            score=1.0 if success else 0.0,
            generated_code=result.get("output", ""),
            test_results=test_results,
            failure_category=failure_category,
            execution_time=time.time() - start_time,
            token_usage=result.get("token_usage", {}),
            tool_call_counts=telemetry_fields["tool_call_counts"],
            tool_errors=telemetry_fields["tool_errors"],
            tool_total_time_ms=telemetry_fields["tool_total_time_ms"],
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

        # Check for infrastructure/environment issues first
        if "timeout" in error_lower:
            return FailureCategory.TIMEOUT
        elif "package" in error_lower and (
            "not found" in error_lower or "not available" in error_lower
        ):
            return FailureCategory.PACKAGE_NOT_FOUND
        elif (
            "api key" in error_lower or "api_key" in error_lower or "authentication" in error_lower
        ):
            return FailureCategory.ENVIRONMENT_ERROR
        elif "config" in error_lower and "error" in error_lower:
            return FailureCategory.CONFIG_ERROR
        # Code generation issues
        elif "syntax" in error_lower or "parse" in error_lower:
            return FailureCategory.SYNTAX_ERROR
        elif "could not find function" in error_lower:
            return FailureCategory.MISSING_IMPORT
        # Test failures (expected behavior)
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
            FailureCategory.CONFIG_ERROR: "Check configuration files for missing or invalid settings",
            FailureCategory.ENVIRONMENT_ERROR: "Ensure all required API keys and environment variables are set",
            FailureCategory.PACKAGE_NOT_FOUND: "Verify required R packages are installed in the evaluation environment",
            FailureCategory.TIMEOUT: "Encourage simpler, more focused test approaches",
            FailureCategory.SYNTAX_ERROR: "Strengthen code examples in the skill to show correct R syntax",
            FailureCategory.MISSING_IMPORT: "Emphasize the importance of loading required libraries",
            FailureCategory.TEST_FAILURE: "Improve guidance on assertion selection and test structure",
            FailureCategory.SNAPSHOT_MISMATCH: "Provide better examples of snapshot testing patterns",
            FailureCategory.INCOMPLETE_SOLUTION: "Guide toward more comprehensive test coverage",
            FailureCategory.WRONG_ASSERTION: "Improve guidance on selecting appropriate testthat assertions",
            FailureCategory.OVERLY_COMPLEX: "Encourage simpler, more focused test patterns",
            FailureCategory.WRONG_FIXTURE_USAGE: "Provide better examples of test fixture patterns",
        }

        failure_cat = result.failure_category
        if failure_cat is None:
            return "Review and improve the skill guidance"
        return suggestions.get(failure_cat, "Review and improve the skill guidance")
