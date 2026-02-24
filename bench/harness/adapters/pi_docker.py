"""Pi Docker harness - wraps DockerPiRunner in AgentHarness protocol."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from evaluation.models import FailureCategory
from evaluation.pi_runner import DockerPiRunner, DockerPiRunnerConfig

from ..base import (
    AgentHarness,
    ErrorCategory,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    TestResult,
    TokenUsage,
)
from ..registry import register_harness

# Mapping from FailureCategory to ErrorCategory
FAILURE_TO_ERROR_CATEGORY: dict[FailureCategory, ErrorCategory] = {
    FailureCategory.CONFIG_ERROR: ErrorCategory.INVALID_REQUEST,
    FailureCategory.ENVIRONMENT_ERROR: ErrorCategory.INVALID_REQUEST,
    FailureCategory.PACKAGE_NOT_FOUND: ErrorCategory.TASK_ERROR,
    FailureCategory.TIMEOUT: ErrorCategory.TIMEOUT,
    FailureCategory.SYNTAX_ERROR: ErrorCategory.TASK_ERROR,
    FailureCategory.MISSING_IMPORT: ErrorCategory.TASK_ERROR,
    FailureCategory.TEST_FAILURE: ErrorCategory.TEST_ERROR,
    FailureCategory.WRONG_ASSERTION: ErrorCategory.TEST_ERROR,
    FailureCategory.SNAPSHOT_MISMATCH: ErrorCategory.TEST_ERROR,
    FailureCategory.INCOMPLETE_SOLUTION: ErrorCategory.TASK_ERROR,
    FailureCategory.OVERLY_COMPLEX: ErrorCategory.TASK_ERROR,
    FailureCategory.WRONG_FIXTURE_USAGE: ErrorCategory.TEST_ERROR,
}


@register_harness("pi_docker")
class PiDockerHarness(AgentHarness):
    """Harness that runs Pi CLI inside Docker container.

    This is the primary harness for R package testing tasks.
    """

    def __init__(self, config: HarnessConfig):
        super().__init__(config)
        self._runner: DockerPiRunner | None = None

    def setup(self) -> None:
        """Initialize Docker runner."""
        # Get config values with defaults, using extra="allow" for custom fields
        model = (
            getattr(self.config, "default_model", None)
            or self.config.model
            or "openrouter/openai/gpt-oss-120b:free"
        )
        docker_image = getattr(self.config, "docker_image", "posit-gskill-eval:latest")
        api_keys = getattr(self.config, "api_keys", None)
        sandbox_profile = getattr(self.config, "sandbox_profile", "networked")

        runner_config = DockerPiRunnerConfig(
            model=model,
            max_turns=self.config.max_turns or 50,
            timeout=int(self.config.timeout),
            docker_image=docker_image,
            api_keys=api_keys,
            sandbox_profile=sandbox_profile,
        )
        self._runner = DockerPiRunner(runner_config)

    def teardown(self) -> None:
        """Clean up runner resources."""
        self._runner = None

    def validate_environment(self) -> tuple[bool, list[str]]:
        """Check Docker and image availability."""
        issues: list[str] = []

        try:
            import subprocess

            # Check Docker is available
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                issues.append("Docker is not available")
                return (False, issues)

            # Check image exists
            docker_image = getattr(self.config, "docker_image", "posit-gskill-eval:latest")
            result = subprocess.run(
                ["docker", "images", "-q", docker_image],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                issues.append(
                    f"Docker image '{docker_image}' not found. Run 'make docker-build' first."
                )
                return (False, issues)

            return (True, [])

        except FileNotFoundError:
            issues.append("Docker command not found")
            return (False, issues)
        except Exception as e:
            issues.append(f"Unexpected error: {e}")
            return (False, issues)

    async def execute(self, request: HarnessRequest) -> HarnessResult:
        """Execute task using DockerPiRunner."""
        if self._runner is None:
            self.setup()

        run_id = str(uuid.uuid4())[:8]

        # Build task-like object for DockerPiRunner
        task = self._build_task(request)

        # Resolve package directory
        package_dir = self._resolve_package_dir(request)

        # Get skill content and context from request metadata
        skill_content = getattr(request, "skill_content", None) or request.metadata.get(
            "skill_content"
        )
        task_context = getattr(request, "context", "") or request.metadata.get("context", "")
        model = (
            getattr(request, "model", None) or request.metadata.get("model") or self.config.model
        )

        # Execute
        start_time = time.time()
        try:
            result_dict = self._runner.run_evaluation(
                skill_content=skill_content,
                task_instruction=request.prompt,
                task_context=task_context,
                package_dir=package_dir,
                model=model,
                task=task,
            )
            execution_time = time.time() - start_time

            return self._convert_result(request, result_dict, execution_time, run_id)

        except Exception as e:
            execution_time = time.time() - start_time
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=False,
                error_message=str(e),
                error_category=ErrorCategory.from_exception(e),
                execution_time=execution_time,
            )

    def _build_task(self, request: HarnessRequest) -> Any:
        """Build task-like object for DockerPiRunner."""

        # Create object with expected attributes
        class TaskLike:
            pass

        task = TaskLike()
        task.source_package = request.metadata.get("source_package", "")
        task.test_patch = getattr(request, "test_patch", "") or request.metadata.get(
            "test_patch", ""
        )
        task.patch = getattr(request, "gold_patch", "") or request.metadata.get("gold_patch", "")
        tests = getattr(request, "tests", {}) or request.metadata.get("tests", {})
        if isinstance(tests, dict):
            task.tests = tests
        else:
            task.tests = {
                "fail_to_pass": getattr(request, "fail_to_pass", [])
                or request.metadata.get("fail_to_pass", []),
                "pass_to_pass": getattr(request, "pass_to_pass", [])
                or request.metadata.get("pass_to_pass", []),
            }
        task.source = request.metadata.get("source", {})
        return task

    def _resolve_package_dir(self, request: HarnessRequest) -> Path:
        """Resolve package directory for task."""
        source_package = request.metadata.get("source_package", "")
        if source_package:
            return Path(f"packages/{source_package}")
        package_dir = request.metadata.get("package_dir", "packages/unknown")
        return Path(package_dir)

    def _convert_result(
        self,
        request: HarnessRequest,
        result_dict: dict[str, Any],
        execution_time: float,
        run_id: str,
    ) -> HarnessResult:
        """Convert DockerPiRunner result dict to HarnessResult."""
        # Convert test results
        test_results_data = result_dict.get("test_results", {})
        test_results: list[TestResult] = []

        # test_results from DockerPiRunner is a dict with passed, num_passed, etc.
        # not a list of individual tests, so we create a summary test result
        if isinstance(test_results_data, dict):
            passed = test_results_data.get("passed", False)
            num_passed = test_results_data.get("num_passed", 0)
            num_failed = test_results_data.get("num_failed", 0)
            test_results.append(
                TestResult(
                    name="test_summary",
                    passed=passed,
                    message=f"Passed: {num_passed}, Failed: {num_failed}",
                    execution_time=execution_time,
                )
            )
        elif isinstance(test_results_data, list):
            test_results = [
                TestResult(
                    name=tr.get("name", "unknown"),
                    passed=tr.get("passed", False),
                    message=tr.get("message", ""),
                    execution_time=tr.get("execution_time", 0.0),
                )
                for tr in test_results_data
            ]

        # Convert token usage
        token_usage_dict = result_dict.get("token_usage", {})
        token_usage = TokenUsage(
            prompt=token_usage_dict.get("input_tokens", 0),
            completion=token_usage_dict.get("output_tokens", 0),
            total=token_usage_dict.get("total_tokens", 0),
            cache_read=token_usage_dict.get("cache_read_tokens", 0),
            cache_write=token_usage_dict.get("cache_write_tokens", 0),
        )

        # Map error category
        error_category = ErrorCategory.NONE
        error_message = result_dict.get("error")
        if not result_dict.get("success", True):
            if error_message:
                # Try to classify based on error message
                if "timeout" in error_message.lower():
                    error_category = ErrorCategory.TIMEOUT
                else:
                    error_category = ErrorCategory.TASK_ERROR
            else:
                error_category = ErrorCategory.TEST_ERROR

        # Determine tests_passed based on test results
        tests_passed = result_dict.get("success", False)
        if test_results:
            tests_passed = all(tr.passed for tr in test_results)

        return HarnessResult(
            task_id=request.task_id,
            run_id=run_id,
            success=result_dict.get("success", False),
            output=result_dict.get("output", ""),
            patch=result_dict.get("generated_code"),
            tests_passed=tests_passed,
            test_results=test_results,
            error_message=error_message,
            error_category=error_category,
            execution_time=execution_time,
            token_usage=token_usage,
            model=result_dict.get("model"),
            metadata={
                "raw_result": result_dict,
                "tool_call_counts": result_dict.get("tool_call_counts", {}),
                "tool_errors": result_dict.get("tool_errors", {}),
                "tool_total_time_ms": result_dict.get("tool_total_time_ms", {}),
            },
        )
