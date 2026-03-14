"""Pi Docker harness - wraps DockerPiRunner in AgentHarness protocol."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.telemetry import (
    LatencyBreakdown,
    TelemetrySchema,
    ToolCallMetrics,
)
from bench.telemetry import (
    TokenUsage as TelemetryTokenUsage,
)
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


@dataclass
class TaskLike:
    """Task shim with the attributes expected by DockerPiRunner."""

    source_package: str = ""
    test_patch: str = ""
    patch: str = ""
    tests: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


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
        forward_github_token = bool(getattr(self.config, "forward_github_token", False))
        auth_policy = getattr(self.config, "auth_policy", "env")
        if hasattr(auth_policy, "value"):
            auth_policy = auth_policy.value

        save_traces = bool(getattr(self.config, "save_traces", False))
        trace_dir = getattr(self.config, "trace_dir", "container_logs")

        runner_config = DockerPiRunnerConfig(
            model=model,
            max_turns=self.config.max_turns or 50,
            timeout=int(self.config.timeout),
            docker_image=docker_image,
            api_keys=api_keys,
            forward_github_token=forward_github_token,
            sandbox_profile=sandbox_profile,
            auth_policy=str(auth_policy),
            keep_workspace_on_failure=bool(
                getattr(self.config, "keep_workspace_on_failure", False)
            ),
            save_traces=save_traces,
            trace_dir=Path(trace_dir),
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
        runner = self._runner
        if runner is None:
            raise RuntimeError("PiDockerHarness runner failed to initialize")

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
            result_dict = runner.run_evaluation(
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
                telemetry=TelemetrySchema(
                    tokens=TelemetryTokenUsage(
                        prompt=0,
                        completion=0,
                        total=0,
                    ),
                    turns=0,
                    tools=ToolCallMetrics(),
                    latency=LatencyBreakdown(
                        total_s=execution_time,
                        execution_s=execution_time,
                    ),
                    provider=self._infer_provider(model),
                    model=model,
                    harness="pi_docker",
                ),
            )

    def _build_task(self, request: HarnessRequest) -> Any:
        """Build task-like object for DockerPiRunner."""
        task = TaskLike(
            source_package=request.metadata.get("source_package", ""),
            test_patch=getattr(request, "test_patch", "") or request.metadata.get("test_patch", ""),
            patch=getattr(request, "gold_patch", "") or request.metadata.get("gold_patch", ""),
        )
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

        tool_call_counts = {
            str(tool): int(count)
            for tool, count in result_dict.get("tool_call_counts", {}).items()
            if isinstance(count, int | float)
        }
        tool_errors = {
            str(tool): int(count)
            for tool, count in result_dict.get("tool_errors", {}).items()
            if isinstance(count, int | float)
        }
        tool_total_time_ms = {
            str(tool): float(duration)
            for tool, duration in result_dict.get("tool_total_time_ms", {}).items()
            if isinstance(duration, int | float)
        }
        total_calls = sum(tool_call_counts.values())
        failed_calls = sum(tool_errors.values())
        successful_calls = max(0, total_calls - failed_calls)
        total_duration_ms = sum(tool_total_time_ms.values())

        # Map error category
        error_category = ErrorCategory.NONE
        error_message = result_dict.get("error")
        if not result_dict.get("success", True):
            if error_message:
                # Try to classify based on error message
                lower_error = error_message.lower()
                if "timeout" in lower_error or "timed out" in lower_error:
                    error_category = ErrorCategory.TIMEOUT
                elif any(
                    marker in lower_error
                    for marker in (
                        "429",
                        "rate limit",
                        "insufficient balance",
                        "no resource package",
                        "recharge",
                    )
                ):
                    error_category = ErrorCategory.RATE_LIMIT
                elif any(
                    marker in lower_error
                    for marker in (
                        "authentication_error",
                        "oidc",
                        "api key",
                        "no api key found",
                        "unauthorized",
                        "user not found",
                        "balance",
                        '"status":401',
                        '"status": 401',
                        '"code":401',
                        '"code": 401',
                        '"status_code":401',
                        '"status_code": 401',
                        '"status":403',
                        '"status": 403',
                        '"code":403',
                        '"code": 403',
                        '"status_code":403',
                        '"status_code": 403',
                    )
                ):
                    error_category = ErrorCategory.AUTH_ERROR
                elif (
                    ("model" in lower_error and "not found" in lower_error)
                    or "model not available" in lower_error
                    or "no such model" in lower_error
                    or "invalid model" in lower_error
                    or "--list-models" in lower_error
                ):
                    error_category = ErrorCategory.MODEL_ERROR
                elif (
                    "error ('test-" in lower_error
                    or "testthat" in lower_error
                    or "expected `" in lower_error
                    or "failed ──" in lower_error
                    or (test_results_data and not test_results_data.get("passed", True))
                ):
                    error_category = ErrorCategory.TEST_ERROR
                else:
                    error_category = ErrorCategory.TASK_ERROR
            else:
                error_category = ErrorCategory.TEST_ERROR

        # Determine tests_passed based on test results
        tests_passed = result_dict.get("success", False)
        if test_results:
            tests_passed = all(tr.passed for tr in test_results)

        model_name = result_dict.get("model")
        telemetry = TelemetrySchema(
            tokens=TelemetryTokenUsage(
                prompt=token_usage.prompt,
                completion=token_usage.completion,
                total=token_usage.total,
                cache_read=token_usage.cache_read,
                cache_write=token_usage.cache_write,
            ),
            turns=int(token_usage_dict.get("turn_count", 0) or 0),
            tools=ToolCallMetrics(
                total_calls=total_calls,
                successful_calls=successful_calls,
                failed_calls=failed_calls,
                total_duration_ms=total_duration_ms,
                by_tool=tool_call_counts,
                errors=tool_errors,
                duration_ms_by_tool=tool_total_time_ms,
            ),
            latency=LatencyBreakdown(
                total_s=execution_time,
                execution_s=execution_time,
            ),
            provider=self._infer_provider(model_name),
            model=model_name,
            harness="pi_docker",
        )

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
            model=model_name,
            telemetry=telemetry,
            metadata={"raw_result": result_dict},
        )

    @staticmethod
    def _infer_provider(model: str | None) -> str | None:
        """Infer provider name from model identifier."""
        if not model:
            return None
        if "/" not in model:
            return None
        return model.split("/", 1)[0]
