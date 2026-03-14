"""Evaluation sandbox using Docker + cc-mirror."""

import logging
import time
import warnings
from pathlib import Path
from typing import Any, TypedDict

from bench.provider import get_env_var
from bench.telemetry import TelemetryCollector
from config import get_llm_config

from .models import EvaluationResult, FailureCategory, TestResult, TrajectoryRecord
from .pi_runner import DockerPiRunner, DockerPiRunnerConfig

logger = logging.getLogger(__name__)


class _TelemetryFields(TypedDict):
    tool_call_counts: dict[str, int]
    tool_errors: dict[str, int]
    tool_total_time_ms: dict[str, float]


def _telemetry_fields(telemetry: TelemetryCollector) -> _TelemetryFields:
    """Extract flat tool metrics for EvaluationResult payloads."""
    schema = telemetry.collect(model=None, harness="evaluation_sandbox")
    return {
        "tool_call_counts": {
            str(tool): int(count) for tool, count in schema.tools.by_tool.items()
        },
        "tool_errors": {
            str(tool): int(count) for tool, count in schema.tools.errors.items()
        },
        "tool_total_time_ms": dict(schema.tools.duration_ms_by_tool),
    }


def _select_env_var(env_vars: list[str]) -> tuple[str | None, str | None]:
    """Choose the first available credential; otherwise return first required var."""
    for env_var in env_vars:
        value = get_env_var(env_var)
        if value:
            return env_var, value
    if env_vars:
        env_var = env_vars[0]
        return env_var, get_env_var(env_var)
    return None, None


def _resolve_env_vars_from_llm_catalog(model: str) -> list[str]:
    """Resolve candidate env vars for a model using llm.yaml name/id mappings."""
    llm_config = get_llm_config()
    models_cfg = llm_config.config.get("models", {})
    providers_cfg = llm_config.config.get("providers", {})
    candidates: list[str] = []

    def add_env_for_provider(provider_name: str | None) -> None:
        if not provider_name:
            return
        env_var = providers_cfg.get(provider_name, {}).get("api_key_env")
        if not env_var:
            try:
                from bench.provider import resolve_api_key_env

                env_var = resolve_api_key_env(provider_name)
            except (ImportError, KeyError):
                env_var = None
        if env_var and env_var not in candidates:
            candidates.append(env_var)

    # Match against resolved model IDs from llm.yaml.
    for model_name, model_cfg in models_cfg.items():
        try:
            full_cfg = llm_config.get_model_config(model_name)
        except ValueError:
            continue
        # Construct full model ID with prefix (e.g., "openai/glm-5")
        raw_id = full_cfg.get("id", model_name)
        prefix = full_cfg.get("litellm_prefix", "")
        model_id = f"{prefix}/{raw_id}" if prefix else raw_id
        if model_id != model:
            continue
        if "providers" in model_cfg:
            for provider_entry in model_cfg.get("providers", []):
                add_env_for_provider(provider_entry.get("provider"))
        else:
            add_env_for_provider(model_cfg.get("provider"))

    # Match against raw model ids from llm.yaml.
    for model_cfg in models_cfg.values():
        if "providers" in model_cfg:
            for provider_entry in model_cfg.get("providers", []):
                if provider_entry.get("id") == model:
                    add_env_for_provider(provider_entry.get("provider"))
        elif model_cfg.get("id") == model:
            add_env_for_provider(model_cfg.get("provider"))

    return candidates


def get_required_api_key(model: str) -> tuple[str | None, str | None]:
    """Get the required API key environment variable for a model.

    Args:
        model: Model name (either a short name from llm.yaml or provider/model format)

    Returns:
        Tuple of (env_var_name, env_var_value) or (None, None) if no key required.
    """
    llm_config = get_llm_config()

    # Try to resolve as a model name from llm.yaml
    try:
        model_cfg = llm_config.get_model_config(model)
        env_var = model_cfg.get("api_key_env")
        if env_var:
            return env_var, get_env_var(env_var)
    except ValueError:
        pass

    # Try to resolve from llm.yaml model id mappings.
    env_var, api_key = _select_env_var(_resolve_env_vars_from_llm_catalog(model))
    if env_var:
        return env_var, api_key

    # Try central resolver for provider prefix
    if "/" in model:
        provider = model.split("/", 1)[0]
        try:
            from bench.provider import resolve_api_key_env

            key_name = resolve_api_key_env(provider)
            return key_name, get_env_var(key_name)
        except (ImportError, KeyError):
            pass

        # Fallback to canonical provider mapping (deprecated)
        warnings.warn(
            f"Using fallback mapping in get_required_api_key(). "
            f"Prefer bench.provider.resolve_api_key_env() for provider '{provider}'.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from bench.provider.resolver import PROVIDER_API_KEY_MAP
        except ImportError:
            provider_api_key_map = {
                "openrouter": "OPENROUTER_API_KEY",
                "opencode": "OPENCODE_API_KEY",
                "zai": "Z_AI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }
        else:
            provider_api_key_map = PROVIDER_API_KEY_MAP

        key_name = provider_api_key_map.get(provider)
        if key_name:
            return key_name, get_env_var(key_name)

    # No provider prefix - assume no key required
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
        telemetry.start()

        # Resolve model from llm.yaml config
        if model is None:
            model = get_llm_config().task_agent
        start_time = time.time()

        # Provider-aware API key check
        env_var, api_key = get_required_api_key(model)
        if env_var and not api_key:
            telemetry.record_tool_call("config_check", success=False, duration_ms=0.0)
            telemetry_fields = _telemetry_fields(telemetry)
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
        package_resolution_start = time.perf_counter()
        package_resolution_success = True
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
            package_resolution_success = False
        telemetry.record_tool_call(
            "package_resolution",
            success=package_resolution_success,
            duration_ms=(time.perf_counter() - package_resolution_start) * 1000.0,
        )
        if not package_resolution_success:
            telemetry_fields = _telemetry_fields(telemetry)
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

        # Run evaluation in Docker with telemetry
        docker_eval_start = time.perf_counter()
        result = self.runner.run_evaluation(
            skill_content=skill_prompt,
            task_instruction=task.instruction,
            task_context=task.context,
            package_dir=package_dir,
            model=model,
            task=task,  # Pass task for repo/commit info
        )
        docker_success = not bool(result.get("error"))
        telemetry.record_tool_call(
            "docker_evaluation",
            success=docker_success,
            duration_ms=(time.perf_counter() - docker_eval_start) * 1000.0,
        )

        # Track test parsing
        test_parsing_start = time.perf_counter()
        test_results = self._parse_test_results(result)
        telemetry.record_tool_call(
            "test_parsing",
            success=True,
            duration_ms=(time.perf_counter() - test_parsing_start) * 1000.0,
        )

        success = result.get("success", False)

        failure_category = None
        if not success:
            failure_category = self._classify_failure(result)

        # Get telemetry fields
        telemetry_fields = _telemetry_fields(telemetry)

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
