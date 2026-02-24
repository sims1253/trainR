"""Main experiment runner.

The ExperimentRunner is the single execution engine for all benchmark runs.
It handles:
- Deterministic execution
- Artifact generation (results.jsonl, summary.json, manifest.json)
- Retry semantics
- Progress tracking
"""

import asyncio
import contextlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.experiments.config import ExperimentConfig, RetryStrategy
from bench.experiments.matrix import ExperimentMatrix, ExperimentRun, generate_matrix
from bench.harness import (
    HarnessConfig,
    HarnessRegistry,
    HarnessRequest,
    HarnessResult,
    ErrorCategory as HarnessErrorCategory,
)
from bench.schema.v1 import (
    CaseResultV1,
    ConfigFingerprintV1,
    EnvironmentFingerprintV1,
    ErrorCategoryV1,
    ManifestV1,
    ResultSummaryV1,
    ResultV1,
    TokenUsageV1,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Unified experiment runner.

    This is the canonical entry point for running benchmarks.
    It produces deterministic outputs and proper artifact structure.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        harness_name: str | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            harness_name: Override harness name (default: from config.execution.harness).
                          Priority: explicit param > config.execution.harness > "pi_docker".
            output_dir: Override output directory

        Raises:
            ValueError: If harness is not registered
        """
        self.config = config
        self.output_dir = output_dir or (
            Path(config.output.dir) / config.name if config.output.dir else None
        )
        self.matrix: ExperimentMatrix | None = None
        self.manifest: ManifestV1 | None = None
        self.results: list[ResultV1] = []

        # Determine harness: explicit param > config > default
        if harness_name is not None:
            self.harness_name = harness_name
        elif hasattr(config, "execution") and hasattr(config.execution, "harness"):
            self.harness_name = config.execution.harness
        else:
            self.harness_name = "pi_docker"

        # Validate harness is registered
        if self.harness_name not in HarnessRegistry.list_available():
            raise ValueError(
                f"Harness '{self.harness_name}' is not registered. "
                f"Available: {HarnessRegistry.list_available()}"
            )

    def setup(self) -> Path:
        """
        Set up the experiment run.

        - Creates output directory
        - Generates experiment matrix
        - Creates initial manifest
        - Validates environment

        Returns:
            Path to output directory
        """
        # Set random seed for determinism
        if self.config.determinism.seed is not None:
            random.seed(self.config.determinism.seed)

        # Generate experiment matrix
        self.matrix = generate_matrix(self.config)

        if not self.matrix.runs:
            raise ValueError("No runs generated. Check task and model configuration.")

        # Create output directory
        self.output_dir = self.config.get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create initial manifest
        self.manifest = self._create_manifest()

        # Validate API keys for selected models
        self._validate_api_keys()

        # Validate harness environment
        harness_config = self._build_harness_config()
        harness = HarnessRegistry.get(self.harness_name, harness_config)
        is_valid, errors = harness.validate_environment()
        if not is_valid:
            raise RuntimeError(
                f"Harness '{self.harness_name}' environment validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Save initial matrix
        self._save_matrix()

        logger.info(f"Experiment setup complete: {len(self.matrix)} runs in {self.output_dir}")

        return self.output_dir

    def _create_manifest(self) -> ManifestV1:
        """Create the initial manifest with fingerprints."""
        import platform
        import subprocess
        import sys

        # Get git info
        git_sha = "unknown"
        git_branch = None
        git_dirty = False

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_sha = result.stdout.strip()
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_branch = result.stdout.strip()
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_dirty = bool(result.stdout.strip()) if result.returncode == 0 else False
        except Exception:
            pass

        # Environment fingerprint
        env_fp = EnvironmentFingerprintV1(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            docker_image=self.config.execution.docker_image,
            dependencies={},
        )

        # Config fingerprint
        config_fp = ConfigFingerprintV1(
            llm_config_hash=None,
            benchmark_config_hash=None,
            skill_hash=self._hash_skill() if self.config.skill.is_active() else None,
            skill_name=self.config.skill.get_name(),
        )

        manifest = ManifestV1(
            run_id=self.config.generate_run_id(),
            run_name=self.config.name,
            models=[m.name for m in self.matrix.models] if self.matrix else [],
            task_count=len(self.matrix.tasks) if self.matrix else 0,
            skill_version=self.config.skill.get_name(),
            config_fingerprint=config_fp,
            environment_fingerprint=env_fp,
            git_sha=git_sha,
            git_branch=git_branch,
            git_dirty=git_dirty,
            config=self._get_config_snapshot(),
        )

        return manifest

    def _hash_skill(self) -> str | None:
        """Compute hash of skill content."""
        import hashlib

        content = self.config.get_skill_content()
        if content:
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        return None

    def _get_config_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the configuration for the manifest."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "models": [m.name for m in self.matrix.models] if self.matrix else [],
            "tasks_selection": self.config.tasks.selection.value,
            "repeats": self.config.execution.repeats,
            "timeout": self.config.execution.timeout,
            "seed": self.config.determinism.seed,
            "matrix_hash": self.matrix.config_hash if self.matrix else None,
        }

    def _validate_api_keys(self) -> None:
        """Validate that required API keys are set."""
        if not self.matrix:
            return

        missing_keys = []
        for model_spec in self.matrix.models:
            if model_spec.api_key_env and not os.environ.get(model_spec.api_key_env):
                missing_keys.append((model_spec.name, model_spec.api_key_env))

        if missing_keys:
            for model_name, key_name in missing_keys:
                logger.error(f"Missing API key: {key_name} (required for model: {model_name})")
            raise ValueError(f"Missing required API keys: {[k for _, k in missing_keys]}")

    def _save_matrix(self) -> None:
        """Save the experiment matrix to a file."""
        if not self.output_dir or not self.matrix:
            return

        matrix_path = self.output_dir / "matrix.json"
        matrix_path.write_text(json.dumps(self.matrix.to_dict(), indent=2))
        logger.debug(f"Saved experiment matrix to {matrix_path}")

    def run(self) -> ManifestV1:
        """
        Execute the experiment.

        Runs all tasks in the experiment matrix and produces:
        - results.jsonl: Individual task results
        - summary.json: Aggregated statistics
        - manifest.json: Run metadata and fingerprints

        Returns:
            Finalized ManifestV1
        """
        if not self.matrix or not self.output_dir or not self.manifest:
            raise RuntimeError("Experiment not set up. Call setup() first.")

        start_time = time.time()
        results_file = self.output_dir / "results.jsonl"

        logger.info(f"Starting experiment: {len(self.matrix)} runs")

        # Execute runs
        if self.config.execution.parallel_workers > 1:
            self._run_parallel(results_file)
        else:
            self._run_sequential(results_file)

        # Finalize manifest
        end_time = datetime.now(timezone.utc)
        self.manifest.finalize(end_time)
        self.manifest.results_path = str(results_file)

        # Save artifacts
        self._save_manifest()
        self._save_summary()

        total_time = time.time() - start_time
        logger.info(f"Experiment complete in {total_time:.1f}s")

        return self.manifest

    def _run_sequential(self, results_file: Path) -> None:
        """Run all tasks sequentially."""
        for run in self.matrix.runs:  # type: ignore
            result = self._execute_run(run)
            self._record_result(result, results_file)

    def _run_parallel(self, results_file: Path) -> None:
        """Run tasks in parallel using thread pool."""
        workers = self.config.execution.parallel_workers

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._execute_run, run): run for run in self.matrix.runs}  # type: ignore

            for future in as_completed(futures):
                run = futures[future]
                try:
                    result = future.result()
                    self._record_result(result, results_file)
                except Exception as e:
                    logger.error(f"Run {run.run_index} failed: {e}")
                    # Record error result
                    error_result = self._create_error_result(run, str(e))
                    self._record_result(error_result, results_file)

    def _execute_run(self, run: ExperimentRun) -> ResultV1:
        """Execute a single run with retry logic."""
        max_attempts = self._get_max_attempts()

        for attempt in range(max_attempts):
            try:
                result = self._run_single_evaluation(run)

                # Check if we should retry
                if attempt < max_attempts - 1 and self._should_retry(result):
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Retrying {run.task.task_id}/{run.model.name} "
                        f"(attempt {attempt + 2}/{max_attempts}) after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                return result

            except Exception as e:
                if attempt < max_attempts - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(f"Exception on attempt {attempt + 1}, retrying: {e}")
                    time.sleep(delay)
                    continue
                raise

        # Should not reach here, but return error result if it does
        return self._create_error_result(run, "Max retries exceeded")

    def _run_single_evaluation(self, run: ExperimentRun) -> ResultV1:
        """Run a single evaluation using the harness abstraction layer."""
        start_time = time.time()
        return self._run_single_evaluation_harness(run, start_time)

    def _run_single_evaluation_harness(self, run: ExperimentRun, start_time: float) -> ResultV1:
        """Run a single evaluation using the harness abstraction layer."""
        harness_config = self._build_harness_config()
        harness = HarnessRegistry.get(self.harness_name, harness_config)

        try:
            harness.setup()
            request = self._build_harness_request(run)
            result = self._execute_harness(harness, request)
            return self._convert_harness_result(result, run, time.time() - start_time)
        finally:
            harness.teardown()

    def _execute_harness(self, harness: Any, request: HarnessRequest) -> HarnessResult:
        """Execute harness with simplified async handling."""
        import inspect

        if inspect.iscoroutinefunction(harness.execute):
            return asyncio.run(harness.execute(request))
        else:
            return harness.execute(request)

    def _build_harness_config(self) -> HarnessConfig:
        """Build HarnessConfig from ExperimentConfig."""
        config_dict = {
            "timeout": self.config.execution.timeout,
            "max_retries": self.config.retry.max_retries,
            "retry_delay": self.config.retry.base_delay,
            "max_tokens": None,
            "max_turns": 50,
            "sandbox_enabled": True,
            "network_access": False,
            # Additional harness-specific config (extra="allow")
            "docker_image": self.config.execution.docker_image,
        }
        return HarnessConfig.model_validate(config_dict)

    def _build_harness_request(self, run: ExperimentRun) -> HarnessRequest:
        """Build HarnessRequest from ExperimentRun."""
        # Load task data for context
        task_path = Path(run.task.task_path)
        task_data = {}

        if task_path.exists():
            with contextlib.suppress(Exception):
                task_data = json.loads(task_path.read_text())

        instruction = task_data.get("task", {}).get("instruction", task_data.get("instruction", ""))
        context = task_data.get("task", {}).get("context", task_data.get("context", ""))

        # Get skill content
        skill_content = self.config.get_skill_content()

        return HarnessRequest(
            task_id=run.task.task_id,
            prompt=instruction,
            system_prompt=skill_content,
            timeout=self.config.execution.timeout,
            metadata={
                "context": context,
                "source_package": run.task.source_package or task_data.get("source_package", ""),
                "split": run.task.split,
                "difficulty": run.task.difficulty,
                "task_type": run.task.task_type,
                "model": run.model.litellm_model,
                "skill_content": skill_content,
                # Include SWE-bench style fields if present
                "test_patch": task_data.get("test_patch", ""),
                "gold_patch": task_data.get("patch", task_data.get("gold_patch", "")),
                "tests": task_data.get("tests", {}),
                "fail_to_pass": task_data.get("fail_to_pass", []),
                "pass_to_pass": task_data.get("pass_to_pass", []),
                "source": task_data.get("source", {}),
                # Include run metadata
                "run_index": run.run_index,
                "repeat_index": run.repeat_index,
                "fingerprint": run.fingerprint,
                "support_profile": run.support_profile,
                "pair_id": run.pair_id,
                "pair_role": run.pair_role,
            },
        )

    def _convert_harness_result(
        self, result: HarnessResult, run: ExperimentRun, latency: float
    ) -> ResultV1:
        """Convert HarnessResult to ResultV1."""
        # Map error categories from harness to schema v1
        error_category_map = {
            HarnessErrorCategory.NONE: None,
            HarnessErrorCategory.TIMEOUT: ErrorCategoryV1.TIMEOUT,
            HarnessErrorCategory.API_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.RATE_LIMIT: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.AUTH_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.NETWORK_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.INVALID_REQUEST: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.AGENT_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.SANDBOX_ERROR: ErrorCategoryV1.SANDBOX_ERROR,
            HarnessErrorCategory.RESOURCE_EXHAUSTED: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.MODEL_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.TASK_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.TEST_ERROR: ErrorCategoryV1.TEST_FAILURE,
            HarnessErrorCategory.UNKNOWN: ErrorCategoryV1.UNKNOWN,
        }
        error_category = error_category_map.get(result.error_category, ErrorCategoryV1.UNKNOWN)

        # Convert test results
        test_results = [
            CaseResultV1(
                name=tr.name,
                passed=tr.passed,
                message=tr.message,
                execution_time=tr.execution_time,
            )
            for tr in result.test_results
        ]

        # Token usage
        token_usage = TokenUsageV1(
            prompt=result.token_usage.prompt,
            completion=result.token_usage.completion,
            total=result.token_usage.total,
        )

        # Save trajectory if enabled
        trajectory_path = None
        if self.config.execution.save_trajectories and result.patch and self.output_dir:
            traj_dir = self.output_dir / "trajectories" / run.model.name
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_file = traj_dir / f"{run.task.task_id}.txt"
            traj_file.write_text(result.patch)
            trajectory_path = str(traj_file.relative_to(self.output_dir))

        # Extract telemetry from metadata
        metadata = result.metadata or {}
        tool_call_counts = metadata.get("tool_call_counts", {})
        tool_errors = metadata.get("tool_errors", {})
        tool_total_time_ms = metadata.get("tool_total_time_ms", {})

        return ResultV1(
            result_id=f"{run.fingerprint}",
            task_id=run.task.task_id,
            model=run.model.name,
            profile_id=run.support_profile or self.config.skill.get_name(),
            passed=result.success,
            score=1.0 if result.success else 0.0,
            error_category=error_category,
            error_message=result.error_message,
            latency_s=latency,
            execution_time=result.execution_time,
            token_usage=token_usage,
            test_results=test_results,
            generated_code=result.patch,
            trajectory_path=trajectory_path,
            repeat_index=run.repeat_index,
            metadata={
                "fingerprint": run.fingerprint,
                "run_index": run.run_index,
                "support_profile": run.support_profile,
                "pair_id": run.pair_id,
                "pair_role": run.pair_role,
                "harness_run_id": result.run_id,
            },
            tool_call_counts=tool_call_counts,
            tool_errors=tool_errors,
            tool_total_time_ms=tool_total_time_ms,
        )

    def _create_error_result(self, run: ExperimentRun, error_message: str) -> ResultV1:
        """Create an error result for a failed run."""
        return ResultV1(
            result_id=f"{run.fingerprint}",
            task_id=run.task.task_id,
            model=run.model.name,
            profile_id=run.support_profile or self.config.skill.get_name(),
            passed=False,
            score=0.0,
            error_category=ErrorCategoryV1.UNKNOWN,
            error_message=error_message,
            latency_s=0.0,
            execution_time=0.0,
            token_usage=TokenUsageV1(),
            test_results=[],
            repeat_index=run.repeat_index,
            metadata={
                "fingerprint": run.fingerprint,
                "run_index": run.run_index,
                "support_profile": run.support_profile,
                "pair_id": run.pair_id,
                "pair_role": run.pair_role,
            },
            # Empty telemetry for error results
            tool_call_counts={},
            tool_errors={},
            tool_total_time_ms={},
        )

    def _get_max_attempts(self) -> int:
        """Get maximum number of attempts based on retry config."""
        if self.config.retry.strategy == RetryStrategy.NONE:
            return 1
        return self.config.retry.max_retries + 1

    def _should_retry(self, result: ResultV1) -> bool:
        """Check if we should retry based on result."""
        if result.passed:
            return False

        if result.error_category:
            return result.error_category.value in self.config.retry.retry_on

        return False

    def _get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry attempt."""
        if self.config.retry.strategy == RetryStrategy.FIXED:
            return self.config.retry.base_delay
        elif self.config.retry.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.retry.base_delay * (2**attempt)
            return min(delay, self.config.retry.max_delay)
        return 0.0

    def _record_result(self, result: ResultV1, results_file: Path) -> None:
        """Record a result to the results file and manifest."""
        self.results.append(result)
        if self.manifest:
            self.manifest.add_result(result)

        # Append to results.jsonl
        with open(results_file, "a") as f:
            f.write(json.dumps(result.model_dump(mode="json")) + "\n")

        # Save intermediate manifest if enabled
        if self.config.output.save_intermediate and self.manifest and self.output_dir:
            self._save_manifest()

    def _save_manifest(self) -> None:
        """Save the manifest to a file."""
        if not self.output_dir or not self.manifest:
            return

        manifest_path = self.output_dir / "manifest.json"
        self.manifest.save(str(manifest_path))
        logger.debug(f"Saved manifest to {manifest_path}")

    def _save_summary(self) -> None:
        """Save the summary to a file."""
        if not self.output_dir or not self.manifest:
            return

        summary = ResultSummaryV1(
            total_tasks=self.manifest.task_count,
            completed=len(self.results),
            passed=sum(1 for r in self.results if r.passed),
            failed=sum(1 for r in self.results if not r.passed),
            errors=sum(1 for r in self.results if r.error_category),
            avg_score=sum(r.score for r in self.results) / len(self.results)
            if self.results
            else 0.0,
            avg_latency_s=sum(r.latency_s for r in self.results) / len(self.results)
            if self.results
            else 0.0,
            total_tokens=sum(r.token_usage.total for r in self.results),
        )

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2))
        logger.debug(f"Saved summary to {summary_path}")


def run_experiment(config: ExperimentConfig) -> ManifestV1:
    """
    Convenience function to run an experiment.

    Args:
        config: Experiment configuration

    Returns:
        Finalized manifest
    """
    runner = ExperimentRunner(config)
    runner.setup()
    return runner.run()
