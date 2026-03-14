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
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Lock
from typing import Any

from bench.experiments.config import ExperimentConfig, RetryStrategy
from bench.experiments.matrix import ExperimentMatrix, ExperimentRun, generate_matrix
from bench.harness import (
    ErrorCategory as HarnessErrorCategory,
)
from bench.harness import (
    HarnessConfig,
    HarnessRegistry,
    HarnessRequest,
    HarnessResult,
)
from bench.schema.v1 import (
    CaseResultV1,
    ConfigFingerprintV1,
    EnvironmentFingerprintV1,
    ErrorCategoryV1,
    ManifestV1,
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

    _RATE_LIMIT_BASE_COOLDOWN_S = 60.0
    _RATE_LIMIT_MAX_COOLDOWN_S = 3600.0
    _RATE_LIMIT_JITTER_RATIO = 0.2

    def __init__(
        self,
        config: ExperimentConfig,
        harness_name: str | None = None,
        output_dir: Path | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            harness_name: Override harness name (default: from config.execution.harness).
                          Priority: explicit param > config.execution.harness > "pi_docker".
            output_dir: Override output directory
            progress_callback: Optional callback for progress updates

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
        self._memory_saving = True  # Enable memory saving by default

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

        # Rate-limit tracking for models
        self._rate_limited_models: dict[str, float] = {}  # model -> cooldown_until_timestamp
        self._rate_limited_providers: dict[str, float] = {}  # provider -> cooldown_until_timestamp
        self._rate_limit_lock = Lock()
        self._provider_rate_limit_streaks: dict[str, int] = {}

        # Provider pacing state (run-start throttling)
        self._provider_next_run_at: dict[str, float] = {}
        self._provider_pacing_lock = Lock()

        # Progress callback
        self._progress_callback = progress_callback

    def _emit_progress(self, event: dict[str, Any]) -> None:
        """Emit a progress event without affecting experiment execution."""
        if not self._progress_callback:
            return
        with contextlib.suppress(Exception):
            self._progress_callback(event)

    def setup(self) -> Path:
        """
        Set up the experiment run.

        - Creates output directory
        - Generates experiment matrix
        - Creates initial manifest
        - Validates environment
        - Runs provider preflight validation

        Returns:
            Path to output directory

        Raises:
            RuntimeError: If harness or credential validation fails
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

        # Validate provider credentials (preflight)
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

        from bench.provider import get_provider_resolver

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

        # Get unique models and resolve providers
        models_used = list({run.model.name for run in self.matrix.runs}) if self.matrix else []
        resolver = get_provider_resolver()
        providers_metadata = {}
        for model in models_used:
            try:
                provider = resolver.resolve_provider(model)
                providers_metadata[model] = provider
            except KeyError:
                providers_metadata[model] = "unknown"

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
            metadata={
                "execution": {
                    "harness": self.harness_name,
                    "sandbox_profile": self.config.execution.sandbox_profile.value,
                    "auth_policy": self.config.execution.auth_policy.value,
                },
                "providers": providers_metadata,
            },
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
            "parallel_workers": self.config.execution.parallel_workers,
            "provider_parallel_limits": dict(self.config.execution.provider_parallel_limits),
            "provider_min_interval_s": dict(self.config.execution.provider_min_interval_s),
            "provider_max_requests_per_second": dict(
                self.config.execution.provider_max_requests_per_second
            ),
            "seed": self.config.determinism.seed,
            "matrix_hash": self.matrix.config_hash if self.matrix else None,
        }

    def _validate_api_keys(self) -> None:
        """Validate provider credentials using preflight validation.

        Raises:
            RuntimeError: If credential validation fails
        """
        if not self.matrix:
            return

        from bench.provider import AuthPolicy, run_preflight

        # Get all unique models from the matrix
        models = list({run.model.name for run in self.matrix.runs})

        # Determine auth policy from config
        auth_policy = AuthPolicy.ENV  # Default
        if hasattr(self.config.execution, "auth_policy"):
            auth_policy = AuthPolicy(self.config.execution.auth_policy)

        # Run preflight validation
        preflight_result = run_preflight(
            models=models,
            auth_policy=auth_policy,
            strict=True,  # Fail on missing credentials
        )

        if not preflight_result.is_valid:
            error_msg = "Preflight validation failed:\n" + "\n".join(
                f"  - {e}" for e in preflight_result.errors
            )
            raise RuntimeError(error_msg)

        # Log warnings
        for warning in preflight_result.warnings:
            logger.warning(f"Preflight warning: {warning}")

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

        model_totals: dict[str, int] = {}
        for run in self.matrix.runs:
            model_totals[run.model.name] = model_totals.get(run.model.name, 0) + 1
        self._emit_progress(
            {
                "type": "start",
                "total_runs": len(self.matrix),
                "models": sorted(model_totals.keys()),
                "model_totals": model_totals,
                "task_count": len({run.task.task_id for run in self.matrix.runs}),
            }
        )

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
        runs = self._interleave_runs_by_model(self.matrix.runs)  # type: ignore[arg-type]
        for run in runs:
            self._emit_progress(
                {"type": "run_start", "model": run.model.name, "task_id": run.task.task_id}
            )
            result = self._execute_run(run)
            self._record_result(result, results_file)

    def _interleave_runs_by_model(self, runs: list[ExperimentRun]) -> list[ExperimentRun]:
        """Interleave runs by model in round-robin order.

        This surfaces model-specific failures earlier and reduces bursty pressure
        on any single model/provider while preserving deterministic ordering.
        """
        model_runs: dict[str, list[ExperimentRun]] = {}
        for run in runs:
            model = run.model.name
            if model not in model_runs:
                model_runs[model] = []
            model_runs[model].append(run)

        if len(model_runs) <= 1:
            return runs

        preferred_order: list[str] = []
        if self.matrix and self.matrix.models:
            preferred_order = [m.name for m in self.matrix.models]

        # Preserve model order from matrix config; append any unexpected models deterministically.
        ordered_models = [m for m in preferred_order if m in model_runs]
        ordered_models.extend(sorted(m for m in model_runs if m not in set(ordered_models)))

        interleaved: list[ExperimentRun] = []
        max_len = max(len(rs) for rs in model_runs.values())
        for i in range(max_len):
            for model in ordered_models:
                rs = model_runs[model]
                if i < len(rs):
                    interleaved.append(rs[i])

        return interleaved

    def _interleave_runs_by_provider(self, runs: list) -> list:
        """Interleave runs by provider to prevent semaphore starvation.

        When runs are grouped by provider, workers can all block on the same
        provider's semaphore while tasks from other providers sit idle. By
        interleaving runs, we ensure workers blocking on one provider don't
        prevent other providers' tasks from executing.
        """
        # Group runs by provider
        provider_runs: dict[str, list] = {}
        for run in runs:
            provider = self._resolve_run_provider(run)
            if provider not in provider_runs:
                provider_runs[provider] = []
            provider_runs[provider].append(run)

        if len(provider_runs) <= 1:
            return runs

        # Interleave in round-robin fashion
        interleaved = []
        max_len = max(len(r) for r in provider_runs.values())
        providers = list(provider_runs.keys())
        # Sort providers for deterministic ordering
        providers.sort()

        for i in range(max_len):
            for provider in providers:
                runs_for_provider = provider_runs[provider]
                if i < len(runs_for_provider):
                    interleaved.append(runs_for_provider[i])

        return interleaved

    def _run_parallel(self, results_file: Path) -> None:
        """Run tasks in parallel using thread pool with rate-limit aware scheduling."""
        workers = self.config.execution.parallel_workers
        provider_limits = self._get_provider_parallel_limits(workers)
        provider_semaphores = {
            provider: BoundedSemaphore(limit) for provider, limit in provider_limits.items()
        }

        if provider_limits:
            logger.info(
                "Provider concurrency limits enabled: %s",
                ", ".join(
                    f"{provider}={limit}" for provider, limit in sorted(provider_limits.items())
                ),
            )

        provider_pacing = {
            provider: self._get_provider_min_interval_s(provider)
            for provider in set(self.config.execution.provider_min_interval_s).union(
                self.config.execution.provider_max_requests_per_second
            )
        }
        provider_pacing = {p: s for p, s in provider_pacing.items() if s > 0}
        if provider_pacing:
            logger.info(
                "Provider pacing enabled (min run-start interval): %s",
                ", ".join(
                    f"{provider}={interval:.2f}s"
                    for provider, interval in sorted(provider_pacing.items())
                ),
            )

        # Interleave by model first, then by provider to avoid model starvation while
        # still preventing provider semaphore starvation.
        runs = self._interleave_runs_by_model(self.matrix.runs)  # type: ignore[arg-type]
        runs = self._interleave_runs_by_provider(runs)

        # Use a queue for dynamic task scheduling
        pending: Queue = Queue()
        for run in runs:
            pending.put(run)

        total = len(runs)
        completed = 0

        # Thread-safe counter for progress tracking
        completed_lock = Lock()

        def get_next_run() -> ExperimentRun | None:
            """Get next non-rate-limited task from queue.

            Returns None if queue is empty. If all remaining tasks are rate-limited,
            waits for the shortest cooldown and returns that task.
            """
            skipped: list[ExperimentRun] = []

            while not pending.empty():
                try:
                    run = pending.get_nowait()
                except Exception:
                    break

                model = run.model.name
                provider = self._resolve_run_provider(run)
                if self._is_model_rate_limited(model) or self._is_provider_rate_limited(provider):
                    skipped.append(run)
                else:
                    # Found a non-rate-limited task; put skipped items back at the front
                    for r in reversed(skipped):
                        pending.queue.insert(0, r)
                    return run

            # All remaining tasks are rate-limited
            if skipped:
                # Find the task with shortest remaining cooldown
                shortest_wait = min(
                    (r for r in skipped),
                    key=self._get_run_rate_limit_remaining,
                )
                wait_time = self._get_run_rate_limit_remaining(shortest_wait)

                if wait_time > 0:
                    logger.info(
                        f"All pending tasks rate-limited, waiting {wait_time:.1f}s for "
                        f"model '{shortest_wait.model.name}'"
                    )
                    time.sleep(wait_time)

                # Put all skipped back and return the one with shortest wait
                for r in skipped:
                    if r is not shortest_wait:
                        pending.put(r)
                return shortest_wait

            return None

        def execute_with_rate_limit_awareness(
            run: ExperimentRun,
        ) -> tuple[ResultV1, ExperimentRun]:
            """Execute a run and return result with run for tracking."""
            self._emit_progress(
                {"type": "run_start", "model": run.model.name, "task_id": run.task.task_id}
            )
            result = self._execute_run_with_provider_limit(run, provider_semaphores)
            return result, run

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict = {}

            # Submit initial batch of tasks (up to worker count)
            for _ in range(min(workers, total)):
                run = get_next_run()
                if run is None:
                    break
                futures[executor.submit(execute_with_rate_limit_awareness, run)] = run

            # Process completed tasks and submit new ones dynamically
            while futures:
                for future in as_completed(futures):
                    run = futures.pop(future)
                    try:
                        result, _completed_run = future.result()
                        self._record_result(result, results_file)
                    except Exception as e:
                        logger.error(f"Run {run.run_index} failed: {e}")
                        error_result = self._create_error_result(run, str(e))
                        self._record_result(error_result, results_file)

                    with completed_lock:
                        completed += 1

                    # Submit next task if available
                    next_run = get_next_run()
                    if next_run is not None:
                        futures[executor.submit(execute_with_rate_limit_awareness, next_run)] = (
                            next_run
                        )

                    # Break out of the for loop to restart as_completed
                    break
                else:
                    # as_completed exhausted but futures not empty - shouldn't happen
                    break

    def _get_provider_parallel_limits(self, worker_limit: int) -> dict[str, int]:
        """Normalize provider limits and clamp to the global worker limit."""
        limits = self.config.execution.provider_parallel_limits
        if not limits:
            return {}
        return {
            provider: min(limit, worker_limit) for provider, limit in limits.items() if limit >= 1
        }

    def _resolve_run_provider(self, run: ExperimentRun) -> str:
        """Resolve provider name for a run, with fallback for raw model IDs."""
        if run.model.provider:
            return run.model.provider.lower()
        if "/" in run.model.litellm_model:
            return run.model.litellm_model.split("/", 1)[0].lower()
        return "unknown"

    def _get_provider_min_interval_s(self, provider: str) -> float:
        """Get effective minimum spacing between run starts for a provider."""
        cfg = self.config.execution
        min_interval = cfg.provider_min_interval_s.get(provider, 0.0)
        rps = cfg.provider_max_requests_per_second.get(provider)
        if rps and rps > 0:
            min_interval = max(min_interval, 1.0 / rps)
        return max(0.0, min_interval)

    def _throttle_provider_run_start(self, provider: str) -> None:
        """Throttle run starts for a provider using configured pacing limits."""
        min_interval = self._get_provider_min_interval_s(provider)
        if min_interval <= 0:
            return

        while True:
            wait_time = 0.0
            with self._provider_pacing_lock:
                now = time.monotonic()
                next_allowed = self._provider_next_run_at.get(provider, 0.0)
                if now >= next_allowed:
                    self._provider_next_run_at[provider] = now + min_interval
                    return
                wait_time = next_allowed - now

            if wait_time > 0:
                time.sleep(wait_time)

    def _mark_model_rate_limited(
        self, model_name: str, cooldown_seconds: float | None = None
    ) -> None:
        """Mark a model as rate-limited with a cooldown period."""
        if cooldown_seconds is None:
            cooldown_seconds = self.config.execution.rate_limit_cooldown_s
        cooldown_until = time.time() + cooldown_seconds
        with self._rate_limit_lock:
            existing = self._rate_limited_models.get(model_name, 0.0)
            self._rate_limited_models[model_name] = max(existing, cooldown_until)
            logger.warning(
                f"Model '{model_name}' rate-limited, cooling down for {cooldown_seconds:.0f}s"
            )
        self._emit_progress(
            {"type": "model_sleep", "model": model_name, "cooldown_s": cooldown_seconds}
        )

    def _mark_provider_rate_limited(
        self, provider: str, cooldown_seconds: float | None = None
    ) -> None:
        """Mark a provider as rate-limited with a cooldown period."""
        if cooldown_seconds is None:
            cooldown_seconds = self.config.execution.rate_limit_cooldown_s
        cooldown_until = time.time() + cooldown_seconds
        with self._rate_limit_lock:
            existing = self._rate_limited_providers.get(provider, 0.0)
            self._rate_limited_providers[provider] = max(existing, cooldown_until)
            logger.warning(
                f"Provider '{provider}' rate-limited, cooling down for {cooldown_seconds:.0f}s"
            )
        self._emit_progress(
            {"type": "provider_sleep", "provider": provider, "cooldown_s": cooldown_seconds}
        )

    def _is_model_rate_limited(self, model_name: str) -> bool:
        """Check if a model is still in cooldown."""
        with self._rate_limit_lock:
            if model_name not in self._rate_limited_models:
                return False
            if time.time() > self._rate_limited_models[model_name]:
                # Cooldown expired
                del self._rate_limited_models[model_name]
                logger.info(f"Model '{model_name}' cooldown expired, resuming tasks")
                self._emit_progress({"type": "model_resume", "model": model_name})
                return False
            return True

    def _is_provider_rate_limited(self, provider: str) -> bool:
        """Check if a provider is still in cooldown."""
        with self._rate_limit_lock:
            if provider not in self._rate_limited_providers:
                return False
            if time.time() > self._rate_limited_providers[provider]:
                del self._rate_limited_providers[provider]
                logger.info(f"Provider '{provider}' cooldown expired, resuming tasks")
                self._emit_progress({"type": "provider_resume", "provider": provider})
                return False
            return True

    def _get_rate_limit_remaining(self, model_name: str) -> float:
        """Get remaining cooldown time for a rate-limited model, or 0 if not limited."""
        with self._rate_limit_lock:
            if model_name not in self._rate_limited_models:
                return 0.0
            remaining = self._rate_limited_models[model_name] - time.time()
            return max(0.0, remaining)

    def _get_provider_rate_limit_remaining(self, provider: str) -> float:
        """Get remaining cooldown time for a rate-limited provider, or 0 if not limited."""
        with self._rate_limit_lock:
            if provider not in self._rate_limited_providers:
                return 0.0
            remaining = self._rate_limited_providers[provider] - time.time()
            return max(0.0, remaining)

    def _get_run_rate_limit_remaining(self, run: ExperimentRun) -> float:
        """Get remaining cooldown for a run from either model or provider cooldowns."""
        provider = self._resolve_run_provider(run)
        return max(
            self._get_rate_limit_remaining(run.model.name),
            self._get_provider_rate_limit_remaining(provider),
        )

    def _extract_retry_after_seconds(self, error_message: str | None) -> float | None:
        """Extract retry-after hint in seconds from provider error messages."""
        if not error_message:
            return None
        text = error_message.lower()
        patterns = (
            r"retry[-_\s]?after[^\d]*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|m|min|minute|minutes|h|hr|hour|hours)?",
            r"try again in[^\d]*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|m|min|minute|minutes|h|hr|hour|hours)?",
            r"reset(?:s|ting)? in[^\d]*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|m|min|minute|minutes|h|hr|hour|hours)?",
        )

        unit_scale = {
            None: 1.0,
            "ms": 0.001,
            "s": 1.0,
            "sec": 1.0,
            "secs": 1.0,
            "second": 1.0,
            "seconds": 1.0,
            "m": 60.0,
            "min": 60.0,
            "minute": 60.0,
            "minutes": 60.0,
            "h": 3600.0,
            "hr": 3600.0,
            "hour": 3600.0,
            "hours": 3600.0,
        }

        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            seconds = float(match.group(1)) * unit_scale.get(match.group(2), 1.0)
            return max(0.0, seconds)

        return None

    def _compute_rate_limit_cooldown_s(self, provider: str, error_message: str | None) -> float:
        """Compute cooldown after rate limiting with Retry-After support and backoff."""
        retry_after_s = self._extract_retry_after_seconds(error_message)
        if retry_after_s is not None:
            return min(max(1.0, retry_after_s), self._RATE_LIMIT_MAX_COOLDOWN_S)

        with self._rate_limit_lock:
            streak = self._provider_rate_limit_streaks.get(provider, 0) + 1
            self._provider_rate_limit_streaks[provider] = streak

        cooldown = min(
            self._RATE_LIMIT_BASE_COOLDOWN_S * (2 ** (streak - 1)),
            self._RATE_LIMIT_MAX_COOLDOWN_S,
        )
        jitter = 1.0 + random.uniform(-self._RATE_LIMIT_JITTER_RATIO, self._RATE_LIMIT_JITTER_RATIO)
        return max(1.0, cooldown * jitter)

    def _record_successful_provider_attempt(self, provider: str) -> None:
        """Gradually decay provider rate-limit streak after successful attempts."""
        with self._rate_limit_lock:
            streak = self._provider_rate_limit_streaks.get(provider, 0)
            if streak <= 1:
                self._provider_rate_limit_streaks.pop(provider, None)
            else:
                self._provider_rate_limit_streaks[provider] = streak - 1

    def _execute_run_with_provider_limit(
        self,
        run: ExperimentRun,
        provider_semaphores: dict[str, BoundedSemaphore],
    ) -> ResultV1:
        """Execute a run while enforcing per-provider concurrency and pacing."""
        provider = self._resolve_run_provider(run)
        provider_sem = provider_semaphores.get(provider)
        if provider_sem is None:
            self._throttle_provider_run_start(provider)
            return self._execute_run(run)

        with provider_sem:
            self._throttle_provider_run_start(provider)
            return self._execute_run(run)

    def _execute_run(self, run: ExperimentRun) -> ResultV1:
        """Execute a single run with retry logic."""
        max_attempts = self._get_max_attempts()
        provider = self._resolve_run_provider(run)

        for attempt in range(max_attempts):
            try:
                result = self._run_single_evaluation(run)

                # Check for rate limit from harness
                harness_error = getattr(result, "_harness_error_category", None)
                if harness_error == HarnessErrorCategory.RATE_LIMIT:
                    cooldown_s = self._compute_rate_limit_cooldown_s(provider, result.error_message)
                    self._mark_model_rate_limited(run.model.name, cooldown_s)
                    self._mark_provider_rate_limited(provider, cooldown_s)
                else:
                    self._record_successful_provider_attempt(provider)

                # Check if we should retry
                if attempt < max_attempts - 1 and self._should_retry(result):
                    delay = self._get_retry_delay(attempt)
                    if harness_error == HarnessErrorCategory.RATE_LIMIT:
                        delay = max(
                            delay,
                            self._get_rate_limit_remaining(run.model.name),
                            self._get_provider_rate_limit_remaining(provider),
                        )
                    logger.warning(
                        f"Retrying {run.task.task_id}/{run.model.name} "
                        f"(attempt {attempt + 2}/{max_attempts}) after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                return result

            except Exception as e:
                if HarnessErrorCategory.from_exception(e) == HarnessErrorCategory.RATE_LIMIT:
                    cooldown_s = self._compute_rate_limit_cooldown_s(provider, str(e))
                    self._mark_model_rate_limited(run.model.name, cooldown_s)
                    self._mark_provider_rate_limited(provider, cooldown_s)
                if attempt < max_attempts - 1:
                    delay = self._get_retry_delay(attempt)
                    delay = max(
                        delay,
                        self._get_rate_limit_remaining(run.model.name),
                        self._get_provider_rate_limit_remaining(provider),
                    )
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
            converted = self._convert_harness_result(result, run, time.time() - start_time)

            # Store original harness error category for rate-limit detection
            converted._harness_error_category = result.error_category  # type: ignore

            return converted
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
            "sandbox_profile": self.config.execution.sandbox_profile.value,
            "forward_github_token": self.config.execution.forward_github_token,
            "auth_policy": self.config.execution.auth_policy.value,
            "keep_workspace_on_failure": self.config.execution.keep_workspace_on_failure,
            "save_traces": self.config.execution.save_traces,
            "trace_dir": str(self.output_dir / "traces") if self.output_dir else "container_logs",
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

        instruction = self._extract_task_instruction(task_data)
        context = self._extract_task_context(task_data)

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
                "problem_statement": task_data.get("problem_statement", {}),
                "grading": task_data.get("grading", {}),
                # Include run metadata
                "run_index": run.run_index,
                "repeat_index": run.repeat_index,
                "fingerprint": run.fingerprint,
                "support_profile": run.support_profile,
                "pair_id": run.pair_id,
                "pair_role": run.pair_role,
            },
        )

    @staticmethod
    def _first_non_empty_text(*values: Any) -> str:
        """Return first non-empty string value, trimmed."""
        for value in values:
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        return ""

    def _extract_task_instruction(self, task_data: dict[str, Any]) -> str:
        """Extract instruction text from heterogeneous task schemas."""
        task_payload = task_data.get("task")
        task_payload = task_payload if isinstance(task_payload, dict) else {}
        problem_statement = task_data.get("problem_statement")
        problem_statement = problem_statement if isinstance(problem_statement, dict) else {}

        return self._first_non_empty_text(
            task_payload.get("instruction"),
            task_data.get("instruction"),
            task_payload.get("prompt"),
            task_data.get("prompt"),
            problem_statement.get("instruction"),
            problem_statement.get("description"),
        )

    def _extract_task_context(self, task_data: dict[str, Any]) -> str:
        """Extract context text from heterogeneous task schemas."""
        task_payload = task_data.get("task")
        task_payload = task_payload if isinstance(task_payload, dict) else {}
        problem_statement = task_data.get("problem_statement")
        problem_statement = problem_statement if isinstance(problem_statement, dict) else {}

        return self._first_non_empty_text(
            task_payload.get("context"),
            task_data.get("context"),
            problem_statement.get("context"),
            problem_statement.get("description"),
        )

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        """Parse a score-like value into float when possible."""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                return float(value.strip())
        return None

    def _extract_harness_score(self, result: HarnessResult) -> tuple[float, str, float | None]:
        """Extract score from harness result, preserving non-binary task scoring."""
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        raw_result = metadata.get("raw_result")
        raw_result = raw_result if isinstance(raw_result, dict) else {}
        raw_response = metadata.get("raw_response")
        raw_response = raw_response if isinstance(raw_response, dict) else {}

        candidates: list[tuple[str, Any]] = [
            ("result.score", getattr(result, "score", None)),
            ("metadata.score", metadata.get("score")),
            ("metadata.normalized_score", metadata.get("normalized_score")),
            ("metadata.raw_score", metadata.get("raw_score")),
            ("metadata.raw_result.score", raw_result.get("score")),
            ("metadata.raw_result.normalized_score", raw_result.get("normalized_score")),
            ("metadata.raw_result.raw_score", raw_result.get("raw_score")),
            ("metadata.raw_response.score", raw_response.get("score")),
            ("metadata.raw_response.normalized_score", raw_response.get("normalized_score")),
            ("metadata.raw_response.raw_score", raw_response.get("raw_score")),
        ]

        for source, raw_value in candidates:
            score = self._coerce_score(raw_value)
            if score is None:
                continue
            normalized_score = min(max(score, 0.0), 1.0)
            if normalized_score != score:
                logger.debug(
                    "Clamped out-of-range score %.4f from %s to %.4f",
                    score,
                    source,
                    normalized_score,
                )
            return normalized_score, source, score

        fallback = 1.0 if result.success else 0.0
        return fallback, "pass_fail_fallback", None

    def _extract_harness_score_tier(self, result: HarnessResult) -> str | None:
        """Extract optional tier label from harness metadata."""
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        raw_result = metadata.get("raw_result")
        raw_result = raw_result if isinstance(raw_result, dict) else {}
        raw_response = metadata.get("raw_response")
        raw_response = raw_response if isinstance(raw_response, dict) else {}

        tier_value = self._first_non_empty_text(
            metadata.get("score_tier"),
            metadata.get("tier"),
            raw_result.get("score_tier"),
            raw_result.get("tier"),
            raw_result.get("grade"),
            raw_response.get("score_tier"),
            raw_response.get("tier"),
            raw_response.get("grade"),
        )
        return tier_value or None

    def _convert_harness_result(
        self, result: HarnessResult, run: ExperimentRun, latency: float
    ) -> ResultV1:
        """Convert HarnessResult to ResultV1."""
        # Map error categories from harness to schema v1
        # Transient API/provider/network errors -> LLM_ERROR (retriable by default)
        # Auth errors -> AUTH_ERROR (NOT retriable)
        # Unknown errors -> UNKNOWN (NOT retriable)
        error_category_map = {
            HarnessErrorCategory.NONE: None,
            HarnessErrorCategory.TIMEOUT: ErrorCategoryV1.TIMEOUT,
            # Transient API/provider errors -> LLM_ERROR (retriable)
            HarnessErrorCategory.API_ERROR: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.RATE_LIMIT: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.NETWORK_ERROR: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.MODEL_ERROR: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.RESOURCE_EXHAUSTED: ErrorCategoryV1.LLM_ERROR,
            # Auth errors -> AUTH_ERROR (NOT retriable)
            HarnessErrorCategory.AUTH_ERROR: ErrorCategoryV1.AUTH_ERROR,
            # Sandbox errors -> SANDBOX_ERROR (retriable)
            HarnessErrorCategory.SANDBOX_ERROR: ErrorCategoryV1.SANDBOX_ERROR,
            # Test failures -> TEST_FAILURE (NOT retriable)
            HarnessErrorCategory.TEST_ERROR: ErrorCategoryV1.TEST_FAILURE,
            # Invalid requests/agent/task errors -> UNKNOWN (NOT retriable)
            HarnessErrorCategory.INVALID_REQUEST: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.AGENT_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.TASK_ERROR: ErrorCategoryV1.UNKNOWN,
            # Fallback for unexpected errors
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

        container_log_path = None
        if self.config.execution.save_container_logs and self.output_dir:
            log_dir = self.output_dir / "container_logs" / run.model.name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{run.task.task_id}.log"
            log_file.write_text(
                self._format_container_log(
                    task_id=run.task.task_id,
                    model=run.model.name,
                    output=result.output,
                    error_message=result.error_message,
                )
            )
            container_log_path = str(log_file.relative_to(self.output_dir))

        # Extract telemetry from canonical telemetry schema
        tool_call_counts: dict[str, int] = {}
        tool_errors: dict[str, int] = {}
        tool_total_time_ms: dict[str, float] = {}
        if result.telemetry is not None:
            tool_call_counts = dict(result.telemetry.tools.by_tool)
            tool_errors = dict(result.telemetry.tools.errors)
            tool_total_time_ms = dict(result.telemetry.tools.duration_ms_by_tool)

        score, score_source, raw_score = self._extract_harness_score(result)
        score_tier = self._extract_harness_score_tier(result)

        result_metadata: dict[str, Any] = {
            "fingerprint": run.fingerprint,
            "run_index": run.run_index,
            "support_profile": run.support_profile,
            "pair_id": run.pair_id,
            "pair_role": run.pair_role,
            "harness_run_id": result.run_id,
        }
        if container_log_path:
            result_metadata["container_log_path"] = container_log_path
        if score_source != "pass_fail_fallback":
            result_metadata["harness_score_source"] = score_source
        if raw_score is not None:
            result_metadata["harness_score_raw"] = raw_score
        if score_tier is not None:
            result_metadata["harness_score_tier"] = score_tier

        return ResultV1(
            result_id=f"{run.fingerprint}",
            task_id=run.task.task_id,
            model=run.model.name,
            profile_id=run.support_profile or self.config.skill.get_name(),
            passed=result.success,
            score=score,
            error_category=error_category,
            error_message=result.error_message,
            latency_s=latency,
            execution_time=result.execution_time,
            token_usage=token_usage,
            test_results=test_results,
            generated_code=result.patch,
            trajectory_path=trajectory_path,
            repeat_index=run.repeat_index,
            metadata=result_metadata,
            telemetry=result.telemetry,
            tool_call_counts=tool_call_counts,
            tool_errors=tool_errors,
            tool_total_time_ms=tool_total_time_ms,
        )

    def _format_container_log(
        self,
        *,
        task_id: str,
        model: str,
        output: str | None,
        error_message: str | None,
    ) -> str:
        """Format raw container output for persisted debug logs."""
        sections = [
            f"task_id: {task_id}",
            f"model: {model}",
            "",
            "=== STDOUT ===",
            output or "",
            "",
            "=== STDERR/ERROR ===",
            error_message or "",
            "",
        ]
        return "\n".join(sections)

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
            telemetry=None,
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
        # Append to results.jsonl FIRST (always full data)
        with open(results_file, "a") as f:
            f.write(json.dumps(result.model_dump(mode="json")) + "\n")

        # Add to manifest (updates summaries)
        if self.manifest:
            self.manifest.add_result(result, keep_in_memory=not self._memory_saving)

        # Keep a list of results in runner ONLY if memory saving is disabled
        if not self._memory_saving:
            self.results.append(result)

        self._emit_progress(
            {
                "type": "result",
                "model": result.model,
                "task_id": result.task_id,
                "passed": result.passed,
                "score": result.score,
                "tokens": result.token_usage.total if result.token_usage else 0,
                "prompt_tokens": result.token_usage.prompt if result.token_usage else 0,
                "completion_tokens": result.token_usage.completion if result.token_usage else 0,
                "latency_s": result.latency_s,
                "runtime_s": result.execution_time or result.latency_s,
                "tool_call_counts": result.tool_call_counts,
                "tool_errors": result.tool_errors,
                "tool_total_time_ms": result.tool_total_time_ms,
                "error_category": result.error_category.value if result.error_category else None,
                "error_message": result.error_message,
                "tests_passed": result.passed_count,
                "tests_total": result.test_count,
            }
        )

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

        # Use the summary already computed in the manifest
        summary = self.manifest.summary

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
