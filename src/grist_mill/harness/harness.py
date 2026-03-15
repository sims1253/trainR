"""Reference Harness implementation that wires the full evaluation loop.

Orchestrates:
    task -> environment preparation -> agent invocation -> test execution -> result capture

Supports retry logic on transient failures (RATE_LIMIT, API_ERROR, NETWORK_ERROR)
with configurable count/delay. Does not retry on deterministic failures
(TEST_FAILURE, TIMEOUT, SKIPPED).

Attaches telemetry to every TaskResult.

Validates:
- VAL-HARNESS-08: Full wiring from task to result
- VAL-HARNESS-09: Retry logic on transient failures
- VAL-TELEM-02: Latency breakdown by phase
- VAL-TELEM-03: Tool call metrics captured
- VAL-TELEM-06: Every TaskResult has non-null telemetry
- VAL-TELEM-07: raw_events populated when trace_enabled=True
"""

from __future__ import annotations

import logging
import time
from typing import Any

from grist_mill.harness.result_parser import ResultParser
from grist_mill.schemas import (
    ErrorCategory,
    HarnessConfig,
    Task,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

# Error categories considered transient / retriable.
_TRANSIENT_ERROR_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.API_ERROR,
        ErrorCategory.NETWORK_ERROR,
    }
)


class Harness:
    """Reference evaluation harness.

    Wires the full evaluation loop:

    1. ``env.prepare(task)`` — prepare the environment
    2. ``agent.run(task, config)`` — invoke the agent (with retries on transient errors)
    3. ``env.execute(test_command)`` — run the test command
    4. ``env.cleanup()`` — tear down the environment (guaranteed)

    Every ``TaskResult`` includes non-null telemetry with latency breakdown,
    tool call metrics, and optional raw events.

    Args:
        config: The harness configuration.
        max_retries: Maximum retry attempts for transient errors (default 3).
        retry_delay: Delay in seconds between retries (default 1.0).
        trace_enabled: If True, populate ``raw_events`` in telemetry (default False).
    """

    def __init__(
        self,
        *,
        config: HarnessConfig,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        trace_enabled: bool = False,
    ) -> None:
        self._config = config
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._trace_enabled = trace_enabled
        self._result_parser = ResultParser()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts for transient errors."""
        return self._max_retries

    @property
    def retry_delay(self) -> float:
        """Delay in seconds between retries."""
        return self._retry_delay

    @property
    def trace_enabled(self) -> bool:
        """Whether raw_events tracing is enabled."""
        return self._trace_enabled

    @property
    def config(self) -> HarnessConfig:
        """The harness configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        task: Task,
        agent: Any,
        env: Any,
        collector: TelemetryCollector | None = None,
    ) -> TaskResult:
        """Execute a task through the full evaluation loop.

        The lifecycle is:

        1. **prepare**: ``env.prepare(task)``
        2. **agent invocation**: ``agent.run(task, config)`` with retries
        3. **test execution**: ``env.execute(task.test_command)``
        4. **cleanup**: ``env.cleanup()`` (guaranteed)

        Telemetry is collected throughout and attached to the result.

        Args:
            task: The task to execute.
            agent: An object with a ``run(task, config) -> TaskResult`` method.
            env: An object with ``prepare(task)``, ``execute(command, timeout)``,
                and ``cleanup()`` methods.
            collector: Optional ``TelemetryCollector`` to use. If ``None``,
                a new one is created.

        Returns:
            A ``TaskResult`` with telemetry attached.
        """
        if collector is None:
            collector = TelemetryCollector()

        # --- Phase: prepare ---
        with collector.track_phase("setup"):
            try:
                env.prepare(task)
                if self._trace_enabled:
                    collector.record_raw_event(
                        {
                            "phase": "prepare",
                            "task_id": task.id,
                            "status": "completed",
                        }
                    )
            except Exception as exc:
                logger.error(
                    "Environment preparation failed for task '%s': %s",
                    task.id,
                    exc,
                )
                if self._trace_enabled:
                    collector.record_raw_event(
                        {
                            "phase": "prepare",
                            "task_id": task.id,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                # Cleanup even on prepare failure, then return error result
                with collector.track_phase("teardown"):
                    env.cleanup()
                return _build_error_result(
                    task_id=task.id,
                    error_message=f"Environment preparation failed: {exc}",
                    error_category=ErrorCategory.ENVIRONMENT_ERROR,
                    collector=collector,
                )

        # --- Phase: agent invocation (with retries) ---
        agent_result = self._run_agent_with_retries(
            task=task,
            agent=agent,
            collector=collector,
        )

        # If the agent returned an error result (not retryable, or retries exhausted),
        # run the test command anyway if the agent produced something useful.
        # If agent returned SUCCESS, run test command to verify.
        # If agent returned TIMEOUT, SKIPPED, or TEST_FAILURE — skip test execution.

        # --- Phase: test execution ---
        test_result: TaskResult | None = None
        should_run_tests = agent_result.status not in (
            TaskStatus.TIMEOUT,
            TaskStatus.SKIPPED,
        )

        if should_run_tests:
            with collector.track_phase("execution"):
                try:
                    exec_output = env.execute(
                        task.test_command,
                        timeout=float(task.timeout),
                    )
                    if self._trace_enabled:
                        collector.record_raw_event(
                            {
                                "phase": "execute",
                                "task_id": task.id,
                                "command": task.test_command,
                                "exit_code": exec_output.exit_code,
                                "timed_out": exec_output.timed_out,
                            }
                        )

                    # Parse the test execution output
                    test_result = self._result_parser.parse(
                        exec_output,
                        task_id=task.id,
                        language=task.language,
                    )
                    # Merge telemetry from test execution into agent result
                    test_result = _merge_telemetry(test_result, collector)
                except Exception as exc:
                    logger.error(
                        "Test execution failed for task '%s': %s",
                        task.id,
                        exc,
                    )
                    if self._trace_enabled:
                        collector.record_raw_event(
                            {
                                "phase": "execute",
                                "task_id": task.id,
                                "status": "failed",
                                "error": str(exc),
                            }
                        )
                    test_result = _build_error_result(
                        task_id=task.id,
                        error_message=f"Test execution failed: {exc}",
                        error_category=ErrorCategory.ENVIRONMENT_ERROR,
                        collector=collector,
                    )

        # --- Phase: cleanup ---
        with collector.track_phase("teardown"):
            env.cleanup()
            if self._trace_enabled:
                collector.record_raw_event(
                    {
                        "phase": "cleanup",
                        "task_id": task.id,
                        "status": "completed",
                    }
                )

        # --- Determine final result ---
        if test_result is not None:
            # Test execution gives us the definitive answer
            # Build final result from test_result but with full telemetry snapshot
            final_telemetry = collector.build()
            final_result = TaskResult(
                task_id=test_result.task_id,
                status=test_result.status,
                score=test_result.score,
                error_category=test_result.error_category,
                telemetry=final_telemetry,
                transcript=test_result.transcript,
            )
        else:
            # No test execution (TIMEOUT/SKIPPED) — use agent result with telemetry
            final_result = _merge_telemetry(agent_result, collector)

        return final_result

    # ------------------------------------------------------------------
    # Agent invocation with retry logic
    # ------------------------------------------------------------------

    def _run_agent_with_retries(
        self,
        *,
        task: Task,
        agent: Any,
        collector: TelemetryCollector,
    ) -> TaskResult:
        """Run the agent with retry logic for transient errors.

        Retries when the agent returns ``ERROR`` status with a transient
        error category (RATE_LIMIT, API_ERROR, NETWORK_ERROR). Does NOT
        retry on deterministic failures (TEST_FAILURE, TIMEOUT, SKIPPED).

        Args:
            task: The task to execute.
            agent: The agent to invoke.
            collector: Telemetry collector for tracking.

        Returns:
            The ``TaskResult`` from the agent (or last retry attempt).
        """
        attempts = 0
        max_attempts = 1 + self._max_retries  # initial + retries

        while attempts < max_attempts:
            attempts += 1

            try:
                agent_result = agent.run(task, self._config)
            except Exception as exc:
                logger.error(
                    "Agent raised exception on attempt %d/%d for task '%s': %s",
                    attempts,
                    max_attempts,
                    task.id,
                    exc,
                )
                if self._trace_enabled:
                    collector.record_raw_event(
                        {
                            "phase": "agent_run",
                            "task_id": task.id,
                            "attempt": attempts,
                            "status": "exception",
                            "error": str(exc),
                        }
                    )

                # Agent exceptions are treated as non-retriable
                # (the agent itself is broken, not a transient error)
                return _build_error_result(
                    task_id=task.id,
                    error_message=f"Agent exception: {exc}",
                    error_category=ErrorCategory.API_ERROR,
                    collector=collector,
                )

            if self._trace_enabled:
                collector.record_raw_event(
                    {
                        "phase": "agent_run",
                        "task_id": task.id,
                        "attempt": attempts,
                        "status": agent_result.status.value,
                        "error_category": (
                            agent_result.error_category.value
                            if agent_result.error_category
                            else None
                        ),
                    }
                )

            # Check if this result is retriable
            if self._should_retry(agent_result) and attempts < max_attempts:
                logger.warning(
                    "Transient error on attempt %d/%d for task '%s' "
                    "(category=%s), retrying in %.1fs...",
                    attempts,
                    max_attempts,
                    task.id,
                    agent_result.error_category,
                    self._retry_delay,
                )
                time.sleep(self._retry_delay)
                continue

            # Not retriable or retries exhausted — return the result
            return agent_result

        # Should not reach here, but return last attempt's result
        return agent_result  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retriable_error(error_category: ErrorCategory | None) -> bool:
        """Check if an error category is transient and should be retried.

        Args:
            error_category: The error category from a TaskResult.

        Returns:
            ``True`` if the error is transient (RATE_LIMIT, API_ERROR, NETWORK_ERROR).
        """
        if error_category is None:
            return False
        return error_category in _TRANSIENT_ERROR_CATEGORIES

    def _should_retry(self, result: TaskResult) -> bool:
        """Determine if a TaskResult should trigger a retry.

        A result is retriable when:
        - status is ERROR
        - error_category is one of RATE_LIMIT, API_ERROR, NETWORK_ERROR

        Args:
            result: The TaskResult to evaluate.

        Returns:
            ``True`` if the result should be retried.
        """
        if result.status != TaskStatus.ERROR:
            return False
        return self._is_retriable_error(result.error_category)

    def __repr__(self) -> str:
        return (
            f"Harness("
            f"max_retries={self._max_retries}, "
            f"retry_delay={self._retry_delay}, "
            f"trace_enabled={self._trace_enabled})"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_error_result(
    *,
    task_id: str,
    error_message: str,
    error_category: ErrorCategory,
    collector: TelemetryCollector,
) -> TaskResult:
    """Build an error TaskResult with telemetry attached."""
    telemetry = collector.build()
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.ERROR,
        score=0.0,
        error_category=error_category,
        telemetry=telemetry,
        transcript=[{"message": error_message}],
    )


def _merge_telemetry(
    result: TaskResult,
    collector: TelemetryCollector,
) -> TaskResult:
    """Merge telemetry from the collector into a TaskResult.

    Creates a new TaskResult with the collector's telemetry snapshot
    attached while preserving all other fields.
    """
    telemetry = collector.build()
    return TaskResult(
        task_id=result.task_id,
        status=result.status,
        score=result.score,
        error_category=result.error_category,
        telemetry=telemetry,
        transcript=result.transcript,
    )


# ---------------------------------------------------------------------------
# run_experiment — convenience function for running multiple tasks
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    tasks: list[Task],
    config: HarnessConfig,
    agent: Any,
    env: Any,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    trace_enabled: bool = False,
) -> list[TaskResult]:
    """Run multiple tasks through the harness and return all results.

    Each task gets its own ``TelemetryCollector`` so metrics are independent.
    The environment is prepared and cleaned up per task.

    Args:
        tasks: List of tasks to execute.
        config: Harness configuration.
        agent: Agent to invoke for each task.
        env: Environment for execution.
        max_retries: Maximum retry attempts per task.
        retry_delay: Delay in seconds between retries.
        trace_enabled: Whether to capture raw events.

    Returns:
        A list of ``TaskResult`` objects, one per task.
    """
    harness = Harness(
        config=config,
        max_retries=max_retries,
        retry_delay=retry_delay,
        trace_enabled=trace_enabled,
    )

    results: list[TaskResult] = []
    for i, task in enumerate(tasks):
        logger.info(
            "Running task %d/%d: '%s'",
            i + 1,
            len(tasks),
            task.id,
        )
        collector = TelemetryCollector()
        result = harness.run(
            task=task,
            agent=agent,
            env=env,
            collector=collector,
        )
        results.append(result)
        logger.info(
            "Task '%s' completed: status=%s score=%.1f",
            task.id,
            result.status.value,
            result.score,
        )

    return results
