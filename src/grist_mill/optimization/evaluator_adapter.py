"""GEPA evaluator adapter for grist-mill.

Wraps the evaluation harness as a GEPA-compatible evaluator function.
The adapter returns ``(float_score, dict_side_info)`` per the GEPA contract,
with side info capturing:

- Execution traces (transcript, raw events)
- Errors (error category, error messages)
- Timing (duration_s, latency breakdown)
- Token usage (prompt, completion, total)
- Tool call metrics (counts, per-tool breakdown)

Supports configurable scoring objectives:

- ``pass-rate``: Simple fraction of tasks that pass (default)
- ``cost-adjusted``: Penalizes high token usage
- ``difficulty-weighted``: Weights by task difficulty

All objectives produce deterministic, reproducible scores.

The evaluator adapter is pluggable: users can provide a custom evaluator
via config that replaces the default without code changes.

Validates:
- VAL-GEPA-01: Evaluator adapter captures actionable side information
- VAL-GEPA-02: Side information is passed to reflection model
- VAL-GEPA-03: Multi-objective evaluation produces composite scores
- VAL-GEPA-06: Evaluator adapter is pluggable
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from grist_mill.schemas import (
    Difficulty,
    Task,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.telemetry import TelemetrySchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class EvaluatorAdapterConfig(BaseModel):
    """Configuration for the GEPA evaluator adapter.

    Controls which objective function to use, cost parameters, tracing,
    and difficulty weights for the weighted objective.

    Attributes:
        objective: Scoring objective — one of ``pass-rate``, ``cost-adjusted``,
            or ``difficulty-weighted``.
        cost_per_token: Cost per token in USD for the cost-adjusted objective.
        trace_enabled: Whether to include raw events in side info.
        difficulty_weights: Per-difficulty weights for difficulty-weighted
            objective. Default: EASY=1.0, MEDIUM=2.0, HARD=3.0.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    objective: Literal["pass-rate", "cost-adjusted", "difficulty-weighted"] = Field(
        default="pass-rate",
        description="Scoring objective function.",
    )
    cost_per_token: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per token in USD (for cost-adjusted objective).",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to include raw events in side info.",
    )
    difficulty_weights: dict[str, float] = Field(
        default_factory=lambda: {"EASY": 1.0, "MEDIUM": 2.0, "HARD": 3.0},
        description="Per-difficulty weights for difficulty-weighted objective.",
    )


# ---------------------------------------------------------------------------
# Objective Functions
# ---------------------------------------------------------------------------


class ObjectiveFunction(ABC):
    """Abstract base class for scoring objectives.

    Each objective takes a list of TaskResults and their corresponding Tasks
    and produces a deterministic scalar score in [0.0, 1.0].
    """

    @abstractmethod
    def compute(self, results: list[TaskResult], tasks: list[Task]) -> float:
        """Compute the aggregate score from results.

        Args:
            results: List of task results.
            tasks: Corresponding list of task definitions.

        Returns:
            A deterministic scalar score in [0.0, 1.0].
        """
        ...

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> ObjectiveFunction:
        """Factory method to create an objective function by name.

        Args:
            name: Objective name — ``pass-rate``, ``cost-adjusted``,
                or ``difficulty-weighted``.
            **kwargs: Additional parameters for the objective.

        Returns:
            An instance of the requested objective function.

        Raises:
            ValueError: If the objective name is unknown.
        """
        # Map objective names to their parameter keys
        objective_params: dict[str, set[str]] = {
            "pass-rate": set(),
            "cost-adjusted": {"cost_per_token"},
            "difficulty-weighted": {"difficulty_weights"},
        }

        if name not in objective_params:
            msg = f"Unknown objective: {name!r}. Must be one of {sorted(objective_params.keys())}."
            raise ValueError(msg)

        # Only pass relevant kwargs
        allowed = objective_params[name]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        known: dict[str, type[ObjectiveFunction]] = {
            "pass-rate": PassRateObjective,
            "cost-adjusted": CostAdjustedObjective,
            "difficulty-weighted": DifficultyWeightedObjective,
        }
        return known[name](**filtered_kwargs)


class PassRateObjective(ObjectiveFunction):
    """Simple pass-rate objective.

    Score = (number of successful tasks) / (total tasks).

    Deterministic: same inputs always produce the same output.
    """

    def __init__(self, **_kwargs: Any) -> None:
        """Accept but ignore extra kwargs for factory compatibility."""

    def compute(self, results: list[TaskResult], tasks: list[Task]) -> float:
        """Compute pass rate.

        Args:
            results: List of task results.
            tasks: Corresponding list of task definitions (used for count).

        Returns:
            Fraction of tasks with status SUCCESS, in [0.0, 1.0].
        """
        if not results:
            return 0.0
        success_count = sum(1 for r in results if r.status == TaskStatus.SUCCESS)
        return success_count / len(results)


class CostAdjustedObjective(ObjectiveFunction):
    """Cost-adjusted objective.

    Penalizes high token usage. For each task:
    - Success: score = 1.0 / (1 + cost)
    - Failure: score = 0.0

    The total score is the average across all tasks. Cost is computed
    as ``total_tokens * cost_per_token``.

    Deterministic: same inputs always produce the same output.
    """

    def __init__(self, cost_per_token: float = 0.0) -> None:
        self.cost_per_token = cost_per_token

    def compute(self, results: list[TaskResult], tasks: list[Task]) -> float:
        """Compute cost-adjusted score.

        Args:
            results: List of task results.
            tasks: Corresponding list of task definitions.

        Returns:
            Cost-adjusted score in [0.0, 1.0].
        """
        if not results:
            return 0.0

        total_score = 0.0
        for result in results:
            if result.status != TaskStatus.SUCCESS:
                total_score += 0.0
                continue

            # Extract token usage from telemetry
            total_tokens = self._get_total_tokens(result)
            cost = total_tokens * self.cost_per_token
            total_score += 1.0 / (1.0 + cost)

        return total_score / len(results)

    @staticmethod
    def _get_total_tokens(result: TaskResult) -> int:
        """Extract total token count from telemetry."""
        if result.telemetry is None:
            return 0
        if isinstance(result.telemetry, TelemetrySchema):
            return result.telemetry.tokens.total
        # Telemetry might be a dict (e.g., from JSON round-trip)
        if isinstance(result.telemetry, dict):
            tokens = result.telemetry.get("tokens", {})
            if isinstance(tokens, dict):
                return tokens.get("total", 0)
        return 0


class DifficultyWeightedObjective(ObjectiveFunction):
    """Difficulty-weighted objective.

    Weights each task's contribution by its difficulty level:
    - EASY: weight 1.0
    - MEDIUM: weight 2.0 (default)
    - HARD: weight 3.0 (default)

    Score = (sum of weights for successful tasks) / (sum of all weights).

    Deterministic: same inputs always produce the same output.
    """

    def __init__(
        self,
        difficulty_weights: dict[str, float] | None = None,
    ) -> None:
        self._weights = difficulty_weights or {
            "EASY": 1.0,
            "MEDIUM": 2.0,
            "HARD": 3.0,
        }

    def compute(self, results: list[TaskResult], tasks: list[Task]) -> float:
        """Compute difficulty-weighted score.

        Args:
            results: List of task results.
            tasks: Corresponding list of task definitions.

        Returns:
            Difficulty-weighted score in [0.0, 1.0].
        """
        if not results or not tasks:
            return 0.0

        total_weight = 0.0
        success_weight = 0.0

        for task, result in zip(tasks, results, strict=True):
            weight = self._get_weight(task)
            total_weight += weight
            if result.status == TaskStatus.SUCCESS:
                success_weight += weight

        if total_weight == 0.0:
            return 0.0

        return success_weight / total_weight

    def _get_weight(self, task: Task) -> float:
        """Get the difficulty weight for a task."""
        difficulty_name = (
            task.difficulty.value
            if isinstance(task.difficulty, Difficulty)
            else str(task.difficulty)
        )
        return self._weights.get(difficulty_name, 1.0)


# ---------------------------------------------------------------------------
# GepaEvaluatorAdapter
# ---------------------------------------------------------------------------


class GepaEvaluatorAdapter:
    """Wraps the evaluation harness as a GEPA-compatible evaluator.

    Conforms to the GEPA ``Evaluator`` protocol::

        __call__(candidate, example, **kwargs) -> tuple[float, dict[str, Any]]

    The adapter:
    1. Runs the evaluation harness with the candidate and task
    2. Extracts structured side information from the TaskResult
    3. Computes a score using the configured objective function
    4. Returns ``(score, side_info)`` for GEPA reflection

    Side info includes:
    - ``task_id``: The evaluated task's ID
    - ``status``: Task status (SUCCESS, FAILURE, etc.)
    - ``difficulty``: Task difficulty level
    - ``score``: Raw task score
    - ``error_category``: Error category (if failed)
    - ``errors``: Error messages for reflection model
    - ``traces``: Execution transcript
    - ``duration_s``: Total execution duration
    - ``token_usage``: Token counts (prompt, completion, total)
    - ``tool_calls``: Tool invocation metrics
    - ``raw_events``: Raw trace events (if trace_enabled=True)

    Args:
        harness: The evaluation harness with a ``run()`` method.
        config: Adapter configuration (objective, costs, etc.).
    """

    def __init__(
        self,
        harness: Any,
        config: EvaluatorAdapterConfig | None = None,
    ) -> None:
        self._harness = harness
        self._config = config or EvaluatorAdapterConfig()
        self._objective_function: ObjectiveFunction = ObjectiveFunction.create(
            self._config.objective,
            cost_per_token=self._config.cost_per_token,
            difficulty_weights=self._config.difficulty_weights,
        )

    @property
    def _objective(self) -> ObjectiveFunction:
        """The configured objective function."""
        return self._objective_function

    def __call__(
        self,
        candidate: str | dict[str, str],
        example: Any,
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate on a single task.

        Conforms to the GEPA ``Evaluator`` protocol.

        Args:
            candidate: The candidate text or parameter dict to evaluate.
            example: A task from the dataset (``Task`` instance).
            **kwargs: Additional keyword arguments (reserved for GEPA).

        Returns:
            Tuple of ``(score, side_info)`` where:
            - ``score`` is a float in [0.0, 1.0]
            - ``side_info`` is a dict with diagnostic information
        """
        start_time = time.perf_counter()

        try:
            result = self._run_harness(candidate, example)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Harness raised exception during evaluation: %s",
                exc,
            )
            side_info = self._build_error_side_info(example, exc, elapsed)
            return 0.0, side_info

        elapsed = time.perf_counter() - start_time
        side_info = self._build_side_info(result, example)

        # Override duration_s with wall-clock measurement
        side_info["duration_s"] = round(elapsed, 4)

        # Compute score using the configured objective
        score = self._objective_function.compute([result], [example])
        score = round(score, 6)

        logger.debug(
            "GEPA evaluation: task=%s status=%s score=%.4f duration=%.2fs",
            side_info.get("task_id", "?"),
            side_info.get("status", "?"),
            score,
            elapsed,
        )

        return score, side_info

    def _run_harness(self, candidate: str | dict[str, str], task: Task) -> TaskResult:
        """Run the evaluation harness.

        Args:
            candidate: The candidate to evaluate (currently unused by harness;
                the harness uses its own config for agent setup).
            task: The task to evaluate.

        Returns:
            The TaskResult from the harness.
        """
        # The harness.run() method may have varying signatures depending
        # on the harness implementation. We try the most common patterns.
        # For the grist-mill Harness, it expects: task, agent, env, collector
        # But the adapter may also wrap simpler harnesses.
        if hasattr(self._harness, "run"):
            run_method = self._harness.run
            import inspect

            sig = inspect.signature(run_method)
            params = list(sig.parameters.keys())

            # If the harness expects (task, agent, env, ...) pattern,
            # we just call with the task directly if it also accepts
            # a single positional argument pattern
            if len(params) >= 1 and params[0] in ("task", "self"):
                # Try calling with task as keyword argument
                try:
                    return run_method(task=task)
                except TypeError:
                    pass
                # Try calling with task as positional argument
                try:
                    return run_method(task)
                except TypeError:
                    pass
            # Fallback: call with task as first positional
            return run_method(task)

        raise RuntimeError("Harness does not have a callable 'run' method.")

    def _build_side_info(
        self,
        result: TaskResult,
        task: Task,
    ) -> dict[str, Any]:
        """Build structured side info from a TaskResult.

        Extracts traces, errors, timing, and token usage for GEPA reflection.

        Args:
            result: The task result.
            task: The task that was evaluated.

        Returns:
            A dict with all side information fields.
        """
        # --- Basic fields ---
        side_info: dict[str, Any] = {
            "task_id": result.task_id,
            "status": result.status.value,
            "score": result.score,
            "difficulty": task.difficulty.value
            if isinstance(task.difficulty, Difficulty)
            else str(task.difficulty),
            "error_category": result.error_category.value if result.error_category else None,
        }

        # --- Errors for reflection ---
        side_info["errors"] = self._extract_errors(result)

        # --- Traces (transcript) ---
        side_info["traces"] = result.transcript or []

        # --- Timing ---
        side_info["duration_s"] = self._extract_duration(result)

        # --- Token usage ---
        side_info["token_usage"] = self._extract_token_usage(result)

        # --- Tool calls ---
        side_info["tool_calls"] = self._extract_tool_calls(result)

        # --- Raw events (if trace_enabled) ---
        if self._config.trace_enabled:
            side_info["raw_events"] = self._extract_raw_events(result)

        return side_info

    def _build_error_side_info(
        self,
        task: Any,
        exc: Exception,
        elapsed: float,
    ) -> dict[str, Any]:
        """Build side info when the harness raises an exception.

        Args:
            task: The task that was being evaluated.
            exc: The exception that was raised.
            elapsed: Wall-clock time elapsed before the exception.

        Returns:
            A dict with error side information.
        """
        task_id = getattr(task, "id", "unknown") if task else "unknown"
        difficulty = (
            task.difficulty.value
            if hasattr(task, "difficulty") and isinstance(task.difficulty, Difficulty)
            else "UNKNOWN"
        )
        return {
            "task_id": task_id,
            "status": "ERROR",
            "score": 0.0,
            "difficulty": difficulty,
            "error_category": "HARNESS_ERROR",
            "errors": [f"Harness exception: {exc}"],
            "traces": [],
            "duration_s": round(elapsed, 4),
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "tool_calls": {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
            },
        }

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_errors(result: TaskResult) -> list[str]:
        """Extract error messages from a TaskResult for reflection.

        Collects error messages from:
        - Transcript messages
        - Error category
        - Status information

        Args:
            result: The task result.

        Returns:
            A list of error message strings.
        """
        errors: list[str] = []

        if result.error_category:
            errors.append(f"Error category: {result.error_category.value}")

        if result.transcript:
            for entry in result.transcript:
                if isinstance(entry, dict):
                    # Extract error messages from transcript entries
                    for key in ("message", "error", "stderr"):
                        if entry.get(key):
                            msg = str(entry[key])
                            # Truncate very long messages
                            if len(msg) > 500:
                                msg = msg[:500] + "... [truncated]"
                            errors.append(msg)
                elif isinstance(entry, str) and entry:
                    if len(entry) > 500:
                        entry = entry[:500] + "... [truncated]"
                    errors.append(entry)

        return errors

    @staticmethod
    def _extract_duration(result: TaskResult) -> float:
        """Extract execution duration from telemetry.

        Args:
            result: The task result.

        Returns:
            Duration in seconds, or 0.0 if not available.
        """
        if result.telemetry is None:
            return 0.0
        if isinstance(result.telemetry, TelemetrySchema):
            return result.telemetry.latency.total_s
        if isinstance(result.telemetry, dict):
            latency = result.telemetry.get("latency", {})
            if isinstance(latency, dict):
                return latency.get("total_s", 0.0)
        return 0.0

    @staticmethod
    def _extract_token_usage(result: TaskResult) -> dict[str, int]:
        """Extract token usage from telemetry.

        Args:
            result: The task result.

        Returns:
            Dict with prompt, completion, and total token counts.
        """
        default: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

        if result.telemetry is None:
            return default
        if isinstance(result.telemetry, TelemetrySchema):
            return {
                "prompt": result.telemetry.tokens.prompt,
                "completion": result.telemetry.tokens.completion,
                "total": result.telemetry.tokens.total,
            }
        if isinstance(result.telemetry, dict):
            tokens = result.telemetry.get("tokens", {})
            if isinstance(tokens, dict):
                return {
                    "prompt": tokens.get("prompt", 0),
                    "completion": tokens.get("completion", 0),
                    "total": tokens.get("total", 0),
                }
        return default

    @staticmethod
    def _extract_tool_calls(result: TaskResult) -> dict[str, Any]:
        """Extract tool call metrics from telemetry.

        Args:
            result: The task result.

        Returns:
            Dict with tool call metrics.
        """
        default: dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
        }

        if result.telemetry is None:
            return default
        if isinstance(result.telemetry, TelemetrySchema):
            tc = result.telemetry.tool_calls
            return {
                "total_calls": tc.total_calls,
                "successful_calls": tc.successful_calls,
                "failed_calls": tc.failed_calls,
                "by_tool": tc.by_tool if tc.by_tool else {},
            }
        if isinstance(result.telemetry, dict):
            tool_calls = result.telemetry.get("tool_calls", {})
            if isinstance(tool_calls, dict):
                return {
                    "total_calls": tool_calls.get("total_calls", 0),
                    "successful_calls": tool_calls.get("successful_calls", 0),
                    "failed_calls": tool_calls.get("failed_calls", 0),
                    "by_tool": tool_calls.get("by_tool", {}),
                }
        return default

    @staticmethod
    def _extract_raw_events(result: TaskResult) -> list[dict[str, Any]]:
        """Extract raw trace events from telemetry.

        Args:
            result: The task result.

        Returns:
            List of raw event dicts.
        """
        if result.telemetry is None:
            return []
        if isinstance(result.telemetry, TelemetrySchema):
            return result.telemetry.raw_events or []
        if isinstance(result.telemetry, dict):
            events = result.telemetry.get("raw_events", [])
            if isinstance(events, list):
                return events
        return []

    def __repr__(self) -> str:
        return (
            f"GepaEvaluatorAdapter("
            f"objective={self._config.objective!r}, "
            f"trace_enabled={self._config.trace_enabled})"
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_evaluator_adapter(
    harness: Any,
    config: EvaluatorAdapterConfig | None = None,
) -> GepaEvaluatorAdapter:
    """Create a GEPA evaluator adapter from a harness and config.

    This is the primary factory for creating evaluator adapters that
    wrap the grist-mill evaluation harness for use with GEPA.

    Args:
        harness: The evaluation harness with a ``run()`` method.
        config: Adapter configuration. Uses defaults if ``None``.

    Returns:
        A ``GepaEvaluatorAdapter`` instance.

    Example::

        from grist_mill.optimization import create_evaluator_adapter, EvaluatorAdapterConfig

        adapter = create_evaluator_adapter(
            harness=my_harness,
            config=EvaluatorAdapterConfig(
                objective="cost-adjusted",
                cost_per_token=0.001,
            ),
        )
        score, side_info = adapter(candidate="my skill", example=task)
    """
    return GepaEvaluatorAdapter(harness=harness, config=config)


def load_custom_evaluator(
    evaluator_fn: Callable[..., tuple[float, dict[str, Any]]],
) -> Callable[..., tuple[float, dict[str, Any]]]:
    """Wrap a custom evaluator function for use with GEPA.

    Validates that the callable conforms to the GEPA evaluator protocol
    and wraps it with basic logging.

    Args:
        evaluator_fn: A callable with signature
            ``(candidate, example, **kwargs) -> tuple[float, dict[str, Any]]``.

    Returns:
        The wrapped callable, ready for use with GEPA.

    Raises:
        TypeError: If ``evaluator_fn`` is not callable.

    Example::

        def my_evaluator(candidate: str, example, **kwargs):
            # Custom evaluation logic
            return 0.8, {"task_id": example.id}


        adapter = load_custom_evaluator(my_evaluator)
        score, info = adapter(candidate="prompt", example=task)
    """
    if not callable(evaluator_fn):
        raise TypeError(f"Custom evaluator must be callable, got {type(evaluator_fn).__name__}")

    def _wrapped(
        candidate: str | dict[str, str],
        example: Any,
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        logger.debug(
            "Custom evaluator called for task=%s",
            getattr(example, "id", "unknown") if example else "unknown",
        )
        result = evaluator_fn(candidate, example, **kwargs)
        score, side_info = result

        if not isinstance(score, (int, float)):
            raise TypeError(
                f"Custom evaluator must return (float, dict), got score type {type(score).__name__}"
            )
        if not isinstance(side_info, dict):
            raise TypeError(
                f"Custom evaluator must return (float, dict), "
                f"got side_info type {type(side_info).__name__}"
            )

        # Ensure task_id is in side info
        if "task_id" not in side_info and example is not None:
            side_info["task_id"] = getattr(example, "id", "unknown")

        logger.debug(
            "Custom evaluator result: score=%.4f task=%s",
            float(score),
            side_info.get("task_id", "?"),
        )
        return float(score), side_info

    return _wrapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CostAdjustedObjective",
    "DifficultyWeightedObjective",
    "EvaluatorAdapterConfig",
    "GepaEvaluatorAdapter",
    "ObjectiveFunction",
    "PassRateObjective",
    "create_evaluator_adapter",
    "load_custom_evaluator",
]
