"""Optimization runtime for grist-mill.

Provides the core optimization loop with:
- Composable budget management (max_calls, timeout, no-improvement)
- Checkpointing and state persistence
- Resume from checkpoint
- Pareto front management
- Multiple target types (skill, system_prompt, tool_policy)
- Graceful termination with signal handling
- Holdout evaluation for overfitting prevention

Validates:
- VAL-OPT-01: optimize command runs end-to-end
- VAL-OPT-02: Budget enforcement triggers graceful termination
- VAL-OPT-03: Timeout budget halts within tolerance
- VAL-OPT-04: No-improvement stop condition detects stagnation
- VAL-OPT-05: Checkpointing persists full optimization state
- VAL-OPT-06: Resume from checkpoint restores exact state
- VAL-OPT-07: Evolved artifact is exportable
- VAL-OPT-08: Optimization supports multiple target types
- VAL-GEPA-04: Holdout evaluation prevents overfitting
- VAL-GEPA-05: Pareto-front accumulation prevents regression
"""

from __future__ import annotations

import json
import logging
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grist_mill.optimization.evaluator_adapter import EvaluatorAdapterConfig
from grist_mill.schemas import Task

logger = logging.getLogger(__name__)

# Valid target types
VALID_TARGET_TYPES: frozenset[str] = frozenset({"skill", "system_prompt", "tool_policy"})


# ---------------------------------------------------------------------------
# Budget Management
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    """Configuration for optimization budget.

    All conditions are optional and composable. When multiple conditions
    are set, the first one to trigger causes termination.

    Attributes:
        max_calls: Maximum number of evaluator calls.
        timeout_s: Maximum wall-clock time in seconds.
        no_improvement_patience: Number of consecutive iterations without
            improvement before stopping.
        checkpoint_interval: Save checkpoint every N iterations (default 1).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    max_calls: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of evaluator calls.",
    )
    timeout_s: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum wall-clock time in seconds.",
    )
    no_improvement_patience: int | None = Field(
        default=None,
        ge=1,
        description="Consecutive iterations without improvement before stopping.",
    )
    checkpoint_interval: int = Field(
        default=1,
        ge=1,
        description="Save checkpoint every N iterations.",
    )


class StopCondition:
    """Composable stop condition checker for the optimization loop.

    Tracks call count, elapsed time, and improvement history to determine
    when the loop should terminate. Reports which condition triggered.

    The ``update_and_check`` method should be called once per iteration with
    the current best score to update the no-improvement counter. The
    ``should_stop`` method is a pure check without side effects.

    Args:
        budget: The budget configuration defining stop conditions.
    """

    def __init__(self, budget: BudgetConfig) -> None:
        self._budget = budget
        self._last_best_score: float | None = None
        self._consecutive_no_improvement: int = 0

    def update_and_check(
        self,
        call_count: int,
        elapsed_s: float,
        best_score: float,
    ) -> bool:
        """Update no-improvement tracking and check stop conditions.

        Should be called once per iteration. Updates the internal
        improvement counter based on the current best score.

        Args:
            call_count: Current number of evaluator calls.
            elapsed_s: Wall-clock time elapsed in seconds.
            best_score: Current best score across all iterations.

        Returns:
            True if any stop condition is met.
        """
        # Update no-improvement tracking
        if self._budget.no_improvement_patience is not None:
            if self._last_best_score is not None:
                if best_score > self._last_best_score:
                    self._consecutive_no_improvement = 0
                else:
                    self._consecutive_no_improvement += 1
            self._last_best_score = best_score

        return self.should_stop(call_count, elapsed_s, best_score)

    def should_stop(
        self,
        call_count: int,
        elapsed_s: float,
        best_score: float,
    ) -> bool:
        """Pure check: determine if any stop condition is met.

        This method has no side effects. For the full update+check cycle,
        use ``update_and_check`` instead.

        Args:
            call_count: Current number of evaluator calls.
            elapsed_s: Wall-clock time elapsed in seconds.
            best_score: Current best score across all iterations.

        Returns:
            True if any stop condition is met.
        """
        if self._budget.max_calls is not None and call_count >= self._budget.max_calls:
            return True

        if self._budget.timeout_s is not None and elapsed_s >= self._budget.timeout_s:
            return True

        return (
            self._budget.no_improvement_patience is not None
            and self._consecutive_no_improvement >= self._budget.no_improvement_patience
        )

    def termination_reason(
        self,
        call_count: int,
        elapsed_s: float,
        best_score: float,
    ) -> str | None:
        """Identify which condition triggered termination.

        Args:
            call_count: Current number of evaluator calls.
            elapsed_s: Wall-clock time elapsed in seconds.
            best_score: Current best score across all iterations.

        Returns:
            The name of the triggered condition, or None if no condition
            is met.
        """
        if self._budget.max_calls is not None and call_count >= self._budget.max_calls:
            return "max_calls"

        if self._budget.timeout_s is not None and elapsed_s >= self._budget.timeout_s:
            return "timeout"

        if (
            self._budget.no_improvement_patience is not None
            and self._consecutive_no_improvement >= self._budget.no_improvement_patience
        ):
            return "no_improvement"

        return None


# ---------------------------------------------------------------------------
# Target Types
# ---------------------------------------------------------------------------


class TargetType(BaseModel):
    """Represents an optimization target (what is being optimized).

    Attributes:
        type: The target type — one of ``skill``, ``system_prompt``,
            or ``tool_policy``.
        content: The current content of the target (e.g., skill text,
            system prompt text, tool policy JSON).
        metadata: Optional metadata about the target.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    type: str = Field(
        ...,
        description="Target type: skill, system_prompt, or tool_policy.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Current content of the target.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the target.",
    )

    @model_validator(mode="after")
    def _validate_target_type(self) -> TargetType:
        """Ensure the target type is one of the known types."""
        if self.type not in VALID_TARGET_TYPES:
            msg = (
                f"Unknown target type: {self.type!r}. Must be one of {sorted(VALID_TARGET_TYPES)}."
            )
            raise ValueError(msg)
        return self


class TargetConfig(BaseModel):
    """Configuration for the optimization target.

    Convenience model that combines type, content, and metadata.

    Attributes:
        type: Target type (skill, system_prompt, tool_policy).
        content: Initial content of the target.
        metadata: Optional metadata.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    type: str = Field(
        ...,
        description="Target type.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Target content.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata.",
    )

    @model_validator(mode="after")
    def _validate_type(self) -> TargetConfig:
        """Validate target type."""
        if self.type not in VALID_TARGET_TYPES:
            msg = (
                f"Unknown target type: {self.type!r}. Must be one of {sorted(VALID_TARGET_TYPES)}."
            )
            raise ValueError(msg)
        return self


def serialize_target(target: TargetType) -> dict[str, Any]:
    """Serialize a TargetType to a JSON-compatible dict.

    Args:
        target: The target to serialize.

    Returns:
        A dict suitable for JSON serialization.
    """
    return {
        "type": target.type,
        "content": target.content,
        "metadata": target.metadata,
    }


# ---------------------------------------------------------------------------
# Checkpoint State and Persistence
# ---------------------------------------------------------------------------


class CheckpointState(BaseModel):
    """Full optimization state for checkpointing.

    Contains all information needed to resume an optimization run:
    candidate pool, Pareto front, iteration count, budget counters,
    trajectory history, and target information.

    Attributes:
        iteration: Current iteration number.
        call_count: Total evaluator calls made.
        best_score: Best score achieved so far.
        candidates: All candidates evaluated so far.
        pareto_front: Current Pareto front candidates.
        best_candidate: The best candidate found.
        trajectory: History of iteration scores.
        target_type: Type of target being optimized.
        seed_content: Original seed content.
        elapsed_s: Wall-clock time elapsed in seconds.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    iteration: int = Field(
        ...,
        ge=0,
        description="Current iteration number.",
    )
    call_count: int = Field(
        ...,
        ge=0,
        description="Total evaluator calls made.",
    )
    best_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Best score achieved so far.",
    )
    candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All candidates evaluated.",
    )
    pareto_front: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current Pareto front.",
    )
    best_candidate: dict[str, Any] | None = Field(
        default=None,
        description="Best candidate found.",
    )
    trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Iteration history.",
    )
    target_type: str = Field(
        ...,
        description="Type of target being optimized.",
    )
    seed_content: str = Field(
        ...,
        description="Original seed content.",
    )
    elapsed_s: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time elapsed in seconds.",
    )


class OptimizationCheckpoint:
    """Manages checkpoint persistence for optimization runs.

    Saves and loads the full optimization state to/from a JSON file.
    The checkpoint is valid JSON and can survive SIGTERM.

    Args:
        path: Path to the checkpoint file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, state: CheckpointState) -> None:
        """Save the optimization state to the checkpoint file.

        Args:
            state: The full optimization state to persist.
        """
        data = state.model_dump(mode="json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Checkpoint saved to %s (iteration=%d)", self._path, state.iteration)

    def load(self) -> CheckpointState | None:
        """Load the optimization state from the checkpoint file.

        Returns:
            The loaded CheckpointState, or None if the file doesn't exist.
        """
        if not self._path.exists():
            logger.debug("No checkpoint found at %s", self._path)
            return None

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            state = CheckpointState(**data)
            logger.debug(
                "Checkpoint loaded from %s (iteration=%d, call_count=%d)",
                self._path,
                state.iteration,
                state.call_count,
            )
            return state
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to load checkpoint from %s: %s", self._path, exc)
            return None


# ---------------------------------------------------------------------------
# Pareto Front
# ---------------------------------------------------------------------------


class ParetoFront:
    """Manages the Pareto front of optimization candidates.

    A candidate enters the Pareto front if it is not dominated by any
    existing front member. A candidate dominates another if it has a
    higher or equal score on all dimensions and strictly higher on at
    least one.

    This prevents regression: no accepted candidate degrades performance
    on any previously-solved task (VAL-GEPA-05).
    """

    def __init__(self) -> None:
        # Map from candidate ID to its scores dict
        self._front: dict[str, dict[str, float]] = {}

    @property
    def front(self) -> dict[str, dict[str, float]]:
        """The current Pareto front (candidate_id -> scores dict)."""
        return dict(self._front)

    def update(self, candidate_id: str, scores: dict[str, float]) -> bool:
        """Try to add a candidate to the Pareto front.

        Args:
            candidate_id: Unique identifier for the candidate.
            scores: Dict mapping dimension names to score values.

        Returns:
            True if the candidate was added to the front.
        """
        # Remove any existing front members dominated by the new candidate
        dominated = [
            cid
            for cid, existing_scores in self._front.items()
            if self._dominates(scores, existing_scores)
        ]
        for cid in dominated:
            del self._front[cid]

        # Check if the new candidate is dominated by any remaining member
        for existing_scores in self._front.values():
            if self._dominates(existing_scores, scores):
                return False

        self._front[candidate_id] = scores
        return True

    def get_best(self) -> str | None:
        """Get the candidate with the highest average score.

        Returns:
            The candidate ID with the highest average score, or None
            if the front is empty.
        """
        if not self._front:
            return None

        best_id: str | None = None
        best_avg = -1.0

        for cid, scores in self._front.items():
            avg = sum(scores.values()) / len(scores) if scores else 0.0
            if avg > best_avg:
                best_avg = avg
                best_id = cid

        return best_id

    def to_list(self) -> list[dict[str, Any]]:
        """Convert the Pareto front to a list of dicts.

        Returns:
            List of dicts with 'candidate_id' and 'scores' keys.
        """
        return [{"candidate_id": cid, "scores": scores} for cid, scores in self._front.items()]

    @staticmethod
    def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
        """Check if scores dict ``a`` dominates ``b``.

        ``a`` dominates ``b`` if a is >= b on all dimensions and
        strictly > on at least one.

        Args:
            a: First candidate's scores.
            b: Second candidate's scores.

        Returns:
            True if ``a`` dominates ``b``.
        """
        if set(a.keys()) != set(b.keys()):
            # Different dimensions — can't compare
            return False

        at_least_one_strictly_better = False
        for key in a:
            if a[key] < b[key]:
                return False
            if a[key] > b[key]:
                at_least_one_strictly_better = True

        return at_least_one_strictly_better


# ---------------------------------------------------------------------------
# Proposer (abstract)
# ---------------------------------------------------------------------------


class BaseProposer(ABC):
    """Abstract base class for candidate proposers.

    Given the current candidate, evaluation history, and side information,
    proposes the next candidate to evaluate.
    """

    @abstractmethod
    def propose(
        self,
        current_content: str,
        side_info: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Propose the next candidate.

        Args:
            current_content: The current candidate's content.
            side_info: Evaluation side information from the last round.
            **kwargs: Additional context.

        Returns:
            The proposed next candidate content.
        """
        ...


class MockProposer(BaseProposer):
    """Mock proposer that returns a slightly modified version of the content.

    Used for testing. Appends an iteration counter to the content.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def propose(
        self,
        current_content: str,
        side_info: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return a slightly modified content for testing."""
        self._call_count += 1
        return f"{current_content}\n# iteration {self._call_count}"


# ---------------------------------------------------------------------------
# Optimization Result
# ---------------------------------------------------------------------------


class OptimizationResult(BaseModel):
    """Result of an optimization run.

    Contains the best candidate, scores, termination info, and trajectory.

    Attributes:
        best_candidate: The best candidate found.
        train_score: Best score on training tasks.
        holdout_score: Best score on holdout tasks (None if no holdout).
        iteration_count: Total iterations completed.
        call_count: Total evaluator calls made.
        termination_reason: Which condition triggered termination.
        trajectory: History of iteration scores.
        pareto_front: Final Pareto front.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    best_candidate: dict[str, Any] = Field(
        ...,
        description="Best candidate found.",
    )
    train_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Best score on training tasks.",
    )
    holdout_score: float | None = Field(
        default=None,
        description="Best score on holdout tasks (None if no holdout).",
    )
    iteration_count: int = Field(
        ...,
        ge=0,
        description="Total iterations completed.",
    )
    call_count: int = Field(
        ...,
        ge=0,
        description="Total evaluator calls made.",
    )
    termination_reason: str | None = Field(
        default=None,
        description="Which condition triggered termination.",
    )
    trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of iteration scores.",
    )
    pareto_front: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Final Pareto front.",
    )


# ---------------------------------------------------------------------------
# Optimization Config
# ---------------------------------------------------------------------------


class OptimizationConfig(BaseModel):
    """Full configuration for an optimization run.

    Attributes:
        target: The target being optimized.
        budget: Budget constraints.
        evaluator: Evaluator adapter configuration.
        output_dir: Directory for output files.
        checkpoint_path: Path to checkpoint file.
        train_split: Name of the training split.
        holdout_split: Name of the holdout split.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    target: TargetConfig = Field(
        ...,
        description="The target being optimized.",
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Budget constraints.",
    )
    evaluator: EvaluatorAdapterConfig = Field(
        default_factory=EvaluatorAdapterConfig,
        description="Evaluator adapter configuration.",
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory for output files.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to checkpoint file.",
    )
    train_split: str = Field(
        default="train",
        description="Name of the training split.",
    )
    holdout_split: str = Field(
        default="holdout",
        description="Name of the holdout split.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_tasks(
    tasks: list[Task],
    train_split: str = "train",
    holdout_split: str = "holdout",
) -> tuple[list[Task], list[Task]]:
    """Split a task list into train and holdout sets based on a convention.

    This uses a simple convention: tasks that have an ID starting with
    the split name go into that set. In practice, tasks would have a
    ``split`` field or be loaded from a dataset with known splits.

    Args:
        tasks: All tasks.
        train_split: Name of the training split.
        holdout_split: Name of the holdout split.

    Returns:
        Tuple of (train_tasks, holdout_tasks).
    """
    # If tasks have a 'split' attribute, use it.
    # Otherwise, use task ID prefix matching.
    train: list[Task] = []
    holdout: list[Task] = []

    for task in tasks:
        task_id_lower = task.id.lower()
        if task_id_lower.startswith(holdout_split.lower()):
            holdout.append(task)
        else:
            train.append(task)

    return train, holdout


def export_best_candidate(
    best: dict[str, Any],
    path: Path,
) -> None:
    """Export the best candidate as a deployable JSON artifact.

    Args:
        best: The best candidate dict with content, score, etc.
        path: Path to write the export file.
    """
    export_data = {
        "schema_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        **best,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(export_data, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Best candidate exported to %s", path)


# ---------------------------------------------------------------------------
# Optimization Runner
# ---------------------------------------------------------------------------


class OptimizationRunner:
    """Core optimization loop with budget management and checkpointing.

    Runs the iterative optimization cycle:
    1. Evaluate current candidate on training tasks
    2. Compute aggregate score
    3. Check stop conditions
    4. Propose next candidate
    5. Save checkpoint periodically
    6. Repeat until a stop condition triggers

    Supports graceful termination via SIGTERM (writes checkpoint).

    Args:
        config: Full optimization configuration.
        train_tasks: Tasks for training/optimization.
        holdout_tasks: Tasks for holdout evaluation (optional).
        evaluate_fn: Function with signature
            ``(candidate, task, **kwargs) -> (float, dict)``.
        propose_fn: Proposer that generates next candidates.
        output_dir: Directory for output files.
        checkpoint_path: Path for checkpoint file.
        resume: Whether to resume from an existing checkpoint.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        train_tasks: list[Task],
        holdout_tasks: list[Task] | None = None,
        evaluate_fn: Callable[..., tuple[float, dict[str, Any]]] | None = None,
        propose_fn: BaseProposer | None = None,
        output_dir: Path | None = None,
        checkpoint_path: Path | None = None,
        resume: bool = False,
    ) -> None:
        self._config = config
        self._train_tasks = train_tasks
        self._holdout_tasks = holdout_tasks
        self._evaluate_fn = evaluate_fn
        self._propose_fn = propose_fn or MockProposer()
        self._output_dir = output_dir or Path(config.output_dir or ".")
        self._checkpoint_path = checkpoint_path or Path(
            config.checkpoint_path or str(self._output_dir / "checkpoint.json")
        )
        self._resume = resume

        # Internal state
        self._stop_condition = StopCondition(config.budget)
        self._pareto_front = ParetoFront()
        self._checkpoint = OptimizationCheckpoint(path=self._checkpoint_path)
        self._terminating = False

    def run(self) -> OptimizationResult:
        """Execute the optimization loop.

        Returns:
            An OptimizationResult with the best candidate and metrics.
        """
        # Set up signal handler for graceful termination
        original_handler = signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            return self._run_loop()
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGTERM, original_handler)

    def _run_loop(self) -> OptimizationResult:
        """The actual optimization loop."""
        # Check for resume
        iteration = 0
        call_count = 0
        best_score = 0.0
        best_candidate_content = self._config.target.content
        candidates: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = []
        start_time = time.monotonic()

        if self._resume:
            state = self._checkpoint.load()
            if state is not None:
                iteration = state.iteration
                call_count = state.call_count
                best_score = state.best_score
                candidates = state.candidates
                trajectory = state.trajectory
                # Restore Pareto front
                for c in state.pareto_front:
                    cid = c.get("candidate_id", f"candidate-{len(self._pareto_front.front)}")
                    self._pareto_front.update(cid, c.get("scores", {}))
                best_candidate_content = (
                    state.best_candidate.get("content", self._config.target.content)
                    if state.best_candidate
                    else self._config.target.content
                )
                start_time = time.monotonic() - state.elapsed_s
                logger.info(
                    "Resumed from checkpoint: iteration=%d, call_count=%d, best_score=%.4f",
                    iteration,
                    call_count,
                    best_score,
                )

        current_content = best_candidate_content

        while True:
            if self._terminating:
                logger.info("Graceful termination requested.")
                break

            elapsed = time.monotonic() - start_time

            # Evaluate current candidate on all training tasks
            iteration_scores: list[float] = []
            side_info_list: list[dict[str, Any]] = []
            for task in self._train_tasks:
                if self._terminating:
                    break
                score, side_info = self._evaluate_fn(  # type: ignore[misc]
                    current_content, task
                )
                iteration_scores.append(score)
                side_info_list.append(side_info)
                call_count += 1

                # Check max_calls after each task evaluation
                elapsed = time.monotonic() - start_time
                if self._stop_condition.should_stop(
                    call_count=call_count,
                    elapsed_s=elapsed,
                    best_score=best_score,
                ):
                    break

            if not iteration_scores:
                break

            # Compute aggregate score (average across tasks)
            iter_score = sum(iteration_scores) / len(iteration_scores)

            # Track candidate
            candidate_record = {
                "content": current_content,
                "score": iter_score,
                "iteration": iteration,
            }
            candidates.append(candidate_record)
            trajectory.append(
                {
                    "iteration": iteration,
                    "score": iter_score,
                    "candidate": current_content[:100],
                }
            )

            # Update Pareto front
            candidate_id = f"candidate-{iteration}"
            task_scores = {f"task-{i}": s for i, s in enumerate(iteration_scores)}
            self._pareto_front.update(candidate_id, task_scores)

            # Check for improvement
            if iter_score > best_score:
                best_score = iter_score
                best_candidate_content = current_content
                logger.info(
                    "New best score: %.4f at iteration %d",
                    best_score,
                    iteration,
                )

            # Check stop conditions (with update for no-improvement tracking)
            elapsed = time.monotonic() - start_time
            if self._stop_condition.update_and_check(
                call_count=call_count,
                elapsed_s=elapsed,
                best_score=best_score,
            ):
                logger.info(
                    "Stop condition triggered: %s",
                    self._stop_condition.termination_reason(
                        call_count=call_count,
                        elapsed_s=elapsed,
                        best_score=best_score,
                    ),
                )
                break

            # Save checkpoint
            if iteration > 0 and iteration % self._config.budget.checkpoint_interval == 0:
                self._save_checkpoint(
                    iteration=iteration + 1,
                    call_count=call_count,
                    best_score=best_score,
                    candidates=candidates,
                    trajectory=trajectory,
                    best_content=best_candidate_content,
                    elapsed_s=elapsed,
                )

            # Propose next candidate
            current_content = self._propose_fn.propose(
                current_content=current_content,
                side_info=side_info_list,
            )
            iteration += 1

        # Final checkpoint
        elapsed = time.monotonic() - start_time
        termination_reason = self._stop_condition.termination_reason(
            call_count=call_count,
            elapsed_s=elapsed,
            best_score=best_score,
        )

        # Evaluate best on holdout
        holdout_score = None
        if self._holdout_tasks and not self._terminating:
            holdout_scores: list[float] = []
            for task in self._holdout_tasks:
                score, _info = self._evaluate_fn(  # type: ignore[misc]
                    best_candidate_content, task
                )
                holdout_scores.append(score)
            if holdout_scores:
                holdout_score = sum(holdout_scores) / len(holdout_scores)

        # Build best candidate
        best_candidate = {
            "content": best_candidate_content,
            "score": best_score,
            "iteration": iteration,
            "target_type": self._config.target.type,
            "metadata": self._config.target.metadata,
        }

        # Save final checkpoint
        self._save_checkpoint(
            iteration=iteration,
            call_count=call_count,
            best_score=best_score,
            candidates=candidates,
            trajectory=trajectory,
            best_content=best_candidate_content,
            elapsed_s=elapsed,
        )

        # Save outputs
        self._output_dir.mkdir(parents=True, exist_ok=True)
        export_best_candidate(
            best_candidate,
            Path(self._output_dir) / "best_candidate.json",
        )
        self._save_trajectory(trajectory)

        return OptimizationResult(
            best_candidate=best_candidate,
            train_score=best_score,
            holdout_score=holdout_score,
            iteration_count=iteration,
            call_count=call_count,
            termination_reason=termination_reason,
            trajectory=trajectory,
            pareto_front=self._pareto_front.to_list(),
        )

    def _save_checkpoint(
        self,
        iteration: int,
        call_count: int,
        best_score: float,
        candidates: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        best_content: str,
        elapsed_s: float,
    ) -> None:
        """Save the current state to checkpoint."""
        state = CheckpointState(
            iteration=iteration,
            call_count=call_count,
            best_score=best_score,
            candidates=candidates,
            pareto_front=self._pareto_front.to_list(),
            best_candidate={
                "content": best_content,
                "score": best_score,
                "target_type": self._config.target.type,
            },
            trajectory=trajectory,
            target_type=self._config.target.type,
            seed_content=self._config.target.content,
            elapsed_s=elapsed_s,
        )
        self._checkpoint.save(state)

    def _save_trajectory(self, trajectory: list[dict[str, Any]]) -> None:
        """Save trajectory as JSONL."""
        traj_path = Path(self._output_dir) / "trajectory.jsonl"
        lines = [json.dumps(entry, default=str) for entry in trajectory]
        traj_path.write_text("\n".join(lines), encoding="utf-8")

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM by setting the terminating flag.

        The main loop checks this flag and gracefully stops, writing a
        checkpoint before exiting.
        """
        logger.info("Received SIGTERM (signal %d). Gracefully shutting down...", signum)
        self._terminating = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BaseProposer",
    "BudgetConfig",
    "CheckpointState",
    "MockProposer",
    "OptimizationCheckpoint",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationRunner",
    "ParetoFront",
    "StopCondition",
    "TargetConfig",
    "TargetType",
    "export_best_candidate",
    "serialize_target",
    "split_tasks",
]
