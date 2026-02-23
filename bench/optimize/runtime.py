"""Runtime support for optimization runs.

This module provides:
- BudgetConfig: Budget limit configuration for optimization runs
- OptimizationState: State tracking for resume capability
- Checkpoint save/load for persistence
- Budget checking utilities
"""

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(self, reason: str, current: float, limit: float) -> None:
        self.reason = reason
        self.current = current
        self.limit = limit
        super().__init__(f"Budget exceeded: {reason} ({current:.2f} > {limit:.2f})")


class StopReason(str, Enum):
    """Reason why optimization stopped."""

    COMPLETED = "completed"
    """Optimization completed normally."""

    MAX_ITERATIONS = "max_iterations"
    """Reached maximum iterations."""

    MAX_TIME = "max_time"
    """Reached maximum wall-clock time."""

    MAX_TOKENS = "max_tokens"
    """Reached maximum token usage."""

    MAX_COST = "max_cost"
    """Reached maximum cost."""

    INTERRUPTED = "interrupted"
    """Received interrupt signal (SIGINT/SIGTERM)."""

    CONVERGED = "converged"
    """Optimization converged (no improvement)."""

    ERROR = "error"
    """Error during optimization."""


class BudgetConfig(BaseModel):
    """Configuration for budget limits during optimization.

    Budget limits provide hard stops to prevent runaway optimization
    runs. All limits are optional - set to None for unlimited.
    """

    max_iterations: int | None = Field(
        default=None,
        description="Maximum optimization iterations",
        ge=1,
    )

    max_time_seconds: int | None = Field(
        default=None,
        description="Maximum wall-clock time in seconds",
        ge=1,
    )

    max_tokens: int | None = Field(
        default=None,
        description="Maximum total token usage",
        ge=1,
    )

    max_cost: float | None = Field(
        default=None,
        description="Maximum cost in dollars",
        ge=0.0,
    )

    checkpoint_interval: int = Field(
        default=5,
        description="Save checkpoint every N iterations",
        ge=1,
    )

    def has_any_limit(self) -> bool:
        """Check if any budget limit is configured."""
        return any(
            [
                self.max_iterations is not None,
                self.max_time_seconds is not None,
                self.max_tokens is not None,
                self.max_cost is not None,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "max_time_seconds": self.max_time_seconds,
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "checkpoint_interval": self.checkpoint_interval,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetConfig":
        """Deserialize from dictionary."""
        return cls(
            max_iterations=data.get("max_iterations"),
            max_time_seconds=data.get("max_time_seconds"),
            max_tokens=data.get("max_tokens"),
            max_cost=data.get("max_cost"),
            checkpoint_interval=data.get("checkpoint_interval", 5),
        )


@dataclass
class BudgetUsage:
    """Current budget usage tracking."""

    iterations: int = 0
    """Number of iterations completed."""

    elapsed_seconds: float = 0.0
    """Wall-clock time elapsed."""

    total_tokens: int = 0
    """Total tokens used."""

    total_cost: float = 0.0
    """Total cost in dollars."""

    start_time: float | None = None
    """Start timestamp (monotonic time)."""

    def start(self) -> None:
        """Start the timer."""
        if self.start_time is None:
            self.start_time = time.monotonic()

    def update_elapsed(self) -> None:
        """Update elapsed time from start."""
        if self.start_time is not None:
            self.elapsed_seconds = time.monotonic() - self.start_time

    def add_tokens(self, tokens: int) -> None:
        """Add token usage."""
        self.total_tokens += tokens

    def add_cost(self, cost: float) -> None:
        """Add cost."""
        self.total_cost += cost

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetUsage":
        """Deserialize from dictionary."""
        usage = cls(
            iterations=data.get("iterations", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
        )
        # Note: start_time is not restored - it's only for the current session
        return usage


def check_budget(usage: BudgetUsage, config: BudgetConfig) -> StopReason | None:
    """Check if any budget limit has been exceeded.

    Args:
        usage: Current budget usage
        config: Budget configuration

    Returns:
        StopReason if budget exceeded, None otherwise
    """
    # Update elapsed time before checking
    usage.update_elapsed()

    # Check iteration limit
    if config.max_iterations is not None and usage.iterations >= config.max_iterations:
        logger.info(
            f"Budget exceeded: max_iterations ({usage.iterations} >= {config.max_iterations})"
        )
        return StopReason.MAX_ITERATIONS

    # Check time limit
    if config.max_time_seconds is not None and usage.elapsed_seconds >= config.max_time_seconds:
        logger.info(
            f"Budget exceeded: max_time ({usage.elapsed_seconds:.1f}s >= {config.max_time_seconds}s)"
        )
        return StopReason.MAX_TIME

    # Check token limit
    if config.max_tokens is not None and usage.total_tokens >= config.max_tokens:
        logger.info(f"Budget exceeded: max_tokens ({usage.total_tokens} >= {config.max_tokens})")
        return StopReason.MAX_TOKENS

    # Check cost limit
    if config.max_cost is not None and usage.total_cost >= config.max_cost:
        logger.info(
            f"Budget exceeded: max_cost (${usage.total_cost:.2f} >= ${config.max_cost:.2f})"
        )
        return StopReason.MAX_COST

    return None


# Type variable for candidate type
CandidateT = TypeVar("CandidateT")


@dataclass
class TrajectoryEntry(Generic[CandidateT]):
    """A single entry in the optimization trajectory."""

    iteration: int
    """Iteration number."""

    candidate: dict[str, Any]
    """Candidate configuration (serialized)."""

    candidate_hash: str
    """Hash of the candidate for deduplication."""

    score: float
    """Evaluation score for this candidate."""

    is_best: bool = False
    """Whether this is the best candidate so far."""

    timestamp: float = field(default_factory=time.time)
    """When this entry was recorded."""

    tokens_used: int = 0
    """Tokens used for this iteration."""

    cost: float = 0.0
    """Cost for this iteration."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "iteration": self.iteration,
            "candidate": self.candidate,
            "candidate_hash": self.candidate_hash,
            "score": self.score,
            "is_best": self.is_best,
            "timestamp": self.timestamp,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryEntry":
        """Deserialize from dictionary."""
        return cls(
            iteration=data["iteration"],
            candidate=data["candidate"],
            candidate_hash=data["candidate_hash"],
            score=data["score"],
            is_best=data.get("is_best", False),
            timestamp=data.get("timestamp", time.time()),
            tokens_used=data.get("tokens_used", 0),
            cost=data.get("cost", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OptimizationState(Generic[CandidateT]):
    """Complete state of an optimization run for resume capability.

    This captures all information needed to resume an interrupted
    optimization run exactly where it left off.
    """

    # Run identification
    run_id: str
    """Unique identifier for this run."""

    target_type: str
    """Type of target being optimized."""

    target_fingerprint: str
    """Fingerprint of the target for validation."""

    # Configuration
    budget_config: BudgetConfig
    """Budget configuration."""

    # Progress tracking
    iteration: int = 0
    """Current iteration number."""

    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    """History of all evaluated candidates."""

    # Best candidate tracking
    best_candidate: dict[str, Any] | None = None
    """Best candidate found so far (serialized)."""

    best_score: float = float("-inf")
    """Score of the best candidate."""

    best_iteration: int = 0
    """Iteration where best was found."""

    # Usage tracking
    budget_usage: BudgetUsage = field(default_factory=BudgetUsage)
    """Budget usage tracking."""

    # Status
    stop_reason: StopReason | None = None
    """Why optimization stopped (if stopped)."""

    started_at: float = field(default_factory=time.time)
    """When optimization started."""

    completed_at: float | None = None
    """When optimization completed."""

    error_message: str | None = None
    """Error message if optimization failed."""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional run metadata."""

    def add_entry(
        self,
        candidate: dict[str, Any],
        candidate_hash: str,
        score: float,
        tokens_used: int = 0,
        cost: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryEntry:
        """Add a trajectory entry and update state.

        Args:
            candidate: Serialized candidate
            candidate_hash: Hash of candidate
            score: Evaluation score
            tokens_used: Tokens used this iteration
            cost: Cost this iteration
            metadata: Additional metadata

        Returns:
            The created TrajectoryEntry
        """
        self.iteration += 1
        self.budget_usage.iterations = self.iteration
        self.budget_usage.add_tokens(tokens_used)
        self.budget_usage.add_cost(cost)

        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_candidate = candidate
            self.best_iteration = self.iteration

        entry = TrajectoryEntry(
            iteration=self.iteration,
            candidate=candidate,
            candidate_hash=candidate_hash,
            score=score,
            is_best=is_best,
            tokens_used=tokens_used,
            cost=cost,
            metadata=metadata or {},
        )

        self.trajectory.append(entry)
        return entry

    def mark_complete(self, reason: StopReason) -> None:
        """Mark optimization as complete."""
        self.stop_reason = reason
        self.completed_at = time.time()
        self.budget_usage.update_elapsed()

    def mark_error(self, error: str) -> None:
        """Mark optimization as failed with error."""
        self.error_message = error
        self.stop_reason = StopReason.ERROR
        self.completed_at = time.time()
        self.budget_usage.update_elapsed()

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "run_id": self.run_id,
            "target_type": self.target_type,
            "target_fingerprint": self.target_fingerprint,
            "budget_config": self.budget_config.to_dict(),
            "iteration": self.iteration,
            "trajectory": [entry.to_dict() for entry in self.trajectory],
            "best_candidate": self.best_candidate,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "budget_usage": self.budget_usage.to_dict(),
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationState":
        """Deserialize state from dictionary."""
        stop_reason = None
        if data.get("stop_reason"):
            stop_reason = StopReason(data["stop_reason"])

        return cls(
            run_id=data["run_id"],
            target_type=data["target_type"],
            target_fingerprint=data["target_fingerprint"],
            budget_config=BudgetConfig.from_dict(data.get("budget_config", {})),
            iteration=data.get("iteration", 0),
            trajectory=[TrajectoryEntry.from_dict(e) for e in data.get("trajectory", [])],
            best_candidate=data.get("best_candidate"),
            best_score=data.get("best_score", float("-inf")),
            best_iteration=data.get("best_iteration", 0),
            budget_usage=BudgetUsage.from_dict(data.get("budget_usage", {})),
            stop_reason=stop_reason,
            started_at=data.get("started_at", time.time()),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


# Checkpoint file name
CHECKPOINT_FILE = "checkpoint.json"
STATE_FILE = "optimization_state.json"


def save_checkpoint(
    state: OptimizationState, run_dir: Path, filename: str = CHECKPOINT_FILE
) -> Path:
    """Save optimization state to checkpoint file.

    Args:
        state: Optimization state to save
        run_dir: Run directory
        filename: Checkpoint filename

    Returns:
        Path to saved checkpoint
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / filename

    # Write atomically using temp file
    temp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        temp_path.rename(checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise

    return checkpoint_path


def load_checkpoint(run_dir: Path, filename: str = CHECKPOINT_FILE) -> OptimizationState | None:
    """Load optimization state from checkpoint file.

    Args:
        run_dir: Run directory
        filename: Checkpoint filename

    Returns:
        OptimizationState if checkpoint exists, None otherwise
    """
    checkpoint_path = run_dir / filename

    if not checkpoint_path.exists():
        logger.debug(f"No checkpoint found at {checkpoint_path}")
        return None

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
        state = OptimizationState.from_dict(data)
        logger.info(f"Loaded checkpoint from {checkpoint_path} (iteration {state.iteration})")
        return state
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def has_checkpoint(run_dir: Path, filename: str = CHECKPOINT_FILE) -> bool:
    """Check if a checkpoint exists.

    Args:
        run_dir: Run directory
        filename: Checkpoint filename

    Returns:
        True if checkpoint exists
    """
    return (run_dir / filename).exists()


class InterruptHandler:
    """Handler for graceful interrupt signals.

    Catches SIGINT and SIGTERM to allow graceful shutdown with
    state preservation.

    Usage:
        handler = InterruptHandler()
        while not handler.interrupted and not done:
            # Do work
            ...
            if handler.should_checkpoint():
                save_checkpoint(state, run_dir)
    """

    def __init__(self) -> None:
        """Initialize the interrupt handler."""
        self.interrupted = False
        self.interrupt_count = 0
        self._original_sigint = None
        self._original_sigterm = None

    def __enter__(self) -> "InterruptHandler":
        """Enter context manager, install signal handlers."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal."""
        self.interrupt_count += 1
        sig_name = signal.Signals(signum).name

        if self.interrupt_count == 1:
            logger.warning(
                f"Received {sig_name}, gracefully shutting down after current iteration..."
            )
            self.interrupted = True
        else:
            logger.warning(f"Received {sig_name} ({self.interrupt_count}x), forcing exit...")
            # Restore original handler and re-raise
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            # Re-raise the signal
            signal.raise_signal(signum)

    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint (on interrupt)."""
        return self.interrupted


class OptimizationRun:
    """Manages a complete optimization run with resume support.

    This class coordinates:
    - Budget tracking
    - Checkpoint persistence
    - Interrupt handling
    - State management

    Usage:
        config = BudgetConfig(max_iterations=100)
        run = OptimizationRun(run_dir, config, target)

        if resume and run.can_resume():
            run.resume()
        else:
            run.start()

        while not run.is_complete():
            candidate = propose_candidate()
            score = evaluate(candidate)
            run.record(candidate, score)

            if run.should_checkpoint():
                run.save_checkpoint()

        run.finalize()
    """

    def __init__(
        self,
        run_dir: Path,
        budget_config: BudgetConfig,
        target_type: str,
        target_fingerprint: str,
        run_id: str | None = None,
    ) -> None:
        """Initialize the optimization run.

        Args:
            run_dir: Directory for run outputs
            budget_config: Budget configuration
            target_type: Type of optimization target
            target_fingerprint: Fingerprint of the target
            run_id: Optional run ID (generated if not provided)
        """
        self.run_dir = Path(run_dir)
        self.budget_config = budget_config
        self.target_type = target_type
        self.target_fingerprint = target_fingerprint

        # Initialize state (will be loaded or created)
        self.state: OptimizationState | None = None
        self._run_id = run_id

        # Interrupt handler
        self._interrupt_handler = InterruptHandler()

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        if self.state:
            return self.state.run_id
        return self._run_id or "unknown"

    def can_resume(self) -> bool:
        """Check if this run can be resumed from a checkpoint."""
        return has_checkpoint(self.run_dir)

    def resume(self) -> OptimizationState:
        """Resume from existing checkpoint.

        Returns:
            The loaded OptimizationState

        Raises:
            FileNotFoundError: If no checkpoint exists
            ValueError: If checkpoint is invalid
        """
        state = load_checkpoint(self.run_dir)
        if state is None:
            raise FileNotFoundError(f"No checkpoint found at {self.run_dir}")

        # Validate fingerprint matches
        if state.target_fingerprint != self.target_fingerprint:
            raise ValueError(
                f"Target fingerprint mismatch: "
                f"checkpoint={state.target_fingerprint}, "
                f"expected={self.target_fingerprint}"
            )

        # Reset the timer for the resumed session
        state.budget_usage.start()
        state.stop_reason = None
        state.error_message = None

        self.state = state
        logger.info(f"Resumed run {state.run_id} from iteration {state.iteration}")
        return state

    def start(self, metadata: dict[str, Any] | None = None) -> OptimizationState:
        """Start a new optimization run.

        Args:
            metadata: Optional metadata to include

        Returns:
            The new OptimizationState
        """
        from datetime import datetime

        run_id = self._run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.state = OptimizationState(
            run_id=run_id,
            target_type=self.target_type,
            target_fingerprint=self.target_fingerprint,
            budget_config=self.budget_config,
            metadata=metadata or {},
        )
        self.state.budget_usage.start()

        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Started new run {run_id}")
        return self.state

    def record(
        self,
        candidate: dict[str, Any],
        candidate_hash: str,
        score: float,
        tokens_used: int = 0,
        cost: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryEntry:
        """Record an evaluation result.

        Args:
            candidate: Serialized candidate
            candidate_hash: Hash of the candidate
            score: Evaluation score
            tokens_used: Tokens used
            cost: Cost
            metadata: Additional metadata

        Returns:
            The created TrajectoryEntry

        Raises:
            RuntimeError: If run not started
        """
        if self.state is None:
            raise RuntimeError("Run not started - call start() or resume() first")

        return self.state.add_entry(
            candidate=candidate,
            candidate_hash=candidate_hash,
            score=score,
            tokens_used=tokens_used,
            cost=cost,
            metadata=metadata,
        )

    def is_complete(self) -> bool:
        """Check if optimization is complete.

        Returns:
            True if optimization should stop
        """
        if self.state is None:
            return True

        # Check for interrupt
        if self._interrupt_handler.interrupted:
            return True

        # Check for error
        if self.state.error_message:
            return True

        # Check budget
        budget_result = check_budget(self.state.budget_usage, self.budget_config)
        return budget_result is not None

    def get_stop_reason(self) -> StopReason:
        """Get the reason optimization stopped."""
        if self.state is None:
            return StopReason.ERROR

        if self.state.error_message:
            return StopReason.ERROR

        if self._interrupt_handler.interrupted:
            return StopReason.INTERRUPTED

        budget_result = check_budget(self.state.budget_usage, self.budget_config)
        if budget_result:
            return budget_result

        return StopReason.COMPLETED

    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint now.

        Returns:
            True if checkpoint should be saved
        """
        # Checkpoint on interrupt
        if self._interrupt_handler.should_checkpoint():
            return True

        # Checkpoint at configured interval
        if self.state is None:
            return False

        interval = self.budget_config.checkpoint_interval
        return self.state.iteration > 0 and self.state.iteration % interval == 0

    def save_checkpoint(self) -> Path:
        """Save current state to checkpoint.

        Returns:
            Path to saved checkpoint

        Raises:
            RuntimeError: If run not started
        """
        if self.state is None:
            raise RuntimeError("Run not started - call start() or resume() first")

        return save_checkpoint(self.state, self.run_dir)

    def finalize(self, reason: StopReason | None = None) -> OptimizationState:
        """Finalize the optimization run.

        Args:
            reason: Optional explicit stop reason

        Returns:
            The final OptimizationState

        Raises:
            RuntimeError: If run not started
        """
        if self.state is None:
            raise RuntimeError("Run not started - call start() or resume() first")

        final_reason = reason or self.get_stop_reason()
        self.state.mark_complete(final_reason)

        # Final checkpoint
        self.save_checkpoint()

        logger.info(f"Run {self.state.run_id} finalized: {final_reason.value}")
        return self.state

    def __enter__(self) -> "OptimizationRun":
        """Enter context manager."""
        self._interrupt_handler.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager with final save on interrupt."""
        if self.state is not None:
            # Save state on any exit
            try:
                self.save_checkpoint()
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")

        self._interrupt_handler.__exit__(exc_type, exc_val, exc_tb)
