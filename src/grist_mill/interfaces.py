"""Abstract base classes for the grist-mill framework.

Defines the core interfaces that all implementations must follow:

- BaseAgent: Agent that runs tasks and produces results
- BaseBenchmark: Benchmark lifecycle (setup, teardown, result collection)
- BaseEnvironment: Execution environment (prepare, execute, cleanup)
- BaseHarness: Orchestrator wiring task -> environment -> agent
- LocalEnvironment: Reference local-process environment implementation
- LocalHarness: Reference harness using LocalEnvironment

Validates VAL-FOUND-06, VAL-FOUND-07, VAL-FOUND-08, VAL-FOUND-09.
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod

from grist_mill.schemas import (
    ExecutionOutput,
    HarnessConfig,
    Task,
    TaskResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseAgent (VAL-FOUND-07)
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract interface for task-executing agents.

    Subclasses must implement the ``run`` method which accepts a task
    definition and harness configuration, producing a ``TaskResult``.

    A subclass that does not implement ``run`` will fail at instantiation
    with a ``TypeError`` indicating the unimplemented method.
    """

    @abstractmethod
    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        """Execute a task and return the result.

        Args:
            task: The task definition to execute.
            config: The harness configuration including agent and environment settings.

        Returns:
            A ``TaskResult`` capturing the outcome of the execution.
        """
        ...


# ---------------------------------------------------------------------------
# BaseBenchmark (VAL-FOUND-06)
# ---------------------------------------------------------------------------


class BaseBenchmark(ABC):
    """Abstract interface for benchmark lifecycle management.

    Subclasses must implement ``setup``, ``teardown``, and ``submit_result``.
    The ``submit_result`` method validates that the input is a ``TaskResult``
    instance, raising ``TypeError`` for any other type.
    """

    @abstractmethod
    def setup(self) -> None:
        """Prepare the benchmark for execution.

        Called before any tasks are run. Use this to initialize resources,
        load configurations, or prepare environments.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources after benchmark execution.

        Called after all tasks have been completed (or the benchmark
        is interrupted). Use this to release resources, write summaries,
        or persist state.
        """
        ...

    @abstractmethod
    def _submit_result(self, result: TaskResult) -> None:
        """Internal method to handle the submitted result.

        Subclasses implement this to add collection logic.

        Args:
            result: The validated task result to submit.
        """
        ...

    def submit_result(self, result: TaskResult) -> None:
        """Submit a task result to the benchmark.

        Validates that ``result`` is a ``TaskResult`` instance before
        delegating to the subclass-specific ``_submit_result`` implementation.

        Args:
            result: The task result to submit.

        Raises:
            TypeError: If ``result`` is not a ``TaskResult`` instance.
        """
        if not isinstance(result, TaskResult):
            raise TypeError(
                f"submit_result expects a TaskResult instance, "
                f"got {type(result).__name__}: {result!r}"
            )
        self._submit_result(result)


# ---------------------------------------------------------------------------
# BaseEnvironment (VAL-FOUND-08)
# ---------------------------------------------------------------------------


class BaseEnvironment(ABC):
    """Abstract interface for execution environments.

    Subclasses must implement ``prepare``, ``execute``, and ``cleanup``.
    The ``execute`` method returns a typed ``ExecutionOutput`` containing
    stdout, stderr, exit_code, and timed_out fields.
    """

    @abstractmethod
    def prepare(self, task: Task) -> None:
        """Prepare the environment for executing a task.

        Args:
            task: The task to prepare the environment for.
                Implementations may use task fields like ``setup_command``,
                ``dependencies``, and ``language`` to configure the environment.
        """
        ...

    @abstractmethod
    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        """Execute a command in the environment.

        Args:
            command: The shell command to execute.
            timeout: Maximum time in seconds before killing the process.

        Returns:
            An ``ExecutionOutput`` with stdout, stderr, exit_code, and timed_out.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources allocated during prepare.

        Must be safe to call even if prepare was never called or
        if the previous execution raised an exception.
        """
        ...


# ---------------------------------------------------------------------------
# BaseHarness (VAL-FOUND-09)
# ---------------------------------------------------------------------------


class BaseHarness(ABC):
    """Abstract orchestrator that wires task -> environment -> agent.

    The ``run`` method orchestrates the full execution lifecycle:

    1. ``env.prepare(task)`` — set up the environment
    2. ``agent.run(task, config)`` — execute the task
    3. ``env.cleanup()`` — tear down the environment

    Cleanup is **guaranteed** to run even when the agent or environment
    raises an exception.
    """

    def _run(
        self,
        task: Task,
        agent: BaseAgent,
        env: BaseEnvironment,
    ) -> TaskResult:
        """Internal orchestration with guaranteed cleanup.

        This method is the reference implementation of the harness pattern.
        It ensures ``env.cleanup()`` is always called, regardless of whether
        ``env.prepare()`` or ``agent.run()`` raises an exception.

        Args:
            task: The task to execute.
            agent: The agent that will run the task.
            env: The environment to prepare and clean up.

        Returns:
            The ``TaskResult`` from the agent.

        Raises:
            Exception: Any exception from ``env.prepare`` or ``agent.run``
                is re-raised after cleanup.
        """
        try:
            env.prepare(task)
        except Exception:
            # Even if prepare fails, ensure cleanup runs.
            env.cleanup()
            raise

        try:
            return agent.run(task, self._config)
        finally:
            env.cleanup()

    @property
    @abstractmethod
    def _config(self) -> HarnessConfig:
        """The harness configuration."""
        ...


# ---------------------------------------------------------------------------
# LocalEnvironment — reference implementation
# ---------------------------------------------------------------------------


class LocalEnvironment(BaseEnvironment):
    """Local-process execution environment.

    Executes commands as subprocesses on the host machine. Suitable for
    fast development iteration without Docker overhead.
    """

    def __init__(self) -> None:
        self._prepared = False

    def prepare(self, task: Task) -> None:
        """Prepare the local environment for a task.

        If the task has a ``setup_command``, it is executed. A failing
        setup command is logged but does not prevent subsequent execution
        (the agent/harness is responsible for handling setup failures).
        """
        self._prepared = True
        logger.info(
            "Preparing local environment for task '%s' (language=%s)",
            task.id,
            task.language,
        )

        if task.setup_command:
            logger.info("Running setup command: %s", task.setup_command)
            output = self.execute(task.setup_command, timeout=task.timeout)
            if output.exit_code != 0:
                logger.warning(
                    "Setup command exited with code %d: %s",
                    output.exit_code,
                    output.stderr[:200] if output.stderr else "(no stderr)",
                )

    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        """Execute a command as a subprocess.

        Args:
            command: The shell command to execute.
            timeout: Maximum time in seconds before killing the process.

        Returns:
            An ``ExecutionOutput`` with captured output and timing info.
        """
        logger.debug("Executing command: %s (timeout=%.1fs)", command, timeout)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecutionOutput(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if exc.stdout else ""
            stderr = exc.stderr.decode() if exc.stderr else ""
            logger.warning("Command timed out after %.1fs: %s", timeout, command)
            return ExecutionOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
            )

    def cleanup(self) -> None:
        """Clean up the local environment.

        Currently a no-op for local environments, but provided for
        interface compliance and future resource cleanup.
        """
        if self._prepared:
            logger.debug("Cleaning up local environment")
            self._prepared = False


# ---------------------------------------------------------------------------
# LocalHarness — reference implementation
# ---------------------------------------------------------------------------


class LocalHarness(BaseHarness):
    """Reference harness implementation using local environment.

    Wires a task to a local-process environment and an agent,
    ensuring proper lifecycle management with guaranteed cleanup.
    """

    def __init__(self, config: HarnessConfig) -> None:
        self._harness_config = config

    @property
    def _config(self) -> HarnessConfig:
        """Return the harness configuration."""
        return self._harness_config

    def run(
        self,
        task: Task,
        agent: BaseAgent,
        env: BaseEnvironment,
    ) -> TaskResult:
        """Run a task through the agent with the given environment.

        Args:
            task: The task to execute.
            agent: The agent that will run the task.
            env: The environment to prepare and clean up.

        Returns:
            The ``TaskResult`` from the agent.
        """
        return self._run(task, agent, env)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BaseAgent",
    "BaseBenchmark",
    "BaseEnvironment",
    "BaseHarness",
    "LocalEnvironment",
    "LocalHarness",
]
