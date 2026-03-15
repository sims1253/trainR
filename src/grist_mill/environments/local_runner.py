"""LocalRunner — host subprocess execution environment.

Executes commands as subprocesses on the host machine, capturing stdout/stderr
separately, detecting exit codes, and enforcing timeouts via SIGTERM then SIGKILL.
Supports configurable working directory.

This is the primary runner for fast development iteration — no Docker overhead.

Validates VAL-HARNESS-06, VAL-HARNESS-07.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
from typing import IO

from grist_mill.harness.result_parser import ResultParser
from grist_mill.interfaces import BaseEnvironment
from grist_mill.schemas import (
    ExecutionOutput,
    Task,
    TaskResult,
)

logger = logging.getLogger(__name__)

# Default grace period between SIGTERM and SIGKILL (seconds).
_DEFAULT_SIGKILL_DELAY: float = 5.0


def _reader_thread(
    stream: IO[str] | None,
    result: list[str],
    lock: threading.Lock,
) -> None:
    """Read lines from a stream in a background thread.

    Accumulates lines into *result* under *lock* so the main thread
    can safely inspect partial output at any time.
    """
    if stream is None:
        return
    try:
        for line in stream:
            with lock:
                result.append(line)
    except (ValueError, OSError):
        # Stream closed — normal during process termination.
        pass


class LocalRunner(BaseEnvironment):
    """Execute commands as host subprocesses.

    Captures stdout and stderr separately, detects exit codes, and enforces
    timeouts using a two-phase kill strategy: SIGTERM first, then SIGKILL
    after a configurable grace period.

    Args:
        working_directory: Directory in which to execute commands.
            If ``None``, the current working directory is used.
        sigkill_delay: Seconds to wait between SIGTERM and SIGKILL
            when a process times out. Defaults to 5 seconds.
        result_parser: Optional ``ResultParser`` instance for converting
            execution output to ``TaskResult``. A default instance is
            created if not provided.

    Example::

        runner = LocalRunner(working_directory="/tmp/myproject")
        runner.prepare(task)
        output = runner.execute("pytest tests/", timeout=60.0)
        result = runner.run_task("pytest tests/", task_id="t1", language="python", timeout=60.0)
        runner.cleanup()
    """

    def __init__(
        self,
        *,
        working_directory: str | None = None,
        sigkill_delay: float = _DEFAULT_SIGKILL_DELAY,
        result_parser: ResultParser | None = None,
    ) -> None:
        self._working_directory: str | None = working_directory
        self._sigkill_delay: float = sigkill_delay
        self._result_parser: ResultParser = result_parser or ResultParser()
        self._prepared: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def working_directory(self) -> str | None:
        """The configured working directory (``None`` means cwd)."""
        return self._working_directory

    @property
    def sigkill_delay(self) -> float:
        """Seconds between SIGTERM and SIGKILL on timeout."""
        return self._sigkill_delay

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare()`` has been called without a matching ``cleanup()``."""
        return self._prepared

    # ------------------------------------------------------------------
    # BaseEnvironment implementation
    # ------------------------------------------------------------------

    def prepare(self, task: Task) -> None:
        """Prepare the local environment for a task.

        Validates that the configured working directory exists. If the task
        has a ``setup_command``, it is executed. A failing setup command is
        logged but does not prevent subsequent execution (the agent/harness
        is responsible for handling setup failures).

        Args:
            task: The task to prepare the environment for.

        Raises:
            FileNotFoundError: If the configured working directory does not exist.
        """
        if self._working_directory is not None and not os.path.isdir(self._working_directory):
            msg = (
                f"Working directory does not exist: {self._working_directory}. "
                f"Please create it or specify a valid path."
            )
            raise FileNotFoundError(msg)

        self._prepared = True
        logger.info(
            "Preparing local environment for task '%s' (language=%s)",
            task.id,
            task.language,
        )

        if task.setup_command:
            logger.info("Running setup command: %s", task.setup_command)
            output = self.execute(task.setup_command, timeout=float(task.timeout))
            if output.exit_code != 0:
                logger.warning(
                    "Setup command exited with code %d: %s",
                    output.exit_code,
                    output.stderr[:200] if output.stderr else "(no stderr)",
                )

    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        """Execute a command as a subprocess.

        Uses background reader threads to continuously capture stdout and
        stderr.  This ensures partial output is available even when the
        process times out.

        If the process exceeds *timeout* seconds, the entire process group
        receives SIGTERM; if it does not terminate within *sigkill_delay*
        seconds, SIGKILL is sent.

        Args:
            command: The shell command to execute.
            timeout: Maximum time in seconds before killing the process.

        Returns:
            An ``ExecutionOutput`` with captured stdout, stderr, exit_code,
            and ``timed_out`` flag.
        """
        logger.debug("Executing command: %s (timeout=%.1fs)", command, timeout)
        cwd = self._working_directory

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                # Create a new process group so we can signal the entire
                # group (shell + children) on timeout.
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            msg = f"Failed to start command: {exc}"
            logger.error(msg)
            return ExecutionOutput(
                stdout="",
                stderr=msg,
                exit_code=127,
                timed_out=False,
            )

        # Spin up reader threads so stdout/stderr are continuously drained.
        # This prevents deadlocks and ensures partial output is captured
        # even when we need to kill the process on timeout.
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        lock = threading.Lock()

        assert proc.stdout is not None  # for type checker
        assert proc.stderr is not None  # for type checker

        t_out = threading.Thread(
            target=_reader_thread, args=(proc.stdout, stdout_lines, lock), daemon=True
        )
        t_err = threading.Thread(
            target=_reader_thread, args=(proc.stderr, stderr_lines, lock), daemon=True
        )
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
            # Process completed within the timeout.
            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)

            with lock:
                stdout = "".join(stdout_lines)
                stderr = "".join(stderr_lines)

            return ExecutionOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Command timed out after %.1fs, sending SIGTERM: %s", timeout, command)
            # Collect whatever partial output has been accumulated so far.
            with lock:
                partial_stdout = "".join(stdout_lines)
                partial_stderr = "".join(stderr_lines)

            return self._terminate_process(
                proc, partial_stdout=partial_stdout, partial_stderr=partial_stderr
            )

    def cleanup(self) -> None:
        """Clean up resources allocated during prepare.

        Safe to call even if prepare was never called or execution raised.
        """
        if self._prepared:
            logger.debug("Cleaning up local environment")
            self._prepared = False

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def run_task(
        self,
        *,
        command: str,
        task_id: str,
        language: str = "python",
        timeout: float,
    ) -> TaskResult:
        """Execute a command and parse the result into a TaskResult.

        Convenience method combining ``execute()`` with ``ResultParser.parse()``.

        Args:
            command: The shell command to execute.
            task_id: The ID of the task (for the resulting TaskResult).
            language: Programming language hint for error categorization.
            timeout: Maximum time in seconds.

        Returns:
            A ``TaskResult`` with status, score, and error_category populated.
        """
        output = self.execute(command, timeout=timeout)
        return self._result_parser.parse(
            output,
            task_id=task_id,
            language=language,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _terminate_process(
        self,
        proc: subprocess.Popen[str],
        *,
        partial_stdout: str = "",
        partial_stderr: str = "",
    ) -> ExecutionOutput:
        """Terminate a timed-out process using SIGTERM, then SIGKILL.

        Signals the entire process group (shell + child processes) to ensure
        that nested commands like ``sleep`` are also terminated.

        Args:
            proc: The process to terminate.
            partial_stdout: Stdout captured before the timeout.
            partial_stderr: Stderr captured before the timeout.

        Returns:
            An ``ExecutionOutput`` with ``timed_out=True`` and any
            output captured.
        """
        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
        except OSError:
            # Process already dead — nothing to do.
            proc.wait()
            return ExecutionOutput(
                stdout=partial_stdout,
                stderr=partial_stderr,
                exit_code=-1,
                timed_out=True,
            )

        # Phase 1: Try SIGTERM on the process group (graceful).
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            # Process may already be dead; fall back to single PID.
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                proc.wait()
                return ExecutionOutput(
                    stdout=partial_stdout,
                    stderr=partial_stderr,
                    exit_code=-1,
                    timed_out=True,
                )

        try:
            proc.wait(timeout=self._sigkill_delay)
            logger.info("Process terminated gracefully after SIGTERM")
            return ExecutionOutput(
                stdout=partial_stdout,
                stderr=partial_stderr,
                exit_code=-1,
                timed_out=True,
            )
        except subprocess.TimeoutExpired:
            # Phase 2: Force SIGKILL on the process group.
            logger.warning("Process did not respond to SIGTERM, sending SIGKILL")
            with contextlib.suppress(OSError):
                os.killpg(pgid, signal.SIGKILL)
            with contextlib.suppress(OSError):
                proc.kill()

            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5.0)

            return ExecutionOutput(
                stdout=partial_stdout,
                stderr=partial_stderr,
                exit_code=-1,
                timed_out=True,
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        wd = self._working_directory or "<cwd>"
        return f"LocalRunner(working_directory={wd!r}, sigkill_delay={self._sigkill_delay!r})"
