"""Tests for LocalRunner (BaseEnvironment subclass).

Validates VAL-HARNESS-06, VAL-HARNESS-07:
- LocalRunner executes commands as subprocesses, capturing stdout/stderr/exit_code
- Returns typed ExecutionOutput with timed_out flag
- Timeout kills process (SIGTERM, then SIGKILL) and marks timed_out=True
- Result parsing converts raw output to TaskResult
- Working directory is configurable per task
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grist_mill.environments.local_runner import LocalRunner
from grist_mill.harness.result_parser import ResultParser
from grist_mill.schemas import (
    Difficulty,
    ErrorCategory,
    ExecutionOutput,
    Task,
    TaskResult,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TASK_ID = "test-local-001"


@pytest.fixture()
def sample_task() -> Task:
    """Return a valid Task for testing."""
    return Task(
        id=_SAMPLE_TASK_ID,
        prompt="Fix the bug in the function.",
        language="python",
        test_command="echo done",
        timeout=30,
        difficulty=Difficulty.EASY,
    )


@pytest.fixture()
def runner() -> LocalRunner:
    """Return a LocalRunner with short sigkill delay for fast tests."""
    return LocalRunner(sigkill_delay=0.5)


# ===========================================================================
# 1. Basic execution — stdout/stderr/exit_code capture
# ===========================================================================


class TestBasicExecution:
    """LocalRunner executes commands as subprocesses capturing stdout/stderr/exit_code."""

    def test_echo_command_captures_stdout(self, runner: LocalRunner, sample_task: Task) -> None:
        """echo command captures stdout correctly."""
        runner.prepare(sample_task)
        output = runner.execute("echo hello world", timeout=10.0)
        runner.cleanup()

        assert isinstance(output, ExecutionOutput)
        assert "hello world" in output.stdout
        assert output.exit_code == 0
        assert output.timed_out is False

    def test_echo_command_stderr_capture(self, runner: LocalRunner, sample_task: Task) -> None:
        """stderr is captured separately from stdout."""
        runner.prepare(sample_task)
        output = runner.execute("echo error_message >&2", timeout=10.0)
        runner.cleanup()

        assert "error_message" in output.stderr
        # stdout should be empty (message went to stderr)
        assert output.stdout.strip() == ""

    def test_both_stdout_and_stderr(self, runner: LocalRunner, sample_task: Task) -> None:
        """Both stdout and stderr are captured independently."""
        runner.prepare(sample_task)
        output = runner.execute("echo out_msg && echo err_msg >&2", timeout=10.0)
        runner.cleanup()

        assert "out_msg" in output.stdout
        assert "err_msg" in output.stderr

    def test_nonzero_exit_code(self, runner: LocalRunner, sample_task: Task) -> None:
        """Non-zero exit codes are captured correctly."""
        runner.prepare(sample_task)
        output = runner.execute("exit 42", timeout=10.0)
        runner.cleanup()

        assert output.exit_code == 42
        assert output.timed_out is False

    def test_exit_code_1(self, runner: LocalRunner, sample_task: Task) -> None:
        """Exit code 1 is captured for failed commands."""
        runner.prepare(sample_task)
        output = runner.execute("false", timeout=10.0)
        runner.cleanup()

        assert output.exit_code == 1
        assert output.timed_out is False

    def test_empty_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Commands producing no output return empty strings."""
        runner.prepare(sample_task)
        output = runner.execute("true", timeout=10.0)
        runner.cleanup()

        assert output.stdout == ""
        assert output.stderr == ""
        assert output.exit_code == 0

    def test_unicode_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Unicode characters in output are preserved."""
        runner.prepare(sample_task)
        output = runner.execute("echo '✓ 测试通过 🎉'", timeout=10.0)
        runner.cleanup()

        assert "✓" in output.stdout
        assert "测试通过" in output.stdout


# ===========================================================================
# 2. Timeout handling — SIGTERM then SIGKILL
# ===========================================================================


class TestTimeoutHandling:
    """Timeout kills process (SIGTERM, then SIGKILL) and marks timed_out=True.

    Validates VAL-HARNESS-07.
    """

    def test_timeout_marks_timed_out_true(self, runner: LocalRunner, sample_task: Task) -> None:
        """When timeout is exceeded, timed_out is True."""
        runner.prepare(sample_task)
        output = runner.execute("sleep 60", timeout=0.5)
        runner.cleanup()

        assert output.timed_out is True

    def test_timeout_captures_partial_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Partial output is captured even when process times out."""
        runner.prepare(sample_task)
        # Command that produces some output then sleeps
        output = runner.execute("echo 'partial' && sleep 60", timeout=0.5)
        runner.cleanup()

        assert output.timed_out is True
        assert "partial" in output.stdout

    def test_timeout_partial_stderr(self, runner: LocalRunner, sample_task: Task) -> None:
        """Partial stderr is captured on timeout."""
        runner.prepare(sample_task)
        output = runner.execute("echo 'err' >&2 && sleep 60", timeout=0.5)
        runner.cleanup()

        assert output.timed_out is True
        assert "err" in output.stderr

    def test_no_timeout_when_fast_command(self, runner: LocalRunner, sample_task: Task) -> None:
        """Fast commands should not be marked as timed out."""
        runner.prepare(sample_task)
        output = runner.execute("echo fast", timeout=10.0)
        runner.cleanup()

        assert output.timed_out is False

    def test_timeout_exit_code(self, runner: LocalRunner, sample_task: Task) -> None:
        """Exit code on timeout should be -1 (signal-based)."""
        runner.prepare(sample_task)
        output = runner.execute("sleep 60", timeout=0.5)
        runner.cleanup()

        assert output.timed_out is True
        assert output.exit_code == -1

    def test_sigterm_then_sigkill_sequence(self, runner: LocalRunner, sample_task: Task) -> None:
        """Verify SIGTERM is sent first, then SIGKILL after grace period.

        We mock Popen and os.killpg/os.getpgid to verify the signal sequence.
        """
        runner.prepare(sample_task)

        signals_sent: list[int] = []

        def mock_getpgid(pid: int) -> int:
            return pid

        def mock_killpg(pgid: int, sig: int) -> None:
            signals_sent.append(sig)
            if sig == signal.SIGTERM:
                # After SIGTERM, wait() will raise TimeoutExpired
                pass
            elif sig == signal.SIGKILL:
                # After SIGKILL, wait() succeeds
                pass

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = -9

        # First wait() call raises TimeoutExpired (simulate timeout)
        # Second wait() call also raises TimeoutExpired (simulate SIGTERM ignored)
        # Third wait() succeeds (after SIGKILL)
        wait_call_count = 0

        def mock_wait(timeout: float = 0) -> None:
            nonlocal wait_call_count
            wait_call_count += 1
            if wait_call_count <= 2:
                raise subprocess.TimeoutExpired(cmd="sleep 60", timeout=timeout)

        mock_proc.wait = mock_wait

        with (
            patch("grist_mill.environments.local_runner.subprocess.Popen", return_value=mock_proc),
            patch("grist_mill.environments.local_runner.os.getpgid", side_effect=mock_getpgid),
            patch("grist_mill.environments.local_runner.os.killpg", side_effect=mock_killpg),
            patch("grist_mill.environments.local_runner._reader_thread"),
        ):
            output = runner.execute("sleep 60", timeout=0.5)

        assert output.timed_out is True
        assert signal.SIGTERM in signals_sent
        assert signal.SIGKILL in signals_sent
        # Verify SIGTERM was sent before SIGKILL
        assert signals_sent.index(signal.SIGTERM) < signals_sent.index(signal.SIGKILL)

        runner.cleanup()


# ===========================================================================
# 3. Configurable working directory
# ===========================================================================


class TestWorkingDirectory:
    """Working directory is configurable per task.

    Validates the expectedBehavior: 'Working directory is configurable per task'.
    """

    def test_execute_in_cwd_by_default(self, runner: LocalRunner, sample_task: Task) -> None:
        """By default, commands execute in the current working directory."""
        runner.prepare(sample_task)
        output = runner.execute("pwd", timeout=10.0)
        runner.cleanup()

        # Should be the current working directory
        cwd = os.getcwd()
        assert output.stdout.strip() == cwd

    def test_execute_in_custom_working_directory(
        self, runner: LocalRunner, sample_task: Task
    ) -> None:
        """Commands execute in the configured working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalRunner(working_directory=tmpdir)
            runner.prepare(sample_task)
            output = runner.execute("pwd", timeout=10.0)
            runner.cleanup()

            assert output.stdout.strip() == tmpdir

    def test_working_directory_with_relative_command(
        self, runner: LocalRunner, sample_task: Task
    ) -> None:
        """Relative paths are resolved against the configured working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in the temp dir
            test_file = Path(tmpdir) / "test_marker.txt"
            test_file.write_text("hello")

            runner = LocalRunner(working_directory=tmpdir)
            runner.prepare(sample_task)
            output = runner.execute("cat test_marker.txt", timeout=10.0)
            runner.cleanup()

            assert output.exit_code == 0
            assert "hello" in output.stdout

    def test_working_directory_nonexistent(self, sample_task: Task) -> None:
        """Non-existent working directory should raise a clear error."""
        runner = LocalRunner(working_directory="/nonexistent/path/that/does/not/exist")
        with pytest.raises(FileNotFoundError):
            runner.prepare(sample_task)


# ===========================================================================
# 4. Result parsing integration
# ===========================================================================


class TestResultParsing:
    """Result parsing converts raw output to TaskResult.

    Validates: 'Result parsing converts raw output to TaskResult'.
    """

    def test_parse_success_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Successful execution (exit 0) produces TaskResult with status=SUCCESS."""
        runner.prepare(sample_task)
        output = runner.execute("echo 'all tests passed'", timeout=10.0)
        runner.cleanup()

        parser = ResultParser()
        result = parser.parse(output, task_id=sample_task.id, language=sample_task.language)

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.error_category is None

    def test_parse_failure_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Failed execution produces TaskResult with status=FAILURE."""
        runner.prepare(sample_task)
        output = runner.execute("echo '2 failed' && exit 1", timeout=10.0)
        runner.cleanup()

        parser = ResultParser()
        result = parser.parse(output, task_id=sample_task.id, language=sample_task.language)

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.FAILURE
        assert result.score == 0.0

    def test_parse_timeout_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Timed-out execution produces TaskResult with status=TIMEOUT."""
        runner.prepare(sample_task)
        output = runner.execute("echo 'partial' && sleep 60", timeout=0.5)
        runner.cleanup()

        parser = ResultParser()
        result = parser.parse(output, task_id=sample_task.id, language=sample_task.language)

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.TIMEOUT
        assert result.score == 0.0

    def test_parse_syntax_error_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """Syntax error in output produces TaskResult with SYNTAX_ERROR."""
        runner.prepare(sample_task)
        output = runner.execute("echo 'SyntaxError: invalid syntax' >&2 && exit 1", timeout=10.0)
        runner.cleanup()

        parser = ResultParser()
        result = parser.parse(output, task_id=sample_task.id, language=sample_task.language)

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR

    def test_run_with_result_parser_method(self, runner: LocalRunner, sample_task: Task) -> None:
        """LocalRunner.run_task convenience method parses results."""
        runner.prepare(sample_task)
        result = runner.run_task(
            command="echo 'tests passed'",
            task_id=sample_task.id,
            language=sample_task.language,
            timeout=10.0,
        )
        runner.cleanup()

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.task_id == sample_task.id


# ===========================================================================
# 5. Environment lifecycle
# ===========================================================================


class TestEnvironmentLifecycle:
    """LocalRunner properly manages prepare/execute/cleanup lifecycle."""

    def test_prepare_sets_up_environment(self, runner: LocalRunner, sample_task: Task) -> None:
        """prepare() initializes the runner for task execution."""
        runner.prepare(sample_task)
        assert runner.is_prepared
        runner.cleanup()

    def test_cleanup_after_prepare(self, runner: LocalRunner, sample_task: Task) -> None:
        """cleanup() after prepare() resets the prepared state."""
        runner.prepare(sample_task)
        runner.cleanup()
        assert not runner.is_prepared

    def test_cleanup_without_prepare(self, runner: LocalRunner) -> None:
        """cleanup() without prepare() is safe (no-op)."""
        runner.cleanup()  # Should not raise
        assert not runner.is_prepared

    def test_execute_without_prepare(self, runner: LocalRunner) -> None:
        """execute() without prepare() still works (prepare is optional for simple commands)."""
        output = runner.execute("echo hello", timeout=10.0)
        assert output.exit_code == 0
        assert "hello" in output.stdout

    def test_setup_command_executed_on_prepare(
        self, runner: LocalRunner, sample_task: Task
    ) -> None:
        """Task's setup_command is executed during prepare."""
        sample_task.setup_command = "echo 'setup ran'"
        runner.prepare(sample_task)

        # The setup command output is logged but doesn't block execution
        output = runner.execute("echo 'main command'", timeout=10.0)
        runner.cleanup()

        assert output.exit_code == 0
        assert "main command" in output.stdout

    def test_setup_command_failure_does_not_block(
        self, runner: LocalRunner, sample_task: Task
    ) -> None:
        """A failing setup_command logs a warning but doesn't prevent execution."""
        sample_task.setup_command = "exit 1"
        runner.prepare(sample_task)

        # Main command should still work
        output = runner.execute("echo 'still running'", timeout=10.0)
        runner.cleanup()

        assert output.exit_code == 0
        assert "still running" in output.stdout

    def test_double_cleanup_is_safe(self, runner: LocalRunner, sample_task: Task) -> None:
        """Calling cleanup() twice is safe."""
        runner.prepare(sample_task)
        runner.cleanup()
        runner.cleanup()  # Second call should not raise
        assert not runner.is_prepared


# ===========================================================================
# 6. Interface compliance
# ===========================================================================


class TestInterfaceCompliance:
    """LocalRunner is a proper BaseEnvironment subclass."""

    def test_is_base_environment_subclass(self) -> None:
        """LocalRunner inherits from BaseEnvironment."""
        from grist_mill.interfaces import BaseEnvironment

        assert issubclass(LocalRunner, BaseEnvironment)

    def test_has_prepare_execute_cleanup(self) -> None:
        """LocalRunner has prepare, execute, and cleanup methods."""
        runner = LocalRunner()
        assert callable(runner.prepare)
        assert callable(runner.execute)
        assert callable(runner.cleanup)

    def test_execute_returns_execution_output(self, runner: LocalRunner, sample_task: Task) -> None:
        """execute() returns an ExecutionOutput instance."""
        runner.prepare(sample_task)
        output = runner.execute("echo test", timeout=10.0)
        runner.cleanup()

        assert isinstance(output, ExecutionOutput)
        assert hasattr(output, "stdout")
        assert hasattr(output, "stderr")
        assert hasattr(output, "exit_code")
        assert hasattr(output, "timed_out")

    def test_can_be_used_with_base_harness(self, sample_task: Task) -> None:
        """LocalRunner can be used as a BaseEnvironment in the harness."""
        from grist_mill.interfaces import BaseAgent, HarnessConfig
        from grist_mill.schemas import AgentConfig, EnvironmentConfig

        class EchoAgent(BaseAgent):
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                return TaskResult(task_id=task.id, status=TaskStatus.SUCCESS, score=1.0)

        from grist_mill.interfaces import LocalHarness

        config = HarnessConfig(
            agent=AgentConfig(model="test", provider="test"),
            environment=EnvironmentConfig(runner_type="local"),
        )
        harness = LocalHarness(config=config)
        env = LocalRunner()
        agent = EchoAgent()

        result = harness.run(sample_task, agent, env)
        assert result.status == TaskStatus.SUCCESS


# ===========================================================================
# 7. Constructor and configuration
# ===========================================================================


class TestConstructor:
    """LocalRunner constructor and configuration."""

    def test_default_working_directory_is_none(self) -> None:
        """Default working_directory is None (uses cwd)."""
        runner = LocalRunner()
        assert runner.working_directory is None

    def test_custom_working_directory(self) -> None:
        """Custom working_directory is stored."""
        runner = LocalRunner(working_directory="/tmp")
        assert runner.working_directory == "/tmp"

    def test_sigkill_delay_default(self) -> None:
        """Default SIGKILL delay is reasonable (e.g., 5 seconds)."""
        runner = LocalRunner()
        assert runner.sigkill_delay > 0

    def test_custom_sigkill_delay(self) -> None:
        """Custom sigkill_delay is stored."""
        runner = LocalRunner(sigkill_delay=2.0)
        assert runner.sigkill_delay == 2.0

    def test_repr(self) -> None:
        """repr provides useful information."""
        runner = LocalRunner(working_directory="/tmp", sigkill_delay=3.0)
        r = repr(runner)
        assert "LocalRunner" in r
        assert "/tmp" in r


# ===========================================================================
# 8. Integration with harness (VAL-HARNESS-06)
# ===========================================================================


class TestHarnessIntegration:
    """LocalRunner provides same interface as Docker runner.

    Validates VAL-HARNESS-06: 'The LocalRunner must execute the agent command
    as a subprocess on the host, capturing stdout/stderr and exit code,
    returning a TaskResult with the same interface as the Docker runner.'
    """

    def test_echo_success_produces_taskresult_success(
        self, runner: LocalRunner, sample_task: Task
    ) -> None:
        """Running echo 'success' produces TaskResult with status=SUCCESS."""
        runner.prepare(sample_task)
        result = runner.run_task(
            command="echo 'success'",
            task_id=sample_task.id,
            language=sample_task.language,
            timeout=10.0,
        )
        runner.cleanup()

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.task_id == sample_task.id

    def test_failing_command_produces_taskresult_failure(
        self, runner: LocalRunner, sample_task: Task
    ) -> None:
        """A failing command produces TaskResult with appropriate status."""
        runner.prepare(sample_task)
        result = runner.run_task(
            command="echo 'SyntaxError: invalid syntax' >&2 && exit 1",
            task_id=sample_task.id,
            language=sample_task.language,
            timeout=10.0,
        )
        runner.cleanup()

        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.ERROR
        assert result.score == 0.0
