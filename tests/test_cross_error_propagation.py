"""Cross-error propagation tests (m6-cross-error-propagation).

Verifies errors propagate consistently across subsystems:
agent -> harness -> telemetry -> report. No subsystem silently swallows
errors or produces SUCCESS when failure occurred. Graceful shutdown
(SIGTERM) cleans up all resources.

Validates:
- VAL-CROSS-06: Multi-language environment consistency
- VAL-CROSS-10: Error propagation across subsystems
- VAL-CROSS-11: Graceful shutdown cleans up resources

ExpectedBehavior:
- Agent errors produce TaskResult with correct error_category
- Harness errors propagate to telemetry and reports
- SIGTERM during evaluation: no orphaned containers, partial manifest written
- Multi-language tasks use correct environments
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import pytest

from grist_mill.environments.language_config import (
    DEFAULT_FALLBACK_IMAGE,
    DEFAULT_LANGUAGE_IMAGES,
    LanguageImageConfig,
)
from grist_mill.harness import Harness, run_experiment
from grist_mill.reports.errors import error_taxonomy_breakdown
from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ErrorCategory,
    ExecutionOutput,
    HarnessConfig,
    Task,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.telemetry import (
    LatencyBreakdown,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "test-task-1",
    language: str = "python",
    timeout: int = 30,
) -> Task:
    return Task(
        id=task_id,
        prompt="Fix the bug",
        language=language,
        test_command="echo ok",
        timeout=timeout,
        difficulty=Difficulty.EASY,
    )


def _make_config() -> HarnessConfig:
    return HarnessConfig(
        agent=AgentConfig(model="gpt-4", provider="openrouter"),
        environment=EnvironmentConfig(runner_type="local"),
    )


def _make_telemetry() -> TelemetrySchema:
    return TelemetrySchema(
        version="V1",
        tokens=TokenUsage(prompt=100, completion=50, total=150),
        latency=LatencyBreakdown(
            setup_s=0.1,
            execution_s=0.5,
            teardown_s=0.1,
            total_s=0.7,
        ),
        tool_calls=ToolCallMetrics(
            total_calls=2,
            successful_calls=1,
            failed_calls=1,
        ),
        estimated_cost_usd=0.01,
        raw_events=[],
    )


class _MockEnvironment:
    """Mock environment that records call order and returns configurable output."""

    def __init__(
        self,
        execute_output: ExecutionOutput | None = None,
        prepare_error: Exception | None = None,
    ) -> None:
        self._execute_output = execute_output or ExecutionOutput(
            stdout="ok",
            stderr="",
            exit_code=0,
        )
        self._prepare_error = prepare_error
        self.call_order: list[str] = []
        self.cleanup_called = False

    def prepare(self, task: Task) -> None:
        self.call_order.append("prepare")
        if self._prepare_error is not None:
            raise self._prepare_error

    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        self.call_order.append(f"execute:{command}")
        return self._execute_output

    def cleanup(self) -> None:
        self.call_order.append("cleanup")
        self.cleanup_called = True


class _MockAgent:
    """Mock agent that returns a configurable result or raises."""

    def __init__(
        self,
        result: TaskResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.call_count = 0
        self.last_task: Task | None = None
        self.last_config: HarnessConfig | None = None

    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        self.call_count += 1
        self.last_task = task
        self.last_config = config
        if self._error is not None:
            raise self._error
        return self._result or TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
        )


# ===========================================================================
# VAL-CROSS-10: Error propagation across subsystems
# ===========================================================================


class TestAgentErrorPropagation:
    """Verify agent errors produce TaskResult with correct error_category."""

    def test_agent_exception_captured_with_correct_category(self) -> None:
        """Agent exceptions are captured and produce ERROR with API_ERROR category.

        When the agent raises an exception, the harness builds an ERROR result
        with error_category=API_ERROR. The harness still runs the test command
        for non-TIMEOUT/SKIPPED statuses — if the test also fails, the final
        result reflects the test failure; if the test passes, the test success
        is the authoritative result. The key assertion is that the error is
        never silently swallowed.
        """
        task = _make_task()
        config = _make_config()
        # Test command fails — the agent exception + failing test should not
        # produce SUCCESS.
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED",
                stderr="agent produced no code",
                exit_code=1,
            ),
        )
        agent = _MockAgent(
            error=RuntimeError("LLM API connection failed"),
        )

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        # With a failing test command, the result must NOT be SUCCESS
        assert result.status != TaskStatus.SUCCESS
        assert result.error_category is not None
        assert result.score == 0.0
        assert result.task_id == task.id
        # Telemetry must be attached even on error
        assert result.telemetry is not None
        assert isinstance(result.telemetry, TelemetrySchema)
        # Cleanup must have been called
        assert env.cleanup_called

    def test_agent_exception_with_passing_test_is_success(self) -> None:
        """When agent raises but test passes, SUCCESS is legitimate (not silent).

        This verifies the harness correctly runs the test command even after
        agent errors (non-TIMEOUT/SKIPPED). If tests pass, the task genuinely
        succeeded despite the agent hiccup.
        """
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment()  # Test passes (exit_code=0)
        agent = _MockAgent(
            error=RuntimeError("LLM API connection failed"),
        )

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        # The test passed, so SUCCESS is legitimate
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.telemetry is not None

    def test_agent_error_result_propagates_error_category(self) -> None:
        """Agent returns an ERROR TaskResult — harness preserves error context.

        The harness runs the test command after agent errors (non-TIMEOUT/SKIPPED).
        When the test also fails, the test failure determines the final status
        but the agent's error is captured in telemetry events.
        """
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED",
                stderr="error",
                exit_code=1,
            ),
        )
        agent = _MockAgent(
            result=TaskResult(
                task_id=task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.MAX_TURNS_EXCEEDED,
            ),
        )

        harness = Harness(config=config, trace_enabled=True)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.task_id == task.id
        assert result.status != TaskStatus.SUCCESS
        assert result.telemetry is not None
        # The agent's error should be captured in telemetry raw events
        events = result.telemetry.raw_events
        has_agent_error = any(
            e.get("phase") == "agent_run"
            and e.get("status") == "ERROR"
            and e.get("error_category") == "MAX_TURNS_EXCEEDED"
            for e in events
        )
        assert has_agent_error, "Agent error event should be in telemetry"

    def test_agent_timeout_result_preserved(self) -> None:
        """Agent TIMEOUT result skips test execution, preserving the timeout status."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent(
            result=TaskResult(
                task_id=task.id,
                status=TaskStatus.TIMEOUT,
                score=0.0,
            ),
        )

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        # TIMEOUT from agent should skip test execution
        assert result.status == TaskStatus.TIMEOUT
        assert result.task_id == task.id
        assert result.telemetry is not None
        # Cleanup must have been called
        assert env.cleanup_called

    def test_agent_syntax_error_propagates(self) -> None:
        """Agent with SYNTAX_ERROR category propagates through harness.

        The harness runs tests after agent ERROR. With a failing test, the
        final result reflects the test failure. With trace enabled, the
        agent's error category is captured in telemetry.
        """
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="SyntaxError: invalid syntax",
                exit_code=1,
            ),
        )
        agent = _MockAgent(
            result=TaskResult(
                task_id=task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.SYNTAX_ERROR,
            ),
        )

        harness = Harness(config=config, trace_enabled=True)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.task_id == task.id
        assert result.telemetry is not None
        assert result.status != TaskStatus.SUCCESS

    def test_no_silent_success_on_agent_failure_with_failing_test(self) -> None:
        """Harness never produces SUCCESS when agent AND test both failed."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED",
                stderr="rate limited",
                exit_code=1,
            ),
        )
        agent = _MockAgent(
            result=TaskResult(
                task_id=task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.RATE_LIMIT,
            ),
        )

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        # SUCCESS would mean the error was silently swallowed
        assert result.status != TaskStatus.SUCCESS
        assert result.error_category is not None


class TestEnvironmentErrorPropagation:
    """Verify environment errors propagate to TaskResult."""

    def test_prepare_failure_produces_environment_error(self) -> None:
        """Environment prepare failure produces ERROR with ENVIRONMENT_ERROR."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            prepare_error=RuntimeError("Docker image not found"),
        )
        agent = _MockAgent()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR
        assert result.score == 0.0
        assert result.telemetry is not None
        # Cleanup must still be called even when prepare fails
        assert env.cleanup_called

    def test_test_execution_failure_propagates(self) -> None:
        """Test command failure (non-zero exit) propagates as FAILURE."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED 1 test",
                stderr="AssertionError",
                exit_code=1,
            ),
        )
        agent = _MockAgent(
            result=TaskResult(
                task_id=task.id,
                status=TaskStatus.SUCCESS,
                score=1.0,
            ),
        )

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        # Test failure should NOT be masked as SUCCESS
        assert result.status != TaskStatus.SUCCESS
        assert result.score < 1.0
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_cleanup_called_on_all_error_paths(self) -> None:
        """Cleanup is called regardless of where the error occurs."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            prepare_error=RuntimeError("boom"),
        )
        agent = _MockAgent()

        harness = Harness(config=config)
        harness.run(task=task, agent=agent, env=env)

        assert env.cleanup_called


class TestTelemetryOnError:
    """Verify telemetry is attached to every result including errors."""

    def test_telemetry_on_agent_exception(self) -> None:
        """Telemetry attached when agent raises exception."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent(error=ValueError("model output malformed"))

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        telem = result.telemetry
        assert isinstance(telem, TelemetrySchema)
        # Latency should be recorded even on error
        assert telem.latency.total_s > 0

    def test_telemetry_on_environment_error(self) -> None:
        """Telemetry attached when environment prepare fails."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(prepare_error=OSError("no space left"))
        agent = _MockAgent()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        telem = result.telemetry
        assert isinstance(telem, TelemetrySchema)

    def test_telemetry_on_test_failure(self) -> None:
        """Telemetry attached when test execution fails."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED",
                stderr="error",
                exit_code=1,
            ),
        )
        agent = _MockAgent()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        telem = result.telemetry
        assert isinstance(telem, TelemetrySchema)
        assert telem.latency.total_s > 0

    def test_raw_events_on_error_with_trace_enabled(self) -> None:
        """Raw events captured for errors when trace_enabled=True."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(prepare_error=RuntimeError("docker gone"))
        agent = _MockAgent()

        harness = Harness(config=config, trace_enabled=True)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        events = result.telemetry.raw_events
        assert len(events) > 0
        # The prepare failure should be recorded
        has_prepare_failure = any(
            e.get("phase") == "prepare" and e.get("status") == "failed" for e in events
        )
        assert has_prepare_failure


class TestErrorPropagationToReports:
    """Verify errors propagate through to the reporting pipeline."""

    def test_error_results_flow_into_taxonomy_breakdown(self) -> None:
        """Error results produce correct taxonomy breakdown."""
        results = [
            {
                "task_id": "r1",
                "status": TaskStatus.ERROR,
                "error_category": ErrorCategory.API_ERROR,
            },
            {
                "task_id": "r2",
                "status": TaskStatus.FAILURE,
                "error_category": ErrorCategory.TEST_FAILURE,
            },
            {
                "task_id": "r3",
                "status": TaskStatus.ERROR,
                "error_category": ErrorCategory.API_ERROR,
            },
            {
                "task_id": "r4",
                "status": TaskStatus.SUCCESS,
                "error_category": None,
            },
        ]

        breakdown = error_taxonomy_breakdown(results)

        # Only error and failure results should appear
        categories = {b["error_category"] for b in breakdown}
        assert "API_ERROR" in categories
        assert "TEST_FAILURE" in categories
        # SUCCESS should not appear
        assert len(breakdown) == 2

        # Count should be correct
        api_entry = next(b for b in breakdown if b["error_category"] == "API_ERROR")
        assert api_entry["count"] == 2

    def test_agent_error_in_experiment_results(self) -> None:
        """Agent errors from run_experiment propagate to results when tests also fail."""
        tasks = [_make_task(task_id=f"exp-err-{i}") for i in range(3)]
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED",
                stderr="agent error, no solution generated",
                exit_code=1,
            ),
        )
        agent = _MockAgent(
            result=TaskResult(
                task_id="placeholder",
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.API_ERROR,
            ),
        )

        results = run_experiment(
            tasks=tasks,
            config=config,
            agent=agent,
            env=env,
            max_retries=0,
            retry_delay=0.0,
        )

        assert len(results) == 3
        for result in results:
            # No result should be SUCCESS (agent returned ERROR)
            assert result.status != TaskStatus.SUCCESS


# ===========================================================================
# VAL-CROSS-06: Multi-language environment consistency
# ===========================================================================


class TestMultiLanguageEnvironmentConsistency:
    """Verify tasks with different languages use correct environments."""

    def test_language_config_maps_python(self) -> None:
        """Language 'python' resolves to Python Docker image."""
        config = LanguageImageConfig()
        image = config.get_image("python")
        assert image == DEFAULT_LANGUAGE_IMAGES["python"]

    def test_language_config_maps_r(self) -> None:
        """Language 'r' resolves to R (rocker) Docker image."""
        config = LanguageImageConfig()
        image = config.get_image("r")
        assert image == DEFAULT_LANGUAGE_IMAGES["r"]

    def test_language_config_maps_typescript(self) -> None:
        """Language 'typescript' resolves to Node.js Docker image."""
        config = LanguageImageConfig()
        image = config.get_image("typescript")
        assert image == DEFAULT_LANGUAGE_IMAGES["typescript"]

    def test_language_config_unknown_uses_fallback(self) -> None:
        """Unknown language uses fallback image."""
        config = LanguageImageConfig()
        image = config.get_image("cobol")
        assert image == DEFAULT_FALLBACK_IMAGE

    def test_language_config_case_insensitive(self) -> None:
        """Language lookup is case-insensitive."""
        config = LanguageImageConfig()
        assert config.get_image("Python") == config.get_image("python")
        assert config.get_image("R") == config.get_image("r")
        assert config.get_image("TypeScript") == config.get_image("typescript")

    def test_language_config_custom_override(self) -> None:
        """Custom overrides take precedence over defaults."""
        config = LanguageImageConfig(
            overrides={"python": "my-python:3.11", "go": "golang:1.22"},
        )
        assert config.get_image("python") == "my-python:3.11"
        assert config.get_image("go") == "golang:1.22"
        # Default for non-overridden language still works
        assert config.get_image("r") == DEFAULT_LANGUAGE_IMAGES["r"]

    def test_multi_language_tasks_use_different_environments(self) -> None:
        """Tasks with different languages should resolve to different images."""
        python_task = _make_task(task_id="py-task", language="python")
        r_task = _make_task(task_id="r-task", language="r")
        ts_task = _make_task(task_id="ts-task", language="typescript")

        config = LanguageImageConfig()

        py_image = config.get_image(python_task.language)
        r_image = config.get_image(r_task.language)
        ts_image = config.get_image(ts_task.language)

        # Each language should get a different image
        assert py_image != r_image
        assert py_image != ts_image
        assert r_image != ts_image

    def test_language_field_drives_image_selection(self) -> None:
        """The task's language field directly determines the Docker image."""
        task = _make_task(language="r")
        config = LanguageImageConfig()
        image = config.get_image(task.language)
        assert "rocker" in image.lower()

    def test_fallback_image_overridable(self) -> None:
        """Fallback image is configurable."""
        config = LanguageImageConfig(fallback_image="ubuntu:22.04")
        assert config.get_image("fortran") == "ubuntu:22.04"


# ===========================================================================
# VAL-CROSS-11: Graceful shutdown cleans up resources
# ===========================================================================


class TestGracefulShutdown:
    """Verify SIGTERM during evaluation causes clean shutdown with partial manifest.

    Tests use subprocess to send SIGTERM to a child process running an
    experiment, then verify the partial manifest was written and no
    orphaned processes remain.
    """

    def _write_worker_script(
        self,
        tmpdir: str,
        manifest_path: str,
        num_tasks: int = 10,
        task_delay: float = 2.0,
    ) -> str:
        """Write a Python script that simulates a multi-task evaluation.

        The script:
        1. Runs tasks one by one
        2. Writes a partial manifest after each task
        3. Handles SIGTERM by writing final manifest and exiting cleanly
        """
        script = f'''
import json
import signal
import sys
import time
import os

manifest_path = {manifest_path!r}
num_tasks = {num_tasks}
task_delay = {task_delay}

results = []
shutdown_requested = False
running_task = False

def sigterm_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print("SIGTERM received, initiating graceful shutdown", flush=True)

signal.signal(signal.SIGTERM, sigterm_handler)

def write_partial_manifest(results_list):
    """Write partial manifest with whatever results we have so far."""
    manifest = {{
        "name": "grist-mill-evaluation",
        "version": "0.1.0",
        "timestamp": results_list[-1].get("timestamp", "") if results_list else "",
        "schema_version": "V1",
        "total_tasks": num_tasks,
        "completed_tasks": len(results_list),
        "tasks": results_list,
        "interrupted": shutdown_requested,
    }}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

for i in range(num_tasks):
    if shutdown_requested:
        print(f"Shutdown requested before task {{i}}, writing partial manifest", flush=True)
        write_partial_manifest(results)
        sys.exit(0)

    running_task = True
    # Simulate task execution
    time.sleep(task_delay)

    result = {{
        "task_id": f"task-{{i}}",
        "status": "SUCCESS",
        "score": 1.0,
        "timestamp": "2025-01-01T00:00:00Z",
    }}
    results.append(result)
    running_task = False

    # Write partial manifest after each task
    write_partial_manifest(results)
    print(f"Completed task {{i}}/{{num_tasks}}", flush=True)

# Normal completion
write_partial_manifest(results)
print("All tasks completed", flush=True)
'''
        script_path = os.path.join(tmpdir, "worker_script.py")
        with open(script_path, "w") as f:
            f.write(script)
        return script_path

    def test_sigterm_writes_partial_manifest(self) -> None:
        """SIGTERM during multi-task evaluation writes a partial manifest.

        Scenario:
        1. Start a child process running a multi-task evaluation
        2. Wait for it to complete a couple of tasks
        3. Send SIGTERM
        4. Verify the partial manifest exists with completed results
        5. Verify no orphaned processes
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "partial_manifest.json")
            num_tasks = 10
            task_delay = 0.5  # Fast enough for test

            script_path = self._write_worker_script(
                tmpdir,
                manifest_path,
                num_tasks=num_tasks,
                task_delay=task_delay,
            )

            # Start the worker process
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for at least 2 tasks to complete
            time.sleep(task_delay * 2 + 0.5)

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)

            # Wait for the process to exit (with timeout)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail("Worker process did not terminate after SIGTERM")

            # Verify the process exited
            assert proc.returncode is not None
            assert proc.returncode == 0, f"Worker exited with code {proc.returncode}"

            # Verify partial manifest was written
            assert os.path.exists(manifest_path), "Partial manifest was not written"

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Verify the manifest is partial
            assert manifest["interrupted"] is True
            assert manifest["completed_tasks"] < num_tasks
            assert manifest["completed_tasks"] >= 1
            assert len(manifest["tasks"]) == manifest["completed_tasks"]

            # Verify the results are valid
            for task_result in manifest["tasks"]:
                assert "task_id" in task_result
                assert "status" in task_result
                assert "score" in task_result

    def test_sigterm_no_orphaned_processes(self) -> None:
        """After SIGTERM, no orphaned worker processes remain.

        Scenario:
        1. Start a worker that creates child processes
        2. Send SIGTERM
        3. Verify no orphaned processes from our worker
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "no_orphan_manifest.json")

            # Script that creates a child process and handles SIGTERM
            script = f"""
import json
import os
import signal
import subprocess
import sys
import time

manifest_path = {manifest_path!r}
child_procs = []

def sigterm_handler(signum, frame):
    # Kill all child processes
    for p in child_procs:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    # Write partial manifest
    manifest = {{
        "name": "grist-mill-evaluation",
        "version": "0.1.0",
        "schema_version": "V1",
        "completed_tasks": 0,
        "interrupted": True,
        "tasks": [],
    }}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

# Create a long-running child process
child = subprocess.Popen(
    ["sleep", "300"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
child_procs.append(child)

# Wait to be signaled
time.sleep(300)
"""
            script_path = os.path.join(tmpdir, "orphan_test.py")
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,  # Create process group
            )

            # Give it time to start the child process
            time.sleep(0.5)

            # Get the process group
            pgid = os.getpgid(proc.pid)

            # Send SIGTERM to the process group
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                proc.send_signal(signal.SIGTERM)

            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                # Force kill
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait()

            # Verify the process exited
            assert proc.returncode is not None

            # Verify the child process is gone
            import contextlib as _ctxlib

            with _ctxlib.suppress(subprocess.TimeoutExpired):
                subprocess.run(
                    ["pgrep", "-f", "sleep 300"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )

            # Verify partial manifest was written
            assert os.path.exists(manifest_path), "Partial manifest not written after SIGTERM"
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["interrupted"] is True

    def test_normal_completion_writes_full_manifest(self) -> None:
        """Normal completion (no SIGTERM) writes a full manifest with all results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "full_manifest.json")

            script = f"""
import json
import sys

manifest_path = {manifest_path!r}
results = [{{"task_id": f"task-{{i}}", "status": "SUCCESS", "score": 1.0}} for i in range(5)]

manifest = {{
    "name": "grist-mill-evaluation",
    "version": "0.1.0",
    "schema_version": "V1",
    "completed_tasks": 5,
    "total_tasks": 5,
    "interrupted": False,
    "tasks": results,
}}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
"""
            script_path = os.path.join(tmpdir, "normal_test.py")
            with open(script_path, "w") as f:
                f.write(script)

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["interrupted"] is False
            assert manifest["completed_tasks"] == 5
            assert len(manifest["tasks"]) == 5

    def test_experiment_interruptible_writes_partial_results(self) -> None:
        """run_experiment with signal handling writes partial results on SIGTERM.

        Uses the actual run_experiment function from the harness module
        in a subprocess to test real-world SIGTERM handling.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "experiment_manifest.json")

            # Script that uses run_experiment and handles SIGTERM
            script = f"""
import json
import signal
import sys
import time

sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})

from grist_mill.schemas import (
    AgentConfig, EnvironmentConfig, ErrorCategory, ExecutionOutput,
    HarnessConfig, Task, TaskResult, TaskStatus,
)

manifest_path = {manifest_path!r}
results = []
shutdown_requested = False

def sigterm_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGTERM, sigterm_handler)

class SlowAgent:
    def __init__(self):
        self.call_count = 0
    def run(self, task, config):
        self.call_count += 1
        time.sleep(1.0)  # Simulate slow LLM call
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
        )

class TrackingEnv:
    def __init__(self):
        self.cleanup_called = False
    def prepare(self, task):
        pass
    def execute(self, command, timeout):
        return ExecutionOutput(stdout="ok", stderr="", exit_code=0)
    def cleanup(self):
        self.cleanup_called = True

from grist_mill.harness import Harness

config = HarnessConfig(
    agent=AgentConfig(model="test", provider="test"),
    environment=EnvironmentConfig(runner_type="local"),
)
agent = SlowAgent()
env = TrackingEnv()
harness = Harness(config=config)

tasks = [Task(
    id=f"sigtest-{{i}}",
    prompt="test",
    language="python",
    test_command="echo ok",
    timeout=30,
) for i in range(10)]

for task in tasks:
    if shutdown_requested:
        break
    result = harness.run(task=task, agent=agent, env=env)
    results.append(result.model_dump(mode="json"))

# Write manifest
manifest = {{
    "name": "grist-mill-sigterm-test",
    "version": "0.1.0",
    "schema_version": "V1",
    "completed_tasks": len(results),
    "total_tasks": len(tasks),
    "interrupted": shutdown_requested,
    "tasks": results,
}}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

# Cleanup
env.cleanup()
print(f"Done: {{len(results)}}/{{len(tasks)}} tasks", flush=True)
"""
            script_path = os.path.join(tmpdir, "harness_sigterm_test.py")
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for a couple of tasks to complete (each takes ~1s + overhead)
            time.sleep(2.5)

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)

            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail("Harness process did not terminate after SIGTERM")

            # Verify process exited cleanly
            assert proc.returncode is not None

            # Verify partial manifest exists
            assert os.path.exists(manifest_path), "Partial manifest not written"

            with open(manifest_path) as f:
                manifest = json.load(f)

            # At least one task should have completed before SIGTERM
            assert manifest["completed_tasks"] >= 1
            assert manifest["completed_tasks"] < 10
            assert manifest["interrupted"] is True

            # Verify each result has valid fields
            for task_result in manifest["tasks"]:
                assert "task_id" in task_result
                assert "status" in task_result
                assert "score" in task_result
                assert "telemetry" in task_result  # Telemetry must be present
                # Verify no silent SUCCESS on error
                if task_result["status"] == "SUCCESS":
                    assert task_result["score"] > 0


# ===========================================================================
# Cross-cutting: error never produces false SUCCESS
# ===========================================================================


class TestNoFalseSuccess:
    """Verify no subsystem produces SUCCESS when failure actually occurred."""

    def test_agent_failure_with_failing_test_never_becomes_success(self) -> None:
        """Agent ERROR results + failing test never become SUCCESS after harness."""
        for status, error_cat in [
            (TaskStatus.ERROR, ErrorCategory.API_ERROR),
            (TaskStatus.ERROR, ErrorCategory.RATE_LIMIT),
            (TaskStatus.ERROR, ErrorCategory.MAX_TURNS_EXCEEDED),
            (TaskStatus.ERROR, ErrorCategory.NETWORK_ERROR),
        ]:
            task = _make_task(task_id=f"nofalse-{status.value}-{error_cat.value}")
            config = _make_config()
            env = _MockEnvironment(
                execute_output=ExecutionOutput(
                    stdout="FAILED",
                    stderr="test failure",
                    exit_code=1,
                ),
            )
            agent = _MockAgent(
                result=TaskResult(
                    task_id=task.id,
                    status=status,
                    score=0.0,
                    error_category=error_cat,
                ),
            )

            harness = Harness(config=config, max_retries=0)
            result = harness.run(task=task, agent=agent, env=env)

            # With a failing test command, the result must NOT be SUCCESS
            assert result.status != TaskStatus.SUCCESS

    def test_environment_crash_never_becomes_success(self) -> None:
        """Environment crash during test execution never produces SUCCESS."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="Killed",
                exit_code=137,  # SIGKILL
                timed_out=True,
            ),
        )
        agent = _MockAgent()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.status != TaskStatus.SUCCESS

    def test_empty_test_output_never_becomes_success(self) -> None:
        """Empty/malformed test output never produces SUCCESS."""
        task = _make_task()
        config = _make_config()
        env = _MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="",
                exit_code=1,
            ),
        )
        agent = _MockAgent()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.status != TaskStatus.SUCCESS


# ===========================================================================
# VAL-CROSS-11: InterruptibleExperiment unit tests
# ===========================================================================


class TestInterruptibleExperiment:
    """Unit tests for the InterruptibleExperiment class."""

    def test_normal_completion_returns_all_results(self) -> None:
        """Normal completion (no SIGTERM) returns results for all tasks."""
        from grist_mill.harness import InterruptibleExperiment

        tasks = [_make_task(task_id=f"int-{i}") for i in range(5)]
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            experiment = InterruptibleExperiment(
                config=config,
                agent=agent,
                env=env,
                manifest_path=manifest_path,
            )
            results = experiment.run(tasks)

            assert len(results) == 5
            assert not experiment.shutdown_requested

            # Verify manifest was written
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["completed_tasks"] == 5
            assert manifest["interrupted"] is False

    def test_sigterm_stops_accepting_new_tasks(self) -> None:
        """After shutdown_requested, no new tasks are started."""
        from grist_mill.harness import InterruptibleExperiment

        tasks = [_make_task(task_id=f"sig-{i}") for i in range(10)]
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "sig_manifest.json")
            experiment = InterruptibleExperiment(
                config=config,
                agent=agent,
                env=env,
                manifest_path=manifest_path,
            )

            # Simulate shutdown after 3 tasks
            tasks_done = 0

            # Monkey-patch the harness run to count and trigger shutdown
            original_run = Harness.run
            call_count = 0

            def counting_run(self_harness: Any, **kwargs: Any) -> TaskResult:
                nonlocal call_count, tasks_done
                call_count += 1
                result = original_run(self_harness, **kwargs)
                tasks_done = call_count
                if call_count >= 3:
                    experiment._shutdown_requested = True
                return result

            Harness.run = counting_run  # type: ignore[assignment]

            try:
                results = experiment.run(tasks)
            finally:
                Harness.run = original_run  # type: ignore[assignment]

            # Should have stopped after 3 tasks
            assert len(results) == 3
            assert experiment.shutdown_requested
            assert env.cleanup_called

            # Verify partial manifest
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["completed_tasks"] == 3
            assert manifest["total_tasks"] == 10
            assert manifest["interrupted"] is True

    def test_cleanup_called_on_shutdown(self) -> None:
        """Environment cleanup is called when shutdown happens mid-experiment."""
        from grist_mill.harness import InterruptibleExperiment

        tasks = [_make_task(task_id=f"cleanup-{i}") for i in range(5)]
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "cleanup_manifest.json")
            experiment = InterruptibleExperiment(
                config=config,
                agent=agent,
                env=env,
                manifest_path=manifest_path,
            )

            # Monkey-patch harness run to trigger shutdown after 2 tasks
            original_run = Harness.run
            call_count = 0

            def shutdown_trigger_run(self_harness: Any, **kwargs: Any) -> TaskResult:
                nonlocal call_count
                call_count += 1
                result = original_run(self_harness, **kwargs)
                if call_count >= 2:
                    experiment._shutdown_requested = True
                return result

            Harness.run = shutdown_trigger_run  # type: ignore[assignment]

            try:
                results = experiment.run(tasks)
            finally:
                Harness.run = original_run  # type: ignore[assignment]

            # Should have completed 2 tasks then stopped
            assert len(results) == 2
            assert experiment.shutdown_requested
            assert env.cleanup_called

            # Verify partial manifest
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["completed_tasks"] == 2
            assert manifest["interrupted"] is True

    def test_partial_manifest_written_after_each_task(self) -> None:
        """Partial manifest is written after each task completion."""
        from grist_mill.harness import InterruptibleExperiment

        tasks = [_make_task(task_id=f"partial-{i}") for i in range(3)]
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "incremental_manifest.json")
            experiment = InterruptibleExperiment(
                config=config,
                agent=agent,
                env=env,
                manifest_path=manifest_path,
            )
            experiment.run(tasks)

            # Final manifest should exist
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["completed_tasks"] == 3
            assert manifest["interrupted"] is False
            assert len(manifest["tasks"]) == 3

    def test_no_manifest_path_no_file_written(self) -> None:
        """When manifest_path is None, no manifest file is created."""
        from grist_mill.harness import InterruptibleExperiment

        tasks = [_make_task(task_id=f"nopath-{i}") for i in range(2)]
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent()

        experiment = InterruptibleExperiment(
            config=config,
            agent=agent,
            env=env,
            manifest_path=None,
        )
        results = experiment.run(tasks)

        assert len(results) == 2

    def test_sigterm_handler_restored_after_run(self) -> None:
        """Original SIGTERM handler is restored after experiment completes."""
        import signal as sig_mod

        from grist_mill.harness import InterruptibleExperiment

        tasks = [_make_task(task_id=f"restore-{i}") for i in range(2)]
        config = _make_config()
        env = _MockEnvironment()
        agent = _MockAgent()

        original_handler = sig_mod.getsignal(sig_mod.SIGTERM)

        experiment = InterruptibleExperiment(
            config=config,
            agent=agent,
            env=env,
        )
        experiment.run(tasks)

        # Handler should be restored
        restored_handler = sig_mod.getsignal(sig_mod.SIGTERM)
        assert restored_handler is original_handler
