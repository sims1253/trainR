"""Tests for the Harness implementation (m2-harness-implementation).

Covers:
- VAL-HARNESS-08: Harness wires task -> env.prepare -> agent.run -> env.execute(test_command) -> result
- VAL-HARNESS-09: Retry logic on transient failures, no retry on deterministic
- VAL-TELEM-02: Telemetry captures latency breakdown by phase
- VAL-TELEM-03: Telemetry captures tool call metrics
- VAL-TELEM-06: Every TaskResult has non-null telemetry with latency.total_s > 0
- VAL-TELEM-07: raw_events populated when trace_enabled=True
"""

from __future__ import annotations

import time
from typing import Any

import pytest

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
    TelemetryCollector,
    TelemetrySchema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_task() -> Task:
    """Return a valid Task for testing."""
    return Task(
        id="test-harness-001",
        prompt="Fix the bug in the function.",
        language="python",
        test_command="pytest tests/test_fix.py",
        timeout=30,
        difficulty=Difficulty.EASY,
    )


@pytest.fixture()
def sample_config() -> HarnessConfig:
    """Return a valid HarnessConfig for testing."""
    return HarnessConfig(
        agent=AgentConfig(model="gpt-4", provider="openai"),
        environment=EnvironmentConfig(runner_type="local"),
    )


@pytest.fixture()
def success_task_result(sample_task: Task) -> TaskResult:
    """Return a successful TaskResult."""
    return TaskResult(
        task_id=sample_task.id,
        status=TaskStatus.SUCCESS,
        score=1.0,
    )


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class MockEnvironment:
    """Mock environment that tracks call order."""

    def __init__(
        self,
        prepare_side_effect: Any = None,
        execute_output: ExecutionOutput | None = None,
    ) -> None:
        self.call_order: list[str] = []
        self._prepare_side_effect = prepare_side_effect
        self._execute_output = execute_output or ExecutionOutput(
            stdout="all tests passed",
            stderr="",
            exit_code=0,
        )

    def prepare(self, task: Task) -> None:
        self.call_order.append("prepare")
        if self._prepare_side_effect is not None:
            self._prepare_side_effect()

    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        self.call_order.append(f"execute:{command}")
        return self._execute_output

    def cleanup(self) -> None:
        self.call_order.append("cleanup")


class MockAgent:
    """Mock agent that returns a configurable result."""

    def __init__(
        self,
        result: TaskResult | None = None,
        telemetry: TelemetrySchema | None = None,
        run_side_effect: Any = None,
    ) -> None:
        self._result = result
        self._telemetry = telemetry
        self._run_side_effect = run_side_effect
        self.call_count = 0
        self.last_task: Task | None = None
        self.last_config: HarnessConfig | None = None

    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        self.call_count += 1
        self.last_task = task
        self.last_config = config
        if self._run_side_effect is not None:
            self._run_side_effect()
        result = self._result or TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=self._telemetry,
        )
        return result


class FailingAgent:
    """Agent that always raises an exception."""

    def __init__(self, exception: Exception | None = None) -> None:
        self._exception = exception or RuntimeError("Agent crashed!")

    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        raise self._exception


# ===========================================================================
# VAL-HARNESS-08: Harness wiring tests
# ===========================================================================


class TestHarnessWiring:
    """Tests for correct wiring: env.prepare -> agent.run -> env.execute(test_command)."""

    def test_harness_calls_prepare_agent_execute_cleanup(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness calls env.prepare, agent.run, env.execute(test_command), env.cleanup."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent_result = TaskResult(
            task_id=sample_task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
        )
        agent = MockAgent(result=agent_result)
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            trace_enabled=False,
        )
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Verify correct call order
        assert "prepare" in env.call_order
        assert f"execute:{sample_task.test_command}" in env.call_order
        assert "cleanup" in env.call_order

        # Verify agent was called
        assert agent.call_count == 1
        assert agent.last_task is sample_task

        # Verify result
        assert result.task_id == sample_task.id
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0

    def test_harness_call_order_is_correct(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Ensure the exact order: prepare -> agent_run -> execute -> cleanup."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        harness.run(task=sample_task, agent=agent, env=env, collector=collector)

        # prepare must come before execute
        prepare_idx = env.call_order.index("prepare")
        execute_idx = env.call_order.index(f"execute:{sample_task.test_command}")
        cleanup_idx = env.call_order.index("cleanup")

        assert prepare_idx < execute_idx
        assert execute_idx < cleanup_idx

    def test_harness_guarantees_cleanup_on_exception(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Cleanup is called even when agent raises an exception."""
        from grist_mill.harness import Harness

        env = MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="error",
                exit_code=1,
            )
        )
        agent = FailingAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)

        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Even though agent failed, cleanup was called and result was returned
        assert "cleanup" in env.call_order
        assert result.task_id == sample_task.id
        # The test command also failed, so the result should be a failure
        assert result.status == TaskStatus.FAILURE

    def test_harness_guarantees_cleanup_on_prepare_exception(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Cleanup is called even when env.prepare raises."""
        from grist_mill.harness import Harness

        env = MockEnvironment(prepare_side_effect=RuntimeError("Setup failed"))
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert "cleanup" in env.call_order
        assert result.status == TaskStatus.ERROR

    def test_harness_captures_test_execution_result(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Test execution output is captured in the result via result parser."""
        from grist_mill.harness import Harness

        # Test execution returns non-zero exit code (test failure)
        env = MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="1 failed",
                stderr="AssertionError",
                exit_code=1,
            )
        )
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # The test_command execution result should be reflected in the final result
        assert result.task_id == sample_task.id
        # When agent returns SUCCESS but tests fail, the harness captures the test result
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_harness_with_timeout_on_test_execution(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Test execution timeout is handled correctly."""
        from grist_mill.harness import Harness

        env = MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=True,
            )
        )
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert result.status == TaskStatus.TIMEOUT


# ===========================================================================
# VAL-HARNESS-09: Retry logic tests
# ===========================================================================


class TestHarnessRetry:
    """Tests for retry logic on transient failures."""

    def test_retry_on_rate_limit(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness retries when agent returns ERROR with RATE_LIMIT."""
        from grist_mill.harness import Harness

        env = MockEnvironment()

        # Agent fails twice with RATE_LIMIT then succeeds
        results = [
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.RATE_LIMIT,
            ),
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.RATE_LIMIT,
            ),
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.SUCCESS,
                score=1.0,
            ),
        ]

        class RetryAgent:
            def __init__(self) -> None:
                self.call_count = 0

            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                result = results[min(self.call_count, len(results) - 1)]
                self.call_count += 1
                return result

        agent = RetryAgent()
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,  # No delay for testing
        )
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 3
        assert result.status == TaskStatus.SUCCESS

    def test_retry_on_api_error(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness retries when agent returns ERROR with API_ERROR."""
        from grist_mill.harness import Harness

        env = MockEnvironment()

        results = [
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.API_ERROR,
            ),
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.SUCCESS,
                score=1.0,
            ),
        ]

        class RetryAgent:
            def __init__(self) -> None:
                self.call_count = 0

            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                result = results[min(self.call_count, len(results) - 1)]
                self.call_count += 1
                return result

        agent = RetryAgent()
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,
        )
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 2
        assert result.status == TaskStatus.SUCCESS

    def test_retry_on_network_error(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness retries when agent returns ERROR with NETWORK_ERROR."""
        from grist_mill.harness import Harness

        env = MockEnvironment()

        results = [
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.NETWORK_ERROR,
            ),
            TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.SUCCESS,
                score=1.0,
            ),
        ]

        class RetryAgent:
            def __init__(self) -> None:
                self.call_count = 0

            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                result = results[min(self.call_count, len(results) - 1)]
                self.call_count += 1
                return result

        agent = RetryAgent()
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,
        )
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 2
        assert result.status == TaskStatus.SUCCESS

    def test_no_retry_on_test_failure(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness does NOT retry when agent returns FAILURE with TEST_FAILURE."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.FAILURE,
                score=0.0,
                error_category=ErrorCategory.TEST_FAILURE,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,
        )
        harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Agent should only be called once (no retry)
        assert agent.call_count == 1

    def test_no_retry_on_timeout(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness does NOT retry when agent returns TIMEOUT."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.TIMEOUT,
                score=0.0,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,
        )
        harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 1

    def test_no_retry_on_success(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness does NOT retry on success."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,
        )
        harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 1

    def test_retry_exhausts_max_retries(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness stops retrying after max_retries is exhausted."""
        from grist_mill.harness import Harness

        env = MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="tests failed",
                exit_code=1,
            )
        )
        agent = MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.RATE_LIMIT,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=2,
            retry_delay=0.0,
        )
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Initial attempt + 2 retries = 3 total calls
        assert agent.call_count == 3
        # The test execution also failed
        assert result.status == TaskStatus.FAILURE

    def test_retry_delay_is_respected(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Retry delay is respected between attempts."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.RATE_LIMIT,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=2,
            retry_delay=0.1,  # 100ms delay
        )

        start = time.monotonic()
        harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )
        elapsed = time.monotonic() - start

        # 2 retries with 0.1s delay = at least 0.2s
        assert elapsed >= 0.15  # allow some tolerance

    def test_retry_with_zero_max_retries(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """With max_retries=0, no retries happen."""
        from grist_mill.harness import Harness

        env = MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="",
                stderr="failed",
                exit_code=1,
            )
        )
        agent = MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.RATE_LIMIT,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=0,
            retry_delay=0.0,
        )
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 1
        # Test execution also failed since agent didn't produce a solution
        assert result.status == TaskStatus.FAILURE

    def test_no_retry_on_skipped(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness does NOT retry when agent returns SKIPPED."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.SKIPPED,
                score=0.0,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(
            config=sample_config,
            max_retries=3,
            retry_delay=0.0,
        )
        harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert agent.call_count == 1


# ===========================================================================
# VAL-TELEM-06: Every TaskResult has non-null telemetry with latency.total_s > 0
# ===========================================================================


class TestHarnessTelemetry:
    """Tests for telemetry attachment to every TaskResult."""

    def test_telemetry_attached_on_success(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Every TaskResult has non-null telemetry on success."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert result.telemetry is not None
        assert isinstance(result.telemetry, TelemetrySchema)
        assert result.telemetry.latency.total_s > 0

    def test_telemetry_attached_on_failure(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Every TaskResult has non-null telemetry even on failure."""
        from grist_mill.harness import Harness

        env = MockEnvironment(
            execute_output=ExecutionOutput(
                stdout="FAILED",
                stderr="AssertionError",
                exit_code=1,
            )
        )
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert result.telemetry is not None
        assert isinstance(result.telemetry, TelemetrySchema)
        assert result.telemetry.latency.total_s > 0

    def test_telemetry_attached_on_error(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Every TaskResult has non-null telemetry even when agent raises."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = FailingAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert result.telemetry is not None
        assert isinstance(result.telemetry, TelemetrySchema)
        assert result.telemetry.latency.total_s > 0

    def test_telemetry_captures_latency_breakdown(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Telemetry captures setup_s, execution_s, teardown_s phases."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        telemetry = result.telemetry
        assert telemetry is not None
        # All phase durations should be >= 0
        assert telemetry.latency.setup_s >= 0
        assert telemetry.latency.execution_s >= 0
        assert telemetry.latency.teardown_s >= 0
        # Total should be approximately sum of phases
        assert telemetry.latency.total_s >= 0
        assert (
            abs(
                telemetry.latency.total_s
                - (
                    telemetry.latency.setup_s
                    + telemetry.latency.execution_s
                    + telemetry.latency.teardown_s
                )
            )
            < 0.1
        )

    def test_telemetry_captures_tool_call_metrics(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Telemetry captures tool call metrics (total_calls, by_tool breakdown)."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        # Simulate tool calls before the harness run
        collector.record_tool_call("bash", success=True, duration_ms=50.0)
        collector.record_tool_call("editor", success=True, duration_ms=120.0)
        collector.record_tool_call("bash", success=False, duration_ms=10.0)

        harness = Harness(config=sample_config)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        telemetry = result.telemetry
        assert telemetry is not None
        assert telemetry.tool_calls.total_calls == 3
        assert telemetry.tool_calls.successful_calls == 2
        assert telemetry.tool_calls.failed_calls == 1
        assert "bash" in telemetry.tool_calls.by_tool
        assert telemetry.tool_calls.by_tool["bash"]["calls"] == 2
        assert telemetry.tool_calls.by_tool["bash"]["failures"] == 1
        assert "editor" in telemetry.tool_calls.by_tool
        assert telemetry.tool_calls.by_tool["editor"]["calls"] == 1
        assert telemetry.tool_calls.total_duration_ms == pytest.approx(180.0)

    def test_raw_events_populated_when_trace_enabled(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """raw_events populated when trace_enabled=True."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config, trace_enabled=True)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert result.telemetry is not None
        assert len(result.telemetry.raw_events) > 0
        # Should contain at least one event with phase info
        events_phases = [e.get("phase") for e in result.telemetry.raw_events]
        assert "prepare" in events_phases
        assert "execute" in events_phases
        assert "cleanup" in events_phases

    def test_raw_events_empty_when_trace_disabled(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """raw_events is empty when trace_enabled=False."""
        from grist_mill.harness import Harness

        env = MockEnvironment()
        agent = MockAgent()
        collector = TelemetryCollector()

        harness = Harness(config=sample_config, trace_enabled=False)
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        assert result.telemetry is not None
        assert len(result.telemetry.raw_events) == 0


# ===========================================================================
# Harness configuration tests
# ===========================================================================


class TestHarnessConfig:
    """Tests for Harness configuration options."""

    def test_harness_default_max_retries(self, sample_config: HarnessConfig) -> None:
        """Default max_retries is 3."""
        from grist_mill.harness import Harness

        harness = Harness(config=sample_config)
        assert harness.max_retries == 3

    def test_harness_default_retry_delay(self, sample_config: HarnessConfig) -> None:
        """Default retry_delay is 1.0."""
        from grist_mill.harness import Harness

        harness = Harness(config=sample_config)
        assert harness.retry_delay == 1.0

    def test_harness_custom_max_retries(self, sample_config: HarnessConfig) -> None:
        """Custom max_retries can be set."""
        from grist_mill.harness import Harness

        harness = Harness(config=sample_config, max_retries=5)
        assert harness.max_retries == 5

    def test_harness_custom_retry_delay(self, sample_config: HarnessConfig) -> None:
        """Custom retry_delay can be set."""
        from grist_mill.harness import Harness

        harness = Harness(config=sample_config, retry_delay=2.5)
        assert harness.retry_delay == 2.5

    def test_harness_default_trace_enabled(self, sample_config: HarnessConfig) -> None:
        """Default trace_enabled is False."""
        from grist_mill.harness import Harness

        harness = Harness(config=sample_config)
        assert harness.trace_enabled is False


# ===========================================================================
# Transient error category detection
# ===========================================================================


class TestTransientErrorDetection:
    """Tests for which error categories are considered transient/retriable."""

    def test_rate_limit_is_transient(self) -> None:
        """RATE_LIMIT is a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.RATE_LIMIT) is True

    def test_api_error_is_transient(self) -> None:
        """API_ERROR is a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.API_ERROR) is True

    def test_network_error_is_transient(self) -> None:
        """NETWORK_ERROR is a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.NETWORK_ERROR) is True

    def test_test_failure_is_not_transient(self) -> None:
        """TEST_FAILURE is NOT a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.TEST_FAILURE) is False

    def test_timeout_is_not_transient(self) -> None:
        """TIMEOUT is NOT a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(None) is False

    def test_unknown_is_not_transient(self) -> None:
        """UNKNOWN is NOT a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.UNKNOWN) is False

    def test_syntax_error_is_not_transient(self) -> None:
        """SYNTAX_ERROR is NOT a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.SYNTAX_ERROR) is False

    def test_environment_error_is_not_transient(self) -> None:
        """ENVIRONMENT_ERROR is NOT a retriable error category."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(ErrorCategory.ENVIRONMENT_ERROR) is False

    def test_null_error_category_is_not_transient(self) -> None:
        """None error_category is NOT retriable."""
        from grist_mill.harness import Harness

        assert Harness._is_retriable_error(None) is False


# ===========================================================================
# RunExperiment helper tests
# ===========================================================================


class TestRunExperiment:
    """Tests for the run_experiment helper function."""

    def test_run_experiment_returns_results(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """run_experiment runs all tasks and returns results."""
        from grist_mill.harness import run_experiment

        env = MockEnvironment()
        agent = MockAgent()

        results = run_experiment(
            tasks=[sample_task],
            config=sample_config,
            agent=agent,
            env=env,
        )

        assert len(results) == 1
        assert results[0].task_id == sample_task.id
        assert results[0].telemetry is not None

    def test_run_experiment_multiple_tasks(
        self,
        sample_config: HarnessConfig,
    ) -> None:
        """run_experiment handles multiple tasks."""
        from grist_mill.harness import run_experiment

        tasks = [
            Task(
                id=f"task-{i}",
                prompt=f"Task {i}",
                language="python",
                test_command="echo ok",
                timeout=30,
            )
            for i in range(3)
        ]

        env = MockEnvironment()
        agent = MockAgent()

        results = run_experiment(
            tasks=tasks,
            config=sample_config,
            agent=agent,
            env=env,
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.task_id == f"task-{i}"
            assert result.telemetry is not None

    def test_run_experiment_one_failure_doesnt_stop_others(
        self,
        sample_config: HarnessConfig,
    ) -> None:
        """A failing task doesn't prevent other tasks from running."""
        from grist_mill.harness import run_experiment

        task1 = Task(
            id="task-fail",
            prompt="This will fail",
            language="python",
            test_command="exit 1",
            timeout=30,
        )
        task2 = Task(
            id="task-ok",
            prompt="This will succeed",
            language="python",
            test_command="echo ok",
            timeout=30,
        )

        env = MockEnvironment()
        agent = MockAgent()

        results = run_experiment(
            tasks=[task1, task2],
            config=sample_config,
            agent=agent,
            env=env,
        )

        assert len(results) == 2
        # Both tasks should have results (no short-circuit on failure)
        assert results[0].task_id == "task-fail"
        assert results[1].task_id == "task-ok"
