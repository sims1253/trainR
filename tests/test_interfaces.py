"""Tests for abstract base classes (VAL-FOUND-06, VAL-FOUND-07, VAL-FOUND-08, VAL-FOUND-09).

Covers:
- BaseAgent: abstract run method, TypeError on incomplete subclass
- BaseBenchmark: setup/teardown/submit_result, submit_result validates TaskResult type
- BaseEnvironment: prepare/execute/cleanup, execute returns ExecutionOutput
- BaseHarness: orchestrates env.prepare -> agent.run -> env.cleanup, guaranteed cleanup on exception
- LocalHarness: reference implementation using LocalEnvironment
"""

from __future__ import annotations

from typing import Any

import pytest

from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ExecutionOutput,
    HarnessConfig,
    Task,
    TaskResult,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_task() -> Task:
    """Return a valid Task for testing."""
    return Task(
        id="test-001",
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
def sample_result(sample_task: Task) -> TaskResult:
    """Return a valid TaskResult for testing."""
    return TaskResult(
        task_id=sample_task.id,
        status=TaskStatus.SUCCESS,
        score=1.0,
    )


# ===========================================================================
# BaseAgent tests (VAL-FOUND-07)
# ===========================================================================


class TestBaseAgent:
    """Tests for BaseAgent abstract interface."""

    def test_cannot_instantiate_base_agent_directly(self) -> None:
        """BaseAgent itself cannot be instantiated."""
        from grist_mill.interfaces import BaseAgent

        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_incomplete_subclass_raises_typeerror(self) -> None:
        """A subclass that does not implement run fails at instantiation."""
        from grist_mill.interfaces import BaseAgent

        class IncompleteAgent(BaseAgent):
            pass  # missing run method

        with pytest.raises(TypeError, match="run"):
            IncompleteAgent()

    def test_complete_subclass_instantiates(self) -> None:
        """A subclass implementing run can be instantiated."""
        from grist_mill.interfaces import BaseAgent

        class CompleteAgent(BaseAgent):
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    score=1.0,
                )

        agent = CompleteAgent()
        assert isinstance(agent, CompleteAgent)

    def test_complete_agent_run_returns_result(
        self, sample_task: Task, sample_config: HarnessConfig
    ) -> None:
        """A complete agent's run method returns a TaskResult."""
        from grist_mill.interfaces import BaseAgent

        expected = TaskResult(
            task_id=sample_task.id,
            status=TaskStatus.FAILURE,
            score=0.0,
        )

        class MyAgent(BaseAgent):
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                return expected

        agent = MyAgent()
        result = agent.run(sample_task, sample_config)
        assert result is expected
        assert result.task_id == "test-001"
        assert result.status == TaskStatus.FAILURE


# ===========================================================================
# BaseBenchmark tests (VAL-FOUND-06)
# ===========================================================================


class TestBaseBenchmark:
    """Tests for BaseBenchmark abstract interface."""

    def test_cannot_instantiate_base_benchmark_directly(self) -> None:
        """BaseBenchmark itself cannot be instantiated."""
        from grist_mill.interfaces import BaseBenchmark

        with pytest.raises(TypeError):
            BaseBenchmark()  # type: ignore[abstract]

    def test_incomplete_subclass_raises_typeerror(self) -> None:
        """A subclass missing all abstract methods fails at instantiation."""
        from grist_mill.interfaces import BaseBenchmark

        class IncompleteBenchmark(BaseBenchmark):
            pass

        with pytest.raises(TypeError):
            IncompleteBenchmark()

    def test_complete_subclass_instantiates(self) -> None:
        """A subclass implementing all abstract methods can be instantiated."""
        from grist_mill.interfaces import BaseBenchmark

        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None:
                pass

            def teardown(self) -> None:
                pass

            def _submit_result(self, result: TaskResult) -> None:
                pass

        bench = MyBenchmark()
        assert isinstance(bench, MyBenchmark)

    def test_submit_result_accepts_taskresult(self, sample_result: TaskResult) -> None:
        """submit_result accepts a valid TaskResult instance."""
        from grist_mill.interfaces import BaseBenchmark

        collected: list[TaskResult] = []

        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None:
                pass

            def teardown(self) -> None:
                pass

            def _submit_result(self, result: TaskResult) -> None:
                collected.append(result)

        bench = MyBenchmark()
        bench.submit_result(sample_result)
        assert len(collected) == 1
        assert collected[0].task_id == "test-001"

    def test_submit_result_rejects_non_taskresult(self) -> None:
        """submit_result raises TypeError for non-TaskResult input."""
        from grist_mill.interfaces import BaseBenchmark

        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None:
                pass

            def teardown(self) -> None:
                pass

            def _submit_result(self, result: TaskResult) -> None:
                pass

        bench = MyBenchmark()

        with pytest.raises(TypeError, match="TaskResult"):
            bench.submit_result("not a result")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="TaskResult"):
            bench.submit_result({"task_id": "x", "status": "SUCCESS"})

        with pytest.raises(TypeError, match="TaskResult"):
            bench.submit_result(42)  # type: ignore[arg-type]

    def test_setup_and_teardown_are_callable(self) -> None:
        """setup and teardown can be called on a concrete benchmark."""
        from grist_mill.interfaces import BaseBenchmark

        calls: list[str] = []

        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None:
                calls.append("setup")

            def teardown(self) -> None:
                calls.append("teardown")

            def _submit_result(self, result: TaskResult) -> None:
                calls.append("submit_result")

        bench = MyBenchmark()
        bench.setup()
        assert calls == ["setup"]
        bench.teardown()
        assert calls == ["setup", "teardown"]


# ===========================================================================
# BaseEnvironment tests (VAL-FOUND-08)
# ===========================================================================


class TestBaseEnvironment:
    """Tests for BaseEnvironment abstract interface."""

    def test_cannot_instantiate_base_environment_directly(self) -> None:
        """BaseEnvironment itself cannot be instantiated."""
        from grist_mill.interfaces import BaseEnvironment

        with pytest.raises(TypeError):
            BaseEnvironment()  # type: ignore[abstract]

    def test_incomplete_subclass_raises_typeerror(self) -> None:
        """A subclass missing abstract methods fails at instantiation."""
        from grist_mill.interfaces import BaseEnvironment

        class IncompleteEnv(BaseEnvironment):
            pass

        with pytest.raises(TypeError):
            IncompleteEnv()

    def test_complete_subclass_instantiates(self) -> None:
        """A subclass implementing all abstract methods can be instantiated."""
        from grist_mill.interfaces import BaseEnvironment

        class MyEnv(BaseEnvironment):
            def prepare(self, task: Task) -> None:
                pass

            def execute(self, command: str, timeout: float) -> ExecutionOutput:
                return ExecutionOutput(stdout="", stderr="", exit_code=0)

            def cleanup(self) -> None:
                pass

        env = MyEnv()
        assert isinstance(env, MyEnv)

    def test_execute_returns_execution_output(self, sample_task: Task) -> None:
        """execute returns a typed ExecutionOutput."""
        from grist_mill.interfaces import BaseEnvironment

        class MyEnv(BaseEnvironment):
            def prepare(self, task: Task) -> None:
                pass

            def execute(self, command: str, timeout: float) -> ExecutionOutput:
                return ExecutionOutput(
                    stdout="hello world",
                    stderr="",
                    exit_code=0,
                )

            def cleanup(self) -> None:
                pass

        env = MyEnv()
        output = env.execute("echo hello", timeout=10.0)
        assert isinstance(output, ExecutionOutput)
        assert output.stdout == "hello world"
        assert output.exit_code == 0
        assert output.timed_out is False

    def test_local_environment_echo_hello(self, sample_task: Task) -> None:
        """LocalEnvironment executes 'echo hello' and returns correct ExecutionOutput."""
        from grist_mill.interfaces import LocalEnvironment

        env = LocalEnvironment()
        env.prepare(sample_task)
        output = env.execute("echo hello", timeout=10.0)
        env.cleanup()

        assert isinstance(output, ExecutionOutput)
        assert "hello" in output.stdout
        assert output.exit_code == 0
        assert output.timed_out is False

    def test_local_environment_timeout(self, sample_task: Task) -> None:
        """LocalEnvironment marks timed_out=True when command exceeds timeout."""
        from grist_mill.interfaces import LocalEnvironment

        env = LocalEnvironment()
        env.prepare(sample_task)
        output = env.execute("sleep 60", timeout=0.5)
        env.cleanup()

        assert isinstance(output, ExecutionOutput)
        assert output.timed_out is True

    def test_local_environment_nonzero_exit(self, sample_task: Task) -> None:
        """LocalEnvironment captures non-zero exit codes."""
        from grist_mill.interfaces import LocalEnvironment

        env = LocalEnvironment()
        env.prepare(sample_task)
        output = env.execute("exit 42", timeout=10.0)
        env.cleanup()

        assert isinstance(output, ExecutionOutput)
        assert output.exit_code == 42
        assert output.timed_out is False


# ===========================================================================
# BaseHarness tests (VAL-FOUND-09)
# ===========================================================================


class TestBaseHarness:
    """Tests for BaseHarness abstract interface."""

    def test_cannot_instantiate_base_harness_directly(self) -> None:
        """BaseHarness itself cannot be instantiated."""
        from grist_mill.interfaces import BaseHarness

        with pytest.raises(TypeError):
            BaseHarness()  # type: ignore[abstract]

    def test_local_harness_instantiates(self, sample_config: HarnessConfig) -> None:
        """LocalHarness can be instantiated with a HarnessConfig."""
        from grist_mill.interfaces import LocalHarness

        harness = LocalHarness(config=sample_config)
        assert isinstance(harness, LocalHarness)

    def test_harness_run_calls_prepare_agent_cleanup_in_order(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
        sample_result: TaskResult,
    ) -> None:
        """BaseHarness.run calls env.prepare, agent.run, env.cleanup in order."""
        from grist_mill.interfaces import BaseHarness

        call_order: list[str] = []

        class MockEnv:
            def prepare(self, task: Task) -> None:
                call_order.append("prepare")

            def execute(self, command: str, timeout: float) -> ExecutionOutput:
                return ExecutionOutput(stdout="test", stderr="", exit_code=0)

            def cleanup(self) -> None:
                call_order.append("cleanup")

        class MockAgent:
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                call_order.append("agent_run")
                return sample_result

        class MyHarness(BaseHarness):
            def __init__(self, config: HarnessConfig) -> None:
                self.__config = config

            @property
            def _config(self) -> HarnessConfig:
                return self.__config

            def run(
                self,
                task: Task,
                agent: Any,
                env: Any,
            ) -> TaskResult:
                return self._run(task, agent, env)

        harness = MyHarness(config=sample_config)
        result = harness.run(sample_task, MockAgent(), MockEnv())

        assert call_order == ["prepare", "agent_run", "cleanup"]
        assert result.task_id == "test-001"

    def test_harness_guarantees_cleanup_on_agent_exception(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """BaseHarness guarantees env.cleanup is called even when agent raises."""
        from grist_mill.interfaces import BaseHarness

        cleanup_called = False

        class MockEnv:
            def prepare(self, task: Task) -> None:
                pass

            def execute(self, command: str, timeout: float) -> ExecutionOutput:
                return ExecutionOutput(stdout="", stderr="", exit_code=0)

            def cleanup(self) -> None:
                nonlocal cleanup_called
                cleanup_called = True

        class FailingAgent:
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                raise RuntimeError("Agent crashed!")

        class MyHarness(BaseHarness):
            def __init__(self, config: HarnessConfig) -> None:
                self.__config = config

            @property
            def _config(self) -> HarnessConfig:
                return self.__config

            def run(
                self,
                task: Task,
                agent: Any,
                env: Any,
            ) -> TaskResult:
                return self._run(task, agent, env)

        harness = MyHarness(config=sample_config)

        with pytest.raises(RuntimeError, match="Agent crashed"):
            harness.run(sample_task, FailingAgent(), MockEnv())

        assert cleanup_called, "cleanup was not called after agent exception"

    def test_harness_guarantees_cleanup_on_agent_exception_with_local_harness(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """LocalHarness guarantees cleanup when agent raises."""
        from grist_mill.interfaces import LocalHarness

        cleanup_called = False

        class TrackedEnv:
            def prepare(self, task: Task) -> None:
                pass

            def execute(self, command: str, timeout: float) -> ExecutionOutput:
                return ExecutionOutput(stdout="", stderr="", exit_code=0)

            def cleanup(self) -> None:
                nonlocal cleanup_called
                cleanup_called = True

        class FailingAgent:
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                raise RuntimeError("Agent failure")

        harness = LocalHarness(config=sample_config)

        with pytest.raises(RuntimeError, match="Agent failure"):
            harness.run(sample_task, FailingAgent(), TrackedEnv())

        assert cleanup_called, "cleanup was not called by LocalHarness after agent exception"

    def test_local_harness_end_to_end(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """LocalHarness runs end-to-end with LocalEnvironment and a working agent."""
        from grist_mill.interfaces import BaseAgent, LocalEnvironment, LocalHarness

        class EchoAgent(BaseAgent):
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    score=1.0,
                )

        harness = LocalHarness(config=sample_config)
        env = LocalEnvironment()
        agent = EchoAgent()

        result = harness.run(sample_task, agent, env)
        assert isinstance(result, TaskResult)
        assert result.task_id == "test-001"
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0

    def test_harness_cleanup_on_prepare_exception(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """BaseHarness still tries cleanup even if env.prepare raises."""
        from grist_mill.interfaces import BaseHarness

        cleanup_called = False

        class FailingPrepareEnv:
            def prepare(self, task: Task) -> None:
                raise RuntimeError("Prepare failed!")

            def execute(self, command: str, timeout: float) -> ExecutionOutput:
                return ExecutionOutput(stdout="", stderr="", exit_code=0)

            def cleanup(self) -> None:
                nonlocal cleanup_called
                cleanup_called = True

        class MyHarness(BaseHarness):
            def __init__(self, config: HarnessConfig) -> None:
                self.__config = config

            @property
            def _config(self) -> HarnessConfig:
                return self.__config

            def run(
                self,
                task: Task,
                agent: Any,
                env: Any,
            ) -> TaskResult:
                return self._run(task, agent, env)

        class MockAgent:
            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                return TaskResult(task_id=task.id, status=TaskStatus.SUCCESS, score=1.0)

        harness = MyHarness(config=sample_config)

        with pytest.raises(RuntimeError, match="Prepare failed"):
            harness.run(sample_task, MockAgent(), FailingPrepareEnv())

        assert cleanup_called, "cleanup was not called after prepare exception"
