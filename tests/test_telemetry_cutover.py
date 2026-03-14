"""Tests for canonical telemetry cutover."""

from bench.experiments import ExperimentConfig, ExperimentRunner
from bench.experiments.matrix import ExperimentRun, ModelSpec, TaskSpec
from bench.harness import (
    ErrorCategory,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    PiDockerHarness,
    TokenUsage,
)
from bench.telemetry import LatencyBreakdown, TelemetryCollector, TelemetrySchema, ToolCallMetrics
from bench.telemetry import TokenUsage as TelemetryTokenUsage


def test_telemetry_collector_tracks_duration_by_tool() -> None:
    """Collector accumulates per-tool duration for canonical export."""
    collector = TelemetryCollector()
    collector.start()
    collector.record_tool_call("read_file", success=True, duration_ms=10.0)
    collector.record_tool_call("read_file", success=False, duration_ms=5.5)
    collector.record_tool_call("run_tests", success=True, duration_ms=20.0)

    telemetry = collector.collect(harness="test_harness")

    assert telemetry.tools.by_tool == {"read_file": 2, "run_tests": 1}
    assert telemetry.tools.errors == {"read_file": 1}
    assert telemetry.tools.duration_ms_by_tool["read_file"] == 15.5
    assert telemetry.tools.duration_ms_by_tool["run_tests"] == 20.0


def test_pi_docker_harness_emits_canonical_telemetry() -> None:
    """Pi Docker adapter emits TelemetrySchema on HarnessResult."""
    harness = PiDockerHarness(HarnessConfig())
    request = HarnessRequest(task_id="task-1", prompt="Write tests")

    result = harness._convert_result(
        request=request,
        result_dict={
            "success": True,
            "output": "ok",
            "generated_code": "print('ok')",
            "test_results": {"passed": True, "num_passed": 1, "num_failed": 0},
            "error": "",
            "model": "openrouter/glm-5",
            "token_usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "cache_read_tokens": 2,
                "cache_write_tokens": 1,
                "turn_count": 3,
            },
            "tool_call_counts": {"docker_evaluation": 1},
            "tool_errors": {},
            "tool_total_time_ms": {"docker_evaluation": 125.0},
        },
        execution_time=0.25,
        run_id="run-1",
    )

    assert result.telemetry is not None
    assert result.telemetry.harness == "pi_docker"
    assert result.telemetry.model == "openrouter/glm-5"
    assert result.telemetry.tools.by_tool == {"docker_evaluation": 1}
    assert result.telemetry.tools.duration_ms_by_tool == {"docker_evaluation": 125.0}


def test_canonical_result_conversion_reads_telemetry_not_metadata() -> None:
    """ResultV1 tool metrics are derived from HarnessResult.telemetry only."""
    runner = ExperimentRunner(ExperimentConfig(name="telemetry-cutover"))
    run = ExperimentRun(
        run_index=0,
        task=TaskSpec(task_id="task-1", task_path="tasks/dev/task-1.json"),
        model=ModelSpec(name="glm-5", litellm_model="openrouter/glm-5"),
        repeat_index=0,
        fingerprint="fp-1",
        support_profile=None,
        pair_id=None,
        pair_role=None,
    )

    harness_result = HarnessResult(
        task_id="task-1",
        run_id="run-1",
        success=True,
        error_category=ErrorCategory.NONE,
        execution_time=1.0,
        token_usage=TokenUsage(prompt=12, completion=8, total=20),
        telemetry=TelemetrySchema(
            tokens=TelemetryTokenUsage(prompt=12, completion=8, total=20),
            turns=2,
            tools=ToolCallMetrics(
                total_calls=3,
                successful_calls=2,
                failed_calls=1,
                total_duration_ms=42.0,
                by_tool={"real_tool": 3},
                errors={"real_tool": 1},
                duration_ms_by_tool={"real_tool": 42.0},
            ),
            latency=LatencyBreakdown(total_s=1.0, execution_s=1.0),
            model="glm-5",
            harness="pi_docker",
        ),
        metadata={
            "tool_call_counts": {"legacy_tool": 99},
            "tool_errors": {"legacy_tool": 99},
            "tool_total_time_ms": {"legacy_tool": 99.0},
        },
    )

    result = runner._convert_harness_result(harness_result, run, latency=1.0)

    assert result.tool_call_counts == {"real_tool": 3}
    assert result.tool_errors == {"real_tool": 1}
    assert result.tool_total_time_ms == {"real_tool": 42.0}
    assert "legacy_tool" not in result.tool_call_counts
    assert result.telemetry is not None


class TestFailurePathTelemetry:
    """Tests for telemetry completeness on failure paths."""

    def test_timeout_result_has_telemetry(self) -> None:
        """Timeout results must include telemetry with latency."""
        import subprocess
        from unittest.mock import patch

        from bench.harness.adapters.cli_base import CliHarnessBase

        # Create a concrete subclass for testing
        class TestCliHarness(CliHarnessBase):
            @property
            def cli_name(self) -> str:
                return "test-cli"

            def build_cli_args(self, request: HarnessRequest) -> list[str]:
                return ["--test"]

            def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
                return HarnessResult(
                    task_id=request.task_id,
                    run_id="test-run",
                    success=True,
                )

        harness = TestCliHarness(HarnessConfig(timeout=1.0, model="test/model"))
        request = HarnessRequest(task_id="task-1", prompt="test prompt")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="test-cli", timeout=1.0)

            import asyncio

            result = asyncio.run(harness.execute(request))

        assert result.telemetry is not None
        assert result.telemetry.latency.total_s > 0
        assert result.telemetry.harness == "test-cli"
        assert result.telemetry.model == "test/model"
        assert result.telemetry.provider == "test"

    def test_exception_result_has_telemetry(self) -> None:
        """Exception results must include telemetry."""
        from unittest.mock import patch

        from bench.harness.adapters.cli_base import CliHarnessBase

        class TestCliHarness(CliHarnessBase):
            @property
            def cli_name(self) -> str:
                return "test-cli"

            def build_cli_args(self, request: HarnessRequest) -> list[str]:
                return ["--test"]

            def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
                return HarnessResult(
                    task_id=request.task_id,
                    run_id="test-run",
                    success=True,
                )

        harness = TestCliHarness(HarnessConfig(timeout=60.0, model="test/model"))
        request = HarnessRequest(task_id="task-1", prompt="test prompt")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("Simulated error")

            import asyncio

            result = asyncio.run(harness.execute(request))

        assert result.telemetry is not None
        assert result.telemetry.tokens.prompt == 0  # default for unknown
        assert result.telemetry.tokens.completion == 0
        assert result.telemetry.latency.total_s >= 0

    def test_nonzero_return_result_has_telemetry(self) -> None:
        """Non-zero return results must include telemetry."""
        from unittest.mock import MagicMock, patch

        from bench.harness.adapters.cli_base import CliHarnessBase

        class TestCliHarness(CliHarnessBase):
            @property
            def cli_name(self) -> str:
                return "test-cli"

            def build_cli_args(self, request: HarnessRequest) -> list[str]:
                return ["--test"]

            def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
                return HarnessResult(
                    task_id=request.task_id,
                    run_id="test-run",
                    success=True,
                )

        harness = TestCliHarness(HarnessConfig(timeout=60.0, model="test/model"))
        request = HarnessRequest(task_id="task-1", prompt="test prompt")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "some output"
        mock_result.stderr = "error message"

        with patch("subprocess.run", return_value=mock_result):
            import asyncio

            result = asyncio.run(harness.execute(request))

        assert result.telemetry is not None
        assert result.telemetry.harness == "test-cli"
        assert result.telemetry.latency.total_s >= 0

    def test_success_result_has_full_telemetry(self) -> None:
        """Success results should have full telemetry (existing behavior)."""
        from unittest.mock import MagicMock, patch

        from bench.harness.adapters.cli_base import CliHarnessBase

        class TestCliHarness(CliHarnessBase):
            @property
            def cli_name(self) -> str:
                return "test-cli"

            def build_cli_args(self, request: HarnessRequest) -> list[str]:
                return ["--test"]

            def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
                return HarnessResult(
                    task_id=request.task_id,
                    run_id="test-run",
                    success=True,
                    token_usage=TokenUsage(prompt=100, completion=50, total=150),
                    turns=2,
                )

        harness = TestCliHarness(HarnessConfig(timeout=60.0, model="test/model"))
        request = HarnessRequest(task_id="task-1", prompt="test prompt")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            import asyncio

            result = asyncio.run(harness.execute(request))

        assert result.telemetry is not None
        assert result.telemetry.tokens.prompt == 100
        assert result.telemetry.tokens.completion == 50
        assert result.telemetry.turns == 2
        assert result.telemetry.harness == "test-cli"

    def test_pi_docker_exception_has_telemetry(self) -> None:
        """Pi Docker exception results must include telemetry."""
        from unittest.mock import MagicMock

        harness = PiDockerHarness(HarnessConfig(model="test/model"))
        harness._runner = MagicMock()
        harness._runner.run_evaluation.side_effect = RuntimeError("Docker error")

        request = HarnessRequest(task_id="task-1", prompt="test prompt")

        import asyncio

        result = asyncio.run(harness.execute(request))

        assert result.telemetry is not None
        assert result.telemetry.harness == "pi_docker"
        assert result.telemetry.model == "test/model"
        assert result.telemetry.latency.total_s >= 0
        assert result.telemetry.tokens.prompt == 0  # default for unknown
