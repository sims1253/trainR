"""Cross-evaluation export tests (m2-cross-eval-export).

Verifies end-to-end: evaluation produces TaskResults that flow correctly into
the export pipeline. A task that runs through the harness produces a
TaskResult that can be exported to JSON/CSV without manual transformation.

Validates:
- VAL-CROSS-01: End-to-end from evaluation to export without manual bridging
- TaskResult from harness round-trips through JSON export without data loss
- Telemetry data is preserved in export
- Error categories are preserved in export
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from grist_mill.export.formats import export_csv, export_json
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
    TelemetryCollector,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_telemetry(
    *,
    prompt: int = 200,
    completion: int = 80,
    setup_s: float = 0.5,
    execution_s: float = 3.0,
    teardown_s: float = 0.3,
    total_calls: int = 4,
    successful_calls: int = 3,
    failed_calls: int = 1,
    cost: float | None = 0.025,
    raw_events: list[dict[str, Any]] | None = None,
) -> TelemetrySchema:
    """Create a TelemetrySchema with realistic values."""
    return TelemetrySchema(
        version="V1",
        tokens=TokenUsage(
            prompt=prompt,
            completion=completion,
            total=prompt + completion,
        ),
        latency=LatencyBreakdown(
            setup_s=setup_s,
            execution_s=execution_s,
            teardown_s=teardown_s,
            total_s=setup_s + execution_s + teardown_s,
        ),
        tool_calls=ToolCallMetrics(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            by_tool={
                "file_read": {"calls": 2, "successes": 2, "failures": 0},
                "shell_exec": {"calls": 1, "successes": 0, "failures": 1},
                "search": {"calls": 1, "successes": 1, "failures": 0},
            },
            total_duration_ms=450.0,
        ),
        estimated_cost_usd=cost,
        raw_events=raw_events or [],
    )


def _make_harness_result_dict(
    *,
    task_id: str,
    status: TaskStatus = TaskStatus.SUCCESS,
    score: float = 1.0,
    error_category: ErrorCategory | None = None,
    model: str = "gpt-4",
    provider: str = "openrouter",
    telemetry: TelemetrySchema | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a result dict matching what the harness + export pipeline expects.

    This mimics the shape produced when a TaskResult from the harness is
    enriched with model/provider metadata and converted to a flat dict for
    the export layer.
    """
    return {
        "task_id": task_id,
        "status": status,
        "score": score,
        "error_category": error_category,
        "model": model,
        "provider": provider,
        "timestamp": timestamp or datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        "telemetry": telemetry or _make_telemetry(),
    }


# ---------------------------------------------------------------------------
# Mock harness infrastructure (mirrors test_harness.py patterns)
# ---------------------------------------------------------------------------


class _MockEnvironment:
    """Mock environment for harness execution."""

    def __init__(
        self,
        execute_output: ExecutionOutput | None = None,
    ) -> None:
        self._execute_output = execute_output or ExecutionOutput(
            stdout="all tests passed",
            stderr="",
            exit_code=0,
        )
        self.call_order: list[str] = []

    def prepare(self, task: Task) -> None:
        self.call_order.append("prepare")

    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        self.call_order.append(f"execute:{command}")
        return self._execute_output

    def cleanup(self) -> None:
        self.call_order.append("cleanup")


class _MockAgent:
    """Mock agent that returns a configurable result."""

    def __init__(self, result: TaskResult | None = None) -> None:
        self._result = result
        self.call_count = 0

    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        self.call_count += 1
        return self._result or TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            score=1.0,
        )


# ===========================================================================
# VAL-CROSS-01: End-to-end harness → export without manual bridging
# ===========================================================================


class TestEndToEndHarnessExport:
    """Verify a task that runs through the harness produces results that
    can be exported to JSON/CSV without manual transformation."""

    @pytest.fixture()
    def sample_task(self) -> Task:
        return Task(
            id="cross-eval-001",
            prompt="Fix the off-by-one error in the parser.",
            language="python",
            test_command="pytest tests/test_parser.py -q",
            timeout=60,
            difficulty=Difficulty.MEDIUM,
        )

    @pytest.fixture()
    def sample_config(self) -> HarnessConfig:
        return HarnessConfig(
            agent=AgentConfig(model="gpt-4", provider="openrouter"),
            environment=EnvironmentConfig(runner_type="local"),
        )

    def test_harness_result_exports_to_json_without_loss(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """TaskResult from harness round-trips through JSON export without data loss."""
        from grist_mill.harness import Harness

        # Run through the harness
        env = _MockEnvironment()
        agent = _MockAgent(
            result=TaskResult(
                task_id=sample_task.id,
                status=TaskStatus.SUCCESS,
                score=1.0,
            )
        )
        collector = TelemetryCollector()
        collector.record_tokens(prompt=200, completion=80)
        collector.record_tool_call("file_read", success=True, duration_ms=50.0)
        collector.record_tool_call("shell_exec", success=True, duration_ms=120.0)
        collector.set_estimated_cost(0.025)

        harness = Harness(config=sample_config, trace_enabled=True)
        harness_result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Convert to export format dict
        result_dict = _make_harness_result_dict(
            task_id=harness_result.task_id,
            status=harness_result.status,
            score=harness_result.score,
            error_category=harness_result.error_category,
            model=sample_config.agent.model,
            provider=sample_config.agent.provider,
            telemetry=harness_result.telemetry,
        )

        # Export to JSON
        json_output = export_json([result_dict])
        data = json.loads(json_output)

        # Verify the exported record preserves all fields
        assert len(data["records"]) == 1
        record = data["records"][0]

        assert record["task_id"] == sample_task.id
        assert record["status"] == "SUCCESS"
        assert record["score"] == 1.0
        assert record["error_category"] is None
        assert record["model"] == "gpt-4"
        assert record["provider"] == "openrouter"
        assert record["telemetry"]["tokens"]["prompt"] == 200
        assert record["telemetry"]["tokens"]["completion"] == 80
        assert record["telemetry"]["tokens"]["total"] == 280
        assert record["telemetry"]["tool_calls"]["total_calls"] == 2
        assert record["telemetry"]["estimated_cost_usd"] == 0.025

        # JSON round-trip: re-serialize and verify it matches
        round_trip_json = json.dumps(record)
        round_trip_data = json.loads(round_trip_json)
        assert round_trip_data["task_id"] == sample_task.id
        assert round_trip_data["status"] == "SUCCESS"
        assert round_trip_data["score"] == 1.0

    def test_harness_result_exports_to_csv_without_loss(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """TaskResult from harness round-trips through CSV export without data loss."""
        from grist_mill.harness import Harness

        env = _MockEnvironment()
        agent = _MockAgent()
        collector = TelemetryCollector()
        collector.record_tokens(prompt=300, completion=150)
        collector.record_tool_call("editor", success=True, duration_ms=200.0)
        collector.set_estimated_cost(0.045)

        harness = Harness(config=sample_config)
        harness_result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        result_dict = _make_harness_result_dict(
            task_id=harness_result.task_id,
            status=harness_result.status,
            score=harness_result.score,
            error_category=harness_result.error_category,
            model=sample_config.agent.model,
            provider=sample_config.agent.provider,
            telemetry=harness_result.telemetry,
        )

        csv_output = export_csv([result_dict])
        lines = csv_output.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row

        header = lines[0].split(",")
        row = lines[1].split(",")

        # Verify key fields are in the CSV
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["task_id"] == sample_task.id
        assert header_map["status"] == "SUCCESS"
        assert header_map["score"] == "1.0"
        assert header_map["tokens_prompt"] == "300"
        assert header_map["tokens_completion"] == "150"
        assert header_map["tokens_total"] == "450"
        assert header_map["model"] == "gpt-4"
        assert header_map["provider"] == "openrouter"

    def test_multiple_harness_results_export_correctly(
        self,
        sample_config: HarnessConfig,
    ) -> None:
        """Multiple harness results export together with correct counts."""
        from grist_mill.harness import run_experiment

        tasks = [
            Task(
                id=f"multi-task-{i}",
                prompt=f"Fix issue {i}",
                language="python",
                test_command="pytest tests/ -q",
                timeout=60,
            )
            for i in range(5)
        ]

        env = _MockEnvironment()
        agent = _MockAgent()

        results = run_experiment(
            tasks=tasks,
            config=sample_config,
            agent=agent,
            env=env,
            max_retries=0,
            retry_delay=0.0,
        )

        assert len(results) == 5

        # Convert all to export dicts
        result_dicts = [
            _make_harness_result_dict(
                task_id=r.task_id,
                status=r.status,
                score=r.score,
                error_category=r.error_category,
                model=sample_config.agent.model,
                provider=sample_config.agent.provider,
                telemetry=r.telemetry,
            )
            for r in results
        ]

        # Export all to JSON
        json_output = export_json(result_dicts)
        data = json.loads(json_output)
        assert len(data["records"]) == 5
        assert data["summary"]["total_tasks"] == 5

        # Export all to CSV
        csv_output = export_csv(result_dicts)
        lines = csv_output.strip().split("\n")
        assert len(lines) == 6  # 1 header + 5 data rows


# ===========================================================================
# Telemetry preservation in export
# ===========================================================================


class TestTelemetryPreservationInExport:
    """Verify telemetry data from harness is preserved in export."""

    def test_token_usage_preserved_in_json_export(self) -> None:
        """Token usage (prompt, completion, total) is preserved in JSON export."""
        telemetry = _make_telemetry(prompt=500, completion=250, cost=0.075)
        result = _make_harness_result_dict(
            task_id="telem-tok-001",
            telemetry=telemetry,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        assert record["telemetry"]["tokens"]["prompt"] == 500
        assert record["telemetry"]["tokens"]["completion"] == 250
        assert record["telemetry"]["tokens"]["total"] == 750

    def test_token_usage_preserved_in_csv_export(self) -> None:
        """Token usage is preserved in CSV export."""
        telemetry = _make_telemetry(prompt=333, completion=111)
        result = _make_harness_result_dict(
            task_id="telem-tok-csv-001",
            telemetry=telemetry,
        )

        csv_output = export_csv([result])
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["tokens_prompt"] == "333"
        assert header_map["tokens_completion"] == "111"
        assert header_map["tokens_total"] == "444"

    def test_latency_breakdown_preserved_in_json_export(self) -> None:
        """Latency breakdown (setup, execution, teardown, total) is preserved."""
        telemetry = _make_telemetry(
            setup_s=1.2,
            execution_s=5.5,
            teardown_s=0.8,
        )
        result = _make_harness_result_dict(
            task_id="telem-lat-001",
            telemetry=telemetry,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        latency = record["telemetry"]["latency"]
        assert latency["setup_s"] == 1.2
        assert latency["execution_s"] == 5.5
        assert latency["teardown_s"] == 0.8
        assert latency["total_s"] == pytest.approx(7.5)

    def test_latency_breakdown_preserved_in_csv_export(self) -> None:
        """Latency breakdown is preserved in CSV export."""
        telemetry = _make_telemetry(
            setup_s=0.9,
            execution_s=4.2,
            teardown_s=0.6,
        )
        result = _make_harness_result_dict(
            task_id="telem-lat-csv-001",
            telemetry=telemetry,
        )

        csv_output = export_csv([result])
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["latency_setup_s"] == "0.9"
        assert header_map["latency_execution_s"] == "4.2"
        assert header_map["latency_teardown_s"] == "0.6"
        assert header_map["latency_total_s"] == "5.7"

    def test_tool_call_metrics_preserved_in_json_export(self) -> None:
        """Tool call metrics (total_calls, successful, failed, by_tool) are preserved."""
        telemetry = _make_telemetry(
            total_calls=7,
            successful_calls=5,
            failed_calls=2,
        )
        result = _make_harness_result_dict(
            task_id="telem-tool-001",
            telemetry=telemetry,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        tc = record["telemetry"]["tool_calls"]
        assert tc["total_calls"] == 7
        assert tc["successful_calls"] == 5
        assert tc["failed_calls"] == 2
        assert "file_read" in tc["by_tool"]
        assert tc["by_tool"]["file_read"]["calls"] == 2
        assert tc["total_duration_ms"] == 450.0

    def test_tool_call_metrics_preserved_in_csv_export(self) -> None:
        """Tool call metrics are preserved in CSV export."""
        telemetry = _make_telemetry(
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
        )
        result = _make_harness_result_dict(
            task_id="telem-tool-csv-001",
            telemetry=telemetry,
        )

        csv_output = export_csv([result])
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["tool_calls_total"] == "10"
        assert header_map["tool_calls_successful"] == "8"
        assert header_map["tool_calls_failed"] == "2"

    def test_cost_preserved_in_json_export(self) -> None:
        """Estimated cost is preserved in JSON export."""
        telemetry = _make_telemetry(cost=0.12345)
        result = _make_harness_result_dict(
            task_id="telem-cost-001",
            telemetry=telemetry,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        assert record["telemetry"]["estimated_cost_usd"] == 0.12345

    def test_cost_preserved_in_csv_export(self) -> None:
        """Estimated cost is preserved in CSV export."""
        telemetry = _make_telemetry(cost=0.09876)
        result = _make_harness_result_dict(
            task_id="telem-cost-csv-001",
            telemetry=telemetry,
        )

        csv_output = export_csv([result])
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["estimated_cost_usd"] == "0.09876"

    def test_raw_events_preserved_in_json_export(self) -> None:
        """Raw events (audit trail) are preserved in JSON export."""
        raw_events = [
            {"phase": "prepare", "task_id": "audit-001", "status": "completed"},
            {"phase": "execute", "task_id": "audit-001", "exit_code": 0},
            {"phase": "cleanup", "task_id": "audit-001", "status": "completed"},
        ]
        telemetry = _make_telemetry(raw_events=raw_events)
        result = _make_harness_result_dict(
            task_id="telem-events-001",
            telemetry=telemetry,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        events = record["telemetry"]["raw_events"]
        assert len(events) == 3
        assert events[0]["phase"] == "prepare"
        assert events[1]["exit_code"] == 0
        assert events[2]["phase"] == "cleanup"

    def test_telemetry_with_zero_tokens_preserved(self) -> None:
        """Telemetry with zero tokens (adapter didn't report) is preserved."""
        telemetry = _make_telemetry(prompt=0, completion=0, cost=None)
        result = _make_harness_result_dict(
            task_id="telem-zero-001",
            telemetry=telemetry,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        assert record["telemetry"]["tokens"]["prompt"] == 0
        assert record["telemetry"]["tokens"]["completion"] == 0
        assert record["telemetry"]["tokens"]["total"] == 0
        assert record["telemetry"]["estimated_cost_usd"] is None

    def test_telemetry_summary_in_json_export(self) -> None:
        """JSON export summary aggregates telemetry correctly."""
        results = [
            _make_harness_result_dict(
                task_id=f"summary-{i}",
                status=TaskStatus.SUCCESS if i < 3 else TaskStatus.FAILURE,
                score=1.0 if i < 3 else 0.0,
                error_category=None if i < 3 else ErrorCategory.TEST_FAILURE,
                telemetry=_make_telemetry(
                    prompt=100 + i * 50,
                    completion=50 + i * 25,
                    cost=0.01 + i * 0.005,
                ),
            )
            for i in range(5)
        ]

        json_output = export_json(results)
        data = json.loads(json_output)

        summary = data["summary"]
        assert summary["total_tasks"] == 5
        assert summary["pass_rate"] == pytest.approx(0.6, abs=0.01)
        assert summary["total_tokens"] > 0
        assert summary["mean_latency_s"] > 0
        assert summary["total_cost_usd"] > 0


# ===========================================================================
# Error category preservation in export
# ===========================================================================


class TestErrorCategoryPreservationInExport:
    """Verify error categories from harness results are preserved in export."""

    @pytest.mark.parametrize(
        "error_category",
        [
            ErrorCategory.TEST_FAILURE,
            ErrorCategory.SYNTAX_ERROR,
            ErrorCategory.ENVIRONMENT_ERROR,
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.API_ERROR,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.MAX_TURNS_EXCEEDED,
            ErrorCategory.UNKNOWN,
        ],
        ids=[
            "TEST_FAILURE",
            "SYNTAX_ERROR",
            "ENVIRONMENT_ERROR",
            "NETWORK_ERROR",
            "API_ERROR",
            "RATE_LIMIT",
            "MAX_TURNS_EXCEEDED",
            "UNKNOWN",
        ],
    )
    def test_error_category_preserved_in_json_export(
        self,
        error_category: ErrorCategory,
    ) -> None:
        """Each error category round-trips through JSON export."""
        result = _make_harness_result_dict(
            task_id=f"err-cat-json-{error_category.value}",
            status=TaskStatus.ERROR
            if error_category != ErrorCategory.TEST_FAILURE
            else TaskStatus.FAILURE,
            score=0.0,
            error_category=error_category,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        assert record["error_category"] == error_category.value

    @pytest.mark.parametrize(
        "error_category",
        [
            ErrorCategory.TEST_FAILURE,
            ErrorCategory.SYNTAX_ERROR,
            ErrorCategory.ENVIRONMENT_ERROR,
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.API_ERROR,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.MAX_TURNS_EXCEEDED,
            ErrorCategory.UNKNOWN,
        ],
        ids=[
            "TEST_FAILURE",
            "SYNTAX_ERROR",
            "ENVIRONMENT_ERROR",
            "NETWORK_ERROR",
            "API_ERROR",
            "RATE_LIMIT",
            "MAX_TURNS_EXCEEDED",
            "UNKNOWN",
        ],
    )
    def test_error_category_preserved_in_csv_export(
        self,
        error_category: ErrorCategory,
    ) -> None:
        """Each error category round-trips through CSV export."""
        result = _make_harness_result_dict(
            task_id=f"err-cat-csv-{error_category.value}",
            status=TaskStatus.ERROR
            if error_category != ErrorCategory.TEST_FAILURE
            else TaskStatus.FAILURE,
            score=0.0,
            error_category=error_category,
        )

        csv_output = export_csv([result])
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["error_category"] == error_category.value

    def test_none_error_category_preserved_in_json(self) -> None:
        """None error_category (successful task) is preserved in JSON."""
        result = _make_harness_result_dict(
            task_id="no-error-json-001",
            status=TaskStatus.SUCCESS,
            score=1.0,
            error_category=None,
        )

        json_output = export_json([result])
        data = json.loads(json_output)
        record = data["records"][0]

        assert record["error_category"] is None

    def test_none_error_category_preserved_in_csv(self) -> None:
        """None error_category is preserved as empty string in CSV."""
        result = _make_harness_result_dict(
            task_id="no-error-csv-001",
            status=TaskStatus.SUCCESS,
            score=1.0,
            error_category=None,
        )

        csv_output = export_csv([result])
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["error_category"] == ""

    def test_mixed_results_with_errors_export_correctly(self) -> None:
        """Mixed success/failure/error results all export with correct categories."""
        results = [
            _make_harness_result_dict(
                task_id="mixed-success",
                status=TaskStatus.SUCCESS,
                score=1.0,
                error_category=None,
            ),
            _make_harness_result_dict(
                task_id="mixed-test-fail",
                status=TaskStatus.FAILURE,
                score=0.0,
                error_category=ErrorCategory.TEST_FAILURE,
            ),
            _make_harness_result_dict(
                task_id="mixed-syntax",
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.SYNTAX_ERROR,
            ),
            _make_harness_result_dict(
                task_id="mixed-timeout",
                status=TaskStatus.TIMEOUT,
                score=0.0,
                error_category=None,
            ),
        ]

        json_output = export_json(results)
        data = json.loads(json_output)
        records = data["records"]

        assert len(records) == 4
        assert records[0]["error_category"] is None
        assert records[0]["status"] == "SUCCESS"
        assert records[1]["error_category"] == "TEST_FAILURE"
        assert records[1]["status"] == "FAILURE"
        assert records[2]["error_category"] == "SYNTAX_ERROR"
        assert records[2]["status"] == "ERROR"
        assert records[3]["status"] == "TIMEOUT"

    def test_harness_error_result_exports_correctly(
        self,
    ) -> None:
        """A harness run that produces an error exports correctly with telemetry."""
        from grist_mill.harness import Harness

        task = Task(
            id="harness-err-001",
            prompt="Do something",
            language="python",
            test_command="echo ok",
            timeout=30,
        )
        config = HarnessConfig(
            agent=AgentConfig(model="gpt-4", provider="openai"),
            environment=EnvironmentConfig(runner_type="local"),
        )

        env = _MockEnvironment(
            execute_output=ExecutionOutput(stdout="", stderr="fail", exit_code=1),
        )
        agent = _MockAgent(
            result=TaskResult(
                task_id=task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.ENVIRONMENT_ERROR,
            )
        )
        collector = TelemetryCollector()

        harness = Harness(config=config)
        harness_result = harness.run(
            task=task,
            agent=agent,
            env=env,
            collector=collector,
        )

        result_dict = _make_harness_result_dict(
            task_id=harness_result.task_id,
            status=harness_result.status,
            score=harness_result.score,
            error_category=harness_result.error_category,
            model=config.agent.model,
            provider=config.agent.provider,
            telemetry=harness_result.telemetry,
        )

        json_output = export_json([result_dict])
        data = json.loads(json_output)
        record = data["records"][0]

        # The harness ran the test command (exit_code=1) after the agent error,
        # so the final status reflects the test execution result.
        assert record["status"] == "FAILURE"
        # Error category comes from the result parser's classification of the
        # test execution output (not necessarily the agent's error_category).
        assert record["error_category"] is not None
        assert record["telemetry"] is not None
        assert record["telemetry"]["latency"]["total_s"] > 0


# ===========================================================================
# Cross-format consistency (VAL-CROSS-01 / VAL-EXPORT-05)
# ===========================================================================


class TestCrossFormatConsistency:
    """JSON and CSV exports of the same harness results are consistent."""

    def test_same_record_count_across_formats(self) -> None:
        """JSON and CSV export produce the same number of records."""
        results = [_make_harness_result_dict(task_id=f"consist-{i}") for i in range(8)]

        json_output = export_json(results)
        csv_output = export_csv(results)

        json_data = json.loads(json_output)
        json_count = len(json_data["records"])

        csv_lines = csv_output.strip().split("\n")
        csv_count = len(csv_lines) - 1  # minus header

        assert json_count == csv_count == 8

    def test_same_task_ids_across_formats(self) -> None:
        """Task IDs are consistent between JSON and CSV exports."""
        results = [_make_harness_result_dict(task_id=f"id-match-{i}") for i in range(4)]

        json_output = export_json(results)
        csv_output = export_csv(results)

        json_data = json.loads(json_output)
        json_ids = [r["task_id"] for r in json_data["records"]]

        csv_lines = csv_output.strip().split("\n")
        header = csv_lines[0].split(",")
        csv_ids = [line.split(",")[header.index("task_id")] for line in csv_lines[1:]]

        assert json_ids == csv_ids

    def test_same_scores_across_formats(self) -> None:
        """Scores are consistent between JSON and CSV exports."""
        results = [
            _make_harness_result_dict(
                task_id=f"score-match-{i}",
                score=1.0 if i % 2 == 0 else 0.5,
            )
            for i in range(6)
        ]

        json_output = export_json(results)
        csv_output = export_csv(results)

        json_data = json.loads(json_output)
        json_scores = [r["score"] for r in json_data["records"]]

        csv_lines = csv_output.strip().split("\n")
        header = csv_lines[0].split(",")
        csv_scores = [float(line.split(",")[header.index("score")]) for line in csv_lines[1:]]

        assert json_scores == csv_scores

    def test_empty_harness_results_consistent(self) -> None:
        """Empty results produce consistent output across formats."""
        json_output = export_json([])
        csv_output = export_csv([])

        json_data = json.loads(json_output)
        assert json_data["records"] == []

        csv_lines = csv_output.strip().split("\n")
        assert len(csv_lines) >= 1  # header still present


# ===========================================================================
# VAL-CROSS-01: Full pipeline traceability
# ===========================================================================


class TestPipelineTraceability:
    """Verify task ID is preserved from harness through to export."""

    def test_task_id_preserved_through_full_pipeline(
        self,
    ) -> None:
        """Task ID from task definition → harness result → export record is identical."""
        from grist_mill.harness import Harness

        original_id = "pipeline-trace-abc123"
        task = Task(
            id=original_id,
            prompt="Implement feature X",
            language="python",
            test_command="pytest tests/test_x.py",
            timeout=45,
        )
        config = HarnessConfig(
            agent=AgentConfig(model="claude-3-opus", provider="anthropic"),
            environment=EnvironmentConfig(runner_type="local"),
        )

        env = _MockEnvironment()
        agent = _MockAgent()
        collector = TelemetryCollector()
        collector.record_tokens(prompt=500, completion=200)
        collector.set_estimated_cost(0.035)

        harness = Harness(config=config, trace_enabled=True)
        harness_result = harness.run(
            task=task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Verify ID preserved in harness result
        assert harness_result.task_id == original_id

        # Build export dict
        result_dict = _make_harness_result_dict(
            task_id=harness_result.task_id,
            status=harness_result.status,
            score=harness_result.score,
            error_category=harness_result.error_category,
            model=config.agent.model,
            provider=config.agent.provider,
            telemetry=harness_result.telemetry,
        )

        # Export to JSON
        json_output = export_json([result_dict])
        data = json.loads(json_output)
        record = data["records"][0]

        # Verify ID preserved through export
        assert record["task_id"] == original_id
        assert record["model"] == "claude-3-opus"
        assert record["provider"] == "anthropic"
        assert record["telemetry"]["tokens"]["total"] == 700
        assert record["telemetry"]["estimated_cost_usd"] == 0.035
        assert len(record["telemetry"]["raw_events"]) > 0

    def test_status_preserved_through_full_pipeline(
        self,
    ) -> None:
        """All TaskStatus values round-trip through the full pipeline."""
        statuses = [
            (TaskStatus.SUCCESS, 1.0, None),
            (TaskStatus.FAILURE, 0.0, ErrorCategory.TEST_FAILURE),
            (TaskStatus.ERROR, 0.0, ErrorCategory.SYNTAX_ERROR),
            (TaskStatus.TIMEOUT, 0.0, None),
            (TaskStatus.SKIPPED, 0.0, None),
        ]

        for status, score, error_cat in statuses:
            result = _make_harness_result_dict(
                task_id=f"status-pipe-{status.value}",
                status=status,
                score=score,
                error_category=error_cat,
            )

            json_output = export_json([result])
            data = json.loads(json_output)
            record = data["records"][0]

            assert record["status"] == status.value
            assert record["score"] == score
            if error_cat is not None:
                assert record["error_category"] == error_cat.value
            else:
                assert record["error_category"] is None
