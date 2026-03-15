"""Cross-agent telemetry tests (m3-cross-agent-telemetry).

Verifies VAL-CROSS-04: A developer registers a custom agent (M3), runs it in
evaluation (M2), and its telemetry (tool invocations, latency, tokens) appears
in reports (M6) without schema errors.

Evidence: evaluation output and generated report containing the custom agent's
telemetry.

Test strategy:
- Register a custom agent via the AgentRegistry
- Run it through the evaluation harness with mock provider that issues tool calls
- Verify tool invocation telemetry appears in the TaskResult
- Verify telemetry data renders correctly in JSON, CSV, and HTML export
- Verify telemetry is consumable by the aggregation and reporting modules
- Verify the telemetry schema round-trips through JSON serialization without errors
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from grist_mill.agents.api_agent import APIAgent
from grist_mill.agents.provider import (
    MockProvider,
    ProviderResponse,
    ProviderToolCall,
)
from grist_mill.agents.registry import AgentRegistry
from grist_mill.export.formats import export_csv, export_html, export_json
from grist_mill.harness.harness import Harness
from grist_mill.reports.aggregation import aggregate_telemetry
from grist_mill.reports.tools import tool_performance_breakdown
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
from grist_mill.schemas.telemetry import (
    TelemetrySchema,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "cross-telem-001",
    prompt: str = "Refactor the parser to use streaming AST.",
    language: str = "python",
    test_command: str = "pytest tests/ -q",
    timeout: int = 60,
) -> Task:
    """Create a test task for cross-agent telemetry tests."""
    return Task(
        id=task_id,
        prompt=prompt,
        language=language,
        test_command=test_command,
        timeout=timeout,
        difficulty=Difficulty.MEDIUM,
    )


def _make_config(
    model: str = "gpt-4",
    provider: str = "openrouter",
    system_prompt: str | None = None,
) -> HarnessConfig:
    """Create a HarnessConfig for testing."""
    return HarnessConfig(
        agent=AgentConfig(
            model=model,
            provider=provider,
            system_prompt=system_prompt,
        ),
        environment=EnvironmentConfig(runner_type="local"),
    )


def _make_mock_provider_with_tool_calls(
    *,
    tool_names: list[str] | None = None,
    final_content: str = "I have completed the refactoring.",
    prompt_tokens: int = 150,
    completion_tokens: int = 75,
) -> MockProvider:
    """Create a MockProvider that issues tool calls then gives a final answer.

    The provider alternates: first it returns tool calls, then a final answer.
    If tool_names is provided, each tool call uses the next name. Otherwise
    a default set is used.
    """
    names = tool_names or ["file_read", "search", "code_edit", "file_read"]

    responses: list[ProviderResponse] = []

    # First response: two tool calls
    responses.append(
        ProviderResponse(
            content="Let me examine the code.",
            tool_calls=[
                ProviderToolCall(id="tc-1", name=names[0], arguments={"path": "parser.py"}),
                ProviderToolCall(id="tc-2", name=names[1], arguments={"query": "streaming"}),
            ],
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )
    )

    # Second response: one more tool call
    if len(names) >= 3:
        responses.append(
            ProviderResponse(
                content="Now I'll apply the changes.",
                tool_calls=[
                    ProviderToolCall(
                        id="tc-3",
                        name=names[2],
                        arguments={"file": "parser.py", "change": "use streaming AST"},
                    ),
                ],
                usage={
                    "prompt_tokens": prompt_tokens + 50,
                    "completion_tokens": completion_tokens + 25,
                },
            )
        )

    # Final response: done
    responses.append(
        ProviderResponse(
            content=final_content,
            tool_calls=[],
            usage={
                "prompt_tokens": prompt_tokens + 100,
                "completion_tokens": completion_tokens + 50,
            },
        )
    )

    return MockProvider(responses=responses)


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


def _result_to_export_dict(
    result: TaskResult,
    *,
    model: str = "gpt-4",
    provider: str = "openrouter",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Convert a TaskResult to the flat dict format expected by the export pipeline."""
    return {
        "task_id": result.task_id,
        "status": result.status,
        "score": result.score,
        "error_category": result.error_category,
        "model": model,
        "provider": provider,
        "timestamp": timestamp or datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        "telemetry": result.telemetry,
    }


# ===========================================================================
# 1. Custom agent registered via config runs in evaluation harness
# ===========================================================================


class TestCustomAgentRegistrationAndExecution:
    """Verify that a custom agent registered through the AgentRegistry
    can be selected by name in config and runs through the evaluation harness."""

    def test_custom_agent_registered_and_created(self) -> None:
        """A custom agent registered via AgentRegistry can be created by name."""
        registry = AgentRegistry()
        config = _make_config()

        # Verify the default 'api' agent is pre-registered
        assert registry.has("api")
        agent = registry.create("api", config)
        assert isinstance(agent, APIAgent)

    def test_custom_agent_factory_registered(self) -> None:
        """A custom agent factory can be registered and creates the right type."""

        def custom_factory(harness_config: HarnessConfig) -> APIAgent:
            return APIAgent(
                provider=MockProvider(
                    responses=[
                        ProviderResponse(
                            content="Done",
                            tool_calls=[],
                            usage={"prompt_tokens": 10, "completion_tokens": 5},
                        ),
                    ]
                ),
                max_turns=5,
                timeout=30,
            )

        registry = AgentRegistry()
        registry.register("my-custom-agent", custom_factory)

        assert registry.has("my-custom-agent")
        assert "my-custom-agent" in registry.list_agents()

        config = _make_config()
        agent = registry.create("my-custom-agent", config)
        assert isinstance(agent, APIAgent)
        assert agent.max_turns == 5
        assert agent.timeout == 30

    def test_custom_agent_runs_in_harness(self) -> None:
        """A custom agent runs through the evaluation harness end-to-end."""
        task = _make_task()
        config = _make_config()

        # Create agent directly with mock provider
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Task completed.",
                    tool_calls=[],
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=5, timeout=30)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.task_id == task.id
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.telemetry is not None

    def test_duplicate_agent_registration_rejected(self) -> None:
        """Registering a duplicate agent name without overwrite=True raises ValueError."""
        registry = AgentRegistry()

        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                "api",
                lambda c: APIAgent(
                    provider=MockProvider(
                        responses=[
                            ProviderResponse(
                                content="Ok",
                                tool_calls=[],
                                usage={"prompt_tokens": 10, "completion_tokens": 5},
                            ),
                        ]
                    ),
                    max_turns=1,
                    timeout=1,
                ),
            )

    def test_duplicate_agent_registration_with_overwrite(self) -> None:
        """Registering a duplicate agent with overwrite=True succeeds."""

        def new_factory(harness_config: HarnessConfig) -> APIAgent:
            return APIAgent(
                provider=MockProvider(
                    responses=[
                        ProviderResponse(
                            content="Ok",
                            tool_calls=[],
                            usage={"prompt_tokens": 10, "completion_tokens": 5},
                        ),
                    ]
                ),
                max_turns=1,
                timeout=1,
            )

        registry = AgentRegistry()
        registry.register("api", new_factory, overwrite=True)

        assert registry.has("api")
        config = _make_config()
        agent = registry.create("api", config)
        assert agent.max_turns == 1

    def test_unknown_agent_lookup_raises(self) -> None:
        """Looking up an unregistered agent raises KeyError."""
        registry = AgentRegistry()
        config = _make_config()

        with pytest.raises(KeyError, match="not_a_real_agent"):
            registry.create("not_a_real_agent", config)

    def test_registry_lists_all_agents(self) -> None:
        """list_agents returns all registered agent names."""
        registry = AgentRegistry()

        # Should have the default 'api' agent
        assert "api" in registry.list_agents()

        # Register additional agents
        registry.register(
            "agent-b",
            lambda c: APIAgent(
                provider=MockProvider(
                    responses=[
                        ProviderResponse(
                            content="B",
                            tool_calls=[],
                            usage={"prompt_tokens": 5, "completion_tokens": 3},
                        ),
                    ]
                ),
                max_turns=5,
                timeout=30,
            ),
        )
        registry.register(
            "agent-c",
            lambda c: APIAgent(
                provider=MockProvider(
                    responses=[
                        ProviderResponse(
                            content="C",
                            tool_calls=[],
                            usage={"prompt_tokens": 5, "completion_tokens": 3},
                        ),
                    ]
                ),
                max_turns=5,
                timeout=30,
            ),
        )

        agents = registry.list_agents()
        assert "api" in agents
        assert "agent-b" in agents
        assert "agent-c" in agents
        assert len(agents) == 3


# ===========================================================================
# 2. Tool invocation telemetry appears in TaskResult
# ===========================================================================


class TestToolInvocationTelemetryInResult:
    """Verify that tool invocations by the agent produce telemetry in the
    TaskResult, including tool call counts, per-tool breakdown, token usage,
    and latency."""

    def test_tool_calls_produce_telemetry(self) -> None:
        """Agent tool calls produce tool call metrics in the TaskResult."""
        task = _make_task()
        config = _make_config()

        provider = _make_mock_provider_with_tool_calls(
            tool_names=["file_read", "search", "code_edit"],
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        # The agent ran through multi-turn with tool calls
        assert result.task_id == task.id
        assert result.status == TaskStatus.SUCCESS

        # Telemetry must be non-null
        assert result.telemetry is not None
        telem = result.telemetry
        assert isinstance(telem, TelemetrySchema)

        # Tool call metrics should reflect the provider's tool calls
        assert telem.tool_calls.total_calls > 0
        assert telem.tool_calls.by_tool is not None

    def test_token_usage_in_telemetry(self) -> None:
        """Token usage from agent turns is recorded in telemetry."""
        task = _make_task()
        config = _make_config()

        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="First turn.",
                    tool_calls=[
                        ProviderToolCall(id="tc-1", name="read", arguments={}),
                    ],
                    usage={"prompt_tokens": 200, "completion_tokens": 100},
                ),
                ProviderResponse(
                    content="Done.",
                    tool_calls=[],
                    usage={"prompt_tokens": 350, "completion_tokens": 150},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        # Agent records prompt + completion tokens across turns
        assert result.telemetry.tokens.prompt == 200 + 350
        assert result.telemetry.tokens.completion == 100 + 150
        assert result.telemetry.tokens.total == 200 + 100 + 350 + 150

    def test_latency_in_telemetry(self) -> None:
        """Latency breakdown is recorded in telemetry after harness execution."""
        task = _make_task()
        config = _make_config()

        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Done.",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 25},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=5, timeout=30)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        latency = result.telemetry.latency
        # Harness records setup + execution + teardown phases
        assert latency.setup_s >= 0.0
        assert latency.execution_s >= 0.0
        assert latency.teardown_s >= 0.0
        # Total should equal the sum of phases (within floating-point tolerance)
        assert latency.total_s == pytest.approx(
            latency.setup_s + latency.execution_s + latency.teardown_s, abs=0.01
        )

    def test_agent_transcript_includes_tool_calls(self) -> None:
        """The agent's transcript includes tool call and result messages."""
        task = _make_task()
        config = _make_config()

        provider = _make_mock_provider_with_tool_calls(
            tool_names=["grep", "edit"],
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        env = _MockEnvironment()

        harness = Harness(config=config, trace_enabled=True)
        result = harness.run(task=task, agent=agent, env=env)

        # Transcript should be present
        assert result.transcript is not None
        assert len(result.transcript) > 0

        # Should contain at least one assistant message with tool_calls
        has_tool_call = any(
            msg.get("role") == "assistant" and msg.get("tool_calls") for msg in result.transcript
        )
        assert has_tool_call, "Transcript should contain at least one tool call"

    def test_per_tool_breakdown_recorded(self) -> None:
        """Per-tool call breakdown is recorded in telemetry."""
        task = _make_task()
        config = _make_config()

        provider = _make_mock_provider_with_tool_calls(
            tool_names=["file_read", "grep_search", "code_edit"],
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        tc = result.telemetry.tool_calls
        assert tc.total_calls > 0
        # by_tool should contain the tools the agent called
        assert len(tc.by_tool) > 0

    def test_telemetry_version_field_present(self) -> None:
        """Telemetry schema includes the version field."""
        task = _make_task()
        config = _make_config()

        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Done.",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=5, timeout=30)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        assert result.telemetry.version == "V1"

    def test_telemetry_on_error_result(self) -> None:
        """Telemetry is attached even when the agent returns an error."""
        task = _make_task()
        config = _make_config()

        # Provider that always returns tool calls (never terminates cleanly)
        # Provide enough responses for max_turns iterations
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Working...",
                    tool_calls=[
                        ProviderToolCall(id=f"tc-{i}", name="tool_a", arguments={}),
                    ],
                    usage={"prompt_tokens": 50, "completion_tokens": 25},
                )
                for i in range(3)  # enough for max_turns=2 with retries
            ]
        )
        agent = APIAgent(provider=provider, max_turns=2, timeout=60)
        env = _MockEnvironment()

        # Disable retries so the first attempt's result is used directly
        harness = Harness(config=config, max_retries=0)
        result = harness.run(task=task, agent=agent, env=env)

        # Max turns exceeded after exhausting the mock
        assert result.telemetry is not None
        assert result.telemetry.tokens.prompt > 0
        assert result.telemetry.latency.total_s > 0


# ===========================================================================
# 3. Telemetry data renders correctly in report export
# ===========================================================================


class TestTelemetryInReportExport:
    """Verify that telemetry data from a custom agent run renders correctly
    in JSON, CSV, and HTML export formats without schema errors."""

    @pytest.fixture()
    def agent_result(self) -> TaskResult:
        """Run a custom agent through the harness and return the result."""
        task = _make_task(task_id="export-telem-001")
        config = _make_config()

        provider = _make_mock_provider_with_tool_calls(
            tool_names=["file_read", "grep", "edit", "file_read"],
            final_content="Refactoring complete.",
            prompt_tokens=200,
            completion_tokens=100,
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        env = _MockEnvironment()

        harness = Harness(config=config, trace_enabled=True)
        return harness.run(task=task, agent=agent, env=env)

    def test_telemetry_in_json_export(self, agent_result: TaskResult) -> None:
        """Telemetry data from the agent is present and correct in JSON export."""
        result_dict = _result_to_export_dict(
            agent_result,
            model="gpt-4",
            provider="openrouter",
        )

        json_output = export_json([result_dict])
        data = json.loads(json_output)

        assert len(data["records"]) == 1
        record = data["records"][0]

        # Verify schema metadata
        assert "schema_version" in data
        assert "generated_at" in data

        # Verify record fields
        assert record["task_id"] == "export-telem-001"
        assert record["status"] == "SUCCESS"
        assert record["score"] == 1.0

        # Verify telemetry is present and well-formed
        assert "telemetry" in record
        telemetry = record["telemetry"]
        assert telemetry["version"] == "V1"
        assert telemetry["tokens"]["prompt"] > 0
        assert telemetry["tokens"]["completion"] > 0
        assert telemetry["tokens"]["total"] > 0
        assert telemetry["latency"]["total_s"] > 0
        assert telemetry["tool_calls"]["total_calls"] > 0
        assert telemetry["tool_calls"]["by_tool"] is not None

        # JSON round-trip: the record must be valid JSON
        round_trip = json.dumps(record)
        parsed_back = json.loads(round_trip)
        assert parsed_back["task_id"] == "export-telem-001"
        assert parsed_back["telemetry"]["version"] == "V1"

    def test_telemetry_in_csv_export(self, agent_result: TaskResult) -> None:
        """Telemetry data from the agent is present and correct in CSV export."""
        result_dict = _result_to_export_dict(
            agent_result,
            model="gpt-4",
            provider="openrouter",
        )

        csv_output = export_csv([result_dict])
        lines = csv_output.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row

        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}

        assert header_map["task_id"] == "export-telem-001"
        assert header_map["status"] == "SUCCESS"
        assert header_map["score"] == "1.0"
        assert header_map["tokens_prompt"] != "0"
        assert header_map["tokens_completion"] != "0"
        assert header_map["tokens_total"] != "0"
        assert float(header_map["latency_total_s"]) > 0
        assert int(header_map["tool_calls_total"]) > 0

    def test_telemetry_in_html_export(self, agent_result: TaskResult) -> None:
        """Telemetry data from the agent renders in HTML export."""
        result_dict = _result_to_export_dict(
            agent_result,
            model="gpt-4",
            provider="openrouter",
        )

        html_output = export_html([result_dict])

        # HTML should be self-contained
        assert "<!DOCTYPE html>" in html_output
        assert "grist-mill Report" in html_output
        assert "export-telem-001" in html_output
        assert "schema_version" in html_output
        assert "SUCCESS" in html_output

    def test_telemetry_schema_no_errors_in_json_serialization(
        self, agent_result: TaskResult
    ) -> None:
        """TelemetrySchema from the agent serializes to JSON without errors."""
        assert agent_result.telemetry is not None

        # Direct model_dump should work
        telem_dict = agent_result.telemetry.model_dump(mode="json")

        # Must have all expected fields
        assert "version" in telem_dict
        assert "tokens" in telem_dict
        assert "latency" in telem_dict
        assert "tool_calls" in telem_dict
        assert "raw_events" in telem_dict

        # JSON serialization must succeed
        json_str = json.dumps(telem_dict)
        parsed = json.loads(json_str)
        assert parsed["version"] == "V1"
        assert parsed["tokens"]["prompt"] == agent_result.telemetry.tokens.prompt

    def test_multiple_agent_results_export_together(self) -> None:
        """Multiple results from different agent configurations export together."""
        results: list[dict[str, Any]] = []

        for i in range(3):
            task = _make_task(task_id=f"multi-export-{i}")
            config = _make_config(model=f"model-{i}", provider=f"provider-{i}")

            provider = MockProvider(
                responses=[
                    ProviderResponse(
                        content=f"Agent {i} done.",
                        tool_calls=[
                            ProviderToolCall(
                                id=f"tc-{i}-1",
                                name=f"tool_{i}_a",
                                arguments={"idx": i},
                            ),
                        ],
                        usage={
                            "prompt_tokens": 100 + i * 50,
                            "completion_tokens": 50 + i * 25,
                        },
                    ),
                    ProviderResponse(
                        content=f"Final answer from agent {i}.",
                        tool_calls=[],
                        usage={
                            "prompt_tokens": 200 + i * 50,
                            "completion_tokens": 100 + i * 25,
                        },
                    ),
                ]
            )
            agent = APIAgent(provider=provider, max_turns=10, timeout=60)
            env = _MockEnvironment()

            harness = Harness(config=config)
            result = harness.run(task=task, agent=agent, env=env)

            results.append(
                _result_to_export_dict(
                    result,
                    model=f"model-{i}",
                    provider=f"provider-{i}",
                )
            )

        # JSON export
        json_output = export_json(results)
        data = json.loads(json_output)
        assert len(data["records"]) == 3
        assert data["summary"]["total_tasks"] == 3

        # CSV export
        csv_output = export_csv(results)
        csv_lines = csv_output.strip().split("\n")
        assert len(csv_lines) == 4  # header + 3 data rows

        # HTML export
        html_output = export_html(results)
        for i in range(3):
            assert f"multi-export-{i}" in html_output


# ===========================================================================
# 4. Telemetry consumed by reporting/aggregation modules
# ===========================================================================


class TestTelemetryInAggregationAndReporting:
    """Verify that telemetry from a custom agent run is correctly consumed
    by the aggregation and reporting modules."""

    def test_aggregate_telemetry_from_agent_results(self) -> None:
        """aggregate_telemetry correctly processes results from a custom agent."""
        results: list[dict[str, Any]] = []

        for i in range(4):
            task = _make_task(task_id=f"agg-{i}")
            config = _make_config(model=f"model-v{i}")

            provider = MockProvider(
                responses=[
                    ProviderResponse(
                        content="Done.",
                        tool_calls=[],
                        usage={"prompt_tokens": 100 + i * 20, "completion_tokens": 50 + i * 10},
                    ),
                ]
            )
            agent = APIAgent(provider=provider, max_turns=5, timeout=30)
            env = _MockEnvironment()

            harness = Harness(config=config)
            result = harness.run(task=task, agent=agent, env=env)

            results.append(
                _result_to_export_dict(
                    result,
                    model=f"model-v{i}",
                    provider="openrouter",
                )
            )

        summaries = aggregate_telemetry(results, group_by="model")

        assert len(summaries) == 4  # 4 different models
        for s in summaries:
            assert s["total_tasks"] == 1
            assert s["pass_rate"] == 1.0
            assert s["total_tokens"] > 0
            assert s["mean_latency_s"] >= 0.0

    def test_tool_performance_breakdown_from_agent(self) -> None:
        """tool_performance_breakdown processes tool telemetry from agent results."""
        task = _make_task(task_id="tool-breakdown-001")
        config = _make_config()

        provider = _make_mock_provider_with_tool_calls(
            tool_names=["file_read", "grep", "code_edit", "file_read"],
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        result_dict = _result_to_export_dict(result)
        breakdown = tool_performance_breakdown([result_dict])

        # Should have at least one tool with metrics
        assert len(breakdown) > 0

        for tool in breakdown:
            assert "tool_name" in tool
            assert tool["total_calls"] > 0
            assert 0.0 <= tool["success_rate"] <= 1.0
            assert 0.0 <= tool["error_rate"] <= 1.0

    def test_aggregate_multiple_tasks_same_model(self) -> None:
        """Multiple tasks from the same model aggregate into one summary row."""
        results: list[dict[str, Any]] = []

        for i in range(5):
            task = _make_task(task_id=f"same-model-{i}")
            config = _make_config(model="gpt-4")

            provider = MockProvider(
                responses=[
                    ProviderResponse(
                        content="Done.",
                        tool_calls=[],
                        usage={"prompt_tokens": 100, "completion_tokens": 50},
                    ),
                ]
            )
            agent = APIAgent(provider=provider, max_turns=5, timeout=30)
            env = _MockEnvironment()

            harness = Harness(config=config)
            result = harness.run(task=task, agent=agent, env=env)

            results.append(_result_to_export_dict(result, model="gpt-4"))

        summaries = aggregate_telemetry(results, group_by="model")
        assert len(summaries) == 1
        assert summaries[0]["group"] == "gpt-4"
        assert summaries[0]["total_tasks"] == 5
        assert summaries[0]["pass_rate"] == 1.0
        assert summaries[0]["total_tokens"] == 750  # 5 * 150


# ===========================================================================
# 5. End-to-end: custom agent registry → harness → export
# ===========================================================================


class TestEndToEndCustomAgentTelemetry:
    """Full end-to-end: register custom agent, run in harness, verify
    telemetry in export without any schema errors."""

    def test_full_pipeline_custom_agent_to_export(self) -> None:
        """Register a custom agent, run through harness, export to all formats."""
        # 1. Register custom agent via registry
        registry = AgentRegistry()

        def custom_factory(harness_config: HarnessConfig) -> APIAgent:
            return APIAgent(
                provider=_make_mock_provider_with_tool_calls(
                    tool_names=["my_tool_a", "my_tool_b", "my_tool_c"],
                    final_content="Custom agent completed the task.",
                    prompt_tokens=300,
                    completion_tokens=150,
                ),
                max_turns=10,
                timeout=60,
            )

        registry.register("my-agent", custom_factory)

        # 2. Create agent from registry
        config = _make_config(model="custom-model", provider="custom-provider")
        agent = registry.create("my-agent", config)

        # 3. Run through harness
        task = _make_task(task_id="e2e-custom-001")
        env = _MockEnvironment()

        harness = Harness(config=config, trace_enabled=True)
        result = harness.run(task=task, agent=agent, env=env)

        # 4. Verify telemetry is attached
        assert result.task_id == "e2e-custom-001"
        assert result.telemetry is not None
        telem = result.telemetry
        assert telem.version == "V1"
        assert telem.tokens.total > 0
        assert telem.latency.total_s > 0
        assert telem.tool_calls.total_calls > 0

        # 5. Build export dict
        result_dict = _result_to_export_dict(
            result,
            model="custom-model",
            provider="custom-provider",
        )

        # 6. Export to JSON and verify
        json_output = export_json([result_dict])
        data = json.loads(json_output)
        record = data["records"][0]
        assert record["task_id"] == "e2e-custom-001"
        assert record["model"] == "custom-model"
        assert record["provider"] == "custom-provider"
        assert record["telemetry"]["version"] == "V1"
        assert record["telemetry"]["tokens"]["total"] > 0
        assert record["telemetry"]["tool_calls"]["total_calls"] > 0
        assert record["telemetry"]["latency"]["total_s"] > 0

        # 7. Export to CSV and verify
        csv_output = export_csv([result_dict])
        lines = csv_output.strip().split("\n")
        assert len(lines) == 2
        header = lines[0].split(",")
        row = lines[1].split(",")
        header_map = {h.strip(): v.strip() for h, v in zip(header, row, strict=True)}
        assert header_map["task_id"] == "e2e-custom-001"
        assert header_map["model"] == "custom-model"
        assert int(header_map["tokens_total"]) > 0
        assert float(header_map["latency_total_s"]) > 0

        # 8. Export to HTML and verify
        html_output = export_html([result_dict])
        assert "e2e-custom-001" in html_output
        assert "custom-model" in html_output
        assert "SUCCESS" in html_output

        # 9. Verify aggregation works
        agg = aggregate_telemetry([result_dict], group_by="model")
        assert len(agg) == 1
        assert agg[0]["group"] == "custom-model"
        assert agg[0]["total_tasks"] == 1
        assert agg[0]["total_tokens"] > 0

        # 10. Verify tool breakdown works
        breakdown = tool_performance_breakdown([result_dict])
        assert len(breakdown) > 0

    def test_telemetry_forward_compatibility(self) -> None:
        """Telemetry with extra (future) fields doesn't cause schema errors."""
        task = _make_task(task_id="fwd-compat-001")
        config = _make_config()

        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Done.",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 25},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=5, timeout=30)
        env = _MockEnvironment()

        harness = Harness(config=config)
        result = harness.run(task=task, agent=agent, env=env)

        assert result.telemetry is not None
        telem_dict = result.telemetry.model_dump()

        # Add extra fields that a future version might have
        telem_dict["future_field"] = "some_value"
        telem_dict["nested_future"] = {"key": "value"}

        # TelemetrySchema should handle extra fields (extra="ignore")
        rebuilt = TelemetrySchema(**telem_dict)
        assert rebuilt.version == "V1"
        assert rebuilt.tokens.total == 75

        # JSON round-trip
        json_str = json.dumps(telem_dict)
        parsed = json.loads(json_str)
        assert parsed["version"] == "V1"
        assert parsed["future_field"] == "some_value"
