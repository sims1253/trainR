"""Tests for the API-backed agent implementation.

Covers:
- VAL-AGENT-01: Multi-turn conversation with tool use
- VAL-AGENT-02: Provider-agnostic routing
- VAL-AGENT-03: Max-turns limit enforcement
- VAL-AGENT-04: Timeout enforcement
- VAL-AGENT-05: Pluggable agent registry
- VAL-AGENT-06: Full conversation transcript
- VAL-AGENT-07: Per-turn token counting

Uses mock providers to avoid real API calls.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from grist_mill.agents.api_agent import APIAgent
from grist_mill.agents.conversation import (
    AssistantMessage,
    Conversation,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from grist_mill.agents.provider import (
    BaseProvider,
    MockProvider,
    ProviderMessage,
    ProviderResponse,
    ProviderToolCall,
)
from grist_mill.agents.registry import AgentRegistry
from grist_mill.interfaces import BaseAgent
from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ErrorCategory,
    HarnessConfig,
    Task,
    TaskResult,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "test-1",
    prompt: str = "Solve this task",
    language: str = "python",
    timeout: int = 30,
) -> Task:
    """Create a simple test task."""
    return Task(
        id=task_id,
        prompt=prompt,
        language=language,
        test_command="echo done",
        timeout=timeout,
        difficulty=Difficulty.EASY,
    )


def _make_config(
    model: str = "test-model",
    provider: str = "mock",
    system_prompt: str | None = None,
    max_turns: int = 10,
    timeout: int = 60,
) -> HarnessConfig:
    """Create a harness config for testing."""
    return HarnessConfig(
        agent=AgentConfig(
            model=model,
            provider=provider,
            system_prompt=system_prompt,
        ),
        environment=EnvironmentConfig(runner_type="local"),
    )


# ---------------------------------------------------------------------------
# Conversation Model Tests (VAL-AGENT-06)
# ---------------------------------------------------------------------------


class TestConversationModel:
    """Tests for the Conversation data model."""

    def test_conversation_starts_empty(self) -> None:
        """A new conversation has no messages."""
        conv = Conversation()
        assert conv.messages == []
        assert conv.turn_count == 0

    def test_add_user_message(self) -> None:
        """User messages can be added and appear in transcript."""
        conv = Conversation()
        conv.add_user("Solve this task")
        assert len(conv.messages) == 1
        assert isinstance(conv.messages[0], UserMessage)
        assert conv.messages[0].content == "Solve this task"
        assert conv.turn_count == 0  # turn_count tracks LLM exchanges

    def test_add_assistant_message(self) -> None:
        """Assistant messages with and without tool calls."""
        conv = Conversation()
        conv.add_assistant("I will help you")
        assert len(conv.messages) == 1
        assert isinstance(conv.messages[0], AssistantMessage)
        assert conv.messages[0].content == "I will help you"
        assert conv.messages[0].tool_calls == []

    def test_add_assistant_with_tool_calls(self) -> None:
        """Assistant messages can carry tool calls."""
        conv = Conversation()
        conv.add_assistant(
            content="Let me check that",
            tool_calls=[{"id": "tc_1", "name": "read_file", "arguments": {"path": "/tmp/f"}}],
        )
        msg = conv.messages[0]
        # When tool_calls are present, a ToolCallMessage is stored
        assert isinstance(msg, ToolCallMessage)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read_file"

    def test_add_tool_result(self) -> None:
        """Tool results can be added."""
        conv = Conversation()
        conv.add_tool_result(tool_call_id="tc_1", content="file contents here")
        assert len(conv.messages) == 1
        assert isinstance(conv.messages[0], ToolResultMessage)
        assert conv.messages[0].content == "file contents here"
        assert conv.messages[0].tool_call_id == "tc_1"

    def test_full_transcript_preserved_in_order(self) -> None:
        """All messages are preserved in order."""
        conv = Conversation()
        conv.add_user("Task prompt")
        conv.add_assistant("I'll use a tool")
        conv.add_assistant(
            content="",
            tool_calls=[{"id": "tc_1", "name": "bash", "arguments": {"cmd": "ls"}}],
        )
        conv.add_tool_result(tool_call_id="tc_1", content="file1.py\nfile2.py")
        conv.add_assistant("Here are the files.")

        assert len(conv.messages) == 5
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"
        assert conv.messages[2].role == "assistant"
        assert conv.messages[3].role == "tool"
        assert conv.messages[4].role == "assistant"

    def test_to_transcript_list(self) -> None:
        """to_transcript returns list of dicts with all fields."""
        conv = Conversation()
        conv.add_user("Hello")
        conv.add_assistant("Hi there")
        transcript = conv.to_transcript()
        assert len(transcript) == 2
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "Hello"
        assert transcript[1]["role"] == "assistant"
        assert transcript[1]["content"] == "Hi there"

    def test_turn_count_tracks_assistant_messages(self) -> None:
        """turn_count increments for each assistant response."""
        conv = Conversation()
        assert conv.turn_count == 0
        conv.add_assistant("First")
        assert conv.turn_count == 1
        conv.add_assistant("Second")
        assert conv.turn_count == 2

    def test_tool_result_messages_ignored_for_turn_count(self) -> None:
        """Tool result messages don't increment turn count."""
        conv = Conversation()
        conv.add_assistant("First")
        conv.add_tool_result("tc_1", "result")
        assert conv.turn_count == 1
        conv.add_assistant("Second")
        assert conv.turn_count == 2


# ---------------------------------------------------------------------------
# Mock Provider Tests
# ---------------------------------------------------------------------------


class TestMockProvider:
    """Tests for the mock provider used in testing."""

    def test_mock_provider_returns_configured_responses(self) -> None:
        """Mock provider returns responses in order."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="First response",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
                ProviderResponse(
                    content="Second response",
                    tool_calls=[],
                    usage={"prompt_tokens": 15, "completion_tokens": 8},
                ),
            ]
        )
        result = provider.complete(messages=[ProviderMessage(role="user", content="test")])
        assert result.content == "First response"
        assert result.usage["prompt_tokens"] == 10

        result2 = provider.complete(messages=[ProviderMessage(role="user", content="test")])
        assert result2.content == "Second response"

    def test_mock_provider_exhaustion_raises(self) -> None:
        """Mock provider raises StopIteration when responses exhausted."""
        provider = MockProvider(responses=[])
        with pytest.raises(StopIteration):
            provider.complete(messages=[ProviderMessage(role="user", content="test")])

    def test_mock_provider_with_delay(self) -> None:
        """Mock provider can simulate delay."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="delayed",
                    tool_calls=[],
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            ],
            delay=0.1,
        )
        start = time.monotonic()
        provider.complete(messages=[ProviderMessage(role="user", content="test")])
        elapsed = time.monotonic() - start
        assert elapsed >= 0.09  # allow small timing variance


# ---------------------------------------------------------------------------
# API Agent Tests (VAL-AGENT-01 through VAL-AGENT-07)
# ---------------------------------------------------------------------------


class TestAPIAgentMultiTurn:
    """VAL-AGENT-01: Multi-turn conversation with tool use."""

    def test_single_turn_conversation(self) -> None:
        """Agent completes in a single turn when LLM returns final answer."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="The answer is 42.",
                    tool_calls=[],
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.SUCCESS
        assert result.task_id == "test-1"

    def test_multi_turn_with_tool_calls(self) -> None:
        """Agent dispatches tool calls, feeds results, continues conversation."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ProviderToolCall(
                            id="tc_1",
                            name="read_file",
                            arguments={"path": "/tmp/main.py"},
                        ),
                    ],
                    usage={"prompt_tokens": 30, "completion_tokens": 15},
                ),
                ProviderResponse(
                    content="I see the code now. Here's the fix: ...",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 25},
                ),
            ]
        )
        tool_handler = MagicMock(return_value="def main(): pass")

        agent = APIAgent(
            provider=provider,
            max_turns=10,
            timeout=60,
            tool_handler=tool_handler,
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.SUCCESS
        assert tool_handler.call_count == 1
        # Tool handler should have been called with the right arguments
        call_args = tool_handler.call_args
        assert call_args[1]["tool_name"] == "read_file"

    def test_multiple_tool_calls_in_single_response(self) -> None:
        """Agent handles multiple tool calls in a single assistant response."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ProviderToolCall(id="tc_1", name="tool_a", arguments={"x": 1}),
                        ProviderToolCall(id="tc_2", name="tool_b", arguments={"y": 2}),
                    ],
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                ),
                ProviderResponse(
                    content="Both tools executed. Final answer.",
                    tool_calls=[],
                    usage={"prompt_tokens": 40, "completion_tokens": 10},
                ),
            ]
        )

        call_log: list[str] = []

        def tool_handler(*, tool_name: str, arguments: dict[str, Any]) -> str:
            call_log.append(tool_name)
            return f"Result from {tool_name}"

        agent = APIAgent(
            provider=provider,
            max_turns=10,
            timeout=60,
            tool_handler=tool_handler,
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.SUCCESS
        assert call_log == ["tool_a", "tool_b"]

    def test_tool_call_failure_continues(self) -> None:
        """Agent continues even when a tool call returns an error."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ProviderToolCall(id="tc_1", name="failing_tool", arguments={}),
                    ],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
                ProviderResponse(
                    content="I see the tool failed. Here's my answer anyway.",
                    tool_calls=[],
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                ),
            ]
        )

        def failing_tool(*, tool_name: str, arguments: dict[str, Any]) -> str:
            raise RuntimeError("Tool execution failed!")

        agent = APIAgent(
            provider=provider,
            max_turns=10,
            timeout=60,
            tool_handler=failing_tool,
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.SUCCESS
        # The error should be passed back to the LLM as tool result
        transcript = result.transcript
        assert transcript is not None
        tool_results = [m for m in transcript if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert "Tool execution failed!" in tool_results[0]["content"]


class TestAPIAgentMaxTurns:
    """VAL-AGENT-03: Max-turns limit enforcement."""

    def test_max_turns_exceeded_returns_error(self) -> None:
        """Agent returns ERROR with MAX_TURNS_EXCEEDED when limit is reached."""
        # Create a provider that always returns tool calls (never terminates)
        responses = [
            ProviderResponse(
                content="",
                tool_calls=[ProviderToolCall(id=f"tc_{i}", name="tool", arguments={})],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
            for i in range(5)
        ]
        provider = MockProvider(responses=responses)
        agent = APIAgent(
            provider=provider,
            max_turns=3,
            timeout=60,
            tool_handler=lambda **kw: "result",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.MAX_TURNS_EXCEEDED

    def test_max_turns_includes_partial_output(self) -> None:
        """When max turns exceeded, result includes whatever partial output was generated."""
        responses = [
            ProviderResponse(
                content="Partial answer so far" if i == 0 else "",
                tool_calls=[ProviderToolCall(id=f"tc_{i}", name="tool", arguments={})],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
            for i in range(5)
        ]
        provider = MockProvider(responses=responses)
        agent = APIAgent(
            provider=provider,
            max_turns=2,
            timeout=60,
            tool_handler=lambda **kw: "result",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.MAX_TURNS_EXCEEDED
        # Transcript should have the partial conversation
        assert result.transcript is not None
        assert len(result.transcript) > 0

    def test_exact_max_turns_completes(self) -> None:
        """Agent completes successfully when it finishes exactly at max_turns."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_0", name="t", arguments={})],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_1", name="t", arguments={})],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                ),
                ProviderResponse(
                    content="Done!",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ]
        )
        agent = APIAgent(
            provider=provider,
            max_turns=3,
            timeout=60,
            tool_handler=lambda **kw: "ok",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.SUCCESS


class TestAPIAgentTimeout:
    """VAL-AGENT-04: Timeout enforcement."""

    def test_timeout_returns_timeout_status(self) -> None:
        """Agent returns TIMEOUT when wall-clock timeout is exceeded."""
        # Provider that takes a long time (> 1s delay per response)
        responses = [
            ProviderResponse(
                content="",
                tool_calls=[ProviderToolCall(id=f"tc_{i}", name="tool", arguments={})],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
            for i in range(100)
        ]
        provider = MockProvider(responses=responses, delay=0.2)
        agent = APIAgent(
            provider=provider,
            max_turns=100,
            timeout=0.5,  # Very short timeout
            tool_handler=lambda **kw: "result",
        )
        task = _make_task(timeout=30)
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.TIMEOUT

    def test_timeout_includes_partial_transcript(self) -> None:
        """When timeout occurs, partial transcript is included."""
        # Provider that always returns tool calls (never terminates)
        responses = [
            ProviderResponse(
                content="Partial",
                tool_calls=[ProviderToolCall(id=f"tc_{i}", name="t", arguments={})],
                usage={"prompt_tokens": 5, "completion_tokens": 3},
            )
            for i in range(10)
        ]
        provider = MockProvider(responses=responses, delay=0.2)
        agent = APIAgent(
            provider=provider,
            max_turns=100,
            timeout=0.5,
            tool_handler=lambda **kw: "result",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.TIMEOUT
        assert result.transcript is not None
        assert len(result.transcript) > 0


class TestAPIAgentTranscript:
    """VAL-AGENT-06: Full conversation transcript."""

    def test_transcript_captures_all_turns(self) -> None:
        """All messages appear in the transcript in order."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_1", name="bash", arguments={"cmd": "ls"})],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
                ProviderResponse(
                    content="Here is my final answer.",
                    tool_calls=[],
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                ),
            ]
        )
        agent = APIAgent(
            provider=provider,
            max_turns=10,
            timeout=60,
            tool_handler=lambda **kw: "file1.py",
        )
        task = _make_task(prompt="List files")
        config = _make_config()

        result = agent.run(task, config)
        assert result.transcript is not None

        # Transcript should have: user, assistant (tool call), tool result, assistant (final)
        roles = [m["role"] for m in result.transcript]
        assert roles[0] == "user"
        assert roles[1] == "assistant"
        assert roles[2] == "tool"
        assert roles[3] == "assistant"

    def test_transcript_content_preserved(self) -> None:
        """Transcript preserves the original content of each message."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Thinking about it...",
                    tool_calls=[],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task(prompt="What is 2+2?")
        config = _make_config()

        result = agent.run(task, config)
        transcript = result.transcript
        assert transcript is not None
        assert transcript[0]["content"] == "What is 2+2?"
        assert transcript[1]["content"] == "Thinking about it..."

    def test_transcript_with_system_prompt(self) -> None:
        """System prompt appears as first message in transcript when provided."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Done",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task()
        config = _make_config(system_prompt="You are a helpful assistant.")

        result = agent.run(task, config)
        transcript = result.transcript
        assert transcript is not None
        assert transcript[0]["role"] == "system"
        assert transcript[0]["content"] == "You are a helpful assistant."


class TestAPIAgentTokenCounting:
    """VAL-AGENT-07: Per-turn token counting."""

    def test_per_turn_token_counts_recorded(self) -> None:
        """Each turn's token usage is recorded in telemetry."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_1", name="tool", arguments={})],
                    usage={"prompt_tokens": 30, "completion_tokens": 15},
                ),
                ProviderResponse(
                    content="Final answer",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 25},
                ),
            ]
        )
        agent = APIAgent(
            provider=provider,
            max_turns=10,
            timeout=60,
            tool_handler=lambda **kw: "ok",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.transcript is not None

        # Check per-turn token usage on assistant messages
        assistant_msgs = [m for m in result.transcript if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 2

        # Each assistant message should have token_usage
        for msg in assistant_msgs:
            assert "token_usage" in msg
            assert "prompt_tokens" in msg["token_usage"]
            assert "completion_tokens" in msg["token_usage"]

    def test_token_counts_sum_to_total(self) -> None:
        """Per-turn token counts sum to total in telemetry."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Step 1",
                    tool_calls=[],
                    usage={"prompt_tokens": 30, "completion_tokens": 15},
                ),
                ProviderResponse(
                    content="Step 2",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 25},
                ),
                ProviderResponse(
                    content="Done",
                    tool_calls=[],
                    usage={"prompt_tokens": 40, "completion_tokens": 20},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.telemetry is not None

        # Sum per-turn tokens from transcript
        total_prompt = 0
        total_completion = 0
        for msg in result.transcript or []:
            if "token_usage" in msg:
                total_prompt += msg["token_usage"]["prompt_tokens"]
                total_completion += msg["token_usage"]["completion_tokens"]

        # These should match telemetry totals
        assert result.telemetry.tokens.prompt == total_prompt
        assert result.telemetry.tokens.completion == total_completion
        assert result.telemetry.tokens.total == total_prompt + total_completion

    def test_zero_tokens_when_not_reported(self) -> None:
        """Telemetry has zeros when provider doesn't report tokens."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="No token info",
                    tool_calls=[],
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.telemetry is not None
        assert result.telemetry.tokens.total >= 0


class TestAPIAgentProviderAgnostic:
    """VAL-AGENT-02: Provider-agnostic routing."""

    def test_same_agent_different_providers(self) -> None:
        """Same agent type works with different provider instances."""
        provider_a = MockProvider(
            responses=[
                ProviderResponse(
                    content="From provider A",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ]
        )
        provider_b = MockProvider(
            responses=[
                ProviderResponse(
                    content="From provider B",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ]
        )

        task = _make_task()

        agent_a = APIAgent(provider=provider_a, max_turns=10, timeout=60)
        config_a = _make_config(provider="provider_a")
        result_a = agent_a.run(task, config_a)
        assert "From provider A" in (result_a.transcript or [{}])[-1].get("content", "")

        agent_b = APIAgent(provider=provider_b, max_turns=10, timeout=60)
        config_b = _make_config(provider="provider_b")
        result_b = agent_b.run(task, config_b)
        assert "From provider B" in (result_b.transcript or [{}])[-1].get("content", "")

    def test_provider_error_handling(self) -> None:
        """Agent handles provider errors gracefully."""
        provider = MagicMock(spec=BaseProvider)
        provider.complete.side_effect = RuntimeError("API connection failed")

        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.API_ERROR


class TestAgentRegistry:
    """VAL-AGENT-05: Pluggable agent registry."""

    def test_register_and_retrieve_agent(self) -> None:
        """Agents can be registered by name and retrieved."""
        registry = AgentRegistry()

        def factory(config: HarnessConfig) -> APIAgent:
            provider = MockProvider(
                responses=[
                    ProviderResponse(
                        content="Custom agent",
                        tool_calls=[],
                        usage={"prompt_tokens": 5, "completion_tokens": 3},
                    ),
                ]
            )
            return APIAgent(provider=provider, max_turns=10, timeout=60)

        registry.register("custom_agent", factory)
        assert registry.has("custom_agent")

        config = _make_config()
        agent = registry.create("custom_agent", config)
        assert isinstance(agent, BaseAgent)
        result = agent.run(_make_task(), config)
        assert result.status == TaskStatus.SUCCESS

    def test_duplicate_registration_fails(self) -> None:
        """Registering a duplicate agent name raises ValueError."""
        registry = AgentRegistry()

        def factory(config):
            return APIAgent(
                provider=MockProvider(responses=[]),
                max_turns=10,
                timeout=60,
            )

        registry.register("my_agent", factory)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("my_agent", factory)

    def test_duplicate_registration_with_overwrite(self) -> None:
        """Registering with overwrite=True replaces existing agent."""
        registry = AgentRegistry()

        def factory1(config):
            return APIAgent(
                provider=MockProvider(responses=[]),
                max_turns=1,
                timeout=60,
            )

        def factory2(config):
            return APIAgent(
                provider=MockProvider(responses=[]),
                max_turns=2,
                timeout=60,
            )

        registry.register("agent", factory1)
        registry.register("agent", factory2, overwrite=True)

        assert registry.has("agent")
        config = _make_config()
        agent = registry.create("agent", config)
        assert agent.max_turns == 2

    def test_list_registered_agents(self) -> None:
        """Registry lists all registered agent names."""
        registry = AgentRegistry()

        def factory(config):
            return APIAgent(
                provider=MockProvider(responses=[]),
                max_turns=10,
                timeout=60,
            )

        registry.register("agent_a", factory)
        registry.register("agent_b", factory)

        names = registry.list_agents()
        # "api" is pre-registered by default
        assert "agent_a" in names
        assert "agent_b" in names
        assert "api" in names

    def test_retrieving_unregistered_agent_raises(self) -> None:
        """Retrieving an unregistered agent raises KeyError."""
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.create("nonexistent", _make_config())

    def test_builtin_api_agent_registered(self) -> None:
        """The builtin 'api' agent is pre-registered."""
        registry = AgentRegistry()
        assert registry.has("api")

    def test_custom_agent_selected_by_config(self) -> None:
        """Custom agents can be selected by name in config."""
        registry = AgentRegistry()

        call_log: list[str] = []

        class CustomAgent(BaseAgent):
            def __init__(self, label: str) -> None:
                self.label = label

            def run(self, task: Task, config: HarnessConfig) -> TaskResult:
                call_log.append(self.label)
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    score=1.0,
                )

        def make_custom(config: HarnessConfig) -> CustomAgent:
            return CustomAgent(label="custom-v1")

        registry.register("custom-v1", make_custom)
        config = _make_config()
        agent = registry.create("custom-v1", config)
        result = agent.run(_make_task(), config)

        assert isinstance(result, TaskResult)
        assert call_log == ["custom-v1"]


class TestAPIAgentIsBaseAgent:
    """API agent implements BaseAgent correctly."""

    def test_api_agent_is_base_agent(self) -> None:
        """APIAgent is a subclass of BaseAgent."""
        agent = APIAgent(
            provider=MockProvider(responses=[]),
            max_turns=10,
            timeout=60,
        )
        assert isinstance(agent, BaseAgent)

    def test_incomplete_subclass_fails(self) -> None:
        """Subclass without run method fails at instantiation."""
        with pytest.raises(TypeError):

            class BadAgent(BaseAgent):
                pass

            BadAgent()


class TestAPIAgentTelemetry:
    """Telemetry is properly attached to results."""

    def test_success_has_telemetry(self) -> None:
        """Successful result includes telemetry."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Done",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ]
        )
        agent = APIAgent(provider=provider, max_turns=10, timeout=60)
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.telemetry is not None
        assert result.telemetry.latency.total_s > 0

    def test_error_has_telemetry(self) -> None:
        """Error result includes telemetry."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_0", name="t", arguments={})],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                ),
            ]
            * 100
        )
        agent = APIAgent(
            provider=provider,
            max_turns=2,
            timeout=60,
            tool_handler=lambda **kw: "r",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.telemetry is not None
        assert result.telemetry.latency.total_s > 0

    def test_timeout_has_telemetry(self) -> None:
        """Timeout result includes telemetry."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_0", name="t", arguments={})],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                ),
            ]
            * 100,
            delay=0.2,
        )
        agent = APIAgent(
            provider=provider,
            max_turns=100,
            timeout=0.5,
            tool_handler=lambda **kw: "r",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.status == TaskStatus.TIMEOUT
        assert result.telemetry is not None
        assert result.telemetry.latency.total_s > 0

    def test_tool_calls_tracked_in_telemetry(self) -> None:
        """Tool invocations are tracked in telemetry."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="",
                    tool_calls=[ProviderToolCall(id="tc_1", name="bash", arguments={"cmd": "ls"})],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
                ProviderResponse(
                    content="Done",
                    tool_calls=[],
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                ),
            ]
        )
        agent = APIAgent(
            provider=provider,
            max_turns=10,
            timeout=60,
            tool_handler=lambda **kw: "file1.py",
        )
        task = _make_task()
        config = _make_config()

        result = agent.run(task, config)
        assert result.telemetry is not None
        assert result.telemetry.tool_calls.total_calls == 1
        assert result.telemetry.tool_calls.successful_calls == 1
