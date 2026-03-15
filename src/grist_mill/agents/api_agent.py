"""API-backed agent implementation with multi-turn conversation support.

Provides:
- APIAgent: The main agent that manages multi-turn LLM conversations
  with tool-use support, provider-agnostic routing, max-turns and timeout
  enforcement, full transcript capture, and per-turn token counting.

The agent is provider-agnostic: it delegates to a ``BaseProvider`` subclass
for LLM communication, and routing to different providers is controlled by
injecting a different provider instance.

Validates:
- VAL-AGENT-01: Multi-turn conversation with tool use
- VAL-AGENT-02: Provider-agnostic routing via config
- VAL-AGENT-03: Max-turns limit → ERROR with MAX_TURNS_EXCEEDED
- VAL-AGENT-04: Timeout enforcement → TIMEOUT with partial output
- VAL-AGENT-06: Full conversation transcript
- VAL-AGENT-07: Per-turn token counting
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from grist_mill.agents.conversation import Conversation
from grist_mill.agents.provider import BaseProvider, ProviderMessage
from grist_mill.interfaces import BaseAgent
from grist_mill.schemas import (
    ErrorCategory,
    HarnessConfig,
    Task,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

# Type for optional tool handler callback
ToolHandler = Callable[..., str]


class APIAgent(BaseAgent):
    """API-backed agent with multi-turn conversation and tool-use support.

    Manages a conversation loop where:

    1. The task prompt is sent as the initial user message
    2. The LLM responds — possibly requesting tool calls
    3. Tool calls are dispatched via the ``tool_handler`` callback
    4. Tool results are fed back to the LLM
    5. The loop continues until the LLM produces a final response (no tool calls),
       the max-turns limit is exceeded, or the wall-clock timeout is reached.

    The agent is provider-agnostic: inject any ``BaseProvider`` subclass to
    route to OpenRouter, OpenAI, Anthropic, etc.

    Args:
        provider: The LLM provider to use for completions.
        max_turns: Maximum number of LLM response turns (tool-call/response cycles).
        timeout: Wall-clock timeout in seconds for the entire conversation.
        tool_handler: Optional callback for dispatching tool calls.
            Signature: ``tool_handler(*, tool_name: str, arguments: dict) -> str``
            If not provided, tool calls are logged and a placeholder result is returned.
    """

    def __init__(
        self,
        *,
        provider: BaseProvider,
        max_turns: int = 10,
        timeout: int = 60,
        tool_handler: ToolHandler | None = None,
    ) -> None:
        self._provider = provider
        self._max_turns = max_turns
        self._timeout = timeout
        self._tool_handler = tool_handler

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_turns(self) -> int:
        """Maximum number of LLM response turns."""
        return self._max_turns

    @property
    def timeout(self) -> int:
        """Wall-clock timeout in seconds."""
        return self._timeout

    @property
    def provider(self) -> BaseProvider:
        """The LLM provider."""
        return self._provider

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, task: Task, config: HarnessConfig) -> TaskResult:
        """Execute a task with multi-turn conversation and tool-use support.

        Manages the full conversation loop:

        1. Initialize conversation with optional system prompt and task prompt
        2. Loop: send messages to LLM, process response
        3. If tool calls requested: dispatch tools, feed results, continue
        4. If final answer: build and return TaskResult
        5. If max-turns exceeded: return ERROR with MAX_TURNS_EXCEEDED
        6. If timeout exceeded: return TIMEOUT with partial output

        Args:
            task: The task to execute.
            config: Harness configuration (provider, system prompt, etc.).

        Returns:
            A ``TaskResult`` with status, telemetry, and full transcript.
        """
        collector = TelemetryCollector()
        conversation = Conversation()
        # Per-turn token tracking: maps assistant message index → usage dict
        turn_token_usage: list[dict[str, int]] = []
        start_time = time.monotonic()

        try:
            # Initialize conversation
            if config.agent.system_prompt:
                conversation.add_system(config.agent.system_prompt)
            conversation.add_user(task.prompt)

            # Conversation loop
            while conversation.turn_count < self._max_turns:
                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= self._timeout:
                    logger.warning(
                        "Agent timeout after %.1fs for task '%s'",
                        elapsed,
                        task.id,
                    )
                    return _build_result(
                        task_id=task.id,
                        status=TaskStatus.TIMEOUT,
                        score=0.0,
                        conversation=conversation,
                        collector=collector,
                        turn_token_usage=turn_token_usage,
                        elapsed_s=time.monotonic() - start_time,
                    )

                # Send messages to provider
                provider_messages = _to_provider_messages(conversation)
                try:
                    response = self._provider.complete(provider_messages)
                except StopIteration:
                    # MockProvider exhausted
                    logger.warning(
                        "Provider exhausted responses for task '%s'",
                        task.id,
                    )
                    return _build_result(
                        task_id=task.id,
                        status=TaskStatus.ERROR,
                        score=0.0,
                        error_category=ErrorCategory.API_ERROR,
                        conversation=conversation,
                        collector=collector,
                        turn_token_usage=turn_token_usage,
                        elapsed_s=time.monotonic() - start_time,
                    )
                except Exception as exc:
                    logger.error(
                        "Provider error for task '%s': %s",
                        task.id,
                        exc,
                    )
                    return _build_result(
                        task_id=task.id,
                        status=TaskStatus.ERROR,
                        score=0.0,
                        error_category=ErrorCategory.API_ERROR,
                        conversation=conversation,
                        collector=collector,
                        turn_token_usage=turn_token_usage,
                        elapsed_s=time.monotonic() - start_time,
                    )

                # Record token usage for this turn
                prompt_tokens = response.usage.get("prompt_tokens", 0)
                completion_tokens = response.usage.get("completion_tokens", 0)
                collector.record_tokens(prompt=prompt_tokens, completion=completion_tokens)

                # Track per-turn token usage (one entry per assistant response)
                turn_token_usage.append(
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                )

                # Add assistant message to conversation
                tool_calls_dicts = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ]
                conversation.add_assistant(
                    content=response.content,
                    tool_calls=tool_calls_dicts if tool_calls_dicts else None,
                )

                # If no tool calls, this is the final response
                if not response.tool_calls:
                    return _build_result(
                        task_id=task.id,
                        status=TaskStatus.SUCCESS,
                        score=1.0,
                        conversation=conversation,
                        collector=collector,
                        turn_token_usage=turn_token_usage,
                        elapsed_s=time.monotonic() - start_time,
                    )

                # Dispatch tool calls and feed results back
                for tc in response.tool_calls:
                    tool_result = self._dispatch_tool(tc.name, tc.arguments)
                    conversation.add_tool_result(tool_call_id=tc.id, content=tool_result)

                    # Record tool call in telemetry
                    collector.record_tool_call(
                        tool_name=tc.name,
                        success=True,
                        duration_ms=0.0,
                    )

            # Max turns exceeded
            logger.warning(
                "Max turns (%d) exceeded for task '%s'",
                self._max_turns,
                task.id,
            )
            return _build_result(
                task_id=task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.MAX_TURNS_EXCEEDED,
                conversation=conversation,
                collector=collector,
                turn_token_usage=turn_token_usage,
                elapsed_s=time.monotonic() - start_time,
            )

        except Exception as exc:
            logger.error(
                "Unexpected error in agent for task '%s': %s",
                task.id,
                exc,
            )
            return _build_result(
                task_id=task.id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.UNKNOWN,
                conversation=conversation,
                collector=collector,
                turn_token_usage=turn_token_usage,
                elapsed_s=time.monotonic() - start_time,
            )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call and return the result.

        If a ``tool_handler`` callback is configured, it is called with the
        tool name and arguments. If not, a placeholder result is returned.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Arguments for the tool invocation.

        Returns:
            The tool result as a string.
        """
        if self._tool_handler is not None:
            try:
                return self._tool_handler(tool_name=tool_name, arguments=arguments)
            except Exception as exc:
                logger.error("Tool '%s' failed: %s", tool_name, exc)
                return f"Tool execution failed: {exc!s}"
        else:
            logger.warning("No tool handler configured, returning placeholder for '%s'", tool_name)
            return f"[Tool '{tool_name}' executed with arguments: {arguments}]"

    def __repr__(self) -> str:
        return (
            f"APIAgent(max_turns={self._max_turns}, "
            f"timeout={self._timeout}, "
            f"provider={type(self._provider).__name__})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_provider_messages(conversation: Conversation) -> list[ProviderMessage]:
    """Convert a Conversation to a list of ProviderMessages."""
    result: list[ProviderMessage] = []
    for msg in conversation.messages:
        if hasattr(msg, "tool_call_id"):
            # ToolResultMessage
            result.append(
                ProviderMessage(
                    role=msg.role,
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,  # type: ignore[attr-defined]
                )
            )
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            # AssistantMessage with tool calls
            result.append(
                ProviderMessage(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=msg.tool_calls,  # type: ignore[attr-defined]
                )
            )
        else:
            result.append(ProviderMessage(role=msg.role, content=msg.content))
    return result


def _build_result(
    *,
    task_id: str,
    status: TaskStatus,
    score: float,
    conversation: Conversation,
    collector: TelemetryCollector,
    turn_token_usage: list[dict[str, int]],
    elapsed_s: float,
    error_category: ErrorCategory | None = None,
) -> TaskResult:
    """Build a TaskResult with transcript, telemetry, and token annotations."""
    transcript = _build_transcript(conversation, turn_token_usage)
    telemetry = collector.build()
    # Record execution latency in telemetry
    telemetry.latency.execution_s = elapsed_s
    telemetry.latency.total_s = (
        telemetry.latency.setup_s + telemetry.latency.execution_s + telemetry.latency.teardown_s
    )

    return TaskResult(
        task_id=task_id,
        status=status,
        score=score,
        error_category=error_category,
        telemetry=telemetry,
        transcript=transcript,
    )


def _build_transcript(
    conversation: Conversation,
    turn_token_usage: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Build the transcript with per-turn token annotations.

    Walks through all messages in the conversation. For each assistant
    message, looks up the corresponding token usage from ``turn_token_usage``
    (indexed by assistant turn number) and attaches it as ``token_usage``.

    Args:
        conversation: The conversation to serialize.
        turn_token_usage: Per-turn token usage, one entry per assistant response.

    Returns:
        A list of message dicts with optional ``token_usage`` on assistant messages.
    """
    transcript: list[dict[str, Any]] = []
    assistant_index = 0

    for msg in conversation.messages:
        d = msg.to_dict()
        if msg.role == "assistant" and assistant_index < len(turn_token_usage):
            d["token_usage"] = turn_token_usage[assistant_index]
            assistant_index += 1
        transcript.append(d)

    return transcript


__all__ = [
    "APIAgent",
]
