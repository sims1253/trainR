"""Conversation state management for multi-turn agent interactions.

Manages the messages array, tracks turn counts, and provides transcript
serialization. Each message is a typed record preserving role, content,
and optional metadata (tool calls, tool call IDs).

Validates VAL-AGENT-06: Full conversation transcript captured in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Message Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UserMessage:
    """A user message in the conversation."""

    role: str = field(default="user", init=False)
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class SystemMessage:
    """A system message in the conversation (e.g., system prompt)."""

    role: str = field(default="system", init=False)
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """An assistant message in the conversation.

    May carry optional tool calls that the assistant wants to dispatch.
    """

    role: str = field(default="assistant", init=False)
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass(frozen=True, slots=True)
class ToolCallMessage:
    """An assistant message that consists solely of tool calls (no text content)."""

    role: str = field(default="assistant", init=False)
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "tool_calls": self.tool_calls}


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """A tool result message fed back to the LLM.

    Contains the result of executing a tool call, identified by the
    original tool call ID.
    """

    role: str = field(default="tool", init=False)
    tool_call_id: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


# Union type for all message types
MessageType = UserMessage | SystemMessage | AssistantMessage | ToolCallMessage | ToolResultMessage


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class Conversation:
    """Manages the state of a multi-turn conversation.

    Tracks all messages in order, counts LLM interaction turns (assistant
    responses), and provides transcript serialization for attachment to
    ``TaskResult``.
    """

    def __init__(self) -> None:
        self._messages: list[MessageType] = []
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[MessageType]:
        """All messages in the conversation, in order."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Number of LLM response turns (assistant messages)."""
        return self._turn_count

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._messages.append(UserMessage(content=content))

    def add_system(self, content: str) -> None:
        """Add a system message to the conversation."""
        self._messages.append(SystemMessage(content=content))

    def add_assistant(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add an assistant message to the conversation.

        If ``tool_calls`` is non-empty, a ``ToolCallMessage`` is stored.
        Otherwise, an ``AssistantMessage`` is stored.

        Args:
            content: The text content of the assistant response.
            tool_calls: Optional list of tool call dicts.
        """
        if tool_calls:
            self._messages.append(ToolCallMessage(content=content, tool_calls=tool_calls))
        else:
            self._messages.append(AssistantMessage(content=content))
        self._turn_count += 1

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message to the conversation.

        Args:
            tool_call_id: The ID of the original tool call.
            content: The result content from the tool execution.
        """
        self._messages.append(ToolResultMessage(tool_call_id=tool_call_id, content=content))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_transcript(self) -> list[dict[str, Any]]:
        """Serialize all messages to a list of dicts.

        Returns:
            A list of message dicts with role, content, and optional fields.
        """
        return [msg.to_dict() for msg in self._messages]

    def to_provider_messages(self) -> list[dict[str, Any]]:
        """Convert to provider-format messages.

        Produces a list of dicts suitable for sending to LLM providers.
        """
        result: list[dict[str, Any]] = []
        for msg in self._messages:
            d = msg.to_dict()
            # ToolResultMessage: rename tool_call_id to tool_call_id for provider format
            if isinstance(msg, ToolResultMessage):
                d["tool_call_id"] = msg.tool_call_id
            result.append(d)
        return result

    def __repr__(self) -> str:
        return f"Conversation(messages={len(self._messages)}, turns={self._turn_count})"


__all__ = [
    "AssistantMessage",
    "Conversation",
    "MessageType",
    "SystemMessage",
    "ToolCallMessage",
    "ToolResultMessage",
    "UserMessage",
]
