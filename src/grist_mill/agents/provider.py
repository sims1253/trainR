"""Provider abstraction for LLM interactions.

Defines the interface for communicating with LLM providers and provides
a ``MockProvider`` for testing. Provider-agnostic: the agent delegates
to a ``BaseProvider`` subclass, and routing is controlled by config.

Validates VAL-AGENT-02: Provider-agnostic agent routing.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """A message sent to an LLM provider."""

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = field(default=None)


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """A response from an LLM provider.

    Contains the text content, any requested tool calls, and token usage.
    """

    content: str
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseProvider (abstract interface)
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract interface for LLM providers.

    Subclasses must implement ``complete`` to send messages and receive
    a response from the LLM.
    """

    @abstractmethod
    def complete(self, messages: list[ProviderMessage]) -> ProviderResponse:
        """Send messages to the LLM and receive a response.

        Args:
            messages: Conversation history to send to the LLM.

        Returns:
            A ``ProviderResponse`` with content, tool calls, and usage info.
        """
        ...


# ---------------------------------------------------------------------------
# MockProvider — for testing
# ---------------------------------------------------------------------------


class MockProvider(BaseProvider):
    """Mock provider that returns pre-configured responses.

    Useful for testing agent behavior without real API calls.

    Args:
        responses: List of responses to return in order.
        delay: Optional delay in seconds before each response (for timeout testing).
    """

    def __init__(
        self,
        responses: list[ProviderResponse],
        delay: float = 0.0,
    ) -> None:
        self._responses = list(responses)
        self._index = 0
        self._delay = delay

    def complete(self, messages: list[ProviderMessage]) -> ProviderResponse:
        """Return the next configured response.

        Args:
            messages: Conversation history (ignored by mock).

        Returns:
            The next ``ProviderResponse`` in the queue.

        Raises:
            StopIteration: If all configured responses have been consumed.
        """
        if self._delay > 0:
            time.sleep(self._delay)

        if self._index >= len(self._responses):
            raise StopIteration(
                f"MockProvider has no more responses (requested {self._index + 1}, "
                f"have {len(self._responses)})"
            )

        response = self._responses[self._index]
        self._index += 1
        logger.debug("MockProvider returning response %d/%d", self._index, len(self._responses))
        return response


__all__ = [
    "BaseProvider",
    "MockProvider",
    "ProviderMessage",
    "ProviderResponse",
    "ProviderToolCall",
]
