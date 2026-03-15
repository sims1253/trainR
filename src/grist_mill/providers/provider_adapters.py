"""LLM provider adapters for OpenAI, OpenRouter, and Anthropic.

Each adapter implements the ``BaseProvider`` interface and handles:
- HTTP communication with the provider's API
- Request/response format normalization
- Token usage extraction and normalization
- Tool call parsing
- Error mapping to the provider error taxonomy
- Retry with exponential backoff on transient errors

All adapters use ``httpx`` directly for HTTP communication, avoiding
the OpenAI Python SDK to keep dependencies minimal and behavior consistent.

Validates:
- VAL-PROVIDER-01: Multi-provider resolution
- VAL-PROVIDER-03: Provider-specific error mapping
- VAL-PROVIDER-05: Failover and retry on transient errors
- VAL-PROVIDER-06: Token counting per provider
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from grist_mill.agents.provider import (
    BaseProvider,
    ProviderMessage,
    ProviderResponse,
    ProviderToolCall,
)
from grist_mill.providers.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    ServerError,
    map_http_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared HTTP client base
# ---------------------------------------------------------------------------


class RetryableProvider(BaseProvider, ABC):
    """Base class for HTTP-based providers with retry support.

    Handles retry logic with exponential backoff for transient errors
    (5xx, connection errors, timeouts). Non-transient errors (401, 403, 429)
    are raised immediately without retry.

    Args:
        api_key: API key for authentication.
        model: Model identifier.
        max_retries: Maximum number of retry attempts (default 3).
        initial_backoff: Initial backoff delay in seconds (default 1.0).
        timeout: HTTP request timeout in seconds (default 60).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._timeout = timeout

    @property
    def model(self) -> str:
        """The model identifier."""
        return self._model

    @abstractmethod
    def _get_api_url(self) -> str:
        """Return the API endpoint URL."""
        ...

    @abstractmethod
    def _build_request(
        self, messages: list[ProviderMessage]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Build HTTP headers and JSON body for the API request.

        Returns:
            Tuple of (headers, body_dict).
        """
        ...

    @abstractmethod
    def _parse_response(self, response_json: dict[str, Any]) -> ProviderResponse:
        """Parse the JSON response body into a ProviderResponse.

        Args:
            response_json: Parsed JSON response from the API.

        Returns:
            A normalized ``ProviderResponse``.
        """
        ...

    def complete(self, messages: list[ProviderMessage]) -> ProviderResponse:
        """Send messages to the provider with retry support.

        Args:
            messages: Conversation history.

        Returns:
            A ``ProviderResponse`` with content, tool calls, and usage.

        Raises:
            AuthenticationError: On 401/403 (not retried).
            RateLimitError: On 429 (not retried).
            ServerError: On 5xx/connection errors after exhausting retries.
            ProviderError: On other HTTP errors.
        """
        url = self._get_api_url()
        headers, body = self._build_request(messages)

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._make_request(url, headers, body)

                if response.status_code >= 400:
                    error = map_http_error(
                        status_code=response.status_code,
                        body=response.text,
                        headers=dict(response.headers),
                        provider_hint=type(self).__name__,
                    )
                    # Non-transient errors: raise immediately
                    if not error.transient:
                        raise error
                    # Transient errors: retry
                    last_error = error
                    logger.warning(
                        "Provider %s transient error (attempt %d/%d): HTTP %d - %s",
                        type(self).__name__,
                        attempt,
                        self._max_retries,
                        response.status_code,
                        error.message,
                    )
                else:
                    response_json = response.json()
                    return self._parse_response(response_json)

            except httpx.TimeoutException as exc:
                last_error = ServerError(
                    f"Request timed out (attempt {attempt}/{self._max_retries}): {exc}",
                    status_code=0,
                    provider_hint=type(self).__name__,
                )
                logger.warning(
                    "Provider %s timeout (attempt %d/%d): %s",
                    type(self).__name__,
                    attempt,
                    self._max_retries,
                    exc,
                )

            except httpx.ConnectError as exc:
                last_error = ServerError(
                    f"Connection error (attempt {attempt}/{self._max_retries}): {exc}",
                    status_code=0,
                    provider_hint=type(self).__name__,
                )
                logger.warning(
                    "Provider %s connection error (attempt %d/%d): %s",
                    type(self).__name__,
                    attempt,
                    self._max_retries,
                    exc,
                )

            except (AuthenticationError, RateLimitError, ProviderError):
                # Non-transient errors: re-raise immediately
                raise

            # Backoff before retry
            if attempt < self._max_retries:
                delay = self._initial_backoff * (2 ** (attempt - 1))
                logger.debug(
                    "Retrying in %.2fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)

        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise ServerError(
            "All retry attempts failed.",
            provider_hint=type(self).__name__,
        )

    def _make_request(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> httpx.Response:
        """Make an HTTP POST request to the provider API.

        Args:
            url: API endpoint URL.
            headers: HTTP headers.
            body: JSON body.

        Returns:
            An ``httpx.Response``.
        """
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(url, headers=headers, json=body)


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(RetryableProvider):
    """OpenAI API provider.

    Uses the OpenAI chat completions endpoint:
    https://api.openai.com/v1/chat/completions
    """

    def _get_api_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def _build_request(
        self, messages: list[ProviderMessage]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        openai_messages = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            openai_messages.append(m)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
        }
        return headers, body

    def _parse_response(self, response_json: dict[str, Any]) -> ProviderResponse:
        choice = response_json["choices"][0]
        message = choice["message"]
        usage = response_json.get("usage", {})
        content = message.get("content") or ""
        tool_calls = self._parse_tool_calls(message.get("tool_calls", []))

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    @staticmethod
    def _parse_tool_calls(tool_calls_json: list[dict[str, Any]]) -> list[ProviderToolCall]:
        """Parse OpenAI-format tool calls into ProviderToolCall objects."""
        result: list[ProviderToolCall] = []
        for tc in tool_calls_json:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = {}
            result.append(
                ProviderToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )
        return result


# ---------------------------------------------------------------------------
# OpenRouter Provider
# ---------------------------------------------------------------------------


class OpenRouterProvider(RetryableProvider):
    """OpenRouter API provider.

    Uses the OpenAI-compatible OpenRouter endpoint:
    https://openrouter.ai/api/v1/chat/completions
    """

    def _get_api_url(self) -> str:
        return "https://openrouter.ai/api/v1/chat/completions"

    def _build_request(
        self, messages: list[ProviderMessage]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        openai_messages = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            openai_messages.append(m)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
        }
        return headers, body

    def _parse_response(self, response_json: dict[str, Any]) -> ProviderResponse:
        choice = response_json["choices"][0]
        message = choice["message"]
        usage = response_json.get("usage", {})
        content = message.get("content") or ""
        tool_calls = OpenAIProvider._parse_tool_calls(message.get("tool_calls", []))

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(RetryableProvider):
    """Anthropic API provider.

    Uses the Anthropic messages endpoint:
    https://api.anthropic.com/v1/messages

    Note: Anthropic uses ``x-api-key`` header instead of ``Authorization``,
    and ``input_tokens``/``output_tokens`` instead of ``prompt_tokens``/``completion_tokens``.
    """

    # Model IDs that support the newer API version
    _API_VERSION = "2023-06-01"

    def _get_api_url(self) -> str:
        return "https://api.anthropic.com/v1/messages"

    def _build_request(
        self, messages: list[ProviderMessage]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "anthropic-version": self._API_VERSION,
        }
        # Extract system prompt from messages
        system_content = ""
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_content += msg.content + "\n"
            else:
                m: dict[str, Any] = {"role": msg.role, "content": msg.content}
                if msg.tool_call_id:
                    # Tool result message
                    m = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                elif msg.tool_calls:
                    # Assistant message with tool use
                    content_blocks: list[dict[str, Any]] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "input": tc.get("arguments", {}),
                            }
                        )
                    m["content"] = content_blocks
                anthropic_messages.append(m)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }
        if system_content.strip():
            body["system"] = system_content.strip()

        return headers, body

    def _parse_response(self, response_json: dict[str, Any]) -> ProviderResponse:
        usage = response_json.get("usage", {})

        # Extract text content from content blocks
        content_parts: list[str] = []
        tool_calls: list[ProviderToolCall] = []
        for block in response_json.get("content", []):
            if block.get("type") == "text":
                content_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ProviderToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        content = "\n".join(content_parts)

        # Normalize Anthropic token names
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "RetryableProvider",
]
