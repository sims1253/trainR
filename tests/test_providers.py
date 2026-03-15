"""Tests for multi-provider LLM resolution.

Covers:
- VAL-PROVIDER-01: Multi-provider LLM resolution abstracts provider differences
- VAL-PROVIDER-02: Provider credential validation at startup
- VAL-PROVIDER-03: Provider-specific error mapping
- VAL-PROVIDER-04: Custom provider registration
- VAL-PROVIDER-05: Provider failover and retry
- VAL-PROVIDER-06: Token counting and cost tracking per provider

All tests use mock HTTP responses to avoid real API calls.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from grist_mill.agents.provider import (
    BaseProvider,
    ProviderMessage,
    ProviderResponse,
)
from grist_mill.providers.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    ServerError,
    map_http_error,
)
from grist_mill.providers.pricing import (
    estimate_cost,
    get_pricing_for_provider,
)
from grist_mill.providers.provider_adapters import (
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from grist_mill.providers.registry import (
    ProviderRegistry,
    validate_credentials,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_messages() -> list[ProviderMessage]:
    """Create a simple conversation for testing."""
    return [
        ProviderMessage(role="system", content="You are a helpful assistant."),
        ProviderMessage(role="user", content="Hello, world!"),
    ]


def _mock_openai_response(
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a mock OpenAI API response."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop" if not tool_calls else "tool_calls",
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _mock_openrouter_response(
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Create a mock OpenRouter API response (OpenAI-compatible)."""
    return _mock_openai_response(content, prompt_tokens, completion_tokens)


def _mock_anthropic_response(
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict[str, Any]:
    """Create a mock Anthropic API response."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# VAL-PROVIDER-01: Multi-provider LLM resolution
# ---------------------------------------------------------------------------


class TestProviderRouting:
    """Test that a single config change routes to the correct backend."""

    def test_registry_resolves_openai_provider(self) -> None:
        """ProviderRegistry resolves 'openai' to OpenAIProvider."""
        registry = ProviderRegistry()
        provider = registry.resolve("openai", api_key="test-key", model="gpt-4")
        assert isinstance(provider, OpenAIProvider)

    def test_registry_resolves_openrouter_provider(self) -> None:
        """ProviderRegistry resolves 'openrouter' to OpenRouterProvider."""
        registry = ProviderRegistry()
        provider = registry.resolve("openrouter", api_key="test-key", model="openai/gpt-4")
        assert isinstance(provider, OpenRouterProvider)

    def test_registry_resolves_anthropic_provider(self) -> None:
        """ProviderRegistry resolves 'anthropic' to AnthropicProvider."""
        registry = ProviderRegistry()
        provider = registry.resolve("anthropic", api_key="test-key", model="claude-3-opus-20240229")
        assert isinstance(provider, AnthropicProvider)

    def test_same_config_different_provider(self) -> None:
        """Same config with different provider name routes differently."""
        registry = ProviderRegistry()

        openai_provider = registry.resolve("openai", api_key="key1", model="gpt-4")
        anthropic_provider = registry.resolve("anthropic", api_key="key2", model="claude-3-opus")

        assert isinstance(openai_provider, OpenAIProvider)
        assert isinstance(anthropic_provider, AnthropicProvider)
        assert type(openai_provider) is not type(anthropic_provider)

    def test_unknown_provider_raises_key_error(self) -> None:
        """Requesting an unregistered provider raises KeyError."""
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="not_a_real_provider"):
            registry.resolve("not_a_real_provider", api_key="key", model="model")

    def test_registry_lists_builtin_providers(self) -> None:
        """ProviderRegistry lists all builtin providers."""
        registry = ProviderRegistry()
        providers = registry.list_providers()
        assert "openai" in providers
        assert "openrouter" in providers
        assert "anthropic" in providers

    def test_registry_has_provider(self) -> None:
        """ProviderRegistry.has() checks for registered providers."""
        registry = ProviderRegistry()
        assert registry.has("openai") is True
        assert registry.has("openrouter") is True
        assert registry.has("nonexistent") is False


# ---------------------------------------------------------------------------
# VAL-PROVIDER-02: Credential validation at startup
# ---------------------------------------------------------------------------


class TestCredentialValidation:
    """Test that missing credentials produce immediate errors."""

    def test_validate_credentials_missing_key(self) -> None:
        """Missing API key produces a descriptive error."""
        registry = ProviderRegistry()
        # Clear any existing env vars to test missing credential
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="OPENAI_API_KEY"),
        ):
            validate_credentials(registry, "openai", {})

    def test_validate_credentials_missing_openrouter_key(self) -> None:
        """Missing OpenRouter key produces error naming OPENROUTER_API_KEY."""
        registry = ProviderRegistry()
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="OPENROUTER_API_KEY"),
        ):
            validate_credentials(registry, "openrouter", {})

    def test_validate_credentials_missing_anthropic_key(self) -> None:
        """Missing Anthropic key produces error naming ANTHROPIC_API_KEY."""
        registry = ProviderRegistry()
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ANTHROPIC_API_KEY"),
        ):
            validate_credentials(registry, "anthropic", {})

    def test_validate_credentials_valid_key(self) -> None:
        """Valid credentials pass without error."""
        registry = ProviderRegistry()
        # Should not raise
        validate_credentials(registry, "openai", {"OPENAI_API_KEY": "dummy-key"})

    def test_validate_credentials_env_vars_take_precedence(self) -> None:
        """Environment variable credentials are used if config key missing."""
        registry = ProviderRegistry()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-env-key"}, clear=True):
            # Should not raise — env var provides the credential
            validate_credentials(registry, "openai", {})

    def test_validate_credentials_custom_provider(self) -> None:
        """Custom provider with no required keys passes validation."""
        registry = ProviderRegistry()

        def custom_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(responses=[])

        registry.register("custom", custom_factory, required_env_vars=[])
        # Should not raise
        validate_credentials(registry, "custom", {})

    def test_validate_credentials_custom_provider_with_keys(self) -> None:
        """Custom provider with required keys checks those keys."""
        registry = ProviderRegistry()

        def custom_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(responses=[])

        registry.register(
            "custom",
            custom_factory,
            required_env_vars=["CUSTOM_LLM_KEY"],
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="CUSTOM_LLM_KEY"),
        ):
            validate_credentials(registry, "custom", {})


# ---------------------------------------------------------------------------
# VAL-PROVIDER-03: Provider-specific error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """Test that HTTP errors map to the error taxonomy."""

    def test_401_maps_to_authentication_error(self) -> None:
        """HTTP 401 maps to AuthenticationError."""
        exc = map_http_error(status_code=401, body="Unauthorized")
        assert isinstance(exc, AuthenticationError)
        assert "401" in str(exc)

    def test_403_maps_to_authentication_error(self) -> None:
        """HTTP 403 maps to AuthenticationError."""
        exc = map_http_error(status_code=403, body="Forbidden")
        assert isinstance(exc, AuthenticationError)

    def test_429_maps_to_rate_limit_error(self) -> None:
        """HTTP 429 maps to RateLimitError."""
        exc = map_http_error(status_code=429, body="Rate limited")
        assert isinstance(exc, RateLimitError)
        assert "rate limit" in str(exc).lower() or "429" in str(exc)

    def test_500_maps_to_server_error(self) -> None:
        """HTTP 500 maps to ServerError (transient)."""
        exc = map_http_error(status_code=500, body="Internal Server Error")
        assert isinstance(exc, ServerError)
        assert exc.transient is True

    def test_502_maps_to_server_error(self) -> None:
        """HTTP 502 maps to ServerError (transient)."""
        exc = map_http_error(status_code=502, body="Bad Gateway")
        assert isinstance(exc, ServerError)
        assert exc.transient is True

    def test_503_maps_to_server_error(self) -> None:
        """HTTP 503 maps to ServerError (transient)."""
        exc = map_http_error(status_code=503, body="Service Unavailable")
        assert isinstance(exc, ServerError)
        assert exc.transient is True

    def test_404_maps_to_provider_error(self) -> None:
        """HTTP 404 maps to ProviderError (not found)."""
        exc = map_http_error(status_code=404, body="Not Found")
        assert isinstance(exc, ProviderError)
        assert exc.transient is False

    def test_422_maps_to_provider_error(self) -> None:
        """HTTP 422 maps to ProviderError (bad request)."""
        exc = map_http_error(status_code=422, body="Unprocessable Entity")
        assert isinstance(exc, ProviderError)
        assert exc.transient is False

    def test_400_maps_to_provider_error(self) -> None:
        """HTTP 400 maps to ProviderError."""
        exc = map_http_error(status_code=400, body="Bad Request")
        assert isinstance(exc, ProviderError)

    def test_error_messages_are_actionable(self) -> None:
        """Error messages contain actionable information."""
        exc = map_http_error(status_code=401, body="Invalid API key")
        msg = str(exc).lower()
        # Should mention the key or how to fix it
        assert "api key" in msg or "401" in msg or "auth" in msg

    def test_rate_limit_error_has_retry_info(self) -> None:
        """RateLimitError includes retry-after info when available."""
        exc = map_http_error(status_code=429, body="Rate limited", headers={"Retry-After": "30"})
        assert isinstance(exc, RateLimitError)
        assert exc.retry_after == 30

    def test_connection_error_maps_to_transient(self) -> None:
        """Network/connection errors are mapped as transient."""
        exc = map_http_error(status_code=0, body="Connection refused")
        assert isinstance(exc, ServerError)
        assert exc.transient is True


# ---------------------------------------------------------------------------
# VAL-PROVIDER-04: Custom provider registration
# ---------------------------------------------------------------------------


class TestCustomProviderRegistration:
    """Test that custom providers can be registered and used."""

    def test_register_custom_provider(self) -> None:
        """Custom provider can be registered and resolved."""
        registry = ProviderRegistry()

        def custom_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(
                responses=[
                    ProviderResponse(
                        content="Custom response",
                        usage={"prompt_tokens": 5, "completion_tokens": 3},
                    )
                ]
            )

        registry.register("my-provider", custom_factory, required_env_vars=[])
        provider = registry.resolve("my-provider", api_key="key", model="test")
        assert isinstance(provider, MockProvider)
        response = provider.complete(_make_messages())
        assert response.content == "Custom response"

    def test_register_custom_provider_with_config(self) -> None:
        """Custom provider is used when referenced in config."""
        registry = ProviderRegistry()

        def custom_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(
                responses=[
                    ProviderResponse(
                        content="Config-driven response",
                        usage={"prompt_tokens": 10, "completion_tokens": 5},
                    )
                ]
            )

        registry.register("custom-via-config", custom_factory, required_env_vars=[])
        provider = registry.resolve("custom-via-config", api_key="key", model="test")
        response = provider.complete(_make_messages())
        assert response.content == "Config-driven response"

    def test_duplicate_registration_without_overwrite(self) -> None:
        """Registering a duplicate without overwrite raises ValueError."""
        registry = ProviderRegistry()

        def factory1(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(responses=[])

        def factory2(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(responses=[])

        registry.register("test-dup", factory1, required_env_vars=[])
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test-dup", factory2, required_env_vars=[])

    def test_duplicate_registration_with_overwrite(self) -> None:
        """Registering a duplicate with overwrite=True succeeds."""
        registry = ProviderRegistry()

        def factory1(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(responses=[])

        def factory2(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
            return MockProvider(
                responses=[
                    ProviderResponse(
                        content="Overwritten",
                        usage={"prompt_tokens": 1, "completion_tokens": 1},
                    )
                ]
            )

        registry.register("test-overwrite", factory1, required_env_vars=[])
        registry.register("test-overwrite", factory2, required_env_vars=[], overwrite=True)
        provider = registry.resolve("test-overwrite", api_key="key", model="test")
        response = provider.complete(_make_messages())
        assert response.content == "Overwritten"


# ---------------------------------------------------------------------------
# VAL-PROVIDER-05: Provider failover and retry
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test that transient errors trigger retry with backoff."""

    def test_retry_on_500_error(self) -> None:
        """Transient 500 errors trigger retry with backoff."""
        registry = ProviderRegistry()

        call_count = 0
        response_after_retries = _mock_openai_response()

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(
                    status_code=500,
                    json={"error": {"message": "Internal Server Error"}},
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                status_code=200,
                json=response_after_retries,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openai", api_key="test-key", model="gpt-4", max_retries=3)
            messages = _make_messages()
            response = provider.complete(messages)

        assert response.content == "Hello!"
        assert call_count == 3  # 2 failures + 1 success

    def test_retry_on_timeout(self) -> None:
        """Timeout errors trigger retry."""
        registry = ProviderRegistry()

        call_count = 0
        response_after_retries = _mock_openai_response()

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise httpx.TimeoutException("Request timed out")
            return httpx.Response(
                status_code=200,
                json=response_after_retries,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve(
                "openai",
                api_key="test-key",
                model="gpt-4",
                max_retries=2,
            )
            messages = _make_messages()
            response = provider.complete(messages)

        assert response.content == "Hello!"
        assert call_count == 2

    def test_no_retry_on_401(self) -> None:
        """Authentication errors (401) are NOT retried."""
        registry = ProviderRegistry()

        call_count = 0

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                status_code=401,
                json={"error": {"message": "Invalid API key"}},
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openai", api_key="test-key", model="gpt-4", max_retries=3)
            messages = _make_messages()
            with pytest.raises(AuthenticationError):
                provider.complete(messages)

        assert call_count == 1  # No retry for 401

    def test_no_retry_on_429_with_retry_after(self) -> None:
        """Rate limit (429) errors are NOT retried automatically (actionable)."""
        registry = ProviderRegistry()

        call_count = 0

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                status_code=429,
                json={"error": {"message": "Rate limited"}},
                headers={"Retry-After": "60"},
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openai", api_key="test-key", model="gpt-4", max_retries=3)
            messages = _make_messages()
            with pytest.raises(RateLimitError):
                provider.complete(messages)

        assert call_count == 1  # No retry for 429

    def test_exhausted_retries_raises(self) -> None:
        """When all retries are exhausted, the last error is raised."""
        registry = ProviderRegistry()

        call_count = 0

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                status_code=500,
                json={"error": {"message": "Internal Server Error"}},
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openai", api_key="test-key", model="gpt-4", max_retries=3)
            messages = _make_messages()
            with pytest.raises(ServerError):
                provider.complete(messages)

        assert call_count == 3  # max_retries attempts

    def test_backoff_delay_increases(self) -> None:
        """Retry delays increase with exponential backoff."""
        registry = ProviderRegistry()
        timestamps: list[float] = []

        call_count = 0

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            timestamps.append(time.monotonic())
            if call_count <= 3:
                return httpx.Response(
                    status_code=503,
                    json={"error": {"message": "Service Unavailable"}},
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                status_code=200,
                json=_mock_openai_response(),
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve(
                "openai",
                api_key="test-key",
                model="gpt-4",
                max_retries=4,
                initial_backoff=0.05,
            )
            messages = _make_messages()
            provider.complete(messages)

        assert len(timestamps) == 4
        # Delays should increase: gap2 > gap1, gap3 > gap2
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        gap3 = timestamps[3] - timestamps[2]
        assert gap2 > gap1
        assert gap3 > gap2


# ---------------------------------------------------------------------------
# VAL-PROVIDER-06: Token counting and cost tracking
# ---------------------------------------------------------------------------


class TestTokenCountingAndCost:
    """Test per-provider token counting and cost estimation."""

    def test_openai_provider_reports_token_usage(self) -> None:
        """OpenAI provider reports token usage in response."""
        registry = ProviderRegistry()
        mock_resp = _mock_openai_response(prompt_tokens=100, completion_tokens=50)

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openai", api_key="test-key", model="gpt-4")
            response = provider.complete(_make_messages())

        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 50
        assert response.usage["total_tokens"] == 150

    def test_openrouter_provider_reports_token_usage(self) -> None:
        """OpenRouter provider reports token usage in response."""
        registry = ProviderRegistry()
        mock_resp = _mock_openrouter_response(prompt_tokens=200, completion_tokens=80)

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openrouter", api_key="test-key", model="openai/gpt-4")
            response = provider.complete(_make_messages())

        assert response.usage["prompt_tokens"] == 200
        assert response.usage["completion_tokens"] == 80
        assert response.usage["total_tokens"] == 280

    def test_anthropic_provider_reports_token_usage(self) -> None:
        """Anthropic provider reports token usage in response."""
        registry = ProviderRegistry()
        mock_resp = _mock_anthropic_response(input_tokens=300, output_tokens=120)

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve(
                "anthropic",
                api_key="test-key",
                model="claude-3-opus-20240229",
            )
            response = provider.complete(_make_messages())

        assert response.usage["prompt_tokens"] == 300
        assert response.usage["completion_tokens"] == 120
        assert response.usage["total_tokens"] == 420

    def test_cost_estimation_openai(self) -> None:
        """Cost estimation for OpenAI uses correct pricing."""
        pricing = get_pricing_for_provider("openai", "gpt-4")
        cost = estimate_cost(
            provider="openai",
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        expected = pricing.input_per_1m * 1000 / 1_000_000 + pricing.output_per_1m * 500 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_cost_estimation_openrouter(self) -> None:
        """Cost estimation for OpenRouter uses default pricing."""
        cost = estimate_cost(
            provider="openrouter",
            model="openai/gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert cost > 0

    def test_cost_estimation_anthropic(self) -> None:
        """Cost estimation for Anthropic uses correct pricing."""
        pricing = get_pricing_for_provider("anthropic", "claude-3-opus-20240229")
        cost = estimate_cost(
            provider="anthropic",
            model="claude-3-opus-20240229",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        expected = pricing.input_per_1m * 1000 / 1_000_000 + pricing.output_per_1m * 500 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_cost_estimation_unknown_model_returns_zero(self) -> None:
        """Cost estimation for unknown model returns 0.0."""
        cost = estimate_cost(
            provider="openai",
            model="unknown-model-xyz",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert cost == 0.0

    def test_token_usage_defaults_zero_on_missing(self) -> None:
        """Token usage defaults to 0 when not reported."""
        registry = ProviderRegistry()
        mock_resp = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                },
            ],
            # No usage field
        }

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = registry.resolve("openai", api_key="test-key", model="gpt-4")
            response = provider.complete(_make_messages())

        assert response.usage["prompt_tokens"] == 0
        assert response.usage["completion_tokens"] == 0
        assert response.usage["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Provider adapter tests (OpenAI, OpenRouter, Anthropic)
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Tests for the OpenAI provider adapter."""

    def test_complete_returns_response(self) -> None:
        """OpenAI provider returns a valid ProviderResponse."""
        mock_resp = _mock_openai_response()

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = OpenAIProvider(api_key="test-key", model="gpt-4")
            response = provider.complete(_make_messages())

        assert response.content == "Hello!"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

    def test_complete_with_tool_calls(self) -> None:
        """OpenAI provider parses tool calls from response."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/test.py"}',
                },
            },
        ]
        mock_resp = _mock_openai_response(
            content="", prompt_tokens=10, completion_tokens=20, tool_calls=tool_calls
        )

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = OpenAIProvider(api_key="test-key", model="gpt-4")
            response = provider.complete(_make_messages())

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "/tmp/test.py"}

    def test_auth_header_set(self) -> None:
        """OpenAI provider sets Authorization header."""
        captured_headers: dict[str, str] = {}

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            captured_headers.update(kwargs.get("headers", {}))
            return httpx.Response(
                status_code=200,
                json=_mock_openai_response(),
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = OpenAIProvider(api_key="dummy-key", model="gpt-4")
            provider.complete(_make_messages())

        assert captured_headers.get("Authorization") == "Bearer dummy-key"


class TestOpenRouterProvider:
    """Tests for the OpenRouter provider adapter."""

    def test_complete_returns_response(self) -> None:
        """OpenRouter provider returns a valid ProviderResponse."""
        mock_resp = _mock_openrouter_response()

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4")
            response = provider.complete(_make_messages())

        assert response.content == "Hello!"

    def test_openrouter_url_correct(self) -> None:
        """OpenRouter provider uses the correct API URL."""
        captured_url: str = ""

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal captured_url
            captured_url = url
            return httpx.Response(
                status_code=200,
                json=_mock_openrouter_response(),
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4")
            provider.complete(_make_messages())

        assert "openrouter.ai" in captured_url


class TestAnthropicProvider:
    """Tests for the Anthropic provider adapter."""

    def test_complete_returns_response(self) -> None:
        """Anthropic provider returns a valid ProviderResponse."""
        mock_resp = _mock_anthropic_response()

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
            response = provider.complete(_make_messages())

        assert response.content == "Hello!"

    def test_anthropic_token_normalization(self) -> None:
        """Anthropic provider normalizes input_tokens to prompt_tokens."""
        mock_resp = _mock_anthropic_response(input_tokens=100, output_tokens=50)

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json=mock_resp,
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
            response = provider.complete(_make_messages())

        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 50
        assert response.usage["total_tokens"] == 150

    def test_anthropic_url_correct(self) -> None:
        """Anthropic provider uses the correct API URL."""
        captured_url: str = ""

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            nonlocal captured_url
            captured_url = url
            return httpx.Response(
                status_code=200,
                json=_mock_anthropic_response(),
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
            provider.complete(_make_messages())

        assert "api.anthropic.com" in captured_url

    def test_anthropic_x_api_key_header(self) -> None:
        """Anthropic provider uses x-api-key header, not Authorization."""
        captured_headers: dict[str, str] = {}

        def mock_post(url: str, **kwargs: Any) -> httpx.Response:
            captured_headers.update(kwargs.get("headers", {}))
            return httpx.Response(
                status_code=200,
                json=_mock_anthropic_response(),
                request=httpx.Request("POST", url),
            )

        with patch("httpx.Client.post", side_effect=mock_post):
            provider = AnthropicProvider(api_key="dummy-key", model="claude-3-opus-20240229")
            provider.complete(_make_messages())

        assert captured_headers.get("x-api-key") == "dummy-key"
        assert "Authorization" not in captured_headers


# ---------------------------------------------------------------------------
# MockProvider import for custom tests
# ---------------------------------------------------------------------------


class MockProvider(BaseProvider):
    """Simple mock provider for testing custom registration."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ProviderMessage]) -> ProviderResponse:
        if not self._responses:
            raise StopIteration("No more responses")
        return self._responses.pop(0)
