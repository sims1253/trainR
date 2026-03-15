# Provider System

## Overview

The `grist_mill.providers` module implements multi-provider LLM resolution for the grist-mill framework. It provides a pluggable architecture for routing to different LLM providers (OpenAI, OpenRouter, Anthropic, or custom) via a single configuration change.

## Architecture

```
grist_mill/providers/
├── __init__.py            # Public API re-exports
├── errors.py              # Error taxonomy (AuthenticationError, RateLimitError, ServerError, ProviderError)
├── pricing.py             # Per-provider, per-model pricing data and cost estimation
├── provider_adapters.py   # OpenAIProvider, OpenRouterProvider, AnthropicProvider (all extend RetryableProvider)
└── registry.py            # ProviderRegistry + validate_credentials()
```

## Key Design Decisions

1. **Direct httpx usage**: Providers use `httpx` directly instead of the OpenAI SDK to keep dependencies minimal and behavior consistent across providers.

2. **RetryableProvider base class**: All HTTP-based providers extend `RetryableProvider` which provides:
   - Automatic retry with exponential backoff for transient errors (5xx, timeouts)
   - No retry for non-transient errors (401, 403, 429)
   - Configurable `max_retries` and `initial_backoff`

3. **Token normalization**: Each provider normalizes its API-specific token names (e.g., Anthropic's `input_tokens`/`output_tokens`) to the common `prompt_tokens`/`completion_tokens`/`total_tokens` format.

4. **Provider factory pattern**: The `ProviderRegistry` uses factory functions, not classes directly. This allows dependency injection and custom configuration.

5. **Credential validation**: `validate_credentials()` checks both config-provided env vars and process environment variables before execution.

## Usage Pattern

```python
from grist_mill.providers import ProviderRegistry, validate_credentials

registry = ProviderRegistry()

# Validate credentials first
validate_credentials(registry, "openai", {})

# Resolve provider from config
provider = registry.resolve("openai", api_key="sk-...", model="gpt-4")

# Use with APIAgent
from grist_mill.agents.api_agent import APIAgent
agent = APIAgent(provider=provider, max_turns=10, timeout=600)
```

## Custom Provider Registration

```python
registry = ProviderRegistry()

def my_custom_provider(api_key: str, model: str, **kwargs) -> BaseProvider:
    # Return a BaseProvider instance
    ...

registry.register("my-custom", my_custom_provider, required_env_vars=["MY_API_KEY"])
```

## Validation Assertions

- VAL-PROVIDER-01: Multi-provider resolution (single config change switches provider)
- VAL-PROVIDER-02: Credential validation at startup (missing key → ValueError naming the key)
- VAL-PROVIDER-03: HTTP error mapping (401→Auth, 429→RateLimit, 5xx→ServerError)
- VAL-PROVIDER-04: Custom provider registration
- VAL-PROVIDER-05: Retry on transient errors with backoff
- VAL-PROVIDER-06: Token counting and cost estimation per provider

## Pricing Database

Pricing is in `providers/pricing.py` with published rates for:
- OpenAI: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- OpenRouter: openai/gpt-4, openai/gpt-4-turbo, openai/gpt-4o, openai/gpt-4o-mini
- Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet

Unknown models return $0.00 cost estimate.
