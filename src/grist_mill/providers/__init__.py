"""Multi-provider LLM resolution for the grist-mill framework.

Provides:
- ProviderRegistry: Central registry for pluggable LLM providers
- OpenAIProvider: OpenAI chat completions adapter
- OpenRouterProvider: OpenRouter (OpenAI-compatible) adapter
- AnthropicProvider: Anthropic messages adapter
- Provider errors: Typed error hierarchy with HTTP error mapping
- Pricing: Per-provider, per-model cost estimation
- Credential validation: Startup-time credential checking

All providers implement ``BaseProvider`` and support:
- Retry with exponential backoff on transient errors (5xx, timeout)
- Token usage reporting normalized across providers
- Cost estimation using published pricing data
- Custom provider registration via config

Validates:
- VAL-PROVIDER-01: Multi-provider LLM resolution
- VAL-PROVIDER-02: Credential validation at startup
- VAL-PROVIDER-03: Provider-specific error mapping
- VAL-PROVIDER-04: Custom provider registration
- VAL-PROVIDER-05: Failover and retry on transient errors
- VAL-PROVIDER-06: Token counting and cost tracking per provider
"""

from grist_mill.providers.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    ServerError,
    map_http_error,
)
from grist_mill.providers.pricing import (
    ProviderPricing,
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

__all__ = [
    "AnthropicProvider",
    "AuthenticationError",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderError",
    "ProviderPricing",
    "ProviderRegistry",
    "RateLimitError",
    "ServerError",
    "estimate_cost",
    "get_pricing_for_provider",
    "map_http_error",
    "validate_credentials",
]
