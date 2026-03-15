"""Pricing data and cost estimation for LLM providers.

Provides per-provider, per-model pricing information and cost estimation
utilities. Pricing is stored per million tokens and uses standard published
rates as of early 2024.

Validates VAL-PROVIDER-06: Token counting and cost tracking per provider.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderPricing:
    """Pricing for a specific model on a specific provider.

    Attributes:
        input_per_1m: Cost per 1 million input tokens (USD).
        output_per_1m: Cost per 1 million output tokens (USD).
    """

    input_per_1m: float
    output_per_1m: float


# ---------------------------------------------------------------------------
# Pricing database
# ---------------------------------------------------------------------------

# Format: {provider: {model: ProviderPricing}}
_PRICING_DB: dict[str, dict[str, ProviderPricing]] = {
    "openai": {
        "gpt-4": ProviderPricing(input_per_1m=30.0, output_per_1m=60.0),
        "gpt-4-turbo": ProviderPricing(input_per_1m=10.0, output_per_1m=30.0),
        "gpt-4o": ProviderPricing(input_per_1m=5.0, output_per_1m=15.0),
        "gpt-4o-mini": ProviderPricing(input_per_1m=0.15, output_per_1m=0.60),
        "gpt-3.5-turbo": ProviderPricing(input_per_1m=0.50, output_per_1m=1.50),
    },
    "openrouter": {
        "openai/gpt-4": ProviderPricing(input_per_1m=30.0, output_per_1m=60.0),
        "openai/gpt-4-turbo": ProviderPricing(input_per_1m=10.0, output_per_1m=30.0),
        "openai/gpt-4o": ProviderPricing(input_per_1m=5.0, output_per_1m=15.0),
        "openai/gpt-4o-mini": ProviderPricing(input_per_1m=0.15, output_per_1m=0.60),
    },
    "anthropic": {
        "claude-3-opus-20240229": ProviderPricing(input_per_1m=15.0, output_per_1m=75.0),
        "claude-3-sonnet-20240229": ProviderPricing(input_per_1m=3.0, output_per_1m=15.0),
        "claude-3-haiku-20240307": ProviderPricing(input_per_1m=0.25, output_per_1m=1.25),
        "claude-3-5-sonnet-20241022": ProviderPricing(input_per_1m=3.0, output_per_1m=15.0),
    },
}


def get_pricing_for_provider(provider: str, model: str) -> ProviderPricing:
    """Get pricing for a specific provider and model.

    Args:
        provider: Provider name (e.g., 'openai', 'openrouter', 'anthropic').
        model: Model name (e.g., 'gpt-4', 'openai/gpt-4').

    Returns:
        A ``ProviderPricing`` for the given model.

    Raises:
        KeyError: If the provider or model is not found in the pricing database.
    """
    provider_pricing = _PRICING_DB.get(provider, {})
    pricing = provider_pricing.get(model)
    if pricing is None:
        raise KeyError(
            f"No pricing found for provider={provider!r}, model={model!r}. "
            f"Known providers: {list(_PRICING_DB.keys())}. "
            f"Known models for {provider}: {list(provider_pricing.keys())}."
        )
    return pricing


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the cost for a provider call.

    Args:
        provider: Provider name.
        model: Model name.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD. Returns 0.0 if the model is not in the
        pricing database.
    """
    try:
        pricing = get_pricing_for_provider(provider, model)
    except KeyError:
        return 0.0

    input_cost = pricing.input_per_1m * prompt_tokens / 1_000_000
    output_cost = pricing.output_per_1m * completion_tokens / 1_000_000
    return input_cost + output_cost


__all__ = [
    "ProviderPricing",
    "estimate_cost",
    "get_pricing_for_provider",
]
