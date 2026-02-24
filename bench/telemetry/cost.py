"""Cost estimation for LLM invocations."""

from __future__ import annotations

# Pricing table (USD per 1M tokens)
PRICING_TABLE: dict[str, dict[str, float]] = {
    # OpenAI models
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "openai/gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "openai/gpt-4": {"prompt": 30.00, "completion": 60.00},
    "openai/gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    # Anthropic models
    "anthropic/claude-3.5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "anthropic/claude-3.5-sonnet-v2": {"prompt": 3.00, "completion": 15.00},
    "anthropic/claude-3-opus": {"prompt": 15.00, "completion": 75.00},
    "anthropic/claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    # OpenRouter free models
    "openrouter/openai/gpt-oss-120b:free": {"prompt": 0.00, "completion": 0.00},
    # Add more models as needed
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate cost in USD for a model invocation.

    Args:
        model: Model identifier (e.g., 'openai/gpt-4o')
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        Estimated cost in USD, or None if pricing is unknown (not zero cost).
    """
    pricing = PRICING_TABLE.get(model)
    if pricing is None:
        return None  # Unknown pricing

    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

    return prompt_cost + completion_cost


class CostEstimator:
    """Estimate costs for benchmark runs."""

    def __init__(self, pricing_table: dict[str, dict[str, float]] | None = None) -> None:
        """Initialize the cost estimator.

        Args:
            pricing_table: Custom pricing table (USD per 1M tokens).
                          Uses default PRICING_TABLE if not provided.
        """
        self.pricing = pricing_table or PRICING_TABLE

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Estimate cost for a model invocation.

        Args:
            model: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD, or None if pricing unknown
        """
        pricing = self.pricing.get(model)
        if pricing is None:
            return None

        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

        return prompt_cost + completion_cost

    def add_pricing(self, model: str, prompt_price: float, completion_price: float) -> None:
        """Add or update pricing for a model.

        Args:
            model: Model identifier
            prompt_price: Price per 1M prompt tokens in USD
            completion_price: Price per 1M completion tokens in USD
        """
        self.pricing[model] = {
            "prompt": prompt_price,
            "completion": completion_price,
        }

    def get_supported_models(self) -> list[str]:
        """Get list of models with known pricing.

        Returns:
            List of model identifiers with pricing information
        """
        return list(self.pricing.keys())
