"""
Inference Gateway - Single entry point for structured LLM output.

Provides a unified interface for calling different LLM providers using their
native SDKs, replacing LiteLLM for mining/judging flows.
"""

from typing import Literal

from pydantic import BaseModel

from bench.provider.adapters.openai_compat import InferenceResult, OpenAICompatAdapter
from bench.provider.env import get_env_var
from bench.provider.resolver import ProviderResolver

# API style types
ApiStyle = Literal["openai_compat", "openai_native", "anthropic", "gemini"]

# Canonical defaults for providers that use OpenAI-compatible APIs.
OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "opencode": "https://opencode.ai/zen/v1",
    "zai": "https://api.z.ai/api/paas/v4",
    "zai_coding_plan": "https://api.z.ai/api/coding/paas/v4",
}


def generate_structured(
    messages: list[dict],
    response_schema: type[BaseModel],
    model_name: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    resolver: ProviderResolver | None = None,
    api_key: str | None = None,
    verbose: bool = False,
) -> InferenceResult:
    """
    Generate structured output from an LLM matching the given schema.

    This is the single entry point for structured LLM inference in the
    mining/judging path. It routes to the appropriate provider adapter
    based on the model's api_style.

    Args:
        messages: Chat messages in OpenAI format
        response_schema: Pydantic model to validate response
        model_name: Model name from llm.yaml (e.g., "glm-5-plan")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        resolver: Optional ProviderResolver instance
        api_key: Optional API key (will be resolved if not provided)
        verbose: Whether to print debug information

    Returns:
        InferenceResult with validated content

    Raises:
        ValueError: If model or provider not found
        KeyError: If API key not available
    """
    if resolver is None:
        resolver = ProviderResolver()

    # Resolve model info
    model_info = resolver.get_model_info(model_name)
    provider_name = model_info.provider
    provider_info = resolver.get_provider_info(provider_name)

    # Get model ID and API style
    model_id = model_info.id
    # Default to openai_compat for providers without explicit api_style
    api_style: ApiStyle = getattr(provider_info, "api_style", None) or "openai_compat"
    json_mode = model_info.json_mode

    # Resolve max_tokens from model capabilities if not provided
    if max_tokens is None:
        max_tokens = model_info.max_tokens_default

    # Resolve API key
    if api_key is None:
        api_key_env = provider_info.api_key_env
        if api_key_env:
            api_key = get_env_var(api_key_env)
        if not api_key:
            raise KeyError(f"API key not found for provider {provider_name} (env: {api_key_env})")

    # Get base URL
    base_url = provider_info.base_url

    if verbose:
        print(f"[inference] model={model_name}, model_id={model_id}, provider={provider_name}")
        print(f"[inference] api_style={api_style}, json_mode={json_mode}, base_url={base_url}")

    # Route to appropriate adapter based on api_style
    if api_style in ("openai_compat", "openai_native"):
        if not base_url and api_style == "openai_compat":
            base_url = OPENAI_COMPAT_BASE_URLS.get(provider_name)
            if not base_url:
                raise ValueError(f"base_url required for openai_compat provider {provider_name}")

        adapter = OpenAICompatAdapter(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key or "",
        )
        return adapter.generate_structured(
            model_id=model_id,
            messages=messages,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    elif api_style == "anthropic":
        # TODO: Implement Anthropic adapter when needed
        raise NotImplementedError("Anthropic adapter not yet implemented")

    elif api_style == "gemini":
        # TODO: Implement Gemini adapter when needed
        raise NotImplementedError("Gemini adapter not yet implemented")

    else:
        raise ValueError(f"Unknown api_style: {api_style}")


def embed_schema_in_messages(
    messages: list[dict],
    response_schema: type[BaseModel],
) -> list[dict]:
    """
    Embed JSON schema instructions in the last user message.

    Used for models that don't support native JSON mode (json_mode: "prompt").

    Args:
        messages: Original messages
        response_schema: Pydantic model defining expected response format

    Returns:
        Modified messages with schema instructions
    """
    schema = response_schema.model_json_schema()
    schema_instruction = (
        f"\n\nRespond with valid JSON matching this schema:\n```json\n{schema}\n```"
    )

    # Make a copy to avoid mutating input
    messages = list(messages)

    # Find last user message and append schema
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            messages[i] = {
                **messages[i],
                "content": messages[i].get("content", "") + schema_instruction,
            }
            break

    return messages
