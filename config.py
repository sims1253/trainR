"""Unified LLM configuration loader.

Single source of truth for all LLM settings in trainR.
Reads from configs/llm.yaml

LiteLLM providers: https://docs.litellm.ai/docs/providers
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# API key environment variables
PROVIDER_API_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "Z_AI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "zai_coding_plan": "Z_AI_API_KEY",  # Same key, different endpoint
}

# LiteLLM model prefixes
PROVIDER_PREFIXES: dict[str, str] = {
    "openrouter": "openrouter/",
    "zai": "zai/",
    "gemini": "gemini/",
    "openai": "",
    "anthropic": "anthropic/",
    "opencode": "openai/",  # OpenAI-compatible
    "zai_coding_plan": "openai/",  # OpenAI-compatible
}


@dataclass
class LLMConfig:
    """LLM configuration from configs/llm.yaml."""

    provider: str = "openrouter"
    task_agent: str = "openrouter/openai/gpt-oss-120b:free"
    reflection: str = "openrouter/openai/gpt-oss-120b:free"
    mining: str | None = None  # Defaults to reflection
    temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/llm.yaml") -> "LLMConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            provider=data.get("provider", "openrouter"),
            task_agent=data.get("models", {}).get("task_agent", cls.task_agent),
            reflection=data.get("models", {}).get("reflection", cls.reflection),
            mining=data.get("models", {}).get("mining"),
            temperature=data.get("parameters", {}).get("temperature", cls.temperature),
            max_tokens=data.get("parameters", {}).get("max_tokens", cls.max_tokens),
        )

    def get_api_key(self) -> str | None:
        """Get API key for current provider."""
        import os

        key_name = PROVIDER_API_KEYS.get(self.provider)
        if not key_name:
            return None
        key = os.environ.get(key_name)
        # Gemini fallback
        if self.provider == "gemini" and not key:
            key = os.environ.get("GOOGLE_API_KEY")
        return key

    def get_model_for_litellm(self, model: str | None = None) -> str:
        """Get model name with LiteLLM prefix."""
        model = model or self.task_agent
        if "/" in model:
            return model
        prefix = PROVIDER_PREFIXES.get(self.provider, "")
        return f"{prefix}{model}"

    def get_base_url(self, config_path: str | Path = "configs/llm.yaml") -> str | None:
        """Get custom base_url if provider uses one.

        Only needed for providers not natively supported by LiteLLM.
        """
        path = Path(config_path)
        if not path.exists():
            return None

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        custom_endpoints = data.get("custom_endpoints", {})
        endpoint_config = custom_endpoints.get(self.provider, {})
        return endpoint_config.get("base_url")

    def get_litellm_kwargs(self) -> dict[str, Any]:
        """Get kwargs for litellm.completion()."""
        base_url = self.get_base_url()
        if base_url:
            return {"api_base": base_url}
        return {}

    def get_mining_model(self) -> str:
        """Get model for PR mining (defaults to reflection)."""
        return self.mining or self.reflection


# Global cached config
_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get global LLM config (cached)."""
    global _config
    if _config is None:
        _config = LLMConfig.from_yaml()
    return _config


def reload_config() -> LLMConfig:
    """Reload config from file."""
    global _config
    _config = LLMConfig.from_yaml()
    return _config


def list_available_providers() -> list[str]:
    """List providers with API keys set."""
    import os

    available = []
    for provider, key_name in PROVIDER_API_KEYS.items():
        key = os.environ.get(key_name)
        if provider == "gemini" and not key:
            key = os.environ.get("GOOGLE_API_KEY")
        if key and provider not in available:
            available.append(provider)
    return available
