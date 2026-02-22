"""Unified LLM configuration loader.

Single source of truth for all LLM settings in trainR.
Reads from configs/llm.yaml

LiteLLM providers: https://docs.litellm.ai/docs/providers
OpenAI-compatible: https://docs.litellm.ai/docs/providers/openai_compatible
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

# Supported providers and their API key environment variables
PROVIDER_API_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "Z_AI_API_KEY",  # Z.AI / Zhipu AI (API or Coding Plan)
    "gemini": "GEMINI_API_KEY",  # Google AI Studio
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "opencode": "OPENCODE_API_KEY",  # OpenCode Zen (OpenAI-compatible)
}

# Model prefixes for LiteLLM
PROVIDER_PREFIXES: dict[str, str] = {
    "openrouter": "openrouter/",
    "zai": "zai/",
    "gemini": "gemini/",
    "openai": "",  # OpenAI models don't need prefix
    "anthropic": "anthropic/",
    "opencode": "openai/",  # OpenAI-compatible mode
}

# Default base URLs for providers (can be overridden in config)
DEFAULT_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "zai": "https://open.bigmodel.cn/api/paas/v4",  # Zhipu API
    "zai_coding_plan": "https://api.zukijourney.com/v1",  # ZAI Coding Plan
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "opencode": None,  # Set in config
}


@dataclass
class LLMConfig:
    """LLM configuration loaded from configs/llm.yaml."""

    provider: Literal["openrouter", "zai", "gemini", "openai", "anthropic", "opencode"] = (
        "openrouter"
    )
    task_agent: str = "openrouter/openai/gpt-oss-120b:free"
    reflection: str = "openrouter/openai/gpt-oss-120b:free"
    temperature: float = 0.3
    max_tokens: int = 4096
    base_url: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/llm.yaml") -> "LLMConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: {path} not found, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        provider = data.get("provider", "openrouter")
        models = data.get("models", {})
        params = data.get("parameters", {})
        providers_config = data.get("providers", {})

        # Get base_url for this provider
        provider_config = providers_config.get(provider, {})
        base_url = provider_config.get("base_url")

        return cls(
            provider=provider,
            task_agent=models.get("task_agent", cls.task_agent),
            reflection=models.get("reflection", cls.reflection),
            temperature=params.get("temperature", cls.temperature),
            max_tokens=params.get("max_tokens", cls.max_tokens),
            base_url=base_url,
        )

    def get_api_key(self) -> str | None:
        """Get API key for current provider."""
        import os

        key_name = PROVIDER_API_KEYS.get(self.provider)
        if not key_name:
            return None

        key = os.environ.get(key_name)

        # Special case: Gemini can use GEMINI_API_KEY or GOOGLE_API_KEY
        if self.provider == "gemini" and not key:
            key = os.environ.get("GOOGLE_API_KEY")

        return key

    def get_model_for_litellm(self, model: str | None = None) -> str:
        """Get model name formatted for LiteLLM."""
        model = model or self.task_agent
        if "/" in model:
            return model  # Already has prefix
        prefix = PROVIDER_PREFIXES.get(self.provider, "")
        return f"{prefix}{model}"

    def get_litellm_kwargs(self) -> dict[str, Any]:
        """Get kwargs for litellm.completion() call.

        Includes api_base for OpenAI-compatible providers.
        """
        kwargs: dict[str, Any] = {}

        # For providers with custom base_url, pass it to LiteLLM
        if self.base_url:
            kwargs["api_base"] = self.base_url

        return kwargs


# Global config instance (lazy loaded)
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
    """List providers that have API keys set in environment."""
    import os

    available = []
    for provider, key_name in PROVIDER_API_KEYS.items():
        key = os.environ.get(key_name)
        if provider == "gemini" and not key:
            key = os.environ.get("GOOGLE_API_KEY")
        if key:
            available.append(provider)

    return available
