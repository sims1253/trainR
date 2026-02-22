"""Unified LLM configuration loader.

Single source of truth for all LLM settings in trainR.
Reads from configs/llm.yaml

LiteLLM providers: https://docs.litellm.ai/docs/providers
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """LLM configuration from configs/llm.yaml."""

    config: dict = field(default_factory=dict)
    _config_path: str | Path = "configs/llm.yaml"

    # Legacy fields for backward compatibility
    provider: str = "openrouter"
    task_agent: str = "openrouter/openai/gpt-oss-120b:free"
    reflection: str = "openrouter/openai/gpt-oss-120b:free"
    mining: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/llm.yaml") -> "LLMConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls(_config_path=path)

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Get legacy fields for backward compatibility
        data.get("models", {})
        defaults = data.get("defaults", {})
        old_models = data.get("models", {}) if "provider" in data else {}

        # Legacy format detection
        if "provider" in data:
            # Old format
            task_agent = old_models.get("task_agent", cls.task_agent)
            reflection = old_models.get("reflection", cls.reflection)
            mining = old_models.get("mining")
        else:
            # New format - use defaults
            task_agent = cls._resolve_model_to_litellm(
                data, defaults.get("task_agent", "glm-5-free")
            )
            reflection = cls._resolve_model_to_litellm(
                data, defaults.get("reflection", "glm-5-free")
            )
            mining = defaults.get("mining")

        return cls(
            config=data,
            _config_path=path,
            task_agent=task_agent,
            reflection=reflection,
            mining=mining,
        )

    @staticmethod
    def _resolve_model_to_litellm(config: dict, model_name: str) -> str:
        """Resolve a model name to its LiteLLM model string."""
        models = config.get("models", {})
        providers = config.get("providers", {})

        if model_name not in models:
            # Not a known model - return as-is
            return model_name

        model_cfg = models[model_name]
        provider_name = model_cfg.get("provider")
        model_id = model_cfg.get("id", model_name)

        if provider_name and provider_name in providers:
            provider_cfg = providers[provider_name]
            prefix = provider_cfg.get("litellm_prefix")
            if prefix:
                return f"{prefix}/{model_id}"

        return model_id

    def get_model_config(self, model_name: str) -> dict:
        """Get full configuration for a model by name."""
        models = self.config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in llm.yaml")

        model_cfg = models[model_name]
        provider_name = model_cfg.get("provider")
        providers = self.config.get("providers", {})
        provider_cfg = providers.get(provider_name, {})

        # Merge model and provider config
        return {
            "name": model_name,
            "id": model_cfg.get("id"),
            "provider": provider_name,
            "base_url": provider_cfg.get("base_url"),
            "api_key_env": provider_cfg.get("api_key_env"),
            "litellm_prefix": provider_cfg.get("litellm_prefix"),
            **model_cfg.get("capabilities", {}),
        }

    def get_litellm_model(self, model_name: str) -> str:
        """Get the LiteLLM-compatible model string."""
        cfg = self.get_model_config(model_name)
        model_id: str = cfg.get("id", model_name)
        prefix = cfg.get("litellm_prefix")

        if prefix:
            return f"{prefix}/{model_id}"
        return model_id

    def get_model_api_key(self, model_name: str) -> str | None:
        """Get the API key for a model."""
        cfg = self.get_model_config(model_name)
        env_var = cfg.get("api_key_env")
        if env_var:
            return os.environ.get(env_var)
        return None

    def get_model_base_url(self, model_name: str) -> str | None:
        """Get the base URL for a model's provider."""
        cfg = self.get_model_config(model_name)
        return cfg.get("base_url")

    def get_default_model(self, purpose: str) -> str | None:
        """Get the default model for a purpose (mining, task_agent, reflection, judge)."""
        defaults = self.config.get("defaults", {})
        return defaults.get(purpose)

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.config.get("models", {}).keys())

    # Legacy methods for backward compatibility
    def get_api_key(self) -> str | None:
        """Get API key for current provider (legacy)."""
        key_name = self._get_api_key_env(self.provider)
        if not key_name:
            return None
        key = os.environ.get(key_name)
        # Gemini fallback
        if self.provider == "gemini" and not key:
            key = os.environ.get("GOOGLE_API_KEY")
        return key

    def _get_api_key_env(self, provider: str) -> str | None:
        """Get the API key environment variable name for a provider."""
        mapping = {
            "openrouter": "OPENROUTER_API_KEY",
            "zai": "Z_AI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "opencode": "OPENCODE_API_KEY",
            "zai_coding_plan": "Z_AI_API_KEY",
        }
        return mapping.get(provider)

    def get_model_for_litellm(self, model: str | None = None) -> str:
        """Get model name with LiteLLM prefix."""
        model = model or self.task_agent
        if "/" in model:
            return model
        prefix = self._get_litellm_prefix(self.provider)
        return f"{prefix}{model}" if prefix else model

    def _get_litellm_prefix(self, provider: str) -> str:
        """Get the LiteLLM prefix for a provider."""
        mapping: dict[str, str] = {
            "openrouter": "openrouter/",
            "zai": "zai/",
            "gemini": "gemini/",
            "openai": "",
            "anthropic": "anthropic/",
            "opencode": "openai/",
            "zai_coding_plan": "openai/",
        }
        return mapping.get(provider, "")

    def get_base_url(
        self, model: str | None = None, config_path: str | Path = "configs/llm.yaml"
    ) -> str | None:
        """Get custom base_url if needed for this model."""
        # Try new format first
        if model:
            for prefix, _provider in [("opencode/", "opencode"), ("zai/", "zai_coding_plan")]:
                if model.startswith(prefix):
                    try:
                        model_name = model.replace(prefix, "").replace("openai/", "")
                        return self.get_model_base_url(model_name)
                    except ValueError:
                        pass

        # Legacy fallback
        provider = self.provider
        if model:
            for prefix, prov in [("opencode/", "opencode"), ("zai/", "zai_coding_plan")]:
                if model.startswith(prefix):
                    provider = prov
                    break

        custom_endpoints = self.config.get("custom_endpoints", {})
        endpoint_config = custom_endpoints.get(provider, {})
        return endpoint_config.get("base_url")

    def get_litellm_kwargs(self, model: str | None = None) -> dict[str, Any]:
        """Get kwargs for litellm.completion()."""
        base_url = self.get_base_url(model)
        if base_url:
            return {"base_url": base_url}
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
    provider_keys = {
        "openrouter": "OPENROUTER_API_KEY",
        "zai": "Z_AI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "opencode": "OPENCODE_API_KEY",
        "zai_coding_plan": "Z_AI_API_KEY",
    }

    available = []
    for provider, key_name in provider_keys.items():
        key = os.environ.get(key_name)
        if provider == "gemini" and not key:
            key = os.environ.get("GOOGLE_API_KEY")
        if key and provider not in available:
            available.append(provider)
    return available
