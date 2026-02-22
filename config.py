"""Unified LLM configuration loader.

Single source of truth for all LLM settings in trainR.
Reads from configs/llm.yaml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class LLMConfig:
    """LLM configuration loaded from configs/llm.yaml."""

    provider: Literal["openrouter", "openai", "anthropic"] = "openrouter"
    task_agent: str = "openrouter/openai/gpt-oss-120b:free"
    reflection: str = "openrouter/openai/gpt-oss-120b:free"
    temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/llm.yaml") -> "LLMConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: {path} not found, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            provider=data.get("provider", "openrouter"),
            task_agent=data.get("models", {}).get("task_agent", cls.task_agent),
            reflection=data.get("models", {}).get("reflection", cls.reflection),
            temperature=data.get("parameters", {}).get("temperature", cls.temperature),
            max_tokens=data.get("parameters", {}).get("max_tokens", cls.max_tokens),
        )

    def get_api_key(self) -> str | None:
        """Get API key for current provider."""
        import os

        key_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        return os.environ.get(key_map.get(self.provider, ""))


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
