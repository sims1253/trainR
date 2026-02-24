"""Data models for provider resolution and credential management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CredentialSource(Enum):
    """Source type for credentials."""

    ENV = "env"
    MOUNTED_FILE = "mounted_file"
    NONE = "none"


@dataclass
class ProviderInfo:
    """Information about a provider."""

    name: str
    """Provider name (e.g., 'openrouter', 'zai')."""

    litellm_prefix: str
    """LiteLLM prefix for the provider (e.g., 'openrouter/', 'openai/')."""

    api_key_env: str
    """Environment variable name for the API key."""

    display_name: str = ""
    """Human-readable display name."""

    base_url: str | None = None
    """Base URL for the provider API (if custom)."""

    supports_response_format: bool = True
    """Whether the provider supports response_format JSON mode."""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.title()


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    """Model identifier (e.g., 'glm-5-free', 'openai/gpt-oss-120b:free')."""

    provider: str
    """Provider name."""

    name: str = ""
    """Short name used in configs (e.g., 'glm-5', 'gpt-oss-120b')."""

    litellm_model: str = ""
    """Full LiteLLM model string (e.g., 'openai/glm-5-free')."""

    display_name: str = ""
    """Human-readable display name."""

    capabilities: dict[str, Any] = field(default_factory=dict)
    """Model capabilities (reasoning, json_mode, max_tokens_default, etc.)."""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id
        if not self.display_name:
            self.display_name = self.name

    @property
    def supports_reasoning(self) -> bool:
        """Whether the model supports reasoning content."""
        return self.capabilities.get("reasoning", False)

    @property
    def json_mode(self) -> str:
        """How JSON output is handled: 'native', 'prompt', or 'none'."""
        return self.capabilities.get("json_mode", "none")

    @property
    def max_tokens_default(self) -> int | None:
        """Default max tokens for the model."""
        return self.capabilities.get("max_tokens_default")
