"""Provider resolution from configuration."""

from pathlib import Path
from typing import Any

import yaml

from .models import ModelInfo, ProviderInfo

# Canonical provider to API key environment variable mappings
PROVIDER_API_KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "Z_AI_API_KEY",  # Note: Z_AI_API_KEY not ZAI_API_KEY
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "kimi": "KIMI_API_KEY",
}

# Canonical provider to LiteLLM prefix mappings
PROVIDER_LITELLM_PREFIX = {
    "openrouter": "openrouter/",
    "zai": "zai/",
    "gemini": "gemini/",
    "openai": "",
    "anthropic": "anthropic/",
    "opencode": "openai/",  # opencode uses OpenAI-compatible API
}


class ProviderResolver:
    """Resolves providers and models from configuration.

    This class is the single source of truth for:
    - Provider to API key environment variable mappings
    - Provider to LiteLLM prefix mappings
    - Model to provider resolution
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the resolver.

        Args:
            config_path: Path to llm.yaml. If None, uses default location.
        """
        self._config_path = self._resolve_config_path(config_path)
        self._config: dict[str, Any] | None = None
        self._providers: dict[str, ProviderInfo] | None = None
        self._models: dict[str, ModelInfo] | None = None

    def _resolve_config_path(self, config_path: str | Path | None) -> Path:
        """Resolve the configuration file path."""
        if config_path is not None:
            return Path(config_path)

        # Try standard locations
        candidates = [
            Path("configs/llm.yaml"),
            Path("config/llm.yaml"),
            Path.cwd() / "configs" / "llm.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Default to configs/llm.yaml (will fail later if not found)
        return Path("configs/llm.yaml")

    def _load_config(self) -> dict[str, Any]:
        """Load the configuration from YAML file."""
        if self._config is None:
            if not self._config_path.exists():
                raise FileNotFoundError(f"LLM configuration not found at {self._config_path}")
            with open(self._config_path) as f:
                self._config = yaml.safe_load(f)
        return self._config or {}

    def _load_providers(self) -> dict[str, ProviderInfo]:
        """Load provider information from configuration."""
        if self._providers is not None:
            return self._providers

        config = self._load_config()
        providers_config = config.get("providers", {})

        self._providers = {}

        for name, prov_config in providers_config.items():
            # Get API key env from config or canonical map
            api_key_env = prov_config.get("api_key_env") or PROVIDER_API_KEY_MAP.get(
                name, f"{name.upper()}_API_KEY"
            )

            # Get LiteLLM prefix from config or canonical map
            litellm_prefix_raw = prov_config.get("litellm_prefix", "")
            # Normalize prefix with trailing slash
            if litellm_prefix_raw and not litellm_prefix_raw.endswith("/"):
                litellm_prefix = f"{litellm_prefix_raw}/"
            else:
                litellm_prefix = litellm_prefix_raw

            # Override with canonical map if available
            if name in PROVIDER_LITELLM_PREFIX:
                litellm_prefix = PROVIDER_LITELLM_PREFIX[name]

            self._providers[name] = ProviderInfo(
                name=name,
                litellm_prefix=litellm_prefix,
                api_key_env=api_key_env,
                base_url=prov_config.get("base_url"),
                supports_response_format=prov_config.get("supports_response_format", True),
            )

        return self._providers

    def _load_models(self) -> dict[str, ModelInfo]:
        """Load model information from configuration."""
        if self._models is not None:
            return self._models

        config = self._load_config()
        models_config = config.get("models", {})
        providers = self._load_providers()

        self._models = {}

        for name, model_config in models_config.items():
            # Skip models with multiple providers (they have 'providers' key)
            if "providers" in model_config:
                # Use the first provider as default
                first_provider = model_config["providers"][0]
                model_id = first_provider["id"]
                provider_name = first_provider["provider"]
            else:
                model_id = model_config.get("id", name)
                provider_name = model_config.get("provider", "")

            provider_info = providers.get(provider_name)
            if provider_info:
                litellm_model = f"{provider_info.litellm_prefix}{model_id}"
            else:
                litellm_model = model_id

            self._models[name] = ModelInfo(
                id=model_id,
                provider=provider_name,
                name=name,
                litellm_model=litellm_model,
                capabilities=model_config.get("capabilities", {}),
            )

        return self._models

    @property
    def providers(self) -> dict[str, ProviderInfo]:
        """Get all loaded providers."""
        return self._load_providers()

    @property
    def models(self) -> dict[str, ModelInfo]:
        """Get all loaded models."""
        return self._load_models()

    def resolve_provider(self, model_name: str) -> str:
        """Resolve the provider name for a given model name.

        Args:
            model_name: The short model name (e.g., 'glm-5', 'gpt-oss-120b').

        Returns:
            The provider name (e.g., 'opencode', 'openrouter').

        Raises:
            KeyError: If the model is not found.
        """
        models = self._load_models()
        if model_name not in models:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        return models[model_name].provider

    def get_provider_info(self, provider_name: str) -> ProviderInfo:
        """Get provider information by name.

        Args:
            provider_name: The provider name (e.g., 'opencode', 'openrouter').

        Returns:
            ProviderInfo for the provider.

        Raises:
            KeyError: If the provider is not found.
        """
        providers = self._load_providers()
        if provider_name not in providers:
            raise KeyError(f"Provider '{provider_name}' not found in configuration")
        return providers[provider_name]

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information by name.

        Args:
            model_name: The short model name (e.g., 'glm-5', 'gpt-oss-120b').

        Returns:
            ModelInfo for the model.

        Raises:
            KeyError: If the model is not found.
        """
        models = self._load_models()
        if model_name not in models:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        return models[model_name]

    def get_api_key_env(self, provider_name: str) -> str:
        """Get the API key environment variable name for a provider.

        Args:
            provider_name: The provider name (e.g., 'opencode', 'openrouter').

        Returns:
            The environment variable name (e.g., 'OPENCODE_API_KEY').

        Raises:
            KeyError: If the provider is not found.
        """
        provider_info = self.get_provider_info(provider_name)
        return provider_info.api_key_env

    def get_litellm_prefix(self, provider_name: str) -> str:
        """Get the LiteLLM prefix for a provider.

        Args:
            provider_name: The provider name (e.g., 'opencode', 'openrouter').

        Returns:
            The LiteLLM prefix (e.g., 'openrouter/', 'openai/').

        Raises:
            KeyError: If the provider is not found.
        """
        provider_info = self.get_provider_info(provider_name)
        return provider_info.litellm_prefix

    def get_litellm_model(self, model_name: str) -> str:
        """Get the full LiteLLM model string for a model name.

        Args:
            model_name: The short model name (e.g., 'glm-5', 'gpt-oss-120b').

        Returns:
            The LiteLLM model string (e.g., 'openai/glm-5-free').

        Raises:
            KeyError: If the model is not found.
        """
        model_info = self.get_model_info(model_name)
        return model_info.litellm_model

    def get_default_model(self, purpose: str = "mining") -> str:
        """Get the default model name for a purpose.

        Args:
            purpose: The purpose (e.g., 'mining', 'task_agent', 'reflection').

        Returns:
            The default model name.
        """
        config = self._load_config()
        defaults = config.get("defaults", {})
        return defaults.get(purpose, defaults.get("mining", "glm-5"))

    def resolve_api_key_env(self, model_name: str) -> str:
        """Resolve the API key environment variable for a model.

        Args:
            model_name: The short model name.

        Returns:
            The environment variable name for the API key.
        """
        provider = self.resolve_provider(model_name)
        return self.get_api_key_env(provider)

    def resolve_litellm_prefix(self, model_name: str) -> str:
        """Resolve the LiteLLM prefix for a model.

        Args:
            model_name: The short model name.

        Returns:
            The LiteLLM prefix for the model's provider.
        """
        provider = self.resolve_provider(model_name)
        return self.get_litellm_prefix(provider)


# Module-level singleton for convenience
_resolver_instance: ProviderResolver | None = None


def get_provider_resolver(config_path: str | Path | None = None) -> ProviderResolver:
    """Get the singleton ProviderResolver instance.

    Args:
        config_path: Optional path to llm.yaml. Only used on first call.

    Returns:
        The ProviderResolver singleton.
    """
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = ProviderResolver(config_path)
    return _resolver_instance


# Convenience functions using the singleton
def resolve_provider(model_name: str) -> str:
    """Resolve the provider name for a model. Uses singleton resolver."""
    return get_provider_resolver().resolve_provider(model_name)


def resolve_api_key_env(model_or_provider: str) -> str:
    """Resolve the API key environment variable. Uses singleton resolver.

    Args:
        model_or_provider: Either a model name or provider name.

    Returns:
        The environment variable name.
    """
    resolver = get_provider_resolver()
    # Try as provider first
    try:
        return resolver.get_api_key_env(model_or_provider)
    except KeyError:
        # Try as model
        return resolver.resolve_api_key_env(model_or_provider)


def resolve_litellm_prefix(model_or_provider: str) -> str:
    """Resolve the LiteLLM prefix. Uses singleton resolver.

    Args:
        model_or_provider: Either a model name or provider name.

    Returns:
        The LiteLLM prefix.
    """
    resolver = get_provider_resolver()
    # Try as provider first
    try:
        return resolver.get_litellm_prefix(model_or_provider)
    except KeyError:
        # Try as model
        return resolver.resolve_litellm_prefix(model_or_provider)
