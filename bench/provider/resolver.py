"""Provider resolution from configuration."""

import warnings
from pathlib import Path
from typing import Any

import yaml

from .models import ModelInfo, ProviderInfo

# Canonical provider to API key environment variable mappings
PROVIDER_API_KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "Z_AI_API_KEY",  # Canonical key name (ZAI_API_KEY alias is normalized)
    "zai_coding_plan": "Z_AI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "kimi": "KIMI_API_KEY",
}

# Canonical provider to API style mappings
PROVIDER_API_STYLE = {
    "openrouter": "openai_compat",
    "zai": "openai_compat",
    "zai_coding_plan": "openai_compat",
    "gemini": "gemini",
    "openai": "openai_native",
    "anthropic": "anthropic",
    "opencode": "openai_compat",  # OpenAI-compatible API
    "kimi": "openai_compat",
}

# Backward compatibility - deprecated
# Note: These are stored WITHOUT trailing slash to match old YAML format
PROVIDER_LITELLM_PREFIX = {
    "openrouter": "openrouter",
    "zai": "openai",  # zai uses OpenAI-compatible API format
    "zai_coding_plan": "openai",
    "gemini": "gemini",
    "openai": "",
    "anthropic": "anthropic",
    "opencode": "openai",  # opencode uses OpenAI-compatible API
    "kimi": "openai",  # kimi uses OpenAI-compatible API
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

            # Get API style from config or canonical map
            api_style = prov_config.get("api_style") or PROVIDER_API_STYLE.get(
                name, "openai_compat"
            )

            # Backward compatibility: compute litellm_prefix from canonical map
            litellm_prefix = PROVIDER_LITELLM_PREFIX.get(name, "")

            self._providers[name] = ProviderInfo(
                name=name,
                api_style=api_style,
                api_key_env=api_key_env,
                base_url=prov_config.get("base_url"),
                supports_response_format=prov_config.get("supports_response_format", True),
                litellm_prefix=litellm_prefix,
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

            # Backward compatibility: compute litellm_model
            if provider_info and provider_info.litellm_prefix:
                litellm_model = f"{provider_info.litellm_prefix}/{model_id}"
            else:
                litellm_model = model_id

            self._models[name] = ModelInfo(
                id=model_id,
                provider=provider_name,
                name=name,
                model_id=model_id,
                litellm_model=litellm_model,
                capabilities=model_config.get("capabilities", {}),
                _provider_info=provider_info,
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

    def get_api_style(self, provider_name: str) -> str:
        """Get the API style for a provider.

        Args:
            provider_name: The provider name (e.g., 'opencode', 'openrouter').

        Returns:
            The API style (e.g., 'openai_compat', 'anthropic', 'gemini').

        Raises:
            KeyError: If the provider is not found.
        """
        provider_info = self.get_provider_info(provider_name)
        return provider_info.api_style

    def get_litellm_prefix(self, provider_name: str) -> str:
        """Get the LiteLLM prefix for a provider.

        DEPRECATED: Use get_api_style() instead.

        Args:
            provider_name: The provider name (e.g., 'opencode', 'openrouter').

        Returns:
            The LiteLLM prefix (e.g., 'openrouter/', 'openai/').

        Raises:
            KeyError: If the provider is not found.
        """
        warnings.warn(
            "get_litellm_prefix() is deprecated. Use get_api_style() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        provider_info = self.get_provider_info(provider_name)
        return provider_info.litellm_prefix

    def get_model_id(self, model_name: str) -> str:
        """Get the raw model ID to send to the API.

        Args:
            model_name: The short model name (e.g., 'glm-5', 'gpt-oss-120b').

        Returns:
            The raw model ID (e.g., 'glm-5-free', 'openai/gpt-oss-120b:free').

        Raises:
            KeyError: If the model is not found.
        """
        model_info = self.get_model_info(model_name)
        return model_info.model_id or model_info.id

    def get_litellm_model(self, model_name: str) -> str:
        """Get the full LiteLLM model string for a model name.

        DEPRECATED: Use get_model_id() instead.

        Args:
            model_name: The short model name (e.g., 'glm-5', 'gpt-oss-120b').

        Returns:
            The LiteLLM model string (e.g., 'openai/glm-5-free').

        Raises:
            KeyError: If the model is not found.
        """
        warnings.warn(
            "get_litellm_model() is deprecated. Use get_model_id() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def resolve_api_style(self, model_name: str) -> str:
        """Resolve the API style for a model.

        Args:
            model_name: The short model name.

        Returns:
            The API style for the model's provider.
        """
        provider = self.resolve_provider(model_name)
        return self.get_api_style(provider)

    def resolve_litellm_prefix(self, model_name: str) -> str:
        """Resolve the LiteLLM prefix for a model.

        DEPRECATED: Use resolve_api_style() instead.

        Args:
            model_name: The short model name.

        Returns:
            The LiteLLM prefix for the model's provider.
        """
        warnings.warn(
            "resolve_litellm_prefix() is deprecated. Use resolve_api_style() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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


def resolve_api_style(model_or_provider: str) -> str:
    """Resolve the API style. Uses singleton resolver.

    Args:
        model_or_provider: Either a model name or provider name.

    Returns:
        The API style (e.g., 'openai_compat', 'anthropic', 'gemini').
    """
    resolver = get_provider_resolver()
    # Try as provider first
    try:
        return resolver.get_api_style(model_or_provider)
    except KeyError:
        # Try as model
        return resolver.resolve_api_style(model_or_provider)


def resolve_litellm_prefix(model_or_provider: str) -> str:
    """Resolve the LiteLLM prefix. Uses singleton resolver.

    DEPRECATED: Use resolve_api_style() instead.

    Args:
        model_or_provider: Either a model name or provider name.

    Returns:
        The LiteLLM prefix.
    """
    warnings.warn(
        "resolve_litellm_prefix() is deprecated. Use resolve_api_style() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    resolver = get_provider_resolver()
    # Try as provider first
    try:
        return resolver.get_litellm_prefix(model_or_provider)
    except KeyError:
        # Try as model
        return resolver.resolve_litellm_prefix(model_or_provider)
