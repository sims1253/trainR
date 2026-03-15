"""Provider registry for pluggable LLM provider resolution.

Provides a central ``ProviderRegistry`` that maps provider names to factory
functions. Builtin providers (OpenAI, OpenRouter, Anthropic) are
pre-registered. Custom providers can be registered at runtime.

The registry is the single entry point for:
- Resolving a provider by name and configuration
- Validating credentials before execution
- Listing available providers

Validates:
- VAL-PROVIDER-01: Multi-provider LLM resolution
- VAL-PROVIDER-02: Credential validation at startup
- VAL-PROVIDER-04: Custom provider registration via config
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from grist_mill.agents.provider import BaseProvider
from grist_mill.providers.provider_adapters import (
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

logger = logging.getLogger(__name__)

# Type alias for provider factory functions
ProviderFactory = Callable[..., BaseProvider]

# Mapping of builtin provider names to required env var keys
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ProviderRegistry:
    """Central registry for LLM provider resolution.

    Providers are registered by name with a factory function that creates
    a provider instance from configuration parameters. Builtin providers
    (openai, openrouter, anthropic) are pre-registered.

    Usage::

        registry = ProviderRegistry()
        provider = registry.resolve("openai", api_key="sk-...", model="gpt-4")
        response = provider.complete(messages)
    """

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}
        self._required_env_vars: dict[str, list[str]] = {}
        # Pre-register builtin providers
        self._register_builtins()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register builtin provider factories."""
        self.register(
            "openai",
            _openai_factory,
            required_env_vars=["OPENAI_API_KEY"],
        )
        self.register(
            "openrouter",
            _openrouter_factory,
            required_env_vars=["OPENROUTER_API_KEY"],
        )
        self.register(
            "anthropic",
            _anthropic_factory,
            required_env_vars=["ANTHROPIC_API_KEY"],
        )

    def register(
        self,
        name: str,
        factory: ProviderFactory,
        *,
        required_env_vars: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a provider factory by name.

        Args:
            name: Unique name for the provider.
            factory: A callable that creates a ``BaseProvider`` instance.
                Must accept at least ``api_key: str`` and ``model: str`` kwargs.
            required_env_vars: List of environment variable names required
                for this provider's credentials.
            overwrite: If True, replace an existing registration.

        Raises:
            ValueError: If a provider with the same name is already registered
                and ``overwrite`` is False.
        """
        if name in self._factories and not overwrite:
            msg = f"Provider '{name}' is already registered. Use overwrite=True to replace it."
            raise ValueError(msg)

        if name in self._factories:
            logger.info("Overwriting provider '%s'", name)
        else:
            logger.info("Registering provider '%s'", name)

        self._factories[name] = factory
        self._required_env_vars[name] = required_env_vars or []

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        **kwargs: Any,
    ) -> BaseProvider:
        """Resolve a provider by name and configuration.

        Args:
            name: The registered provider name.
            api_key: API key for the provider.
            model: Model identifier.
            **kwargs: Additional provider-specific arguments
                (e.g., ``max_retries``, ``initial_backoff``).

        Returns:
            A ``BaseProvider`` instance.

        Raises:
            KeyError: If the provider name is not registered.
        """
        if name not in self._factories:
            available = self.list_providers()
            msg = f"Provider '{name}' is not registered. Available providers: {available}"
            raise KeyError(msg)

        factory = self._factories[name]
        provider = factory(api_key=api_key, model=model, **kwargs)
        logger.debug(
            "Resolved provider '%s' to %s (model=%s)",
            name,
            type(provider).__name__,
            model,
        )
        return provider

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: The provider name to check.

        Returns:
            True if the provider is registered.
        """
        return name in self._factories

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            A sorted list of provider name strings.
        """
        return sorted(self._factories.keys())

    def get_required_env_vars(self, name: str) -> list[str]:
        """Get the required environment variable names for a provider.

        Args:
            name: The provider name.

        Returns:
            A list of environment variable names.

        Raises:
            KeyError: If the provider name is not registered.
        """
        if name not in self._required_env_vars:
            msg = f"Provider '{name}' is not registered."
            raise KeyError(msg)
        return list(self._required_env_vars[name])

    def __repr__(self) -> str:
        return f"ProviderRegistry(providers={self.list_providers()})"


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------


def validate_credentials(
    registry: ProviderRegistry,
    provider_name: str,
    config_env: dict[str, str] | None = None,
) -> None:
    """Validate that required credentials are available for a provider.

    Checks both the provided config_env dict and the process environment
    variables. Raises ``ValueError`` with a descriptive message if any
    required credential is missing.

    Args:
        registry: The provider registry.
        provider_name: The provider name to validate.
        config_env: Optional dict of config-provided environment variables
            (e.g., from YAML config's ``api_key`` field).

    Raises:
        ValueError: If a required credential is missing.
        KeyError: If the provider name is not registered.
    """
    required_vars = registry.get_required_env_vars(provider_name)
    config_env = config_env or {}

    for var_name in required_vars:
        # Check config_env first, then environment variables
        if config_env.get(var_name):
            continue
        env_val = os.environ.get(var_name, "")
        if env_val:
            continue

        # Credential not found
        msg = (
            f"Missing required credential for provider '{provider_name}': "
            f"environment variable '{var_name}' is not set. "
            f"Set it in your environment or provide it in your config file."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Builtin provider factories
# ---------------------------------------------------------------------------


def _openai_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
    """Create an OpenAI provider."""
    return OpenAIProvider(
        api_key=api_key,
        model=model,
        max_retries=kwargs.get("max_retries", 3),
        initial_backoff=kwargs.get("initial_backoff", 1.0),
        timeout=kwargs.get("timeout", 60.0),
    )


def _openrouter_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
    """Create an OpenRouter provider."""
    return OpenRouterProvider(
        api_key=api_key,
        model=model,
        max_retries=kwargs.get("max_retries", 3),
        initial_backoff=kwargs.get("initial_backoff", 1.0),
        timeout=kwargs.get("timeout", 60.0),
    )


def _anthropic_factory(api_key: str, model: str, **kwargs: Any) -> BaseProvider:
    """Create an Anthropic provider."""
    return AnthropicProvider(
        api_key=api_key,
        model=model,
        max_retries=kwargs.get("max_retries", 3),
        initial_backoff=kwargs.get("initial_backoff", 1.0),
        timeout=kwargs.get("timeout", 60.0),
    )


__all__ = [
    "ProviderFactory",
    "ProviderRegistry",
    "validate_credentials",
]
