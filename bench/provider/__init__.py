"""Central provider resolution and credential management.

This module provides a single source of truth for:
- Provider to API key environment variable mappings
- Provider to LiteLLM prefix mappings
- Model to provider resolution
- Credential validation and redaction
"""

from .resolver import (
    ProviderResolver,
    get_provider_resolver,
    resolve_provider,
    resolve_api_key_env,
    resolve_litellm_prefix,
)
from .auth import (
    AuthPolicy,
    CredentialResolver,
    get_credentials,
    validate_credentials,
)
from .models import (
    ProviderInfo,
    ModelInfo,
    CredentialSource,
)
from .preflight import (
    PreflightResult,
    run_preflight,
    validate_model_provider_config,
)

__all__ = [
    # Resolver
    "ProviderResolver",
    "get_provider_resolver",
    "resolve_provider",
    "resolve_api_key_env",
    "resolve_litellm_prefix",
    # Auth
    "AuthPolicy",
    "CredentialResolver",
    "get_credentials",
    "validate_credentials",
    # Models
    "ProviderInfo",
    "ModelInfo",
    "CredentialSource",
    # Preflight
    "PreflightResult",
    "run_preflight",
    "validate_model_provider_config",
]
