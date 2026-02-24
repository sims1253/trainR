"""Central provider resolution and credential management.

This module provides a single source of truth for:
- Provider to API key environment variable mappings
- Provider to LiteLLM prefix mappings
- Model to provider resolution
- Credential validation and redaction
"""

from .auth import (
    AuthPolicy,
    CredentialResolver,
    get_credentials,
    validate_credentials,
)
from .models import (
    CredentialSource,
    ModelInfo,
    ProviderInfo,
)
from .preflight import (
    PreflightResult,
    run_preflight,
    validate_model_provider_config,
)
from .resolver import (
    ProviderResolver,
    get_provider_resolver,
    resolve_api_key_env,
    resolve_litellm_prefix,
    resolve_provider,
)

__all__ = [
    # Auth
    "AuthPolicy",
    "CredentialResolver",
    "CredentialSource",
    "ModelInfo",
    # Preflight
    "PreflightResult",
    # Models
    "ProviderInfo",
    # Resolver
    "ProviderResolver",
    "get_credentials",
    "get_provider_resolver",
    "resolve_api_key_env",
    "resolve_litellm_prefix",
    "resolve_provider",
    "run_preflight",
    "validate_credentials",
    "validate_model_provider_config",
]
