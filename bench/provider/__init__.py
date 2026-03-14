"""Central provider resolution and credential management.

This module provides a single source of truth for:
- Provider to API key environment variable mappings
- Provider to API style mappings
- Model to provider resolution
- Credential validation and redaction
"""

from .auth import (
    AuthPolicy,
    CredentialResolver,
    clear_credential_resolver_cache,
    get_credentials,
    validate_credentials,
)
from .env import (
    api_key_aliases,
    bootstrap_environment,
    get_env_var,
    normalize_api_key_aliases,
)
from .inference import (
    InferenceResult,
    embed_schema_in_messages,
    generate_structured,
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
    resolve_api_style,
    resolve_litellm_prefix,
    resolve_provider,
)

__all__ = [
    "AuthPolicy",
    "CredentialResolver",
    "CredentialSource",
    "InferenceResult",
    "ModelInfo",
    "PreflightResult",
    "ProviderInfo",
    "ProviderResolver",
    "api_key_aliases",
    "bootstrap_environment",
    "clear_credential_resolver_cache",
    "embed_schema_in_messages",
    "generate_structured",
    "get_credentials",
    "get_env_var",
    "get_provider_resolver",
    "normalize_api_key_aliases",
    "resolve_api_key_env",
    "resolve_api_style",
    "resolve_litellm_prefix",
    "resolve_provider",
    "run_preflight",
    "validate_credentials",
    "validate_model_provider_config",
]
