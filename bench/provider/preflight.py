"""Preflight validation for benchmark execution.

Validates that all required credentials and environment settings are in place
before starting execution.
"""

from dataclasses import dataclass, field
from typing import Any

from .auth import AuthPolicy, CredentialResolver
from .resolver import get_provider_resolver


@dataclass
class PreflightResult:
    """Result of preflight validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def run_preflight(
    models: list[str],
    auth_policy: AuthPolicy = AuthPolicy.ENV,
    strict: bool = True,
) -> PreflightResult:
    """Run preflight validation for the given models.

    Args:
        models: List of model names to validate credentials for
        auth_policy: How credentials should be loaded
        strict: If True, fail on missing credentials. If False, warn only.

    Returns:
        PreflightResult with validation status and any errors/warnings
    """
    result = PreflightResult(is_valid=True)
    resolver = get_provider_resolver()
    cred_resolver = CredentialResolver(auth_policy)

    # Track which providers we need
    required_providers = set()

    for model in models:
        try:
            model_info = resolver.get_model_info(model)
        except KeyError:
            if strict:
                result.errors.append(f"Unknown model: {model}")
                result.is_valid = False
            else:
                result.warnings.append(f"Unknown model: {model}")
            continue
        required_providers.add(model_info.provider)

    # Validate credentials for each provider
    for provider in required_providers:
        try:
            api_key_env = resolver.get_api_key_env(provider)
        except KeyError:
            result.warnings.append(f"No API key mapping for provider: {provider}")
            continue

        if not api_key_env:
            result.warnings.append(f"No API key mapping for provider: {provider}")
            continue

        cred_source = cred_resolver.resolve(api_key_env)
        if cred_source.is_valid:
            result.details[f"{provider}_credential_source"] = cred_source.source.value
            result.details[f"{provider}_credential_location"] = cred_source.location
        else:
            msg = f"Missing credential for {provider}: {api_key_env}"
            if strict:
                result.errors.append(msg)
                result.is_valid = False
            else:
                result.warnings.append(msg)

    return result


def validate_model_provider_config(model: str, provider: str | None = None) -> PreflightResult:
    """Validate that a model/provider combination is properly configured.

    Args:
        model: Model name to validate
        provider: Optional provider hint

    Returns:
        PreflightResult with validation status
    """
    result = PreflightResult(is_valid=True)
    resolver = get_provider_resolver()

    # Check model is known
    try:
        model_info = resolver.get_model_info(model)
    except KeyError:
        result.errors.append(f"Unknown model: {model}")
        result.is_valid = False
        return result

    # Check provider matches
    if provider and model_info.provider != provider:
        result.errors.append(
            f"Model {model} is from provider {model_info.provider}, not {provider}"
        )
        result.is_valid = False

    # Check API key
    api_key_env = resolver.get_api_key_env(model_info.provider)
    if api_key_env:
        cred_resolver = CredentialResolver()
        cred_source = cred_resolver.resolve(api_key_env)
        if not cred_source.is_valid:
            result.errors.append(f"Missing API key: {api_key_env}")
            result.is_valid = False

    return result
