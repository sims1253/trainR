"""Tests for provider environment bootstrap and key alias normalization."""

import os
from unittest.mock import patch

from bench.provider import clear_credential_resolver_cache, get_env_var
from bench.provider.auth import AuthPolicy, CredentialResolver
from bench.provider.env import normalize_api_key_aliases


def test_normalize_alias_promotes_to_canonical() -> None:
    """Alias-only values should populate canonical names."""
    env = {"ZAI_API_KEY": "alias-value"}
    writes = normalize_api_key_aliases(env)

    assert env["Z_AI_API_KEY"] == "alias-value"
    assert env["ZAI_API_KEY"] == "alias-value"
    assert writes["Z_AI_API_KEY"] == "ZAI_API_KEY"


def test_normalize_canonical_overrides_conflicting_alias() -> None:
    """Canonical value should win when canonical and alias disagree."""
    env = {"Z_AI_API_KEY": "canonical", "ZAI_API_KEY": "alias"}
    normalize_api_key_aliases(env)

    assert env["Z_AI_API_KEY"] == "canonical"
    assert env["ZAI_API_KEY"] == "canonical"


def test_normalize_google_alias_promotes_to_gemini() -> None:
    """GOOGLE_API_KEY should populate GEMINI_API_KEY."""
    env = {"GOOGLE_API_KEY": "google-key"}
    normalize_api_key_aliases(env)

    assert env["GEMINI_API_KEY"] == "google-key"
    assert env["GOOGLE_API_KEY"] == "google-key"


def test_normalize_anthropic_auth_alias() -> None:
    """ANTHROPIC_AUTH_TOKEN should populate ANTHROPIC_API_KEY."""
    env = {"ANTHROPIC_AUTH_TOKEN": "anthropic-token"}
    normalize_api_key_aliases(env)

    assert env["ANTHROPIC_API_KEY"] == "anthropic-token"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "anthropic-token"


def test_normalize_opencode_token_alias() -> None:
    """OPENCODE_API_TOKEN should populate OPENCODE_API_KEY."""
    env = {"OPENCODE_API_TOKEN": "opencode-token"}
    normalize_api_key_aliases(env)

    assert env["OPENCODE_API_KEY"] == "opencode-token"
    assert env["OPENCODE_API_TOKEN"] == "opencode-token"


@patch.dict(os.environ, {"ZAI_API_KEY": "alias-value"}, clear=True)
def test_get_env_var_resolves_zai_alias() -> None:
    """Reading canonical key should work when only alias exists."""
    assert get_env_var("Z_AI_API_KEY") == "alias-value"


@patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True)
def test_get_env_var_resolves_google_alias() -> None:
    """Reading GEMINI_API_KEY should work when only GOOGLE_API_KEY exists."""
    assert get_env_var("GEMINI_API_KEY") == "google-key"


@patch.dict(os.environ, {"ZAI_API_KEY": "alias-value"}, clear=True)
def test_credential_resolver_reads_normalized_alias() -> None:
    """Credential resolver should treat alias as valid canonical credential."""
    resolver = CredentialResolver(AuthPolicy.ENV)
    info = resolver.resolve("Z_AI_API_KEY")

    assert info.value == "alias-value"
    assert info.is_valid is True


@patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "anthropic-token"}, clear=True)
def test_credential_resolver_reads_anthropic_auth_alias() -> None:
    """Credential resolver should resolve ANTHROPIC_AUTH_TOKEN alias."""
    resolver = CredentialResolver(AuthPolicy.ENV)
    info = resolver.resolve("ANTHROPIC_API_KEY")

    assert info.value == "anthropic-token"
    assert info.is_valid is True


def test_get_credential_resolver_isolated_per_policy() -> None:
    """Policy-specific resolver singletons should not bleed policy state."""
    from bench.provider.auth import get_credential_resolver

    clear_credential_resolver_cache()
    env_resolver = get_credential_resolver(AuthPolicy.ENV)
    mounted_resolver = get_credential_resolver(AuthPolicy.MOUNTED_FILE)

    assert env_resolver.policy == AuthPolicy.ENV
    assert mounted_resolver.policy == AuthPolicy.MOUNTED_FILE
    assert env_resolver is not mounted_resolver
