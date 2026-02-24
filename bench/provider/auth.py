"""Authentication and credential management."""

import os
from dataclasses import dataclass
from enum import Enum

from .models import CredentialSource


class AuthPolicy(Enum):
    """Policy for how credentials should be loaded."""

    ENV = "env"
    """Load credentials from environment variables."""

    MOUNTED_FILE = "mounted_file"
    """Load credentials from mounted files (e.g., Kubernetes secrets)."""


@dataclass
class CredentialInfo:
    """Information about a resolved credential."""

    source: CredentialSource
    """Where the credential came from."""

    location: str
    """Specific location (env var name or file path)."""

    value: str | None
    """The credential value (may be None if not set)."""

    is_valid: bool
    """Whether the credential is valid (non-empty)."""

    is_redacted: bool = False
    """Whether the value has been redacted."""

    def redacted(self) -> "CredentialInfo":
        """Return a copy with the value redacted."""
        if self.value is None:
            return CredentialInfo(
                source=self.source,
                location=self.location,
                value=None,
                is_valid=self.is_valid,
                is_redacted=True,
            )
        return CredentialInfo(
            source=self.source,
            location=self.location,
            value=self._redact_value(self.value),
            is_valid=self.is_valid,
            is_redacted=True,
        )

    @staticmethod
    def _redact_value(value: str) -> str:
        """Redact a credential value for safe logging."""
        if len(value) <= 8:
            return "*" * len(value)
        # Show first 4 and last 4 characters
        return f"{value[:4]}...{value[-4:]}"


class CredentialResolver:
    """Resolves and validates credentials for providers.

    This class handles:
    - Loading credentials based on auth policy
    - Redacting sensitive values for logging
    - Tracking credential sources
    """

    def __init__(self, policy: AuthPolicy = AuthPolicy.ENV) -> None:
        """Initialize the credential resolver.

        Args:
            policy: The authentication policy to use.
        """
        self._policy = policy
        self._cache: dict[str, CredentialInfo] = {}

    @property
    def policy(self) -> AuthPolicy:
        """Get the current auth policy."""
        return self._policy

    def resolve(self, env_var: str) -> CredentialInfo:
        """Resolve a credential by environment variable name.

        Args:
            env_var: The environment variable name.

        Returns:
            CredentialInfo with source, location, value, and validity.
        """
        # Check cache first
        cache_key = f"{self._policy.value}:{env_var}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._policy == AuthPolicy.ENV:
            info = self._resolve_from_env(env_var)
        elif self._policy == AuthPolicy.MOUNTED_FILE:
            info = self._resolve_from_mounted_file(env_var)
        else:
            info = CredentialInfo(
                source=CredentialSource.NONE,
                location=env_var,
                value=None,
                is_valid=False,
            )

        self._cache[cache_key] = info
        return info

    def _resolve_from_env(self, env_var: str) -> CredentialInfo:
        """Resolve credential from environment variable."""
        value = os.environ.get(env_var)
        return CredentialInfo(
            source=CredentialSource.ENV,
            location=env_var,
            value=value,
            is_valid=value is not None and len(value) > 0,
        )

    def _resolve_from_mounted_file(self, env_var: str) -> CredentialInfo:
        """Resolve credential from mounted file.

        Looks for a file at /run/secrets/{lowercase_env_var_name}
        """
        # Convert env var name to expected file path
        # e.g., OPENROUTER_API_KEY -> /run/secrets/openrouter_api_key
        file_name = env_var.lower()
        file_path = f"/run/secrets/{file_name}"

        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    value = f.read().strip()
                return CredentialInfo(
                    source=CredentialSource.MOUNTED_FILE,
                    location=file_path,
                    value=value,
                    is_valid=len(value) > 0,
                )
            except OSError:
                pass

        # Fall back to environment variable
        return self._resolve_from_env(env_var)

    def get(self, env_var: str) -> str | None:
        """Get the credential value.

        Args:
            env_var: The environment variable name.

        Returns:
            The credential value, or None if not set.
        """
        return self.resolve(env_var).value

    def is_valid(self, env_var: str) -> bool:
        """Check if a credential is valid.

        Args:
            env_var: The environment variable name.

        Returns:
            True if the credential is set and non-empty.
        """
        return self.resolve(env_var).is_valid

    def clear_cache(self) -> None:
        """Clear the credential cache."""
        self._cache.clear()


# Module-level singleton
_resolver_instance: CredentialResolver | None = None


def get_credential_resolver(policy: AuthPolicy = AuthPolicy.ENV) -> CredentialResolver:
    """Get the singleton CredentialResolver instance.

    Args:
        policy: The auth policy to use. Only used on first call.

    Returns:
        The CredentialResolver singleton.
    """
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = CredentialResolver(policy)
    return _resolver_instance


def get_credentials(env_var: str, policy: AuthPolicy = AuthPolicy.ENV) -> str | None:
    """Get credential value by environment variable name.

    Args:
        env_var: The environment variable name.
        policy: The auth policy to use.

    Returns:
        The credential value, or None if not set.
    """
    return get_credential_resolver(policy).get(env_var)


def validate_credentials(
    env_vars: str | list[str], policy: AuthPolicy = AuthPolicy.ENV
) -> dict[str, bool]:
    """Validate that credentials are set.

    Args:
        env_vars: Environment variable name(s) to check.
        policy: The auth policy to use.

    Returns:
        Dictionary mapping env var names to validity.
    """
    resolver = get_credential_resolver(policy)
    if isinstance(env_vars, str):
        env_vars = [env_vars]
    return {var: resolver.is_valid(var) for var in env_vars}
