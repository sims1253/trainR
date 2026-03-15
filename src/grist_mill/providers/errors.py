"""Provider-specific error types and HTTP error mapping.

Defines the error taxonomy for LLM providers:
- AuthenticationError: 401, 403 (invalid credentials)
- RateLimitError: 429 (rate limited, includes retry-after)
- ServerError: 5xx, connection errors (transient, retriable)
- ProviderError: 4xx other errors (not retriable)

All errors inherit from ProviderError for consistent catching.

Validates VAL-PROVIDER-03: Provider-specific error mapping to error taxonomy.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for all LLM provider failures.

    Attributes:
        status_code: HTTP status code (0 for non-HTTP errors like connection failures).
        transient: Whether the error is transient and may be resolved by retrying.
        message: Human-readable error description.
        provider_hint: Optional hint about which provider caused the error.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        transient: bool = False,
        provider_hint: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.transient = transient
        self.message = message
        self.provider_hint = provider_hint
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Authentication failure (HTTP 401, 403).

    The user's API key is missing, invalid, or expired.
    Not retriable — the user must fix their credentials.
    """

    def __init__(
        self,
        message: str = "Authentication failed. Check your API key.",
        *,
        status_code: int = 401,
        provider_hint: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            transient=False,
            provider_hint=provider_hint,
        )


class RateLimitError(ProviderError):
    """Rate limit exceeded (HTTP 429).

    The user has made too many requests. The ``retry_after`` attribute
    indicates how many seconds to wait before retrying.

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header).
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded.",
        *,
        status_code: int = 429,
        retry_after: int | None = None,
        provider_hint: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message,
            status_code=status_code,
            transient=False,
            provider_hint=provider_hint,
        )


class ServerError(ProviderError):
    """Server-side error (HTTP 5xx, connection errors, timeouts).

    These are transient and may be resolved by retrying with backoff.
    """

    def __init__(
        self,
        message: str = "Server error. This may be transient.",
        *,
        status_code: int = 500,
        provider_hint: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            transient=True,
            provider_hint=provider_hint,
        )


def map_http_error(
    status_code: int,
    body: str,
    headers: dict[str, str] | None = None,
    provider_hint: str | None = None,
) -> ProviderError:
    """Map an HTTP status code and body to a typed provider error.

    Args:
        status_code: HTTP status code (0 for non-HTTP errors).
        body: Response body text.
        headers: Optional response headers (e.g., Retry-After).
        provider_hint: Optional hint about which provider caused the error.

    Returns:
        A ``ProviderError`` subclass instance.
    """
    safe_headers = headers or {}

    if status_code in (401, 403):
        hint = "Check your API key and ensure it is valid."
        if status_code == 403:
            hint = "Your API key may not have permission for this operation."
        return AuthenticationError(
            f"Authentication failed (HTTP {status_code}): {body[:200]}. {hint}",
            status_code=status_code,
            provider_hint=provider_hint,
        )

    if status_code == 429:
        retry_after = safe_headers.get("Retry-After")
        retry_str = ""
        if retry_after is not None:
            try:
                retry_str = f" Retry after {int(retry_after)}s."
            except (ValueError, TypeError):
                retry_str = f" Retry after {retry_after}."
        return RateLimitError(
            f"Rate limit exceeded (HTTP 429): {body[:200]}.{retry_str}",
            status_code=429,
            retry_after=int(retry_after) if retry_after is not None else None,
            provider_hint=provider_hint,
        )

    if status_code >= 500 or status_code == 0:
        if status_code == 0:
            return ServerError(
                f"Connection error: {body[:200]}. This is typically a transient network issue.",
                status_code=0,
                provider_hint=provider_hint,
            )
        return ServerError(
            f"Server error (HTTP {status_code}): {body[:200]}. "
            "This is a transient error that may resolve on retry.",
            status_code=status_code,
            provider_hint=provider_hint,
        )

    # All other 4xx errors
    return ProviderError(
        f"Provider error (HTTP {status_code}): {body[:200]}.",
        status_code=status_code,
        transient=False,
        provider_hint=provider_hint,
    )


__all__ = [
    "AuthenticationError",
    "ProviderError",
    "RateLimitError",
    "ServerError",
    "map_http_error",
]
