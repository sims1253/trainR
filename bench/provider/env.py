"""Environment bootstrap and API key normalization.

This module provides a single environment initialization path for:
- Loading a project-local `.env` file
- Normalizing API key aliases across providers
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

# Canonical API key names and accepted aliases.
# Canonical names are used across trainR internals.
API_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": ("ANTHROPIC_AUTH_TOKEN",),
    "GEMINI_API_KEY": ("GOOGLE_API_KEY",),
    "OPENCODE_API_KEY": ("OPENCODE_API_TOKEN",),
    "Z_AI_API_KEY": ("ZAI_API_KEY",),
}

_DOTENV_INIT_LOCK = threading.Lock()
_DOTENV_INITIALIZED = False


def _repo_root() -> Path:
    """Get repository root from this module location."""
    return Path(__file__).resolve().parents[2]


def _as_bool(raw: str | None, default: bool = True) -> bool:
    """Parse a truthy/falsy string value."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_load_dotenv() -> bool:
    """Determine whether `.env` should be loaded."""
    # Keep test execution hermetic unless explicitly opted in.
    if "pytest" in sys.modules and os.environ.get("TRAINR_LOAD_DOTENV") is None:
        return False
    return _as_bool(os.environ.get("TRAINR_LOAD_DOTENV"), default=True)


def _get_dotenv_path() -> Path:
    """Resolve dotenv path from env override or project root."""
    configured = os.environ.get("TRAINR_DOTENV_PATH")
    if configured:
        return Path(configured).expanduser()
    return _repo_root() / ".env"


def _ensure_dotenv_loaded() -> None:
    """Load `.env` once per process."""
    global _DOTENV_INITIALIZED
    if _DOTENV_INITIALIZED:
        return

    with _DOTENV_INIT_LOCK:
        if _DOTENV_INITIALIZED:
            return

        if _should_load_dotenv():
            dotenv_path = _get_dotenv_path()
            if dotenv_path.exists():
                # Shell/CI environment wins over .env defaults.
                load_dotenv(dotenv_path=dotenv_path, override=False)

        _DOTENV_INITIALIZED = True


def normalize_api_key_aliases(env: dict[str, str] | None = None) -> dict[str, str]:
    """Normalize API key aliases into canonical + alias names.

    Resolution order for each canonical name:
    1. Canonical variable value (if set)
    2. First non-empty alias value

    The resolved value is then written back to canonical and all aliases to
    keep downstream tools consistent.

    Args:
        env: Optional mutable environment mapping. Defaults to ``os.environ``.

    Returns:
        Mapping of environment variable names that were written to their source
        variable name.
    """
    target_env = os.environ if env is None else env
    writes: dict[str, str] = {}

    for canonical, aliases in API_KEY_ALIASES.items():
        candidates = [canonical, *aliases]
        source: str | None = None
        resolved: str | None = None

        for candidate in candidates:
            value = target_env.get(candidate)
            if value:
                source = candidate
                resolved = value
                break

        if not resolved or not source:
            continue

        for candidate in candidates:
            current = target_env.get(candidate)
            if current != resolved:
                target_env[candidate] = resolved
                writes[candidate] = source

    return writes


def bootstrap_environment() -> None:
    """Initialize environment defaults and normalize key aliases."""
    _ensure_dotenv_loaded()
    normalize_api_key_aliases()


def get_env_var(name: str, default: str | None = None) -> str | None:
    """Get an environment variable after env bootstrap."""
    bootstrap_environment()
    return os.environ.get(name, default)


def api_key_aliases() -> dict[str, tuple[str, ...]]:
    """Get configured API key alias mapping."""
    return API_KEY_ALIASES
