"""Language-to-Docker-image mapping for multi-language environment support.

Provides configurable Docker image selection based on task language.
Each supported language maps to a default image, with support for
custom overrides and fallback images.

Validates VAL-ENV-01: Multi-language environment support via configurable Docker images.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default language-to-image mapping
# ---------------------------------------------------------------------------

DEFAULT_LANGUAGE_IMAGES: dict[str, str] = {
    "python": "python:3.12-slim",
    "r": "rocker/r-ver:latest",
    "typescript": "node:22-slim",
}

DEFAULT_FALLBACK_IMAGE: str = "python:3.12-slim"


class LanguageImageConfig(BaseModel):
    """Maps programming languages to Docker images.

    Supports custom overrides for any language and a configurable
    fallback image for unknown languages.  Lookups are case-insensitive.

    Args:
        overrides: Custom language-to-image mappings that override defaults.
        fallback_image: Image to use for unknown languages (default ``python:3.12-slim``).

    Example::

        config = LanguageImageConfig(
            overrides={"python": "my-python:3.11"},
            fallback_image="ubuntu:22.04",
        )
        image = config.get_image("python")  # "my-python:3.11"
        image = config.get_image("unknown")  # "ubuntu:22.04"
    """

    model_config = {"str_strip_whitespace": True}

    overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Custom language-to-image mappings that override defaults.",
    )
    fallback_image: str = Field(
        default=DEFAULT_FALLBACK_IMAGE,
        min_length=1,
        description="Image to use for unknown languages.",
    )

    def get_image(self, language: str) -> str:
        """Resolve the Docker image for a given language.

        Lookup order:
        1. Custom overrides (case-insensitive key match)
        2. Default language mapping (case-insensitive key match)
        3. Fallback image

        Args:
            language: Programming language identifier (e.g., ``"python"``, ``"r"``).

        Returns:
            The Docker image name to use for that language.
        """
        lang_lower = language.lower()

        # 1. Check custom overrides
        for key, image in self.overrides.items():
            if key.lower() == lang_lower:
                logger.debug("Language '%s' resolved to '%s' (custom override)", language, image)
                return image

        # 2. Check default mapping
        for key, image in DEFAULT_LANGUAGE_IMAGES.items():
            if key.lower() == lang_lower:
                logger.debug("Language '%s' resolved to '%s' (default mapping)", language, image)
                return image

        # 3. Fallback
        logger.debug("Language '%s' not found, using fallback '%s'", language, self.fallback_image)
        return self.fallback_image

    def __repr__(self) -> str:
        return (
            f"LanguageImageConfig(overrides={self.overrides!r}, "
            f"fallback_image={self.fallback_image!r})"
        )


__all__ = [
    "DEFAULT_FALLBACK_IMAGE",
    "DEFAULT_LANGUAGE_IMAGES",
    "LanguageImageConfig",
]
