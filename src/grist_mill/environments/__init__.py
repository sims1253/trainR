"""Environments module — execution environment implementations.

Provides runners for executing commands in different environments:
- LocalRunner: Host subprocess execution (fast dev iteration)
- DockerRunner: Containerized execution with resource limits

Also provides:
- LanguageImageConfig: Multi-language Docker image mapping
- EnvironmentSetupError: Raised when environment setup fails
"""

from grist_mill.environments.docker_runner import (
    DockerDaemonError,
    DockerImageError,
    DockerRunner,
    EnvironmentSetupError,
)
from grist_mill.environments.language_config import (
    DEFAULT_FALLBACK_IMAGE,
    DEFAULT_LANGUAGE_IMAGES,
    LanguageImageConfig,
)
from grist_mill.environments.local_runner import LocalRunner

__all__ = [
    "DEFAULT_FALLBACK_IMAGE",
    "DEFAULT_LANGUAGE_IMAGES",
    "DockerDaemonError",
    "DockerImageError",
    "DockerRunner",
    "EnvironmentSetupError",
    "LanguageImageConfig",
    "LocalRunner",
]
