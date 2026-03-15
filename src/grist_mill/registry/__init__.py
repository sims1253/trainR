"""Artifact registry and management.

Provides the central ``ArtifactRegistry`` for dynamic registration,
lookup, filtering, and wiring of artifacts into harness configurations.

Validates VAL-FOUND-10 and VAL-FOUND-11.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from grist_mill.schemas.artifact import (
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)

if TYPE_CHECKING:
    from grist_mill.schemas import HarnessConfig

logger = logging.getLogger(__name__)


class ArtifactRegistry:
    """Central registry for managing pluggable artifacts.

    Supports dynamic registration, lookup by name or type, duplicate
    detection, and automatic wiring into ``HarnessConfig`` and agent
    execution context.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, ToolArtifact | MCPServerArtifact | SkillArtifact] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        artifact: ToolArtifact | MCPServerArtifact | SkillArtifact,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an artifact in the registry.

        Args:
            artifact: The artifact to register.
            overwrite: If True, replace an existing artifact with the same name.
                If False (default), raise ``ValueError`` on duplicate names.

        Raises:
            ValueError: If an artifact with the same name is already registered
                and ``overwrite`` is False.
        """
        if artifact.name in self._artifacts and not overwrite:
            msg = (
                f"Artifact '{artifact.name}' is already registered. "
                f"Use overwrite=True to replace it."
            )
            raise ValueError(msg)

        if artifact.name in self._artifacts:
            logger.info("Overwriting artifact '%s'", artifact.name)
        else:
            logger.info("Registering artifact '%s' (type=%s)", artifact.name, artifact.type)

        self._artifacts[artifact.name] = artifact

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolArtifact | MCPServerArtifact | SkillArtifact | None:
        """Retrieve an artifact by name.

        Args:
            name: The unique name of the artifact.

        Returns:
            The artifact if found, or ``None`` if not registered.
        """
        return self._artifacts.get(name)

    def has(self, name: str) -> bool:
        """Check whether an artifact with the given name is registered.

        Args:
            name: The artifact name to check.

        Returns:
            ``True`` if the artifact is registered, ``False`` otherwise.
        """
        return name in self._artifacts

    def get_by_type(
        self, artifact_type: str
    ) -> list[ToolArtifact | MCPServerArtifact | SkillArtifact]:
        """Retrieve all artifacts of a specific type.

        Args:
            artifact_type: The type discriminator (``"tool"``, ``"mcp_server"``, ``"skill"``).

        Returns:
            A list of matching artifacts. Empty list if none match.
        """
        return [artifact for artifact in self._artifacts.values() if artifact.type == artifact_type]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_artifacts(
        self, *, filter_type: str | None = None
    ) -> list[ToolArtifact | MCPServerArtifact | SkillArtifact]:
        """List registered artifacts, optionally filtered by type.

        Args:
            filter_type: If provided, return only artifacts of this type.

        Returns:
            A list of artifacts matching the filter (or all if no filter).
        """
        if filter_type is None:
            return list(self._artifacts.values())
        return self.get_by_type(filter_type)

    @property
    def count(self) -> int:
        """Number of registered artifacts."""
        return len(self._artifacts)

    def count_by_type(self, artifact_type: str) -> int:
        """Count artifacts of a specific type.

        Args:
            artifact_type: The type discriminator to count.

        Returns:
            The number of artifacts of the given type.
        """
        return len(self.get_by_type(artifact_type))

    @property
    def names(self) -> list[str]:
        """List of all registered artifact names."""
        return list(self._artifacts.keys())

    # ------------------------------------------------------------------
    # Deregistration
    # ------------------------------------------------------------------

    def deregister(self, name: str, *, ignore_missing: bool = False) -> None:
        """Remove an artifact from the registry by name.

        Args:
            name: The name of the artifact to remove.
            ignore_missing: If True, do not raise when the artifact is not found.

        Raises:
            KeyError: If the artifact is not registered and ``ignore_missing`` is False.
        """
        if name not in self._artifacts:
            if ignore_missing:
                return
            msg = f"Artifact '{name}' is not registered."
            raise KeyError(msg)

        del self._artifacts[name]
        logger.info("Deregistered artifact '%s'", name)

    def clear(self) -> None:
        """Remove all artifacts from the registry."""
        self._artifacts.clear()
        logger.info("Cleared all artifacts from registry")

    # ------------------------------------------------------------------
    # Wiring into HarnessConfig (VAL-FOUND-11)
    # ------------------------------------------------------------------

    def build_harness_config(
        self,
        *,
        model: str,
        provider: str,
        runner_type: str,
        artifact_names: list[str] | None = None,
        system_prompt: str | None = None,
        docker_image: str | None = None,
    ) -> HarnessConfig:
        """Build a ``HarnessConfig`` with artifacts wired into artifact_bindings.

        If ``artifact_names`` is ``None``, all registered artifacts are wired.
        If ``artifact_names`` is provided, only those artifacts are wired
        (after validating they exist in the registry).

        Args:
            model: The LLM model identifier.
            provider: The LLM provider name.
            runner_type: The environment runner type (``"local"`` or ``"docker"``).
            artifact_names: Optional list of artifact names to wire. ``None`` means all.
            system_prompt: Optional system prompt for the agent.
            docker_image: Optional Docker image for the environment.

        Returns:
            A ``HarnessConfig`` with the specified artifacts wired.

        Raises:
            ValueError: If any specified artifact name is not found in the registry.
        """
        from grist_mill.schemas import AgentConfig, EnvironmentConfig, HarnessConfig

        names_to_wire: list[str]
        if artifact_names is None:
            names_to_wire = self.names
        else:
            # Validate that all requested artifacts exist
            missing = [n for n in artifact_names if n not in self._artifacts]
            if missing:
                msg = f"Artifact(s) not found in registry: {', '.join(missing)}"
                raise ValueError(msg)
            names_to_wire = list(artifact_names)

        agent_config = AgentConfig(
            model=model,
            provider=provider,
            system_prompt=system_prompt,
        )
        env_config = EnvironmentConfig(
            runner_type=runner_type,
            docker_image=docker_image,
        )

        return HarnessConfig(
            agent=agent_config,
            environment=env_config,
            artifact_bindings=names_to_wire,
        )

    def get_agent_context(
        self, artifact_names: list[str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a structured context of artifacts for agent consumption.

        Returns a dict with keys ``"tools"``, ``"mcp_servers"``, and ``"skills"``,
        each containing a list of artifact data dicts suitable for injection
        into the agent's execution context.

        Args:
            artifact_names: Optional list of artifact names to include. ``None`` means all.

        Returns:
            A dict with categorized artifact data for the agent.

        Raises:
            ValueError: If any specified artifact name is not found in the registry.
        """
        if artifact_names is None:
            artifacts = list(self._artifacts.values())
        else:
            missing = [n for n in artifact_names if n not in self._artifacts]
            if missing:
                msg = f"Artifact(s) not found in registry: {', '.join(missing)}"
                raise ValueError(msg)
            artifacts = [self._artifacts[n] for n in artifact_names]

        tools: list[dict[str, Any]] = []
        mcp_servers: list[dict[str, Any]] = []
        skills: list[dict[str, Any]] = []

        for artifact in artifacts:
            data = artifact.model_dump()
            if isinstance(artifact, ToolArtifact):
                tools.append(data)
            elif isinstance(artifact, MCPServerArtifact):
                mcp_servers.append(data)
            elif isinstance(artifact, SkillArtifact):
                skills.append(data)

        return {
            "tools": tools,
            "mcp_servers": mcp_servers,
            "skills": skills,
        }


__all__ = ["ArtifactRegistry"]
