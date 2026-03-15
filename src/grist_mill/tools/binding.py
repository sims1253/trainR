"""Artifact-to-tool binding layer.

Wires registered artifacts into the agent's execution context:

- **SkillArtifact**: SKILL.md file content is read and injected into the
  agent's system prompt / context.
- **MCPServerArtifact**: The server is started as a subprocess via
  ``MCPServerManager``, and the transport URL (``stdio://<name>``) is
  injected into the tool configuration.
- **ToolArtifact**: The declared command is verified as available in the
  execution environment (via ``shutil.which``), and the tool definition
  is registered in the ``ToolRegistry``.

All three artifact types coexist without naming conflicts (they live in
separate namespaces: *skills*, *mcp_servers*, *tools*).

Artifact validation happens during ``setup()`` — **before** the agent
runs — so missing files or unavailable commands produce early, actionable
errors.

All resources (subprocesses, temp files) are cleaned up in ``cleanup()``,
even on failure or timeout.

Validates:
- VAL-BIND-01: SKILL.md files injected as agent context
- VAL-BIND-02: MCP server config translated to running server
- VAL-BIND-03: CLI tool definitions verified as available commands
- VAL-BIND-04: Multiple artifact types coexist without conflict
- VAL-BIND-05: Artifact validation fails early with actionable errors
- VAL-BIND-06: Artifact cleanup happens after execution regardless of outcome
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from grist_mill.registry import ArtifactRegistry
from grist_mill.schemas.artifact import (
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)
from grist_mill.tools.exceptions import MCPServerError
from grist_mill.tools.mcp import MCPServerManager
from grist_mill.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ArtifactBindingError(Exception):
    """Raised when artifact binding fails during setup.

    Setup-time failures produce this error **before** the agent runs,
    ensuring early detection of missing files, unavailable commands,
    or misconfigured artifacts.
    """

    def __init__(self, message: str, *, artifact_name: str | None = None) -> None:
        self.artifact_name = artifact_name
        super().__init__(message)


class ArtifactValidationError(ArtifactBindingError):
    """Raised when an artifact fails validation (missing file, unavailable command)."""

    pass


# ---------------------------------------------------------------------------
# ArtifactBinder
# ---------------------------------------------------------------------------


class ArtifactBinder:
    """Binds artifacts from the registry into the agent's execution context.

    Usage (context manager — recommended)::

        registry = ArtifactRegistry()
        registry.register(skill_artifact)
        registry.register(mcp_artifact)
        registry.register(tool_artifact)

        binder = ArtifactBinder(registry=registry, tool_registry=tool_registry)
        with binder:
            context = binder.build_agent_context()
            # ... run agent with context ...
        # All MCP servers stopped, resources cleaned up

    Usage (manual)::

        binder = ArtifactBinder(registry=registry, tool_registry=tool_registry)
        binder.setup()
        try:
            context = binder.build_agent_context()
            agent.run(task, config)
        finally:
            binder.cleanup()

    The ``setup()`` phase:
    1. Validates all bound artifacts (files exist, commands available)
    2. Reads SKILL.md content for injection
    3. Starts MCP server subprocesses
    4. Registers ToolArtifact commands in the ToolRegistry

    The ``cleanup()`` phase:
    1. Stops all MCP server subprocesses
    2. Deregisters tools that were added by this binder
    """

    def __init__(
        self,
        *,
        registry: ArtifactRegistry,
        tool_registry: ToolRegistry | None = None,
        artifact_names: list[str] | None = None,
    ) -> None:
        """Initialize the binder.

        Args:
            registry: The artifact registry containing bound artifacts.
            tool_registry: Optional ToolRegistry to register tool artifacts into.
                If ``None``, tool artifacts are validated but not registered.
            artifact_names: Specific artifact names to bind. ``None`` means all.
        """
        self._registry = registry
        self._tool_registry = tool_registry
        self._artifact_names = artifact_names

        # State tracking
        self._setup_complete = False
        self._mcp_manager = MCPServerManager()
        self._skill_contents: dict[str, str] = {}
        self._mcp_urls: dict[str, str] = {}
        self._registered_tool_names: list[str] = []

        # Resolved artifacts (populated during setup)
        self._tools: list[ToolArtifact] = []
        self._mcp_servers: list[MCPServerArtifact] = []
        self._skills: list[SkillArtifact] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def skill_contents(self) -> dict[str, str]:
        """Mapping of skill name → SKILL.md file content."""
        return dict(self._skill_contents)

    @property
    def mcp_urls(self) -> dict[str, str]:
        """Mapping of MCP server name → transport URL."""
        return dict(self._mcp_urls)

    @property
    def is_setup(self) -> bool:
        """Whether setup has been completed."""
        return self._setup_complete

    @property
    def registered_tool_names(self) -> list[str]:
        """Names of tools registered in the ToolRegistry by this binder."""
        return list(self._registered_tool_names)

    @property
    def mcp_manager(self) -> MCPServerManager:
        """The underlying MCP server manager (for inspecting running servers)."""
        return self._mcp_manager

    # ------------------------------------------------------------------
    # Setup (VAL-BIND-05: early validation)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Validate and bind all artifacts.

        This method:
        1. Resolves artifacts from the registry
        2. Validates each artifact (files exist, commands available)
        3. Reads SKILL.md content
        4. Starts MCP servers as subprocesses
        5. Registers tool commands in the ToolRegistry

        Raises:
            ArtifactValidationError: If any artifact fails validation.
            MCPServerError: If an MCP server fails to start.
            ValueError: If referenced artifacts don't exist in the registry.
        """
        if self._setup_complete:
            logger.debug("ArtifactBinder setup already complete, skipping")
            return

        # Resolve artifacts
        self._resolve_artifacts()

        # Validate and bind each type
        self._validate_and_bind_tools()
        self._validate_and_bind_skills()
        self._validate_and_bind_mcp_servers()

        self._setup_complete = True
        logger.info(
            "Artifact binding complete: %d tools, %d MCP servers, %d skills",
            len(self._tools),
            len(self._mcp_servers),
            len(self._skills),
        )

    def _resolve_artifacts(self) -> None:
        """Resolve artifact objects from the registry by name."""
        if self._artifact_names is None:
            # Bind all registered artifacts
            all_artifacts = self._registry.list_artifacts()
        else:
            all_artifacts = []
            for name in self._artifact_names:
                artifact = self._registry.get(name)
                if artifact is None:
                    msg = (
                        f"Artifact '{name}' referenced in binding but not found "
                        f"in registry. Register the artifact before binding."
                    )
                    raise ValueError(msg)
                all_artifacts.append(artifact)

        for artifact in all_artifacts:
            if isinstance(artifact, ToolArtifact):
                self._tools.append(artifact)
            elif isinstance(artifact, MCPServerArtifact):
                self._mcp_servers.append(artifact)
            elif isinstance(artifact, SkillArtifact):
                self._skills.append(artifact)

    def _validate_and_bind_tools(self) -> None:
        """Validate ToolArtifacts: verify commands are available."""
        for tool_artifact in self._tools:
            # Check for naming conflicts
            if (
                self._tool_registry is not None
                and self._tool_registry.has(tool_artifact.name)
                and tool_artifact.name not in self._registered_tool_names
            ):
                msg = (
                    f"Tool artifact '{tool_artifact.name}' conflicts with an "
                    f"existing tool in the ToolRegistry. Use a different name or "
                    f"set overwrite=True when registering."
                )
                raise ArtifactValidationError(msg, artifact_name=tool_artifact.name)

            # Validate command availability
            if tool_artifact.command is not None:
                cmd_binary = (
                    tool_artifact.command.split()[0]
                    if " " in tool_artifact.command
                    else tool_artifact.command
                )
                if shutil.which(cmd_binary) is None:
                    msg = (
                        f"Tool artifact '{tool_artifact.name}' references command "
                        f"'{cmd_binary}' which is not available in the current "
                        f"environment PATH. Install the command or update the artifact."
                    )
                    raise ArtifactValidationError(msg, artifact_name=tool_artifact.name)

            # Register in tool registry
            if self._tool_registry is not None:
                handler = _create_tool_handler(tool_artifact)
                self._tool_registry.register_from_artifact(
                    artifact=tool_artifact,
                    handler=handler,
                    overwrite=False,
                )
                self._registered_tool_names.append(tool_artifact.name)
                logger.info(
                    "Bound tool artifact '%s' to ToolRegistry",
                    tool_artifact.name,
                )

    def _validate_and_bind_skills(self) -> None:
        """Validate SkillArtifacts: verify files exist and read content."""
        for skill_artifact in self._skills:
            skill_path = Path(skill_artifact.skill_file_path)

            if not skill_path.exists():
                msg = (
                    f"Skill artifact '{skill_artifact.name}' references file "
                    f"'{skill_artifact.skill_file_path}' which does not exist. "
                    f"Create the file or update the artifact's skill_file_path."
                )
                raise ArtifactValidationError(msg, artifact_name=skill_artifact.name)

            if not skill_path.is_file():
                msg = (
                    f"Skill artifact '{skill_artifact.name}' references "
                    f"'{skill_artifact.skill_file_path}' which is not a file. "
                    f"Ensure the path points to a regular file."
                )
                raise ArtifactValidationError(msg, artifact_name=skill_artifact.name)

            try:
                content = skill_path.read_text(encoding="utf-8")
            except OSError as exc:
                msg = (
                    f"Skill artifact '{skill_artifact.name}' could not read file "
                    f"'{skill_artifact.skill_file_path}': {exc}. "
                    f"Ensure the file is readable."
                )
                raise ArtifactValidationError(msg, artifact_name=skill_artifact.name) from exc

            self._skill_contents[skill_artifact.name] = content
            logger.info(
                "Bound skill artifact '%s' (%d chars from %s)",
                skill_artifact.name,
                len(content),
                skill_artifact.skill_file_path,
            )

    def _validate_and_bind_mcp_servers(self) -> None:
        """Validate and start MCPServerArtifacts as subprocesses."""
        for mcp_artifact in self._mcp_servers:
            # Verify command is available
            cmd_binary = (
                mcp_artifact.command.split()[0]
                if " " in mcp_artifact.command
                else mcp_artifact.command
            )
            if shutil.which(cmd_binary) is None:
                msg = (
                    f"MCP server artifact '{mcp_artifact.name}' references command "
                    f"'{cmd_binary}' which is not available in the current "
                    f"environment PATH. Install the command or update the artifact."
                )
                raise ArtifactValidationError(msg, artifact_name=mcp_artifact.name)

            # Start the server subprocess
            try:
                self._mcp_manager.start(mcp_artifact)
            except MCPServerError:
                raise
            except Exception as exc:
                msg = (
                    f"Failed to start MCP server '{mcp_artifact.name}': {exc}. "
                    f"Check the command and arguments."
                )
                raise ArtifactBindingError(msg, artifact_name=mcp_artifact.name) from exc

            # Build the transport URL (stdio://<name>)
            url = f"stdio://{mcp_artifact.name}"
            self._mcp_urls[mcp_artifact.name] = url
            server_info = self._mcp_manager.get_server(mcp_artifact.name)
            server_pid = server_info.pid if server_info is not None else None
            logger.info(
                "Bound MCP server artifact '%s' → %s (pid=%s)",
                mcp_artifact.name,
                url,
                server_pid,
            )

    # ------------------------------------------------------------------
    # Build agent context (VAL-BIND-01, VAL-BIND-02, VAL-BIND-04)
    # ------------------------------------------------------------------

    def build_agent_context(self) -> dict[str, Any]:
        """Build the complete agent context from all bound artifacts.

        Returns a dict with:
        - ``"skills"``: list of ``{"name": ..., "content": ...}`` dicts
        - ``"mcp_servers"``: list of ``{"name": ..., "url": ...}`` dicts
        - ``"tools"``: list of tool capability dicts from the ToolRegistry

        Raises:
            ArtifactBindingError: If setup has not been called.
        """
        if not self._setup_complete:
            msg = "ArtifactBinder.setup() must be called before build_agent_context()."
            raise ArtifactBindingError(msg)

        skills: list[dict[str, str]] = []
        for name, content in self._skill_contents.items():
            skills.append({"name": name, "content": content})

        mcp_servers: list[dict[str, str]] = []
        for name, url in self._mcp_urls.items():
            mcp_servers.append({"name": name, "url": url})

        tools: list[dict[str, Any]] = []
        if self._tool_registry is not None:
            # Only include tools registered by this binder
            for tool_name in self._registered_tool_names:
                cap = self._tool_registry.get_capability(tool_name)
                if cap is not None:
                    tools.append(cap)

        return {
            "skills": skills,
            "mcp_servers": mcp_servers,
            "tools": tools,
        }

    def build_system_prompt_extension(self) -> str:
        """Build skill content as a system prompt extension.

        Concatenates all skill contents into a single string suitable
        for appending to the agent's system prompt.

        Returns:
            A string containing all skill contents, separated by markers.

        Raises:
            ArtifactBindingError: If setup has not been called.
        """
        if not self._setup_complete:
            msg = "ArtifactBinder.setup() must be called before build_system_prompt_extension()."
            raise ArtifactBindingError(msg)

        if not self._skill_contents:
            return ""

        parts: list[str] = []
        for name, content in self._skill_contents.items():
            parts.append(f"## Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Cleanup (VAL-BIND-06: cleanup regardless of outcome)
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up all artifact resources.

        Stops all MCP server subprocesses and deregisters tools that
        were added by this binder. Safe to call multiple times or even
        if setup was never completed.
        """
        # Stop MCP servers
        if self._mcp_manager.server_count > 0:
            self._mcp_manager.stop_all()
            logger.info("Stopped all MCP servers managed by ArtifactBinder")

        # Deregister tools
        if self._tool_registry is not None:
            for tool_name in self._registered_tool_names:
                try:
                    self._tool_registry.deregister(tool_name)
                    logger.debug("Deregistered tool '%s' from ToolRegistry", tool_name)
                except Exception:
                    logger.debug(
                        "Tool '%s' was already deregistered or not found, skipping",
                        tool_name,
                    )
        self._registered_tool_names.clear()

        # Clear state
        self._skill_contents.clear()
        self._mcp_urls.clear()
        self._tools.clear()
        self._mcp_servers.clear()
        self._skills.clear()
        self._setup_complete = False

        logger.info("ArtifactBinder cleanup complete")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> ArtifactBinder:
        """Enter the context manager, calling setup()."""
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager, calling cleanup() regardless of outcome."""
        self.cleanup()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_tool_handler(tool_artifact: ToolArtifact) -> Any:
    """Create a handler function for a ToolArtifact.

    If the artifact has a ``command``, the handler executes that command
    via subprocess. Otherwise, returns a placeholder handler.

    Args:
        tool_artifact: The tool artifact to create a handler for.

    Returns:
        A callable handler.
    """
    import subprocess

    if tool_artifact.command is None:

        def placeholder_handler(**kwargs: Any) -> dict[str, Any]:
            return {
                "result": f"Tool '{tool_artifact.name}' executed (no command configured)",
                "arguments": kwargs,
            }

        return placeholder_handler

    tool_command: str = tool_artifact.command  # guarded by None check above

    def command_handler(**kwargs: Any) -> dict[str, Any]:
        try:
            cmd_parts: list[str] = [tool_command]
            # Simple argument passing — could be enhanced
            for key, value in kwargs.items():
                cmd_parts.extend([f"--{key}", str(value)])

            proc = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30.0,
                encoding="utf-8",
                errors="replace",
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout.decode() if exc.stdout else "",
                "stderr": exc.stderr.decode() if exc.stderr else "",
                "exit_code": -1,
                "error": "Command timed out",
            }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -1,
                "error": f"Command execution failed: {exc}",
            }

    return command_handler


__all__ = [
    "ArtifactBinder",
    "ArtifactBindingError",
    "ArtifactValidationError",
]
