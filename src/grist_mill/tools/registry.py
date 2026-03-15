"""Tool registry with capability advertisement and dynamic registration.

Provides:
- ToolRegistry: Central registry for managing callable tools.
  Supports dynamic registration, capability advertisement for agent
  discovery, and lookup by name.

Validates:
- VAL-TOOL-01: Tool registry supports capability advertisement
- VAL-TOOL-02: Tool registry supports dynamic registration at runtime
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from grist_mill.schemas.artifact import ToolArtifact
from grist_mill.tools.exceptions import ToolNotFoundError, ToolRegistrationError
from grist_mill.tools.models import ToolDefinition

logger = logging.getLogger(__name__)

# Type alias for tool handler callables
ToolHandler = Callable[..., dict[str, Any]]

# Default timeout for tools that don't specify one (in seconds)
DEFAULT_TOOL_TIMEOUT: float = 30.0


class ToolRegistry:
    """Central registry for managing callable tools.

    Supports dynamic registration at any point before execution,
    capability advertisement for agent discovery, and lookup by name.
    Tools can be registered directly or from ``ToolArtifact`` schema objects.

    Usage::

        registry = ToolRegistry()
        registry.register(
            name="search",
            description="Search the web",
            input_schema={"type": "object", ...},
            handler=lambda query: {"results": [...]},
        )

        # Agent discovers available tools
        capabilities = registry.get_capabilities()
        tool_defs = registry.get_tool_definitions_for_agent()

        # Individual tool lookup
        definition = registry.get_definition("search")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        handler: ToolHandler,
        timeout: float = DEFAULT_TOOL_TIMEOUT,
        overwrite: bool = False,
    ) -> None:
        """Register a tool with the registry.

        Args:
            name: Unique name for the tool.
            description: Human-readable description for agent discovery.
            input_schema: JSON Schema for validating tool inputs.
            output_schema: Optional JSON Schema for validating tool outputs.
            handler: Callable implementing the tool logic.
                Signature: ``handler(**kwargs) -> dict``
            timeout: Default timeout in seconds for this tool.
            overwrite: If True, replace an existing tool with the same name.

        Raises:
            ToolRegistrationError: If a tool with the same name is already
                registered and ``overwrite`` is False.
        """
        if name in self._tools and not overwrite:
            msg = f"Tool '{name}' is already registered. Use overwrite=True to replace it."
            raise ToolRegistrationError(msg, tool_name=name, existing=True)

        if name in self._tools:
            logger.info("Overwriting tool '%s'", name)
        else:
            logger.info("Registering tool '%s' (timeout=%.1fs)", name, timeout)

        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout=timeout,
            handler=handler,
        )
        self._tools[name] = definition

    def register_from_artifact(
        self,
        *,
        artifact: ToolArtifact,
        handler: ToolHandler,
        timeout: float = DEFAULT_TOOL_TIMEOUT,
        overwrite: bool = False,
    ) -> None:
        """Register a tool from a ``ToolArtifact`` schema object.

        The artifact's name, description, input_schema, and output_schema
        are used directly.

        Args:
            artifact: A ``ToolArtifact`` instance with tool metadata.
            handler: Callable implementing the tool logic.
            timeout: Default timeout in seconds.
            overwrite: If True, replace an existing tool with the same name.

        Raises:
            ToolRegistrationError: If a tool with the same name is already
                registered and ``overwrite`` is False.
        """
        self.register(
            name=artifact.name,
            description=artifact.description,
            input_schema=artifact.input_schema,
            output_schema=artifact.output_schema,
            handler=handler,
            timeout=timeout,
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Retrieve a tool definition by name.

        Args:
            name: The unique name of the tool.

        Returns:
            The ``ToolDefinition`` if found, or ``None``.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check whether a tool with the given name is registered.

        Args:
            name: The tool name to check.

        Returns:
            ``True`` if the tool is registered.
        """
        return name in self._tools

    # ------------------------------------------------------------------
    # Capability Advertisement (VAL-TOOL-01)
    # ------------------------------------------------------------------

    def get_capabilities(self) -> dict[str, dict[str, Any]]:
        """Return capability advertisements for all registered tools.

        Each entry contains name, description, input_schema, output_schema
        (if defined), and timeout. The handler is NOT exposed.

        Returns:
            A dict mapping tool names to capability dicts.
        """
        return {name: tool.get_capabilities() for name, tool in self._tools.items()}

    def get_capability(self, name: str) -> dict[str, Any] | None:
        """Return capability advertisement for a single tool.

        Args:
            name: The tool name.

        Returns:
            A capability dict, or ``None`` if the tool is not registered.
        """
        definition = self._tools.get(name)
        if definition is None:
            return None
        return definition.get_capabilities()

    def get_tool_definitions_for_agent(self) -> list[dict[str, Any]]:
        """Return tool definitions formatted for LLM agent consumption.

        Produces a list of dicts suitable for inclusion in a tool-use
        API payload (name, description, input_schema).

        Returns:
            A list of tool definition dicts.
        """
        return [tool.get_agent_definition() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            A list of tool name strings.
        """
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    # ------------------------------------------------------------------
    # Deregistration
    # ------------------------------------------------------------------

    def deregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: The name of the tool to remove.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            msg = f"Tool '{name}' is not registered."
            raise ToolNotFoundError(msg, tool_name=name)

        del self._tools[name]
        logger.info("Deregistered tool '%s'", name)

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
        logger.info("Cleared all tools from registry")


__all__ = [
    "DEFAULT_TOOL_TIMEOUT",
    "ToolRegistry",
]
