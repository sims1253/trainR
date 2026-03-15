"""Tool definition and invocation result schemas.

Provides:
- ToolDefinition: Internal representation of a registered tool
- ToolInvocationResult: Result of a tool invocation attempt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """Internal representation of a registered tool.

    Stores the tool's metadata (name, description, schemas, timeout)
    along with its handler function. This is the internal representation
    used by the ToolRegistry, separate from the public-facing capability
    advertisement.

    Attributes:
        name: Unique tool name used for lookup and invocation.
        description: Human-readable description for agent discovery.
        input_schema: JSON Schema for validating tool inputs.
        output_schema: Optional JSON Schema for validating tool outputs.
        timeout: Default timeout in seconds for this tool's invocation.
        handler: Callable that implements the tool's logic.
            Signature: ``handler(**kwargs) -> dict``
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    timeout: float
    handler: Any  # Callable[..., dict[str, Any]] - using Any to avoid complex Protocol

    def get_capabilities(self) -> dict[str, Any]:
        """Return the public-facing capability advertisement for this tool.

        Exposes name, description, schemas, and timeout, but NOT the handler.

        Returns:
            A dict with capability information suitable for agent discovery.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "timeout": self.timeout,
        }
        if self.output_schema is not None:
            result["output_schema"] = self.output_schema
        return result

    def get_agent_definition(self) -> dict[str, Any]:
        """Return a tool definition formatted for LLM agent consumption.

        Similar to capabilities but formatted for tool-use API payloads.

        Returns:
            A dict with name, description, and input_schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolInvocationResult:
    """Result of a tool invocation attempt.

    Captures whether the invocation succeeded, the output or error,
    the tool name, and the duration.

    Attributes:
        tool_name: Name of the tool that was invoked.
        success: Whether the invocation completed successfully.
        output: The tool's output dict, if successful.
        error: Error message string, if the invocation failed.
        duration_ms: Duration of the invocation in milliseconds.
    """

    tool_name: str
    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the invocation result to a dict.

        Returns:
            A dict with all fields, suitable for telemetry or logging.
        """
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "ToolDefinition",
    "ToolInvocationResult",
]
