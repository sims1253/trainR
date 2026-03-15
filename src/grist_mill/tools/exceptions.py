"""Custom exception classes for the tool orchestration layer.

Provides specific, actionable error types for tool registration,
invocation, and lifecycle management.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base exception for all tool-related errors."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        self.tool_name = tool_name
        super().__init__(message)


class ToolRegistrationError(ToolError):
    """Raised when a tool cannot be registered (e.g., duplicate name)."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        existing: bool = False,
    ) -> None:
        self.existing = existing
        super().__init__(message, tool_name=tool_name)


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found in the registry."""

    def __init__(self, message: str, *, tool_name: str) -> None:
        super().__init__(message, tool_name=tool_name)


class ToolInvocationError(ToolError):
    """Raised when a tool invocation fails."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        cause: Exception | None = None,
    ) -> None:
        self.cause = cause
        super().__init__(message, tool_name=tool_name)


class ToolTimeoutError(ToolError):
    """Raised when a tool invocation exceeds its timeout."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        timeout_s: float,
    ) -> None:
        self.timeout_s = timeout_s
        super().__init__(message, tool_name=tool_name)


class ToolValidationError(ToolError):
    """Raised when tool input or output fails schema validation."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        validation_errors: list[dict[str, str]] | None = None,
    ) -> None:
        self.validation_errors = validation_errors or []
        super().__init__(message, tool_name=tool_name)


class MCPServerError(ToolError):
    """Raised when an MCP server operation fails."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str | None = None,
    ) -> None:
        super().__init__(message, tool_name=server_name)


__all__ = [
    "MCPServerError",
    "ToolError",
    "ToolInvocationError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolTimeoutError",
    "ToolValidationError",
]
