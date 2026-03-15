"""Tool orchestration layer for the grist-mill framework.

Provides:
- ToolRegistry: Dynamic tool registration with capability advertisement
- ToolInvoker: Invocation routing with input/output validation and timeout
- MCPServerManager: MCP server subprocess lifecycle management
- ArtifactBinder: Artifact-to-tool binding (skills, MCP servers, CLI tools)
- ToolDefinition / ToolInvocationResult: Data models for tools
- Custom exception hierarchy for tool errors

Validates:
- VAL-TOOL-01: Tool registry supports capability advertisement
- VAL-TOOL-02: Tool registry supports dynamic registration at runtime
- VAL-TOOL-03: Tool invocation routes through registry with validation
- VAL-TOOL-04: Tool invocation timeout is enforced per-call
- VAL-TOOL-05: Per-tool telemetry is captured
- VAL-TOOL-06: MCP server lifecycle management
- VAL-BIND-01 through VAL-BIND-06: Artifact binding
"""

from grist_mill.tools.exceptions import (
    MCPServerError,
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolTimeoutError,
    ToolValidationError,
)
from grist_mill.tools.invocation import ToolInvoker
from grist_mill.tools.mcp import MCPServerInfo, MCPServerManager
from grist_mill.tools.models import ToolDefinition, ToolInvocationResult
from grist_mill.tools.binding import (
    ArtifactBinder,
    ArtifactBindingError,
    ArtifactValidationError,
)
from grist_mill.tools.registry import DEFAULT_TOOL_TIMEOUT, ToolRegistry

__all__ = [
    "DEFAULT_TOOL_TIMEOUT",
    "ArtifactBinder",
    "ArtifactBindingError",
    "ArtifactValidationError",
    "MCPServerError",
    "MCPServerInfo",
    "MCPServerManager",
    "ToolDefinition",
    "ToolError",
    "ToolInvocationError",
    "ToolInvocationResult",
    "ToolInvoker",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolTimeoutError",
    "ToolValidationError",
]
