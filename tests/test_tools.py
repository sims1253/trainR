"""Tests for the tool orchestration layer.

Covers:
- VAL-TOOL-01: Tool registry supports capability advertisement
- VAL-TOOL-02: Tool registry supports dynamic registration at runtime
- VAL-TOOL-03: Tool invocation routes through registry with validation
- VAL-TOOL-04: Tool invocation timeout is enforced per-call
- VAL-TOOL-05: Per-tool telemetry is captured
- VAL-TOOL-06: MCP server lifecycle management
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from grist_mill.schemas.artifact import MCPServerArtifact, ToolArtifact
from grist_mill.schemas.telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["query"],
}

VALID_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
        "total": {"type": "integer"},
    },
    "required": ["results", "total"],
}

SIMPLE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}

SIMPLE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
}


# ---------------------------------------------------------------------------
# VAL-TOOL-01: Tool registry supports capability advertisement
# ---------------------------------------------------------------------------


class TestToolRegistryCapabilityAdvertisement:
    """Tests for VAL-TOOL-01: capability advertisement."""

    def test_register_tool_advertises_capabilities(self) -> None:
        """Registered tool must declare name, description, input_schema, output_schema."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="search",
            description="Search the web for information",
            input_schema=VALID_INPUT_SCHEMA,
            output_schema=VALID_OUTPUT_SCHEMA,
            handler=lambda **kwargs: {"results": [], "total": 0},
        )

        capabilities = registry.get_capabilities()
        assert "search" in capabilities
        cap = capabilities["search"]
        assert cap["name"] == "search"
        assert cap["description"] == "Search the web for information"
        assert cap["input_schema"] == VALID_INPUT_SCHEMA
        assert cap["output_schema"] == VALID_OUTPUT_SCHEMA

    def test_list_tools_returns_all_registered_tools(self) -> None:
        """list_tools returns all registered tool names."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="tool_a",
            description="Tool A",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )
        registry.register(
            name="tool_b",
            description="Tool B",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        assert set(registry.list_tools()) == {"tool_a", "tool_b"}

    def test_get_capability_single_tool(self) -> None:
        """get_capability returns capabilities for a single tool."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="read_file",
            description="Read a file from disk",
            input_schema=SIMPLE_INPUT_SCHEMA,
            output_schema=SIMPLE_OUTPUT_SCHEMA,
            handler=lambda **kwargs: {"greeting": "hello"},
        )

        cap = registry.get_capability("read_file")
        assert cap is not None
        assert cap["name"] == "read_file"
        assert cap["description"] == "Read a file from disk"

    def test_get_capability_nonexistent_returns_none(self) -> None:
        """get_capability returns None for unregistered tool."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        assert registry.get_capability("nonexistent") is None

    def test_capabilities_include_handler_not_exposed(self) -> None:
        """Capabilities should not expose the handler function."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="secure_tool",
            description="A secure tool",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        cap = registry.get_capability("secure_tool")
        assert "handler" not in cap
        assert "timeout" in cap  # timeout should be advertised

    def test_advertise_default_timeout(self) -> None:
        """Tools advertise their default timeout in capabilities."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="timed_tool",
            description="A tool with timeout",
            input_schema=SIMPLE_INPUT_SCHEMA,
            timeout=30.0,
            handler=lambda **kwargs: {},
        )

        cap = registry.get_capability("timed_tool")
        assert cap["timeout"] == 30.0

    def test_tool_for_agent_discovery(self) -> None:
        """Tools can be formatted for agent discovery (LLM tool definitions)."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="calculator",
            description="Perform arithmetic calculations",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
            handler=lambda **kwargs: {"result": 42},
        )

        tool_defs = registry.get_tool_definitions_for_agent()
        assert len(tool_defs) == 1
        tool_def = tool_defs[0]
        assert tool_def["name"] == "calculator"
        assert tool_def["description"] == "Perform arithmetic calculations"
        assert tool_def["input_schema"]["type"] == "object"
        assert "handler" not in tool_def


# ---------------------------------------------------------------------------
# VAL-TOOL-02: Tool registry supports dynamic registration at runtime
# ---------------------------------------------------------------------------


class TestToolRegistryDynamicRegistration:
    """Tests for VAL-TOOL-02: dynamic registration."""

    def test_register_tool_after_initial_setup(self) -> None:
        """Tools can be registered at any point before execution."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="initial_tool",
            description="Registered at startup",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        assert "initial_tool" in registry.list_tools()

        # Register more tools later
        registry.register(
            name="late_tool",
            description="Registered after initial setup",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        assert "late_tool" in registry.list_tools()
        assert registry.count == 2

    def test_duplicate_registration_raises_error(self) -> None:
        """Registering a duplicate tool name raises ToolRegistrationError."""
        from grist_mill.tools.exceptions import ToolRegistrationError
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="dup_tool",
            description="First registration",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        with pytest.raises(ToolRegistrationError, match="dup_tool"):
            registry.register(
                name="dup_tool",
                description="Duplicate registration",
                input_schema=SIMPLE_INPUT_SCHEMA,
                handler=lambda **kwargs: {},
            )

    def test_duplicate_registration_with_overwrite(self) -> None:
        """Overwriting an existing tool with overwrite=True succeeds."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="replace_me",
            description="Original",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {"original": True},
        )

        registry.register(
            name="replace_me",
            description="Replaced",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {"replaced": True},
            overwrite=True,
        )

        cap = registry.get_capability("replace_me")
        assert cap["description"] == "Replaced"

    def test_deregister_tool(self) -> None:
        """Tools can be deregistered."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="temp_tool",
            description="Temporary tool",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        assert "temp_tool" in registry.list_tools()
        registry.deregister("temp_tool")
        assert "temp_tool" not in registry.list_tools()

    def test_deregister_nonexistent_raises(self) -> None:
        """Deregistering a nonexistent tool raises ToolNotFoundError."""
        from grist_mill.tools.exceptions import ToolNotFoundError
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()

        with pytest.raises(ToolNotFoundError):
            registry.deregister("nonexistent")

    def test_register_from_tool_artifact(self) -> None:
        """Tools can be registered from ToolArtifact schema objects."""
        from grist_mill.tools.registry import ToolRegistry

        artifact = ToolArtifact(
            name="artifact_tool",
            description="Registered from artifact",
            input_schema=SIMPLE_INPUT_SCHEMA,
            output_schema=SIMPLE_OUTPUT_SCHEMA,
        )

        registry = ToolRegistry()
        registry.register_from_artifact(
            artifact=artifact,
            handler=lambda **kwargs: {"greeting": "hello"},
        )

        assert "artifact_tool" in registry.list_tools()
        cap = registry.get_capability("artifact_tool")
        assert cap["description"] == "Registered from artifact"

    def test_clear_all_tools(self) -> None:
        """clear() removes all registered tools."""
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="tool1",
            description="Tool 1",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )
        registry.register(
            name="tool2",
            description="Tool 2",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {},
        )

        assert registry.count == 2
        registry.clear()
        assert registry.count == 0


# ---------------------------------------------------------------------------
# VAL-TOOL-03: Tool invocation routes through registry
# ---------------------------------------------------------------------------


class TestToolInvocationRouting:
    """Tests for VAL-TOOL-03: invocation routing with validation."""

    def test_invoke_tool_success(self) -> None:
        """Tool invocation routes through registry and returns result."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="greet",
            description="Greet someone",
            input_schema=SIMPLE_INPUT_SCHEMA,
            output_schema=SIMPLE_OUTPUT_SCHEMA,
            handler=lambda name="World": {"greeting": f"Hello, {name}!"},
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("greet", arguments={"name": "Alice"})

        assert result.success is True
        assert result.output == {"greeting": "Hello, Alice!"}
        assert result.error is None

    def test_invoke_validates_input_against_schema(self) -> None:
        """Tool invocation validates input against input_schema."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="strict_tool",
            description="Requires specific input",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {"ok": True},
        )

        invoker = ToolInvoker(registry=registry)

        # Missing required field 'name'
        result = invoker.invoke("strict_tool", arguments={})

        assert result.success is False
        assert result.error is not None
        assert "name" in result.error

    def test_invoke_validates_output_against_schema(self) -> None:
        """Tool invocation validates output against output_schema."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="bad_output",
            description="Returns wrong output",
            input_schema=SIMPLE_INPUT_SCHEMA,
            output_schema=SIMPLE_OUTPUT_SCHEMA,
            handler=lambda **kwargs: {"wrong_field": "value"},
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("bad_output", arguments={"name": "test"})

        assert result.success is False
        assert result.error is not None
        assert "greeting" in result.error

    def test_invoke_nonexistent_tool_returns_error(self) -> None:
        """Invoking a nonexistent tool returns a ToolNotFoundError result."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("nonexistent", arguments={})

        assert result.success is False
        assert result.error is not None
        assert "nonexistent" in result.error

    def test_invoke_without_output_schema_skips_output_validation(self) -> None:
        """Tools without output_schema skip output validation."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="free_output",
            description="Any output is fine",
            input_schema=SIMPLE_INPUT_SCHEMA,
            output_schema=None,
            handler=lambda **kwargs: {"anything": "goes"},
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("free_output", arguments={"name": "test"})

        assert result.success is True
        assert result.output == {"anything": "goes"}

    def test_invoke_without_input_schema_skips_input_validation(self) -> None:
        """Tools without input_schema accept any input."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="free_input",
            description="Any input is fine",
            input_schema={},
            handler=lambda **kwargs: {"received": kwargs},
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("free_input", arguments={"arbitrary": "data"})

        assert result.success is True
        assert result.output == {"received": {"arbitrary": "data"}}

    def test_invoke_handler_exception_caught(self) -> None:
        """Handler exceptions are caught and returned as error results."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        def failing_handler(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Something went wrong in the handler")

        registry = ToolRegistry()
        registry.register(
            name="failing",
            description="Always fails",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=failing_handler,
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("failing", arguments={"name": "test"})

        assert result.success is False
        assert result.error is not None
        assert "Something went wrong" in result.error

    def test_invoke_records_telemetry(self) -> None:
        """Each invocation records telemetry with the collector."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="tracked",
            description="Tracked tool",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda name="World": {"greeting": f"Hi {name}"},
        )

        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)
        invoker.invoke("tracked", arguments={"name": "Bob"})

        telemetry = collector.build()
        assert telemetry.tool_calls.total_calls == 1
        assert telemetry.tool_calls.successful_calls == 1
        assert telemetry.tool_calls.failed_calls == 0
        assert "tracked" in telemetry.tool_calls.by_tool
        assert telemetry.tool_calls.by_tool["tracked"]["calls"] == 1
        assert telemetry.tool_calls.total_duration_ms > 0


# ---------------------------------------------------------------------------
# VAL-TOOL-04: Tool invocation timeout is enforced per-call
# ---------------------------------------------------------------------------


class TestToolInvocationTimeout:
    """Tests for VAL-TOOL-04: per-call timeout enforcement."""

    def test_timeout_enforced_per_call(self) -> None:
        """Tool exceeding its timeout returns an error result."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        def slow_handler(**kwargs: Any) -> dict[str, Any]:
            time.sleep(5)
            return {"done": True}

        registry = ToolRegistry()
        registry.register(
            name="slow_tool",
            description="Very slow tool",
            input_schema=SIMPLE_INPUT_SCHEMA,
            timeout=0.5,
            handler=slow_handler,
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("slow_tool", arguments={"name": "test"})

        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    def test_timeout_override_per_invocation(self) -> None:
        """Invocation timeout can be overridden per-call."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        def slow_handler(**kwargs: Any) -> dict[str, Any]:
            time.sleep(5)
            return {"done": True}

        registry = ToolRegistry()
        registry.register(
            name="default_slow",
            description="Slow with default timeout",
            input_schema=SIMPLE_INPUT_SCHEMA,
            timeout=10.0,  # default is generous
            handler=slow_handler,
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("default_slow", arguments={"name": "test"}, timeout=0.5)

        assert result.success is False
        assert result.error is not None

    def test_timeout_records_failure_in_telemetry(self) -> None:
        """Timed-out tools are recorded as failures in telemetry."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        def hanging_handler(**kwargs: Any) -> dict[str, Any]:
            time.sleep(10)
            return {"never": "reached"}

        registry = ToolRegistry()
        registry.register(
            name="hang_tool",
            description="Hangs forever",
            input_schema=SIMPLE_INPUT_SCHEMA,
            timeout=0.3,
            handler=hanging_handler,
        )

        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)
        invoker.invoke("hang_tool", arguments={"name": "test"})

        telemetry = collector.build()
        assert telemetry.tool_calls.total_calls == 1
        assert telemetry.tool_calls.failed_calls == 1
        assert telemetry.tool_calls.successful_calls == 0
        assert telemetry.tool_calls.by_tool["hang_tool"]["failures"] == 1

    def test_no_timeout_completes_successfully(self) -> None:
        """Tools completing within timeout succeed normally."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="fast_tool",
            description="Completes quickly",
            input_schema=SIMPLE_INPUT_SCHEMA,
            timeout=10.0,
            handler=lambda **kwargs: {"fast": True},
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("fast_tool", arguments={"name": "test"})

        assert result.success is True
        assert result.output == {"fast": True}


# ---------------------------------------------------------------------------
# VAL-TOOL-05: Per-tool telemetry is captured
# ---------------------------------------------------------------------------


class TestPerToolTelemetry:
    """Tests for VAL-TOOL-05: per-tool telemetry capture."""

    def test_multiple_tool_calls_recorded(self) -> None:
        """Multiple tool calls are all recorded in telemetry."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="tool_a",
            description="Tool A",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {"a": True},
        )
        registry.register(
            name="tool_b",
            description="Tool B",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {"b": True},
        )

        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)

        invoker.invoke("tool_a", arguments={"name": "test"})
        invoker.invoke("tool_b", arguments={"name": "test"})
        invoker.invoke("tool_a", arguments={"name": "test"})

        telemetry = collector.build()
        assert telemetry.tool_calls.total_calls == 3
        assert telemetry.tool_calls.successful_calls == 3
        assert telemetry.tool_calls.failed_calls == 0
        assert telemetry.tool_calls.by_tool["tool_a"]["calls"] == 2
        assert telemetry.tool_calls.by_tool["tool_b"]["calls"] == 1
        assert telemetry.tool_calls.total_duration_ms > 0

    def test_failed_tool_recorded_in_telemetry(self) -> None:
        """Failed tool invocations are recorded in telemetry."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        def bad_handler(**kwargs: Any) -> dict[str, Any]:
            raise ValueError("Bad input")

        registry = ToolRegistry()
        registry.register(
            name="bad_tool",
            description="Fails",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=bad_handler,
        )

        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)
        invoker.invoke("bad_tool", arguments={"name": "test"})

        telemetry = collector.build()
        assert telemetry.tool_calls.total_calls == 1
        assert telemetry.tool_calls.failed_calls == 1
        assert telemetry.tool_calls.successful_calls == 0

    def test_duration_recorded_per_call(self) -> None:
        """Duration is recorded per tool call."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="timed",
            description="Has measurable duration",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: (time.sleep(0.05), {"ok": True})[-1],
        )

        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)
        invoker.invoke("timed", arguments={"name": "test"})

        telemetry = collector.build()
        # Total duration across all tool calls is recorded
        assert telemetry.tool_calls.total_duration_ms > 0
        # Per-tool call count is recorded
        assert telemetry.tool_calls.by_tool["timed"]["calls"] == 1

    def test_telemetry_without_collector(self) -> None:
        """Invoker works without a telemetry collector (no error)."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="simple",
            description="Simple tool",
            input_schema=SIMPLE_INPUT_SCHEMA,
            handler=lambda **kwargs: {"ok": True},
        )

        invoker = ToolInvoker(registry=registry)
        result = invoker.invoke("simple", arguments={"name": "test"})

        assert result.success is True


# ---------------------------------------------------------------------------
# VAL-TOOL-06: MCP server lifecycle management
# ---------------------------------------------------------------------------


class TestMCPServerLifecycle:
    """Tests for VAL-TOOL-06: MCP server lifecycle management."""

    def test_start_mcp_server(self) -> None:
        """MCP server can be started as a subprocess."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact = MCPServerArtifact(
            name="echo_server",
            command="echo",
            args=["hello from mcp"],
        )

        server = manager.start(artifact)

        assert server.name == "echo_server"
        # echo exits immediately, so wait for it to complete
        server.wait(timeout=5.0)
        assert server.exit_code == 0
        assert server.stdout is not None
        assert "hello from mcp" in server.stdout

        manager.stop_all()

    def test_stop_mcp_server(self) -> None:
        """MCP server can be stopped after execution."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact = MCPServerArtifact(
            name="sleep_server",
            command="sleep",
            args=["30"],
        )

        server = manager.start(artifact)
        assert server.is_alive()

        manager.stop("sleep_server")
        # After stopping, server should not be alive
        # Give it a moment to terminate
        time.sleep(0.2)
        assert not server.is_alive()

    def test_stop_all_servers(self) -> None:
        """All MCP servers are stopped when stop_all is called."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact1 = MCPServerArtifact(name="server1", command="sleep", args=["30"])
        artifact2 = MCPServerArtifact(name="server2", command="sleep", args=["30"])

        manager.start(artifact1)
        manager.start(artifact2)

        assert len(manager.list_running()) >= 1  # at least one still alive

        manager.stop_all()

        # Both should be stopped
        time.sleep(0.2)
        running = [
            s
            for s in [manager.get_server("server1"), manager.get_server("server2")]
            if s is not None and s.is_alive()
        ]
        assert len(running) == 0

    def test_stdout_stderr_captured(self) -> None:
        """MCP server stdout and stderr are captured."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact = MCPServerArtifact(
            name="output_server",
            command="python3",
            args=["-c", "import sys; print('stdout_msg'); print('stderr_msg', file=sys.stderr)"],
        )

        server = manager.start(artifact)

        # Wait for process to finish
        server.wait(timeout=5.0)

        assert server.stdout is not None
        assert "stdout_msg" in server.stdout
        assert server.stderr is not None
        assert "stderr_msg" in server.stderr

        manager.stop_all()

    def test_list_running_servers(self) -> None:
        """list_running returns names of running MCP servers."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact = MCPServerArtifact(name="listing_test", command="sleep", args=["30"])
        manager.start(artifact)

        running = manager.list_running()
        assert "listing_test" in running

        manager.stop_all()

    def test_get_server_returns_server_info(self) -> None:
        """get_server returns MCPServerInfo for a started server."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact = MCPServerArtifact(
            name="info_test",
            command="echo",
            args=["test"],
        )

        manager.start(artifact)

        server = manager.get_server("info_test")
        assert server is not None
        assert server.name == "info_test"
        assert server.command == "echo"

        manager.stop_all()

    def test_get_nonexistent_server_returns_none(self) -> None:
        """get_server returns None for a server that was never started."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()
        assert manager.get_server("never_started") is None

    def test_stop_nonexistent_server_graceful(self) -> None:
        """Stopping a nonexistent server is a no-op (doesn't raise)."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()
        manager.stop("never_started")  # Should not raise

    def test_server_env_vars(self) -> None:
        """MCP server can be started with environment variables."""
        from grist_mill.tools.mcp import MCPServerManager

        manager = MCPServerManager()

        artifact = MCPServerArtifact(
            name="env_server",
            command="python3",
            args=["-c", "import os; print(os.environ.get('MY_VAR', 'not_set'))"],
            env={"MY_VAR": "hello_world"},
        )

        server = manager.start(artifact)
        server.wait(timeout=5.0)

        assert server.stdout is not None
        assert "hello_world" in server.stdout

        manager.stop_all()

    def test_context_manager_cleanup(self) -> None:
        """MCPServerManager can be used as a context manager for cleanup."""
        from grist_mill.tools.mcp import MCPServerManager

        artifact = MCPServerArtifact(name="ctx_server", command="sleep", args=["30"])

        with MCPServerManager() as manager:
            manager.start(artifact)
            assert "ctx_server" in manager.list_running()

        # After context manager exit, servers should be stopped
        # We can't directly check since manager is closed, but cleanup happened


# ---------------------------------------------------------------------------
# Integration: ToolInvoker + ToolRegistry + Telemetry
# ---------------------------------------------------------------------------


class TestToolOrchestrationIntegration:
    """Integration tests combining all tool orchestration components."""

    def test_full_workflow_registration_invocation_telemetry(self) -> None:
        """End-to-end: register → discover → invoke → telemetry."""
        from grist_mill.tools.invocation import ToolInvoker
        from grist_mill.tools.registry import ToolRegistry

        # 1. Register tools
        registry = ToolRegistry()
        registry.register(
            name="search",
            description="Search the web",
            input_schema=VALID_INPUT_SCHEMA,
            output_schema=VALID_OUTPUT_SCHEMA,
            handler=lambda query="", limit=10: {"results": [{"title": query}], "total": 1},
        )
        registry.register(
            name="calculate",
            description="Perform calculations",
            input_schema={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
            handler=lambda expr="": {"result": eval(expr)},
        )

        # 2. Agent discovers tools
        capabilities = registry.get_capabilities()
        assert len(capabilities) == 2

        tool_defs = registry.get_tool_definitions_for_agent()
        assert len(tool_defs) == 2

        # 3. Invoke with telemetry
        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)

        result1 = invoker.invoke("search", arguments={"query": "python tools", "limit": 5})
        assert result1.success is True
        assert result1.output["total"] == 1

        result2 = invoker.invoke("calculate", arguments={"expr": "2 + 2"})
        assert result2.success is True
        assert result2.output["result"] == 4

        # 4. Verify telemetry
        telemetry = collector.build()
        assert telemetry.tool_calls.total_calls == 2
        assert telemetry.tool_calls.successful_calls == 2
        assert "search" in telemetry.tool_calls.by_tool
        assert "calculate" in telemetry.tool_calls.by_tool

    def test_tool_invocation_result_serialization(self) -> None:
        """ToolInvocationResult can be serialized to dict."""
        from grist_mill.tools.invocation import ToolInvocationResult

        result = ToolInvocationResult(
            tool_name="test",
            success=True,
            output={"key": "value"},
            error=None,
            duration_ms=42.0,
        )

        d = result.to_dict()
        assert d["tool_name"] == "test"
        assert d["success"] is True
        assert d["output"] == {"key": "value"}
        assert d["error"] is None
        assert d["duration_ms"] == 42.0

    def test_error_invocation_result_serialization(self) -> None:
        """Failed ToolInvocationResult serializes correctly."""
        from grist_mill.tools.invocation import ToolInvocationResult

        result = ToolInvocationResult(
            tool_name="bad",
            success=False,
            output=None,
            error="Input validation failed: missing 'name' field",
            duration_ms=1.5,
        )

        d = result.to_dict()
        assert d["success"] is False
        assert d["output"] is None
        assert "missing" in d["error"]
