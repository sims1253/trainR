"""Tool invocation with input/output validation and timeout enforcement.

Provides:
- ToolInvoker: Routes tool calls through the registry, validates input/output,
  enforces timeouts, and records per-call telemetry.

Validates:
- VAL-TOOL-03: Tool invocation routes through registry with validation
- VAL-TOOL-04: Tool invocation timeout is enforced per-call
- VAL-TOOL-05: Per-tool telemetry is captured
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any

from grist_mill.schemas.telemetry import TelemetryCollector
from grist_mill.tools.exceptions import ToolTimeoutError
from grist_mill.tools.models import ToolDefinition, ToolInvocationResult
from grist_mill.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _validate_json_schema(
    data: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    """Validate data against a JSON Schema, returning a list of error messages.

    Uses a minimal inline validator rather than requiring the ``jsonschema``
    package. Supports ``type``, ``required``, ``properties``, ``minimum``,
    ``maximum``, and ``minLength`` keywords.

    Args:
        data: The data to validate.
        schema: The JSON Schema to validate against.

    Returns:
        A list of error message strings. Empty list means validation passed.
    """
    errors: list[dict[str, str]] = []

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(
                {"field": "(root)", "message": f"Expected object, got {type(data).__name__}"}
            )
            return errors

        required = schema.get("required", [])
        for req_field in required:
            if req_field not in data:
                errors.append(
                    {"field": req_field, "message": f"Required field '{req_field}' is missing"}
                )

        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                prop_value = data[prop_name]
                prop_type = prop_schema.get("type")

                if prop_type and not _check_type(prop_value, prop_type):
                    errors.append(
                        {
                            "field": prop_name,
                            "message": f"Expected {prop_type}, got {type(prop_value).__name__}",
                        }
                    )

                if prop_type == "integer" and isinstance(prop_value, (int, float)):
                    minimum = prop_schema.get("minimum")
                    maximum = prop_schema.get("maximum")
                    if minimum is not None and prop_value < minimum:
                        errors.append(
                            {
                                "field": prop_name,
                                "message": f"Value {prop_value} is less than minimum {minimum}",
                            }
                        )
                    if maximum is not None and prop_value > maximum:
                        errors.append(
                            {
                                "field": prop_name,
                                "message": f"Value {prop_value} exceeds maximum {maximum}",
                            }
                        )

                if prop_type == "string" and isinstance(prop_value, str):
                    min_length = prop_schema.get("minLength")
                    if min_length is not None and len(prop_value) < min_length:
                        errors.append(
                            {
                                "field": prop_name,
                                "message": f"String length {len(prop_value)} is less than minLength {min_length}",
                            }
                        )
    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(
                {"field": "(root)", "message": f"Expected string, got {type(data).__name__}"}
            )
    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(
                {"field": "(root)", "message": f"Expected integer, got {type(data).__name__}"}
            )
    elif schema_type == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            errors.append(
                {"field": "(root)", "message": f"Expected number, got {type(data).__name__}"}
            )
    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(
                {"field": "(root)", "message": f"Expected boolean, got {type(data).__name__}"}
            )
    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(
                {"field": "(root)", "message": f"Expected array, got {type(data).__name__}"}
            )

    return errors


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected JSON Schema type."""
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": (int,),
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True  # Unknown type, skip check
    if expected_type == "integer":
        # int but not bool
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return isinstance(value, expected)


def _format_validation_errors(errors: list[dict[str, str]]) -> str:
    """Format validation errors into a human-readable message."""
    parts = [f"{e['field']}: {e['message']}" for e in errors]
    return "Input validation failed: " + "; ".join(parts)


class ToolInvoker:
    """Routes tool calls through the registry with validation and telemetry.

    For each tool invocation:

    1. Looks up the tool in the registry
    2. Validates input against the tool's input_schema
    3. Executes the tool handler with timeout enforcement
    4. Validates output against the tool's output_schema
    5. Records telemetry (duration, success/failure)

    Usage::

        registry = ToolRegistry()
        registry.register(name="search", ...)

        collector = TelemetryCollector()
        invoker = ToolInvoker(registry=registry, telemetry_collector=collector)

        result = invoker.invoke("search", arguments={"query": "python"})
        if result.success:
            print(result.output)
        else:
            print(result.error)
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        telemetry_collector: TelemetryCollector | None = None,
    ) -> None:
        self._registry = registry
        self._telemetry_collector = telemetry_collector

    def invoke(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> ToolInvocationResult:
        """Invoke a tool through the registry.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Input arguments for the tool.
            timeout: Optional per-call timeout override in seconds.
                If not provided, uses the tool's default timeout.

        Returns:
            A ``ToolInvocationResult`` with success/output or error.
        """
        start_time = time.perf_counter()

        # 1. Look up the tool
        definition = self._registry.get_definition(tool_name)
        if definition is None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ToolInvocationResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' is not registered in the registry.",
                duration_ms=duration_ms,
            )
            self._record_telemetry(tool_name, success=False, duration_ms=duration_ms)
            return result

        # Use per-call timeout override or the tool's default
        effective_timeout = timeout if timeout is not None else definition.timeout

        # 2. Validate input
        if definition.input_schema:
            validation_errors = _validate_json_schema(arguments, definition.input_schema)
            if validation_errors:
                duration_ms = (time.perf_counter() - start_time) * 1000
                error_msg = _format_validation_errors(validation_errors)
                result = ToolInvocationResult(
                    tool_name=tool_name,
                    success=False,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
                self._record_telemetry(tool_name, success=False, duration_ms=duration_ms)
                return result

        # 3. Execute with timeout
        try:
            output = self._execute_with_timeout(
                definition,
                arguments,
                effective_timeout,
            )
        except ToolTimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = (
                f"Tool '{tool_name}' timed out after {effective_timeout:.1f}s. "
                f"Consider increasing the timeout or optimizing the handler."
            )
            result = ToolInvocationResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
            self._record_telemetry(tool_name, success=False, duration_ms=duration_ms)
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = (
                f"Tool '{tool_name}' execution failed: {exc!s}. Check the tool handler for errors."
            )
            logger.error("Tool '%s' handler error: %s", tool_name, exc)
            result = ToolInvocationResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
            self._record_telemetry(tool_name, success=False, duration_ms=duration_ms)
            return result

        # 4. Validate output
        if definition.output_schema is not None:
            output_errors = _validate_json_schema(output, definition.output_schema)
            if output_errors:
                duration_ms = (time.perf_counter() - start_time) * 1000
                error_msg = _format_validation_errors(output_errors).replace(
                    "Input validation failed",
                    "Output validation failed",
                )
                result = ToolInvocationResult(
                    tool_name=tool_name,
                    success=False,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
                self._record_telemetry(tool_name, success=False, duration_ms=duration_ms)
                return result

        # 5. Success
        duration_ms = (time.perf_counter() - start_time) * 1000
        result = ToolInvocationResult(
            tool_name=tool_name,
            success=True,
            output=output,
            duration_ms=duration_ms,
        )
        self._record_telemetry(tool_name, success=True, duration_ms=duration_ms)
        return result

    def _execute_with_timeout(
        self,
        definition: ToolDefinition,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Execute a tool handler with timeout enforcement.

        Uses ``concurrent.futures.ThreadPoolExecutor`` to run the handler
        in a separate thread, allowing timeout enforcement without
        blocking the main thread.

        Args:
            definition: The tool definition to execute.
            arguments: Validated input arguments.
            timeout: Maximum execution time in seconds.

        Returns:
            The handler's output dict.

        Raises:
            ToolTimeoutError: If the handler exceeds the timeout.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(definition.handler, **arguments)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # Cancel the future (best-effort; thread may still be running)
                future.cancel()
                raise ToolTimeoutError(
                    f"Tool '{definition.name}' timed out after {timeout:.1f}s",
                    tool_name=definition.name,
                    timeout_s=timeout,
                ) from None

    def _record_telemetry(
        self,
        tool_name: str,
        *,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record tool invocation in the telemetry collector.

        Args:
            tool_name: Name of the invoked tool.
            success: Whether the invocation succeeded.
            duration_ms: Duration in milliseconds.
        """
        if self._telemetry_collector is not None:
            self._telemetry_collector.record_tool_call(
                tool_name=tool_name,
                success=success,
                duration_ms=duration_ms,
            )


__all__ = [
    "ToolInvoker",
]
