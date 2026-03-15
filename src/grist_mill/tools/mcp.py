"""MCP server lifecycle management.

Provides:
- MCPServerInfo: Runtime information about a running MCP server process
- MCPServerManager: Starts, stops, and monitors MCP server subprocesses

Validates:
- VAL-TOOL-06: MCP server lifecycle management (start/stop subprocesses,
  stdout/stderr captured, stopped after execution)
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from grist_mill.schemas.artifact import MCPServerArtifact
from grist_mill.tools.exceptions import MCPServerError

logger = logging.getLogger(__name__)


@dataclass
class MCPServerInfo:
    """Runtime information about a running (or completed) MCP server process.

    Attributes:
        name: Name of the MCP server (from the artifact).
        command: The command used to start the server.
        args: Arguments passed to the command.
        pid: Process ID, or None if the process hasn't started.
        stdout: Captured stdout output (None until process completes or is polled).
        stderr: Captured stderr output (None until process completes or is polled).
        exit_code: Exit code of the process, or None if still running.
        start_time: Unix timestamp when the process was started.
    """

    name: str
    command: str
    args: list[str]
    pid: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    start_time: float | None = None
    _process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def is_alive(self) -> bool:
        """Check if the MCP server process is still running.

        Returns:
            True if the process is running, False otherwise.
        """
        if self._process is None:
            return False
        return self._process.poll() is None

    def wait(self, timeout: float | None = None) -> None:
        """Wait for the process to complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.
        """
        if self._process is not None:
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("MCP server '%s' did not complete within timeout", self.name)
            self._capture_output()

    def _capture_output(self) -> None:
        """Capture stdout/stderr from the process if available."""
        if self._process is not None:
            if self._process.stdout is not None:
                self.stdout = self._process.stdout.read()
            if self._process.stderr is not None:
                self.stderr = self._process.stderr.read()
            self.exit_code = self._process.returncode

    def terminate(self) -> None:
        """Terminate the MCP server process.

        Sends SIGTERM, then SIGKILL if the process doesn't exit within 5 seconds.
        """
        if self._process is None:
            return

        if not self.is_alive():
            self._capture_output()
            return

        logger.info("Terminating MCP server '%s' (pid=%d)", self.name, self.pid)
        self._process.terminate()

        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning(
                "MCP server '%s' did not respond to SIGTERM, sending SIGKILL",
                self.name,
            )
            self._process.kill()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                logger.error("MCP server '%s' did not respond to SIGKILL", self.name)

        self._capture_output()
        logger.info(
            "MCP server '%s' terminated (exit_code=%d)",
            self.name,
            self.exit_code,
        )


class MCPServerManager:
    """Manages MCP server subprocess lifecycle.

    Starts MCP servers as subprocesses based on ``MCPServerArtifact``
    definitions, captures their stdout/stderr, and ensures they are
    stopped after execution.

    Can be used as a context manager for automatic cleanup::

        with MCPServerManager() as manager:
            manager.start(artifact)
            # ... use the server ...
        # All servers automatically stopped

    Or manually::

        manager = MCPServerManager()
        manager.start(artifact)
        # ... use the server ...
        manager.stop_all()
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerInfo] = {}

    def start(self, artifact: MCPServerArtifact) -> MCPServerInfo:
        """Start an MCP server as a subprocess.

        Args:
            artifact: The ``MCPServerArtifact`` defining the server's
                command, args, and optional environment variables.

        Returns:
            An ``MCPServerInfo`` with runtime information about the process.

        Raises:
            MCPServerError: If the server fails to start.
        """
        if artifact.name in self._servers and self._servers[artifact.name].is_alive():
            msg = (
                f"MCP server '{artifact.name}' is already running. "
                f"Stop it first or use a different name."
            )
            raise MCPServerError(msg, server_name=artifact.name)

        env: dict[str, str] | None = None
        if artifact.env:
            import os

            env = {**os.environ, **artifact.env}

        try:
            logger.info(
                "Starting MCP server '%s': %s %s",
                artifact.name,
                artifact.command,
                " ".join(artifact.args),
            )

            process = subprocess.Popen(
                [artifact.command, *artifact.args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            info = MCPServerInfo(
                name=artifact.name,
                command=artifact.command,
                args=list(artifact.args),
                pid=process.pid,
                start_time=time.time(),
                _process=process,
            )

            self._servers[artifact.name] = info
            logger.info(
                "MCP server '%s' started (pid=%d)",
                artifact.name,
                process.pid,
            )

            return info

        except FileNotFoundError as exc:
            msg = (
                f"Failed to start MCP server '{artifact.name}': "
                f"Command '{artifact.command}' not found. "
                f"Ensure the command is available in the system PATH."
            )
            raise MCPServerError(msg, server_name=artifact.name) from exc
        except OSError as exc:
            msg = f"Failed to start MCP server '{artifact.name}': OS error: {exc!s}"
            raise MCPServerError(msg, server_name=artifact.name) from exc

    def stop(self, name: str) -> None:
        """Stop a specific MCP server.

        If the server is not registered or not running, this is a no-op.

        Args:
            name: The name of the server to stop.
        """
        info = self._servers.get(name)
        if info is None:
            logger.debug("MCP server '%s' not found in manager, nothing to stop", name)
            return

        info.terminate()

    def stop_all(self) -> None:
        """Stop all running MCP servers.

        Ensures all server processes are terminated and their output
        is captured, regardless of whether they were running or had
        already exited.
        """
        for name, info in self._servers.items():
            if info.is_alive():
                logger.info("Stopping MCP server '%s'", name)
            info.terminate()

        logger.info("All MCP servers stopped")

    def get_server(self, name: str) -> MCPServerInfo | None:
        """Get information about a specific MCP server.

        Args:
            name: The name of the server.

        Returns:
            An ``MCPServerInfo`` if the server was started, or ``None``.
        """
        return self._servers.get(name)

    def list_running(self) -> list[str]:
        """List names of currently running MCP servers.

        Returns:
            A list of server names that are still running.
        """
        return [name for name, info in self._servers.items() if info.is_alive()]

    @property
    def server_count(self) -> int:
        """Total number of servers that have been started."""
        return len(self._servers)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> MCPServerManager:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager, stopping all servers."""
        self.stop_all()


__all__ = [
    "MCPServerInfo",
    "MCPServerManager",
]
