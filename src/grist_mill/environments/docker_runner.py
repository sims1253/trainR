"""DockerRunner — Docker-based execution environment.

Creates Docker containers per task: pulls/uses configured image, injects
artifacts into the container filesystem, enforces CPU/memory/timeout limits,
captures output, and cleans up containers on success/failure/timeout.

Supports volume mounts for workspace isolation. Verifies Docker daemon
availability before execution. Handles missing images (auto-pull or error).

Validates VAL-HARNESS-02, VAL-HARNESS-03, VAL-HARNESS-04, VAL-ENV-05, VAL-ENV-07.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import docker.errors
import docker.models.containers
import docker.models.images
import docker.models.volumes

import docker
from grist_mill.harness.result_parser import ResultParser
from grist_mill.interfaces import BaseEnvironment
from grist_mill.schemas import (
    ExecutionOutput,
    Task,
    TaskResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DockerDaemonError(Exception):
    """Raised when the Docker daemon is not available."""

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(
            f"Docker daemon is not available: {message}. "
            "Please ensure Docker is installed and running."
        )


class DockerImageError(Exception):
    """Raised when a required Docker image is not available."""

    def __init__(self, image: str, reason: str = "") -> None:
        self.image = image
        self.reason = reason
        super().__init__(
            f"Docker image '{image}' is not available: {reason}. "
            f"Run 'docker pull {image}' to download it."
        )


# ---------------------------------------------------------------------------
# DockerRunner
# ---------------------------------------------------------------------------

# Default container name prefix
_CONTAINER_NAME_PREFIX = "grist-mill-"


class DockerRunner(BaseEnvironment):
    """Execute tasks inside isolated Docker containers.

    Each task gets its own container and workspace volume.  Artifacts are
    injected into the container's workspace before execution.  Resource
    limits (CPU, memory, timeout) are enforced at the container level.

    Args:
        image: Docker image to use (default ``"python:3.12-slim"``).
        cpu_limit: CPU limit in cores (default ``1.0``).
        memory_limit: Memory limit string, e.g. ``"4g"``, ``"512m"`` (default ``"4g"``).
        auto_pull: If ``True``, pull missing images automatically (default ``True``).
        keep_workspace_on_failure: If ``True``, preserve workspace volumes on cleanup
            for debugging (default ``False``).
        workspace_path: Path inside the container for the workspace
            (default ``"/workspace"``).
        artifacts: Mapping of ``{relative_path: content}`` to inject into the
            workspace before execution.
        result_parser: Optional ``ResultParser`` for converting execution output
            to ``TaskResult``.  A default instance is created if not provided.

    Example::

        runner = DockerRunner(image="python:3.12-slim", cpu_limit=2.0)
        runner.prepare(task)
        output = runner.execute("pytest tests/", timeout=60.0)
        result = runner.run_task(
            command="pytest tests/",
            task_id="t1",
            language="python",
            timeout=60.0,
        )
        runner.cleanup()
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        cpu_limit: float = 1.0,
        memory_limit: str = "4g",
        auto_pull: bool = True,
        keep_workspace_on_failure: bool = False,
        workspace_path: str = "/workspace",
        artifacts: dict[str, str] | None = None,
        result_parser: ResultParser | None = None,
    ) -> None:
        self._image: str = image
        self._cpu_limit: float = cpu_limit
        self._memory_limit: str = memory_limit
        self._auto_pull: bool = auto_pull
        self._keep_workspace_on_failure: bool = keep_workspace_on_failure
        self._workspace_path: str = workspace_path
        self._artifacts: dict[str, str] = artifacts or {}
        self._result_parser: ResultParser = result_parser or ResultParser()

        # Internal state
        self._container_id: str | None = None
        self._workspace_volume_name: str | None = None
        self._prepared: bool = False
        self._client: docker.DockerClient | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def image(self) -> str:
        """The configured Docker image."""
        return self._image

    @property
    def cpu_limit(self) -> float:
        """CPU limit in cores."""
        return self._cpu_limit

    @property
    def memory_limit(self) -> str:
        """Memory limit string."""
        return self._memory_limit

    @property
    def auto_pull(self) -> bool:
        """Whether to auto-pull missing images."""
        return self._auto_pull

    @property
    def keep_workspace_on_failure(self) -> bool:
        """Whether to preserve workspace volumes on cleanup."""
        return self._keep_workspace_on_failure

    @property
    def workspace_path(self) -> str:
        """Path inside the container for the workspace."""
        return self._workspace_path

    @property
    def artifacts(self) -> dict[str, str]:
        """Artifact mapping to inject into the workspace."""
        return dict(self._artifacts)

    @property
    def result_parser(self) -> ResultParser:
        """The result parser instance."""
        return self._result_parser

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare()`` has been called without a matching ``cleanup()``."""
        return self._prepared

    # ------------------------------------------------------------------
    # Docker client management
    # ------------------------------------------------------------------

    def _get_client(self) -> docker.DockerClient:
        """Get or create a Docker client connection.

        Returns:
            A connected ``docker.DockerClient`` instance.

        Raises:
            DockerDaemonError: If the Docker daemon is not reachable.
        """
        if self._client is not None:
            return self._client

        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as exc:
            self._client = None
            raise DockerDaemonError(str(exc)) from exc

        return self._client

    def check_docker_available(self) -> bool:
        """Check whether the Docker daemon is reachable.

        Returns:
            ``True`` if Docker is available, ``False`` otherwise.
        """
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Image management (VAL-ENV-07)
    # ------------------------------------------------------------------

    def _ensure_image_available(self, image: str | None = None) -> None:
        """Ensure the Docker image is available locally.

        If the image is not present locally:
        - If ``auto_pull=True``, pulls it automatically.
        - If ``auto_pull=False``, raises ``DockerImageError`` with a
          suggestion to run ``docker pull``.

        Args:
            image: Image name to check. Defaults to ``self._image``.

        Raises:
            DockerImageError: If the image is missing and ``auto_pull=False``.
            DockerDaemonError: If Docker daemon is unreachable.
        """
        image_name = image or self._image
        client = self._get_client()

        # Check if image exists locally
        try:
            images = client.images.list(name=image_name)
            if images:
                logger.debug("Image '%s' found locally", image_name)
                return
        except Exception as exc:
            raise DockerDaemonError(f"Failed to list images: {exc}") from exc

        # Image not found locally
        if self._auto_pull:
            logger.info("Pulling image '%s' ...", image_name)
            try:
                client.images.pull(image_name)
                logger.info("Successfully pulled image '%s'", image_name)
            except Exception as exc:
                raise DockerImageError(
                    image_name,
                    f"Failed to pull: {exc}",
                ) from exc
        else:
            raise DockerImageError(
                image_name,
                "Image not found locally and auto_pull is disabled",
            )

    # ------------------------------------------------------------------
    # BaseEnvironment implementation
    # ------------------------------------------------------------------

    def prepare(self, task: Task) -> None:
        """Prepare a Docker container for task execution.

        Ensures the configured image is available, creates an isolated
        workspace volume, injects artifacts, and creates the container.

        Args:
            task: The task to prepare the environment for.

        Raises:
            DockerDaemonError: If Docker daemon is unreachable.
            DockerImageError: If the image is missing and auto-pull is disabled.
        """
        if self._prepared:
            logger.warning("DockerRunner already prepared, skipping")
            return

        # 1. Ensure image is available
        self._ensure_image_available()

        client = self._get_client()

        # 2. Create an isolated workspace volume
        volume_name = f"{_CONTAINER_NAME_PREFIX}workspace-{uuid.uuid4().hex[:12]}"
        self._workspace_volume_name = volume_name

        try:
            workspace_vol: docker.models.volumes.Volume = client.volumes.create(
                name=volume_name,
                driver="local",
            )
        except Exception as exc:
            raise DockerDaemonError(f"Failed to create workspace volume: {exc}") from exc

        # 3. Create a helper container to inject artifacts into the volume
        if self._artifacts:
            self._inject_artifacts(client, workspace_vol)

        # 4. Create the main execution container
        container_name = f"{_CONTAINER_NAME_PREFIX}{task.id}-{uuid.uuid4().hex[:8]}"

        try:
            container: docker.models.containers.Container = client.containers.create(
                image=self._image,
                name=container_name,
                volumes={
                    volume_name: {"bind": self._workspace_path, "mode": "rw"},
                },
                working_dir=self._workspace_path,
                mem_limit=self._memory_limit,
                nano_cpus=int(self._cpu_limit * 1e9),
                detach=True,
                stdin_open=True,
                tty=False,
            )
            self._container_id = container.id
        except Exception as exc:
            # Clean up the volume if container creation fails
            with contextlib.suppress(Exception):
                workspace_vol.remove(force=True)
            raise DockerDaemonError(f"Failed to create container: {exc}") from exc

        self._prepared = True
        logger.info(
            "Prepared Docker container '%s' (image=%s, cpu=%.1f, mem=%s, workspace=%s)",
            container_name,
            self._image,
            self._cpu_limit,
            self._memory_limit,
            self._workspace_path,
        )

        # 5. Run setup_command if specified
        if task.setup_command:
            logger.info("Running setup command: %s", task.setup_command)
            output = self.execute(task.setup_command, timeout=float(task.timeout))
            if output.exit_code != 0:
                logger.warning(
                    "Setup command exited with code %d: %s",
                    output.exit_code,
                    output.stderr[:200] if output.stderr else "(no stderr)",
                )

    def execute(self, command: str, timeout: float) -> ExecutionOutput:
        """Execute a command inside the prepared Docker container.

        The command runs as an exec inside the running container.  If
        the command exceeds *timeout* seconds, the container is killed.

        Uses a background thread to run ``exec_run`` so we can enforce
        the timeout without blocking indefinitely.

        Args:
            command: The shell command to execute inside the container.
            timeout: Maximum time in seconds before killing the container.

        Returns:
            An ``ExecutionOutput`` with captured stdout, stderr, exit_code,
            and ``timed_out`` flag.
        """
        if not self._prepared or self._container_id is None:
            raise RuntimeError(
                "DockerRunner has not been prepared. Call prepare() before execute()."
            )

        client = self._get_client()

        try:
            container = client.containers.get(self._container_id)
        except docker.errors.NotFound:
            return ExecutionOutput(
                stdout="",
                stderr=f"Container {self._container_id} no longer exists",
                exit_code=-1,
                timed_out=False,
            )

        # Start the container before exec
        with contextlib.suppress(Exception):
            container.start()

        # Use a thread to run exec_run with a timeout
        exec_result_holder: list[Any] = []
        exec_error_holder: list[Exception] = []

        def _run_exec() -> None:
            try:
                result = container.exec_run(
                    cmd=["/bin/sh", "-c", command],
                    workdir=self._workspace_path,
                    demux=True,
                )
                exec_result_holder.append(result)
            except Exception as exc:
                exec_error_holder.append(exc)

        thread = threading.Thread(target=_run_exec, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running — we've timed out
            logger.warning(
                "Command timed out after %.1fs, killing container",
                timeout,
            )
            return self._kill_container(container)

        if exec_error_holder:
            exc = exec_error_holder[0]
            if isinstance(exc, docker.errors.APIError) and (
                "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            ):
                return self._kill_container(container)
            return ExecutionOutput(
                stdout="",
                stderr=f"Docker API error: {exc}",
                exit_code=-1,
                timed_out=False,
            )

        if not exec_result_holder:
            return ExecutionOutput(
                stdout="",
                stderr="exec_run returned no result",
                exit_code=-1,
                timed_out=False,
            )

        exec_result = exec_result_holder[0]

        # Decode output
        stdout_bytes, stderr_bytes = exec_result.output if exec_result.output else (b"", b"")
        if isinstance(stdout_bytes, bytes):
            stdout = stdout_bytes.decode("utf-8", errors="replace")
        else:
            stdout = str(stdout_bytes) if stdout_bytes else ""

        if isinstance(stderr_bytes, bytes):
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        else:
            stderr = str(stderr_bytes) if stderr_bytes else ""

        return ExecutionOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=exec_result.exit_code or 0,
            timed_out=False,
        )

    def cleanup(self) -> None:
        """Clean up the Docker container and workspace volume.

        Removes the container (force=True) and, unless
        ``keep_workspace_on_failure`` is ``True``, also removes the
        workspace volume.  Safe to call multiple times or without
        a preceding ``prepare()``.
        """
        if not self._prepared:
            return

        client: docker.DockerClient | None = None
        try:
            client = self._get_client()
        except DockerDaemonError:
            # Docker not available — reset state and return
            self._container_id = None
            self._workspace_volume_name = None
            self._prepared = False
            return

        # 1. Remove the container
        if self._container_id is not None:
            try:
                container = client.containers.get(self._container_id)
                container.remove(force=True)
                logger.debug("Removed container %s", self._container_id)
            except docker.errors.NotFound:
                logger.debug("Container %s already removed", self._container_id)
            except Exception as exc:
                logger.warning("Failed to remove container %s: %s", self._container_id, exc)

        # 2. Remove the workspace volume (unless keep_workspace_on_failure)
        if self._workspace_volume_name is not None and not self._keep_workspace_on_failure:
            try:
                vol = client.volumes.get(self._workspace_volume_name)
                vol.remove(force=True)
                logger.debug("Removed workspace volume %s", self._workspace_volume_name)
            except docker.errors.NotFound:
                logger.debug("Workspace volume %s already removed", self._workspace_volume_name)
            except Exception as exc:
                logger.warning("Failed to remove volume %s: %s", self._workspace_volume_name, exc)

        # Reset state
        self._container_id = None
        self._workspace_volume_name = None
        self._prepared = False

        # Close the Docker client
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
        self._client = None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def run_task(
        self,
        *,
        command: str,
        task_id: str,
        language: str = "python",
        timeout: float,
    ) -> TaskResult:
        """Execute a command and parse the result into a TaskResult.

        Convenience method combining ``execute()`` with ``ResultParser.parse()``.

        Args:
            command: The shell command to execute inside the container.
            task_id: The ID of the task (for the resulting TaskResult).
            language: Programming language hint for error categorization.
            timeout: Maximum time in seconds.

        Returns:
            A ``TaskResult`` with status, score, and error_category populated.
        """
        output = self.execute(command, timeout=timeout)
        return self._result_parser.parse(
            output,
            task_id=task_id,
            language=language,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject_artifacts(
        self,
        client: docker.DockerClient,
        volume: docker.models.volumes.Volume,
    ) -> None:
        """Inject artifacts into the workspace volume using a helper container.

        Creates a temporary container that mounts the volume and writes
        artifact files to it, then removes itself.

        Args:
            client: The Docker client.
            volume: The workspace volume to inject artifacts into.
        """
        # Build a shell script that creates all artifact files
        commands: list[str] = []
        for rel_path, content in self._artifacts.items():
            full_path = f"{self._workspace_path}/{rel_path}"
            parent_dir = str(Path(rel_path).parent)
            commands.append(f"mkdir -p '{self._workspace_path}/{parent_dir}'")
            # Use heredoc to write content safely (avoids shell escaping issues)
            commands.append(f"cat > '{full_path}' << 'GRIST_MILL_EOF'\n{content}\nGRIST_MILL_EOF")

        script = "\n".join(commands)

        helper_name = f"{_CONTAINER_NAME_PREFIX}artifact-inject-{uuid.uuid4().hex[:8]}"
        helper = None
        try:
            helper = client.containers.create(
                image=self._image,
                name=helper_name,
                volumes={
                    volume.name: {"bind": self._workspace_path, "mode": "rw"},
                },
                command=["/bin/sh", "-c", script],
            )
            helper.start()
            exit_status = helper.wait(timeout=30)
            if exit_status["StatusCode"] != 0:
                logs = helper.logs(stderr=True).decode("utf-8", errors="replace")[:200]
                logger.warning(
                    "Artifact injection helper exited with code %d: %s",
                    exit_status["StatusCode"],
                    logs,
                )
        except Exception as exc:
            logger.warning("Failed to inject artifacts: %s", exc)
        finally:
            # Clean up the helper container
            if helper is not None:
                with contextlib.suppress(Exception):
                    helper = client.containers.get(helper_name)
                    helper.remove(force=True)

    def _kill_container(
        self,
        container: docker.models.containers.Container,
        *,
        partial_stdout: str = "",
        partial_stderr: str = "",
    ) -> ExecutionOutput:
        """Kill a container that has exceeded its timeout.

        Sends SIGKILL, waits for removal, and returns a timed-out
        ``ExecutionOutput`` with any partial output captured.

        Args:
            container: The container to kill.
            partial_stdout: Any stdout captured before the timeout.
            partial_stderr: Any stderr captured before the timeout.

        Returns:
            An ``ExecutionOutput`` with ``timed_out=True``.
        """
        logger.warning("Killing container %s due to timeout", container.id)
        try:
            container.kill()
        except docker.errors.NotFound:
            pass  # Already dead
        except Exception as exc:
            logger.warning("Failed to kill container: %s", exc)

        with contextlib.suppress(Exception):
            container.remove(force=True)

        return ExecutionOutput(
            stdout=partial_stdout,
            stderr=partial_stderr,
            exit_code=-1,
            timed_out=True,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DockerRunner(image={self._image!r}, cpu_limit={self._cpu_limit!r}, "
            f"memory_limit={self._memory_limit!r}, auto_pull={self._auto_pull!r}, "
            f"keep_workspace_on_failure={self._keep_workspace_on_failure!r})"
        )
