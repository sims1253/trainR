"""DockerRunner — Docker-based execution environment.

Creates Docker containers per task: pulls/uses configured image, injects
artifacts into the container filesystem, enforces CPU/memory/timeout limits,
captures output, and cleans up containers on success/failure/timeout.

Supports volume mounts for workspace isolation. Verifies Docker daemon
availability before execution. Handles missing images (auto-pull or error).

Multi-language support: configurable Docker images per language (python, r,
typescript). Network access control (--network=none by default). Custom
working directory. Environment health check with structured diagnostics.
Setup command runs before agent; failure marks task ERROR.

Validates VAL-HARNESS-02, VAL-HARNESS-03, VAL-HARNESS-04, VAL-ENV-01,
VAL-ENV-02, VAL-ENV-03, VAL-ENV-04, VAL-ENV-05, VAL-ENV-06, VAL-ENV-07,
VAL-ENV-08.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import docker.errors
import docker.models.containers
import docker.models.images
import docker.models.volumes

import docker
from grist_mill.environments.language_config import LanguageImageConfig
from grist_mill.harness.result_parser import ResultParser
from grist_mill.interfaces import BaseEnvironment
from grist_mill.schemas import (
    EnvironmentHealth,
    ErrorCategory,
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


class EnvironmentSetupError(Exception):
    """Raised when the environment setup command fails.

    Indicates that the task should be marked as ERROR with
    error_category=ENVIRONMENT_ERROR without invoking the agent.
    """

    def __init__(
        self,
        message: str,
        error_category: ErrorCategory = ErrorCategory.ENVIRONMENT_ERROR,
        task_id: str = "",
    ) -> None:
        self.error_category = error_category
        self.task_id = task_id
        super().__init__(
            f"Environment setup failed for task '{task_id}': {message}. "
            f"Error category: {error_category.value}. The agent will NOT be invoked."
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

    Supports multi-language image selection, network access control,
    custom working directories, and structured health checks.

    Args:
        image: Docker image to use (default ``"python:3.12-slim"``).
        cpu_limit: CPU limit in cores (default ``1.0``).
        memory_limit: Memory limit string, e.g. ``"4g"``, ``"512m"`` (default ``"4g"``).
        auto_pull: If ``True``, pull missing images automatically (default ``True``).
        keep_workspace_on_failure: If ``True``, preserve workspace volumes on cleanup
            for debugging (default ``False``).
        workspace_path: Path inside the container for the workspace
            (default ``"/workspace"``).
        working_dir: Working directory inside the container for command execution
            (default ``workspace_path``).
        network_access: If ``False`` (default), run with ``--network=none``.
            If ``True``, allow full network access.
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
        working_dir: str | None = None,
        network_access: bool = False,
        artifacts: dict[str, str] | None = None,
        result_parser: ResultParser | None = None,
    ) -> None:
        self._image: str = image
        self._cpu_limit: float = cpu_limit
        self._memory_limit: str = memory_limit
        self._auto_pull: bool = auto_pull
        self._keep_workspace_on_failure: bool = keep_workspace_on_failure
        self._workspace_path: str = workspace_path
        self._working_dir: str = working_dir if working_dir is not None else workspace_path
        self._network_access: bool = network_access
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
    def working_dir(self) -> str:
        """Working directory inside the container for command execution."""
        return self._working_dir

    @property
    def network_access(self) -> bool:
        """Whether network access is enabled (True) or disabled (False)."""
        return self._network_access

    @property
    def network_mode(self) -> str | None:
        """Docker network mode. ``'none'`` when network_access=False, ``None`` otherwise."""
        if not self._network_access:
            return "none"
        return None

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
    # Factory method: create_from_task (VAL-ENV-01)
    # ------------------------------------------------------------------

    @classmethod
    def create_from_task(
        cls,
        task: Task,
        *,
        language_overrides: dict[str, str] | None = None,
        network_access: bool = False,
        working_dir: str | None = None,
        cpu_limit: float = 1.0,
        memory_limit: str = "4g",
        auto_pull: bool = True,
        artifacts: dict[str, str] | None = None,
    ) -> DockerRunner:
        """Create a DockerRunner configured for a specific task's language.

        Uses the task's ``language`` field to select the appropriate Docker
        image via ``LanguageImageConfig``.

        Args:
            task: The task to create a runner for.
            language_overrides: Custom language-to-image mappings.
            network_access: Whether to enable network access.
            working_dir: Custom working directory inside the container.
            cpu_limit: CPU limit in cores.
            memory_limit: Memory limit string.
            auto_pull: Whether to auto-pull missing images.
            artifacts: Artifacts to inject into the workspace.

        Returns:
            A ``DockerRunner`` configured with the correct image for the
            task's language.
        """
        lang_config = LanguageImageConfig(overrides=language_overrides or {})
        image = lang_config.get_image(task.language)

        return cls(
            image=image,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            auto_pull=auto_pull,
            network_access=network_access,
            working_dir=working_dir,
            artifacts=artifacts,
        )

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
    # Health check (VAL-ENV-08)
    # ------------------------------------------------------------------

    def health_check(self) -> EnvironmentHealth:
        """Check the health of the Docker environment.

        Returns a structured ``EnvironmentHealth`` result with:
        - ``docker_available``: Whether the Docker daemon is reachable.
        - ``images_ready``: Whether the configured image is available locally.
        - ``disk_free_gb``: Free disk space in gigabytes.

        Returns:
            An ``EnvironmentHealth`` model with structured diagnostics.
        """
        docker_available = self.check_docker_available()
        images_ready = False

        if docker_available:
            try:
                client = self._get_client()
                images = client.images.list(name=self._image)
                images_ready = len(images) > 0
            except Exception:
                images_ready = False

        # Get disk free space
        disk_free_gb = 0.0
        try:
            usage = shutil.disk_usage("/")
            disk_free_gb = round(usage.free / (1024**3), 2)
        except Exception:
            disk_free_gb = 0.0

        return EnvironmentHealth(
            docker_available=docker_available,
            images_ready=images_ready,
            disk_free_gb=disk_free_gb,
        )

    # ------------------------------------------------------------------
    # BaseEnvironment implementation
    # ------------------------------------------------------------------

    def prepare(self, task: Task) -> None:
        """Prepare a Docker container for task execution.

        Ensures the configured image is available, creates an isolated
        workspace volume, injects artifacts, and creates the container.
        If the task has a ``setup_command``, it is executed.  A failing
        setup command raises ``EnvironmentSetupError`` with
        ``error_category=ENVIRONMENT_ERROR``, and the container is
        cleaned up without invoking the agent (VAL-ENV-03).

        Args:
            task: The task to prepare the environment for.

        Raises:
            DockerDaemonError: If Docker daemon is unreachable.
            DockerImageError: If the image is missing and auto-pull is disabled.
            EnvironmentSetupError: If the setup command fails (VAL-ENV-03).
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

        # Build container creation kwargs
        create_kwargs: dict[str, Any] = {
            "image": self._image,
            "name": container_name,
            "volumes": {volume_name: {"bind": self._workspace_path, "mode": "rw"}},
            "working_dir": self._working_dir,
            "mem_limit": self._memory_limit,
            "nano_cpus": int(self._cpu_limit * 1e9),
            "detach": True,
            "stdin_open": True,
            "tty": False,
        }

        # 5. Apply network access control (VAL-ENV-04)
        network_mode = self.network_mode
        if network_mode is not None:
            create_kwargs["network_mode"] = network_mode

        try:
            container: docker.models.containers.Container = client.containers.create(
                **create_kwargs,
            )
            self._container_id = container.id
        except Exception as exc:
            # Clean up the volume if container creation fails
            with contextlib.suppress(Exception):
                workspace_vol.remove(force=True)
            raise DockerDaemonError(f"Failed to create container: {exc}") from exc

        self._prepared = True
        logger.info(
            "Prepared Docker container '%s' (image=%s, cpu=%.1f, mem=%s, "
            "workspace=%s, working_dir=%s, network=%s)",
            container_name,
            self._image,
            self._cpu_limit,
            self._memory_limit,
            self._workspace_path,
            self._working_dir,
            "none" if self.network_mode == "none" else "default",
        )

        # 6. Run setup_command if specified (VAL-ENV-03)
        if task.setup_command:
            logger.info("Running setup command for task '%s': %s", task.id, task.setup_command)
            output = self.execute(task.setup_command, timeout=float(task.timeout))
            if output.exit_code != 0:
                stderr_preview = output.stderr[:200] if output.stderr else "(no stderr)"
                msg = f"Setup command exited with code {output.exit_code}: {stderr_preview}"
                logger.error(msg)
                # Clean up container and volume before raising
                self.cleanup()
                raise EnvironmentSetupError(
                    message=msg,
                    error_category=ErrorCategory.ENVIRONMENT_ERROR,
                    task_id=task.id,
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
                    workdir=self._working_dir,
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
            f"network_access={self._network_access!r}, "
            f"working_dir={self._working_dir!r}, "
            f"keep_workspace_on_failure={self._keep_workspace_on_failure!r})"
        )


__all__ = [
    "DockerDaemonError",
    "DockerImageError",
    "DockerRunner",
    "EnvironmentSetupError",
]
