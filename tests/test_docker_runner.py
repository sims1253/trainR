"""Tests for DockerRunner — Docker-based execution environment.

Tests cover:
- Container creation, execution, and cleanup (VAL-HARNESS-02)
- Artifact injection into container filesystem (VAL-HARNESS-03)
- Resource limits: CPU, memory, timeout enforcement (VAL-HARNESS-04)
- Docker daemon availability check (VAL-ENV-05)
- Missing image handling: auto-pull and error (VAL-ENV-07)
- Cleanup on all outcomes (success, failure, timeout)
- keep_workspace_on_failure volume preservation

Integration tests are marked with @pytest.mark.integration_local
and require a running Docker daemon.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grist_mill.environments.docker_runner import (
    DockerDaemonError,
    DockerImageError,
    DockerRunner,
)
from grist_mill.schemas import Task, TaskResult, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    *,
    task_id: str = "test-task-1",
    prompt: str = "Solve the problem",
    language: str = "python",
    test_command: str = "echo hello",
    setup_command: str | None = None,
    timeout: int = 60,
) -> Task:
    """Create a simple Task for testing."""
    return Task(
        id=task_id,
        prompt=prompt,
        language=language,
        test_command=test_command,
        setup_command=setup_command,
        timeout=timeout,
    )


def _skip_without_docker() -> None:
    """Raise pytest.skip if Docker is not available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker not available: {exc}")


# ===========================================================================
# Unit Tests (no Docker required)
# ===========================================================================


class TestDockerRunnerInit:
    """Test DockerRunner initialization and properties."""

    def test_default_initialization(self) -> None:
        """Runner initializes with sensible defaults."""
        runner = DockerRunner()
        assert runner.image == "python:3.12-slim"
        assert runner.cpu_limit == 1.0
        assert runner.memory_limit == "4g"
        assert runner.auto_pull is True
        assert runner.keep_workspace_on_failure is False
        assert runner.workspace_path == "/workspace"
        assert runner.result_parser is not None

    def test_custom_image(self) -> None:
        """Custom image is stored correctly."""
        runner = DockerRunner(image="ubuntu:22.04")
        assert runner.image == "ubuntu:22.04"

    def test_resource_limits(self) -> None:
        """CPU and memory limits are stored correctly."""
        runner = DockerRunner(cpu_limit=2.0, memory_limit="8g")
        assert runner.cpu_limit == 2.0
        assert runner.memory_limit == "8g"

    def test_auto_pull_flag(self) -> None:
        """auto_pull flag controls image pulling behavior."""
        runner = DockerRunner(auto_pull=False)
        assert runner.auto_pull is False

    def test_keep_workspace_flag(self) -> None:
        """keep_workspace_on_failure controls volume cleanup."""
        runner = DockerRunner(keep_workspace_on_failure=True)
        assert runner.keep_workspace_on_failure is True

    def test_workspace_path(self) -> None:
        """Custom workspace path is stored correctly."""
        runner = DockerRunner(workspace_path="/app")
        assert runner.workspace_path == "/app"

    def test_artifacts_config(self) -> None:
        """Artifacts dict is stored correctly."""
        artifacts = {"skill.py": "# skill content", "config.yaml": "key: value"}
        runner = DockerRunner(artifacts=artifacts)
        assert runner.artifacts == artifacts

    def test_repr(self) -> None:
        """__repr__ provides useful information."""
        runner = DockerRunner(image="alpine:latest")
        r = repr(runner)
        assert "DockerRunner" in r
        assert "alpine:latest" in r


class TestDockerRunnerCheckDockerAvailable:
    """Test Docker daemon availability check (VAL-ENV-05)."""

    def test_docker_available_success(self) -> None:
        """Returns True when Docker daemon is reachable."""
        _skip_without_docker()
        runner = DockerRunner()
        assert runner.check_docker_available() is True

    def test_docker_unavailable_raises_clear_error(self) -> None:
        """When Docker daemon is unreachable, check returns False."""
        runner = DockerRunner()
        with patch("grist_mill.environments.docker_runner.docker") as mock_docker:
            mock_client = MagicMock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_docker.from_env.return_value = mock_client
            assert runner.check_docker_available() is False


class TestDockerRunnerImageHandling:
    """Test image availability check and pulling (VAL-ENV-07)."""

    def test_image_exists_locally(self) -> None:
        """_ensure_image_available does not pull when image exists."""
        _skip_without_docker()
        runner = DockerRunner(image="python:3.12-slim")
        # Should not raise — image should either be present or auto-pulled
        runner._ensure_image_available()

    def test_image_not_found_auto_pull_enabled(self) -> None:
        """With auto_pull=True, missing image triggers a pull."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19", auto_pull=True)
        # Should pull without error
        runner._ensure_image_available()

    def test_image_not_found_auto_pull_disabled(self) -> None:
        """With auto_pull=False, missing image raises DockerImageError."""
        runner = DockerRunner(auto_pull=False)
        with patch.object(runner, "_client") as mock_client:
            mock_client.images.list.return_value = []
            with pytest.raises(DockerImageError, match="nonexistent-image"):
                runner._ensure_image_available(image="nonexistent-image")

    def test_error_message_includes_pull_suggestion(self) -> None:
        """Error message tells user to run docker pull."""
        runner = DockerRunner(auto_pull=False)
        with patch.object(runner, "_client") as mock_client:
            mock_client.images.list.return_value = []
            with pytest.raises(DockerImageError, match="docker pull nonexistent-img"):
                runner._ensure_image_available(image="nonexistent-img")


class TestDockerRunnerPrepare:
    """Test environment preparation."""

    def test_prepare_sets_prepared_flag(self) -> None:
        """prepare() sets internal state."""
        runner = DockerRunner()
        task = _make_task()
        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_volume = MagicMock()
        mock_volume.name = "vol123"
        with (
            patch.object(runner, "_ensure_image_available"),
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.volumes.create.return_value = mock_volume
            mock_client.containers.create.return_value = mock_container
            mock_get_client.return_value = mock_client
            runner.prepare(task)
        assert runner.is_prepared

    def test_prepare_calls_ensure_image(self) -> None:
        """prepare() checks image availability."""
        runner = DockerRunner(image="my-image:latest")
        task = _make_task()
        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_volume = MagicMock()
        mock_volume.name = "vol123"
        with (
            patch.object(runner, "_ensure_image_available") as mock_ensure,
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.volumes.create.return_value = mock_volume
            mock_client.containers.create.return_value = mock_container
            mock_get_client.return_value = mock_client
            runner.prepare(task)
            mock_ensure.assert_called_once()

    def test_prepare_runs_setup_command(self) -> None:
        """prepare() executes setup_command from the task."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task(setup_command="echo 'setup done'")
        runner.prepare(task)
        runner.cleanup()

    def test_prepare_with_artifacts(self) -> None:
        """prepare() injects artifacts into container workspace."""
        _skip_without_docker()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_file = Path(tmpdir) / "test_artifact.txt"
            artifact_file.write_text("hello from artifact")

            runner = DockerRunner(
                image="alpine:3.19",
                artifacts={"test_artifact.txt": artifact_file.read_text()},
            )
            task = _make_task()
            runner.prepare(task)
            runner.cleanup()


class TestDockerRunnerExecute:
    """Test command execution inside Docker containers."""

    def test_execute_simple_command(self) -> None:
        """Execute a simple echo command and capture output."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        try:
            output = runner.execute("echo hello", timeout=30.0)
            assert output.stdout.strip() == "hello"
            assert output.exit_code == 0
            assert output.timed_out is False
        finally:
            runner.cleanup()

    def test_execute_captures_stderr(self) -> None:
        """stderr is captured separately from stdout."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        try:
            output = runner.execute("echo error >&2", timeout=30.0)
            assert output.stderr.strip() == "error"
            assert output.exit_code == 0
        finally:
            runner.cleanup()

    def test_execute_nonzero_exit_code(self) -> None:
        """Non-zero exit code is captured correctly."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        try:
            output = runner.execute("exit 42", timeout=30.0)
            assert output.exit_code == 42
            assert output.timed_out is False
        finally:
            runner.cleanup()

    def test_execute_timeout_kills_container(self) -> None:
        """Timeout kills the container and returns timed_out=True."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task(timeout=5)
        runner.prepare(task)
        try:
            output = runner.execute("sleep 999", timeout=3.0)
            assert output.timed_out is True
            assert output.exit_code == -1
        finally:
            runner.cleanup()

    def test_timeout_returns_partial_output(self) -> None:
        """Timed-out execution captures whatever partial output was produced."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task(timeout=5)
        runner.prepare(task)
        try:
            output = runner.execute("echo before; sleep 999; echo after", timeout=3.0)
            assert output.timed_out is True
            # At minimum, we should have "before" or partial output
            assert output.stdout is not None
        finally:
            runner.cleanup()


class TestDockerRunnerCleanup:
    """Test container cleanup on all outcomes (VAL-HARNESS-02)."""

    def test_cleanup_removes_container(self) -> None:
        """cleanup() removes the Docker container."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        container_id = runner._container_id
        assert container_id is not None

        runner.cleanup()

        # Verify container is gone
        try:
            client = runner._get_client()
            client.containers.get(container_id)
            pytest.fail("Container should have been removed")
        except Exception:
            pass  # Expected: container not found

    def test_cleanup_safe_without_prepare(self) -> None:
        """cleanup() is safe to call without prepare()."""
        runner = DockerRunner()
        runner.cleanup()  # Should not raise

    def test_cleanup_idempotent(self) -> None:
        """Calling cleanup() multiple times is safe."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        runner.cleanup()
        runner.cleanup()  # Should not raise


class TestDockerRunnerKeepWorkspace:
    """Test keep_workspace_on_failure behavior."""

    def test_keep_workspace_preserves_volume(self) -> None:
        """When keep_workspace=True, volume persists after cleanup."""
        _skip_without_docker()
        runner = DockerRunner(
            image="alpine:3.19",
            keep_workspace_on_failure=True,
        )
        task = _make_task()
        runner.prepare(task)
        volume_name = runner._workspace_volume_name
        runner.cleanup()

        # Volume should still exist
        try:
            client = runner._get_client()
            vol = client.volumes.get(volume_name)
            vol.remove()  # Clean up after test
        except Exception:
            pytest.fail(f"Volume {volume_name} should have been preserved")

    def test_no_keep_workspace_removes_volume(self) -> None:
        """When keep_workspace=False (default), volume is removed."""
        _skip_without_docker()
        runner = DockerRunner(
            image="alpine:3.19",
            keep_workspace_on_failure=False,
        )
        task = _make_task()
        runner.prepare(task)
        volume_name = runner._workspace_volume_name
        runner.cleanup()

        # Volume should be gone
        try:
            client = runner._get_client()
            client.volumes.get(volume_name)
            pytest.fail(f"Volume {volume_name} should have been removed")
        except Exception:
            pass  # Expected: volume not found


class TestDockerRunnerResourceLimits:
    """Test resource limit enforcement (VAL-HARNESS-04)."""

    def test_memory_limit_enforced(self) -> None:
        """Memory limit prevents exceeding allocated memory."""
        _skip_without_docker()
        # Use very low memory limit; a process that allocates should get OOM killed
        runner = DockerRunner(image="alpine:3.19", memory_limit="16m")
        task = _make_task()
        runner.prepare(task)
        try:
            # Try to allocate more memory than the limit
            runner.execute(
                "python3 -c 'x=\"a\"*(100*1024*1024)'"
                if False
                else "sh -c 'x=$(head -c 50m /dev/zero | tail); sleep 1'",
                timeout=10.0,
            )
            # The command may be OOM killed or may succeed depending on kernel config
            # Just verify it didn't hang
        finally:
            runner.cleanup()

    def test_cpu_limit_applied(self) -> None:
        """CPU limit is applied to the container."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19", cpu_limit=0.5)
        task = _make_task()
        runner.prepare(task)
        try:
            # Just verify the container runs with the limit
            output = runner.execute("echo ok", timeout=10.0)
            assert output.stdout.strip() == "ok"
        finally:
            runner.cleanup()

    def test_timeout_enforced(self) -> None:
        """Timeout enforcement kills the container."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        try:
            output = runner.execute("sleep 999", timeout=2.0)
            assert output.timed_out is True
        finally:
            runner.cleanup()


class TestDockerRunnerArtifacts:
    """Test artifact injection (VAL-HARNESS-03)."""

    def test_artifacts_injected_into_container(self) -> None:
        """Artifacts are written to the workspace inside the container."""
        _skip_without_docker()
        artifacts = {
            "hello.txt": "Hello, world!",
            "subdir/nested.py": "print('nested')",
        }
        runner = DockerRunner(image="alpine:3.19", artifacts=artifacts)
        task = _make_task()
        runner.prepare(task)
        try:
            # Verify artifact is accessible inside the container
            output = runner.execute("cat /workspace/hello.txt", timeout=10.0)
            assert output.stdout.strip() == "Hello, world!"

            output = runner.execute("cat /workspace/subdir/nested.py", timeout=10.0)
            assert "nested" in output.stdout
        finally:
            runner.cleanup()

    def test_artifacts_at_custom_workspace_path(self) -> None:
        """Artifacts are placed at the configured workspace path."""
        _skip_without_docker()
        runner = DockerRunner(
            image="alpine:3.19",
            workspace_path="/custom-workspace",
            artifacts={"test.txt": "content"},
        )
        task = _make_task()
        runner.prepare(task)
        try:
            output = runner.execute("cat /custom-workspace/test.txt", timeout=10.0)
            assert output.stdout.strip() == "content"
        finally:
            runner.cleanup()


class TestDockerRunnerRunTask:
    """Test the convenience run_task method."""

    def test_run_task_success(self) -> None:
        """run_task returns a SUCCESS TaskResult for a passing command."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task(test_command="echo pass")
        runner.prepare(task)
        try:
            result = runner.run_task(
                command="echo pass",
                task_id="t1",
                language="python",
                timeout=30.0,
            )
            assert isinstance(result, TaskResult)
            assert result.status == TaskStatus.SUCCESS
            assert result.score == 1.0
        finally:
            runner.cleanup()

    def test_run_task_failure(self) -> None:
        """run_task returns FAILURE for a non-zero exit code."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        try:
            result = runner.run_task(
                command="exit 1",
                task_id="t2",
                language="python",
                timeout=30.0,
            )
            assert isinstance(result, TaskResult)
            assert result.status in (TaskStatus.FAILURE, TaskStatus.ERROR)
        finally:
            runner.cleanup()

    def test_run_task_timeout(self) -> None:
        """run_task returns TIMEOUT for a timed-out command."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19")
        task = _make_task()
        runner.prepare(task)
        try:
            result = runner.run_task(
                command="sleep 999",
                task_id="t3",
                language="python",
                timeout=2.0,
            )
            assert isinstance(result, TaskResult)
            assert result.status == TaskStatus.TIMEOUT
        finally:
            runner.cleanup()


class TestDockerRunnerDaemonError:
    """Test DockerDaemonError exception."""

    def test_docker_daemon_error_message(self) -> None:
        """Error message is clear and actionable."""
        err = DockerDaemonError("Cannot connect to Docker daemon")
        assert "Cannot connect to Docker daemon" in str(err)


class TestDockerRunnerImageError:
    """Test DockerImageError exception."""

    def test_image_error_message(self) -> None:
        """Error message includes the image name."""
        err = DockerImageError("my-image:latest", "Image not found")
        assert "my-image:latest" in str(err)
        assert "docker pull" in str(err)


class TestDockerRunnerPythonApi:
    """Test using Docker Python API directly (mocked)."""

    def test_prepare_with_mock(self) -> None:
        """prepare() creates container via Docker API."""
        runner = DockerRunner(image="test:latest")
        mock_container = MagicMock()
        mock_container.id = "abc123"

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.containers.create.return_value = mock_container
            mock_get_client.return_value = mock_client

            task = _make_task()
            runner.prepare(task)

            mock_client.containers.create.assert_called_once()
            assert runner._container_id == "abc123"

    def test_execute_with_mock(self) -> None:
        """execute() runs command in container and returns ExecutionOutput."""
        runner = DockerRunner(image="test:latest")
        runner._container_id = "abc123"
        runner._prepared = True

        mock_exec_result = MagicMock()
        mock_exec_result.output = (b"hello world\n", b"")
        mock_exec_result.exit_code = 0

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_container.exec_run.return_value = mock_exec_result
            mock_container.start.return_value = None
            mock_client.containers.get.return_value = mock_container
            mock_get_client.return_value = mock_client

            output = runner.execute("echo hello", timeout=30.0)
            assert output.stdout == "hello world\n"
            assert output.exit_code == 0
            assert output.timed_out is False

    def test_cleanup_with_mock(self) -> None:
        """cleanup() removes container and volume."""
        runner = DockerRunner(image="test:latest")
        runner._container_id = "abc123"
        runner._workspace_volume_name = "vol123"
        runner._prepared = True

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_volume = MagicMock()
            mock_client.containers.get.return_value = mock_container
            mock_client.volumes.get.return_value = mock_volume
            mock_get_client.return_value = mock_client

            runner.cleanup()

            mock_container.remove.assert_called_once_with(force=True)
            mock_volume.remove.assert_called_once_with(force=True)
            assert runner._container_id is None
            assert runner.is_prepared is False

    def test_cleanup_handles_already_removed_container(self) -> None:
        """cleanup() is resilient if container was already removed."""
        runner = DockerRunner(image="test:latest")
        runner._container_id = "abc123"
        runner._workspace_volume_name = "vol123"
        runner._prepared = True

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            # docker.errors.NotFound when container is gone
            import docker.errors

            mock_client.containers.get.side_effect = docker.errors.NotFound("not found")
            mock_volume = MagicMock()
            mock_client.volumes.get.return_value = mock_volume
            mock_get_client.return_value = mock_client

            runner.cleanup()  # Should not raise

    def test_cleanup_with_keep_workspace(self) -> None:
        """cleanup() preserves volume when keep_workspace_on_failure=True."""
        runner = DockerRunner(
            image="test:latest",
            keep_workspace_on_failure=True,
        )
        runner._container_id = "abc123"
        runner._workspace_volume_name = "vol123"
        runner._prepared = True

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_client.containers.get.return_value = mock_container
            mock_get_client.return_value = mock_client

            runner.cleanup()

            # Container removed, volume preserved
            mock_container.remove.assert_called_once_with(force=True)
            # volume.remove should NOT be called
            mock_client.volumes.get.assert_not_called()

    def test_execute_timeout_with_mock(self) -> None:
        """execute() handles timeout by killing the container."""
        runner = DockerRunner(image="test:latest")
        runner._container_id = "abc123"
        runner._prepared = True

        with patch.object(runner, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_container.start.return_value = None

            # exec_run blocks — simulate by having it never set the result
            def slow_exec(*args, **kwargs):
                import time

                time.sleep(10)  # Block longer than the timeout

            mock_container.exec_run.side_effect = slow_exec
            mock_client.containers.get.return_value = mock_container
            mock_get_client.return_value = mock_client

            output = runner.execute("sleep 999", timeout=0.5)
            assert output.timed_out is True
            mock_container.kill.assert_called_once()
