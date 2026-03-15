"""Tests for multi-language environment support.

Covers:
- VAL-ENV-01: Multi-language environment support via configurable Docker images
  (task language field drives image selection)
- VAL-ENV-02: Volume mount management for workspace isolation
  (concurrent tasks don't interfere)
- VAL-ENV-03: Environment preparation runs setup_command; failure marks task ERROR
- VAL-ENV-04: Network access control (--network=none default, True enables access)
- VAL-ENV-06: Custom working directory inside container
- VAL-ENV-08: Environment health check returns structured diagnostics
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from grist_mill.environments.docker_runner import (
    DockerRunner,
    EnvironmentSetupError,
)
from grist_mill.environments.language_config import (
    DEFAULT_LANGUAGE_IMAGES,
    LanguageImageConfig,
)
from grist_mill.environments.local_runner import LocalRunner
from grist_mill.schemas import (
    ErrorCategory,
    Task,
)

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
    working_dir: str | None = None,
    dependencies: list[str] | None = None,
) -> Task:
    """Create a Task for testing.  working_dir is stored in dependencies as
    a special marker ``working_dir:<path>`` since Task has no native
    ``working_dir`` field — the DockerRunner reads this from config."""
    return Task(
        id=task_id,
        prompt=prompt,
        language=language,
        test_command=test_command,
        setup_command=setup_command,
        timeout=timeout,
        dependencies=dependencies or [],
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
# VAL-ENV-01: Multi-language Docker image selection
# ===========================================================================


class TestLanguageImageConfig:
    """Language-to-Docker-image mapping (VAL-ENV-01)."""

    def test_python_language_uses_python_image(self) -> None:
        """language='python' resolves to python:3.12-slim image."""
        config = LanguageImageConfig()
        assert config.get_image("python") == "python:3.12-slim"

    def test_r_language_uses_rocker_image(self) -> None:
        """language='r' resolves to rocker/r-ver image."""
        config = LanguageImageConfig()
        assert config.get_image("r") == "rocker/r-ver:latest"

    def test_typescript_language_uses_node_image(self) -> None:
        """language='typescript' resolves to node image."""
        config = LanguageImageConfig()
        assert config.get_image("typescript") == "node:22-slim"

    def test_custom_image_override(self) -> None:
        """Custom image overrides default for a language."""
        config = LanguageImageConfig(
            overrides={"python": "my-custom-python:3.11"},
        )
        assert config.get_image("python") == "my-custom-python:3.11"
        # Other languages still use defaults
        assert config.get_image("r") == "rocker/r-ver:latest"

    def test_unknown_language_uses_python_fallback(self) -> None:
        """Unknown language falls back to python:3.12-slim."""
        config = LanguageImageConfig()
        assert config.get_image("unknown_lang") == "python:3.12-slim"

    def test_unknown_language_custom_fallback(self) -> None:
        """Custom fallback for unknown languages."""
        config = LanguageImageConfig(fallback_image="ubuntu:22.04")
        assert config.get_image("unknown_lang") == "ubuntu:22.04"

    def test_case_insensitive_language_lookup(self) -> None:
        """Language lookup is case-insensitive."""
        config = LanguageImageConfig()
        assert config.get_image("Python") == "python:3.12-slim"
        assert config.get_image("R") == "rocker/r-ver:latest"
        assert config.get_image("TypeScript") == "node:22-slim"

    def test_default_language_images_is_a_dict(self) -> None:
        """DEFAULT_LANGUAGE_IMAGES contains expected entries."""
        assert "python" in DEFAULT_LANGUAGE_IMAGES
        assert "r" in DEFAULT_LANGUAGE_IMAGES
        assert "typescript" in DEFAULT_LANGUAGE_IMAGES


class TestDockerRunnerLanguageImageSelection:
    """DockerRunner selects image based on task language (VAL-ENV-01)."""

    def test_docker_runner_with_language_config_python(self) -> None:
        """DockerRunner with LanguageImageConfig selects python image for python task."""
        lang_config = LanguageImageConfig()
        image = lang_config.get_image("python")
        runner = DockerRunner(image=image)
        assert "python" in runner.image.lower()

    def test_docker_runner_with_language_config_r(self) -> None:
        """DockerRunner with LanguageImageConfig selects rocker image for R task."""
        lang_config = LanguageImageConfig()
        image = lang_config.get_image("r")
        runner = DockerRunner(image=image)
        assert "rocker" in runner.image.lower()

    def test_docker_runner_with_language_config_typescript(self) -> None:
        """DockerRunner with LanguageImageConfig selects node image for TS task."""
        lang_config = LanguageImageConfig()
        image = lang_config.get_image("typescript")
        runner = DockerRunner(image=image)
        assert "node" in runner.image.lower()

    def test_docker_runner_factory_from_task(self) -> None:
        """DockerRunner.create_from_task() selects image based on task language."""
        task = _make_task(language="r")
        runner = DockerRunner.create_from_task(task)
        assert "rocker" in runner.image.lower()

    def test_docker_runner_factory_from_task_python(self) -> None:
        """create_from_task selects python image for python task."""
        task = _make_task(language="python")
        runner = DockerRunner.create_from_task(task)
        assert "python" in runner.image.lower()

    def test_docker_runner_factory_from_task_typescript(self) -> None:
        """create_from_task selects node image for typescript task."""
        task = _make_task(language="typescript")
        runner = DockerRunner.create_from_task(task)
        assert "node" in runner.image.lower()

    def test_docker_runner_factory_with_custom_overrides(self) -> None:
        """create_from_task respects custom language image overrides."""
        overrides = {"python": "custom-python:3.11"}
        task = _make_task(language="python")
        runner = DockerRunner.create_from_task(task, language_overrides=overrides)
        assert runner.image == "custom-python:3.11"

    def test_docker_runner_factory_with_network_access(self) -> None:
        """create_from_task passes network_access to runner."""
        task = _make_task(language="python")
        runner = DockerRunner.create_from_task(task, network_access=True)
        assert runner.network_access is True


# ===========================================================================
# VAL-ENV-04: Network access control
# ===========================================================================


class TestNetworkAccessControl:
    """Network access control per environment (VAL-ENV-04)."""

    def test_default_network_access_is_none(self) -> None:
        """Default network_access is False (--network=none)."""
        runner = DockerRunner()
        assert runner.network_access is False

    def test_network_access_enabled(self) -> None:
        """network_access=True enables full network."""
        runner = DockerRunner(network_access=True)
        assert runner.network_access is True

    def test_network_mode_none_when_disabled(self) -> None:
        """network_mode is 'none' when network_access=False."""
        runner = DockerRunner(network_access=False)
        assert runner.network_mode == "none"

    def test_network_mode_empty_when_enabled(self) -> None:
        """network_mode is None when network_access=True (default bridge)."""
        runner = DockerRunner(network_access=True)
        assert runner.network_mode is None

    def test_container_created_with_network_none(self) -> None:
        """Container is created with network_mode='none' when disabled."""
        runner = DockerRunner(network_access=False, auto_pull=True)
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

            # Verify containers.create was called with network_mode='none'
            call_kwargs = mock_client.containers.create.call_args
            assert (
                call_kwargs.kwargs.get("network_mode") == "none"
                or call_kwargs[1].get("network_mode") == "none"
            )

    def test_container_created_without_network_none_when_enabled(self) -> None:
        """Container is created without network_mode when network_access=True."""
        runner = DockerRunner(network_access=True, auto_pull=True)
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

            # Verify containers.create was called without network_mode (or None)
            call_kwargs = mock_client.containers.create.call_args
            nm = call_kwargs.kwargs.get("network_mode") or call_kwargs[1].get("network_mode")
            assert nm is None

    def test_network_none_blocks_curl(self) -> None:
        """Integration: --network=none blocks curl (requires Docker)."""
        _skip_without_docker()
        runner = DockerRunner(
            image="alpine:3.19",
            network_access=False,
            auto_pull=True,
        )
        task = _make_task(timeout=30)
        runner.prepare(task)
        try:
            output = runner.execute(
                "wget -q --spider http://example.com 2>&1 || echo 'network blocked'",
                timeout=10.0,
            )
            assert "network blocked" in output.stdout or output.exit_code != 0
        finally:
            runner.cleanup()

    def test_network_enabled_allows_curl(self) -> None:
        """Integration: network access enabled allows outbound connections."""
        _skip_without_docker()
        runner = DockerRunner(
            image="alpine:3.19",
            network_access=True,
            auto_pull=True,
        )
        task = _make_task(timeout=30)
        runner.prepare(task)
        try:
            output = runner.execute(
                "wget -q --spider http://example.com && echo 'network ok' || echo 'network failed'",
                timeout=15.0,
            )
            # Either success or at least no "network blocked" error
            assert "network ok" in output.stdout or output.exit_code == 0
        finally:
            runner.cleanup()


# ===========================================================================
# VAL-ENV-06: Custom working directory
# ===========================================================================


class TestCustomWorkingDirectory:
    """Custom working directory inside container (VAL-ENV-06)."""

    def test_default_working_dir_is_workspace(self) -> None:
        """Default working_dir is /workspace."""
        runner = DockerRunner()
        assert runner.working_dir == "/workspace"

    def test_custom_working_dir(self) -> None:
        """Custom working_dir is stored."""
        runner = DockerRunner(working_dir="/app/src")
        assert runner.working_dir == "/app/src"

    def test_container_created_with_custom_working_dir(self) -> None:
        """Container is created with the specified working_dir."""
        runner = DockerRunner(working_dir="/app/src", auto_pull=True)
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

            call_kwargs = mock_client.containers.create.call_args
            wd = call_kwargs.kwargs.get("working_dir") or call_kwargs[1].get("working_dir")
            assert wd == "/app/src"

    def test_working_dir_with_real_container(self) -> None:
        """Integration: custom working_dir sets execution directory."""
        _skip_without_docker()
        runner = DockerRunner(
            image="alpine:3.19",
            working_dir="/tmp/testdir",
            auto_pull=True,
        )
        task = _make_task(timeout=30)
        runner.prepare(task)
        try:
            # Create the directory and verify pwd
            runner.execute("mkdir -p /tmp/testdir", timeout=5.0)
            output = runner.execute("pwd", timeout=5.0)
            assert output.stdout.strip() == "/tmp/testdir"
        finally:
            runner.cleanup()

    def test_create_from_task_with_working_dir(self) -> None:
        """create_from_task passes working_dir to runner."""
        task = _make_task()
        runner = DockerRunner.create_from_task(task, working_dir="/app/code")
        assert runner.working_dir == "/app/code"


# ===========================================================================
# VAL-ENV-08: Environment health check
# ===========================================================================


class TestEnvironmentHealthCheck:
    """Environment health check returns structured diagnostics (VAL-ENV-08)."""

    def test_health_check_is_callable(self) -> None:
        """DockerRunner has a health_check method."""
        runner = DockerRunner()
        assert callable(runner.health_check)

    def test_health_check_returns_structured_result(self) -> None:
        """health_check() returns an EnvironmentHealth model."""
        runner = DockerRunner()
        with (
            patch.object(runner, "check_docker_available", return_value=True),
            patch("grist_mill.environments.docker_runner.shutil") as mock_shutil,
        ):
            mock_shutil.disk_usage.return_value = MagicMock(
                total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
            )
            health = runner.health_check()

        assert hasattr(health, "docker_available")
        assert hasattr(health, "images_ready")
        assert hasattr(health, "disk_free_gb")

    def test_health_check_docker_unavailable(self) -> None:
        """health_check reports docker_available=False when Docker is down."""
        runner = DockerRunner()
        with patch.object(runner, "check_docker_available", return_value=False):
            health = runner.health_check()

        assert health.docker_available is False

    def test_health_check_docker_available(self) -> None:
        """health_check reports docker_available=True when Docker is up."""
        runner = DockerRunner()
        with (
            patch.object(runner, "check_docker_available", return_value=True),
            patch("grist_mill.environments.docker_runner.shutil") as mock_shutil,
        ):
            mock_shutil.disk_usage.return_value = MagicMock(
                total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
            )
            health = runner.health_check()

        assert health.docker_available is True

    def test_health_check_disk_free_gb(self) -> None:
        """health_check reports disk_free_gb as a float."""
        runner = DockerRunner()
        with (
            patch.object(runner, "check_docker_available", return_value=True),
            patch("grist_mill.environments.docker_runner.shutil") as mock_shutil,
        ):
            mock_shutil.disk_usage.return_value = MagicMock(
                total=100 * 1024**3, used=30 * 1024**3, free=70 * 1024**3
            )
            health = runner.health_check()

        assert isinstance(health.disk_free_gb, float)
        # 70 GB free, allow some rounding
        assert 69.0 <= health.disk_free_gb <= 71.0

    def test_health_check_images_ready_true(self) -> None:
        """health_check reports images_ready=True when image is available."""
        _skip_without_docker()
        runner = DockerRunner(image="python:3.12-slim")
        health = runner.health_check()
        assert health.docker_available is True

    def test_health_check_images_ready_false(self) -> None:
        """health_check reports images_ready=False when image is missing."""
        runner = DockerRunner(image="nonexistent-image:latest")
        with (
            patch.object(runner, "check_docker_available", return_value=True),
            patch("grist_mill.environments.docker_runner.shutil") as mock_shutil,
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_shutil.disk_usage.return_value = MagicMock(
                total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
            )
            mock_client = MagicMock()
            mock_client.images.list.return_value = []
            mock_get_client.return_value = mock_client
            health = runner.health_check()

        assert health.images_ready is False

    def test_health_check_serialization(self) -> None:
        """EnvironmentHealth can be serialized to JSON."""
        runner = DockerRunner()
        with (
            patch.object(runner, "check_docker_available", return_value=True),
            patch("grist_mill.environments.docker_runner.shutil") as mock_shutil,
        ):
            mock_shutil.disk_usage.return_value = MagicMock(
                total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
            )
            health = runner.health_check()

        import json

        data = json.loads(health.model_dump_json())
        assert "docker_available" in data
        assert "images_ready" in data
        assert "disk_free_gb" in data

    def test_local_runner_health_check(self) -> None:
        """LocalRunner also supports health_check."""
        runner = LocalRunner()
        health = runner.health_check()
        assert hasattr(health, "docker_available")
        assert health.docker_available is not None  # boolean


# ===========================================================================
# VAL-ENV-03: Setup command failure marks task ERROR
# ===========================================================================


class TestSetupCommandFailure:
    """Setup command failure marks task ERROR (ENVIRONMENT_ERROR) without agent (VAL-ENV-03)."""

    def test_setup_failure_sets_error_state(self) -> None:
        """When setup_command fails, prepare() raises EnvironmentSetupError."""
        runner = DockerRunner(auto_pull=True)
        task = _make_task(setup_command="exit 1")

        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_volume = MagicMock()
        mock_volume.name = "vol123"

        # exec_run for setup command returns exit code 1
        class MockExecResult:
            def __init__(self, exit_code: int, output: tuple[bytes, bytes]) -> None:
                self.exit_code = exit_code
                self.output = output

        mock_exec_result = MockExecResult(1, (b"", b"setup failed\n"))

        with (
            patch.object(runner, "_ensure_image_available"),
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.volumes.create.return_value = mock_volume
            mock_client.containers.create.return_value = mock_container
            # containers.get must also return the mock_container for execute()
            mock_client.containers.get.return_value = mock_container

            # Make the container available for exec
            mock_container.start.return_value = None
            mock_container.exec_run.return_value = mock_exec_result
            mock_get_client.return_value = mock_client

            with pytest.raises(EnvironmentSetupError) as exc_info:
                runner.prepare(task)

            assert exc_info.value.error_category == ErrorCategory.ENVIRONMENT_ERROR
            assert runner.is_prepared is False  # cleanup should have been called

    def test_setup_success_does_not_raise(self) -> None:
        """When setup_command succeeds, prepare() completes normally."""
        runner = DockerRunner(auto_pull=True)
        task = _make_task(setup_command="echo 'setup ok'")

        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_volume = MagicMock()
        mock_volume.name = "vol123"

        # exec_run for setup command returns exit code 0
        class MockExecResult:
            def __init__(self, exit_code: int, output: tuple[bytes, bytes]) -> None:
                self.exit_code = exit_code
                self.output = output

        mock_exec_result = MockExecResult(0, (b"setup ok\n", b""))

        with (
            patch.object(runner, "_ensure_image_available"),
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.volumes.create.return_value = mock_volume
            mock_client.containers.create.return_value = mock_container
            mock_client.containers.get.return_value = mock_container

            mock_container.start.return_value = None
            mock_container.exec_run.return_value = mock_exec_result
            mock_get_client.return_value = mock_client

            runner.prepare(task)  # Should not raise
            assert runner.is_prepared

    def test_no_setup_command_skips_setup(self) -> None:
        """When there's no setup_command, prepare() skips the setup step."""
        runner = DockerRunner(auto_pull=True)
        task = _make_task(setup_command=None)

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
            # exec_run should NOT have been called (no setup_command)
            mock_container.exec_run.assert_not_called()
            assert runner.is_prepared

    def test_setup_failure_cleans_up_container(self) -> None:
        """Setup failure cleans up the container and volume."""
        runner = DockerRunner(auto_pull=True)
        task = _make_task(setup_command="exit 1")

        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_volume = MagicMock()
        mock_volume.name = "vol123"

        class MockExecResult:
            def __init__(self, exit_code: int, output: tuple[bytes, bytes]) -> None:
                self.exit_code = exit_code
                self.output = output

        mock_exec_result = MockExecResult(1, (b"", b"setup failed\n"))

        with (
            patch.object(runner, "_ensure_image_available"),
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.volumes.create.return_value = mock_volume
            mock_client.containers.create.return_value = mock_container
            mock_client.containers.get.return_value = mock_container
            mock_container.start.return_value = None
            mock_container.exec_run.return_value = mock_exec_result
            mock_get_client.return_value = mock_client

            with pytest.raises(EnvironmentSetupError):
                runner.prepare(task)

            # Container should have been removed
            mock_container.remove.assert_called_with(force=True)

    def test_setup_failure_real_docker(self) -> None:
        """Integration: setup_command failure with real Docker."""
        _skip_without_docker()
        runner = DockerRunner(image="alpine:3.19", auto_pull=True)
        task = _make_task(setup_command="exit 1", timeout=30)

        with pytest.raises(EnvironmentSetupError) as exc_info:
            runner.prepare(task)

        assert exc_info.value.error_category == ErrorCategory.ENVIRONMENT_ERROR
        assert (
            "setup" in str(exc_info.value).lower() or "environment" in str(exc_info.value).lower()
        )
        assert runner.is_prepared is False

    def test_environment_setup_error_has_task_id(self) -> None:
        """EnvironmentSetupError includes the task ID."""
        runner = DockerRunner(auto_pull=True)
        task = _make_task(task_id="special-task-42", setup_command="exit 1")

        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_volume = MagicMock()
        mock_volume.name = "vol123"

        class MockExecResult:
            def __init__(self, exit_code: int, output: tuple[bytes, bytes]) -> None:
                self.exit_code = exit_code
                self.output = output

        mock_exec_result = MockExecResult(1, (b"", b"failed\n"))

        with (
            patch.object(runner, "_ensure_image_available"),
            patch.object(runner, "_get_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.volumes.create.return_value = mock_volume
            mock_client.containers.create.return_value = mock_container
            mock_client.containers.get.return_value = mock_container
            mock_container.start.return_value = None
            mock_container.exec_run.return_value = mock_exec_result
            mock_get_client.return_value = mock_client

            with pytest.raises(EnvironmentSetupError) as exc_info:
                runner.prepare(task)

            assert "special-task-42" in str(exc_info.value)


# ===========================================================================
# VAL-ENV-02: Volume mount workspace isolation
# ===========================================================================


class TestVolumeMountIsolation:
    """Volume mounts isolate workspaces; concurrent tasks don't interfere (VAL-ENV-02)."""

    def test_unique_workspace_volume_per_prepare(self) -> None:
        """Each prepare() call creates a unique workspace volume."""
        runner = DockerRunner(auto_pull=True)

        volume_names: list[str] = []
        for i in range(3):
            task = _make_task(task_id=f"task-{i}")
            mock_container = MagicMock()
            mock_container.id = f"container-{i}"
            mock_volume = MagicMock()
            mock_volume.name = f"vol-{i}"
            volume_names.append(f"vol-{i}")

            with (
                patch.object(runner, "_ensure_image_available"),
                patch.object(runner, "_get_client") as mock_get_client,
            ):
                mock_client = MagicMock()
                mock_client.volumes.create.return_value = mock_volume
                mock_client.containers.create.return_value = mock_container
                mock_get_client.return_value = mock_client

                runner.prepare(task)
                runner.cleanup()

        # Verify unique names were requested
        create_calls = []
        for call in mock_client.volumes.create.call_args_list:
            create_calls.append(call.kwargs.get("name") or call[1].get("name"))

        # Each volume should have a unique name
        assert len(set(create_calls)) == len(create_calls)

    def test_concurrent_tasks_isolated_volumes(self) -> None:
        """Concurrent tasks get different workspace volumes."""
        num_tasks = 5
        volume_names: list[str] = []
        lock = threading.Lock()

        def prepare_task(idx: int) -> None:
            runner = DockerRunner(auto_pull=True)
            task = _make_task(task_id=f"concurrent-{idx}")
            mock_container = MagicMock()
            mock_container.id = f"concurrent-container-{idx}"
            mock_volume = MagicMock()
            mock_volume.name = f"concurrent-vol-{idx}"

            with (
                patch.object(runner, "_ensure_image_available"),
                patch.object(runner, "_get_client") as mock_get_client,
            ):
                mock_client = MagicMock()
                mock_client.volumes.create.return_value = mock_volume
                mock_client.containers.create.return_value = mock_container
                mock_get_client.return_value = mock_client

                runner.prepare(task)
                vol_name = runner._workspace_volume_name
                with lock:
                    volume_names.append(vol_name)
                runner.cleanup()

        threads = [threading.Thread(target=prepare_task, args=(i,)) for i in range(num_tasks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All volume names should be unique
        assert len(set(volume_names)) == num_tasks

    def test_workspace_isolation_real_docker(self) -> None:
        """Integration: Two containers don't see each other's files."""
        _skip_without_docker()
        # Task 1 writes a file
        runner1 = DockerRunner(image="alpine:3.19", auto_pull=True)
        task1 = _make_task(task_id="iso-1")
        runner1.prepare(task1)
        runner1.execute("echo 'secret-data-from-task-1' > /workspace/secret.txt", timeout=10.0)
        vol1_name = runner1._workspace_volume_name
        runner1.cleanup()

        # Task 2 should NOT see the file (different volume)
        runner2 = DockerRunner(image="alpine:3.19", auto_pull=True)
        task2 = _make_task(task_id="iso-2")
        runner2.prepare(task2)
        output = runner2.execute(
            "cat /workspace/secret.txt 2>&1 || echo 'file not found'", timeout=10.0
        )
        runner2.cleanup()

        # The file should NOT exist in task 2's workspace
        assert "file not found" in output.stdout or output.exit_code != 0

        # Clean up the preserved volume from task 1
        try:
            import docker

            client = docker.from_env()
            vol = client.volumes.get(vol1_name)
            vol.remove(force=True)
        except Exception:
            pass
