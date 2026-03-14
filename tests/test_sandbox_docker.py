"""Unit tests for Docker sandbox policy enforcement."""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bench.experiments import ExperimentConfig
from bench.experiments.runner import ExperimentRunner
from bench.harness import ErrorCategory, HarnessConfig, HarnessRequest
from bench.harness.adapters.pi_docker import PiDockerHarness
from bench.sandbox import DockerCommandBuilder, SandboxPolicy, SandboxProfile
from evaluation.pi_runner import DockerPiRunner, DockerPiRunnerConfig


def test_strict_policy_builds_hardened_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict profile should include baseline hardening flags."""
    monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
    monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

    policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
    builder = DockerCommandBuilder(policy)

    cmd = builder.build_run_command(
        image="test-image:latest",
        command=["echo", "ok"],
        volumes=[
            ("/tmp/work", "/workspace", "rw"),
            ("/tmp/readonly", "/data", "ro"),
        ],
    )

    assert "--user" in cmd
    assert "1000:1000" in cmd
    assert "--cap-drop" in cmd
    assert "ALL" in cmd
    assert "--read-only" in cmd
    assert "--security-opt" in cmd
    assert "no-new-privileges:true" in cmd
    assert "--network=none" in cmd
    assert "--memory-swap" in cmd
    assert "4g" in cmd
    assert "--ulimit" in cmd
    assert "nofile=1024:4096" in cmd
    assert "/tmp/work:/workspace:rw" in cmd
    assert "/tmp/readonly:/data:ro" in cmd


def test_networked_profile_uses_bridge_network() -> None:
    policy = SandboxPolicy.from_profile(SandboxProfile.NETWORKED)
    builder = DockerCommandBuilder(policy)

    cmd = builder.build_run_command(
        image="test-image:latest",
        command=[],
    )

    assert "--network=bridge" in cmd
    assert "--network=none" not in cmd


def test_blocks_docker_socket_mount_by_default() -> None:
    policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
    builder = DockerCommandBuilder(policy)

    with pytest.raises(ValueError, match="Docker socket mounts are not allowed"):
        builder.build_run_command(
            image="test-image:latest",
            command=[],
            volumes=[("/var/run/docker.sock", "/var/run/docker.sock", "ro")],
        )


def test_blocks_unapproved_rw_mount_destination() -> None:
    policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
    builder = DockerCommandBuilder(policy)

    with pytest.raises(ValueError, match="Writable bind mounts are restricted"):
        builder.build_run_command(
            image="test-image:latest",
            command=[],
            volumes=[("/tmp/work", "/other", "rw")],
        )


def test_developer_profile_allows_rw_mounts() -> None:
    policy = SandboxPolicy.from_profile(SandboxProfile.DEVELOPER)
    builder = DockerCommandBuilder(policy)

    cmd = builder.build_run_command(
        image="test-image:latest",
        command=[],
        volumes=[("/tmp/work", "/anywhere", "rw")],
    )

    assert "/tmp/work:/anywhere:rw" in cmd


def test_docker_runner_config_does_not_forward_github_token_by_default() -> None:
    config = DockerPiRunnerConfig()
    assert config.forward_github_token is False


def test_pi_docker_harness_forwards_token_flag_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, DockerPiRunnerConfig] = {}

    class DummyRunner:
        def __init__(self, config: DockerPiRunnerConfig):
            captured["config"] = config

    monkeypatch.setattr("bench.harness.adapters.pi_docker.DockerPiRunner", DummyRunner)

    harness = PiDockerHarness(
        HarnessConfig(
            model="openrouter/openai/gpt-oss-120b:free",
            forward_github_token=True,
        )
    )
    harness.setup()

    assert captured["config"].forward_github_token is True


def test_pi_docker_harness_forwards_auth_policy_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, DockerPiRunnerConfig] = {}

    class DummyRunner:
        def __init__(self, config: DockerPiRunnerConfig):
            captured["config"] = config

    monkeypatch.setattr("bench.harness.adapters.pi_docker.DockerPiRunner", DummyRunner)

    harness = PiDockerHarness(
        HarnessConfig(
            model="openrouter/openai/gpt-oss-120b:free",
            auth_policy="mounted_file",
        )
    )
    harness.setup()

    assert captured["config"].auth_policy == "mounted_file"


def test_experiment_runner_propagates_sandbox_and_token_flags() -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "sandbox-propagation",
            "execution": {
                "harness": "pi_docker",
                "sandbox_profile": "strict",
                "forward_github_token": True,
                "auth_policy": "mounted_file",
            },
        }
    )
    runner = ExperimentRunner(config)
    harness_config = runner._build_harness_config()

    assert harness_config.sandbox_profile == "strict"
    assert harness_config.forward_github_token is True
    assert harness_config.auth_policy == "mounted_file"


class TestWorkspaceLifecycle:
    """Tests for ephemeral workspace lifecycle."""

    def test_parallel_runs_use_unique_workspaces(self, tmp_path: Path) -> None:
        """Verify parallel runs for same repo/model use unique workspace paths."""
        captured_paths: list[str] = []

        def mock_run_evaluation(
            skill_content: str | None,
            task_instruction: str,
            task_context: str,
            package_dir: Path,
            model: str | None = None,
            task: Any = None,
        ) -> dict[str, Any]:
            # Capture the workspace path that would be created
            # Since we can't easily intercept tempfile.mkdtemp, we verify uniqueness
            # by checking the prefix pattern in a real run
            import uuid

            safe_repo = "test_repo".replace("/", "_")
            safe_model = (model or "test_model").replace("/", "_").replace(":", "_")
            workspace_path = tempfile.mkdtemp(
                prefix=f"trainr_{safe_repo}_{safe_model}_{uuid.uuid4().hex[:8]}_",
                dir=tempfile.gettempdir(),
            )
            captured_paths.append(workspace_path)
            # Clean up immediately for this test
            import shutil

            shutil.rmtree(workspace_path, ignore_errors=True)
            return {"success": True, "score": 1.0, "output": "", "error": ""}

        # Run multiple "evaluations" in parallel
        models = ["model_a", "model_b", "model_a", "model_c", "model_a"]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(mock_run_evaluation, None, "test", "context", tmp_path, model, None)
                for model in models
            ]
            for future in futures:
                future.result()

        # All paths should be unique
        assert len(captured_paths) == len(set(captured_paths)), (
            f"Duplicate workspace paths found: {captured_paths}"
        )

    def test_success_path_removes_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify workspace is cleaned up on success."""
        import json

        config = DockerPiRunnerConfig(
            model="test-model",
            timeout=60,
            keep_workspace_on_failure=False,
        )

        created_workspace: list[Path] = []

        def mock_subprocess_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            # Handle _check_docker call (docker images -q)
            if "images" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc123"  # Fake image ID
                return result
            # Handle run_evaluation call
            # Capture the workspace path from the docker command
            for _i, arg in enumerate(cmd):
                if str(arg).startswith("/tmp") and ":/workspace" in str(arg):
                    workspace = Path(str(arg).split(":")[0])
                    created_workspace.append(workspace)
                    break
            # Return success
            result = MagicMock()
            result.returncode = 0
            usage_event = {"type": "message_end", "usage": {"input": 10, "output": 5}}
            result.stdout = f"{json.dumps(usage_event)}\n1 passed, 0 failed"
            result.stderr = ""
            return result

        with patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run):
            runner = DockerPiRunner(config)

            # Create a mock task
            task = MagicMock()
            task.source = {"repo": "test/repo", "base_commit": "abc"}
            task.test_patch = ""
            task.patch = ""
            task.tests = {"fail_to_pass": [], "pass_to_pass": []}

            result = runner.run_evaluation(
                skill_content="test skill",
                task_instruction="test instruction",
                task_context="test context",
                package_dir=Path("/tmp/pkg"),
                model="test-model",
                task=task,
            )

        # On success, workspace should be cleaned up
        assert result["success"] is True
        if created_workspace:
            assert not created_workspace[0].exists(), (
                f"Workspace should be removed on success: {created_workspace[0]}"
            )

    def test_timeout_path_removes_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify workspace is cleaned up on timeout."""
        import subprocess

        config = DockerPiRunnerConfig(
            model="test-model",
            timeout=1,
            keep_workspace_on_failure=False,
        )

        run_call_count = [0]

        def mock_subprocess_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            # Handle _check_docker call (docker images -q)
            if "images" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc123"
                return result
            # Second call is the actual run - raise timeout
            run_call_count[0] += 1
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

        with patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run):
            runner = DockerPiRunner(config)

            task = MagicMock()
            task.source = {"repo": "test/repo", "base_commit": "abc"}
            task.test_patch = ""
            task.patch = ""
            task.tests = {"fail_to_pass": [], "pass_to_pass": []}

            result = runner.run_evaluation(
                skill_content="test skill",
                task_instruction="test instruction",
                task_context="test context",
                package_dir=Path("/tmp/pkg"),
                model="test-model",
                task=task,
            )

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_exception_path_removes_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify workspace is cleaned up on exception."""
        config = DockerPiRunnerConfig(
            model="test-model",
            timeout=60,
            keep_workspace_on_failure=False,
        )

        run_call_count = [0]

        def mock_subprocess_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            # Handle _check_docker call (docker images -q)
            if "images" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc123"
                return result
            # Second call is the actual run - raise exception
            run_call_count[0] += 1
            raise RuntimeError("Simulated failure")

        with patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run):
            runner = DockerPiRunner(config)

            task = MagicMock()
            task.source = {"repo": "test/repo", "base_commit": "abc"}
            task.test_patch = ""
            task.patch = ""
            task.tests = {"fail_to_pass": [], "pass_to_pass": []}

            result = runner.run_evaluation(
                skill_content="test skill",
                task_instruction="test instruction",
                task_context="test context",
                package_dir=Path("/tmp/pkg"),
                model="test-model",
                task=task,
            )

        assert result["success"] is False
        assert "Simulated failure" in result["error"]

    def test_retention_on_failure_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify workspace is retained when keep_workspace_on_failure=True and run fails."""
        config = DockerPiRunnerConfig(
            model="test-model",
            timeout=60,
            keep_workspace_on_failure=True,
        )

        created_workspace: list[Path] = []

        def mock_subprocess_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            # Handle _check_docker call (docker images -q)
            if "images" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc123"
                return result
            # Handle run_evaluation call
            # Capture the workspace path from the docker command
            for _i, arg in enumerate(cmd):
                if str(arg).startswith("/tmp") and ":/workspace" in str(arg):
                    workspace = Path(str(arg).split(":")[0])
                    created_workspace.append(workspace)
                    # Create the directory so we can verify it's retained
                    workspace.mkdir(parents=True, exist_ok=True)
                    break
            # Return failure
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "Error occurred"
            return result

        with (
            caplog.at_level(logging.WARNING),
            patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run),
        ):
            runner = DockerPiRunner(config)

            task = MagicMock()
            task.source = {"repo": "test/repo", "base_commit": "abc"}
            task.test_patch = ""
            task.patch = ""
            task.tests = {"fail_to_pass": [], "pass_to_pass": []}

            result = runner.run_evaluation(
                skill_content="test skill",
                task_instruction="test instruction",
                task_context="test context",
                package_dir=Path("/tmp/pkg"),
                model="test-model",
                task=task,
            )

        # On failure with keep_workspace_on_failure=True, workspace should be retained
        assert result["success"] is False
        if created_workspace:
            # Workspace should still exist
            assert created_workspace[0].exists(), (
                f"Workspace should be retained on failure: {created_workspace[0]}"
            )
            # Clean up after test
            import shutil

            shutil.rmtree(created_workspace[0], ignore_errors=True)

        # Check that retention was logged
        assert any("Retaining failed workspace" in record.message for record in caplog.records), (
            "Expected warning log for retained workspace"
        )

    def test_failure_without_retention_flag_removes_workspace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify workspace is cleaned up on failure when keep_workspace_on_failure=False."""
        config = DockerPiRunnerConfig(
            model="test-model",
            timeout=60,
            keep_workspace_on_failure=False,  # Default, but explicit here
        )

        created_workspace: list[Path] = []

        def mock_subprocess_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            # Handle _check_docker call (docker images -q)
            if "images" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc123"
                return result
            # Handle run_evaluation call
            # Capture the workspace path from the docker command
            for _i, arg in enumerate(cmd):
                if str(arg).startswith("/tmp") and ":/workspace" in str(arg):
                    workspace = Path(str(arg).split(":")[0])
                    created_workspace.append(workspace)
                    break
            # Return failure
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "Error occurred"
            return result

        with patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run):
            runner = DockerPiRunner(config)

            task = MagicMock()
            task.source = {"repo": "test/repo", "base_commit": "abc"}
            task.test_patch = ""
            task.patch = ""
            task.tests = {"fail_to_pass": [], "pass_to_pass": []}

            result = runner.run_evaluation(
                skill_content="test skill",
                task_instruction="test instruction",
                task_context="test context",
                package_dir=Path("/tmp/pkg"),
                model="test-model",
                task=task,
            )

        # On failure without retention flag, workspace should be cleaned up
        assert result["success"] is False
        if created_workspace:
            assert not created_workspace[0].exists(), (
                f"Workspace should be removed on failure without retention flag: {created_workspace[0]}"
            )


def test_run_evaluation_detects_agent_auth_error_from_json_events() -> None:
    """Pi JSON auth errors should fail the run even if process exit code is 0."""
    import json

    config = DockerPiRunnerConfig(model="glm-5-plan", timeout=60)

    oidc_error = "401 authentication_error: Error verifying OIDC token"

    def mock_subprocess_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("cmd", [])
        if "images" in cmd:
            result = MagicMock()
            result.returncode = 0
            result.stdout = "abc123"
            return result

        result = MagicMock()
        result.returncode = 0
        error_event = {
            "type": "message_end",
            "message": {
                "stopReason": "error",
                "errorMessage": oidc_error,
            },
        }
        result.stdout = f"{json.dumps(error_event)}\n1 passed, 0 failed"
        result.stderr = ""
        return result

    with patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run):
        runner = DockerPiRunner(config)
        task = MagicMock()
        task.source = {"repo": "test/repo", "base_commit": "abc"}
        task.test_patch = ""
        task.patch = ""
        task.tests = {"fail_to_pass": [], "pass_to_pass": []}

        result = runner.run_evaluation(
            skill_content="test skill",
            task_instruction="test instruction",
            task_context="test context",
            package_dir=Path("/tmp/pkg"),
            model="glm-5-plan",
            task=task,
        )

    assert result["success"] is False
    assert "agent error" in result["error"].lower()
    assert "oidc" in result["error"].lower()


def test_parse_agent_error_ignores_bare_status_codes_without_error_context() -> None:
    """Plain numbers like file sizes should not be misread as HTTP errors."""
    runner = object.__new__(DockerPiRunner)

    assert runner._parse_agent_error("Wrote 403 bytes to tests/testthat/test-generated.R") is None


def test_parse_agent_error_detects_contextual_http_status_codes() -> None:
    """HTTP status errors with surrounding context should still be detected."""
    runner = object.__new__(DockerPiRunner)

    error_text = runner._parse_agent_error("HTTP 403 Forbidden while contacting provider")

    assert error_text is not None
    assert "403" in error_text


def test_parse_agent_error_detects_model_not_found() -> None:
    """Model lookup failures should be surfaced as agent errors."""
    runner = object.__new__(DockerPiRunner)

    error_text = runner._parse_agent_error("Error: Model openrouter/foo not found")

    assert error_text is not None
    assert "model" in error_text.lower()
    assert "not found" in error_text.lower()


def test_run_evaluation_marks_zero_turn_zero_token_runs_as_agent_failures() -> None:
    """Runs with no usage events should not be scored from test output alone."""
    config = DockerPiRunnerConfig(model="glm-5-plan", timeout=60)

    def mock_subprocess_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("cmd", [])
        if "images" in cmd:
            result = MagicMock()
            result.returncode = 0
            result.stdout = "abc123"
            return result

        result = MagicMock()
        result.returncode = 0
        result.stdout = "1 passed, 0 failed"
        result.stderr = ""
        return result

    with patch("evaluation.pi_runner.subprocess.run", mock_subprocess_run):
        runner = DockerPiRunner(config)
        task = MagicMock()
        task.source = {"repo": "test/repo", "base_commit": "abc"}
        task.test_patch = ""
        task.patch = ""
        task.tests = {"fail_to_pass": [], "pass_to_pass": []}

        result = runner.run_evaluation(
            skill_content="test skill",
            task_instruction="test instruction",
            task_context="test context",
            package_dir=Path("/tmp/pkg"),
            model="glm-5-plan",
            task=task,
        )

    assert result["success"] is False
    assert result["agent_error"] == "Agent produced no output (0 tokens generated)"


def test_pi_docker_harness_classifies_auth_errors_as_auth_error() -> None:
    """Auth failures should be classified as AUTH_ERROR."""
    harness = PiDockerHarness(HarnessConfig())
    request = HarnessRequest(task_id="task-1", prompt="test prompt")
    result = harness._convert_result(
        request=request,
        result_dict={
            "success": False,
            "error": "Agent error: 401 authentication_error: Error verifying OIDC token",
            "test_results": {"passed": False, "num_passed": 0, "num_failed": 0},
            "token_usage": {},
            "model": "zai/glm-5",
        },
        execution_time=1.0,
        run_id="run-123",
    )

    assert result.error_category == ErrorCategory.AUTH_ERROR


def test_pi_docker_harness_classifies_model_not_found_as_model_error() -> None:
    """Model lookup failures should be classified as MODEL_ERROR."""
    harness = PiDockerHarness(HarnessConfig())
    request = HarnessRequest(task_id="task-1", prompt="test prompt")
    result = harness._convert_result(
        request=request,
        result_dict={
            "success": False,
            "error": "Agent error: Model openrouter/foo not found. Try --list-models.",
            "test_results": {"passed": False, "num_passed": 0, "num_failed": 0},
            "token_usage": {},
            "model": "zai/glm-5",
        },
        execution_time=1.0,
        run_id="run-123",
    )

    assert result.error_category == ErrorCategory.MODEL_ERROR
