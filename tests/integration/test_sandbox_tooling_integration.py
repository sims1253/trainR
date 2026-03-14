"""Integration tests for sandbox and tooling behavior contracts.

These tests validate that sandbox policies are correctly enforced at the
behavior level, not just at the unit level. They ensure:

1. Strict profile includes all required security flags
2. Networked profile uses proper network configuration
3. Mount policies are enforced correctly
4. Config settings flow through to actual docker commands
5. The canonical path can initialize and preflight without bypasses

Test categories:
- Contract tests (no Docker required): Fast, pure unit-level behavior validation
- Integration tests (@pytest.mark.integration_local): Require Docker availability
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from bench.experiments.config import ExecutionConfig, ExperimentConfig
from bench.experiments.runner import ExperimentRunner
from bench.harness import HarnessConfig, HarnessRegistry
from bench.harness.adapters.pi_docker import PiDockerHarness
from bench.sandbox import (
    DockerCommandBuilder,
    SandboxPolicy,
    SandboxProfile,
    build_docker_command,
    get_sandbox_policy,
)

if TYPE_CHECKING:
    from evaluation.pi_runner import DockerPiRunnerConfig

# Check Docker availability for integration tests
DOCKER_AVAILABLE = False
try:
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    DOCKER_AVAILABLE = result.returncode == 0
except (subprocess.SubprocessError, FileNotFoundError, OSError):
    pass


def skip_if_no_docker() -> None:
    """Skip test if Docker is not available."""
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker not available")


@pytest.fixture
def docker_available() -> bool:
    """Fixture that skips test if Docker is not available."""
    skip_if_no_docker()
    return True


# =============================================================================
# Strict Profile Contract Tests
# =============================================================================


class TestStrictProfileContract:
    """Tests for strict sandbox profile behavior contracts.

    These are pure contract tests that do not require Docker.
    They validate that the strict profile includes all required security flags.
    """

    def test_strict_profile_includes_readonly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict profile must include --read-only flag."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        cmd = build_docker_command(
            image="test:latest",
            command=["echo", "test"],
            profile=policy,
        )
        assert "--read-only" in cmd, "Strict profile must include --read-only"

    def test_strict_profile_drops_capabilities(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict profile must drop all capabilities."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        cmd = build_docker_command(
            image="test:latest",
            command=["echo", "test"],
            profile=policy,
        )
        assert "--cap-drop" in cmd, "Strict profile must include --cap-drop"
        assert "ALL" in cmd, "Strict profile must drop ALL capabilities"

    def test_strict_profile_disables_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict profile must disable network."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        cmd = build_docker_command(
            image="test:latest",
            command=["echo", "test"],
            profile=policy,
        )
        assert "--network=none" in cmd, "Strict profile must include --network=none"

    def test_strict_profile_no_new_privileges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict profile must set no-new-privileges."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        cmd = build_docker_command(
            image="test:latest",
            command=["echo", "test"],
            profile=policy,
        )
        assert "--security-opt" in cmd, "Strict profile must include --security-opt"
        assert "no-new-privileges" in " ".join(cmd), "Strict profile must set no-new-privileges"

    def test_strict_profile_runs_as_non_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict profile must run as non-root user."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        cmd = build_docker_command(
            image="test:latest",
            command=["echo", "test"],
            profile=policy,
        )
        assert "--user" in cmd, "Strict profile must include --user flag"
        # Should have a non-root user (not 0:0)
        user_idx = cmd.index("--user")
        user_value = cmd[user_idx + 1]
        assert not user_value.startswith("0:"), "Strict profile must not run as root"

    def test_strict_profile_default_from_get_sandbox_policy(self) -> None:
        """get_sandbox_policy() must return strict profile by default."""
        policy = get_sandbox_policy()
        assert policy.profile == SandboxProfile.STRICT
        assert policy.read_only_root_fs is True
        assert policy.network_enabled is False
        assert policy.no_new_privileges is True
        assert policy.drop_capabilities == ["ALL"]


# =============================================================================
# Networked Profile Contract Tests
# =============================================================================


class TestNetworkedProfileContract:
    """Tests for networked sandbox profile behavior contracts.

    These are pure contract tests that do not require Docker.
    They validate that the networked profile uses proper network configuration.
    """

    def test_networked_profile_uses_bridge_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Networked profile must use bridge/custom network, not none."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.NETWORKED)
        cmd = build_docker_command(
            image="test:latest",
            command=["echo", "test"],
            profile=policy,
        )
        assert "--network=none" not in cmd, "Networked profile must NOT use --network=none"
        # Should have some network config (bridge or custom)
        assert "--network=bridge" in cmd, "Networked profile should use bridge network"

    def test_networked_profile_enables_network_flag(self) -> None:
        """Networked profile must have network_enabled=True."""
        policy = SandboxPolicy.from_profile(SandboxProfile.NETWORKED)
        assert policy.network_enabled is True, "Networked profile must have network_enabled=True"

    def test_networked_profile_keeps_other_security_settings(self) -> None:
        """Networked profile should keep other security hardening."""
        policy = SandboxPolicy.from_profile(SandboxProfile.NETWORKED)
        # Should still have most security settings
        assert policy.read_only_root_fs is True, "Networked should still have read-only FS"
        assert policy.no_new_privileges is True, "Networked should still have no-new-privileges"
        assert policy.drop_capabilities == ["ALL"], "Networked should still drop capabilities"


# =============================================================================
# Mount Policy Enforcement Tests
# =============================================================================


class TestMountPolicyEnforcement:
    """Tests for mount policy enforcement.

    These are pure contract tests that do not require Docker.
    They validate that mount policies are correctly enforced.
    """

    def test_docker_socket_mount_blocked_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Docker socket mount must be blocked by default."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        with pytest.raises(ValueError, match="Docker socket mounts are not allowed"):
            builder.build_run_command(
                image="test:latest",
                command=["echo", "test"],
                volumes=[("/var/run/docker.sock", "/var/run/docker.sock", "ro")],
            )

    def test_docker_socket_mount_blocked_alternate_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Docker socket mount must be blocked even with alternate path."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        with pytest.raises(ValueError, match="Docker socket mounts are not allowed"):
            builder.build_run_command(
                image="test:latest",
                command=["echo", "test"],
                volumes=[("/run/docker.sock", "/run/docker.sock", "rw")],
            )

    def test_strict_profile_blocks_unauthorized_rw_mounts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict profile must block unauthorized RW mount destinations."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        # /etc is not in writable_bind_mount_destinations by default
        with pytest.raises(ValueError, match="Writable bind mounts are restricted"):
            builder.build_run_command(
                image="test:latest",
                command=["echo", "test"],
                volumes=[("/host/etc", "/etc", "rw")],
            )

    def test_strict_profile_allows_workspace_rw_mount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict profile must allow /workspace as RW mount destination."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        # /workspace is in writable_bind_mount_destinations by default
        cmd = builder.build_run_command(
            image="test:latest",
            command=["echo", "test"],
            volumes=[("/host/work", "/workspace", "rw")],
        )
        assert "/host/work:/workspace:rw" in cmd

    def test_strict_profile_allows_ro_mounts_anywhere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict profile must allow RO mounts to any destination."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        # RO mounts should be allowed to any destination
        cmd = builder.build_run_command(
            image="test:latest",
            command=["echo", "test"],
            volumes=[("/host/data", "/data", "ro"), ("/host/etc", "/etc", "ro")],
        )
        assert "/host/data:/data:ro" in cmd
        assert "/host/etc:/etc:ro" in cmd


# =============================================================================
# Harness + Policy Wiring Tests
# =============================================================================


class TestHarnessPolicyWiring:
    """Tests for harness + policy wiring integration.

    These tests validate that execution.sandbox_profile in config
    reaches the actual docker command builder path.
    """

    def test_sandbox_profile_reaches_docker_builder_via_experiment_runner(self) -> None:
        """execution.sandbox_profile in config must reach harness config."""
        config = ExperimentConfig(
            name="test-sandbox-wiring",
            execution=ExecutionConfig(
                harness="pi_docker",
                sandbox_profile=SandboxProfile.STRICT,
            ),
        )
        runner = ExperimentRunner(config)
        harness_config = runner._build_harness_config()

        assert harness_config.sandbox_profile == "strict"

    def test_networked_profile_reaches_docker_builder_via_experiment_runner(self) -> None:
        """Networked profile must flow through to harness config."""
        config = ExperimentConfig(
            name="test-networked-wiring",
            execution=ExecutionConfig(
                harness="pi_docker",
                sandbox_profile=SandboxProfile.NETWORKED,
            ),
        )
        runner = ExperimentRunner(config)
        harness_config = runner._build_harness_config()

        assert harness_config.sandbox_profile == "networked"

    def test_pi_docker_harness_forwards_sandbox_profile_to_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PiDockerHarness must forward sandbox_profile to DockerPiRunner."""
        captured: dict[str, DockerPiRunnerConfig] = {}

        class DummyRunner:
            def __init__(self, config: DockerPiRunnerConfig):
                captured["config"] = config

        monkeypatch.setattr("bench.harness.adapters.pi_docker.DockerPiRunner", DummyRunner)

        harness = PiDockerHarness(
            HarnessConfig(
                model="test-model",
                sandbox_profile="strict",
            )
        )
        harness.setup()

        assert captured["config"].sandbox_profile == "strict"

    def test_developer_profile_flows_through_harness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Developer profile must flow through harness to runner."""
        captured: dict[str, DockerPiRunnerConfig] = {}

        class DummyRunner:
            def __init__(self, config: DockerPiRunnerConfig):
                captured["config"] = config

        monkeypatch.setattr("bench.harness.adapters.pi_docker.DockerPiRunner", DummyRunner)

        harness = PiDockerHarness(
            HarnessConfig(
                model="test-model",
                sandbox_profile="developer",
            )
        )
        harness.setup()

        assert captured["config"].sandbox_profile == "developer"

    def test_harness_registry_has_pi_docker(self) -> None:
        """HarnessRegistry must have pi_docker registered."""
        assert HarnessRegistry.is_registered("pi_docker")
        harness_cls = HarnessRegistry.get_class("pi_docker")
        assert harness_cls is PiDockerHarness


# =============================================================================
# End-to-End Dry-run Integration Tests
# =============================================================================


class TestEndToEndDryRunIntegration:
    """End-to-end integration tests that require Docker.

    These tests confirm the canonical path can initialize harness
    and preflight without bypasses.
    """

    @pytest.mark.integration_local
    def test_canonical_path_harness_initialization(self, docker_available: bool) -> None:
        """Confirm canonical path can initialize harness from registry."""
        config = HarnessConfig(
            model="test-model",
            sandbox_profile="strict",
            timeout=60,
        )

        # Should be able to get harness from registry
        harness = HarnessRegistry.get("pi_docker", config)
        assert harness is not None
        assert isinstance(harness, PiDockerHarness)

    @pytest.mark.integration_local
    def test_canonical_path_harness_validate_environment(self, docker_available: bool) -> None:
        """Confirm harness can validate environment (Docker check)."""
        config = HarnessConfig(
            model="test-model",
            sandbox_profile="strict",
            timeout=60,
        )

        harness = HarnessRegistry.get("pi_docker", config)
        is_valid, issues = harness.validate_environment()

        # Docker should be available (we passed docker_available fixture)
        # Image might not exist, but Docker itself should work
        if not is_valid:
            # Should only fail due to missing image, not Docker itself
            assert any("image" in issue.lower() for issue in issues), f"Unexpected issues: {issues}"

    @pytest.mark.integration_local
    def test_canonical_path_experiment_runner_initialization(self, docker_available: bool) -> None:
        """Confirm ExperimentConfig can be created and used with pi_docker."""
        config = ExperimentConfig(
            name="test-canonical-path",
            execution=ExecutionConfig(
                harness="pi_docker",
                sandbox_profile=SandboxProfile.STRICT,
                timeout=60,
            ),
        )

        # Should be able to create runner
        runner = ExperimentRunner(config)
        assert runner is not None

        # Should be able to build harness config
        harness_config = runner._build_harness_config()
        assert harness_config.sandbox_profile == "strict"

    @pytest.mark.integration_local
    def test_strict_policy_docker_command_builds_correctly(
        self, docker_available: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict profile command must build correctly in Docker context."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = get_sandbox_policy(SandboxProfile.STRICT)
        cmd = build_docker_command(
            image="posit-gskill-eval:latest",
            command=["pi", "eval", "--help"],
            profile=policy,
            volumes=[("/tmp/workspace", "/workspace", "rw")],
        )

        # Verify all strict profile flags are present
        cmd_str = " ".join(cmd)
        assert "--read-only" in cmd
        assert "--network=none" in cmd
        assert "--cap-drop" in cmd and "ALL" in cmd
        assert "no-new-privileges" in cmd_str
        assert "/tmp/workspace:/workspace:rw" in cmd

    @pytest.mark.integration_local
    def test_networked_policy_docker_command_builds_correctly(
        self, docker_available: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Networked profile command must build correctly in Docker context."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = get_sandbox_policy(SandboxProfile.NETWORKED)
        cmd = build_docker_command(
            image="posit-gskill-eval:latest",
            command=["pi", "eval", "--help"],
            profile=policy,
        )

        # Verify network is enabled
        assert "--network=bridge" in cmd
        assert "--network=none" not in cmd


# =============================================================================
# Regression Tests for Sandbox Policy
# =============================================================================


class TestSandboxPolicyRegression:
    """Regression tests to catch sandbox policy bypasses.

    These tests ensure that any changes to sandbox policy don't
    accidentally weaken security.
    """

    def test_strict_profile_defaults_are_secure(self) -> None:
        """Strict profile must have secure defaults."""
        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)

        # All security flags should be enabled
        assert policy.run_as_non_root is True
        assert policy.read_only_root_fs is True
        assert policy.no_new_privileges is True
        assert policy.network_enabled is False
        assert policy.allow_docker_socket_mount is False
        assert policy.read_only_bind_mounts is True

        # Capabilities should be dropped
        assert "ALL" in policy.drop_capabilities

    def test_developer_profile_is_explicitly_relaxed(self) -> None:
        """Developer profile must explicitly relax restrictions (not by accident)."""
        policy = SandboxPolicy.from_profile(SandboxProfile.DEVELOPER)

        # Developer profile intentionally relaxes security
        assert policy.run_as_non_root is False
        assert policy.read_only_root_fs is False
        assert policy.no_new_privileges is False
        assert policy.network_enabled is True
        assert policy.read_only_bind_mounts is False
        assert policy.drop_capabilities == []

    def test_policy_cannot_mount_docker_socket_unless_explicitly_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Docker socket mount must require explicit opt-in."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        # Default policy should block
        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        with pytest.raises(ValueError):
            builder.build_run_command(
                image="test:latest",
                command=["echo", "test"],
                volumes=[("/var/run/docker.sock", "/var/run/docker.sock", "ro")],
            )

        # Explicitly allowing should work
        policy.allow_docker_socket_mount = True
        builder2 = DockerCommandBuilder(policy)
        cmd = builder2.build_run_command(
            image="test:latest",
            command=["echo", "test"],
            volumes=[("/var/run/docker.sock", "/var/run/docker.sock", "ro")],
        )
        assert "/var/run/docker.sock:/var/run/docker.sock:ro" in cmd

    def test_invalid_volume_mode_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid volume modes must be rejected."""
        monkeypatch.setattr("bench.sandbox.docker.os.getuid", lambda: 1000)
        monkeypatch.setattr("bench.sandbox.docker.os.getgid", lambda: 1000)

        policy = SandboxPolicy.from_profile(SandboxProfile.STRICT)
        builder = DockerCommandBuilder(policy)

        with pytest.raises(ValueError, match="Invalid volume mode"):
            builder.build_run_command(
                image="test:latest",
                command=["echo", "test"],
                volumes=[("/tmp/work", "/workspace", "invalid")],
            )
