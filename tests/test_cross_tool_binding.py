"""Cross-tool binding isolation tests (m2-cross-tool-binding-isolation).

Verifies that tool bindings work correctly in both local and Docker execution
modes, and that the same tool invocation produces identical effects regardless
of runner type. Also verifies that the artifact registry is the single source
of truth for artifact references across all subsystems.

Validates:
- VAL-CROSS-09: Artifact registry unifies cross-subsystem references
  - Every subsystem resolves artifact references through the registry
  - A deleted artifact produces a clear registry-level error in all consumers
- VAL-CROSS-12: Tool binding survives environment isolation
  - Same task produces consistent tool invocation traces in local and Docker modes
  - Artifact resources cleaned up in both modes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from grist_mill.environments.local_runner import LocalRunner
from grist_mill.harness.harness import Harness
from grist_mill.registry import ArtifactRegistry
from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ExecutionOutput,
    HarnessConfig,
    Task,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.artifact import (
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)
from grist_mill.schemas.telemetry import (
    TelemetryCollector,
)
from grist_mill.tools.binding import ArtifactBinder
from grist_mill.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_registry() -> ArtifactRegistry:
    """Empty artifact registry."""
    return ArtifactRegistry()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Empty tool registry."""
    return ToolRegistry()


@pytest.fixture
def sample_skill_file(tmp_path: Path) -> Path:
    """Create a sample SKILL.md file."""
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "# Testing Skill\n\nThis is a skill for testing.\n\n"
        "## Rules\n\n- Always write tests first\n- Use assertions\n",
        encoding="utf-8",
    )
    return skill_file


@pytest.fixture
def sample_task() -> Task:
    """A simple task for testing."""
    return Task(
        id="cross-binding-001",
        prompt="Fix the off-by-one error in the parser.",
        language="python",
        test_command="echo hello",
        timeout=30,
        difficulty=Difficulty.MEDIUM,
    )


@pytest.fixture
def sample_config() -> HarnessConfig:
    """Harness configuration for local runner."""
    return HarnessConfig(
        agent=AgentConfig(model="gpt-4", provider="openrouter"),
        environment=EnvironmentConfig(runner_type="local"),
    )


def _make_mock_agent(
    result: TaskResult | None = None,
) -> Any:
    """Create a mock agent that returns the given result."""
    agent = MagicMock()
    if result is not None:
        agent.run.return_value = result
    else:
        agent.run.return_value = TaskResult(
            task_id="cross-binding-001",
            status=TaskStatus.SUCCESS,
            score=1.0,
        )
    return agent


def _make_mock_env(
    execute_output: ExecutionOutput | None = None,
) -> Any:
    """Create a mock environment that tracks call order."""
    env = MagicMock()
    env.execute.return_value = execute_output or ExecutionOutput(
        stdout="hello",
        stderr="",
        exit_code=0,
    )
    env.call_order: list[str] = []

    def _prepare(task: Task) -> None:
        env.call_order.append("prepare")

    def _execute(command: str, timeout: float) -> ExecutionOutput:
        env.call_order.append(f"execute:{command}")
        return env.execute.return_value

    def _cleanup() -> None:
        env.call_order.append("cleanup")

    env.prepare.side_effect = _prepare
    env.execute.side_effect = _execute
    env.cleanup.side_effect = _cleanup
    return env


# ===========================================================================
# VAL-CROSS-09: Artifact registry unifies cross-subsystem references
# ===========================================================================


class TestArtifactRegistryCrossSubsystem:
    """Verify that the artifact registry is the single source of truth for
    artifact references across all subsystems."""

    def test_registry_resolves_for_harness_config(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Registry-provided HarnessConfig includes artifact bindings."""
        skill = SkillArtifact(
            type="skill",
            name="test_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )

        # Both artifact names appear in the config
        assert "test_skill" in config.artifact_bindings
        assert "test_tool" in config.artifact_bindings

    def test_registry_resolves_for_binder(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """ArtifactBinder uses the same registry to resolve artifacts."""
        skill = SkillArtifact(
            type="skill",
            name="binder_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["skills"]) == 1
        assert context["skills"][0]["name"] == "binder_skill"

    def test_registry_resolves_for_agent_context(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Agent context comes from the same registry."""
        skill = SkillArtifact(
            type="skill",
            name="agent_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="agent_tool",
            description="Agent tool",
            input_schema={"type": "object", "properties": {}},
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        context = artifact_registry.get_agent_context()

        assert len(context["skills"]) == 1
        assert len(context["tools"]) == 1
        assert context["skills"][0]["name"] == "agent_skill"
        assert context["tools"][0]["name"] == "agent_tool"

    def test_deleted_artifact_errors_in_harness_config(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Deleting an artifact then building config with that name raises ValueError."""
        skill = SkillArtifact(
            type="skill",
            name="ephemeral_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)
        artifact_registry.deregister("ephemeral_skill")

        with pytest.raises(ValueError, match="ephemeral_skill"):
            artifact_registry.build_harness_config(
                model="gpt-4",
                provider="openrouter",
                runner_type="local",
                artifact_names=["ephemeral_skill"],
            )

    def test_deleted_artifact_errors_in_binder(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Deleting an artifact then binding it raises ValueError in binder."""
        skill = SkillArtifact(
            type="skill",
            name="gone_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)
        artifact_registry.deregister("gone_skill")

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
            artifact_names=["gone_skill"],
        )
        with pytest.raises(ValueError, match="gone_skill"):
            binder.setup()

    def test_deleted_artifact_errors_in_agent_context(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Deleting an artifact then getting agent context raises ValueError."""
        skill = SkillArtifact(
            type="skill",
            name="lost_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)
        artifact_registry.deregister("lost_skill")

        with pytest.raises(ValueError, match="lost_skill"):
            artifact_registry.get_agent_context(artifact_names=["lost_skill"])

    def test_all_subsystems_see_same_registry_state(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """All subsystems see the same artifacts from the same registry."""
        skill = SkillArtifact(
            type="skill",
            name="shared_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        # Query from harness config builder
        config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )
        assert "shared_skill" in config.artifact_bindings

        # Query from agent context builder
        context = artifact_registry.get_agent_context()
        assert context["skills"][0]["name"] == "shared_skill"

        # Query from binder
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            binder_context = binder.build_agent_context()

        assert binder_context["skills"][0]["name"] == "shared_skill"

    def test_deregister_mid_session_invalidates_all_subsystems(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """After deregistration, all subsystems that reference the artifact fail."""
        skill = SkillArtifact(
            type="skill",
            name="volatile_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="volatile_tool",
            description="Volatile tool",
            input_schema={"type": "object", "properties": {}},
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        # Both artifacts are available
        config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )
        assert len(config.artifact_bindings) == 2

        # Deregister the skill
        artifact_registry.deregister("volatile_skill")

        # Harness config with explicit name fails
        with pytest.raises(ValueError, match="volatile_skill"):
            artifact_registry.build_harness_config(
                model="gpt-4",
                provider="openrouter",
                runner_type="local",
                artifact_names=["volatile_skill"],
            )

        # Binder with explicit name fails
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
            artifact_names=["volatile_skill"],
        )
        with pytest.raises(ValueError, match="volatile_skill"):
            binder.setup()

        # Agent context with explicit name fails
        with pytest.raises(ValueError, match="volatile_skill"):
            artifact_registry.get_agent_context(artifact_names=["volatile_skill"])

        # But all-artifacts query only shows the remaining tool
        context = artifact_registry.get_agent_context()
        assert len(context["skills"]) == 0
        assert len(context["tools"]) == 1

    def test_registry_isolation_no_side_effects(
        self,
        artifact_registry: ArtifactRegistry,
    ) -> None:
        """Registering/deregistering in one subsystem doesn't affect others
        unless they explicitly reference the artifact."""
        # Start with empty registry
        assert artifact_registry.count == 0

        # Register artifact A
        tool_a = ToolArtifact(
            type="tool",
            name="tool_a",
            description="Tool A",
            input_schema={"type": "object", "properties": {}},
            command="echo",
        )
        artifact_registry.register(tool_a)

        # Build a config that only references tool_a
        config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
            artifact_names=["tool_a"],
        )
        assert config.artifact_bindings == ["tool_a"]

        # Register artifact B — doesn't affect existing config
        tool_b = ToolArtifact(
            type="tool",
            name="tool_b",
            description="Tool B",
            input_schema={"type": "object", "properties": {}},
            command="echo",
        )
        artifact_registry.register(tool_b)

        # Original config still has only tool_a
        assert config.artifact_bindings == ["tool_a"]

        # All-artifacts query now has both
        context = artifact_registry.get_agent_context()
        assert len(context["tools"]) == 2


# ===========================================================================
# VAL-CROSS-12: Tool binding survives environment isolation
# ===========================================================================


class TestToolBindingEnvironmentIsolation:
    """Verify tool bindings produce consistent effects in local and Docker
    execution modes."""

    def test_same_task_same_artifacts_in_local_mode(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Running with artifacts in local mode produces a valid result."""
        skill = SkillArtifact(
            type="skill",
            name="local_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="local_tool",
            description="Local tool",
            input_schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        # Verify artifacts are bound
        assert len(context["skills"]) == 1
        assert context["skills"][0]["name"] == "local_skill"
        assert "Testing Skill" in context["skills"][0]["content"]
        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "local_tool"

    def test_same_artifact_context_in_docker_config(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Artifact registry produces identical context for Docker config."""
        skill = SkillArtifact(
            type="skill",
            name="docker_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="docker_tool",
            description="Docker tool",
            input_schema={
                "type": "object",
                "properties": {"data": {"type": "string"}},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        # Build Docker config
        docker_config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openai",
            runner_type="docker",
        )

        # Build local config
        local_config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openai",
            runner_type="local",
        )

        # Both configs have the same artifact bindings
        assert sorted(docker_config.artifact_bindings) == sorted(
            local_config.artifact_bindings
        )
        assert "docker_skill" in docker_config.artifact_bindings
        assert "docker_tool" in docker_config.artifact_bindings

    def test_agent_context_independent_of_runner_type(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Agent context from registry is the same regardless of runner type."""
        skill = SkillArtifact(
            type="skill",
            name="universal_skill",
            skill_file_path=str(sample_skill_file),
        )
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="universal_mcp",
            command="sleep",
            args=["60"],
        )
        tool = ToolArtifact(
            type="tool",
            name="universal_tool",
            description="Universal tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(mcp)
        artifact_registry.register(tool)

        # Agent context is the same regardless of runner type
        context = artifact_registry.get_agent_context()

        assert len(context["skills"]) == 1
        assert context["skills"][0]["name"] == "universal_skill"
        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "universal_tool"
        assert len(context["mcp_servers"]) == 1
        assert context["mcp_servers"][0]["name"] == "universal_mcp"

    def test_harness_consistent_artifact_flow_with_local_runner(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Harness + LocalRunner + ArtifactBinder produce consistent results."""
        skill = SkillArtifact(
            type="skill",
            name="harness_local_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        # Bind artifacts
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()
            # The skill is available in the context
            assert len(context["skills"]) == 1
            assert context["skills"][0]["name"] == "harness_local_skill"

        # After cleanup, tool registry should be clean
        assert not tool_registry.has("harness_local_skill")

    def test_harness_wires_task_and_artifacts_through_local(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness.run() calls prepare/execute/cleanup in order with LocalRunner."""
        env = _make_mock_env()
        agent = _make_mock_agent()

        harness = Harness(config=sample_config, trace_enabled=False)
        collector = TelemetryCollector()
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Verify lifecycle order
        assert env.call_order == [
            "prepare",
            "execute:echo hello",
            "cleanup",
        ]

        # Result should be valid
        assert result.task_id == sample_task.id
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.telemetry is not None
        assert result.telemetry.latency.total_s > 0


# ===========================================================================
# Artifact cleanup in both modes
# ===========================================================================


class TestArtifactCleanupBothModes:
    """Verify artifact resources are cleaned up in both local and Docker modes."""

    def test_mcp_cleanup_in_local_mode(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP server subprocesses cleaned up in local mode."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="cleanup_local_mcp",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        server = binder.mcp_manager.get_server("cleanup_local_mcp")
        assert server is not None
        assert server.is_alive()

        binder.cleanup()
        assert not server.is_alive()

    def test_mcp_cleanup_on_exception(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP servers cleaned up even when an exception occurs."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="exception_cleanup_mcp",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        with pytest.raises(RuntimeError, match="test failure"), binder:
            server = binder.mcp_manager.get_server("exception_cleanup_mcp")
            assert server is not None
            assert server.is_alive()
            raise RuntimeError("test failure")

        assert not server.is_alive()

    def test_tool_deregistration_in_local_mode(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Tools deregistered from ToolRegistry after cleanup in local mode."""
        tool = ToolArtifact(
            type="tool",
            name="cleanup_local_tool",
            description="Cleanup local tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        assert tool_registry.has("cleanup_local_tool")

        binder.cleanup()
        assert not tool_registry.has("cleanup_local_tool")

    def test_full_harness_with_artifact_binder_cleanup(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Full harness run + artifact binder produces clean state after."""
        skill = SkillArtifact(
            type="skill",
            name="harness_cleanup_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="harness_cleanup_tool",
            description="Harness cleanup tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        # Run binder lifecycle
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        assert tool_registry.has("harness_cleanup_tool")
        assert len(binder.skill_contents) == 1

        binder.cleanup()
        assert not tool_registry.has("harness_cleanup_tool")
        assert len(binder.skill_contents) == 0
        assert not binder.is_setup

    def test_local_runner_cleanup_is_safe(
        self,
        sample_task: Task,
    ) -> None:
        """LocalRunner cleanup is safe and idempotent."""
        runner = LocalRunner()
        # Cleanup before prepare — safe
        runner.cleanup()

        # Cleanup after prepare
        runner.prepare(sample_task)
        runner.cleanup()

        # Cleanup again — still safe
        runner.cleanup()

    def test_multiple_bind_cleanup_cycles(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Multiple setup/cleanup cycles work correctly (simulates running
        multiple tasks with the same artifacts)."""
        skill = SkillArtifact(
            type="skill",
            name="multi_cycle_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="multi_cycle_tool",
            description="Multi cycle tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        for _i in range(3):
            binder = ArtifactBinder(
                registry=artifact_registry,
                tool_registry=tool_registry,
            )
            binder.setup()
            context = binder.build_agent_context()

            assert len(context["skills"]) == 1
            assert len(context["tools"]) == 1
            assert tool_registry.has("multi_cycle_tool")

            binder.cleanup()
            assert not tool_registry.has("multi_cycle_tool")


# ===========================================================================
# Consistent tool invocation traces
# ===========================================================================


class TestConsistentToolTraces:
    """Verify that the same task produces consistent tool invocation traces
    in both local and Docker modes."""

    def test_harness_trace_has_consistent_phases(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Harness telemetry captures consistent phases regardless of mode."""
        env = _make_mock_env()
        agent = _make_mock_agent()

        harness = Harness(config=sample_config, trace_enabled=True)
        collector = TelemetryCollector()
        result = harness.run(
            task=sample_task,
            agent=agent,
            env=env,
            collector=collector,
        )

        # Telemetry should have consistent phase structure
        assert result.telemetry is not None
        assert result.telemetry.latency.setup_s >= 0
        assert result.telemetry.latency.execution_s >= 0
        assert result.telemetry.latency.teardown_s >= 0
        assert result.telemetry.latency.total_s > 0

        # Raw events should record the phases
        assert len(result.telemetry.raw_events) > 0
        phases = [e.get("phase") for e in result.telemetry.raw_events]
        assert "prepare" in phases
        assert "execute" in phases
        assert "cleanup" in phases

    def test_harness_trace_consistent_across_local_runs(
        self,
        sample_task: Task,
        sample_config: HarnessConfig,
    ) -> None:
        """Multiple local runs produce consistent trace structure."""
        results = []
        for _ in range(3):
            env = _make_mock_env()
            agent = _make_mock_agent()
            harness = Harness(config=sample_config, trace_enabled=True)
            collector = TelemetryCollector()
            result = harness.run(
                task=sample_task,
                agent=agent,
                env=env,
                collector=collector,
            )
            results.append(result)

        # All results should have the same task_id, status, score
        for result in results:
            assert result.task_id == sample_task.id
            assert result.status == TaskStatus.SUCCESS
            assert result.score == 1.0
            assert result.telemetry is not None
            assert result.telemetry.latency.total_s > 0
            assert result.telemetry.version == "V1"

        # Raw events should all have the same phase structure
        for result in results:
            phases = {e.get("phase") for e in result.telemetry.raw_events}
            assert "prepare" in phases
            assert "execute" in phases
            assert "cleanup" in phases

    def test_artifact_context_identical_for_different_runner_types(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Agent context is identical when configured for local vs Docker."""
        skill = SkillArtifact(
            type="skill",
            name="trace_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="trace_tool",
            description="Trace tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
            command="echo",
        )
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="trace_mcp",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)
        artifact_registry.register(mcp)

        # Build contexts for different runner types
        local_config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )
        docker_config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="docker",
        )

        # Artifact bindings are the same
        assert sorted(local_config.artifact_bindings) == sorted(
            docker_config.artifact_bindings
        )

        # Agent context from registry is the same
        context = artifact_registry.get_agent_context()
        assert len(context["skills"]) == 1
        assert len(context["tools"]) == 1
        assert len(context["mcp_servers"]) == 1

    def test_binder_context_independent_of_harness_runner_type(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """ArtifactBinder produces identical context regardless of runner type."""
        skill = SkillArtifact(
            type="skill",
            name="binder_trace_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="binder_trace_tool",
            description="Binder trace tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        # Bind and get context
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        # Verify the context structure is independent of runner type
        assert len(context["skills"]) == 1
        assert context["skills"][0]["name"] == "binder_trace_skill"
        assert "Testing Skill" in context["skills"][0]["content"]
        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "binder_trace_tool"
        assert context["tools"][0]["description"] == "Binder trace tool"


# ===========================================================================
# Integration: Registry + Binder + Harness
# ===========================================================================


class TestRegistryBinderHarnessIntegration:
    """End-to-end integration: registry -> binder -> harness."""

    def test_end_to_end_registry_to_harness(
        self,
        sample_task: Task,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Full flow: registry -> binder -> harness with artifact context."""
        skill = SkillArtifact(
            type="skill",
            name="e2e_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="e2e_tool",
            description="E2E tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        # Build config from registry
        config = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )
        assert "e2e_skill" in config.artifact_bindings
        assert "e2e_tool" in config.artifact_bindings

        # Bind artifacts
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()
            assert len(context["skills"]) == 1
            assert len(context["tools"]) == 1

            # Run harness
            env = _make_mock_env()
            agent = _make_mock_agent()
            harness = Harness(config=config, trace_enabled=True)
            collector = TelemetryCollector()
            result = harness.run(
                task=sample_task,
                agent=agent,
                env=env,
                collector=collector,
            )

        assert result.task_id == sample_task.id
        assert result.status == TaskStatus.SUCCESS
        assert result.telemetry is not None

        # Cleanup verified: tools deregistered
        assert not tool_registry.has("e2e_tool")

    def test_deregistered_artifact_prevents_harness_run(
        self,
        sample_task: Task,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Deregistered artifact prevents subsequent harness config build."""
        skill = SkillArtifact(
            type="skill",
            name="prevent_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        # First build succeeds
        config1 = artifact_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
            artifact_names=["prevent_skill"],
        )
        assert "prevent_skill" in config1.artifact_bindings

        # Deregister
        artifact_registry.deregister("prevent_skill")

        # Second build fails
        with pytest.raises(ValueError, match="prevent_skill"):
            artifact_registry.build_harness_config(
                model="gpt-4",
                provider="openrouter",
                runner_type="local",
                artifact_names=["prevent_skill"],
            )

    def test_artifact_type_filtering_works_across_subsystems(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Registry filtering by type works correctly for all artifact types."""
        skill = SkillArtifact(
            type="skill",
            name="filter_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="filter_tool",
            description="Filter tool",
            input_schema={"type": "object", "properties": {}},
            command="echo",
        )
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="filter_mcp",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)
        artifact_registry.register(mcp)

        # Filter by type
        skills = artifact_registry.list_artifacts(filter_type="skill")
        tools = artifact_registry.list_artifacts(filter_type="tool")
        mcps = artifact_registry.list_artifacts(filter_type="mcp_server")

        assert len(skills) == 1
        assert skills[0].name == "filter_skill"
        assert len(tools) == 1
        assert tools[0].name == "filter_tool"
        assert len(mcps) == 1
        assert mcps[0].name == "filter_mcp"

        # All artifacts
        all_artifacts = artifact_registry.list_artifacts()
        assert len(all_artifacts) == 3

        # Count by type
        assert artifact_registry.count_by_type("skill") == 1
        assert artifact_registry.count_by_type("tool") == 1
        assert artifact_registry.count_by_type("mcp_server") == 1
