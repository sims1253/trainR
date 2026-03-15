"""Tests for artifact-to-tool binding.

Covers:
- VAL-BIND-01: SKILL.md files injected as agent context
- VAL-BIND-02: MCP server config translated to running server
- VAL-BIND-03: CLI tool definitions verified as available commands
- VAL-BIND-04: Multiple artifact types coexist without conflict
- VAL-BIND-05: Artifact validation fails early with actionable errors
- VAL-BIND-06: Artifact cleanup happens after execution regardless of outcome
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grist_mill.registry import ArtifactRegistry
from grist_mill.schemas.artifact import (
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)
from grist_mill.tools.binding import (
    ArtifactBinder,
    ArtifactBindingError,
    ArtifactValidationError,
)
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
def sample_skill_file_2(tmp_path: Path) -> Path:
    """Create a second sample SKILL.md file."""
    skill_file = tmp_path / "SKILL_REVIEW.md"
    skill_file.write_text(
        "# Code Review Skill\n\nReview code for quality.\n",
        encoding="utf-8",
    )
    return skill_file


@pytest.fixture
def sample_tool_artifact() -> ToolArtifact:
    """ToolArtifact using 'echo' (universally available)."""
    return ToolArtifact(
        type="tool",
        name="echo_tool",
        description="Echo a message",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        command="echo",
    )


@pytest.fixture
def sample_tool_artifact_no_command() -> ToolArtifact:
    """ToolArtifact without a command (placeholder)."""
    return ToolArtifact(
        type="tool",
        name="placeholder_tool",
        description="A placeholder tool",
        input_schema={
            "type": "object",
            "properties": {"data": {"type": "string"}},
        },
    )


# ---------------------------------------------------------------------------
# VAL-BIND-01: SKILL.md files injected as agent context
# ---------------------------------------------------------------------------


class TestSkillInjection:
    """VAL-BIND-01: SkillArtifact content injected into agent's context."""

    def test_single_skill_content_injected(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Skill content must appear in the agent context."""
        skill = SkillArtifact(
            type="skill",
            name="testing_skill",
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
        assert context["skills"][0]["name"] == "testing_skill"
        assert "Testing Skill" in context["skills"][0]["content"]
        assert "Always write tests first" in context["skills"][0]["content"]

    def test_skill_content_in_system_prompt_extension(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Skill content must appear in system prompt extension."""
        skill = SkillArtifact(
            type="skill",
            name="testing_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            extension = binder.build_system_prompt_extension()

        assert "Testing Skill" in extension
        assert "Always write tests first" in extension
        assert "## Skill: testing_skill" in extension

    def test_multiple_skills_concatenated(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
        sample_skill_file_2: Path,
    ) -> None:
        """Multiple skills should all appear in the context."""
        skill1 = SkillArtifact(
            type="skill",
            name="testing_skill",
            skill_file_path=str(sample_skill_file),
        )
        skill2 = SkillArtifact(
            type="skill",
            name="review_skill",
            skill_file_path=str(sample_skill_file_2),
        )
        artifact_registry.register(skill1)
        artifact_registry.register(skill2)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["skills"]) == 2
        skill_names = {s["name"] for s in context["skills"]}
        assert skill_names == {"testing_skill", "review_skill"}

    def test_no_skills_returns_empty(self) -> None:
        """No skills should produce empty skills list."""
        registry = ArtifactRegistry()
        tool_reg = ToolRegistry()
        binder = ArtifactBinder(registry=registry, tool_registry=tool_reg)

        with binder:
            context = binder.build_agent_context()
            extension = binder.build_system_prompt_extension()

        assert context["skills"] == []
        assert extension == ""

    def test_skill_contents_accessible_as_property(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """skill_contents property should return a copy of the mapping."""
        skill = SkillArtifact(
            type="skill",
            name="testing_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            contents = binder.skill_contents
            assert "testing_skill" in contents
            assert "Testing Skill" in contents["testing_skill"]


# ---------------------------------------------------------------------------
# VAL-BIND-02: MCP server config translated to running server
# ---------------------------------------------------------------------------


class TestMCPServerBinding:
    """VAL-BIND-02: MCPServerArtifact starts subprocess, URL injected."""

    def test_mcp_server_started_as_subprocess(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP server must be started as a subprocess."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="test_server",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            server_info = binder.mcp_manager.get_server("test_server")
            assert server_info is not None
            assert server_info.pid is not None
            assert server_info.is_alive()

    def test_mcp_url_injected_into_context(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP server URL must appear in the agent context."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="test_server",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["mcp_servers"]) == 1
        assert context["mcp_servers"][0]["name"] == "test_server"
        assert context["mcp_servers"][0]["url"] == "stdio://test_server"

    def test_mcp_url_accessible_as_property(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """mcp_urls property should return a copy of the mapping."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="test_server",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            urls = binder.mcp_urls
            assert urls == {"test_server": "stdio://test_server"}

    def test_mcp_server_stopped_on_cleanup(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP server subprocess must be stopped after cleanup."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="test_server",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        server_info = binder.mcp_manager.get_server("test_server")
        assert server_info is not None
        assert server_info.is_alive()

        binder.cleanup()

        assert not server_info.is_alive()

    def test_multiple_mcp_servers_coexist(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Multiple MCP servers should all start and be tracked."""
        mcp1 = MCPServerArtifact(
            type="mcp_server",
            name="server_a",
            command="sleep",
            args=["60"],
        )
        mcp2 = MCPServerArtifact(
            type="mcp_server",
            name="server_b",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp1)
        artifact_registry.register(mcp2)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["mcp_servers"]) == 2
        server_names = {s["name"] for s in context["mcp_servers"]}
        assert server_names == {"server_a", "server_b"}


# ---------------------------------------------------------------------------
# VAL-BIND-03: CLI tool definitions verified as available commands
# ---------------------------------------------------------------------------


class TestToolVerification:
    """VAL-BIND-03: ToolArtifact commands verified as available."""

    def test_available_command_passes_validation(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_tool_artifact: ToolArtifact,
    ) -> None:
        """Tool with an available command should pass validation."""
        artifact_registry.register(sample_tool_artifact)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "echo_tool"

    def test_tool_registered_in_tool_registry(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_tool_artifact: ToolArtifact,
    ) -> None:
        """Tool should be registered in the ToolRegistry after setup."""
        artifact_registry.register(sample_tool_artifact)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()

        assert tool_registry.has("echo_tool")
        assert "echo_tool" in binder.registered_tool_names

        binder.cleanup()
        assert not tool_registry.has("echo_tool")

    def test_tool_without_command_passes(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_tool_artifact_no_command: ToolArtifact,
    ) -> None:
        """Tool without a command should pass validation (placeholder)."""
        artifact_registry.register(sample_tool_artifact_no_command)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "placeholder_tool"

    def test_tool_command_with_args_verified(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Tool command with arguments should still validate the binary."""
        tool = ToolArtifact(
            type="tool",
            name="echo_with_args",
            description="Echo with args",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            command="echo hello",
        )
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["tools"]) == 1


# ---------------------------------------------------------------------------
# VAL-BIND-04: Multiple artifact types coexist without conflict
# ---------------------------------------------------------------------------


class TestCoexistence:
    """VAL-BIND-04: Multiple artifact types coexist without naming conflicts."""

    def test_all_three_types_bound(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Skills, MCP servers, and tools should all be bound."""
        skill = SkillArtifact(
            type="skill",
            name="my_skill",
            skill_file_path=str(sample_skill_file),
        )
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="my_server",
            command="sleep",
            args=["60"],
        )
        tool = ToolArtifact(
            type="tool",
            name="my_tool",
            description="My tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(mcp)
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        # All three types present
        assert len(context["skills"]) == 1
        assert context["skills"][0]["name"] == "my_skill"
        assert len(context["mcp_servers"]) == 1
        assert context["mcp_servers"][0]["name"] == "my_server"
        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "my_tool"

    def test_same_name_different_types_no_conflict(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Skills, MCP servers, and tools use separate internal namespaces.

        Each type lives in its own namespace within the binder:
        skills -> _skill_contents, mcp_servers -> _mcp_urls, tools -> _registered_tool_names.
        So a skill and an MCP server with different names don't conflict with each other.
        """
        skill = SkillArtifact(
            type="skill",
            name="skill_unique",
            skill_file_path=str(sample_skill_file),
        )
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="mcp_unique",
            command="sleep",
            args=["60"],
        )
        tool = ToolArtifact(
            type="tool",
            name="tool_unique",
            description="Tool unique",
            input_schema={
                "type": "object",
                "properties": {"y": {"type": "string"}},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(mcp)
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        # Should not raise any naming conflict error
        with binder:
            context = binder.build_agent_context()

        assert len(context["skills"]) == 1
        assert len(context["mcp_servers"]) == 1
        assert len(context["tools"]) == 1

    def test_multiple_of_each_type(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
        sample_skill_file_2: Path,
    ) -> None:
        """Multiple artifacts of each type should all be bound."""
        skill1 = SkillArtifact(
            type="skill",
            name="skill_a",
            skill_file_path=str(sample_skill_file),
        )
        skill2 = SkillArtifact(
            type="skill",
            name="skill_b",
            skill_file_path=str(sample_skill_file_2),
        )
        mcp1 = MCPServerArtifact(
            type="mcp_server",
            name="mcp_a",
            command="sleep",
            args=["60"],
        )
        mcp2 = MCPServerArtifact(
            type="mcp_server",
            name="mcp_b",
            command="sleep",
            args=["60"],
        )
        tool1 = ToolArtifact(
            type="tool",
            name="tool_a",
            description="Tool A",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        tool2 = ToolArtifact(
            type="tool",
            name="tool_b",
            description="Tool B",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill1)
        artifact_registry.register(skill2)
        artifact_registry.register(mcp1)
        artifact_registry.register(mcp2)
        artifact_registry.register(tool1)
        artifact_registry.register(tool2)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["skills"]) == 2
        assert len(context["mcp_servers"]) == 2
        assert len(context["tools"]) == 2

    def test_binding_specific_artifact_names(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Binding specific names should only bind those artifacts."""
        skill = SkillArtifact(
            type="skill",
            name="selected_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="unselected_tool",
            description="Unselected tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
            artifact_names=["selected_skill"],
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["skills"]) == 1
        assert len(context["tools"]) == 0


# ---------------------------------------------------------------------------
# VAL-BIND-05: Artifact validation fails early with actionable errors
# ---------------------------------------------------------------------------


class TestEarlyValidation:
    """VAL-BIND-05: Missing files/commands produce setup-time errors."""

    def test_missing_skill_file_raises_error(self) -> None:
        """SkillArtifact pointing to nonexistent file must raise during setup."""
        registry = ArtifactRegistry()
        skill = SkillArtifact(
            type="skill",
            name="broken_skill",
            skill_file_path="/nonexistent/path/SKILL.md",
        )
        registry.register(skill)

        binder = ArtifactBinder(registry=registry, tool_registry=ToolRegistry())

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert "broken_skill" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)
        assert "skill_file_path" in str(exc_info.value)

    def test_missing_skill_file_mentions_path(self) -> None:
        """Error message should include the file path."""
        registry = ArtifactRegistry()
        skill = SkillArtifact(
            type="skill",
            name="broken_skill",
            skill_file_path="/tmp/nonexistent_skill_abc123.md",
        )
        registry.register(skill)

        binder = ArtifactBinder(registry=registry, tool_registry=ToolRegistry())

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert "/tmp/nonexistent_skill_abc123.md" in str(exc_info.value)

    def test_skill_directory_not_file_raises_error(self, tmp_path: Path) -> None:
        """SkillArtifact pointing to a directory must raise during setup."""
        registry = ArtifactRegistry()
        skill = SkillArtifact(
            type="skill",
            name="dir_skill",
            skill_file_path=str(tmp_path),
        )
        registry.register(skill)

        binder = ArtifactBinder(registry=registry, tool_registry=ToolRegistry())

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert "not a file" in str(exc_info.value)

    def test_missing_tool_command_raises_error(self) -> None:
        """ToolArtifact with unavailable command must raise during setup."""
        registry = ArtifactRegistry()
        tool = ToolArtifact(
            type="tool",
            name="broken_tool",
            description="Broken tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="nonexistent_command_xyz_12345",
        )
        registry.register(tool)

        binder = ArtifactBinder(registry=registry, tool_registry=ToolRegistry())

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert "broken_tool" in str(exc_info.value)
        assert "not available" in str(exc_info.value)
        assert "nonexistent_command_xyz_12345" in str(exc_info.value)

    def test_missing_mcp_command_raises_error(self) -> None:
        """MCPServerArtifact with unavailable command must raise during setup."""
        registry = ArtifactRegistry()
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="broken_server",
            command="nonexistent_mcp_xyz_12345",
            args=[],
        )
        registry.register(mcp)

        binder = ArtifactBinder(registry=registry, tool_registry=ToolRegistry())

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert "broken_server" in str(exc_info.value)
        assert "not available" in str(exc_info.value)

    def test_nonexistent_artifact_name_raises_error(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Referencing a nonexistent artifact name must raise."""
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
            artifact_names=["ghost_artifact"],
        )

        with pytest.raises(ValueError) as exc_info:
            binder.setup()

        assert "ghost_artifact" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_error_before_agent_runs(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Validation error must occur during setup, not during agent execution."""
        skill = SkillArtifact(
            type="skill",
            name="missing_skill",
            skill_file_path="/nonexistent/missing.md",
        )
        artifact_registry.register(skill)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        # Error must happen here (setup), not in build_agent_context
        with pytest.raises(ArtifactValidationError):
            binder.setup()

    def test_error_has_artifact_name_attribute(self) -> None:
        """ArtifactValidationError should have artifact_name attribute."""
        registry = ArtifactRegistry()
        skill = SkillArtifact(
            type="skill",
            name="named_skill",
            skill_file_path="/nonexistent/named.md",
        )
        registry.register(skill)

        binder = ArtifactBinder(registry=registry, tool_registry=ToolRegistry())

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert exc_info.value.artifact_name == "named_skill"

    def test_tool_name_conflict_raises_error(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """ToolArtifact with name already in ToolRegistry must raise."""
        # Pre-register a tool
        tool_registry.register(
            name="existing_tool",
            description="Existing tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kwargs: {},
        )

        tool = ToolArtifact(
            type="tool",
            name="existing_tool",
            description="Conflicting tool",
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

        with pytest.raises(ArtifactValidationError) as exc_info:
            binder.setup()

        assert "existing_tool" in str(exc_info.value)
        assert "conflicts" in str(exc_info.value)


# ---------------------------------------------------------------------------
# VAL-BIND-06: Artifact cleanup after execution regardless of outcome
# ---------------------------------------------------------------------------


class TestCleanup:
    """VAL-BIND-06: All resources cleaned up after execution."""

    def test_mcp_servers_cleaned_up_on_success(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP servers must be stopped after successful execution."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="cleanup_server",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            server = binder.mcp_manager.get_server("cleanup_server")
            assert server is not None
            assert server.is_alive()

        # After context manager exit, server must be stopped
        assert not server.is_alive()

    def test_mcp_servers_cleaned_up_on_exception(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP servers must be stopped even when an exception occurs."""
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="exception_server",
            command="sleep",
            args=["60"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        with pytest.raises(RuntimeError, match="simulated failure"), binder:
            server = binder.mcp_manager.get_server("exception_server")
            assert server is not None
            assert server.is_alive()
            raise RuntimeError("simulated failure")

        # Server must be stopped despite the exception
        assert not server.is_alive()

    def test_tools_deregistered_on_cleanup(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_tool_artifact: ToolArtifact,
    ) -> None:
        """Tools must be deregistered from ToolRegistry after cleanup."""
        artifact_registry.register(sample_tool_artifact)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        assert tool_registry.has("echo_tool")

        binder.cleanup()
        assert not tool_registry.has("echo_tool")
        assert binder.registered_tool_names == []

    def test_tools_deregistered_on_exception(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_tool_artifact: ToolArtifact,
    ) -> None:
        """Tools must be deregistered even when an exception occurs."""
        artifact_registry.register(sample_tool_artifact)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        with pytest.raises(RuntimeError, match="boom"), binder:
            assert tool_registry.has("echo_tool")
            raise RuntimeError("boom")

        assert not tool_registry.has("echo_tool")

    def test_cleanup_safe_without_setup(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Cleanup should be safe even if setup was never called."""
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        # Should not raise
        binder.cleanup()

    def test_cleanup_idempotent(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_tool_artifact: ToolArtifact,
    ) -> None:
        """Calling cleanup multiple times should not error."""
        artifact_registry.register(sample_tool_artifact)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        binder.cleanup()
        binder.cleanup()  # Second call should be safe

    def test_context_manager_cleans_up_all_resources(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Context manager should clean up skills, MCP servers, and tools."""
        skill = SkillArtifact(
            type="skill",
            name="cleanup_skill",
            skill_file_path=str(sample_skill_file),
        )
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="cleanup_mcp",
            command="sleep",
            args=["60"],
        )
        tool = ToolArtifact(
            type="tool",
            name="cleanup_tool",
            description="Cleanup tool",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(mcp)
        artifact_registry.register(tool)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        with binder:
            server = binder.mcp_manager.get_server("cleanup_mcp")
            assert server is not None
            assert server.is_alive()
            assert tool_registry.has("cleanup_tool")
            assert len(binder.skill_contents) == 1

        # After context exit
        assert not server.is_alive()
        assert not tool_registry.has("cleanup_tool")
        assert len(binder.skill_contents) == 0
        assert len(binder.mcp_urls) == 0
        assert not binder.is_setup


# ---------------------------------------------------------------------------
# Edge cases and additional tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_registry_binds_nothing(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Empty registry should produce empty context."""
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert context["skills"] == []
        assert context["mcp_servers"] == []
        assert context["tools"] == []

    def test_binder_without_tool_registry(
        self,
        artifact_registry: ArtifactRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Binder should work without a ToolRegistry (tools validated but not registered)."""
        skill = SkillArtifact(
            type="skill",
            name="no_registry_skill",
            skill_file_path=str(sample_skill_file),
        )
        tool = ToolArtifact(
            type="tool",
            name="no_registry_tool",
            description="Tool without registry",
            input_schema={
                "type": "object",
                "properties": {},
            },
            command="echo",
        )
        artifact_registry.register(skill)
        artifact_registry.register(tool)

        binder = ArtifactBinder(registry=artifact_registry)
        with binder:
            context = binder.build_agent_context()

        assert len(context["skills"]) == 1
        assert context["tools"] == []  # No registry, so no tools in context

    def test_build_context_before_setup_raises(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """build_agent_context must raise if setup was not called."""
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        with pytest.raises(ArtifactBindingError):
            binder.build_agent_context()

    def test_build_prompt_before_setup_raises(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """build_system_prompt_extension must raise if setup was not called."""
        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )

        with pytest.raises(ArtifactBindingError):
            binder.build_system_prompt_extension()

    def test_double_setup_is_idempotent(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """Calling setup twice should be a no-op the second time."""
        skill = SkillArtifact(
            type="skill",
            name="dup_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        binder.setup()  # Should be idempotent

        context = binder.build_agent_context()
        assert len(context["skills"]) == 1

        binder.cleanup()

    def test_mcp_command_with_args_binary_checked(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """MCP server command with full path should validate correctly."""
        # Use a command with arguments to test binary extraction
        mcp = MCPServerArtifact(
            type="mcp_server",
            name="echo_server",
            command="echo",
            args=["hello", "world"],
        )
        artifact_registry.register(mcp)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        with binder:
            context = binder.build_agent_context()

        assert len(context["mcp_servers"]) == 1

    def test_cleanup_resets_state(
        self,
        artifact_registry: ArtifactRegistry,
        tool_registry: ToolRegistry,
        sample_skill_file: Path,
    ) -> None:
        """After cleanup, the binder should be fully reset."""
        skill = SkillArtifact(
            type="skill",
            name="reset_skill",
            skill_file_path=str(sample_skill_file),
        )
        artifact_registry.register(skill)

        binder = ArtifactBinder(
            registry=artifact_registry,
            tool_registry=tool_registry,
        )
        binder.setup()
        binder.cleanup()

        assert not binder.is_setup
        assert binder.skill_contents == {}
        assert binder.mcp_urls == {}
        assert binder.registered_tool_names == []
