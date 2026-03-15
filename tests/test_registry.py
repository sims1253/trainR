"""Tests for ArtifactRegistry.

Covers VAL-FOUND-10 and VAL-FOUND-11:
- Dynamic registration and lookup by name/type
- Duplicate detection (rejects unless overwrite=True)
- list_artifacts(filter_type=...) returns filtered results
- Automatic wiring into HarnessConfig/agent context
"""

from __future__ import annotations

import pytest

from grist_mill.registry import ArtifactRegistry
from grist_mill.schemas.artifact import (
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_artifact() -> ToolArtifact:
    """A sample ToolArtifact for testing."""
    return ToolArtifact(
        name="grep",
        description="Search files using ripgrep",
        input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}},
    )


@pytest.fixture
def mcp_artifact() -> MCPServerArtifact:
    """A sample MCPServerArtifact for testing."""
    return MCPServerArtifact(
        name="fs-server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )


@pytest.fixture
def skill_artifact() -> SkillArtifact:
    """A sample SkillArtifact for testing."""
    return SkillArtifact(
        name="code-review",
        skill_file_path="/skills/code-review/SKILL.md",
    )


@pytest.fixture
def empty_registry() -> ArtifactRegistry:
    """A fresh empty ArtifactRegistry."""
    return ArtifactRegistry()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for ArtifactRegistry.register()."""

    def test_register_tool_artifact(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Registering a tool artifact makes it retrievable by name."""
        empty_registry.register(tool_artifact)
        result = empty_registry.get("grep")
        assert result is not None
        assert result is tool_artifact

    def test_register_mcp_artifact(
        self, empty_registry: ArtifactRegistry, mcp_artifact: MCPServerArtifact
    ) -> None:
        """Registering an MCP artifact makes it retrievable by name."""
        empty_registry.register(mcp_artifact)
        result = empty_registry.get("fs-server")
        assert result is not None
        assert result is mcp_artifact

    def test_register_skill_artifact(
        self, empty_registry: ArtifactRegistry, skill_artifact: SkillArtifact
    ) -> None:
        """Registering a skill artifact makes it retrievable by name."""
        empty_registry.register(skill_artifact)
        result = empty_registry.get("code-review")
        assert result is not None
        assert result is skill_artifact

    def test_register_multiple_artifacts(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """Multiple artifacts of different types can coexist."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)
        assert empty_registry.get("grep") is tool_artifact
        assert empty_registry.get("fs-server") is mcp_artifact
        assert empty_registry.get("code-review") is skill_artifact

    def test_register_duplicate_raises(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Registering a duplicate name without overwrite raises ValueError."""
        empty_registry.register(tool_artifact)
        with pytest.raises(ValueError, match="grep"):
            empty_registry.register(tool_artifact)

    def test_register_duplicate_with_overwrite(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Registering a duplicate with overwrite=True replaces the artifact."""
        empty_registry.register(tool_artifact)
        new_tool = ToolArtifact(
            name="grep",
            description="Updated description",
            input_schema={"type": "object"},
        )
        empty_registry.register(new_tool, overwrite=True)
        result = empty_registry.get("grep")
        assert result is new_tool
        assert result.description == "Updated description"

    def test_register_duplicate_with_overwrite_false(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Explicitly setting overwrite=False still raises on duplicate."""
        empty_registry.register(tool_artifact)
        with pytest.raises(ValueError):
            empty_registry.register(tool_artifact, overwrite=False)


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


class TestLookup:
    """Tests for ArtifactRegistry.get() and lookup methods."""

    def test_get_nonexistent_returns_none(self, empty_registry: ArtifactRegistry) -> None:
        """Looking up a non-existent name returns None."""
        assert empty_registry.get("nonexistent") is None

    def test_get_by_type_tool(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Lookup by type returns only artifacts of that type."""
        empty_registry.register(tool_artifact)
        tools = empty_registry.get_by_type("tool")
        assert len(tools) == 1
        assert tools[0] is tool_artifact

    def test_get_by_type_mcp_server(
        self, empty_registry: ArtifactRegistry, mcp_artifact: MCPServerArtifact
    ) -> None:
        """Lookup by type returns only MCP server artifacts."""
        empty_registry.register(mcp_artifact)
        mcps = empty_registry.get_by_type("mcp_server")
        assert len(mcps) == 1
        assert mcps[0] is mcp_artifact

    def test_get_by_type_skill(
        self, empty_registry: ArtifactRegistry, skill_artifact: SkillArtifact
    ) -> None:
        """Lookup by type returns only skill artifacts."""
        empty_registry.register(skill_artifact)
        skills = empty_registry.get_by_type("skill")
        assert len(skills) == 1
        assert skills[0] is skill_artifact

    def test_get_by_type_filters_correctly(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """Filtering by type returns only matching artifacts."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)

        assert len(empty_registry.get_by_type("tool")) == 1
        assert len(empty_registry.get_by_type("mcp_server")) == 1
        assert len(empty_registry.get_by_type("skill")) == 1

    def test_get_by_type_no_matches(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Filtering by a type with no artifacts returns empty list."""
        empty_registry.register(tool_artifact)
        assert empty_registry.get_by_type("mcp_server") == []

    def test_has_true_when_registered(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """has() returns True for a registered artifact."""
        empty_registry.register(tool_artifact)
        assert empty_registry.has("grep") is True

    def test_has_false_when_not_registered(self, empty_registry: ArtifactRegistry) -> None:
        """has() returns False for a non-registered artifact."""
        assert empty_registry.has("nonexistent") is False


# ---------------------------------------------------------------------------
# List and count tests
# ---------------------------------------------------------------------------


class TestListArtifacts:
    """Tests for ArtifactRegistry.list_artifacts() and related methods."""

    def test_list_all(self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact) -> None:
        """list_artifacts() without filter returns all artifacts."""
        empty_registry.register(tool_artifact)
        all_artifacts = empty_registry.list_artifacts()
        assert len(all_artifacts) == 1
        assert all_artifacts[0] is tool_artifact

    def test_list_filtered_by_type(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """list_artifacts(filter_type=...) returns only matching artifacts."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)

        tools = empty_registry.list_artifacts(filter_type="tool")
        assert len(tools) == 1
        assert isinstance(tools[0], ToolArtifact)

        mcps = empty_registry.list_artifacts(filter_type="mcp_server")
        assert len(mcps) == 1
        assert isinstance(mcps[0], MCPServerArtifact)

        skills = empty_registry.list_artifacts(filter_type="skill")
        assert len(skills) == 1
        assert isinstance(skills[0], SkillArtifact)

    def test_list_empty_registry(self, empty_registry: ArtifactRegistry) -> None:
        """list_artifacts() on empty registry returns empty list."""
        assert empty_registry.list_artifacts() == []

    def test_count(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
    ) -> None:
        """count returns the number of registered artifacts."""
        assert empty_registry.count == 0
        empty_registry.register(tool_artifact)
        assert empty_registry.count == 1
        empty_registry.register(mcp_artifact)
        assert empty_registry.count == 2

    def test_count_by_type(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
    ) -> None:
        """count_by_type returns the number of artifacts of a specific type."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        assert empty_registry.count_by_type("tool") == 1
        assert empty_registry.count_by_type("mcp_server") == 1
        assert empty_registry.count_by_type("skill") == 0

    def test_names(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
    ) -> None:
        """names returns all registered artifact names."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        names = empty_registry.names
        assert "grep" in names
        assert "fs-server" in names
        assert len(names) == 2


# ---------------------------------------------------------------------------
# Deregistration tests
# ---------------------------------------------------------------------------


class TestDeregistration:
    """Tests for ArtifactRegistry.deregister()."""

    def test_deregister_existing(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Deregistering an existing artifact removes it."""
        empty_registry.register(tool_artifact)
        empty_registry.deregister("grep")
        assert empty_registry.get("grep") is None
        assert empty_registry.count == 0

    def test_deregister_nonexistent_raises(self, empty_registry: ArtifactRegistry) -> None:
        """Deregistering a non-existent artifact raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            empty_registry.deregister("nonexistent")

    def test_deregister_nonexistent_ignore(
        self, empty_registry: ArtifactRegistry, tool_artifact: ToolArtifact
    ) -> None:
        """Deregister with ignore_missing=True does not raise for nonexistent."""
        empty_registry.register(tool_artifact)
        empty_registry.deregister("nonexistent", ignore_missing=True)
        # Original artifact still there
        assert empty_registry.get("grep") is tool_artifact


# ---------------------------------------------------------------------------
# Clear tests
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for ArtifactRegistry.clear()."""

    def test_clear_removes_all(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
    ) -> None:
        """clear() removes all registered artifacts."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        assert empty_registry.count == 2
        empty_registry.clear()
        assert empty_registry.count == 0
        assert empty_registry.list_artifacts() == []


# ---------------------------------------------------------------------------
# Wiring into HarnessConfig (VAL-FOUND-11)
# ---------------------------------------------------------------------------


class TestWiring:
    """Tests for automatic wiring of artifacts into HarnessConfig.

    Validates VAL-FOUND-11: Registered artifacts are wired into agent
    execution context without manual config editing.
    """

    def test_wire_into_harness_config(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """Artifacts are wired into the agent context when building config."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)

        config = empty_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )

        # The artifact names should appear in artifact_bindings
        assert "grep" in config.artifact_bindings
        assert "fs-server" in config.artifact_bindings
        assert "code-review" in config.artifact_bindings

    def test_wire_selected_artifacts(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
    ) -> None:
        """Only selected artifact names are wired when specified."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)

        config = empty_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
            artifact_names=["grep"],
        )

        assert "grep" in config.artifact_bindings
        assert "fs-server" not in config.artifact_bindings

    def test_wire_nonexistent_artifact_raises(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
    ) -> None:
        """Wiring a nonexistent artifact name raises ValueError."""
        empty_registry.register(tool_artifact)
        with pytest.raises(ValueError, match="nonexistent-tool"):
            empty_registry.build_harness_config(
                model="gpt-4",
                provider="openrouter",
                runner_type="local",
                artifact_names=["grep", "nonexistent-tool"],
            )

    def test_wire_empty_registry_empty_bindings(self, empty_registry: ArtifactRegistry) -> None:
        """Wiring with no artifacts produces empty artifact_bindings."""
        config = empty_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
        )
        assert config.artifact_bindings == []

    def test_wire_all_with_no_selection_wires_everything(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """When artifact_names is None, all registered artifacts are wired."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)

        config = empty_registry.build_harness_config(
            model="gpt-4",
            provider="openrouter",
            runner_type="local",
            # artifact_names defaults to None -> all artifacts
        )

        assert len(config.artifact_bindings) == 3

    def test_get_agent_context(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """get_agent_context returns structured artifact data for the agent."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)

        context = empty_registry.get_agent_context(["grep", "code-review"])

        # Should have tools and skills
        assert "tools" in context
        assert "skills" in context
        assert "mcp_servers" in context

        assert len(context["tools"]) == 1
        assert context["tools"][0]["name"] == "grep"

        assert len(context["skills"]) == 1
        assert context["skills"][0]["name"] == "code-review"

        # fs-server was not requested
        assert len(context["mcp_servers"]) == 0

    def test_get_agent_context_all(
        self,
        empty_registry: ArtifactRegistry,
        tool_artifact: ToolArtifact,
        mcp_artifact: MCPServerArtifact,
        skill_artifact: SkillArtifact,
    ) -> None:
        """get_agent_context with no selection returns all artifacts."""
        empty_registry.register(tool_artifact)
        empty_registry.register(mcp_artifact)
        empty_registry.register(skill_artifact)

        context = empty_registry.get_agent_context()
        assert len(context["tools"]) == 1
        assert len(context["mcp_servers"]) == 1
        assert len(context["skills"]) == 1
