"""Tests for Artifact discriminated union schema.

Covers VAL-FOUND-04 and VAL-FOUND-12:
- Discriminated union routing based on type field
- Each variant enforces its own required fields
- Helpful error messages for developer experience
- JSON round-trip serialization
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grist_mill.schemas.artifact import (
    Artifact,
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)

# ---------------------------------------------------------------------------
# ToolArtifact — valid construction
# ---------------------------------------------------------------------------


class TestToolArtifact:
    """Tests for ToolArtifact variant."""

    def test_valid_construction_minimal(self) -> None:
        """ToolArtifact with required fields only."""
        artifact = ToolArtifact(
            name="grep",
            description="Search files using ripgrep",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}},
        )
        assert artifact.type == "tool"
        assert artifact.name == "grep"
        assert artifact.description == "Search files using ripgrep"
        assert artifact.input_schema["type"] == "object"

    def test_valid_construction_with_optional(self) -> None:
        """ToolArtifact with all optional fields."""
        artifact = ToolArtifact(
            name="ls",
            description="List directory contents",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            command="ls",
        )
        assert artifact.output_schema is not None
        assert artifact.command == "ls"

    def test_missing_name_raises(self) -> None:
        """ToolArtifact requires 'name' field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolArtifact(
                description="A tool",
                input_schema={"type": "object"},
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "name" in field_names

    def test_missing_description_raises(self) -> None:
        """ToolArtifact requires 'description' field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolArtifact(
                name="mytool",
                input_schema={"type": "object"},
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "description" in field_names

    def test_missing_input_schema_raises(self) -> None:
        """ToolArtifact requires 'input_schema' field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolArtifact(
                name="mytool",
                description="Does things",
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "input_schema" in field_names

    def test_empty_name_raises(self) -> None:
        """ToolArtifact name must be non-empty."""
        with pytest.raises(ValidationError):
            ToolArtifact(
                name="",
                description="A tool",
                input_schema={"type": "object"},
            )

    def test_json_round_trip(self) -> None:
        """ToolArtifact serializes and deserializes without data loss."""
        original = ToolArtifact(
            name="find",
            description="Find files by name",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        json_str = original.model_dump_json()
        restored = ToolArtifact.model_validate_json(json_str)
        assert restored == original

    def test_type_field_is_tool(self) -> None:
        """ToolArtifact always has type='tool'."""
        artifact = ToolArtifact(
            name="test",
            description="desc",
            input_schema={},
        )
        assert artifact.type == "tool"


# ---------------------------------------------------------------------------
# MCPServerArtifact — valid construction
# ---------------------------------------------------------------------------


class TestMCPServerArtifact:
    """Tests for MCPServerArtifact variant."""

    def test_valid_construction_minimal(self) -> None:
        """MCPServerArtifact with required fields only."""
        artifact = MCPServerArtifact(
            name="my-mcp-server",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        assert artifact.type == "mcp_server"
        assert artifact.name == "my-mcp-server"
        assert artifact.command == "npx"
        assert artifact.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    def test_valid_construction_with_optional_env(self) -> None:
        """MCPServerArtifact with environment variables."""
        artifact = MCPServerArtifact(
            name="custom-server",
            command="python",
            args=["-m", "my_mcp_server"],
            env={"API_KEY": "secret123", "PORT": "3000"},
        )
        assert artifact.env is not None
        assert artifact.env["PORT"] == "3000"

    def test_missing_command_raises(self) -> None:
        """MCPServerArtifact requires 'command' field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerArtifact(
                name="server",
                args=["--help"],
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "command" in field_names

    def test_missing_args_raises(self) -> None:
        """MCPServerArtifact requires 'args' field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerArtifact(
                name="server",
                command="python",
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "args" in field_names

    def test_empty_command_raises(self) -> None:
        """MCPServerArtifact command must be non-empty."""
        with pytest.raises(ValidationError):
            MCPServerArtifact(
                name="server",
                command="",
                args=["-h"],
            )

    def test_json_round_trip(self) -> None:
        """MCPServerArtifact serializes and deserializes without data loss."""
        original = MCPServerArtifact(
            name="fs-server",
            command="node",
            args=["server.js"],
            env={"DEBUG": "1"},
        )
        json_str = original.model_dump_json()
        restored = MCPServerArtifact.model_validate_json(json_str)
        assert restored == original

    def test_type_field_is_mcp_server(self) -> None:
        """MCPServerArtifact always has type='mcp_server'."""
        artifact = MCPServerArtifact(
            name="test",
            command="python",
            args=[],
        )
        assert artifact.type == "mcp_server"


# ---------------------------------------------------------------------------
# SkillArtifact — valid construction
# ---------------------------------------------------------------------------


class TestSkillArtifact:
    """Tests for SkillArtifact variant."""

    def test_valid_construction_minimal(self) -> None:
        """SkillArtifact with required field only."""
        artifact = SkillArtifact(
            name="code-review",
            skill_file_path="/skills/code-review/SKILL.md",
        )
        assert artifact.type == "skill"
        assert artifact.name == "code-review"
        assert artifact.skill_file_path == "/skills/code-review/SKILL.md"

    def test_valid_construction_with_optional(self) -> None:
        """SkillArtifact with description field."""
        artifact = SkillArtifact(
            name="debugging",
            skill_file_path="/skills/debugging/SKILL.md",
            description="A debugging skill for Python code.",
        )
        assert artifact.description == "A debugging skill for Python code."

    def test_missing_skill_file_path_raises(self) -> None:
        """SkillArtifact requires 'skill_file_path' field."""
        with pytest.raises(ValidationError) as exc_info:
            SkillArtifact(
                name="my-skill",
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "skill_file_path" in field_names

    def test_empty_skill_file_path_raises(self) -> None:
        """SkillArtifact skill_file_path must be non-empty."""
        with pytest.raises(ValidationError):
            SkillArtifact(
                name="my-skill",
                skill_file_path="",
            )

    def test_json_round_trip(self) -> None:
        """SkillArtifact serializes and deserializes without data loss."""
        original = SkillArtifact(
            name="refactoring",
            skill_file_path="/skills/refactoring/SKILL.md",
            description="Refactoring patterns and practices.",
        )
        json_str = original.model_dump_json()
        restored = SkillArtifact.model_validate_json(json_str)
        assert restored == original

    def test_type_field_is_skill(self) -> None:
        """SkillArtifact always has type='skill'."""
        artifact = SkillArtifact(
            name="test",
            skill_file_path="/path/to/skill.md",
        )
        assert artifact.type == "skill"


# ---------------------------------------------------------------------------
# Artifact discriminated union
# ---------------------------------------------------------------------------


class TestArtifactDiscriminatedUnion:
    """Tests for the Artifact discriminated union.

    Validates VAL-FOUND-04: discriminated union routing based on type field.
    """

    def test_type_tool_routes_to_tool_artifact(self) -> None:
        """type='tool' creates a ToolArtifact."""
        artifact = Artifact.model_validate(
            {
                "type": "tool",
                "name": "grep",
                "description": "Search tool",
                "input_schema": {"type": "object"},
            }
        )
        assert isinstance(artifact, ToolArtifact)
        assert artifact.type == "tool"

    def test_type_mcp_server_routes_to_mcp_server_artifact(self) -> None:
        """type='mcp_server' creates an MCPServerArtifact."""
        artifact = Artifact.model_validate(
            {
                "type": "mcp_server",
                "name": "fs-server",
                "command": "node",
                "args": ["server.js"],
            }
        )
        assert isinstance(artifact, MCPServerArtifact)
        assert artifact.type == "mcp_server"

    def test_type_skill_routes_to_skill_artifact(self) -> None:
        """type='skill' creates a SkillArtifact."""
        artifact = Artifact.model_validate(
            {
                "type": "skill",
                "name": "review",
                "skill_file_path": "/skills/review/SKILL.md",
            }
        )
        assert isinstance(artifact, SkillArtifact)
        assert artifact.type == "skill"

    def test_invalid_type_raises(self) -> None:
        """An unknown type value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Artifact.model_validate(
                {
                    "type": "unknown_type",
                    "name": "x",
                }
            )
        # Should mention the discriminator field
        error_str = str(exc_info.value).lower()
        assert "type" in error_str

    def test_tool_missing_required_field_in_union(self) -> None:
        """Tool artifact missing 'input_schema' when routed through union."""
        with pytest.raises(ValidationError) as exc_info:
            Artifact.model_validate(
                {
                    "type": "tool",
                    "name": "test",
                    "description": "desc",
                    # missing input_schema
                }
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][-1] if e["loc"] else "" for e in errors}
        assert "input_schema" in field_names

    def test_mcp_missing_required_field_in_union(self) -> None:
        """MCP artifact missing 'args' when routed through union."""
        with pytest.raises(ValidationError) as exc_info:
            Artifact.model_validate(
                {
                    "type": "mcp_server",
                    "name": "server",
                    "command": "python",
                    # missing args
                }
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][-1] if e["loc"] else "" for e in errors}
        assert "args" in field_names

    def test_skill_missing_required_field_in_union(self) -> None:
        """Skill artifact missing 'skill_file_path' when routed through union."""
        with pytest.raises(ValidationError) as exc_info:
            Artifact.model_validate(
                {
                    "type": "skill",
                    "name": "my-skill",
                    # missing skill_file_path
                }
            )
        errors = exc_info.value.errors()
        field_names = {e["loc"][-1] if e["loc"] else "" for e in errors}
        assert "skill_file_path" in field_names

    def test_union_json_round_trip(self) -> None:
        """Artifact discriminated union round-trips through JSON."""
        original = Artifact(
            type="tool",
            name="cat",
            description="Read files",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        json_str = original.model_dump_json()
        restored = Artifact.model_validate_json(json_str)
        assert restored == original

    def test_all_variants_in_json_list_round_trip(self) -> None:
        """A list of mixed artifact types round-trips through JSON."""
        originals: list[Artifact] = [
            ToolArtifact(name="grep", description="Search", input_schema={}),
            MCPServerArtifact(name="server", command="python", args=["-m", "server"]),
            SkillArtifact(name="skill", skill_file_path="/skills/SKILL.md"),
        ]
        json_str = "["
        for i, art in enumerate(originals):
            if i > 0:
                json_str += ","
            json_str += art.model_dump_json()
        json_str += "]"
        restored = [Artifact.model_validate(obj) for obj in __import__("json").loads(json_str)]
        assert len(restored) == 3
        assert isinstance(restored[0], ToolArtifact)
        assert isinstance(restored[1], MCPServerArtifact)
        assert isinstance(restored[2], SkillArtifact)

    def test_error_message_includes_field_name(self) -> None:
        """ValidationError messages include field names for DX (VAL-FOUND-12)."""
        with pytest.raises(ValidationError) as exc_info:
            Artifact.model_validate(
                {
                    "type": "tool",
                    "name": "my-tool",
                    # missing description and input_schema
                }
            )
        error_str = str(exc_info.value)
        # Should reference the missing fields
        assert "description" in error_str.lower() or "input_schema" in error_str.lower()


# ---------------------------------------------------------------------------
# Artifact common fields
# ---------------------------------------------------------------------------


class TestArtifactCommonFields:
    """Tests for fields shared across all artifact variants."""

    def test_all_artifacts_have_name(self) -> None:
        """Every artifact variant has a 'name' field."""
        tool = ToolArtifact(name="t", description="d", input_schema={})
        mcp = MCPServerArtifact(name="m", command="c", args=[])
        skill = SkillArtifact(name="s", skill_file_path="/path")
        assert tool.name == "t"
        assert mcp.name == "m"
        assert skill.name == "s"

    def test_artifact_model_dump_includes_type(self) -> None:
        """model_dump always includes the discriminator 'type' field."""
        tool = ToolArtifact(name="t", description="d", input_schema={})
        data = tool.model_dump()
        assert data["type"] == "tool"

        mcp = MCPServerArtifact(name="m", command="c", args=[])
        data = mcp.model_dump()
        assert data["type"] == "mcp_server"

        skill = SkillArtifact(name="s", skill_file_path="/p")
        data = skill.model_dump()
        assert data["type"] == "skill"
