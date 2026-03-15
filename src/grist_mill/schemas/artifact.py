"""Artifact discriminated union schema.

Defines the polymorphic Artifact type using Pydantic v2 discriminated unions.
Each variant represents a different kind of pluggable artifact:

- ToolArtifact: A CLI tool with name, description, and input/output schemas
- MCPServerArtifact: An MCP server process with command and args
- SkillArtifact: A skill defined in a SKILL.md file

The discriminator field is ``type`` with values ``tool``, ``mcp_server``, ``skill``.

Validates VAL-FOUND-04 and VAL-FOUND-12.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

# ---------------------------------------------------------------------------
# ToolArtifact
# ---------------------------------------------------------------------------


class ToolArtifact(BaseModel):
    """A CLI tool available to the agent.

    Represents a command-line tool with a structured input schema
    that the agent can discover and invoke.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    type: Literal["tool"] = "tool"
    name: str = Field(
        ...,
        min_length=1,
        description="Unique name of the tool.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of what the tool does.",
    )
    input_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema describing the tool's input parameters.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema describing the tool's output.",
    )
    command: str | None = Field(
        default=None,
        description="Optional command to execute for this tool.",
    )


# ---------------------------------------------------------------------------
# MCPServerArtifact
# ---------------------------------------------------------------------------


class MCPServerArtifact(BaseModel):
    """An MCP (Model Context Protocol) server process.

    Represents a subprocess that provides tools/resources via the MCP protocol.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    type: Literal["mcp_server"] = "mcp_server"
    name: str = Field(
        ...,
        min_length=1,
        description="Unique name of the MCP server.",
    )
    command: str = Field(
        ...,
        min_length=1,
        description="Command to start the MCP server (e.g., 'npx', 'python').",
    )
    args: list[str] = Field(
        ...,
        min_length=0,
        description="Arguments to pass to the command.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the server process.",
    )


# ---------------------------------------------------------------------------
# SkillArtifact
# ---------------------------------------------------------------------------


class SkillArtifact(BaseModel):
    """A skill defined in a SKILL.md file.

    Represents a skill document that gets injected into the agent's context
    as system prompt or additional instructions.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    type: Literal["skill"] = "skill"
    name: str = Field(
        ...,
        min_length=1,
        description="Unique name of the skill.",
    )
    skill_file_path: str = Field(
        ...,
        min_length=1,
        description="Path to the SKILL.md file defining this skill.",
    )
    description: str | None = Field(
        default=None,
        description="Optional human-readable description of the skill.",
    )


# ---------------------------------------------------------------------------
# Discriminated Union
# ---------------------------------------------------------------------------


_ArtifactUnion = Annotated[
    ToolArtifact | MCPServerArtifact | SkillArtifact,
    Field(discriminator="type"),
]


class _ArtifactWrapper(RootModel[_ArtifactUnion]):
    """Internal root model that enables discriminated union validation."""

    pass


# Artifact acts as the discriminated union type with BaseModel-like interface
class Artifact:
    """Discriminated union of ToolArtifact, MCPServerArtifact, and SkillArtifact.

    Use ``Artifact.model_validate(data)`` to parse a dict with a ``type`` discriminator,
    or instantiate directly with ``Artifact(type="tool", ...)``.
    """

    __slots__ = ()

    @staticmethod
    def model_validate(
        data: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolArtifact | MCPServerArtifact | SkillArtifact:
        """Validate a dict or object into the correct artifact variant.

        Uses the ``type`` field as the discriminator.
        """
        return _ArtifactWrapper.model_validate(
            data, strict=strict, from_attributes=from_attributes, context=context
        ).root

    @staticmethod
    def model_validate_json(
        json_data: str | bytes,
        *,
        strict: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolArtifact | MCPServerArtifact | SkillArtifact:
        """Validate a JSON string into the correct artifact variant."""
        return _ArtifactWrapper.model_validate_json(json_data, strict=strict, context=context).root

    def __new__(
        cls,
        *,
        type: str,
        **kwargs: Any,
    ) -> ToolArtifact | MCPServerArtifact | SkillArtifact:
        """Construct an Artifact variant based on the type discriminator.

        This is syntactic sugar that dispatches to the correct variant class.
        """
        if type == "tool":
            return ToolArtifact(type="tool", **kwargs)
        if type == "mcp_server":
            return MCPServerArtifact(type="mcp_server", **kwargs)
        if type == "skill":
            return SkillArtifact(type="skill", **kwargs)
        from pydantic import ValidationError

        raise ValidationError.from_exception_data(
            title="Artifact",
            line_errors=[
                {
                    "type": "literal_error",
                    "loc": ("type",),
                    "input": type,
                    "ctx": {
                        "expected": "'tool', 'mcp_server', or 'skill'",
                    },
                }
            ],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Artifact",
    "MCPServerArtifact",
    "SkillArtifact",
    "ToolArtifact",
]
