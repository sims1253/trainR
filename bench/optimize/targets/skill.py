"""Skill text optimization target.

This target optimizes the text content of skills - the markdown/text
that guides agents in completing tasks. The optimization can explore:
- Section reordering
- Content phrasing
- Example inclusion
- Instruction clarity

Target Isolation:
----------------
SkillTarget ONLY modifies the skill content (inline content or path).
It does NOT modify:
- System prompt configuration
- Tool availability
- Agent configuration
- Any other profile dimension
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from bench.optimize.targets.base import (
    OptimizableTarget,
    ParamSpace,
    ParamSpec,
    ParamType,
    TargetFingerprint,
)


@dataclass
class SkillCandidate:
    """Candidate value for skill optimization.

    A skill candidate represents a potential skill configuration
    that can be evaluated during optimization.
    """

    content: str
    """The skill text/markdown content."""

    name: str | None = None
    """Optional name for the skill."""

    sections: dict[str, str] = field(default_factory=dict)
    """Named sections within the skill (if structured)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., version, author)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "name": self.name,
            "sections": self.sections,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCandidate":
        """Deserialize from dictionary."""
        return cls(
            content=data.get("content", ""),
            name=data.get("name"),
            sections=data.get("sections", {}),
            metadata=data.get("metadata", {}),
        )

    def get_full_content(self) -> str:
        """Get the full skill content.

        If sections are defined, assembles them in order.
        Otherwise returns the main content.
        """
        if self.sections:
            # Assemble sections in a defined order
            section_order = [
                "overview",
                "instructions",
                "examples",
                "patterns",
                "anti_patterns",
                "notes",
            ]
            parts = []
            for section_name in section_order:
                if section_name in self.sections:
                    parts.append(f"## {section_name.title()}\n\n{self.sections[section_name]}")
            # Add any remaining sections not in the standard order
            for section_name, content in self.sections.items():
                if section_name not in section_order:
                    parts.append(f"## {section_name.title()}\n\n{content}")
            return "\n\n".join(parts) if parts else self.content
        return self.content

    def compute_hash(self) -> str:
        """Compute hash of content for change detection."""
        return sha256(self.get_full_content().encode()).hexdigest()[:16]


class SkillTarget(OptimizableTarget[SkillCandidate]):
    """
    Target for optimizing skill text content.

    This target allows optimization of the markdown/text content
    that guides agents in completing tasks.

    Target Isolation:
    ----------------
    When applying a candidate, this target ONLY modifies:
    - skill.content (inline content)

    It preserves ALL other configuration:
    - skill.path (cleared when using inline content)
    - skill.no_skill (preserved)
    - All other experiment configuration
    """

    def __init__(
        self,
        skill_name: str | None = None,
        min_content_length: int = 50,
        max_content_length: int = 50000,
        allowed_sections: list[str] | None = None,
    ) -> None:
        """Initialize the skill target.

        Args:
            skill_name: Optional name for the skill being optimized
            min_content_length: Minimum valid content length
            max_content_length: Maximum valid content length
            allowed_sections: Sections that can be optimized (None = any)
        """
        self.skill_name = skill_name
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.allowed_sections = allowed_sections

        # Compute config hash for fingerprint
        config_str = f"{skill_name}:{min_content_length}:{max_content_length}:{allowed_sections}"
        self._config_hash = sha256(config_str.encode()).hexdigest()[:16]

    @property
    def target_type(self) -> str:
        """Get target type identifier."""
        return "skill"

    def fingerprint(self) -> TargetFingerprint:
        """Get unique fingerprint for this target."""
        return TargetFingerprint(
            target_type=self.target_type,
            config_hash=self._config_hash,
            version="1.0",
        )

    def serialize_candidate(self, candidate: SkillCandidate) -> dict[str, Any]:
        """Serialize a skill candidate for storage.

        Args:
            candidate: Skill candidate to serialize

        Returns:
            Dictionary representation
        """
        return candidate.to_dict()

    def deserialize_candidate(self, data: dict[str, Any]) -> SkillCandidate:
        """Deserialize a skill candidate from storage.

        Args:
            data: Dictionary from serialize_candidate

        Returns:
            SkillCandidate instance

        Raises:
            ValueError: If required fields missing
        """
        if "content" not in data and not data.get("sections"):
            raise ValueError("Candidate must have 'content' or 'sections'")

        return SkillCandidate.from_dict(data)

    def apply_candidate_to_config(
        self,
        config: "ExperimentConfig",
        candidate: SkillCandidate,
    ) -> "ExperimentConfig":
        """Apply a skill candidate to an experiment configuration.

        CRITICAL: This method ONLY modifies the skill content.
        All other configuration remains unchanged.

        Args:
            config: Base experiment configuration
            candidate: Skill candidate to apply

        Returns:
            New ExperimentConfig with skill content applied
        """
        from bench.experiments.config import ExperimentConfig, SkillConfig

        # Create a deep copy of the config
        config_dict = config.model_dump(mode="json")

        # ONLY modify skill.content - preserve everything else
        config_dict["skill"] = {
            "path": None,  # Clear path when using inline content
            "no_skill": config.skill.no_skill,  # Preserve
            "content": candidate.get_full_content(),  # Apply candidate
        }

        return ExperimentConfig.from_dict(config_dict)

    def get_current_value(self, config: "ExperimentConfig") -> SkillCandidate:
        """Extract current skill content from config.

        Args:
            config: Experiment configuration

        Returns:
            SkillCandidate with current content
        """
        content = config.get_skill_content()
        if content is None:
            content = ""

        return SkillCandidate(
            content=content,
            name=config.skill.get_name() if config.skill.get_name() != "none" else None,
        )

    def get_param_space(self) -> ParamSpace:
        """Define the searchable parameter space for skill optimization.

        The parameter space includes:
        - content: The main skill text (TEXT type)
        - sections: Structured sections (TEXT type, optional)
        """
        space = ParamSpace()

        # Main content parameter
        space.add_param(
            ParamSpec(
                name="content",
                param_type=ParamType.TEXT,
                description="Main skill text/markdown content",
                required=True,
                min_value=self.min_content_length,
                max_value=self.max_content_length,
            )
        )

        # Standard sections (all optional)
        section_specs = [
            ("overview", "Brief overview of what this skill teaches"),
            ("instructions", "Step-by-step instructions for using the skill"),
            ("examples", "Concrete examples demonstrating the skill"),
            ("patterns", "Recommended patterns and best practices"),
            ("anti_patterns", "Common mistakes to avoid"),
            ("notes", "Additional notes and tips"),
        ]

        for section_name, description in section_specs:
            if self.allowed_sections is None or section_name in self.allowed_sections:
                space.add_param(
                    ParamSpec(
                        name=f"section_{section_name}",
                        param_type=ParamType.TEXT,
                        description=description,
                        required=False,
                    )
                )

        return space

    def validate_candidate(self, candidate: SkillCandidate) -> tuple[bool, list[str]]:
        """Validate a skill candidate.

        Args:
            candidate: Candidate to validate

        Returns:
            Tuple of (is_valid, error messages)
        """
        errors: list[str] = []

        # Check content length
        content = candidate.get_full_content()
        if len(content) < self.min_content_length:
            errors.append(f"Content too short: {len(content)} < {self.min_content_length}")
        if len(content) > self.max_content_length:
            errors.append(f"Content too long: {len(content)} > {self.max_content_length}")

        # Check content is not empty
        if not content.strip():
            errors.append("Content cannot be empty")

        # Check sections if restricted
        if self.allowed_sections is not None:
            for section_name in candidate.sections:
                if section_name not in self.allowed_sections:
                    errors.append(f"Section '{section_name}' not in allowed sections")

        return len(errors) == 0, errors


def load_skill_from_file(path: str | Path) -> SkillCandidate:
    """Load a skill candidate from a file.

    Args:
        path: Path to skill file (markdown)

    Returns:
        SkillCandidate with file content
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    content = path.read_text()
    return SkillCandidate(
        content=content,
        name=path.stem,
    )


def save_skill_to_file(candidate: SkillCandidate, path: str | Path) -> None:
    """Save a skill candidate to a file.

    Args:
        candidate: Skill candidate to save
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate.get_full_content())
