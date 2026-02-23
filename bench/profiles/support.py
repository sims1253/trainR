"""Support profile module for agent/tool support configurations.

This module defines support modes and profiles that control how skills,
agents, and system prompts are composed for evaluation runs.

Support Modes:
- none: No support, just the raw task
- system_only: System prompt with skill context
- agents_only: Agent-style support without system prompt
- system_plus_agents: Both system prompt and agents
- single_skill: One specific skill
- collection_forced: Force all skills in collection
- collection_selective: Selectively pick skills based on task
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from bench.eval.skill_policy import (
        SelectionResult,
        SkillSelectionPolicy,
    )

logger = logging.getLogger(__name__)


class SupportMode(str, Enum):
    """Modes of support for evaluation runs."""

    NONE = "none"
    """No support, just the raw task."""

    SYSTEM_ONLY = "system_only"
    """System prompt with skill context injected."""

    AGENTS_ONLY = "agents_only"
    """Agent-style support without system prompt modification."""

    SYSTEM_PLUS_AGENTS = "system_plus_agents"
    """Both system prompt and agent-style support."""

    SINGLE_SKILL = "single_skill"
    """One specific skill, referenced by path or name."""

    COLLECTION_FORCED = "collection_forced"
    """Force all skills in collection to be included."""

    COLLECTION_SELECTIVE = "collection_selective"
    """Selectively pick skills based on task characteristics."""


class SkillReference(BaseModel):
    """Reference to a skill file or inline content."""

    path: str | None = Field(default=None, description="Path to skill file")
    name: str | None = Field(default=None, description="Skill name/identifier")
    content: str | None = Field(default=None, description="Inline skill content")
    enabled: bool = Field(default=True, description="Whether this skill is active")
    priority: int = Field(default=0, description="Priority for ordering (higher = earlier)")

    @model_validator(mode="after")
    def validate_reference(self) -> "SkillReference":
        """Ensure at least one source is specified if enabled."""
        if self.enabled and not any([self.path, self.content, self.name]):
            # Allow empty references for placeholder skills
            pass
        return self

    def load_content(self, base_path: Path | None = None) -> str | None:
        """Load skill content from path or return inline content."""
        if not self.enabled:
            return None
        if self.content:
            return self.content
        if self.path:
            skill_path = Path(self.path)
            if not skill_path.is_absolute() and base_path:
                skill_path = base_path / skill_path
            if skill_path.exists():
                return skill_path.read_text()
        return None

    def get_identifier(self) -> str:
        """Get a unique identifier for this skill reference."""
        if self.name:
            return self.name
        if self.path:
            return Path(self.path).stem
        if self.content:
            # Hash first 100 chars of content for identification
            content_preview = self.content[:100]
            return sha256(content_preview.encode()).hexdigest()[:8]
        return "unknown"


class AgentConfig(BaseModel):
    """Configuration for agent-style support."""

    enabled: bool = Field(default=True, description="Whether agent support is enabled")
    agent_type: str = Field(default="default", description="Type of agent (default, router, etc.)")
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    tools: list[str] = Field(default_factory=list, description="Tools available to agent")
    prompt_template: str | None = Field(default=None, description="Custom prompt template")


class SystemPromptConfig(BaseModel):
    """Configuration for system prompt composition."""

    enabled: bool = Field(default=True, description="Whether to modify system prompt")
    base_prompt: str | None = Field(default=None, description="Base system prompt text")
    injection_point: str = Field(
        default="after_context",
        description="Where to inject skill content: before_context, after_context, replace",
    )
    include_skill_metadata: bool = Field(
        default=True,
        description="Include skill metadata in prompt",
    )


class SupportFingerprint(BaseModel):
    """Deterministic fingerprint of support configuration."""

    mode: str = Field(description="Support mode")
    skill_hashes: list[str] = Field(
        default_factory=list,
        description="Hashes of skill content in order",
    )
    skill_order: list[str] = Field(
        default_factory=list,
        description="Skill identifiers in order",
    )
    config_hash: str = Field(
        default="",
        description="Hash of the full configuration",
    )

    @classmethod
    def compute(
        cls,
        mode: SupportMode,
        skills: list[SkillReference],
        agent_config: AgentConfig | None,
        system_config: SystemPromptConfig | None,
        loaded_contents: dict[str, str] | None = None,
    ) -> "SupportFingerprint":
        """Compute fingerprint from configuration.

        Args:
            mode: Support mode
            skills: List of skill references
            agent_config: Agent configuration
            system_config: System prompt configuration
            loaded_contents: Optional dict of skill identifier -> content

        Returns:
            SupportFingerprint instance
        """
        # Compute skill hashes in order (sorted by priority, then by identifier)
        sorted_skills = sorted(
            skills,
            key=lambda s: (-s.priority, s.get_identifier()),
        )

        skill_hashes: list[str] = []
        skill_order: list[str] = []
        hash_components: list[str] = [mode.value]

        for skill in sorted_skills:
            if not skill.enabled:
                continue
            identifier = skill.get_identifier()
            skill_order.append(identifier)

            # Hash content if available
            content = None
            if loaded_contents and identifier in loaded_contents:
                content = loaded_contents[identifier]
            elif skill.content:
                content = skill.content

            if content:
                content_hash = sha256(content.encode()).hexdigest()[:16]
                skill_hashes.append(content_hash)
                hash_components.append(f"{identifier}:{content_hash}")
            else:
                hash_components.append(f"{identifier}:none")

        # Include agent config in hash
        if agent_config and agent_config.enabled:
            agent_str = f"agent:{agent_config.agent_type}:{agent_config.max_iterations}"
            hash_components.append(agent_str)

        # Include system config in hash
        if system_config and system_config.enabled:
            system_str = f"system:{system_config.injection_point}"
            if system_config.base_prompt:
                prompt_hash = sha256(system_config.base_prompt.encode()).hexdigest()[:8]
                system_str += f":{prompt_hash}"
            hash_components.append(system_str)

        # Compute overall config hash
        combined = "|".join(hash_components)
        config_hash = sha256(combined.encode()).hexdigest()[:16]

        return cls(
            mode=mode.value,
            skill_hashes=skill_hashes,
            skill_order=skill_order,
            config_hash=config_hash,
        )

    def to_compact_string(self) -> str:
        """Get a compact string representation for logging."""
        return f"{self.mode}:{self.config_hash[:8]}"


class SelectionMetadata(BaseModel):
    """Metadata about skill selection for COLLECTION_SELECTIVE mode."""

    selected_skills: list[str] = Field(
        default_factory=list,
        description="List of selected skill IDs",
    )
    selection_policy: str = Field(
        default="",
        description="Name of the selection policy used",
    )
    selection_rationale: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rationale for each skill selection",
    )
    config_hash: str = Field(
        default="",
        description="Hash of selection configuration",
    )
    seed: int | None = Field(
        default=None,
        description="Seed used for deterministic selection",
    )

    @classmethod
    def from_selection_result(cls, result: SelectionResult) -> "SelectionMetadata":
        """Create SelectionMetadata from a SelectionResult.

        Args:
            result: SelectionResult from skill policy

        Returns:
            SelectionMetadata instance
        """
        return cls(
            selected_skills=result.get_ordered_skills(),
            selection_policy=result.selection_policy,
            selection_rationale=[r.to_dict() for r in result.selection_rationale],
            config_hash=result.config_hash,
            seed=result.seed,
        )

    def is_reproducible(self) -> bool:
        """Check if selection is reproducible (has seed)."""
        return self.seed is not None


class ComposedSupportArtifact(BaseModel):
    """Persisted artifact of composed support configuration."""

    fingerprint: SupportFingerprint = Field(description="Support fingerprint")
    system_prompt: str | None = Field(default=None, description="Composed system prompt")
    skills_content: dict[str, str] = Field(
        default_factory=dict,
        description="Skill identifier -> content",
    )
    agent_instructions: str | None = Field(
        default=None,
        description="Agent-specific instructions",
    )
    composition_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about composition",
    )
    selection_metadata: SelectionMetadata | None = Field(
        default=None,
        description="Selection metadata for COLLECTION_SELECTIVE mode",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When artifact was created",
    )


class SupportProfile(BaseModel):
    """
    Profile defining how support (skills, agents, system prompts) is composed.

    Support profiles make support structures first-class and reproducible,
    enabling deterministic fingerprinting of evaluation runs.
    """

    # Profile identification
    profile_id: str = Field(description="Unique identifier for this profile")
    name: str = Field(default="", description="Human-readable name")
    description: str = Field(default="", description="Profile description")

    # Mode and configuration
    mode: SupportMode = Field(description="Support mode")
    skills: list[SkillReference] = Field(
        default_factory=list,
        description="Skill references",
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent configuration",
    )
    system_prompt: SystemPromptConfig = Field(
        default_factory=SystemPromptConfig,
        description="System prompt configuration",
    )

    # Collection mode settings
    collection_path: str | None = Field(
        default=None,
        description="Path to skill collection (for collection modes)",
    )
    selection_criteria: dict[str, Any] = Field(
        default_factory=dict,
        description="Criteria for selective skill selection",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: Any) -> SupportMode:
        """Convert string to SupportMode if needed."""
        if isinstance(v, str):
            return SupportMode(v)
        return v

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "SupportProfile":
        """Validate that required fields are present for each mode."""
        if self.mode == SupportMode.SINGLE_SKILL:
            if not self.skills:
                raise ValueError("single_skill mode requires at least one skill reference")
        elif self.mode in (SupportMode.COLLECTION_FORCED, SupportMode.COLLECTION_SELECTIVE):
            if not self.collection_path:
                raise ValueError(f"{self.mode.value} mode requires collection_path")
        return self

    @cached_property
    def fingerprint(self) -> SupportFingerprint:
        """Compute the fingerprint for this profile."""
        # Load content for fingerprinting
        loaded_contents: dict[str, str] = {}
        for skill in self.skills:
            content = skill.load_content()
            if content:
                loaded_contents[skill.get_identifier()] = content

        return SupportFingerprint.compute(
            mode=self.mode,
            skills=self.skills,
            agent_config=self.agent,
            system_config=self.system_prompt,
            loaded_contents=loaded_contents,
        )

    def load_skill_contents(self, base_path: Path | None = None) -> dict[str, str]:
        """Load all skill contents.

        Args:
            base_path: Base path for resolving relative skill paths

        Returns:
            Dict of skill identifier -> content
        """
        contents: dict[str, str] = {}
        for skill in self.skills:
            if not skill.enabled:
                continue
            content = skill.load_content(base_path)
            if content:
                contents[skill.get_identifier()] = content
        return contents

    def get_ordered_skills(self) -> list[SkillReference]:
        """Get skills ordered by priority (descending)."""
        return sorted(self.skills, key=lambda s: (-s.priority, s.get_identifier()))

    def get_selection_policy(self, policies_path: Path | None = None) -> "SkillSelectionPolicy":
        """Get the skill selection policy for this profile.

        Args:
            policies_path: Optional path to policies config file

        Returns:
            Configured SkillSelectionPolicy instance
        """
        # Lazy import to avoid circular dependency
        from bench.eval.skill_policy import (
            HeuristicPolicy,
            load_policy_from_config,
            load_policy_from_yaml,
        )

        # Check if policy is specified in selection_criteria
        policy_name = self.selection_criteria.get("policy", "heuristic_balanced")

        # Try to load from config file
        if policies_path is None:
            policies_path = Path("configs/profiles/support/selective_policies.yaml")

        try:
            return load_policy_from_yaml(policies_path, policy_name)
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Could not load policy '{policy_name}': {e}. Using default.")
            # Fall back to inline config or default
            policy_config = self.selection_criteria.get("policy_config", {})
            if policy_config:
                return load_policy_from_config(policy_config)
            return HeuristicPolicy()

    def select_skills_for_task(
        self,
        task_context: dict[str, Any],
        base_path: Path | None = None,
        policies_path: Path | None = None,
    ) -> tuple[list[SkillReference], "SelectionMetadata"]:
        """Select skills based on task context for COLLECTION_SELECTIVE mode.

        Args:
            task_context: Task information including instruction, context, etc.
            base_path: Base path for resolving skill paths
            policies_path: Optional path to policies config file

        Returns:
            Tuple of (selected skill references, selection metadata)
        """
        # Lazy import to avoid circular dependency
        from bench.eval.skill_policy import SelectionResult, discover_skills

        # Discover available skills from collection path
        skills_dir = Path(self.collection_path) if self.collection_path else Path("skills")
        if base_path and not skills_dir.is_absolute():
            skills_dir = base_path / skills_dir

        available_skills = discover_skills(skills_dir)

        # Get selection policy
        policy = self.get_selection_policy(policies_path)

        # Perform selection
        result: SelectionResult = policy.select(task_context, available_skills)

        # Map selected skill IDs to SkillReferences
        skill_id_to_ref: dict[str, SkillReference] = {s.get_identifier(): s for s in self.skills}

        selected_refs: list[SkillReference] = []
        for skill_id in result.get_ordered_skills():
            if skill_id in skill_id_to_ref:
                selected_refs.append(skill_id_to_ref[skill_id])
            else:
                # Create reference from discovered skill
                for meta in available_skills:
                    if meta.skill_id == skill_id:
                        selected_refs.append(
                            SkillReference(
                                name=meta.name,
                                path=meta.path,
                                priority=meta.priority,
                                enabled=True,
                            )
                        )
                        break

        metadata = SelectionMetadata.from_selection_result(result)

        logger.info(
            f"Selected {len(selected_refs)} skills using '{result.selection_policy}' policy: "
            f"{[s.get_identifier() for s in selected_refs]}"
        )

        return selected_refs, metadata

    def compose(
        self,
        base_path: Path | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> ComposedSupportArtifact:
        """
        Compose the support artifact from this profile.

        Args:
            base_path: Base path for resolving relative paths
            task_context: Optional task context for COLLECTION_SELECTIVE mode

        Returns:
            ComposedSupportArtifact with all composed elements
        """
        selection_metadata: SelectionMetadata | None = None
        ordered_skills = self.get_ordered_skills()

        # For COLLECTION_SELECTIVE mode, select skills based on task context
        if self.mode == SupportMode.COLLECTION_SELECTIVE:
            if task_context is None:
                # Default task context if none provided
                task_context = self.selection_criteria.get("default_task_context", {})
                logger.warning(
                    "COLLECTION_SELECTIVE mode requires task_context. "
                    "Using default from selection_criteria."
                )

            selected_skills, selection_metadata = self.select_skills_for_task(
                task_context=task_context,
                base_path=base_path,
            )
            ordered_skills = selected_skills
            self.skills = selected_skills  # Update for fingerprint computation

        skill_contents: dict[str, str] = {}
        for skill in ordered_skills:
            if not skill.enabled:
                continue
            content = skill.load_content(base_path)
            if content:
                skill_contents[skill.get_identifier()] = content

        # Compose system prompt based on mode
        system_prompt: str | None = None
        if self.mode in (
            SupportMode.SYSTEM_ONLY,
            SupportMode.SYSTEM_PLUS_AGENTS,
            SupportMode.SINGLE_SKILL,
            SupportMode.COLLECTION_FORCED,
            SupportMode.COLLECTION_SELECTIVE,
        ):
            system_prompt = self._compose_system_prompt(skill_contents, ordered_skills)

        # Compose agent instructions if applicable
        agent_instructions: str | None = None
        if self.mode in (
            SupportMode.AGENTS_ONLY,
            SupportMode.SYSTEM_PLUS_AGENTS,
        ):
            agent_instructions = self._compose_agent_instructions(skill_contents, ordered_skills)

        # Compute fingerprint with loaded contents
        fingerprint = SupportFingerprint.compute(
            mode=self.mode,
            skills=ordered_skills,
            agent_config=self.agent,
            system_config=self.system_prompt,
            loaded_contents=skill_contents,
        )

        composition_metadata = {
            "mode": self.mode.value,
            "skill_count": len(ordered_skills),
            "agent_enabled": self.agent.enabled,
            "system_enabled": self.system_prompt.enabled,
        }

        # Add selection info to metadata for COLLECTION_SELECTIVE
        if selection_metadata:
            composition_metadata["selection_reproducible"] = selection_metadata.is_reproducible()

        return ComposedSupportArtifact(
            fingerprint=fingerprint,
            system_prompt=system_prompt,
            skills_content=skill_contents,
            agent_instructions=agent_instructions,
            composition_metadata=composition_metadata,
            selection_metadata=selection_metadata,
        )

    def compose_with_task(
        self,
        task_context: dict[str, Any],
        base_path: Path | None = None,
    ) -> ComposedSupportArtifact:
        """Compose support artifact with explicit task context.

        This is the preferred method for COLLECTION_SELECTIVE mode.

        Args:
            task_context: Task information including:
                - instruction: Task instruction text
                - context: Additional context
                - source_package: Source package name
                - task_id: Optional task identifier
                - difficulty: Optional difficulty level
            base_path: Base path for resolving relative paths

        Returns:
            ComposedSupportArtifact with selected skills and selection metadata
        """
        return self.compose(base_path=base_path, task_context=task_context)

    def _compose_system_prompt(
        self,
        skill_contents: dict[str, str],
        ordered_skills: list[SkillReference],
    ) -> str:
        """Compose the system prompt with skill content."""
        if not self.system_prompt.enabled:
            return ""

        parts: list[str] = []

        # Add base prompt if specified
        if self.system_prompt.base_prompt:
            parts.append(self.system_prompt.base_prompt)

        # Add skill content based on injection point
        skill_sections: list[str] = []
        for skill in ordered_skills:
            if not skill.enabled:
                continue
            identifier = skill.get_identifier()
            content = skill_contents.get(identifier)
            if content:
                if self.system_prompt.include_skill_metadata:
                    skill_sections.append(f"## {identifier}\n\n{content}")
                else:
                    skill_sections.append(content)

        skill_text = "\n\n---\n\n".join(skill_sections) if skill_sections else ""

        if self.system_prompt.injection_point == "replace":
            return skill_text
        elif self.system_prompt.injection_point == "before_context":
            if skill_text:
                parts.insert(0, skill_text)
        else:  # after_context (default)
            if skill_text:
                parts.append(skill_text)

        return "\n\n".join(parts) if parts else ""

    def _compose_agent_instructions(
        self,
        skill_contents: dict[str, str],
        ordered_skills: list[SkillReference],
    ) -> str:
        """Compose agent-specific instructions."""
        if not self.agent.enabled:
            return ""

        parts: list[str] = [f"Agent Type: {self.agent.agent_type}"]

        if self.agent.max_iterations:
            parts.append(f"Maximum Iterations: {self.agent.max_iterations}")

        if self.agent.tools:
            parts.append(f"Available Tools: {', '.join(self.agent.tools)}")

        # Include skill content as agent instructions
        skill_instructions: list[str] = []
        for skill in ordered_skills:
            if not skill.enabled:
                continue
            identifier = skill.get_identifier()
            content = skill_contents.get(identifier)
            if content:
                skill_instructions.append(f"[{identifier}]\n{content}")

        if skill_instructions:
            parts.append("\nSkill Instructions:\n")
            parts.append("\n\n".join(skill_instructions))

        return "\n".join(parts)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SupportProfile":
        """Load a support profile from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Support profile not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupportProfile":
        """Create a support profile from dictionary."""
        # Parse mode
        mode_data = data.get("mode", "none")
        if isinstance(mode_data, str):
            mode = SupportMode(mode_data)
        else:
            mode = mode_data

        # Parse skills
        skills_data = data.get("skills", [])
        skills = [SkillReference(**s) if isinstance(s, dict) else s for s in skills_data]

        # Parse agent config
        agent_data = data.get("agent", {})
        agent = AgentConfig(**agent_data) if isinstance(agent_data, dict) else agent_data

        # Parse system prompt config
        system_data = data.get("system_prompt", {})
        system_prompt = (
            SystemPromptConfig(**system_data) if isinstance(system_data, dict) else system_data
        )

        return cls(
            profile_id=data.get("profile_id", "default"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mode=mode,
            skills=skills,
            agent=agent,
            system_prompt=system_prompt,
            collection_path=data.get("collection_path"),
            selection_criteria=data.get("selection_criteria", {}),
            metadata=data.get("metadata", {}),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save the support profile to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json", exclude_none=True)
        # Remove cached property from serialization
        data.pop("fingerprint", None)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default profiles for common use cases
DEFAULT_PROFILES: dict[str, SupportProfile] = {
    "none": SupportProfile(
        profile_id="none",
        name="No Support",
        description="No support provided, just the raw task",
        mode=SupportMode.NONE,
    ),
    "system_only": SupportProfile(
        profile_id="system_only",
        name="System Prompt Only",
        description="System prompt with skill context injected",
        mode=SupportMode.SYSTEM_ONLY,
        system_prompt=SystemPromptConfig(
            enabled=True,
            injection_point="after_context",
        ),
    ),
    "agents_only": SupportProfile(
        profile_id="agents_only",
        name="Agents Only",
        description="Agent-style support without system prompt modification",
        mode=SupportMode.AGENTS_ONLY,
        agent=AgentConfig(
            enabled=True,
            agent_type="default",
            max_iterations=10,
        ),
        system_prompt=SystemPromptConfig(enabled=False),
    ),
}


def load_support_profile(name_or_path: str, profiles_dir: Path | None = None) -> SupportProfile:
    """
    Load a support profile by name or path.

    Args:
        name_or_path: Profile name (for built-ins) or path to YAML file
        profiles_dir: Directory to search for profile files

    Returns:
        SupportProfile instance

    Raises:
        FileNotFoundError: If profile cannot be found
    """
    # Check built-in profiles first
    if name_or_path in DEFAULT_PROFILES:
        return DEFAULT_PROFILES[name_or_path]

    # Check if it's a path
    path = Path(name_or_path)
    if path.exists():
        return SupportProfile.from_yaml(path)

    # Search in profiles directory
    if profiles_dir:
        profile_path = profiles_dir / f"{name_or_path}.yaml"
        if profile_path.exists():
            return SupportProfile.from_yaml(profile_path)

    # Default search location
    default_dir = Path("configs/profiles/support")
    profile_path = default_dir / f"{name_or_path}.yaml"
    if profile_path.exists():
        return SupportProfile.from_yaml(profile_path)

    raise FileNotFoundError(
        f"Support profile not found: {name_or_path}. "
        f"Searched: built-ins, {name_or_path}, {profile_path}"
    )
