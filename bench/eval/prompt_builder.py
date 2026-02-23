"""Prompt builder for composing prompts from support profiles.

This module provides the PromptBuilder class that takes SupportProfiles
and composes the final prompts for evaluation runs.

Usage:
    from bench.eval.prompt_builder import PromptBuilder
    from bench.profiles import SupportProfile, SupportMode

    profile = SupportProfile(mode=SupportMode.SYSTEM_ONLY, ...)
    builder = PromptBuilder(profile)
    system_prompt = builder.build_system_prompt(task_context)
"""

from pathlib import Path
from typing import Any

from bench.profiles.support import (
    ComposedSupportArtifact,
    SkillReference,
    SupportMode,
    SupportProfile,
    SystemPromptConfig,
)


class PromptBuilder:
    """
    Builds prompts from support profiles for evaluation runs.

    The PromptBuilder takes a SupportProfile and composes the appropriate
    prompts based on the support mode and configuration.

    Attributes:
        profile: The support profile to build from
        base_path: Base path for resolving relative paths
        _composed_cache: Cached composed artifact
    """

    def __init__(
        self,
        profile: SupportProfile,
        base_path: Path | None = None,
    ):
        """
        Initialize the prompt builder.

        Args:
            profile: Support profile to build from
            base_path: Base path for resolving relative skill paths
        """
        self.profile = profile
        self.base_path = base_path or Path.cwd()
        self._composed_cache: ComposedSupportArtifact | None = None

    @property
    def composed(self) -> ComposedSupportArtifact:
        """Get the composed support artifact (cached)."""
        if self._composed_cache is None:
            self._composed_cache = self.profile.compose(self.base_path)
        return self._composed_cache

    @property
    def fingerprint(self) -> str:
        """Get the fingerprint string for this builder's configuration."""
        return self.composed.fingerprint.to_compact_string()

    def build_system_prompt(
        self,
        task_context: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the system prompt for the given task.

        Args:
            task_context: Context about the task (e.g., task description)
            additional_context: Additional context to include

        Returns:
            Composed system prompt string
        """
        mode = self.profile.mode

        # Handle modes that don't produce system prompts
        if mode == SupportMode.NONE:
            return self._build_minimal_prompt(task_context)

        if mode == SupportMode.AGENTS_ONLY:
            # Agents only mode - no system prompt modification
            return task_context or ""

        # Get base composed prompt
        base_prompt = self.composed.system_prompt or ""

        # Combine with task context based on injection point
        injection = self.profile.system_prompt.injection_point

        parts: list[str] = []

        if injection == "before_context":
            if base_prompt:
                parts.append(base_prompt)
            if task_context:
                parts.append(task_context)
        elif injection == "replace":
            # Use skill content directly, ignore task context
            parts.append(base_prompt)
        else:  # after_context (default)
            if task_context:
                parts.append(task_context)
            if base_prompt:
                parts.append(base_prompt)

        # Add additional context
        if additional_context:
            context_parts = self._format_additional_context(additional_context)
            if context_parts:
                parts.append(context_parts)

        return "\n\n".join(filter(None, parts))

    def build_agent_prompt(
        self,
        task_context: str | None = None,
    ) -> str:
        """
        Build agent-specific instructions.

        Args:
            task_context: Context about the task

        Returns:
            Composed agent instructions string
        """
        mode = self.profile.mode

        # Check if agent mode is applicable
        if mode not in (
            SupportMode.AGENTS_ONLY,
            SupportMode.SYSTEM_PLUS_AGENTS,
        ):
            return ""

        agent_instructions = self.composed.agent_instructions or ""

        if task_context:
            return f"{task_context}\n\n{agent_instructions}"

        return agent_instructions

    def build_full_prompt(
        self,
        task_context: str,
        include_agent: bool = True,
        additional_context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Build the complete prompt configuration.

        Args:
            task_context: Context about the task
            include_agent: Whether to include agent instructions
            additional_context: Additional context to include

        Returns:
            Dict with 'system' and optionally 'agent' prompts
        """
        result: dict[str, str] = {
            "system": self.build_system_prompt(task_context, additional_context),
        }

        if include_agent and self.profile.agent.enabled:
            agent_prompt = self.build_agent_prompt()
            if agent_prompt:
                result["agent"] = agent_prompt

        return result

    def get_skill_context(self) -> dict[str, str]:
        """
        Get the loaded skill content.

        Returns:
            Dict of skill identifier -> content
        """
        return self.composed.skills_content

    def get_ordered_skill_names(self) -> list[str]:
        """
        Get skill names in priority order.

        Returns:
            List of skill identifiers in order
        """
        return self.composed.fingerprint.skill_order

    def _build_minimal_prompt(self, task_context: str | None) -> str:
        """Build a minimal prompt for 'none' mode."""
        if task_context:
            return task_context
        return "Complete the task."

    def _format_additional_context(self, context: dict[str, Any]) -> str:
        """Format additional context as a string."""
        if not context:
            return ""

        lines: list[str] = ["## Additional Context"]
        for key, value in context.items():
            if isinstance(value, str):
                lines.append(f"- **{key}**: {value}")
            elif isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
                lines.append(f"- **{key}**: {value_str}")
            else:
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def to_manifest_entry(self) -> dict[str, Any]:
        """
        Create a manifest entry for this prompt configuration.

        Returns:
            Dict suitable for inclusion in run manifest
        """
        return {
            "support_mode": self.profile.mode.value,
            "support_fingerprint": self.fingerprint,
            "support_profile_id": self.profile.profile_id,
            "skill_order": self.get_ordered_skill_names(),
            "agent_enabled": self.profile.agent.enabled,
            "system_prompt_modified": self.profile.system_prompt.enabled,
        }


def build_prompt_from_profile(
    profile: SupportProfile,
    task_context: str,
    base_path: Path | None = None,
    additional_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Convenience function to build prompts from a support profile.

    Args:
        profile: Support profile to use
        task_context: Task context/description
        base_path: Base path for resolving relative paths
        additional_context: Additional context to include

    Returns:
        Dict with 'system' and optionally 'agent' prompts
    """
    builder = PromptBuilder(profile, base_path)
    return builder.build_full_prompt(task_context, additional_context=additional_context)


def compose_system_prompt(
    mode: SupportMode,
    skills: list[SkillReference],
    task_context: str,
    system_config: SystemPromptConfig | None = None,
    base_path: Path | None = None,
) -> str:
    """
    Low-level function to compose a system prompt.

    Args:
        mode: Support mode
        skills: List of skill references
        task_context: Task context/description
        system_config: System prompt configuration
        base_path: Base path for resolving paths

    Returns:
        Composed system prompt string
    """
    profile = SupportProfile(
        profile_id="temp",
        mode=mode,
        skills=skills,
        system_prompt=system_config or SystemPromptConfig(),
    )
    builder = PromptBuilder(profile, base_path)
    return builder.build_system_prompt(task_context)
