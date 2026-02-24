"""Contract tests for bench.eval module's prompt_builder.py and __init__.py surface.

These tests verify the contract between PromptBuilder, the prompt composition
functions, and the public export surface from bench.eval.__init__.py.
"""

import pytest

from bench.eval import PromptBuilder, build_prompt_from_profile, compose_system_prompt
from bench.eval.prompt_builder import PromptBuilder as DirectPromptBuilder
from bench.profiles.support import (
    AgentConfig,
    SkillReference,
    SupportMode,
    SupportProfile,
    SystemPromptConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def skill_content() -> str:
    """Sample skill content for testing."""
    return """# R Testing Skill

This skill helps write tests using testthat.

## Guidelines
- Use expect_* functions
- Organize with describe/it blocks
"""


@pytest.fixture
def skill_ref(skill_content: str) -> SkillReference:
    """SkillReference with inline content."""
    return SkillReference(
        name="r-testing",
        content=skill_content,
        enabled=True,
        priority=1,
    )


@pytest.fixture
def task_context() -> str:
    """Sample task context for testing."""
    return "Write unit tests for the data processing function."


@pytest.fixture
def basic_system_config() -> SystemPromptConfig:
    """Basic system prompt configuration."""
    return SystemPromptConfig(
        enabled=True,
        injection_point="after_context",
    )


@pytest.fixture
def basic_agent_config() -> AgentConfig:
    """Basic agent configuration."""
    return AgentConfig(
        enabled=True,
        agent_type="default",
        max_iterations=10,
    )


@pytest.fixture
def system_only_profile(
    skill_ref: SkillReference,
    basic_system_config: SystemPromptConfig,
) -> SupportProfile:
    """SupportProfile in SYSTEM_ONLY mode."""
    return SupportProfile(
        profile_id="test-system-only",
        mode=SupportMode.SYSTEM_ONLY,
        skills=[skill_ref],
        system_prompt=basic_system_config,
        agent=AgentConfig(enabled=False),  # Explicitly disable agent
    )


@pytest.fixture
def none_mode_profile() -> SupportProfile:
    """SupportProfile in NONE mode."""
    return SupportProfile(
        profile_id="test-none",
        mode=SupportMode.NONE,
    )


@pytest.fixture
def agents_only_profile(
    skill_ref: SkillReference,
    basic_agent_config: AgentConfig,
) -> SupportProfile:
    """SupportProfile in AGENTS_ONLY mode."""
    return SupportProfile(
        profile_id="test-agents-only",
        mode=SupportMode.AGENTS_ONLY,
        skills=[skill_ref],
        agent=basic_agent_config,
        system_prompt=SystemPromptConfig(enabled=False),
    )


@pytest.fixture
def system_plus_agents_profile(
    skill_ref: SkillReference,
    basic_system_config: SystemPromptConfig,
    basic_agent_config: AgentConfig,
) -> SupportProfile:
    """SupportProfile in SYSTEM_PLUS_AGENTS mode."""
    return SupportProfile(
        profile_id="test-system-plus-agents",
        mode=SupportMode.SYSTEM_PLUS_AGENTS,
        skills=[skill_ref],
        system_prompt=basic_system_config,
        agent=basic_agent_config,
    )


# =============================================================================
# Test: PromptBuilder Composition Invariants
# =============================================================================


class TestPromptBuilderInjectionPoint:
    """Tests for injection_point controlling ordering in system prompt."""

    def test_before_context_puts_skill_first(
        self,
        skill_ref: SkillReference,
        task_context: str,
    ):
        """Test that before_context puts skill content before task context."""
        profile = SupportProfile(
            profile_id="test-before",
            mode=SupportMode.SYSTEM_ONLY,
            skills=[skill_ref],
            system_prompt=SystemPromptConfig(
                enabled=True,
                injection_point="before_context",
            ),
        )
        builder = PromptBuilder(profile)
        result = builder.build_system_prompt(task_context)

        # Skill content (contains "# R Testing Skill") should come before task context
        skill_marker = "# R Testing Skill"
        task_marker = "Write unit tests"

        skill_pos = result.find(skill_marker)
        task_pos = result.find(task_marker)

        assert skill_pos != -1, "Skill content should be present"
        assert task_pos != -1, "Task context should be present"
        assert skill_pos < task_pos, "Skill content should come before task context"

    def test_after_context_puts_task_first(
        self,
        skill_ref: SkillReference,
        task_context: str,
    ):
        """Test that after_context puts task context before skill content."""
        profile = SupportProfile(
            profile_id="test-after",
            mode=SupportMode.SYSTEM_ONLY,
            skills=[skill_ref],
            system_prompt=SystemPromptConfig(
                enabled=True,
                injection_point="after_context",
            ),
        )
        builder = PromptBuilder(profile)
        result = builder.build_system_prompt(task_context)

        skill_marker = "# R Testing Skill"
        task_marker = "Write unit tests"

        skill_pos = result.find(skill_marker)
        task_pos = result.find(task_marker)

        assert skill_pos != -1, "Skill content should be present"
        assert task_pos != -1, "Task context should be present"
        assert task_pos < skill_pos, "Task context should come before skill content"

    def test_replace_uses_only_skill_content(
        self,
        skill_ref: SkillReference,
        task_context: str,
    ):
        """Test that replace mode uses only skill content, ignoring task context."""
        profile = SupportProfile(
            profile_id="test-replace",
            mode=SupportMode.SYSTEM_ONLY,
            skills=[skill_ref],
            system_prompt=SystemPromptConfig(
                enabled=True,
                injection_point="replace",
            ),
        )
        builder = PromptBuilder(profile)
        result = builder.build_system_prompt(task_context)

        # Should contain skill content
        assert "# R Testing Skill" in result
        # Should NOT contain task context
        assert "Write unit tests" not in result


class TestPromptBuilderSupportMode:
    """Tests for SupportMode handling in PromptBuilder."""

    def test_none_mode_returns_minimal_prompt_with_context(
        self,
        none_mode_profile: SupportProfile,
        task_context: str,
    ):
        """Test that NONE mode with task context returns just the context."""
        builder = PromptBuilder(none_mode_profile)
        result = builder.build_system_prompt(task_context)

        assert result == task_context

    def test_none_mode_returns_minimal_prompt_without_context(
        self,
        none_mode_profile: SupportProfile,
    ):
        """Test that NONE mode without task context returns default message."""
        builder = PromptBuilder(none_mode_profile)
        result = builder.build_system_prompt(None)

        assert result == "Complete the task."

    def test_agents_only_returns_empty_or_task_context(
        self,
        agents_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that AGENTS_ONLY mode returns task context (no system prompt modification)."""
        builder = PromptBuilder(agents_only_profile)
        result = builder.build_system_prompt(task_context)

        # Should return task context (no modification for agents_only)
        assert result == task_context

    def test_agents_only_without_context_returns_empty(
        self,
        agents_only_profile: SupportProfile,
    ):
        """Test AGENTS_ONLY without task context returns empty string."""
        builder = PromptBuilder(agents_only_profile)
        result = builder.build_system_prompt(None)

        assert result == ""


class TestPromptBuilderFullPrompt:
    """Tests for build_full_prompt() method."""

    def test_returns_dict_with_system_key(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that build_full_prompt returns dict with 'system' key."""
        builder = PromptBuilder(system_only_profile)
        result = builder.build_full_prompt(task_context)

        assert isinstance(result, dict)
        assert "system" in result

    def test_includes_agent_key_when_enabled(
        self,
        system_plus_agents_profile: SupportProfile,
        task_context: str,
    ):
        """Test that 'agent' key is included when agent is enabled."""
        builder = PromptBuilder(system_plus_agents_profile)
        result = builder.build_full_prompt(task_context, include_agent=True)

        assert "system" in result
        assert "agent" in result

    def test_excludes_agent_key_when_not_enabled(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that 'agent' key is excluded when agent is not enabled."""
        builder = PromptBuilder(system_only_profile)
        result = builder.build_full_prompt(task_context, include_agent=True)

        assert "system" in result
        assert "agent" not in result

    def test_include_agent_false_excludes_agent(
        self,
        system_plus_agents_profile: SupportProfile,
        task_context: str,
    ):
        """Test that include_agent=False excludes agent key."""
        builder = PromptBuilder(system_plus_agents_profile)
        result = builder.build_full_prompt(task_context, include_agent=False)

        assert "system" in result
        assert "agent" not in result


class TestPromptBuilderAdditionalContext:
    """Tests for additional_context formatting."""

    def test_additional_context_string_values(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test formatting of string values in additional_context."""
        builder = PromptBuilder(system_only_profile)
        additional = {"version": "1.0", "author": "test"}
        result = builder.build_system_prompt(task_context, additional)

        assert "## Additional Context" in result
        assert "**version**: 1.0" in result
        assert "**author**: test" in result

    def test_additional_context_list_values(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test formatting of list values in additional_context."""
        builder = PromptBuilder(system_only_profile)
        additional = {"tags": ["testing", "r", "unit"]}
        result = builder.build_system_prompt(task_context, additional)

        assert "**tags**: testing, r, unit" in result

    def test_empty_additional_context_no_section(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that empty additional_context dict adds no section."""
        builder = PromptBuilder(system_only_profile)
        result = builder.build_system_prompt(task_context, {})

        assert "## Additional Context" not in result


# =============================================================================
# Test: Metadata Inclusion
# =============================================================================


class TestPromptBuilderManifestEntry:
    """Tests for to_manifest_entry() method."""

    def test_returns_expected_keys(
        self,
        system_only_profile: SupportProfile,
    ):
        """Test that to_manifest_entry returns all expected keys."""
        builder = PromptBuilder(system_only_profile)
        entry = builder.to_manifest_entry()

        expected_keys = [
            "support_mode",
            "support_fingerprint",
            "support_profile_id",
            "skill_order",
            "agent_enabled",
            "system_prompt_modified",
        ]

        for key in expected_keys:
            assert key in entry, f"Missing key: {key}"

    def test_support_mode_is_string_value(
        self,
        system_only_profile: SupportProfile,
    ):
        """Test that support_mode is the string value."""
        builder = PromptBuilder(system_only_profile)
        entry = builder.to_manifest_entry()

        assert entry["support_mode"] == "system_only"

    def test_profile_id_matches(
        self,
        system_only_profile: SupportProfile,
    ):
        """Test that support_profile_id matches profile."""
        builder = PromptBuilder(system_only_profile)
        entry = builder.to_manifest_entry()

        assert entry["support_profile_id"] == "test-system-only"

    def test_agent_enabled_reflects_profile(
        self,
        system_only_profile: SupportProfile,
        agents_only_profile: SupportProfile,
    ):
        """Test that agent_enabled reflects the profile configuration."""
        builder_system = PromptBuilder(system_only_profile)
        builder_agents = PromptBuilder(agents_only_profile)

        assert builder_system.to_manifest_entry()["agent_enabled"] is False
        assert builder_agents.to_manifest_entry()["agent_enabled"] is True


class TestPromptBuilderSkillContext:
    """Tests for skill context retrieval methods."""

    def test_get_ordered_skill_names_returns_list(
        self,
        system_only_profile: SupportProfile,
    ):
        """Test that get_ordered_skill_names returns a list."""
        builder = PromptBuilder(system_only_profile)
        result = builder.get_ordered_skill_names()

        assert isinstance(result, list)
        assert "r-testing" in result

    def test_get_skill_context_returns_dict(
        self,
        system_only_profile: SupportProfile,
    ):
        """Test that get_skill_context returns a dict."""
        builder = PromptBuilder(system_only_profile)
        result = builder.get_skill_context()

        assert isinstance(result, dict)
        assert "r-testing" in result

    def test_skill_context_contains_content(
        self,
        system_only_profile: SupportProfile,
        skill_content: str,
    ):
        """Test that skill context contains the skill content."""
        builder = PromptBuilder(system_only_profile)
        result = builder.get_skill_context()

        assert result["r-testing"] == skill_content


# =============================================================================
# Test: Public Export Surface Stability
# =============================================================================


class TestPublicExportsPromptBuilder:
    """Tests for PromptBuilder-related exports."""

    def test_prompt_builder_importable(self):
        """Test that PromptBuilder is importable from bench.eval."""
        from bench.eval import PromptBuilder

        assert PromptBuilder is DirectPromptBuilder

    def test_build_prompt_from_profile_importable(self):
        """Test that build_prompt_from_profile is importable."""
        from bench.eval import build_prompt_from_profile

        assert callable(build_prompt_from_profile)

    def test_compose_system_prompt_importable(self):
        """Test that compose_system_prompt is importable."""
        from bench.eval import compose_system_prompt

        assert callable(compose_system_prompt)


class TestPublicExportsPolicy:
    """Tests for skill policy exports."""

    def test_heuristic_policy_importable(self):
        """Test that HeuristicPolicy is importable."""
        from bench.eval import HeuristicPolicy

        assert HeuristicPolicy is not None

    def test_keyword_match_policy_importable(self):
        """Test that KeywordMatchPolicy is importable."""
        from bench.eval import KeywordMatchPolicy

        assert KeywordMatchPolicy is not None

    def test_policy_type_importable(self):
        """Test that PolicyType is importable."""
        from bench.eval import PolicyType

        assert PolicyType.HEURISTIC.value == "heuristic"
        assert PolicyType.KEYWORD.value == "keyword"

    def test_selection_rationale_importable(self):
        """Test that SelectionRationale is importable."""
        from bench.eval import SelectionRationale

        rationale = SelectionRationale(skill_id="test", score=0.5)
        assert rationale.skill_id == "test"

    def test_selection_result_importable(self):
        """Test that SelectionResult is importable."""
        from bench.eval import SelectionResult

        result = SelectionResult(selected_skills=["a"], selection_policy="test")
        assert result.selected_skills == ["a"]

    def test_skill_metadata_importable(self):
        """Test that SkillMetadata is importable."""
        from bench.eval import SkillMetadata

        meta = SkillMetadata(skill_id="test", name="Test")
        assert meta.skill_id == "test"

    def test_skill_selection_policy_importable(self):
        """Test that SkillSelectionPolicy is importable."""
        from bench.eval import SkillSelectionPolicy

        assert SkillSelectionPolicy is not None


class TestPublicExportsTelemetry:
    """Tests for telemetry exports."""

    def test_telemetry_collector_importable(self):
        """Test that TelemetryCollector is importable."""
        from bench.eval import TelemetryCollector

        assert TelemetryCollector is not None

    def test_tool_call_context_importable(self):
        """Test that ToolCallContext is importable."""
        from bench.eval import ToolCallContext

        assert ToolCallContext is not None

    def test_tool_call_event_importable(self):
        """Test that ToolCallEvent is importable."""
        from bench.eval import ToolCallEvent

        assert ToolCallEvent is not None

    def test_tool_error_event_importable(self):
        """Test that ToolErrorEvent is importable."""
        from bench.eval import ToolErrorEvent

        assert ToolErrorEvent is not None

    def test_outcome_type_importable(self):
        """Test that OutcomeType is importable."""
        from bench.eval import OutcomeType

        assert OutcomeType is not None

    def test_tool_error_type_importable(self):
        """Test that ToolErrorType is importable."""
        from bench.eval import ToolErrorType

        assert ToolErrorType is not None

    def test_tool_metrics_importable(self):
        """Test that ToolMetrics is importable."""
        from bench.eval import ToolMetrics

        assert ToolMetrics is not None

    def test_classify_error_type_importable(self):
        """Test that classify_error_type is importable."""
        from bench.eval import classify_error_type

        assert callable(classify_error_type)


class TestPublicExportsToolRegistry:
    """Tests for tool registry exports."""

    def test_tool_registry_importable(self):
        """Test that ToolRegistry is importable."""
        from bench.eval import ToolRegistry

        assert ToolRegistry is not None

    def test_tool_registry_error_importable(self):
        """Test that ToolRegistryError is importable."""
        from bench.eval import ToolRegistryError

        assert issubclass(ToolRegistryError, Exception)

    def test_tool_not_found_error_importable(self):
        """Test that ToolNotFoundError is importable."""
        from bench.eval import ToolNotFoundError

        assert issubclass(ToolNotFoundError, Exception)

    def test_tool_validation_error_importable(self):
        """Test that ToolValidationError is importable."""
        from bench.eval import ToolValidationError

        assert issubclass(ToolValidationError, Exception)

    def test_tool_version_not_found_error_importable(self):
        """Test that ToolVersionNotFoundError is importable."""
        from bench.eval import ToolVersionNotFoundError

        assert issubclass(ToolVersionNotFoundError, Exception)

    def test_get_registry_importable(self):
        """Test that get_registry is importable."""
        from bench.eval import get_registry

        assert callable(get_registry)

    def test_reset_registry_importable(self):
        """Test that reset_registry is importable."""
        from bench.eval import reset_registry

        assert callable(reset_registry)


class TestPublicExportsDiscovery:
    """Tests for skill discovery exports."""

    def test_discover_skills_importable(self):
        """Test that discover_skills is importable."""
        from bench.eval import discover_skills

        assert callable(discover_skills)

    def test_load_policy_from_config_importable(self):
        """Test that load_policy_from_config is importable."""
        from bench.eval import load_policy_from_config

        assert callable(load_policy_from_config)

    def test_load_policy_from_yaml_importable(self):
        """Test that load_policy_from_yaml is importable."""
        from bench.eval import load_policy_from_yaml

        assert callable(load_policy_from_yaml)


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestBuildPromptFromProfile:
    """Tests for build_prompt_from_profile convenience function."""

    def test_returns_dict_with_system_key(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that build_prompt_from_profile returns dict with system key."""
        result = build_prompt_from_profile(system_only_profile, task_context)

        assert isinstance(result, dict)
        assert "system" in result

    def test_accepts_additional_context(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that additional_context is accepted and processed."""
        additional = {"key": "value"}
        result = build_prompt_from_profile(
            system_only_profile,
            task_context,
            additional_context=additional,
        )

        assert "**key**: value" in result["system"]


class TestComposeSystemPrompt:
    """Tests for compose_system_prompt low-level function."""

    def test_composes_from_components(
        self,
        skill_ref: SkillReference,
        task_context: str,
    ):
        """Test composing system prompt from individual components."""
        result = compose_system_prompt(
            mode=SupportMode.SYSTEM_ONLY,
            skills=[skill_ref],
            task_context=task_context,
        )

        assert isinstance(result, str)
        assert task_context in result

    def test_with_custom_system_config(
        self,
        skill_ref: SkillReference,
        task_context: str,
    ):
        """Test composing with custom SystemPromptConfig."""
        config = SystemPromptConfig(
            enabled=True,
            injection_point="before_context",
        )
        result = compose_system_prompt(
            mode=SupportMode.SYSTEM_ONLY,
            skills=[skill_ref],
            task_context=task_context,
            system_config=config,
        )

        skill_pos = result.find("# R Testing Skill")
        task_pos = result.find("Write unit tests")
        assert skill_pos < task_pos


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestPromptBuilderEdgeCases:
    """Tests for edge cases in PromptBuilder."""

    def test_none_task_context_with_skills(
        self,
        system_only_profile: SupportProfile,
    ):
        """Test handling None task_context with skills present."""
        builder = PromptBuilder(system_only_profile)
        result = builder.build_system_prompt(None)

        # Should still include skill content
        assert "# R Testing Skill" in result

    def test_empty_skills_list(self, task_context: str):
        """Test handling empty skills list."""
        profile = SupportProfile(
            profile_id="empty-skills",
            mode=SupportMode.SYSTEM_ONLY,
            skills=[],
            system_prompt=SystemPromptConfig(enabled=True),
        )
        builder = PromptBuilder(profile)
        result = builder.build_system_prompt(task_context)

        # Should just return task context
        assert task_context in result

    def test_disabled_skill_excluded(self, task_context: str):
        """Test that disabled skills are excluded."""
        disabled_skill = SkillReference(
            name="disabled",
            content="This should not appear",
            enabled=False,
        )
        profile = SupportProfile(
            profile_id="disabled-test",
            mode=SupportMode.SYSTEM_ONLY,
            skills=[disabled_skill],
            system_prompt=SystemPromptConfig(enabled=True),
        )
        builder = PromptBuilder(profile)
        result = builder.build_system_prompt(task_context)

        assert "This should not appear" not in result

    def test_fingerprint_is_cached(self, system_only_profile: SupportProfile):
        """Test that composed artifact is cached."""
        builder = PromptBuilder(system_only_profile)

        # Access composed twice
        composed1 = builder.composed
        composed2 = builder.composed

        # Should be the same object (cached)
        assert composed1 is composed2

    def test_fingerprint_string_format(self, system_only_profile: SupportProfile):
        """Test that fingerprint returns a string."""
        builder = PromptBuilder(system_only_profile)
        fingerprint = builder.fingerprint

        assert isinstance(fingerprint, str)
        assert ":" in fingerprint  # Format is "mode:hash"

    def test_multiple_skills_ordered_by_priority(self, task_context: str):
        """Test that skills are ordered by priority."""
        skills = [
            SkillReference(name="low", content="Low priority", priority=1),
            SkillReference(name="high", content="High priority", priority=10),
            SkillReference(name="medium", content="Medium priority", priority=5),
        ]
        profile = SupportProfile(
            profile_id="priority-test",
            mode=SupportMode.SYSTEM_ONLY,
            skills=skills,
            system_prompt=SystemPromptConfig(enabled=True, injection_point="after_context"),
        )
        builder = PromptBuilder(profile)
        ordered = builder.get_ordered_skill_names()

        # Higher priority should come first
        assert ordered == ["high", "medium", "low"]

    def test_agent_prompt_for_non_agent_mode(
        self,
        system_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that agent prompt is empty for non-agent modes."""
        builder = PromptBuilder(system_only_profile)
        result = builder.build_agent_prompt(task_context)

        assert result == ""

    def test_agent_prompt_for_agents_only_mode(
        self,
        agents_only_profile: SupportProfile,
        task_context: str,
    ):
        """Test that agent prompt is populated for AGENTS_ONLY mode."""
        builder = PromptBuilder(agents_only_profile)
        result = builder.build_agent_prompt(task_context)

        # Should include agent configuration info
        assert "Agent Type" in result or "default" in result


class TestProfileValidation:
    """Tests for profile validation edge cases."""

    def test_single_skill_mode_requires_skill(self):
        """Test that SINGLE_SKILL mode requires at least one skill."""
        with pytest.raises(ValueError, match="single_skill mode requires"):
            SupportProfile(
                profile_id="invalid",
                mode=SupportMode.SINGLE_SKILL,
                skills=[],
            )

    def test_single_skill_mode_with_skill(self, skill_ref: SkillReference):
        """Test that SINGLE_SKILL mode works with a skill."""
        profile = SupportProfile(
            profile_id="valid-single",
            mode=SupportMode.SINGLE_SKILL,
            skills=[skill_ref],
        )
        builder = PromptBuilder(profile)
        result = builder.build_system_prompt("task")

        assert "task" in result

    def test_collection_forced_requires_path(self):
        """Test that COLLECTION_FORCED mode requires collection_path."""
        with pytest.raises(ValueError, match="collection_forced mode requires"):
            SupportProfile(
                profile_id="invalid",
                mode=SupportMode.COLLECTION_FORCED,
                collection_path=None,
            )

    def test_collection_selective_requires_path(self):
        """Test that COLLECTION_SELECTIVE mode requires collection_path."""
        with pytest.raises(ValueError, match="collection_selective mode requires"):
            SupportProfile(
                profile_id="invalid",
                mode=SupportMode.COLLECTION_SELECTIVE,
                collection_path=None,
            )
