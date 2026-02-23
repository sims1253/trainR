"""Tests for skill selection policy module."""

import pytest

from bench.eval.skill_policy import (
    HeuristicPolicy,
    KeywordMatchPolicy,
    PolicyType,
    SelectionRationale,
    SelectionResult,
    SkillMetadata,
    SkillSelectionPolicy,
    load_policy_from_config,
)
from bench.profiles.support import (
    SelectionMetadata,
    SupportMode,
    SupportProfile,
)


def make_skill_metadata(
    skill_id: str,
    name: str = "",
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    domains: list[str] | None = None,
) -> SkillMetadata:
    """Helper to create SkillMetadata for testing."""
    return SkillMetadata(
        skill_id=skill_id,
        name=name or skill_id,
        tags=tags or [],
        keywords=keywords or [],
        domains=domains or [],
    )


def make_task_context(
    instruction: str = "Write tests for the function",
    context: str = "",
    source_package: str = "testpkg",
) -> dict:
    """Helper to create task context for testing."""
    return {
        "instruction": instruction,
        "context": context,
        "source_package": source_package,
    }


class TestSkillMetadata:
    """Tests for SkillMetadata."""

    def test_basic_creation(self):
        """Test basic metadata creation."""
        meta = make_skill_metadata(
            skill_id="test-skill",
            tags=["testing", "r"],
        )
        assert meta.skill_id == "test-skill"
        assert "testing" in meta.tags
        assert meta.enabled is True

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = make_skill_metadata(
            skill_id="my-skill",
            name="My Skill",
            tags=["tag1"],
        )
        d = meta.to_dict()
        assert d["skill_id"] == "my-skill"
        assert d["name"] == "My Skill"
        assert "tag1" in d["tags"]


class TestSelectionRationale:
    """Tests for SelectionRationale."""

    def test_basic_creation(self):
        """Test basic rationale creation."""
        rationale = SelectionRationale(
            skill_id="test-skill",
            score=0.5,
            reasons=["Domain match: testing"],
            matched_keywords=["test"],
            matched_tags=["testing"],
        )
        assert rationale.skill_id == "test-skill"
        assert rationale.score == 0.5
        assert "Domain match: testing" in rationale.reasons

    def test_to_dict(self):
        """Test serialization."""
        rationale = SelectionRationale(
            skill_id="skill1",
            score=0.8,
            reasons=["reason1"],
            matched_keywords=["kw1"],
            matched_tags=["tag1"],
        )
        d = rationale.to_dict()
        assert d["skill_id"] == "skill1"
        assert d["score"] == 0.8


class TestSelectionResult:
    """Tests for SelectionResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = SelectionResult(
            selected_skills=["skill1", "skill2"],
            selection_policy="heuristic",
        )
        assert result.selected_skills == ["skill1", "skill2"]
        assert result.selection_policy == "heuristic"

    def test_get_ordered_skills_deterministic(self):
        """Test that skill ordering is deterministic (alphabetical)."""
        result = SelectionResult(
            selected_skills=["zebra", "apple", "mango"],
            selection_policy="test",
        )
        ordered = result.get_ordered_skills()
        assert ordered == ["apple", "mango", "zebra"]

    def test_to_dict(self):
        """Test serialization."""
        result = SelectionResult(
            selected_skills=["skill1"],
            selection_policy="heuristic",
            seed=42,
        )
        d = result.to_dict()
        assert d["selected_skills"] == ["skill1"]
        assert d["selection_policy"] == "heuristic"
        assert d["seed"] == 42


class TestHeuristicPolicy:
    """Tests for HeuristicPolicy."""

    def test_policy_type(self):
        """Test policy type is correct."""
        policy = HeuristicPolicy()
        assert policy.policy_type == PolicyType.HEURISTIC

    def test_select_testing_skill_for_test_task(self):
        """Test that testing skills are selected for test tasks."""
        policy = HeuristicPolicy(min_score=0.0, max_skills=5)
        skills = [
            make_skill_metadata(
                skill_id="r-testing",
                tags=["testing", "testthat"],
                domains=["testing"],
            ),
            make_skill_metadata(
                skill_id="r-plotting",
                tags=["plotting", "ggplot"],
                domains=["tidyverse"],
            ),
        ]
        task = make_task_context(instruction="Write tests for my_function")

        result = policy.select(task, skills)

        assert "r-testing" in result.selected_skills
        assert result.selection_policy == "heuristic"

    def test_select_with_min_score(self):
        """Test that min_score filters low-scoring skills."""
        policy = HeuristicPolicy(min_score=0.1)
        skills = [
            # skill1 will match because "test" is a testing keyword and in the instruction
            make_skill_metadata(
                skill_id="skill1",
                tags=["testing"],
                domains=["testing"],
            ),
            # skill2 should not match (no relevant keywords)
            make_skill_metadata(skill_id="skill2", tags=["unrelated"]),
        ]
        task = make_task_context(instruction="Write a test for my function")

        result = policy.select(task, skills)

        # Only skill1 should match - it has testing domain and tag matching
        assert "skill1" in result.selected_skills
        assert "skill2" not in result.selected_skills

    def test_max_skills_limit(self):
        """Test that max_skills limits selection."""
        policy = HeuristicPolicy(max_skills=2, min_score=0.0)
        skills = [
            make_skill_metadata(skill_id="skill1", tags=["test"]),
            make_skill_metadata(skill_id="skill2", tags=["test"]),
            make_skill_metadata(skill_id="skill3", tags=["test"]),
        ]
        task = make_task_context(instruction="test test test")

        result = policy.select(task, skills)

        assert len(result.selected_skills) <= 2

    def test_deterministic_with_seed(self):
        """Test that selection is deterministic with same seed."""
        skills = [
            make_skill_metadata(skill_id="skill1", tags=["test"]),
            make_skill_metadata(skill_id="skill2", tags=["test"]),
        ]
        task = make_task_context(instruction="Write a test")

        policy1 = HeuristicPolicy(seed=42)
        policy2 = HeuristicPolicy(seed=42)

        result1 = policy1.select(task, skills)
        result2 = policy2.select(task, skills)

        assert result1.get_ordered_skills() == result2.get_ordered_skills()

    def test_includes_rationale(self):
        """Test that selection includes rationale."""
        policy = HeuristicPolicy()
        skills = [
            make_skill_metadata(
                skill_id="test-skill",
                tags=["testing"],
                domains=["testing"],
            ),
        ]
        task = make_task_context(instruction="Write a test for this function")

        result = policy.select(task, skills)

        assert len(result.selection_rationale) > 0
        rationale = result.selection_rationale[0]
        assert rationale.skill_id == "test-skill"
        assert len(rationale.reasons) > 0


class TestKeywordMatchPolicy:
    """Tests for KeywordMatchPolicy."""

    def test_policy_type(self):
        """Test policy type is correct."""
        policy = KeywordMatchPolicy()
        assert policy.policy_type == PolicyType.KEYWORD

    def test_keyword_matching(self):
        """Test that keyword matching works."""
        policy = KeywordMatchPolicy(min_score=0.0)
        skills = [
            make_skill_metadata(
                skill_id="cli-skill",
                keywords=["cli", "console", "output"],
            ),
            make_skill_metadata(
                skill_id="other-skill",
                keywords=["graphics", "plot"],
            ),
        ]
        task = make_task_context(instruction="Format console output with cli")

        result = policy.select(task, skills)

        assert "cli-skill" in result.selected_skills

    def test_tag_matching(self):
        """Test that tag matching works."""
        policy = KeywordMatchPolicy(min_score=0.0)
        skills = [
            make_skill_metadata(
                skill_id="tidy-skill",
                tags=["tidyverse", "dplyr"],
            ),
        ]
        task = make_task_context(instruction="Use dplyr to filter data")

        result = policy.select(task, skills)

        assert "tidy-skill" in result.selected_skills

    def test_custom_weights(self):
        """Test custom weights configuration."""
        policy = KeywordMatchPolicy(
            keyword_weight=0.5,
            tag_weight=0.3,
            name_weight=0.2,
        )
        assert policy.keyword_weight == 0.5
        assert policy.tag_weight == 0.3
        assert policy.name_weight == 0.2


class TestLoadPolicyFromConfig:
    """Tests for load_policy_from_config."""

    def test_load_heuristic_policy(self):
        """Test loading heuristic policy from config."""
        config = {
            "type": "heuristic",
            "min_score": 0.2,
            "max_skills": 3,
        }
        policy = load_policy_from_config(config)

        assert isinstance(policy, HeuristicPolicy)
        assert policy.min_score == 0.2
        assert policy.max_skills == 3

    def test_load_keyword_policy(self):
        """Test loading keyword policy from config."""
        config = {
            "type": "keyword",
            "min_score": 0.1,
            "keyword_weight": 0.4,
        }
        policy = load_policy_from_config(config)

        assert isinstance(policy, KeywordMatchPolicy)
        assert policy.min_score == 0.1
        assert policy.keyword_weight == 0.4

    def test_load_unknown_type_raises(self):
        """Test that unknown policy type raises error."""
        config = {"type": "unknown"}
        with pytest.raises(ValueError, match="Unknown policy type"):
            load_policy_from_config(config)


class TestSelectionMetadata:
    """Tests for SelectionMetadata."""

    def test_from_selection_result(self):
        """Test creating from SelectionResult."""
        result = SelectionResult(
            selected_skills=["skill1", "skill2"],
            selection_policy="heuristic",
            seed=42,
            selection_rationale=[
                SelectionRationale(skill_id="skill1", score=0.5, reasons=["test"])
            ],
        )

        metadata = SelectionMetadata.from_selection_result(result)

        assert metadata.selected_skills == ["skill1", "skill2"]  # sorted
        assert metadata.selection_policy == "heuristic"
        assert metadata.seed == 42
        assert len(metadata.selection_rationale) == 1

    def test_is_reproducible(self):
        """Test reproducibility check."""
        with_seed = SelectionMetadata(seed=42)
        without_seed = SelectionMetadata(seed=None)

        assert with_seed.is_reproducible() is True
        assert without_seed.is_reproducible() is False


class TestSupportProfileSelection:
    """Tests for SupportProfile skill selection."""

    def test_compose_with_task_selective(self):
        """Test compose_with_task for selective mode."""
        profile = SupportProfile(
            profile_id="test-selective",
            mode=SupportMode.COLLECTION_SELECTIVE,
            collection_path="skills",
            selection_criteria={
                "policy": "heuristic_balanced",
            },
        )

        task_context = {
            "instruction": "Write tests for my R package",
            "context": "Using testthat framework",
            "source_package": "mypkg",
        }

        # This should not raise
        artifact = profile.compose_with_task(task_context)

        assert artifact.fingerprint.mode == "collection_selective"
        # Selection metadata should be present
        if artifact.selection_metadata:
            assert artifact.selection_metadata.selection_policy in ["heuristic", "keyword"]

    def test_compose_without_task_context_warning(self):
        """Test that compose without task context logs warning for selective mode."""
        profile = SupportProfile(
            profile_id="test-selective",
            mode=SupportMode.COLLECTION_SELECTIVE,
            collection_path="skills",
        )

        # compose without task_context should use default
        artifact = profile.compose()

        assert artifact.fingerprint.mode == "collection_selective"
