"""Tests for mined task schemas."""

import pytest
from pydantic import ValidationError

from task_generator.mined_task import (
    Difficulty,
    MinedTaskSchema,
    MiningConfig,
    MiningStats,
    PRAnalysisInput,
    QualityScore,
    RepoConfig,
    TaskDefinition,
    TaskFiles,
    TaskMetadata,
    TaskSolution,
    TaskSource,
    TaskTests,
    TaskType,
)


def test_task_type_enum():
    """Test TaskType enum values."""
    assert TaskType.BUG_FIX == "bug_fix"
    assert TaskType.FEATURE_IMPL == "feature_impl"
    assert TaskType.TEST_WRITING == "test_writing"
    assert TaskType.REFACTOR == "refactor"
    assert TaskType.DOCUMENTATION == "documentation"
    assert TaskType.PERFORMANCE == "performance"
    assert TaskType.SECURITY == "security"
    assert TaskType.DEPENDENCY == "dependency"


def test_difficulty_enum():
    """Test Difficulty enum values."""
    assert Difficulty.EASY == "easy"
    assert Difficulty.MEDIUM == "medium"
    assert Difficulty.HARD == "hard"


def test_quality_score_enum():
    """Test QualityScore enum values."""
    assert QualityScore.MINIMUM == 1
    assert QualityScore.ACCEPTABLE == 5
    assert QualityScore.GOOD == 7
    assert QualityScore.EXCELLENT == 9
    assert QualityScore.MAXIMUM == 10


def test_pr_analysis_input_defaults():
    """Test PRAnalysisInput with minimal required fields."""
    inp = PRAnalysisInput(pr_title="Fix bug")
    assert inp.pr_title == "Fix bug"
    assert inp.pr_body == ""
    assert inp.issue_title == ""
    assert inp.issue_body == ""
    assert inp.files_changed == []
    assert inp.code_diff == ""
    assert inp.test_diff is None


def test_pr_analysis_input_full():
    """Test PRAnalysisInput with all fields."""
    inp = PRAnalysisInput(
        pr_title="Fix bug",
        pr_body="Body text",
        issue_title="Bug report",
        issue_body="Issue description",
        files_changed=["test.R"],
        code_diff="+ added line",
        test_diff="+ test line",
    )
    assert inp.pr_title == "Fix bug"
    assert inp.pr_body == "Body text"
    assert inp.issue_title == "Bug report"
    assert inp.issue_body == "Issue description"
    assert inp.files_changed == ["test.R"]
    assert inp.code_diff == "+ added line"
    assert inp.test_diff == "+ test line"


def test_pr_analysis_input_files_changed_validator():
    """Test that files_changed accepts list."""
    inp = PRAnalysisInput(
        pr_title="Test",
        pr_body="",
        issue_title="",
        issue_body="",
        files_changed=["single_file.R"],
        code_diff="",
        test_diff=None,
    )
    assert inp.files_changed == ["single_file.R"]


def test_mined_task_schema_defaults():
    """Test MinedTaskSchema with required fields and defaults."""
    schema = MinedTaskSchema(
        task_type=TaskType.BUG_FIX,
        difficulty=Difficulty.MEDIUM,
        quality_score=7,
        instruction="Test instruction",
        context="Test context",
        key_changes=["change1"],
        potential_pitfalls=["pitfall1"],
        is_good_task=True,
    )
    assert schema.task_type == TaskType.BUG_FIX
    assert schema.difficulty == Difficulty.MEDIUM
    assert schema.quality_score == 7
    assert schema.instruction == "Test instruction"
    assert schema.context == "Test context"
    assert schema.key_changes == ["change1"]
    assert schema.potential_pitfalls == ["pitfall1"]
    assert schema.is_good_task is True
    assert schema.fail_to_pass == []
    assert schema.pass_to_pass == []
    assert schema.rejection_reason is None


def test_mined_task_schema_rejection():
    """Test MinedTaskSchema with rejection reason."""
    schema = MinedTaskSchema(
        task_type=TaskType.DOCUMENTATION,
        difficulty=Difficulty.EASY,
        quality_score=3,
        instruction="Test",
        context="Test",
        key_changes=[],
        potential_pitfalls=[],
        is_good_task=False,
        rejection_reason="Documentation only changes",
    )
    assert schema.is_good_task is False
    assert schema.rejection_reason == "Documentation only changes"


def test_quality_score_validation_valid():
    """Test that valid quality scores are accepted."""
    for score in [1, 5, 10]:
        schema = MinedTaskSchema(
            task_type=TaskType.BUG_FIX,
            difficulty=Difficulty.EASY,
            quality_score=score,
            instruction="Test",
            context="Test",
            key_changes=[],
            potential_pitfalls=[],
            is_good_task=True,
        )
        assert schema.quality_score == score


def test_quality_score_validation_invalid():
    """Test that quality_score must be 1-10."""
    with pytest.raises(ValidationError):
        MinedTaskSchema(
            task_type=TaskType.BUG_FIX,
            difficulty=Difficulty.EASY,
            quality_score=0,  # Invalid: below minimum
            instruction="Test",
            context="Test",
            key_changes=[],
            potential_pitfalls=[],
            is_good_task=True,
        )

    with pytest.raises(ValidationError):
        MinedTaskSchema(
            task_type=TaskType.BUG_FIX,
            difficulty=Difficulty.EASY,
            quality_score=11,  # Invalid: above maximum
            instruction="Test",
            context="Test",
            key_changes=[],
            potential_pitfalls=[],
            is_good_task=True,
        )


def test_task_source():
    """Test TaskSource model."""
    source = TaskSource(
        repo="owner/repo",
        pr_number=42,
        pr_url="https://github.com/owner/repo/pull/42",
    )
    assert source.type == "github_pr"
    assert source.repo == "owner/repo"
    assert source.pr_number == 42
    assert source.merged_at is None
    assert source.issue_number is None


def test_task_metadata():
    """Test TaskMetadata model."""
    metadata = TaskMetadata(
        task_type="bug_fix",
        difficulty="medium",
        quality_score=7,
        mined_at="2024-01-01T00:00:00",
    )
    assert metadata.task_type == "bug_fix"
    assert metadata.difficulty == "medium"
    assert metadata.quality_score == 7
    assert metadata.miner_version == "1.0.0"


def test_task_definition():
    """Test TaskDefinition model."""
    definition = TaskDefinition(
        instruction="Write a test",
        context="Function code here",
        hints=["hint1", "hint2"],
    )
    assert definition.instruction == "Write a test"
    assert definition.context == "Function code here"
    assert definition.hints == ["hint1", "hint2"]


def test_task_tests():
    """Test TaskTests model."""
    tests = TaskTests(
        fail_to_pass=["test1", "test2"],
        pass_to_pass=["test3"],
        test_files=["tests/test.R"],
    )
    assert tests.fail_to_pass == ["test1", "test2"]
    assert tests.pass_to_pass == ["test3"]
    assert tests.test_files == ["tests/test.R"]


def test_task_solution():
    """Test TaskSolution model."""
    solution = TaskSolution(
        key_changes=["change1"],
        potential_pitfalls=["pitfall1"],
        reference_diff="+ added code",
    )
    assert solution.key_changes == ["change1"]
    assert solution.potential_pitfalls == ["pitfall1"]
    assert solution.reference_diff == "+ added code"


def test_task_files():
    """Test TaskFiles model."""
    files = TaskFiles(
        changed=["R/code.R"],
        test_changes="+ test code",
        source_files=["R/code.R"],
    )
    assert files.changed == ["R/code.R"]
    assert files.test_changes == "+ test code"
    assert files.source_files == ["R/code.R"]


def test_mining_stats():
    """Test MiningStats model."""
    stats = MiningStats(
        start_time="2024-01-01T00:00:00",
        repos_processed=10,
        prs_found=100,
        tasks_created=25,
    )
    assert stats.start_time == "2024-01-01T00:00:00"
    assert stats.repos_processed == 10
    assert stats.prs_found == 100
    assert stats.tasks_created == 25
    assert stats.errors == []


def test_mining_stats_to_summary():
    """Test MiningStats to_summary method."""
    stats = MiningStats(
        start_time="2024-01-01T00:00:00",
        repos_processed=5,
        prs_found=50,
        tasks_created=10,
        tasks_rejected=5,
    )
    summary = stats.to_summary()
    assert "Mining Statistics" in summary
    assert "Repositories: 5" in summary
    assert "PRs found: 50" in summary
    assert "Tasks created: 10" in summary


def test_repo_config():
    """Test RepoConfig model."""
    config = RepoConfig(owner="tidyverse", repo="dplyr")
    assert config.owner == "tidyverse"
    assert config.repo == "dplyr"
    assert config.full_name == "tidyverse/dplyr"
    assert config.min_stars is None
    assert config.reason is None


def test_repo_config_with_options():
    """Test RepoConfig with optional fields."""
    config = RepoConfig(
        owner="tidyverse",
        repo="ggplot2",
        min_stars=1000,
        reason="Popular visualization package",
    )
    assert config.min_stars == 1000
    assert config.reason == "Popular visualization package"


def test_mining_config():
    """Test MiningConfig model."""
    config = MiningConfig(
        high_priority=[
            RepoConfig(owner="tidyverse", repo="dplyr"),
        ]
    )
    assert len(config.high_priority) == 1
    assert config.high_priority[0].repo == "dplyr"
    assert "since_days" in config.settings


def test_mining_config_all_repos():
    """Test MiningConfig all_repos method."""
    config = MiningConfig(
        high_priority=[RepoConfig(owner="a", repo="1")],
        medium_priority=[RepoConfig(owner="b", repo="2")],
    )
    all_repos = config.all_repos()
    assert len(all_repos) == 2
