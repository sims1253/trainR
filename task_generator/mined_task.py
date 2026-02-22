"""
Pydantic models for structured LLM output from PR mining.

These models define the schema for:
1. Input to the LLM judge (PR analysis input)
2. Output from the LLM judge (mined task schema)
3. Final task format for storage

Usage:
    from task_generator.mined_task import MinedTaskSchema, PRAnalysisInput

    # Create input for LLM
    input_data = PRAnalysisInput(
        pr_title="Fix bug in data filtering",
        pr_body="This PR fixes...",
        ...
    )

    # Get structured output from LLM
    result = llm_client.parse(input_data, MinedTaskSchema)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Classification of the task type based on PR content"""

    BUG_FIX = "bug_fix"
    FEATURE_IMPL = "feature_impl"
    TEST_WRITING = "test_writing"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DEPENDENCY = "dependency"


class Difficulty(str, Enum):
    """Estimated difficulty level for solving the task"""

    EASY = "easy"  # Single file, < 20 lines changed, obvious fix
    MEDIUM = "medium"  # Multiple files, 20-100 lines, some investigation needed
    HARD = "hard"  # Complex logic, > 100 lines, domain knowledge required


class QualityScore(int, Enum):
    """Quality score thresholds for task filtering"""

    MINIMUM = 1
    ACCEPTABLE = 5
    GOOD = 7
    EXCELLENT = 9
    MAXIMUM = 10


class PRAnalysisInput(BaseModel):
    """
    Input data for LLM judge to analyze a pull request.

    This model contains all the information the LLM needs to evaluate
    whether a PR would make a good testing task.
    """

    pr_title: str = Field(description="Title of the pull request")
    pr_body: str = Field(
        description="Body/description of the pull request",
        default="",
    )
    issue_title: str = Field(
        description="Title of the linked issue (if any)",
        default="",
    )
    issue_body: str = Field(
        description="Body/description of the linked issue",
        default="",
    )
    files_changed: list[str] = Field(
        description="List of file paths modified in the PR",
        default_factory=list,
    )
    code_diff: str = Field(
        description="The complete diff of source code changes",
        default="",
    )
    test_diff: str | None = Field(
        description="The diff of test file changes (if any)",
        default=None,
    )

    @field_validator("files_changed", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class MinedTaskSchema(BaseModel):
    """
    Structured output from LLM judge for PR analysis.

    This is the primary schema used for structured output from the LLM.
    It captures both the evaluation (quality, difficulty) and the
    extracted task information (instruction, tests).
    """

    # Classification
    task_type: TaskType = Field(
        description="Type of task this PR represents",
    )
    difficulty: Difficulty = Field(
        description="Estimated difficulty for solving this task",
    )
    quality_score: int = Field(
        ge=1,
        le=10,
        description="Task quality rating from 1-10. "
        "7+ is good for evaluation. "
        "Consider: clarity, testability, self-contained, realistic.",
    )

    # Task content
    instruction: str = Field(
        description="Human-readable task description. "
        "Should be clear enough for a developer to understand what needs to be done. "
        "Include context about what feature/bug is being addressed.",
    )
    context: str = Field(
        description="Relevant code context that would help solve the task. "
        "Include function signatures, relevant imports, or related code snippets.",
        default="",
    )

    # Test specifications
    fail_to_pass: list[str] = Field(
        default_factory=list,
        description="Tests that should FAIL before the fix and PASS after. "
        "These are the core verification tests for the task.",
    )
    pass_to_pass: list[str] = Field(
        default_factory=list,
        description="Tests that should PASS both before and after the fix. "
        "These verify that existing functionality isn't broken.",
    )

    # Solution hints
    key_changes: list[str] = Field(
        default_factory=list,
        description="Key code changes needed to solve the task. "
        "These are hints about what the solution involves.",
    )
    potential_pitfalls: list[str] = Field(
        default_factory=list,
        description="Common mistakes developers might make when solving this task. "
        "Useful for understanding edge cases and tricky aspects.",
    )

    # Evaluation
    is_good_task: bool = Field(
        description="Whether this PR makes a good evaluation task. "
        "False if the task is too ambiguous, too large, or unsuitable for other reasons.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="If is_good_task is False, explain why. "
        "E.g., 'Documentation only', 'Too many files changed', 'Unclear requirements'.",
    )

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Quality score must be between 1 and 10")
        return v

    @field_validator("is_good_task")
    @classmethod
    def validate_rejection(cls, v, info):
        # If task is rejected, there should be a reason
        if not v and not info.data.get("rejection_reason"):
            # Set a default reason if missing
            pass  # Allow for now, but log warning
        return v


class TaskSource(BaseModel):
    """Information about where the task came from"""

    type: str = Field(default="github_pr", description="Source type")
    repo: str = Field(description="Repository name (owner/repo)")
    pr_number: int = Field(description="Pull request number")
    pr_url: str = Field(description="URL to the pull request")
    merged_at: str | None = Field(default=None, description="When the PR was merged (ISO format)")
    issue_number: int | None = Field(default=None, description="Linked issue number (if any)")
    issue_url: str | None = Field(default=None, description="URL to the linked issue")


class TaskMetadata(BaseModel):
    """Metadata about the task"""

    task_type: str = Field(description="Type of task")
    difficulty: str = Field(description="Difficulty level")
    quality_score: int = Field(description="Quality rating 1-10")
    mined_at: str = Field(description="When this task was mined (ISO format)")
    miner_version: str = Field(default="1.0.0", description="Version of the mining script")


class TaskDefinition(BaseModel):
    """The actual task definition"""

    instruction: str = Field(description="Task instruction for the developer")
    context: str = Field(default="", description="Code context to help understand the task")
    hints: list[str] = Field(default_factory=list, description="Optional hints for solving")


class TaskTests(BaseModel):
    """Test specifications for the task"""

    fail_to_pass: list[str] = Field(
        default_factory=list,
        description="Tests that should fail before fix, pass after",
    )
    pass_to_pass: list[str] = Field(
        default_factory=list, description="Tests that should always pass"
    )
    test_files: list[str] = Field(default_factory=list, description="Test file paths to run")


class TaskSolution(BaseModel):
    """Solution-related information"""

    key_changes: list[str] = Field(default_factory=list, description="Key changes needed")
    potential_pitfalls: list[str] = Field(
        default_factory=list, description="Common mistakes to avoid"
    )
    reference_diff: str | None = Field(
        default=None, description="Reference solution diff (truncated)"
    )


class TaskFiles(BaseModel):
    """Files related to the task"""

    changed: list[str] = Field(default_factory=list, description="Files that were changed")
    test_changes: str | None = Field(default=None, description="Test file changes (truncated)")
    source_files: list[str] = Field(default_factory=list, description="Source files to modify")


class MinedTask(BaseModel):
    """
    Complete mined task ready for storage and use.

    This is the final output format that gets saved as JSON.
    """

    task_id: str = Field(description="Unique identifier for the task")
    source: TaskSource = Field(description="Source information")
    metadata: TaskMetadata = Field(description="Task metadata")
    task: TaskDefinition = Field(description="Task definition")
    tests: TaskTests = Field(description="Test specifications")
    solution: TaskSolution = Field(default_factory=TaskSolution, description="Solution hints")
    files: TaskFiles = Field(default_factory=TaskFiles, description="Related files")

    @classmethod
    def from_pr_and_schema(
        cls,
        pr_details: Any,  # PRDetails from mine_prs.py
        llm_schema: MinedTaskSchema,
    ) -> "MinedTask":
        """
        Create a MinedTask from PR details and LLM evaluation.

        Args:
            pr_details: PRDetails dataclass with PR information
            llm_schema: MinedTaskSchema from LLM evaluation

        Returns:
            Complete MinedTask ready for storage
        """
        task_id = f"{pr_details.repo_name.replace('/', '_')}_{pr_details.number}"

        return cls(
            task_id=task_id,
            source=TaskSource(
                type="github_pr",
                repo=pr_details.repo_name,
                pr_number=pr_details.number,
                pr_url=pr_details.url,
                merged_at=(pr_details.merged_at.isoformat() if pr_details.merged_at else None),
                issue_number=(
                    pr_details.linked_issue.get("number") if pr_details.linked_issue else None
                ),
                issue_url=(pr_details.linked_issue.get("url") if pr_details.linked_issue else None),
            ),
            metadata=TaskMetadata(
                task_type=llm_schema.task_type.value,
                difficulty=llm_schema.difficulty.value,
                quality_score=llm_schema.quality_score,
                mined_at=datetime.utcnow().isoformat(),
            ),
            task=TaskDefinition(
                instruction=llm_schema.instruction,
                context=llm_schema.context,
            ),
            tests=TaskTests(
                fail_to_pass=llm_schema.fail_to_pass,
                pass_to_pass=llm_schema.pass_to_pass,
            ),
            solution=TaskSolution(
                key_changes=llm_schema.key_changes,
                potential_pitfalls=llm_schema.potential_pitfalls,
                reference_diff=pr_details.code_diff[:10000] if pr_details.code_diff else None,
            ),
            files=TaskFiles(
                changed=pr_details.files_changed,
                test_changes=pr_details.test_diff[:5000] if pr_details.test_diff else None,
            ),
        )


class MiningStats(BaseModel):
    """Statistics from a mining run"""

    start_time: str = Field(description="When mining started")
    end_time: str | None = Field(default=None, description="When mining ended")
    repos_processed: int = Field(default=0, description="Number of repositories processed")
    prs_found: int = Field(default=0, description="Total PRs found")
    prs_with_tests: int = Field(default=0, description="PRs with test changes")
    prs_with_issues: int = Field(default=0, description="PRs with linked issues")
    tasks_created: int = Field(default=0, description="Tasks successfully created")
    tasks_rejected: int = Field(default=0, description="Tasks rejected by LLM")
    errors: list[str] = Field(default_factory=list, description="Error messages")

    def to_summary(self) -> str:
        """Generate a human-readable summary"""
        lines = [
            "Mining Statistics",
            "=" * 40,
            f"Repositories: {self.repos_processed}",
            f"PRs found: {self.prs_found}",
            f"PRs with tests: {self.prs_with_tests}",
            f"PRs with issues: {self.prs_with_issues}",
            f"Tasks created: {self.tasks_created}",
            f"Tasks rejected: {self.tasks_rejected}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


class RepoConfig(BaseModel):
    """Configuration for a single repository to mine"""

    owner: str = Field(description="Repository owner")
    repo: str = Field(description="Repository name")
    min_stars: int | None = Field(default=None, description="Minimum star count")
    reason: str | None = Field(default=None, description="Why this repo is included")

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


class MiningConfig(BaseModel):
    """Full configuration for a mining run"""

    high_priority: list[RepoConfig] = Field(default_factory=list)
    medium_priority: list[RepoConfig] = Field(default_factory=list)
    bayesian_stan: list[RepoConfig] = Field(default_factory=list)
    specialized: list[RepoConfig] = Field(default_factory=list)
    data_table: list[RepoConfig] = Field(default_factory=list)
    visualization: list[RepoConfig] = Field(default_factory=list)

    settings: dict[str, Any] = Field(
        default_factory=lambda: {
            "since_days": 30,
            "min_quality_score": 7,
            "max_prs_per_repo": 50,
            "require_test_changes": True,
            "require_linked_issue": True,
        }
    )

    def all_repos(self) -> list[RepoConfig]:
        """Get all repositories from all priority levels"""
        return (
            self.high_priority
            + self.medium_priority
            + self.bayesian_stan
            + self.specialized
            + self.data_table
            + self.visualization
        )
