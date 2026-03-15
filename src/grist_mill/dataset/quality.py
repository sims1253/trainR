"""Dataset quality reports.

Generates quality reports summarizing dataset health: task counts by
language, difficulty, and category; flags potential issues such as empty
descriptions, missing setup commands, etc.

Validates VAL-DATASET-06.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grist_mill.dataset.core import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetQualityIssue:
    """A potential issue found in the dataset.

    Attributes:
        severity: 'warning' or 'info'.
        description: Human-readable description of the issue.
        task_ids: Task IDs affected by this issue.
    """

    severity: str
    description: str
    task_ids: list[str] = field(default_factory=list)


@dataclass
class DatasetQualityReport:
    """Quality report summarizing dataset health.

    Attributes:
        total_tasks: Total number of tasks in the dataset.
        by_language: Count of tasks per language.
        by_difficulty: Count of tasks per difficulty level.
        by_category: Count of tasks per category (derived from constraints).
        issues: List of quality issues found.
    """

    total_tasks: int = 0
    by_language: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    issues: list[DatasetQualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the report to a dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "by_language": self.by_language,
            "by_difficulty": self.by_difficulty,
            "by_category": self.by_category,
            "issues": [
                {
                    "severity": issue.severity,
                    "description": issue.description,
                    "task_ids": issue.task_ids,
                }
                for issue in self.issues
            ],
        }

    @classmethod
    def generate(cls, dataset: Dataset) -> DatasetQualityReport:
        """Generate a quality report for the given dataset.

        Analyzes task distribution by language, difficulty, and category.
        Flags potential issues such as empty descriptions, missing setup
        commands, and imbalanced distributions.

        Args:
            dataset: The dataset to analyze.

        Returns:
            A DatasetQualityReport with all analysis results.
        """
        tasks = dataset.list_tasks()
        report = cls(total_tasks=len(tasks))

        # Count by language
        lang_counter: Counter[str] = Counter()
        diff_counter: Counter[str] = Counter()
        category_counter: Counter[str] = Counter()

        empty_prompt_ids: list[str] = []
        short_prompt_ids: list[str] = []
        no_setup_ids: list[str] = []
        no_dependencies_ids: list[str] = []

        for task in tasks:
            lang_counter[task.language] += 1
            diff_counter[task.difficulty.value] += 1
            for constraint in task.constraints:
                category_counter[constraint] += 1

            # Check for empty/whitespace-only prompts
            if not task.prompt.strip():
                empty_prompt_ids.append(task.id)

            # Check for very short prompts (likely insufficient detail)
            elif len(task.prompt.split()) < 4:
                short_prompt_ids.append(task.id)

            # Check for missing setup commands
            if task.setup_command is None:
                no_setup_ids.append(task.id)

            # Check for tasks without dependencies
            if not task.dependencies:
                no_dependencies_ids.append(task.id)

        report.by_language = dict(lang_counter)
        report.by_difficulty = dict(diff_counter)
        report.by_category = dict(category_counter)

        # Add issues
        if short_prompt_ids:
            report.issues.append(
                DatasetQualityIssue(
                    severity="info",
                    description=(
                        f"{len(short_prompt_ids)} task(s) have very short prompts "
                        f"(<4 words): {', '.join(short_prompt_ids[:10])}. "
                        "Consider providing more context for the agent."
                    ),
                    task_ids=short_prompt_ids,
                )
            )

        if empty_prompt_ids:
            report.issues.append(
                DatasetQualityIssue(
                    severity="warning",
                    description=(
                        f"{len(empty_prompt_ids)} task(s) have empty or whitespace-only "
                        f"prompts: {', '.join(empty_prompt_ids[:10])}"
                    ),
                    task_ids=empty_prompt_ids,
                )
            )

        if no_setup_ids:
            report.issues.append(
                DatasetQualityIssue(
                    severity="info",
                    description=(
                        f"{len(no_setup_ids)} task(s) have no setup_command. "
                        "These tasks may fail if dependencies need to be installed."
                    ),
                    task_ids=no_setup_ids[:20],  # Limit to avoid huge lists
                )
            )

        if no_dependencies_ids:
            report.issues.append(
                DatasetQualityIssue(
                    severity="info",
                    description=(
                        f"{len(no_dependencies_ids)} task(s) have no dependencies declared. "
                        "These tasks may rely on the default environment only."
                    ),
                    task_ids=no_dependencies_ids[:20],
                )
            )

        # Check for difficulty distribution imbalance
        if len(diff_counter) == 1:
            report.issues.append(
                DatasetQualityIssue(
                    severity="warning",
                    description=(
                        f"All tasks have the same difficulty ({next(iter(diff_counter.keys()))}). "
                        "Consider adding tasks with varied difficulty levels."
                    ),
                )
            )

        # Check for single-language dataset
        if len(lang_counter) == 1:
            report.issues.append(
                DatasetQualityIssue(
                    severity="info",
                    description=(
                        f"Dataset contains only one language ({next(iter(lang_counter.keys()))}). "
                        "Consider adding multi-language tasks."
                    ),
                )
            )

        return report
