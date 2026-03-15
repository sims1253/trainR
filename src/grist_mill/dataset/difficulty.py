"""Difficulty estimation for tasks.

Estimates task difficulty based on heuristics derived from task metadata:
prompt length, timeout, test command complexity, dependency count, etc.
Produces labels: EASY, MEDIUM, HARD.

Validates VAL-DATASET-04.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from grist_mill.schemas import Difficulty, Task

if TYPE_CHECKING:
    from grist_mill.dataset.core import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic scoring weights
# ---------------------------------------------------------------------------

# Thresholds for difficulty classification
_EASY_THRESHOLD = 3
_HARD_THRESHOLD = 7


def _score_prompt_complexity(prompt: str) -> float:
    """Score prompt complexity based on length and keyword density."""
    score = 0.0

    # Length component (longer prompts tend to be harder)
    word_count = len(prompt.split())
    if word_count > 50:
        score += 2.0
    elif word_count > 25:
        score += 1.0
    elif word_count > 10:
        score += 0.5

    # Complexity keywords (indicate harder tasks)
    complexity_keywords = [
        "race condition",
        "concurrent",
        "distributed",
        "synchronization",
        "deadlock",
        "architecture",
        "microservice",
        "backward compatibility",
        "legacy",
        "edge case",
        "partial failure",
        "high load",
        "multiple service",
        "complex",
        "refactor",
        "redesign",
        "error handling",
        "malformed",
        "modular",
        "abstract",
        "optimize",
    ]
    prompt_lower = prompt.lower()
    for keyword in complexity_keywords:
        if keyword in prompt_lower:
            score += 1.0

    # Very short/simple prompts indicate easy tasks
    simple_keywords = [
        "fix the typo",
        "fix typo",
        "correct the spelling",
        "add a comment",
        "fix the naming",
        "rename",
    ]
    for keyword in simple_keywords:
        if keyword in prompt_lower:
            score -= 1.5

    return max(score, 0.0)


def _score_timeout(timeout: int) -> float:
    """Score based on timeout (longer timeout suggests harder task)."""
    if timeout <= 30:
        return 0.0
    elif timeout <= 60:
        return 0.5
    elif timeout <= 120:
        return 1.0
    elif timeout <= 300:
        return 1.5
    else:
        return 2.5


def _score_test_complexity(test_command: str) -> float:
    """Score based on test command complexity."""
    score = 0.0

    # Multiple test files
    test_files = re.findall(r"[\w./-]+\.(?:py|r|R|ts|js|test)", test_command)
    if len(test_files) > 3:
        score += 2.0
    elif len(test_files) > 1:
        score += 1.0

    # Complex test flags
    complex_flags = ["-x", "-v", "--verbose", "-k", "--cov", "--tb=long"]
    flag_count = sum(1 for flag in complex_flags if flag in test_command)
    score += flag_count * 0.3

    # Test directory patterns
    if re.search(r"tests?[/_]", test_command):
        score += 0.5

    return score


def _score_dependencies(task: Task) -> float:
    """Score based on number of dependencies."""
    dep_count = len(task.dependencies)
    if dep_count >= 5:
        return 2.0
    elif dep_count >= 3:
        return 1.0
    elif dep_count >= 1:
        return 0.5
    return 0.0


def _score_constraints(task: Task) -> float:
    """Score based on constraints."""
    constraint_count = len(task.constraints)
    if constraint_count >= 3:
        return 1.5
    elif constraint_count >= 1:
        return 0.5
    return 0.0


def _compute_overall_score(task: Task) -> float:
    """Compute the overall difficulty score for a task."""
    score = 0.0
    score += _score_prompt_complexity(task.prompt)
    score += _score_timeout(task.timeout)
    score += _score_test_complexity(task.test_command)
    score += _score_dependencies(task)
    score += _score_constraints(task)
    return score


def _score_to_difficulty(score: float) -> Difficulty:
    """Convert a numerical score to a Difficulty enum."""
    if score <= _EASY_THRESHOLD:
        return Difficulty.EASY
    elif score <= _HARD_THRESHOLD:
        return Difficulty.MEDIUM
    else:
        return Difficulty.HARD


# ---------------------------------------------------------------------------
# DifficultyEstimator
# ---------------------------------------------------------------------------


class DifficultyEstimator:
    """Estimates difficulty for tasks based on heuristic scoring.

    Uses a combination of prompt complexity, timeout, test command
    complexity, dependency count, and constraint count to produce
    difficulty labels: EASY, MEDIUM, HARD.

    Attributes:
        overwrite: If True (default), overwrite existing difficulty labels.
                   If False, preserve existing labels.
    """

    def __init__(self, overwrite: bool = True) -> None:
        self.overwrite = overwrite

    def estimate(self, task: Task) -> Difficulty:
        """Estimate the difficulty of a single task.

        Args:
            task: The task to estimate difficulty for.

        Returns:
            The estimated Difficulty.
        """
        if not self.overwrite:
            return task.difficulty

        score = _compute_overall_score(task)
        return _score_to_difficulty(score)

    def estimate_dataset(self, dataset: Dataset) -> None:
        """Estimate difficulty for all tasks in the dataset.

        Modifies tasks in place.

        Args:
            dataset: The dataset whose tasks should be estimated.
        """
        count = 0
        for task in dataset.list_tasks():
            if self.overwrite or task.difficulty == Difficulty.EASY:
                task.difficulty = self.estimate(task)
                count += 1

        logger.info("Estimated difficulty for %d tasks in dataset.", count)
