"""Dataset decontamination: duplicate and near-duplicate detection.

Detects and flags duplicate or near-duplicate tasks based on prompt
similarity. Uses a simple character n-gram Jaccard similarity approach
that requires no external dependencies.

Validates VAL-DATASET-03.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grist_mill.dataset.core import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetDecontaminationResult:
    """Result of a decontamination run.

    Attributes:
        duplicate_pairs: List of (task_id_a, task_id_b) pairs that are duplicates.
        flagged_task_ids: Set of task IDs that are involved in any duplicate.
        total_tasks_checked: Total number of tasks that were compared.
    """

    duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)
    flagged_task_ids: set[str] = field(default_factory=set)
    total_tasks_checked: int = 0


class DatasetDecontamination:
    """Detects duplicate and near-duplicate tasks in a dataset.

    Uses character n-gram Jaccard similarity on the task prompts to identify
    duplicates. The similarity threshold determines how similar two prompts
    must be to be flagged as duplicates.

    Attributes:
        threshold: Similarity threshold (0.0 to 1.0). Lower values are more
                   strict (fewer matches). Default is 0.8.
        ngram_size: Size of character n-grams for comparison. Default is 3.
    """

    def __init__(self, threshold: float = 0.8, ngram_size: int = 3) -> None:
        self.threshold = threshold
        self.ngram_size = ngram_size

    def run(self, dataset: Dataset) -> DatasetDecontaminationResult:
        """Run decontamination on the dataset.

        Compares all pairs of task prompts and flags those with similarity
        above the threshold.

        Args:
            dataset: The dataset to decontaminate.

        Returns:
            A DatasetDecontaminationResult with duplicate pairs and flagged IDs.
        """
        tasks = dataset.list_tasks()
        result = DatasetDecontaminationResult(total_tasks_checked=len(tasks))

        if len(tasks) < 2:
            return result

        # Pre-compute n-grams for all tasks
        ngrams_list: list[frozenset[str]] = []
        for task in tasks:
            ngrams = self._compute_ngrams(task.prompt)
            ngrams_list.append(ngrams)

        # Compare all pairs
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                similarity = self._jaccard_similarity(ngrams_list[i], ngrams_list[j])
                if similarity >= self.threshold:
                    pair = (tasks[i].id, tasks[j].id)
                    result.duplicate_pairs.append(pair)
                    result.flagged_task_ids.add(tasks[i].id)
                    result.flagged_task_ids.add(tasks[j].id)

        logger.info(
            "Decontamination complete. Checked %d tasks, found %d duplicate pairs.",
            result.total_tasks_checked,
            len(result.duplicate_pairs),
        )

        return result

    def _compute_ngrams(self, text: str) -> frozenset[str]:
        """Compute character n-grams from text.

        Args:
            text: The input text.

        Returns:
            A frozenset of n-gram strings.
        """
        normalized = text.lower().strip()
        if len(normalized) < self.ngram_size:
            return frozenset({normalized})
        return frozenset(
            normalized[i : i + self.ngram_size]
            for i in range(len(normalized) - self.ngram_size + 1)
        )

    @staticmethod
    def _jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
        """Compute Jaccard similarity between two sets.

        Args:
            set_a: First set.
            set_b: Second set.

        Returns:
            Jaccard similarity (0.0 to 1.0).
        """
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union
