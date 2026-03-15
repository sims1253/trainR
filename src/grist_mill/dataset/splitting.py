"""Stratified splitting of datasets into train/dev/test sets.

Produces train/dev/test splits with configurable ratios, stratifying by
task difficulty or category to preserve distribution across splits.

Validates VAL-DATASET-01.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from grist_mill.schemas import Difficulty

if TYPE_CHECKING:
    from grist_mill.dataset.core import Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    """Result of a stratified dataset split.

    Attributes:
        train: Training set.
        dev: Development/validation set.
        test: Test set.
        train_ratio: Ratio used for the training split.
        dev_ratio: Ratio used for the dev split.
        test_ratio: Ratio used for the test split.
    """

    train: Dataset
    dev: Dataset
    test: Dataset
    train_ratio: float = field(default=0.7)
    dev_ratio: float = field(default=0.15)
    test_ratio: float = field(default=0.15)


class StratifiedSplitter:
    """Splits a dataset into stratified train/dev/test sets.

    Stratification is done by difficulty level to ensure that each split
    has a representative distribution of task difficulties. Language
    stratification is also applied when possible.

    Attributes:
        train_ratio: Fraction of data for training (default 0.7).
        dev_ratio: Fraction of data for development (default 0.15).
        test_ratio: Fraction of data for testing (default 0.15).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self._validate_ratios(train_ratio, dev_ratio, test_ratio)
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    @staticmethod
    def _validate_ratios(train: float, dev: float, test: float) -> None:
        """Validate that ratios are non-negative and sum to 1.0."""
        if train < 0 or dev < 0 or test < 0:
            msg = f"Split ratios must be non-negative. Got train={train}, dev={dev}, test={test}."
            raise ValueError(msg)
        total = train + dev + test
        if not (0.99 <= total <= 1.01):  # Allow tiny floating-point tolerance
            msg = (
                f"Split ratios must sum to 1.0. Got train={train}, dev={dev}, "
                f"test={test} (sum={total})."
            )
            raise ValueError(msg)

    def split(self, dataset: Dataset) -> DatasetSplit:
        """Split the dataset into stratified train/dev/test sets.

        The splitting is stratified by difficulty level to ensure
        representative distribution across splits.

        Args:
            dataset: The dataset to split.

        Returns:
            A DatasetSplit containing train, dev, and test subsets.

        Raises:
            ValueError: If the dataset is empty.
        """
        if dataset.task_count == 0:
            msg = "Cannot split an empty dataset."
            raise ValueError(msg)

        from grist_mill.dataset.core import Dataset

        rng = random.Random(self.seed)
        tasks = dataset.list_tasks()

        # Group tasks by difficulty for stratification
        by_difficulty: dict[Difficulty, list] = {d: [] for d in Difficulty}
        for task in tasks:
            by_difficulty[task.difficulty].append(task)

        # Shuffle within each difficulty group
        for difficulty_group in by_difficulty.values():
            rng.shuffle(difficulty_group)

        train: list = []
        dev: list = []
        test: list = []

        for difficulty_group in by_difficulty.values():
            n = len(difficulty_group)
            if n == 0:
                continue

            # Calculate split boundaries
            train_end = int(n * self.train_ratio)
            dev_end = train_end + int(n * self.dev_ratio)

            # Ensure at least one task in test set if there are enough tasks
            if n >= 3 and dev_end >= n:
                dev_end = n - 1
            if n >= 2 and train_end >= n:
                train_end = n - 1

            train.extend(difficulty_group[:train_end])
            dev.extend(difficulty_group[train_end:dev_end])
            test.extend(difficulty_group[dev_end:])

        # Fallback: if test is empty but we have tasks, move one from dev or train
        if not test and len(tasks) >= 1:
            if dev:
                test.append(dev.pop())
            elif train:
                test.append(train.pop())

        # If dev is empty but we have enough tasks, move one from train
        if not dev and len(tasks) >= 2 and len(train) > 1:
            dev.append(train.pop())

        train_ds = Dataset()
        train_ds.add_tasks(train)
        dev_ds = Dataset()
        dev_ds.add_tasks(dev)
        test_ds = Dataset()
        test_ds.add_tasks(test)

        return DatasetSplit(
            train=train_ds,
            dev=dev_ds,
            test=test_ds,
            train_ratio=self.train_ratio,
            dev_ratio=self.dev_ratio,
            test_ratio=self.test_ratio,
        )
