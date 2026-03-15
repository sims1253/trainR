"""Core Dataset container with ID uniqueness enforcement.

The Dataset is the primary container for tasks. It enforces task ID uniqueness
across the entire dataset, as required by VAL-TASKFMT-04.
"""

from __future__ import annotations

import logging

from grist_mill.schemas import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DuplicateEntry(Exception):  # noqa: N818
    """Raised when a task with a duplicate ID is added to the dataset."""

    pass


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset:
    """A collection of tasks with unique ID enforcement.

    The Dataset is the core container for benchmark tasks. It enforces that
    every task ID is unique within the dataset (VAL-TASKFMT-04). It provides
    filtering, iteration, and conversion utilities for working with task
    collections.

    Attributes:
        _tasks: Ordered mapping of task ID to Task.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        """Add a single task to the dataset.

        Args:
            task: The task to add.

        Raises:
            DuplicateEntry: If a task with the same ID already exists.
        """
        if task.id in self._tasks:
            msg = f"Duplicate task ID: '{task.id}'. Each task must have a unique ID."
            raise DuplicateEntry(msg)
        self._tasks[task.id] = task

    def add_tasks(self, tasks: list[Task]) -> None:
        """Add multiple tasks to the dataset.

        Args:
            tasks: List of tasks to add.

        Raises:
            DuplicateEntry: If any task has a duplicate ID.
        """
        for task in tasks:
            self.add_task(task)

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the dataset by ID.

        Does nothing if the task ID doesn't exist.

        Args:
            task_id: ID of the task to remove.
        """
        self._tasks.pop(task_id, None)

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: ID of the task to retrieve.

        Returns:
            The Task, or None if not found.
        """
        return self._tasks.get(task_id)

    def has_task(self, task_id: str) -> bool:
        """Check whether a task exists in the dataset.

        Args:
            task_id: ID of the task to check.

        Returns:
            True if the task exists, False otherwise.
        """
        return task_id in self._tasks

    @property
    def task_count(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self._tasks)

    @property
    def task_ids(self) -> set[str]:
        """Return the set of all task IDs in the dataset."""
        return set(self._tasks.keys())

    def list_tasks(self) -> list[Task]:
        """Return all tasks as a list.

        Returns:
            List of all tasks in the dataset.
        """
        return list(self._tasks.values())

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_language(self, language: str) -> Dataset:
        """Create a new Dataset containing only tasks matching the given language.

        Args:
            language: Language to filter by (e.g., 'python', 'r').

        Returns:
            A new Dataset with filtered tasks.
        """
        result = Dataset()
        for task in self._tasks.values():
            if task.language == language:
                result.add_task(task)
        return result

    def filter_by_difficulty(self, difficulty: object) -> Dataset:
        """Create a new Dataset containing only tasks matching the given difficulty.

        Args:
            difficulty: Difficulty enum value to filter by.

        Returns:
            A new Dataset with filtered tasks.
        """
        result = Dataset()
        for task in self._tasks.values():
            if task.difficulty == difficulty:
                result.add_task(task)
        return result

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_task_list(self) -> list[Task]:
        """Return all tasks as a list (alias for list_tasks)."""
        return self.list_tasks()

    @classmethod
    def from_task_list(cls, tasks: list[Task]) -> Dataset:
        """Create a Dataset from a list of tasks.

        Args:
            tasks: List of tasks to include.

        Returns:
            A new Dataset containing the given tasks.

        Raises:
            DuplicateEntry: If duplicate task IDs are found.
        """
        ds = cls()
        ds.add_tasks(tasks)
        return ds

    def __len__(self) -> int:
        return self.task_count

    def __iter__(self):
        return iter(self._tasks.values())

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def __repr__(self) -> str:
        return f"Dataset(tasks={self.task_count})"
