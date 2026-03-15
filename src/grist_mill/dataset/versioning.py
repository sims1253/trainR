"""Dataset versioning with immutable snapshots.

Once a dataset is versioned, its contents are immutable. Any modification
to the dataset creates a new version. Versions are stored in a versioning
store that tracks the full history.

Validates VAL-DATASET-02.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grist_mill.dataset.core import Dataset
    from grist_mill.schemas import Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetVersion:
    """An immutable snapshot of a dataset at a point in time.

    Attributes:
        version_number: Sequential version number.
        description: Human-readable description of this version.
        tasks: Deep copy of tasks in this version.
        created_at: Timestamp when this version was created.
        task_count: Number of tasks in this version.
    """

    version_number: int
    description: str
    tasks: list[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_count: int = 0

    def __post_init__(self) -> None:
        # Since we're frozen, we can't set in __init__ directly after dataclass init.
        # Use object.__setattr__ for the computed field.
        if self.task_count == 0 and self.tasks:
            object.__setattr__(self, "task_count", len(self.tasks))


class DatasetVersioningStore:
    """Store for immutable dataset versions.

    Manages version creation, retrieval, and listing. Each version is a
    deep copy (immutable snapshot) of the dataset at the time of versioning.
    """

    def __init__(self) -> None:
        self._versions: dict[int, DatasetVersion] = {}
        self._next_version: int = 1

    def create_version(
        self,
        dataset: Dataset,
        description: str = "",
    ) -> DatasetVersion:
        """Create a new immutable version of the dataset.

        The version is a deep copy of the dataset's tasks. Subsequent
        modifications to the dataset will not affect the version.

        Args:
            dataset: The Dataset object to version.
            description: Human-readable description.

        Returns:
            The created DatasetVersion.
        """
        # Deep copy all tasks to ensure immutability
        tasks_copy = copy.deepcopy(dataset.to_task_list())

        version = DatasetVersion(
            version_number=self._next_version,
            description=description,
            tasks=tasks_copy,
            task_count=len(tasks_copy),
        )

        self._versions[self._next_version] = version
        self._next_version += 1

        logger.info(
            "Created dataset version %d with %d tasks: %s",
            version.version_number,
            version.task_count,
            description,
        )

        return version

    def get_version(self, version_number: int) -> DatasetVersion | None:
        """Get a specific version by its number.

        Args:
            version_number: The version number to retrieve.

        Returns:
            The DatasetVersion, or None if not found.
        """
        return self._versions.get(version_number)

    def list_versions(self) -> list[DatasetVersion]:
        """List all versions in order.

        Returns:
            List of all DatasetVersion objects, ordered by version number.
        """
        return [self._versions[k] for k in sorted(self._versions.keys())]

    @property
    def latest_version(self) -> DatasetVersion | None:
        """Get the most recent version, or None if no versions exist."""
        if not self._versions:
            return None
        max_version = max(self._versions.keys())
        return self._versions[max_version]
