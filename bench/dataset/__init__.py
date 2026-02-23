"""Dataset management module for benchmark tasks.

This module provides tools for:
- Migrating legacy tasks to canonical v1 format
- Task validation and normalization
- Dataset integrity checking
"""

from bench.dataset.migrate import (
    MigrationCounter,
    MigrationReport,
    MigrationWarning,
    TaskFormat,
    detect_task_format,
    migrate_task,
    migrate_tasks_directory,
)

__all__ = [
    "MigrationCounter",
    "MigrationReport",
    "MigrationWarning",
    "TaskFormat",
    "detect_task_format",
    "migrate_task",
    "migrate_tasks_directory",
]
