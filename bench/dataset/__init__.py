"""Dataset management module for benchmark tasks.

This module provides tools for:
- Migrating legacy tasks to canonical v1 format
- Task validation and normalization
- Dataset integrity checking
- Dataset splitting and lifecycle management
- Decontamination analysis
"""

from bench.dataset.decontam import (
    DecontaminationReport,
    analyze_decontamination,
    check_test_contamination,
    compute_content_fingerprint,
    compute_task_fingerprint,
    deduplicate_tasks,
    find_cross_split_overlap,
    find_duplicates,
)
from bench.dataset.manager import (
    DatasetConfig,
    DatasetManifest,
    SplitStats,
    ValidationReport,
    ValidationResult,
    compute_dataset_fingerprint,
    export_dataset,
    load_tasks_from_directory,
    run_export_command,
    run_split_command,
    run_validate_command,
    split_dataset,
    validate_dataset,
    validate_task_file,
)
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
    # Manager
    "DatasetConfig",
    "DatasetManifest",
    # Decontamination
    "DecontaminationReport",
    # Migration
    "MigrationCounter",
    "MigrationReport",
    "MigrationWarning",
    "SplitStats",
    "TaskFormat",
    "ValidationReport",
    "ValidationResult",
    "analyze_decontamination",
    "check_test_contamination",
    "compute_content_fingerprint",
    "compute_dataset_fingerprint",
    "compute_task_fingerprint",
    "deduplicate_tasks",
    "detect_task_format",
    "export_dataset",
    "find_cross_split_overlap",
    "find_duplicates",
    "load_tasks_from_directory",
    "migrate_task",
    "migrate_tasks_directory",
    "run_export_command",
    "run_split_command",
    "run_validate_command",
    "split_dataset",
    "validate_dataset",
    "validate_task_file",
]
