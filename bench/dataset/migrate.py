"""Migration module for converting legacy tasks to canonical v1 format.

This module provides:
- Dual-read/single-write path: read legacy + v1, normalize to TaskV1 immediately
- Migration observability: counters for "legacy parsed", "adapter fallback used", etc.
- Explicit warnings over silent coercion
- Validation of migrated tasks
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bench.schema.v1 import TaskV1


class TaskFormat(str, Enum):
    """Detected format of legacy task files."""

    MINED = "mined"  # GitHub PR mined tasks with source, problem_statement, patch, etc.
    TESTING = "testing"  # TestingTask format (dev/held_out/train)
    V1 = "v1"  # Already in v1 format
    UNKNOWN = "unknown"  # Cannot determine format


@dataclass
class MigrationWarning:
    """A warning generated during task migration."""

    task_id: str
    field: str
    message: str
    severity: str = "warning"  # warning, info, error


@dataclass
class MigrationCounter:
    """Counters for migration observability."""

    total_processed: int = 0
    legacy_parsed: int = 0
    v1_already: int = 0
    migrated_successfully: int = 0
    failed: int = 0
    adapter_fallback_used: int = 0

    # Validation failures by field
    validation_failures: dict[str, int] = field(default_factory=dict)

    # Warnings by type
    warnings_by_type: dict[str, int] = field(default_factory=dict)

    # Per-format counts
    mined_tasks: int = 0
    testing_tasks: int = 0
    unknown_format: int = 0


@dataclass
class MigrationReport:
    """Complete migration report with statistics and details."""

    timestamp: str
    input_directory: str
    output_directory: str

    # Counts
    total_tasks: int = 0
    migrated_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0

    # Detailed failure info
    failures: list[dict[str, Any]] = field(default_factory=list)

    # Warnings
    warnings: list[dict[str, Any]] = field(default_factory=list)

    # Counters for observability
    counters: MigrationCounter = field(default_factory=MigrationCounter)

    # Per-split statistics
    split_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "input_directory": self.input_directory,
            "output_directory": self.output_directory,
            "summary": {
                "total_tasks": self.total_tasks,
                "migrated_count": self.migrated_count,
                "failed_count": self.failed_count,
                "skipped_count": self.skipped_count,
                "success_rate": (
                    self.migrated_count / self.total_tasks * 100 if self.total_tasks > 0 else 0
                ),
            },
            "counters": {
                "total_processed": self.counters.total_processed,
                "legacy_parsed": self.counters.legacy_parsed,
                "v1_already": self.counters.v1_already,
                "migrated_successfully": self.counters.migrated_successfully,
                "failed": self.counters.failed,
                "adapter_fallback_used": self.counters.adapter_fallback_used,
                "mined_tasks": self.counters.mined_tasks,
                "testing_tasks": self.counters.testing_tasks,
                "unknown_format": self.counters.unknown_format,
                "validation_failures": self.counters.validation_failures,
                "warnings_by_type": self.counters.warnings_by_type,
            },
            "split_stats": self.split_stats,
            "failures": self.failures,
            "warnings": self.warnings,
        }


def detect_task_format(data: dict[str, Any]) -> TaskFormat:
    """
    Detect the format of a task dictionary.

    Args:
        data: Task data dictionary

    Returns:
        Detected TaskFormat
    """
    # Check for v1 format (has schema_version starting with "1.")
    if "schema_version" in data and str(data["schema_version"]).startswith("1."):
        return TaskFormat.V1

    # Check for mined task format (has source, problem_statement, patch)
    if all(key in data for key in ["source", "problem_statement", "patch"]):
        return TaskFormat.MINED

    # Check for testing task format (has source_package, source_file, test_type)
    if all(key in data for key in ["source_package", "source_file", "instruction"]) and (
        "test_type" in data or "patterns" in data
    ):
        return TaskFormat.TESTING

    # Check if it's a testing task with minimal fields
    if "source_package" in data and "instruction" in data and "task_id" in data:
        return TaskFormat.TESTING

    return TaskFormat.UNKNOWN


def migrate_task(
    data: dict[str, Any],
    source_path: Path | None = None,
    counters: MigrationCounter | None = None,
    warnings_list: list[MigrationWarning] | None = None,
) -> tuple[TaskV1 | None, list[MigrationWarning]]:
    """
    Migrate a single task to v1 format.

    Args:
        data: Legacy task data dictionary
        source_path: Path to source file (for error messages)
        counters: MigrationCounter to update (optional)
        warnings_list: List to append warnings to (optional)

    Returns:
        Tuple of (migrated TaskV1 or None if failed, list of warnings)
    """
    local_warnings: list[MigrationWarning] = []
    task_id = data.get("task_id", "unknown")

    def add_warning(field: str, message: str, severity: str = "warning") -> None:
        warning = MigrationWarning(
            task_id=task_id,
            field=field,
            message=message,
            severity=severity,
        )
        local_warnings.append(warning)
        if warnings_list is not None:
            warnings_list.append(warning)

    if counters is not None:
        counters.total_processed += 1

    # Detect format
    task_format = detect_task_format(data)

    if task_format == TaskFormat.V1:
        # Already in v1 format, just validate
        if counters is not None:
            counters.v1_already += 1
        try:
            task = TaskV1.model_validate(data)
            if counters is not None:
                counters.migrated_successfully += 1
            return task, local_warnings
        except ValidationError as e:
            add_warning("schema", f"V1 validation failed: {e}")
            if counters is not None:
                counters.failed += 1
                _record_validation_failures(counters, e)
            return None, local_warnings

    if task_format == TaskFormat.UNKNOWN:
        add_warning("format", "Cannot detect task format, attempting fallback")
        if counters is not None:
            counters.unknown_format += 1
            counters.adapter_fallback_used += 1

    # Record format type
    if counters is not None:
        counters.legacy_parsed += 1
        if task_format == TaskFormat.MINED:
            counters.mined_tasks += 1
        elif task_format == TaskFormat.TESTING:
            counters.testing_tasks += 1

    # Check for missing/optional fields and emit warnings
    _check_missing_fields(data, task_format, add_warning)

    # Attempt migration using appropriate adapter
    task: TaskV1 | None = None

    try:
        if task_format == TaskFormat.MINED:
            task = TaskV1.from_legacy_mined_task(data)
            # Fix for mined tasks where instruction/context are at top level
            # (not under "task" key as the adapter expects)
            if task and not task.instruction and data.get("instruction"):
                task = task.model_copy(update={"instruction": data["instruction"]})
            if task and not task.context and data.get("context"):
                task = task.model_copy(update={"context": data["context"]})
            if task and not task.hints and data.get("hints"):
                task = task.model_copy(update={"hints": data["hints"]})
            # Extract solution info from metadata if present
            metadata = data.get("metadata", {})
            if task and metadata:
                from bench.schema.v1 import TaskSolutionV1

                key_changes = metadata.get("key_changes", [])
                potential_pitfalls = metadata.get("potential_pitfalls", [])
                if key_changes or potential_pitfalls:
                    task = task.model_copy(
                        update={
                            "solution": TaskSolutionV1(
                                key_changes=key_changes,
                                potential_pitfalls=potential_pitfalls,
                                reference_diff=data.get("patch"),
                            )
                        }
                    )
        elif task_format == TaskFormat.TESTING:
            task = TaskV1.from_legacy_testing_task(data)
        else:
            # Fallback: try both adapters
            add_warning("format", "Unknown format, attempting mined task adapter")
            if counters is not None:
                counters.adapter_fallback_used += 1

            try:
                task = TaskV1.from_legacy_mined_task(data)
                # Apply same fix for top-level instruction/context
                if task and not task.instruction and data.get("instruction"):
                    task = task.model_copy(update={"instruction": data["instruction"]})
                if task and not task.context and data.get("context"):
                    task = task.model_copy(update={"context": data["context"]})
            except Exception:
                add_warning("format", "Mined adapter failed, trying testing task adapter")
                try:
                    task = TaskV1.from_legacy_testing_task(data)
                except Exception as e:
                    add_warning("format", f"All adapters failed: {e}")
                    if counters is not None:
                        counters.failed += 1
                    return None, local_warnings

        # Validate the migrated task
        try:
            TaskV1.model_validate(task.model_dump())
        except ValidationError as e:
            add_warning("validation", f"Post-migration validation failed: {e}")
            if counters is not None:
                _record_validation_failures(counters, e)

        if counters is not None:
            counters.migrated_successfully += 1

        return task, local_warnings

    except Exception as e:
        add_warning("migration", f"Migration failed with error: {e}")
        if counters is not None:
            counters.failed += 1
        return None, local_warnings


def _check_missing_fields(
    data: dict[str, Any],
    task_format: TaskFormat,
    add_warning: Any,
) -> None:
    """Check for missing/optional fields and emit explicit warnings."""

    # Common required fields
    if "task_id" not in data or not data["task_id"]:
        add_warning("task_id", "Missing or empty task_id", severity="error")

    if "instruction" not in data or not data["instruction"]:
        add_warning("instruction", "Missing or empty instruction", severity="error")

    if task_format == TaskFormat.MINED:
        # Mined task specific checks
        if "source" not in data:
            add_warning("source", "Missing source information")
        elif not data["source"].get("repo"):
            add_warning("source.repo", "Missing repository name in source")

        if "tests" not in data:
            add_warning("tests", "Missing tests specification")
        else:
            if not data["tests"].get("fail_to_pass"):
                add_warning(
                    "tests.fail_to_pass", "No fail_to_pass tests specified", severity="info"
                )

        if "metadata" not in data:
            add_warning("metadata", "Missing metadata section")
        elif not data["metadata"].get("task_type"):
            add_warning("metadata.task_type", "Missing task_type in metadata")

    elif task_format == TaskFormat.TESTING:
        # Testing task specific checks
        if not data.get("source_package"):
            add_warning("source_package", "Missing source package name")

        if not data.get("source_file"):
            add_warning("source_file", "Missing source file")

        if not data.get("dependencies"):
            add_warning("dependencies", "No dependencies specified", severity="info")


def _record_validation_failures(counters: MigrationCounter, error: ValidationError) -> None:
    """Record validation failures by field."""
    for err in error.errors():
        field_path = ".".join(str(loc) for loc in err.get("loc", ["unknown"]))
        counters.validation_failures[field_path] = (
            counters.validation_failures.get(field_path, 0) + 1
        )


def migrate_tasks_directory(
    input_dir: Path,
    output_dir: Path,
    counters: MigrationCounter | None = None,
    warnings_list: list[MigrationWarning] | None = None,
) -> MigrationReport:
    """
    Migrate all tasks from input directory to output directory.

    Preserves directory structure and generates a migration report.

    Args:
        input_dir: Directory containing legacy tasks
        output_dir: Directory to write migrated tasks
        counters: MigrationCounter to use (optional, creates new if None)
        warnings_list: List to collect warnings (optional)

    Returns:
        MigrationReport with statistics and details
    """
    if counters is None:
        counters = MigrationCounter()
    if warnings_list is None:
        warnings_list = []

    report = MigrationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        counters=counters,
    )

    # Find all JSON files recursively
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    json_files = list(input_path.rglob("*.json"))
    report.total_tasks = len(json_files)

    # Track per-split statistics
    split_counts: dict[str, dict[str, int]] = {}

    for json_file in json_files:
        # Determine relative path and output location
        rel_path = json_file.relative_to(input_path)
        output_file = output_path / rel_path

        # Determine split from path
        parts = rel_path.parts
        split = parts[0] if len(parts) > 1 else "unknown"

        if split not in split_counts:
            split_counts[split] = {"total": 0, "migrated": 0, "failed": 0}
        split_counts[split]["total"] += 1

        try:
            # Read legacy task
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Migrate
            task, task_warnings = migrate_task(
                data,
                source_path=json_file,
                counters=counters,
                warnings_list=warnings_list,
            )

            # Record warnings
            for w in task_warnings:
                report.warnings.append(
                    {
                        "task_id": w.task_id,
                        "field": w.field,
                        "message": w.message,
                        "severity": w.severity,
                        "source_file": str(json_file),
                    }
                )
                # Track warning types
                counters.warnings_by_type[w.field] = counters.warnings_by_type.get(w.field, 0) + 1

            if task is not None:
                # Write migrated task
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(task.model_dump(mode="json"), f, indent=2)

                report.migrated_count += 1
                split_counts[split]["migrated"] += 1
            else:
                report.failed_count += 1
                split_counts[split]["failed"] += 1
                report.failures.append(
                    {
                        "task_id": data.get("task_id", "unknown"),
                        "source_file": str(json_file),
                        "reason": "Migration returned None",
                        "format": detect_task_format(data).value,
                    }
                )

        except json.JSONDecodeError as e:
            report.failed_count += 1
            split_counts[split]["failed"] += 1
            report.failures.append(
                {
                    "task_id": "unknown",
                    "source_file": str(json_file),
                    "reason": f"JSON decode error: {e}",
                    "format": "invalid_json",
                }
            )
            counters.failed += 1
        except Exception as e:
            report.failed_count += 1
            split_counts[split]["failed"] += 1
            report.failures.append(
                {
                    "task_id": "unknown",
                    "source_file": str(json_file),
                    "reason": f"Unexpected error: {e}",
                    "format": "unknown",
                }
            )
            counters.failed += 1

    report.split_stats = split_counts
    return report
