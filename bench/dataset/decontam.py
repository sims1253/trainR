"""Decontamination utilities for dataset management.

This module provides tools for:
- Detecting and removing duplicate tasks
- Finding overlap between splits
- Computing content fingerprints
- Cross-validation decontamination
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.schema.v1 import TaskV1


@dataclass
class DecontaminationReport:
    """Report from decontamination analysis."""

    total_tasks: int = 0
    duplicate_groups: int = 0
    duplicates_found: int = 0
    cross_split_overlaps: int = 0
    contaminated_tasks: list[str] = field(default_factory=list)
    overlap_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tasks": self.total_tasks,
            "duplicate_groups": self.duplicate_groups,
            "duplicates_found": self.duplicates_found,
            "cross_split_overlaps": self.cross_split_overlaps,
            "contaminated_task_count": len(self.contaminated_tasks),
            "contaminated_tasks": self.contaminated_tasks,
            "overlap_matrix": self.overlap_matrix,
        }


def compute_task_fingerprint(task: TaskV1) -> str:
    """
    Compute a deterministic fingerprint for a task.

    The fingerprint is based on task content that should be unique
    for semantically different tasks.

    Args:
        task: TaskV1 instance to fingerprint

    Returns:
        SHA256 hash string
    """
    # Normalize content for fingerprinting
    content_parts = [
        task.task_id,
        task.task_type.value,
        _normalize_text(task.instruction),
        _normalize_text(task.context),
        task.source_package,
        task.source_file or "",
    ]

    # Sort and join
    content = "|".join(content_parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def compute_content_fingerprint(task: TaskV1) -> str:
    """
    Compute a fingerprint based purely on task content.

    This is useful for detecting semantically similar tasks that
    may have different IDs.

    Args:
        task: TaskV1 instance

    Returns:
        SHA256 hash string
    """
    # Focus on the actual content
    content_parts = [
        _normalize_text(task.instruction),
        _normalize_text(task.context),
        task.source_package,
        task.source_file or "",
    ]

    content = "|".join(content_parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Collapse whitespace
    - Remove common formatting
    """
    if not text:
        return ""

    # Lowercase and collapse whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def find_duplicates(tasks: list[TaskV1]) -> dict[str, list[str]]:
    """
    Find duplicate tasks based on content fingerprint.

    Args:
        tasks: List of TaskV1 instances

    Returns:
        Dictionary mapping fingerprint to list of task_ids that share it
    """
    fingerprint_map: dict[str, list[str]] = {}

    for task in tasks:
        fp = compute_content_fingerprint(task)
        if fp not in fingerprint_map:
            fingerprint_map[fp] = []
        fingerprint_map[fp].append(task.task_id)

    # Return only groups with duplicates
    return {fp: ids for fp, ids in fingerprint_map.items() if len(ids) > 1}


def find_cross_split_overlap(
    tasks_by_split: dict[str, list[TaskV1]],
) -> dict[str, list[tuple[str, str]]]:
    """
    Find tasks that appear in multiple splits.

    Args:
        tasks_by_split: Dictionary mapping split name to list of tasks

    Returns:
        Dictionary mapping task_id to list of (split, task_id) tuples
    """
    # Build fingerprint index
    fingerprint_to_splits: dict[str, list[tuple[str, str]]] = {}

    for split_name, tasks in tasks_by_split.items():
        for task in tasks:
            fp = compute_content_fingerprint(task)
            if fp not in fingerprint_to_splits:
                fingerprint_to_splits[fp] = []
            fingerprint_to_splits[fp].append((split_name, task.task_id))

    # Return only tasks in multiple splits
    return {
        fp: splits
        for fp, splits in fingerprint_to_splits.items()
        if len(set(s[0] for s in splits)) > 1
    }


def analyze_decontamination(tasks_dir: Path) -> DecontaminationReport:
    """
    Analyze a directory of tasks for contamination issues.

    Args:
        tasks_dir: Directory containing task files organized by split

    Returns:
        DecontaminationReport with findings
    """
    report = DecontaminationReport()

    # Load all tasks
    tasks_by_split: dict[str, list[TaskV1]] = {}
    all_tasks: list[TaskV1] = []

    for split_dir in tasks_dir.iterdir():
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        tasks_by_split[split_name] = []

        for task_file in split_dir.glob("*.json"):
            try:
                with open(task_file, encoding="utf-8") as f:
                    data = json.load(f)
                task = TaskV1.model_validate(data)
                tasks_by_split[split_name].append(task)
                all_tasks.append(task)
            except Exception:
                continue

    report.total_tasks = len(all_tasks)

    # Find duplicates within all tasks
    duplicates = find_duplicates(all_tasks)
    report.duplicate_groups = len(duplicates)
    report.duplicates_found = sum(len(ids) - 1 for ids in duplicates.values())

    # Record contaminated tasks
    for fp, task_ids in duplicates.items():
        # Keep first, mark others as contaminated
        for task_id in task_ids[1:]:
            report.contaminated_tasks.append(task_id)

    # Find cross-split overlap
    overlaps = find_cross_split_overlap(tasks_by_split)
    report.cross_split_overlaps = len(overlaps)

    for fp, splits in overlaps.items():
        for split_name, task_id in splits[1:]:  # Keep first split
            if task_id not in report.contaminated_tasks:
                report.contaminated_tasks.append(task_id)

    # Build overlap matrix
    split_names = list(tasks_by_split.keys())
    report.overlap_matrix = {s1: {s2: 0 for s2 in split_names} for s1 in split_names}

    for fp, splits in overlaps.items():
        split_set = list(set(s[0] for s in splits))
        for i, s1 in enumerate(split_set):
            for s2 in split_set[i + 1 :]:
                report.overlap_matrix[s1][s2] += 1
                report.overlap_matrix[s2][s1] += 1

    return report


def deduplicate_tasks(tasks: list[TaskV1], keep: str = "first") -> tuple[list[TaskV1], list[str]]:
    """
    Remove duplicate tasks from a list.

    Args:
        tasks: List of TaskV1 instances
        keep: Which duplicate to keep ("first" or "last")

    Returns:
        Tuple of (deduplicated tasks, removed task IDs)
    """
    seen_fps: set[str] = set()
    deduped: list[TaskV1] = []
    removed: list[str] = []

    # Reverse if keeping last
    task_iter = reversed(tasks) if keep == "last" else tasks

    for task in task_iter:
        fp = compute_content_fingerprint(task)
        if fp not in seen_fps:
            seen_fps.add(fp)
            if keep == "last":
                deduped.insert(0, task)
            else:
                deduped.append(task)
        else:
            removed.append(task.task_id)

    return deduped, removed


def check_test_contamination(task: TaskV1, reference_tests: set[str]) -> list[str]:
    """
    Check if a task's tests overlap with a reference set.

    This is useful for ensuring test sets don't leak into training data.

    Args:
        task: TaskV1 instance to check
        reference_tests: Set of test identifiers to check against

    Returns:
        List of overlapping test identifiers
    """
    overlaps: list[str] = []

    for test in task.tests.fail_to_pass:
        if test in reference_tests:
            overlaps.append(test)

    for test in task.tests.pass_to_pass:
        if test in reference_tests:
            overlaps.append(test)

    return overlaps
