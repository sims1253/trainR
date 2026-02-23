"""Dataset lifecycle manager for benchmark tasks.

This module provides:
- Dataset splitting (train/dev/test) with deterministic seeds
- Validation of task files
- Export to dataset manifest format
- CLI interface for dataset operations

Usage:
    uv run python -m bench.dataset.manager split --config configs/dataset/default.yaml
    uv run python -m bench.dataset.manager validate --input tasks_v1
    uv run python -m bench.dataset.manager export --input tasks_v1 --output dataset.json
"""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from bench.dataset.decontam import (
    DecontaminationReport,
    analyze_decontamination,
)
from bench.schema.v1 import TaskV1


@dataclass
class DatasetConfig:
    """Configuration for dataset management."""

    # Split ratios (should sum to 1.0)
    train_ratio: float = 0.7
    dev_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed for reproducibility
    seed: int = 42

    # Input/output paths
    input_dir: str = "tasks_v1"
    output_dir: str = "tasks_v1"

    # Validation settings
    strict_validation: bool = False
    skip_invalid: bool = True

    # Decontamination
    check_contamination: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        """Load configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            train_ratio=data.get("splits", {}).get("train", 0.7),
            dev_ratio=data.get("splits", {}).get("dev", 0.15),
            test_ratio=data.get("splits", {}).get("test", 0.15),
            seed=data.get("seed", 42),
            input_dir=data.get("input_dir", "tasks_v1"),
            output_dir=data.get("output_dir", "tasks_v1"),
            strict_validation=data.get("validation", {}).get("strict", False),
            skip_invalid=data.get("validation", {}).get("skip_invalid", True),
            check_contamination=data.get("decontamination", {}).get("check", True),
        )


@dataclass
class SplitStats:
    """Statistics for a dataset split."""

    name: str
    task_count: int
    task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task_count": self.task_count,
            "task_ids": self.task_ids,
        }


@dataclass
class DatasetManifest:
    """Manifest for a processed dataset."""

    # Version info
    version: str = "1.0"
    schema_version: str = "1.0"

    # Fingerprint (hash of all task IDs)
    fingerprint: str = ""

    # Creation metadata
    created_at: str = ""
    config_used: dict[str, Any] = field(default_factory=dict)

    # Split statistics
    splits: dict[str, SplitStats] = field(default_factory=dict)

    # Total counts
    total_tasks: int = 0

    # Decontamination status
    decontamination_status: str = "unknown"
    decontamination_report: DecontaminationReport | None = None

    # Source fingerprint
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "config_used": self.config_used,
            "splits": {name: stats.to_dict() for name, stats in self.splits.items()},
            "total_tasks": self.total_tasks,
            "decontamination_status": self.decontamination_status,
            "decontamination_report": (
                self.decontamination_report.to_dict() if self.decontamination_report else None
            ),
            "source_fingerprint": self.source_fingerprint,
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls(
            version=data.get("version", "1.0"),
            schema_version=data.get("schema_version", "1.0"),
            fingerprint=data.get("fingerprint", ""),
            created_at=data.get("created_at", ""),
            config_used=data.get("config_used", {}),
            total_tasks=data.get("total_tasks", 0),
            decontamination_status=data.get("decontamination_status", "unknown"),
            source_fingerprint=data.get("source_fingerprint", ""),
        )

        for split_name, split_data in data.get("splits", {}).items():
            manifest.splits[split_name] = SplitStats(
                name=split_name,
                task_count=split_data.get("task_count", 0),
                task_ids=split_data.get("task_ids", []),
            )

        return manifest


@dataclass
class ValidationResult:
    """Result of validating a task file."""

    valid: bool
    task_id: str
    file_path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Report from validating a dataset."""

    total_files: int = 0
    valid_tasks: int = 0
    invalid_tasks: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_files": self.total_files,
            "valid_tasks": self.valid_tasks,
            "invalid_tasks": self.invalid_tasks,
            "results": [
                {
                    "valid": r.valid,
                    "task_id": r.task_id,
                    "file_path": r.file_path,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in self.results
            ],
        }


def load_tasks_from_directory(directory: Path) -> list[tuple[TaskV1, Path]]:
    """
    Load all tasks from a directory.

    Args:
        directory: Directory containing task JSON files

    Returns:
        List of (TaskV1, Path) tuples
    """
    tasks: list[tuple[TaskV1, Path]] = []

    # Files to skip (non-task files)
    skip_files = {"manifest.json", "dataset.json", "metadata.json"}

    for json_file in directory.rglob("*.json"):
        # Skip non-task files
        if json_file.name in skip_files:
            continue

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            task = TaskV1.model_validate(data)
            tasks.append((task, json_file))
        except Exception:
            continue

    return tasks


def compute_dataset_fingerprint(task_ids: list[str]) -> str:
    """
    Compute a deterministic fingerprint for a set of tasks.

    Args:
        task_ids: List of task IDs

    Returns:
        SHA256 hash string
    """
    sorted_ids = sorted(task_ids)
    content = "|".join(sorted_ids)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def split_dataset(
    tasks: list[tuple[TaskV1, Path]],
    config: DatasetConfig,
) -> dict[str, list[tuple[TaskV1, Path]]]:
    """
    Split tasks into train/dev/test sets deterministically.

    Args:
        tasks: List of (TaskV1, Path) tuples
        config: Dataset configuration

    Returns:
        Dictionary mapping split name to list of tasks
    """
    # Set random seed for reproducibility
    random.seed(config.seed)

    # Shuffle tasks deterministically
    shuffled = tasks.copy()
    random.shuffle(shuffled)

    # Calculate split sizes
    n = len(shuffled)
    train_size = int(n * config.train_ratio)
    dev_size = int(n * config.dev_ratio)

    # Create non-overlapping splits from a single shuffled list
    # Remainder (due to int truncation) goes to test split
    splits: dict[str, list[tuple[TaskV1, Path]]] = {
        "train": shuffled[:train_size],
        "dev": shuffled[train_size : train_size + dev_size],
        "test": shuffled[train_size + dev_size :],
    }

    return splits


def validate_task_file(file_path: Path, strict: bool = False) -> ValidationResult:
    """
    Validate a single task file.

    Args:
        file_path: Path to task JSON file
        strict: Whether to fail on warnings

    Returns:
        ValidationResult
    """
    result = ValidationResult(
        valid=False,
        task_id="unknown",
        file_path=str(file_path),
    )

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result.errors.append(f"Failed to read file: {e}")
        return result

    # Check required fields
    if "task_id" not in data:
        result.errors.append("Missing required field: task_id")
        return result

    result.task_id = data["task_id"]

    # Validate against schema
    try:
        task = TaskV1.model_validate(data)

        # Additional semantic checks
        if not task.instruction.strip():
            result.warnings.append("Empty instruction")

        if not task.source_package:
            result.warnings.append("Missing source_package")

        if task.quality_score < 0 or task.quality_score > 10:
            result.warnings.append(f"Unusual quality_score: {task.quality_score}")

        # Valid only if NO errors; in strict mode, also require NO warnings
        if strict:
            result.valid = len(result.errors) == 0 and len(result.warnings) == 0
        else:
            result.valid = len(result.errors) == 0

    except ValidationError as e:
        for err in e.errors():
            field_path = ".".join(str(loc) for loc in err.get("loc", ["unknown"]))
            result.errors.append(
                f"Validation error in {field_path}: {err.get('msg', 'Unknown error')}"
            )

    return result


def validate_dataset(input_dir: Path, strict: bool = False) -> ValidationReport:
    """
    Validate all tasks in a directory.

    Args:
        input_dir: Directory containing task files
        strict: Whether to fail on warnings

    Returns:
        ValidationReport
    """
    report = ValidationReport()

    # Files to skip (non-task files)
    skip_files = {"manifest.json", "dataset.json", "metadata.json"}

    for json_file in input_dir.rglob("*.json"):
        # Skip non-task files
        if json_file.name in skip_files:
            continue

        report.total_files += 1
        result = validate_task_file(json_file, strict)
        report.results.append(result)

        if result.valid:
            report.valid_tasks += 1
        else:
            report.invalid_tasks += 1

    return report


def export_dataset(
    input_dir: Path,
    output_path: Path,
    config: DatasetConfig | None = None,
) -> DatasetManifest:
    """
    Export a dataset to a manifest file.

    Args:
        input_dir: Directory containing task files
        output_path: Path to write manifest
        config: Optional configuration used

    Returns:
        DatasetManifest
    """
    # Load all tasks
    tasks_by_split: dict[str, list[TaskV1]] = {}
    all_task_ids: list[str] = []

    for split_dir in input_dir.iterdir():
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
                all_task_ids.append(task.task_id)
            except Exception:
                continue

    # Create manifest
    manifest = DatasetManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=compute_dataset_fingerprint(all_task_ids),
        total_tasks=len(all_task_ids),
        config_used=config.__dict__ if config else {},
    )

    # Add split stats
    for split_name, tasks in tasks_by_split.items():
        manifest.splits[split_name] = SplitStats(
            name=split_name,
            task_count=len(tasks),
            task_ids=[t.task_id for t in tasks],
        )

    # Compute source fingerprint
    manifest.source_fingerprint = compute_dataset_fingerprint(sorted(all_task_ids))

    # Check decontamination
    if config and config.check_contamination:
        decon_report = analyze_decontamination(input_dir)
        manifest.decontamination_report = decon_report
        if decon_report.duplicates_found > 0 or decon_report.cross_split_overlaps > 0:
            manifest.decontamination_status = "contaminated"
        else:
            manifest.decontamination_status = "clean"

    # Save manifest
    manifest.save(output_path)

    return manifest


def run_split_command(config_path: Path) -> int:
    """
    Run the split command.

    Args:
        config_path: Path to config YAML file

    Returns:
        Exit code
    """
    # Load config
    config = DatasetConfig.from_yaml(config_path)

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    # Load all tasks
    print(f"Loading tasks from {input_dir}...")
    tasks = load_tasks_from_directory(input_dir)
    print(f"Found {len(tasks)} tasks")

    if not tasks:
        print("Error: No valid tasks found")
        return 1

    # Split tasks
    print(f"Splitting with seed={config.seed}...")
    print(f"  Train: {config.train_ratio * 100:.0f}%")
    print(f"  Dev: {config.dev_ratio * 100:.0f}%")
    print(f"  Test: {config.test_ratio * 100:.0f}%")

    splits = split_dataset(tasks, config)

    # Write splits to output
    for split_name, split_tasks in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing {len(split_tasks)} tasks to {split_dir}...")

        for task, _ in split_tasks:
            task_file = split_dir / f"{task.task_id}.json"
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task.model_dump(mode="json"), f, indent=2)

    # Create manifest
    manifest_path = output_dir / "manifest.json"
    manifest = DatasetManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=compute_dataset_fingerprint([t.task_id for t, _ in tasks]),
        total_tasks=len(tasks),
        config_used={
            "train_ratio": config.train_ratio,
            "dev_ratio": config.dev_ratio,
            "test_ratio": config.test_ratio,
            "seed": config.seed,
            "source_config": str(config_path),
        },
    )

    for split_name, split_tasks in splits.items():
        manifest.splits[split_name] = SplitStats(
            name=split_name,
            task_count=len(split_tasks),
            task_ids=[t.task_id for t, _ in split_tasks],
        )

    manifest.save(manifest_path)
    print(f"Manifest written to {manifest_path}")

    return 0


def run_validate_command(input_dir: Path, strict: bool = False) -> int:
    """
    Run the validate command.

    Args:
        input_dir: Directory containing task files
        strict: Whether to fail on warnings

    Returns:
        Exit code
    """
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    print(f"Validating tasks in {input_dir}...")
    report = validate_dataset(input_dir, strict)

    print("\nValidation Report:")
    print(f"  Total files: {report.total_files}")
    print(f"  Valid: {report.valid_tasks}")
    print(f"  Invalid: {report.invalid_tasks}")

    if report.invalid_tasks > 0:
        print("\nInvalid tasks:")
        for result in report.results:
            if not result.valid:
                print(f"\n  {result.file_path}:")
                for err in result.errors:
                    print(f"    ERROR: {err}")
                for warn in result.warnings:
                    print(f"    WARNING: {warn}")

    return 1 if report.invalid_tasks > 0 else 0


def run_export_command(input_dir: Path, output_path: Path, check_contamination: bool = True) -> int:
    """
    Run the export command.

    Args:
        input_dir: Directory containing task files
        output_path: Path to write manifest
        check_contamination: Whether to check for contamination

    Returns:
        Exit code
    """
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    print(f"Exporting dataset from {input_dir}...")

    config = DatasetConfig(check_contamination=check_contamination)
    manifest = export_dataset(input_dir, output_path, config)

    print("\nDataset Manifest:")
    print(f"  Version: {manifest.version}")
    print(f"  Fingerprint: {manifest.fingerprint}")
    print(f"  Created: {manifest.created_at}")
    print(f"  Total tasks: {manifest.total_tasks}")
    print(f"  Decontamination: {manifest.decontamination_status}")

    print("\n  Splits:")
    for split_name, stats in manifest.splits.items():
        print(f"    {split_name}: {stats.task_count} tasks")

    print(f"\nManifest written to {output_path}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dataset lifecycle manager for benchmark tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m bench.dataset.manager split --config configs/dataset/default.yaml
    uv run python -m bench.dataset.manager validate --input tasks_v1
    uv run python -m bench.dataset.manager export --input tasks_v1 --output dataset.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset into train/dev/test")
    split_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/default.yaml"),
        help="Path to config YAML file",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate task files")
    validate_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing task files",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export dataset manifest")
    export_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing task files",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for manifest JSON",
    )
    export_parser.add_argument(
        "--no-contamination-check",
        action="store_true",
        help="Skip contamination checking",
    )

    args = parser.parse_args()

    if args.command == "split":
        return run_split_command(args.config)
    elif args.command == "validate":
        return run_validate_command(args.input, args.strict)
    elif args.command == "export":
        return run_export_command(
            args.input,
            args.output,
            check_contamination=not args.no_contamination_check,
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
