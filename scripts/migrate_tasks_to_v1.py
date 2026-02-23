#!/usr/bin/env python
"""Migrate legacy tasks to canonical v1 format.

Usage:
    uv run python scripts/migrate_tasks_to_v1.py --in tasks --out tasks_v1 --report results/migration_report.json

This script:
- Reads all legacy task files from the input directory
- Converts them to TaskV1 format using appropriate adapters
- Writes migrated tasks to the output directory (preserving structure)
- Generates a detailed migration report
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bench.dataset.migrate import (
    MigrationCounter,
    MigrationWarning,
    migrate_tasks_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy tasks to canonical v1 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate all tasks from tasks/ to tasks_v1/
    uv run python scripts/migrate_tasks_to_v1.py --in tasks --out tasks_v1 --report results/migration_report.json

    # Dry run to see what would happen
    uv run python scripts/migrate_tasks_to_v1.py --in tasks --out tasks_v1 --report results/migration_report.json --dry-run

    # Verbose output with all warnings
    uv run python scripts/migrate_tasks_to_v1.py --in tasks --out tasks_v1 --report results/migration_report.json --verbose
        """,
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Input directory containing legacy tasks",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Output directory for migrated tasks",
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        required=True,
        help="Path to write migration report JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including all warnings",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing even if individual tasks fail (default: True)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    report_path = Path(args.report_path)

    # Validate input directory
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_path}")
        return 1

    if not input_path.is_dir():
        print(f"ERROR: Input path is not a directory: {input_path}")
        return 1

    # Count tasks to process
    json_files = list(input_path.rglob("*.json"))
    print(f"Found {len(json_files)} task files to process")

    if len(json_files) == 0:
        print("No task files found. Nothing to do.")
        return 0

    # Show dry-run message and exit
    if args.dry_run:
        print("\n[DRY RUN] Would migrate tasks:")
        for jf in sorted(json_files)[:10]:
            print(f"  - {jf.relative_to(input_path)}")
        if len(json_files) > 10:
            print(f"  ... and {len(json_files) - 10} more files")
        print(f"\n[DRY RUN] Would write output to: {output_path}")
        print(f"[DRY RUN] Would write report to: {report_path}")
        return 0

    # Perform migration
    print(f"\nMigrating tasks from {input_path} to {output_path}...")

    counters = MigrationCounter()
    warnings_list: list[MigrationWarning] = []

    report = migrate_tasks_directory(
        input_dir=input_path,
        output_dir=output_path,
        counters=counters,
        warnings_list=warnings_list,
    )

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks:     {report.total_tasks}")
    print(f"Migrated:        {report.migrated_count}")
    print(f"Failed:          {report.failed_count}")
    print(f"Success rate:    {report.migrated_count / report.total_tasks * 100:.1f}%")
    print()

    # Print per-split statistics
    print("Per-split statistics:")
    for split, stats in report.split_stats.items():
        rate = stats["migrated"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(
            f"  {split:12} - Total: {stats['total']:3}, Migrated: {stats['migrated']:3}, Failed: {stats['failed']:3} ({rate:.1f}%)"
        )
    print()

    # Print observability counters
    print("Migration observability:")
    print(f"  Legacy tasks parsed:    {counters.legacy_parsed}")
    print(f"  Already v1 format:      {counters.v1_already}")
    print(f"  Mined tasks:            {counters.mined_tasks}")
    print(f"  Testing tasks:          {counters.testing_tasks}")
    print(f"  Unknown format:         {counters.unknown_format}")
    print(f"  Adapter fallback used:  {counters.adapter_fallback_used}")
    print()

    # Print validation failures if any
    if counters.validation_failures:
        print("Validation failures by field:")
        for field, count in sorted(counters.validation_failures.items(), key=lambda x: -x[1]):
            print(f"  {field}: {count}")
        print()

    # Print warnings summary
    if counters.warnings_by_type:
        print("Warnings by type:")
        for field, count in sorted(counters.warnings_by_type.items(), key=lambda x: -x[1]):
            print(f"  {field}: {count}")
        print()

    # Print failures if any
    if report.failures:
        print(f"Failures ({len(report.failures)}):")
        for failure in report.failures[:10]:
            print(f"  - {failure['task_id']}: {failure['reason']}")
            if args.verbose:
                print(f"    File: {failure['source_file']}")
                print(f"    Format: {failure['format']}")
        if len(report.failures) > 10:
            print(f"  ... and {len(report.failures) - 10} more failures")
        print()

    # Print detailed warnings if verbose
    if args.verbose and report.warnings:
        print(f"Warnings ({len(report.warnings)}):")
        for warning in report.warnings[:20]:
            print(
                f"  - [{warning['severity']}] {warning['task_id']}: {warning['field']} - {warning['message']}"
            )
        if len(report.warnings) > 20:
            print(f"  ... and {len(report.warnings) - 20} more warnings")

    print(f"\nReport written to: {report_path}")

    # Return code based on success rate
    success_rate = report.migrated_count / report.total_tasks if report.total_tasks > 0 else 0
    if success_rate >= 0.95:
        print("\nMigration completed successfully (>=95% tasks migrated)")
        return 0
    elif success_rate >= 0.80:
        print("\nMigration completed with some failures (>=80% but <95% migrated)")
        return 0
    else:
        print("\nMigration completed with significant failures (<80% migrated)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
