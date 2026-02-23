#!/usr/bin/env python3
"""
Export adapter for visualizer data.

Consumes canonical backend outputs (manifest.json, results.jsonl, summary.json)
and transforms them to VisualizerDataV1 format for the visualizer.

Usage:
    uv run python scripts/export_visualizer_data.py --input results --output visualizer/src/data/benchmark-results.json
    uv run python scripts/export_visualizer_data.py --sample  # Generate sample data for testing
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Type aliases for clarity
VisualizerDataV1 = dict[str, Any]
ModelResultV1 = dict[str, Any]
SkillResultV1 = dict[str, Any]

VISUALIZER_DATA_VERSION = 1

# Canonical artifact file names
MANIFEST_FILE = "manifest.json"
RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"

# Legacy format support
LEGACY_RESULTS_DIR = "baselines"

# Skill mapping for canonical format
SKILL_MAP = {
    "no_skill": "no_skill",
    "posit_skill": "posit_skill",
    "testing-r-packages-orig": "posit_skill",
}

# Display name mappings
DISPLAY_NAME_MAP = {
    "gpt-oss-120b:free": "OpenAI GPT-OSS-120B",
    "minimax-m2.5-free": "Minimax M2.5",
    "nemotron-3-nano-30b-a3b:free": "NVIDIA Nemotron Nano",
    "step-3.5-flash:free": "StepFun Step-3.5-Flash",
    "glm-4.5": "GLM-4.5",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "gpt-4o": "GPT-4o",
    "gpt-4-turbo": "GPT-4 Turbo",
}

PROVIDER_MAP = {
    "gpt-oss-120b:free": "openrouter",
    "minimax-m2.5-free": "opencode",
    "nemotron-3-nano-30b-a3b:free": "openrouter",
    "step-3.5-flash:free": "openrouter",
    "glm-4.5": "opencode",
    "claude-3-5-sonnet": "anthropic",
    "gpt-4o": "openai",
    "gpt-4-turbo": "openai",
}


class ExportError(Exception):
    """Raised when export fails with actionable error message."""

    pass


def get_display_name(model: str) -> str:
    """Get human-readable display name for a model."""
    if model in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[model]

    # Clean up model name
    name = model.replace(":free", "").replace("-", " ")
    parts = name.split()
    return " ".join(p.capitalize() for p in parts)


def get_provider(model: str) -> str:
    """Get provider name for a model."""
    if model in PROVIDER_MAP:
        return PROVIDER_MAP[model]
    if ":free" in model:
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt"):
        return "openai"
    return "unknown"


def calculate_skill_result(results: list[dict[str, Any]]) -> SkillResultV1:
    """Calculate aggregated skill result from individual task results."""
    if not results:
        return {
            "overall": {"pass_rate": 0.0, "total": 0, "passed": 0, "failed": 0},
            "by_difficulty": {"easy": 0.0, "medium": 0.0, "hard": 0.0},
            "by_package": {},
        }

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", r.get("success", False)))
    failed = total - passed
    pass_rate = round(passed / total, 3) if total > 0 else 0.0

    # Aggregate by difficulty
    by_difficulty: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    # Aggregate by package
    by_package: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})

    for r in results:
        # Handle both canonical (difficulty) and legacy (difficulty) formats
        difficulty = r.get("difficulty", r.get("metadata", {}).get("difficulty", "medium"))
        # Handle both canonical (source_package) and legacy formats
        package = r.get("source_package", r.get("metadata", {}).get("source_package", "unknown"))
        success = r.get("passed", r.get("success", False))

        by_difficulty[difficulty]["total"] += 1
        by_package[package]["total"] += 1
        if success:
            by_difficulty[difficulty]["passed"] += 1
            by_package[package]["passed"] += 1

    # Convert to rates
    by_difficulty_rates = {
        "easy": 0.0,
        "medium": 0.0,
        "hard": 0.0,
    }
    for d, stats in by_difficulty.items():
        if d in by_difficulty_rates:
            by_difficulty_rates[d] = (
                round(stats["passed"] / stats["total"], 3) if stats["total"] > 0 else 0.0
            )

    by_package_rates = {
        p: round(stats["passed"] / stats["total"], 3) if stats["total"] > 0 else 0.0
        for p, stats in by_package.items()
    }

    return {
        "overall": {"pass_rate": pass_rate, "total": total, "passed": passed, "failed": failed},
        "by_difficulty": by_difficulty_rates,
        "by_package": by_package_rates,
    }


def read_canonical_results(input_dir: Path) -> list[dict[str, Any]]:
    """Read results from canonical results.jsonl file."""
    results_file = input_dir / RESULTS_FILE
    if not results_file.exists():
        return []

    results = []
    with open(results_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                warnings.warn(f"Invalid JSON at line {line_num} in {results_file}: {e}")

    return results


def read_canonical_manifest(input_dir: Path) -> dict[str, Any] | None:
    """Read manifest from canonical manifest.json file."""
    manifest_file = input_dir / MANIFEST_FILE
    if not manifest_file.exists():
        return None

    try:
        with open(manifest_file) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"Error reading manifest: {e}")
        return None


def read_canonical_summary(input_dir: Path) -> dict[str, Any] | None:
    """Read summary from canonical summary.json file."""
    summary_file = input_dir / SUMMARY_FILE
    if not summary_file.exists():
        return None

    try:
        with open(summary_file) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"Error reading summary: {e}")
        return None


def discover_run_directories(input_dir: Path) -> list[Path]:
    """Discover all run directories containing canonical artifacts."""
    run_dirs = []

    # Look for directories containing manifest.json or results.jsonl
    for path in input_dir.rglob("*"):
        if path.is_dir():
            if (path / MANIFEST_FILE).exists() or (path / RESULTS_FILE).exists():
                run_dirs.append(path)

    # Also check input_dir itself
    if (input_dir / MANIFEST_FILE).exists() or (input_dir / RESULTS_FILE).exists():
        if input_dir not in run_dirs:
            run_dirs.append(input_dir)

    return sorted(run_dirs)


def parse_legacy_filename(filename: str) -> tuple[str, str, str] | None:
    """Parse legacy eval filename to extract model, skill, and timestamp."""
    if not filename.endswith(".json"):
        return None

    base = filename[:-5]
    if filename.startswith("eval_"):
        base = base[5:]

    # Try to parse timestamp (last two parts: date_time)
    parts = base.split("_")
    if len(parts) < 3:
        return None

    # Last part is time (6 digits), second to last is date (8 digits)
    time_part = parts[-1]
    date_part = parts[-2] if len(parts) >= 2 else ""

    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        timestamp = f"{date_part}_{time_part}"
        # Skill is typically the second to last before date
        # Format: model_skill_date_time or model_skill
        skill = parts[-3] if len(parts) >= 3 else "unknown"
        model = "_".join(parts[:-3]) if len(parts) > 3 else parts[0]
        return (model, skill, timestamp)

    return None


def read_legacy_results(input_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Read results from legacy format (baselines directory with eval_*.json files)."""
    legacy_dir = input_dir / LEGACY_RESULTS_DIR
    if not legacy_dir.exists():
        legacy_dir = input_dir

    model_skill_results: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Find all eval files
    eval_files = (
        list(legacy_dir.rglob("eval_*.json"))
        + list(legacy_dir.rglob("*_no_skill_*.json"))
        + list(legacy_dir.rglob("*_testing-r-packages-orig_*.json"))
    )

    for filepath in eval_files:
        parsed = parse_legacy_filename(filepath.name)
        if not parsed:
            warnings.warn(f"Skipping file with unexpected name: {filepath.name}")
            continue

        model, file_skill, _timestamp = parsed

        try:
            with open(filepath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"Error reading {filepath}: {e}")
            continue

        # Map skill name
        skill_key = SKILL_MAP.get(file_skill, file_skill)

        # Extract results
        for r in data.get("results", []):
            # Add metadata from config if available
            config = data.get("config", {})
            if "model" not in r:
                r["model"] = config.get("model", model)
            model_skill_results[model][skill_key].append(r)

    return model_skill_results


def export_from_canonical(input_dir: Path) -> VisualizerDataV1:
    """Export visualizer data from canonical v1 format."""
    # Discover all run directories
    run_dirs = discover_run_directories(input_dir)

    if not run_dirs:
        raise ExportError(
            f"No canonical artifacts found in {input_dir}\n"
            f"Expected files: {MANIFEST_FILE}, {RESULTS_FILE}, or {SUMMARY_FILE}\n"
            f"Run a benchmark first or use --sample to generate test data."
        )

    # Aggregate results across all runs
    all_results: list[dict[str, Any]] = []
    all_packages: set[str] = set()
    total_tasks = 0
    runs_included = 0
    latest_timestamp = None

    for run_dir in run_dirs:
        results = read_canonical_results(run_dir)
        manifest = read_canonical_manifest(run_dir)

        if results:
            all_results.extend(results)
            runs_included += 1

        if manifest:
            if manifest.get("timestamp"):
                ts = manifest["timestamp"]
                if latest_timestamp is None or ts > latest_timestamp:
                    latest_timestamp = ts
            total_tasks = max(total_tasks, manifest.get("task_count", 0))

        # Extract packages from results metadata
        for r in results:
            pkg = r.get("source_package", r.get("metadata", {}).get("source_package"))
            if pkg:
                all_packages.add(pkg)

    if not all_results:
        raise ExportError(
            f"No results found in canonical format.\n"
            f"Discovered {len(run_dirs)} run directories but no results.jsonl entries.\n"
            f"Check that your benchmark runs completed successfully."
        )

    # Group by model and skill
    model_skill_results: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in all_results:
        model = r.get("model", "unknown")
        # Derive skill from metadata or default to posit_skill for canonical v1
        skill = r.get("skill", r.get("metadata", {}).get("skill", "posit_skill"))
        skill_key = SKILL_MAP.get(skill, skill)
        model_skill_results[model][skill_key].append(r)

    # Build output
    models_output = []
    for model_name, skills in model_skill_results.items():
        # Ensure both skill types exist
        no_skill_results = skills.get("no_skill", [])
        posit_skill_results = skills.get("posit_skill", [])

        # If no explicit skill split, treat all as posit_skill
        if not no_skill_results and not posit_skill_results:
            posit_skill_results = skills.get("unknown", [])

        model_data: ModelResultV1 = {
            "name": model_name,
            "display_name": get_display_name(model_name),
            "provider": get_provider(model_name),
            "results": {
                "no_skill": calculate_skill_result(no_skill_results),
                "posit_skill": calculate_skill_result(posit_skill_results),
            },
        }
        models_output.append(model_data)

    # Sort by posit_skill pass rate (descending)
    def sort_key(m: ModelResultV1) -> tuple[float, float]:
        posit_rate = m["results"]["posit_skill"]["overall"]["pass_rate"]
        no_skill_rate = m["results"]["no_skill"]["overall"]["pass_rate"]
        return (-posit_rate, -no_skill_rate)

    models_output.sort(key=sort_key)

    # Build metadata
    if not latest_timestamp:
        latest_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "visualizer_data_version": VISUALIZER_DATA_VERSION,
        "models": models_output,
        "metadata": {
            "last_updated": latest_timestamp,
            "total_tasks": total_tasks or len(all_results),
            "packages": sorted(all_packages),
            "runs_included": runs_included,
        },
    }


def export_from_legacy(input_dir: Path) -> VisualizerDataV1:
    """Export visualizer data from legacy format (baselines directory)."""
    model_skill_results = read_legacy_results(input_dir)

    if not model_skill_results:
        raise ExportError(
            f"No legacy results found in {input_dir}\n"
            f"Expected files in {LEGACY_RESULTS_DIR}/ directory with eval_*.json pattern.\n"
            f"Run a benchmark first or use --sample to generate test data."
        )

    all_packages: set[str] = set()
    total_tasks = 0

    # Build output
    models_output = []
    for model_name, skills in model_skill_results.items():
        no_skill_results = skills.get("no_skill", [])
        posit_skill_results = skills.get("posit_skill", [])

        # Extract packages
        for r in no_skill_results + posit_skill_results:
            pkg = r.get("source_package")
            if pkg:
                all_packages.add(pkg)
            total_tasks += 1

        model_data: ModelResultV1 = {
            "name": model_name,
            "display_name": get_display_name(model_name),
            "provider": get_provider(model_name),
            "results": {
                "no_skill": calculate_skill_result(no_skill_results),
                "posit_skill": calculate_skill_result(posit_skill_results),
            },
        }
        models_output.append(model_data)

    # Sort by posit_skill pass rate (descending)
    def sort_key(m: ModelResultV1) -> tuple[float, float]:
        posit_rate = m["results"]["posit_skill"]["overall"]["pass_rate"]
        no_skill_rate = m["results"]["no_skill"]["overall"]["pass_rate"]
        return (-posit_rate, -no_skill_rate)

    models_output.sort(key=sort_key)

    return {
        "visualizer_data_version": VISUALIZER_DATA_VERSION,
        "models": models_output,
        "metadata": {
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_tasks": total_tasks,
            "packages": sorted(all_packages),
            "runs_included": len(model_skill_results),
        },
    }


def generate_sample_data() -> VisualizerDataV1:
    """Generate sample data for testing the visualizer."""
    return {
        "visualizer_data_version": VISUALIZER_DATA_VERSION,
        "models": [
            {
                "name": "gpt-4o",
                "display_name": "GPT-4o",
                "provider": "openai",
                "results": {
                    "no_skill": {
                        "overall": {"pass_rate": 0.45, "total": 100, "passed": 45, "failed": 55},
                        "by_difficulty": {"easy": 0.75, "medium": 0.40, "hard": 0.20},
                        "by_package": {
                            "dplyr": 0.50,
                            "ggplot2": 0.40,
                            "testthat": 0.45,
                            "cli": 0.50,
                        },
                    },
                    "posit_skill": {
                        "overall": {"pass_rate": 0.68, "total": 100, "passed": 68, "failed": 32},
                        "by_difficulty": {"easy": 0.90, "medium": 0.65, "hard": 0.50},
                        "by_package": {
                            "dplyr": 0.72,
                            "ggplot2": 0.65,
                            "testthat": 0.68,
                            "cli": 0.70,
                        },
                    },
                },
            },
            {
                "name": "claude-3-5-sonnet",
                "display_name": "Claude 3.5 Sonnet",
                "provider": "anthropic",
                "results": {
                    "no_skill": {
                        "overall": {"pass_rate": 0.50, "total": 100, "passed": 50, "failed": 50},
                        "by_difficulty": {"easy": 0.80, "medium": 0.45, "hard": 0.25},
                        "by_package": {
                            "dplyr": 0.55,
                            "ggplot2": 0.45,
                            "testthat": 0.50,
                            "cli": 0.55,
                        },
                    },
                    "posit_skill": {
                        "overall": {"pass_rate": 0.75, "total": 100, "passed": 75, "failed": 25},
                        "by_difficulty": {"easy": 0.95, "medium": 0.72, "hard": 0.58},
                        "by_package": {
                            "dplyr": 0.78,
                            "ggplot2": 0.72,
                            "testthat": 0.75,
                            "cli": 0.77,
                        },
                    },
                },
            },
            {
                "name": "glm-4.5",
                "display_name": "GLM-4.5",
                "provider": "opencode",
                "results": {
                    "no_skill": {
                        "overall": {"pass_rate": 0.35, "total": 100, "passed": 35, "failed": 65},
                        "by_difficulty": {"easy": 0.60, "medium": 0.30, "hard": 0.15},
                        "by_package": {
                            "dplyr": 0.38,
                            "ggplot2": 0.32,
                            "testthat": 0.35,
                            "cli": 0.38,
                        },
                    },
                    "posit_skill": {
                        "overall": {"pass_rate": 0.55, "total": 100, "passed": 55, "failed": 45},
                        "by_difficulty": {"easy": 0.78, "medium": 0.52, "hard": 0.35},
                        "by_package": {
                            "dplyr": 0.58,
                            "ggplot2": 0.52,
                            "testthat": 0.55,
                            "cli": 0.58,
                        },
                    },
                },
            },
        ],
        "metadata": {
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_tasks": 100,
            "packages": ["cli", "dplyr", "ggplot2", "testthat"],
            "runs_included": 2,
        },
    }


def validate_visualizer_data(data: VisualizerDataV1) -> list[str]:
    """Validate visualizer data against V1 schema. Returns list of errors."""
    errors = []

    # Check version
    if data.get("visualizer_data_version") != VISUALIZER_DATA_VERSION:
        errors.append(
            f"Invalid visualizer_data_version: expected {VISUALIZER_DATA_VERSION}, "
            f"got {data.get('visualizer_data_version')}"
        )
        return errors  # Version mismatch - can't validate further

    # Check models
    if "models" not in data or not isinstance(data["models"], list):
        errors.append("Missing or invalid 'models' array")
        return errors

    if len(data["models"]) == 0:
        errors.append("'models' array is empty")
        return errors

    for i, model in enumerate(data["models"]):
        prefix = f"models[{i}]"

        # Required string fields
        for field in ["name", "display_name", "provider"]:
            if field not in model or not isinstance(model[field], str) or not model[field]:
                errors.append(f"{prefix}.{field} must be a non-empty string")

        # Check results
        if "results" not in model or not isinstance(model["results"], dict):
            errors.append(f"{prefix}.results must be an object")
            continue

        for skill in ["no_skill", "posit_skill"]:
            skill_prefix = f"{prefix}.results.{skill}"
            if skill not in model["results"]:
                errors.append(f"{skill_prefix} is required")
                continue

            skill_data = model["results"][skill]

            # Check overall
            if "overall" not in skill_data:
                errors.append(f"{skill_prefix}.overall is required")
            else:
                overall = skill_data["overall"]
                for field in ["pass_rate", "total", "passed", "failed"]:
                    if field not in overall or not isinstance(overall[field], (int, float)):
                        errors.append(f"{skill_prefix}.overall.{field} must be a number")

                # Validate consistency
                if (
                    isinstance(overall.get("passed"), (int, float))
                    and isinstance(overall.get("failed"), (int, float))
                    and isinstance(overall.get("total"), (int, float))
                ):
                    if overall["passed"] + overall["failed"] != overall["total"]:
                        errors.append(
                            f"{skill_prefix}.overall has inconsistent counts: "
                            f"passed ({overall['passed']}) + failed ({overall['failed']}) "
                            f"!= total ({overall['total']})"
                        )

            # Check by_difficulty
            if "by_difficulty" not in skill_data:
                errors.append(f"{skill_prefix}.by_difficulty is required")
            else:
                for level in ["easy", "medium", "hard"]:
                    if level not in skill_data["by_difficulty"]:
                        errors.append(f"{skill_prefix}.by_difficulty.{level} is required")
                    elif not isinstance(skill_data["by_difficulty"][level], (int, float)):
                        errors.append(f"{skill_prefix}.by_difficulty.{level} must be a number")

            # Check by_package (can be empty object)
            if "by_package" not in skill_data or not isinstance(skill_data["by_package"], dict):
                errors.append(f"{skill_prefix}.by_package must be an object")

    # Check metadata
    if "metadata" not in data or not isinstance(data["metadata"], dict):
        errors.append("Missing or invalid 'metadata' object")
    else:
        meta = data["metadata"]
        for field in ["last_updated", "total_tasks", "packages", "runs_included"]:
            if field not in meta:
                errors.append(f"metadata.{field} is required")

        if "packages" in meta and not isinstance(meta["packages"], list):
            errors.append("metadata.packages must be an array")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Export visualizer data from canonical backend outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from canonical format
  uv run python scripts/export_visualizer_data.py --input results --output visualizer/src/data/benchmark-results.json

  # Generate sample data for testing
  uv run python scripts/export_visualizer_data.py --sample --output visualizer/src/data/benchmark-results.json

  # Validate existing output
  uv run python scripts/export_visualizer_data.py --validate visualizer/src/data/benchmark-results.json
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results"),
        help="Input directory containing canonical artifacts (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualizer/src/data/benchmark-results.json"),
        help="Output JSON file path (default: visualizer/src/data/benchmark-results.json)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate sample data for testing instead of reading from input",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        help="Validate an existing output file instead of generating",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even with warnings (e.g., missing optional fields)",
    )

    args = parser.parse_args()

    # Validation mode
    if args.validate:
        if not args.validate.exists():
            print(f"Error: File not found: {args.validate}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.validate) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {args.validate}: {e}", file=sys.stderr)
            sys.exit(1)

        errors = validate_visualizer_data(data)
        if errors:
            print(f"Validation failed for {args.validate}:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Validation passed: {args.validate}")
            sys.exit(0)

    # Generate sample or export from input
    if args.sample:
        print("Generating sample data for testing...")
        data = generate_sample_data()
    else:
        # Try canonical format first, then fall back to legacy
        try:
            print(f"Reading canonical artifacts from {args.input}...")
            data = export_from_canonical(args.input)
            print("Successfully exported from canonical format")
        except ExportError as e:
            print(f"Canonical format not found: {e}", file=sys.stderr)
            print("Trying legacy format...", file=sys.stderr)
            try:
                data = export_from_legacy(args.input)
                print("Successfully exported from legacy format")
            except ExportError as e2:
                print(f"Error: {e2}", file=sys.stderr)
                print("\nTo fix this issue:", file=sys.stderr)
                print(
                    "  1. Run a benchmark: uv run python scripts/run_benchmark.py --config configs/benchmark.yaml",
                    file=sys.stderr,
                )
                print("  2. Or use --sample to generate test data", file=sys.stderr)
                sys.exit(1)

    # Validate output
    errors = validate_visualizer_data(data)
    if errors:
        print("Warning: Generated data has validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if not args.force:
            print("Use --force to write anyway", file=sys.stderr)
            sys.exit(1)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Output written to {args.output}")
    print(f"  Models: {len(data['models'])}")
    print(f"  Packages: {', '.join(data['metadata']['packages'])}")
    print(f"  Runs included: {data['metadata']['runs_included']}")


if __name__ == "__main__":
    main()
