#!/usr/bin/env python3
"""Aggregate benchmark results into JSON for the visualizer."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results/baselines")
OUTPUT_FILE = Path("visualizer/src/data/benchmark-results.json")

SKILL_MAP = {
    "no_skill": "no_skill",
    "testing-r-packages-orig": "posit_skill",
}

DISPLAY_NAME_MAP = {
    "gpt-oss-120b:free": "OpenAI GPT-OSS-120B",
    "minimax-m2.5-free": "Minimax M2.5",
    "nemotron-3-nano-30b-a3b:free": "NVIDIA Nemotron Nano",
    "step-3.5-flash:free": "StepFun Step-3.5-Flash",
    "glm-4.5": "GLM-4.5",
}

PROVIDER_MAP = {
    "gpt-oss-120b:free": "openrouter",
    "minimax-m2.5-free": "opencode",
    "nemotron-3-nano-30b-a3b:free": "openrouter",
    "step-3.5-flash:free": "openrouter",
    "glm-4.5": "opencode",
}


def get_display_name(model: str) -> str:
    if model in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[model]

    name = model.replace(":free", "").replace("-", " ")
    parts = name.split()
    return " ".join(p.capitalize() for p in parts)


def get_provider(model: str) -> str:
    if model in PROVIDER_MAP:
        return PROVIDER_MAP[model]
    if ":free" in model:
        return "openrouter"
    return "unknown"


def parse_filename(filename: str) -> str | None:
    if not filename.endswith(".json"):
        return None

    base = filename[:-5]
    if filename.startswith("eval_"):
        base = base[5:]

    parts = base.split("_")

    if len(parts) < 3:
        return None

    time_part = parts[-1]
    date_part = parts[-2] if len(parts) >= 2 else ""

    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        return f"{date_part}_{time_part}"

    return None


def calculate_aggregations(results: list[dict]) -> dict[str, Any]:
    if not results:
        return {
            "overall": {"pass_rate": 0.0, "total": 0, "passed": 0, "failed": 0},
            "by_difficulty": {},
            "by_package": {},
        }

    total = len(results)
    passed = sum(1 for r in results if r.get("success", False))
    failed = total - passed
    pass_rate = round(passed / total, 3) if total > 0 else 0.0

    by_difficulty: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    by_package: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})

    for r in results:
        difficulty = r.get("difficulty", "unknown")
        package = r.get("source_package", "unknown")
        success = r.get("success", False)

        by_difficulty[difficulty]["total"] += 1
        by_package[package]["total"] += 1
        if success:
            by_difficulty[difficulty]["passed"] += 1
            by_package[package]["passed"] += 1

    by_difficulty_rates = {
        d: round(stats["passed"] / stats["total"], 3) if stats["total"] > 0 else 0.0
        for d, stats in by_difficulty.items()
    }

    by_package_rates = {
        p: round(stats["passed"] / stats["total"], 3) if stats["total"] > 0 else 0.0
        for p, stats in by_package.items()
    }

    return {
        "overall": {"pass_rate": pass_rate, "total": total, "passed": passed, "failed": failed},
        "by_difficulty": by_difficulty_rates,
        "by_package": by_package_rates,
    }


def main():
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    eval_files = (
        list(results_dir.rglob("eval_*.json"))
        + list(results_dir.rglob("*_no_skill_*.json"))
        + list(results_dir.rglob("*_testing-r-packages-orig_*.json"))
    )
    if not eval_files:
        print(f"No eval files found in {results_dir}")
        return

    model_skill_runs: dict[str, dict[str, tuple[str, dict]]] = defaultdict(dict)
    all_packages: set[str] = set()
    total_tasks = 0

    for filepath in eval_files:
        timestamp = parse_filename(filepath.name)
        if not timestamp:
            print(f"Skipping file with unexpected name: {filepath.name}")
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error reading {filepath}: {e}")
            continue

        config = data.get("config", {})
        model_full = config.get("model", "unknown")
        model_name = model_full.split("/")[-1] if "/" in model_full else model_full
        file_skill = config.get("skill", "unknown")
        skill_key = SKILL_MAP.get(file_skill, file_skill)

        existing = model_skill_runs[model_name].get(skill_key)
        if existing is None or timestamp > existing[0]:
            model_skill_runs[model_name][skill_key] = (timestamp, data)

        for r in data.get("results", []):
            if pkg := r.get("source_package"):
                all_packages.add(pkg)
            total_tasks += 1

    models_output = []

    for model_name, skills in model_skill_runs.items():
        model_data: dict[str, Any] = {
            "name": model_name,
            "display_name": get_display_name(model_name),
            "provider": get_provider(model_name),
            "results": {},
        }

        for skill_key, (_timestamp, data) in skills.items():
            model_data["results"][skill_key] = calculate_aggregations(data.get("results", []))

        models_output.append(model_data)

    def sort_key(m):
        results = m.get("results", {})
        posit_rate = results.get("posit_skill", {}).get("overall", {}).get("pass_rate", 0)
        no_skill_rate = results.get("no_skill", {}).get("overall", {}).get("pass_rate", 0)
        return (-posit_rate, -no_skill_rate)

    models_output.sort(key=sort_key)

    skills_included = set()
    runs_count = 0
    for model_data in models_output:
        skills_included.update(model_data["results"].keys())
        runs_count += len(model_data["results"])

    output = {
        "models": models_output,
        "metadata": {
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_tasks": total_tasks,
            "packages": sorted(all_packages),
            "runs_included": runs_count,
            "skills": sorted(skills_included),
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Processed {len(eval_files)} files")
    print(f"Found {len(models_output)} models with {runs_count} runs")
    print(f"Packages: {', '.join(sorted(all_packages))}")
    print(f"Output written to {OUTPUT_FILE}")

    for m in models_output:
        print(f"\n{m['display_name']} ({m['name']}):")
        for skill, results in m["results"].items():
            overall = results["overall"]
            print(f"  {skill}: {overall['passed']}/{overall['total']} = {overall['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
