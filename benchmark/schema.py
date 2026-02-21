"""Standardized result schema for benchmark runs."""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a single task-model evaluation."""

    task_id: str
    model: str
    passed: bool
    score: float
    latency_s: float
    error_category: str | None = None
    error_message: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    trajectory_path: str | None = None
    repeat_index: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRun:
    """Metadata for a complete benchmark run."""

    run_id: str
    models: list[str]
    task_count: int
    skill_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_sha: str = field(default_factory=lambda: _get_git_sha())
    config: dict[str, Any] = field(default_factory=dict)
    results: list[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def save(self, path: Path) -> None:
        """Save the benchmark run to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "models": self.models,
            "task_count": self.task_count,
            "skill_version": self.skill_version,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "BenchmarkRun":
        """Load a benchmark run from a JSON file."""
        data = json.loads(path.read_text())
        results = [BenchmarkResult(**r) for r in data.pop("results", [])]
        run = cls(**data)
        run.results = results
        return run

    def pass_rate(self, model: str | None = None) -> float:
        """Calculate pass rate, optionally filtered by model."""
        results = self.results
        if model:
            results = [r for r in results if r.model == model]
        if not results:
            return 0.0
        return sum(1 for r in results if r.passed) / len(results)

    def avg_latency(self, model: str | None = None) -> float:
        """Calculate average latency, optionally filtered by model."""
        results = self.results
        if model:
            results = [r for r in results if r.model == model]
        if not results:
            return 0.0
        return sum(r.latency_s for r in results) / len(results)


def _get_git_sha() -> str:
    """Get current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
