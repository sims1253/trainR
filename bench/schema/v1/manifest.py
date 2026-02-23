"""Run manifest schema for benchmark run metadata.

The manifest captures:
- Run identification and provenance
- Configuration fingerprints
- Environment details
- Result summaries
"""

import subprocess
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from bench.schema.v1.results import ResultV1


class EnvironmentFingerprintV1(BaseModel):
    """Fingerprint of the execution environment."""

    python_version: str = Field(default="", description="Python version")
    platform: str = Field(default="", description="Platform/OS")
    docker_image: str | None = Field(default=None, description="Docker image used")
    dependencies: dict[str, str] = Field(
        default_factory=dict,
        description="Key dependency versions",
    )


class ConfigFingerprintV1(BaseModel):
    """Fingerprint of the configuration used."""

    llm_config_hash: str | None = Field(default=None, description="Hash of llm.yaml")
    benchmark_config_hash: str | None = Field(default=None, description="Hash of benchmark.yaml")
    skill_hash: str | None = Field(default=None, description="Hash of skill file")
    skill_name: str | None = Field(default=None, description="Skill file name")


class ResultSummaryV1(BaseModel):
    """Summary statistics for benchmark results."""

    total_tasks: int = Field(default=0, description="Total number of tasks")
    completed: int = Field(default=0, description="Completed evaluations")
    passed: int = Field(default=0, description="Passed evaluations")
    failed: int = Field(default=0, description="Failed evaluations")
    errors: int = Field(default=0, description="Evaluations with errors")
    avg_score: float = Field(default=0.0, description="Average score")
    avg_latency_s: float = Field(default=0.0, description="Average latency in seconds")
    total_tokens: int = Field(default=0, description="Total tokens used")

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.completed == 0:
            return 0.0
        return self.passed / self.completed


class ModelSummaryV1(BaseModel):
    """Per-model result summary."""

    model: str = Field(description="Model name")
    total: int = Field(default=0, description="Total evaluations")
    passed: int = Field(default=0, description="Passed evaluations")
    avg_score: float = Field(default=0.0, description="Average score")
    avg_latency_s: float = Field(default=0.0, description="Average latency")
    total_tokens: int = Field(default=0, description="Total tokens used")

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


class ManifestV1(BaseModel):
    """
    Canonical run manifest schema.

    The manifest captures all metadata about a benchmark run,
    including configuration, environment, and result summaries.
    """

    # Schema versioning
    schema_version: str = Field(default="1.0", description="Schema version")

    # Run identification
    run_id: str = Field(description="Unique identifier for this run")
    run_name: str | None = Field(default=None, description="Human-readable run name")

    # Configuration
    models: list[str] = Field(default_factory=list, description="Models evaluated")
    task_count: int = Field(default=0, description="Number of tasks")
    skill_version: str = Field(default="none", description="Skill version or identifier")

    # Fingerprints
    config_fingerprint: ConfigFingerprintV1 | None = Field(
        default=None,
        description="Configuration fingerprint",
    )
    environment_fingerprint: EnvironmentFingerprintV1 | None = Field(
        default=None,
        description="Environment fingerprint",
    )

    # Git provenance
    git_sha: str = Field(default="unknown", description="Git commit SHA")
    git_branch: str | None = Field(default=None, description="Git branch")
    git_dirty: bool = Field(default=False, description="Whether repo has uncommitted changes")

    # Timestamps
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the run started",
    )
    end_timestamp: str | None = Field(default=None, description="When the run ended")
    duration_s: float | None = Field(default=None, description="Run duration in seconds")

    # Summary statistics
    summary: ResultSummaryV1 = Field(
        default_factory=ResultSummaryV1,
        description="Overall result summary",
    )
    model_summaries: list[ModelSummaryV1] = Field(
        default_factory=list,
        description="Per-model result summaries",
    )

    # Configuration snapshot
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration snapshot")

    # Results reference
    results_path: str | None = Field(default=None, description="Path to results file")
    results: list[ResultV1] = Field(default_factory=list, description="Individual results")

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Ensure schema version is valid."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported schema version: {v}. Expected 1.x")
        return v

    def to_json_schema(self) -> dict[str, Any]:
        """Export this model's JSON schema."""
        return ManifestV1.model_json_schema()

    def add_result(self, result: ResultV1) -> None:
        """Add a result to the manifest."""
        self.results.append(result)
        self._update_summary()

    def _update_summary(self) -> None:
        """Update summary statistics from results."""
        if not self.results:
            return

        self.summary.total_tasks = self.task_count
        self.summary.completed = len(self.results)
        self.summary.passed = sum(1 for r in self.results if r.passed)
        self.summary.failed = self.summary.completed - self.summary.passed
        self.summary.errors = sum(1 for r in self.results if r.error_category)
        self.summary.avg_score = sum(r.score for r in self.results) / len(self.results)
        self.summary.avg_latency_s = sum(r.latency_s for r in self.results) / len(self.results)
        self.summary.total_tokens = sum(r.token_usage.total for r in self.results)

        # Update model summaries
        model_results: dict[str, list[ResultV1]] = {}
        for r in self.results:
            if r.model not in model_results:
                model_results[r.model] = []
            model_results[r.model].append(r)

        self.model_summaries = [
            ModelSummaryV1(
                model=model,
                total=len(results),
                passed=sum(1 for r in results if r.passed),
                avg_score=sum(r.score for r in results) / len(results),
                avg_latency_s=sum(r.latency_s for r in results) / len(results),
                total_tokens=sum(r.token_usage.total for r in results),
            )
            for model, results in sorted(model_results.items())
        ]

    def finalize(self, end_time: datetime | None = None) -> None:
        """Finalize the manifest with end time and duration."""
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        self.end_timestamp = end_time.isoformat()

        if self.timestamp:
            start = datetime.fromisoformat(self.timestamp)
            self.duration_s = (end_time - start).total_seconds()

        self._update_summary()

    def save(self, path: str) -> None:
        """Save the manifest to a JSON file."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = self.model_dump(mode="json")

        # Convert results to dicts
        data["results"] = [r.model_dump(mode="json") for r in self.results]

        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "ManifestV1":
        """Load a manifest from a JSON file."""
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())

        # Convert results back to ResultV1 objects
        results_data = data.pop("results", [])
        results = [ResultV1.model_validate(r) for r in results_data]

        manifest = cls.model_validate(data)
        manifest.results = results

        return manifest

    @classmethod
    def from_legacy_benchmark_run(cls, data: dict[str, Any]) -> "ManifestV1":
        """
        Convert from legacy BenchmarkRun format to ManifestV1.

        Args:
            data: Dictionary from BenchmarkRun save format

        Returns:
            ManifestV1 instance
        """
        from bench.schema.v1.results import ResultV1

        # Convert results
        results = [ResultV1.from_legacy_benchmark_result(r) for r in data.get("results", [])]

        # Create summary
        summary = ResultSummaryV1(
            total_tasks=data.get("task_count", 0),
            completed=len(results),
            passed=sum(1 for r in results if r.passed),
            avg_score=sum(r.score for r in results) / len(results) if results else 0.0,
            avg_latency_s=sum(r.latency_s for r in results) / len(results) if results else 0.0,
        )

        return cls(
            run_id=data.get("run_id", ""),
            models=data.get("models", []),
            task_count=data.get("task_count", 0),
            skill_version=data.get("skill_version", "none"),
            git_sha=data.get("git_sha", "unknown"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            config=data.get("config", {}),
            summary=summary,
            results=results,
        )

    @classmethod
    def create_new(
        cls,
        run_id: str,
        models: list[str],
        task_count: int,
        skill_version: str = "none",
    ) -> "ManifestV1":
        """
        Create a new manifest with automatic environment detection.

        Args:
            run_id: Unique run identifier
            models: List of models to evaluate
            task_count: Number of tasks
            skill_version: Skill version or identifier

        Returns:
            ManifestV1 instance
        """
        import platform
        import sys

        # Get git info
        git_sha = cls._get_git_sha()
        git_branch = cls._get_git_branch()
        git_dirty = cls._is_git_dirty()

        # Environment fingerprint
        env_fp = EnvironmentFingerprintV1(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
        )

        return cls(
            run_id=run_id,
            models=models,
            task_count=task_count,
            skill_version=skill_version,
            git_sha=git_sha,
            git_branch=git_branch,
            git_dirty=git_dirty,
            environment_fingerprint=env_fp,
        )

    @staticmethod
    def _get_git_sha() -> str:
        """Get current git SHA."""
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

    @staticmethod
    def _get_git_branch() -> str | None:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _is_git_dirty() -> bool:
        """Check if repo has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return bool(result.stdout.strip()) if result.returncode == 0 else False
        except Exception:
            return False


def validate_manifest(data: dict[str, Any]) -> ManifestV1:
    """
    Validate manifest data and return a ManifestV1 instance.

    Args:
        data: Raw manifest data dictionary

    Returns:
        Validated ManifestV1 instance

    Raises:
        ValidationError: If data doesn't conform to ManifestV1 schema
    """
    return ManifestV1.model_validate(data)
