"""Canonical schema v1 module.

This module provides the canonical schemas for benchmark data structures:
- TaskV1: Task definition schema
- ProfileV1: Agent/tool profile schema
- ResultV1: Evaluation result schema
- ManifestV1: Run manifest schema

All schemas support:
- Versioning via schema_version field
- JSON schema export
- Validation helpers
- Legacy format adapters
"""

from pathlib import Path
from typing import Any

from bench.schema.v1.manifest import (
    ConfigFingerprintV1,
    EnvironmentFingerprintV1,
    ManifestV1,
    ModelSummaryV1,
    ResultSummaryV1,
    validate_manifest,
)
from bench.schema.v1.profiles import (
    ExecutionConfigV1,
    JudgeConfigV1,
    JudgeModeV1,
    ModelCapabilityV1,
    ModelConfigV1,
    ProfileTypeV1,
    ProfileV1,
    SkillConfigV1,
    VotingStrategyV1,
    WorkerConfigV1,
    validate_profile,
)
from bench.schema.v1.results import (
    ErrorCategoryV1,
    ResultV1,
    CaseResultV1,
    TokenUsageV1,
    validate_result,
)
from bench.schema.v1.task import (
    DifficultyV1,
    TaskFilesV1,
    TaskSolutionV1,
    TaskSourceV1,
    TaskTestsV1,
    TaskTypeV1,
    TaskV1,
    validate_task,
)

__all__ = [
    # Task schemas
    "TaskV1",
    "TaskTypeV1",
    "DifficultyV1",
    "TaskSourceV1",
    "TaskTestsV1",
    "TaskSolutionV1",
    "TaskFilesV1",
    "validate_task",
    # Profile schemas
    "ProfileV1",
    "ProfileTypeV1",
    "ModelConfigV1",
    "ModelCapabilityV1",
    "SkillConfigV1",
    "JudgeConfigV1",
    "JudgeModeV1",
    "VotingStrategyV1",
    "ExecutionConfigV1",
    "WorkerConfigV1",
    "validate_profile",
    # Result schemas
    "ResultV1",
    "ErrorCategoryV1",
    "CaseResultV1",
    "TokenUsageV1",
    "validate_result",
    # Manifest schemas
    "ManifestV1",
    "ResultSummaryV1",
    "ModelSummaryV1",
    "ConfigFingerprintV1",
    "EnvironmentFingerprintV1",
    "validate_manifest",
    # Helpers
    "load_json_schema",
    "adapt_from_legacy_result",
    "adapt_from_legacy_task",
]

# JSON Schema directory
_JSON_SCHEMA_DIR = Path(__file__).parent / "jsonschemas"


def load_json_schema(schema_type: str) -> dict[str, Any]:
    """
    Load a JSON schema file by type.

    Args:
        schema_type: One of "task", "profile", "result", "manifest"

    Returns:
        JSON schema dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema_type is invalid
    """
    import json

    valid_types = {"task", "profile", "result", "manifest"}
    if schema_type not in valid_types:
        raise ValueError(f"Invalid schema type: {schema_type}. Must be one of {valid_types}")

    schema_path = _JSON_SCHEMA_DIR / f"{schema_type}.schema.json"

    if not schema_path.exists():
        # Generate schema on-the-fly if file doesn't exist
        return _generate_json_schema(schema_type)

    return json.loads(schema_path.read_text())


def _generate_json_schema(schema_type: str) -> dict[str, Any]:
    """Generate JSON schema from Pydantic model."""
    schema_map = {
        "task": TaskV1,
        "profile": ProfileV1,
        "result": ResultV1,
        "manifest": ManifestV1,
    }

    model = schema_map.get(schema_type)
    if not model:
        raise ValueError(f"Unknown schema type: {schema_type}")

    return model.model_json_schema()


def adapt_from_legacy_result(data: dict[str, Any], source: str = "benchmark") -> ResultV1:
    """
    Adapt legacy result format to ResultV1.

    Args:
        data: Legacy result data
        source: Source format ("benchmark" or "evaluation")

    Returns:
        ResultV1 instance
    """
    if source == "evaluation":
        return ResultV1.from_legacy_evaluation_result(data)
    return ResultV1.from_legacy_benchmark_result(data)


def adapt_from_legacy_task(data: dict[str, Any], source: str = "mined") -> TaskV1:
    """
    Adapt legacy task format to TaskV1.

    Args:
        data: Legacy task data
        source: Source format ("mined" or "testing")

    Returns:
        TaskV1 instance
    """
    if source == "testing":
        return TaskV1.from_legacy_testing_task(data)
    return TaskV1.from_legacy_mined_task(data)


def export_all_json_schemas(output_dir: Path | None = None) -> None:
    """
    Export all JSON schemas to files.

    Args:
        output_dir: Directory to write schema files (defaults to jsonschemas/)
    """
    import json

    if output_dir is None:
        output_dir = _JSON_SCHEMA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = {
        "task": TaskV1,
        "profile": ProfileV1,
        "result": ResultV1,
        "manifest": ManifestV1,
    }

    for name, model in schemas.items():
        schema_path = output_dir / f"{name}.schema.json"
        schema = model.model_json_schema()
        schema_path.write_text(json.dumps(schema, indent=2))
        print(f"Exported {name} schema to {schema_path}")
