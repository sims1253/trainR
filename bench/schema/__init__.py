"""Canonical schema package for benchmark data structures."""

from bench.schema.v1 import (
    ManifestV1,
    ProfileV1,
    ResultV1,
    TaskV1,
    adapt_from_legacy_result,
    adapt_from_legacy_task,
    load_json_schema,
    validate_manifest,
    validate_profile,
    validate_result,
    validate_task,
)

__all__ = [
    "ManifestV1",
    "ProfileV1",
    "ResultV1",
    "TaskV1",
    "adapt_from_legacy_result",
    "adapt_from_legacy_task",
    "load_json_schema",
    "validate_manifest",
    "validate_profile",
    "validate_result",
    "validate_task",
]
