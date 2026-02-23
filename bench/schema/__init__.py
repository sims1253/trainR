"""Canonical schema package for benchmark data structures."""

from bench.schema.v1 import (
    TaskV1,
    ProfileV1,
    ResultV1,
    ManifestV1,
    validate_task,
    validate_result,
    validate_profile,
    validate_manifest,
    load_json_schema,
    adapt_from_legacy_result,
    adapt_from_legacy_task,
)

__all__ = [
    "TaskV1",
    "ProfileV1",
    "ResultV1",
    "ManifestV1",
    "validate_task",
    "validate_result",
    "validate_profile",
    "validate_manifest",
    "load_json_schema",
    "adapt_from_legacy_result",
    "adapt_from_legacy_task",
]
