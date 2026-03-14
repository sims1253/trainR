"""Tests for provider preflight validation behavior."""

from __future__ import annotations

from unittest.mock import patch

from bench.provider.preflight import run_preflight, validate_model_provider_config


class TestStrictPreflight:
    """Tests for strict preflight mode behavior."""

    def test_strict_mode_unknown_model_is_error(self) -> None:
        """In strict mode, unknown model should be an error not warning."""
        result = run_preflight(models=["nonexistent-model-xyz"], strict=True)

        assert result.is_valid is False
        assert any("unknown model" in e.lower() for e in result.errors)

    def test_non_strict_mode_unknown_model_is_warning(self) -> None:
        """In non-strict mode, unknown model should be a warning."""
        result = run_preflight(models=["nonexistent-model-xyz"], strict=False)

        assert result.is_valid is True  # Warnings don't fail validation
        assert any("unknown model" in w.lower() for w in result.warnings)


@patch("bench.provider.preflight.CredentialResolver.resolve")
def test_validate_model_provider_config_unknown_model_is_structured(
    mock_resolve,
) -> None:
    """Unknown model validation should return a structured error."""
    # Ensure no accidental credential lookup is required for this test.
    mock_resolve.return_value.is_valid = False

    result = validate_model_provider_config("definitely-unknown-model")

    assert result.is_valid is False
    assert result.errors == ["Unknown model: definitely-unknown-model"]
