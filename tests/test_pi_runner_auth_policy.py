"""Tests for auth policy behavior in DockerPiRunner."""

from __future__ import annotations

import pytest

from bench.provider import AuthPolicy
from bench.provider.auth import CredentialResolver
from config import get_llm_config
from evaluation.pi_runner import (
    DockerPiRunner,
    DockerPiRunnerConfig,
    _resolve_credential_value,
    _resolve_pi_model,
)


class TestMountedFileAuthStrictness:
    """Tests for mounted_file auth strictness."""

    def test_mounted_file_strict_fails_without_fallback(self) -> None:
        """mounted_file policy should fail if file not found (no fallback)."""
        resolver = CredentialResolver(
            policy=AuthPolicy.MOUNTED_FILE,
            allow_env_fallback=False,
        )

        result = resolver.resolve("TEST_PROVIDER_API_KEY")

        assert result.value is None
        assert result.is_valid is False

    def test_mounted_file_with_fallback_uses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """mounted_file with fallback should use env when file not found."""
        monkeypatch.setenv("TEST_FALLBACK_API_KEY", "test-key-value")

        resolver = CredentialResolver(
            policy=AuthPolicy.MOUNTED_FILE,
            allow_env_fallback=True,
        )

        result = resolver.resolve("TEST_FALLBACK_API_KEY")

        assert result.value == "test-key-value"
        assert result.is_valid is True


def test_resolve_credential_value_uses_requested_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential lookup should pass through the requested auth policy."""
    calls: list[tuple[str, AuthPolicy]] = []

    def fake_get_credentials(env_var: str, policy: AuthPolicy = AuthPolicy.ENV) -> str | None:
        calls.append((env_var, policy))
        return "secret"

    monkeypatch.setattr("bench.provider.get_credentials", fake_get_credentials)

    value = _resolve_credential_value("OPENROUTER_API_KEY", "mounted_file")

    assert value == "secret"
    assert calls == [("OPENROUTER_API_KEY", AuthPolicy.MOUNTED_FILE)]


def test_docker_runner_config_rejects_invalid_auth_policy() -> None:
    """Runner config should validate auth policy values."""
    with pytest.raises(ValueError, match="Invalid auth_policy"):
        DockerPiRunnerConfig(auth_policy="invalid")


class TestPiModelResolution:
    """Tests for provider-aware Pi model resolution."""

    def test_resolve_pi_model_from_short_name(self) -> None:
        """Short llm.yaml model names should map to provider-prefixed Pi models."""
        llm_config = get_llm_config()

        provider, pi_model = _resolve_pi_model(llm_config, "gpt-5-nano")

        assert provider == "opencode"
        assert pi_model == "opencode/gpt-5-nano"

    def test_resolve_pi_model_from_legacy_litellm_alias(self) -> None:
        """Legacy openai-prefixed aliases should map back to canonical provider prefixes."""
        llm_config = get_llm_config()

        provider, pi_model = _resolve_pi_model(llm_config, "openai/glm-5")

        assert provider == "zai_coding_plan"
        assert pi_model == "zai/glm-5"

    def test_resolve_pi_model_prefixed_passthrough(self) -> None:
        """Explicit provider-prefixed IDs should preserve provider/prefix."""
        llm_config = get_llm_config()

        provider, pi_model = _resolve_pi_model(llm_config, "openrouter/openai/gpt-oss-120b:free")

        assert provider == "openrouter"
        assert pi_model == "openrouter/openai/gpt-oss-120b:free"


def test_sanitize_agent_error_compacts_oversized_payloads() -> None:
    """Malformed multi-line payloads should collapse to a short summary."""
    noisy = 'error"\n [98] "as.character.rlang_message"\n [99] "as.character.rlang_warning"'

    assert (
        DockerPiRunner._sanitize_agent_error(noisy)
        == "Agent emitted oversized structured error payload"
    )


def test_malformed_tool_payload_is_non_fatal_after_clean_success() -> None:
    """A terminal stream artifact should not fail a clean passing run."""
    assert DockerPiRunner._is_non_fatal_agent_error(
        "Agent emitted malformed tool payload in error stream",
        returncode=0,
        test_result={"passed": True},
    )


def test_malformed_tool_payload_is_still_fatal_when_tests_fail() -> None:
    """The same artifact should not mask an actual failed run."""
    assert not DockerPiRunner._is_non_fatal_agent_error(
        "Agent emitted malformed tool payload in error stream",
        returncode=0,
        test_result={"passed": False},
    )


def test_detect_repetitive_read_loop_returns_concise_summary() -> None:
    """Repeated read calls on the same path should be summarized cleanly."""
    output = "\n".join(
        (
            '{"type":"tool_execution_start","toolCallId":"call-'
            f'{idx}","toolName":"read","args":{{"path":"'
            '/workspace/package/tests/testthat/helper-cli.R"}}'
        )
        for idx in range(10)
    )

    assert (
        DockerPiRunner._detect_repetitive_read_loop(output)
        == "Detected repetitive file-read loop on /workspace/package/tests/testthat/helper-cli.R"
    )


def test_detect_repetitive_read_loop_ignores_duplicate_events_for_same_call() -> None:
    """The detector should not count repeated log events for one tool call as a loop."""
    repeated_line = (
        '{"type":"tool_execution_start","toolCallId":"call-1","toolName":"read",'
        '"args":{"path":"/workspace/package/tests/testthat/helper-cli.R"}}'
    )
    output = "\n".join(repeated_line for _ in range(8))

    assert DockerPiRunner._detect_repetitive_read_loop(output) is None


def test_sanitize_output_for_artifacts_truncates_large_lines() -> None:
    """Stored output should be bounded even when a tool emits massive lines."""
    long_line = "x" * (DockerPiRunner._MAX_OUTPUT_LINE_CHARS + 200)

    sanitized = DockerPiRunner._sanitize_output_for_artifacts(long_line)

    assert "... [truncated]" in sanitized
    assert len(sanitized) <= DockerPiRunner._MAX_OUTPUT_LINE_CHARS
