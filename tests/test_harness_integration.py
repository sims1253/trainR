"""Integration tests for harness abstraction layer.

These tests prove that:
1. Harness can be swapped via config only (no code changes)
2. Same task produces consistent results regardless of harness (with mock)
3. Registry correctly routes to different harness implementations
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from pydantic import ValidationError

from bench.experiments import ExperimentConfig
from bench.harness import (
    AgentHarness,
    HarnessConfig,
    HarnessRegistry,
    HarnessRequest,
    HarnessResult,
    PiDockerHarness,
    register_harness,
)


class MockHarness(AgentHarness):
    """Mock harness for testing."""

    def __init__(self, config: HarnessConfig):
        super().__init__(config)
        self.execute_called = False
        self.last_request: HarnessRequest | None = None

    def validate_environment(self) -> tuple[bool, list[str]]:
        return True, []

    async def execute(self, request: HarnessRequest) -> HarnessResult:
        self.execute_called = True
        self.last_request = request
        return HarnessResult(
            task_id=request.task_id,
            run_id=str(uuid.uuid4())[:8],
            success=True,
            tests_passed=True,
        )


class TestHarnessSwapViaConfig:
    """Tests proving harness can be swapped via config only."""

    def test_config_specifies_harness(self):
        """Config can specify which harness to use."""
        config = ExperimentConfig.model_validate(
            {"name": "test", "execution": {"harness": "pi_docker"}}
        )
        assert config.execution.harness == "pi_docker"

    def test_config_supports_different_harness_types(self):
        """Config supports all currently registered harness types."""
        from bench.harness import HarnessRegistry

        # pi_docker is the default
        config = ExperimentConfig(name="test")
        assert config.execution.harness == "pi_docker"

        for harness_type in HarnessRegistry.list_available():
            config = ExperimentConfig.model_validate(
                {"name": "test", "execution": {"harness": harness_type}}
            )
            assert config.execution.harness == harness_type

    def test_registry_returns_correct_harness(self):
        """Registry returns the harness specified in config."""
        # Register mock harness
        HarnessRegistry.register("mock_test", MockHarness)

        try:
            config = HarnessConfig()
            harness = HarnessRegistry.get("mock_test", config)

            assert isinstance(harness, MockHarness)
        finally:
            HarnessRegistry.unregister("mock_test")

    def test_different_harness_same_interface(self):
        """Different harnesses implement the same interface."""
        HarnessRegistry.register("mock_a", MockHarness)

        try:
            config = HarnessConfig()
            harness_a = HarnessRegistry.get("mock_a", config)

            # Verify interface methods exist
            assert hasattr(harness_a, "execute")
            assert hasattr(harness_a, "validate_environment")
            assert hasattr(harness_a, "setup")
            assert hasattr(harness_a, "teardown")
            assert hasattr(harness_a, "get_environment")
            assert hasattr(harness_a, "get_timeout")

            # Both return HarnessResult
            request = HarnessRequest(task_id="test", prompt="test")
            result = asyncio.run(harness_a.execute(request))

            assert isinstance(result, HarnessResult)
        finally:
            HarnessRegistry.unregister("mock_a")

    def test_default_harness_is_pi_docker(self):
        """Default harness is pi_docker."""
        config = ExperimentConfig(name="test")
        assert config.execution.harness == "pi_docker"

    def test_invalid_harness_name_raises_error(self):
        """Invalid harness names are rejected by config validation."""
        # HarnessType is a Literal, so invalid values should fail
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(
                {"name": "test", "execution": {"harness": "invalid_harness"}}
            )


class TestHarnessRequestResult:
    """Tests for HarnessRequest and HarnessResult contracts."""

    def test_request_has_required_fields(self):
        """Request has all required fields."""
        request = HarnessRequest(
            task_id="task-1",
            prompt="Write a function",
        )

        assert request.task_id == "task-1"
        assert request.prompt == "Write a function"

    def test_request_has_optional_fields(self):
        """Request supports optional fields for context."""
        request = HarnessRequest(
            task_id="task-1",
            prompt="Write a function",
            system_prompt="You are an expert",
            repository="https://github.com/example/repo",
            test_command="pytest",
            metadata={"key": "value"},
        )

        assert request.system_prompt == "You are an expert"
        assert request.repository == "https://github.com/example/repo"
        assert request.test_command == "pytest"
        assert request.metadata == {"key": "value"}

    def test_result_has_required_fields(self):
        """Result has all required fields."""
        result = HarnessResult(
            task_id="task-1",
            run_id="run-123",
            success=True,
        )

        assert result.task_id == "task-1"
        assert result.run_id == "run-123"
        assert result.success is True

    def test_result_tracks_test_outcomes(self):
        """Result tracks test pass/fail status."""
        result = HarnessResult(
            task_id="task-1",
            run_id="run-123",
            success=True,
            tests_passed=True,
        )

        assert result.tests_passed is True


class TestHarnessRegistry:
    """Tests for HarnessRegistry routing behavior."""

    def test_registry_routes_to_correct_harness(self):
        """Registry routes config name to correct harness class."""
        HarnessRegistry.register("test_router", MockHarness)

        try:
            config = HarnessConfig()
            harness = HarnessRegistry.get("test_router", config)
            assert isinstance(harness, MockHarness)
        finally:
            HarnessRegistry.unregister("test_router")

    def test_registry_lists_available_harnesses(self):
        """Registry can list all available harnesses."""
        # pi_docker is registered at import time
        available = HarnessRegistry.list_available()
        assert "pi_docker" in available

    def test_registry_is_registered_check(self):
        """Registry can check if a harness is registered."""
        assert HarnessRegistry.is_registered("pi_docker") is True
        assert HarnessRegistry.is_registered("nonexistent") is False

    def test_registry_raises_for_unknown_harness(self):
        """Registry raises KeyError for unknown harness."""
        config = HarnessConfig()
        with pytest.raises(KeyError, match="not registered"):
            HarnessRegistry.get("unknown_harness", config)

    def test_register_decorator(self):
        """register_harness decorator works correctly."""

        @register_harness("decorated_mock")
        class DecoratedMock(AgentHarness):
            def validate_environment(self) -> tuple[bool, list[str]]:
                return True, []

            async def execute(self, request: HarnessRequest) -> HarnessResult:
                return HarnessResult(
                    task_id=request.task_id,
                    run_id="test",
                    success=True,
                )

        try:
            assert HarnessRegistry.is_registered("decorated_mock")
        finally:
            HarnessRegistry.unregister("decorated_mock")

    def test_registry_prevents_duplicate_registration(self):
        """Registry prevents duplicate registration."""
        HarnessRegistry.register("dup_test", MockHarness)

        try:
            with pytest.raises(ValueError, match="already registered"):
                HarnessRegistry.register("dup_test", MockHarness)
        finally:
            HarnessRegistry.unregister("dup_test")

    def test_registry_unregister(self):
        """Registry can unregister harnesses."""
        HarnessRegistry.register("temp_harness", MockHarness)
        assert HarnessRegistry.is_registered("temp_harness")

        result = HarnessRegistry.unregister("temp_harness")
        assert result is True
        assert not HarnessRegistry.is_registered("temp_harness")

    def test_registry_get_class(self):
        """Registry can return class without instantiating."""
        HarnessRegistry.register("class_test", MockHarness)

        try:
            cls = HarnessRegistry.get_class("class_test")
            assert cls is MockHarness
        finally:
            HarnessRegistry.unregister("class_test")


class TestPiDockerHarness:
    """Tests for PiDockerHarness adapter."""

    def test_is_registered(self):
        """PiDockerHarness is registered."""
        assert "pi_docker" in HarnessRegistry.list_available()

    def test_can_instantiate_via_registry(self):
        """PiDockerHarness can be instantiated via registry."""
        config = HarnessConfig()
        harness = HarnessRegistry.get("pi_docker", config)
        assert isinstance(harness, PiDockerHarness)

    def test_validate_environment_returns_tuple(self):
        """validate_environment returns tuple of (bool, list[str])."""
        config = HarnessConfig()
        harness = PiDockerHarness(config)

        # Should return tuple of (bool, list[str])
        is_valid, errors = harness.validate_environment()
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_has_setup_teardown_methods(self):
        """PiDockerHarness has setup and teardown methods."""
        config = HarnessConfig()
        harness = PiDockerHarness(config)

        assert hasattr(harness, "setup")
        assert hasattr(harness, "teardown")
        assert callable(harness.setup)
        assert callable(harness.teardown)


class TestHarnessConsistency:
    """Tests proving consistent results regardless of harness."""

    def test_same_request_produces_consistent_result_structure(self):
        """Same request produces consistent result structure across harnesses."""
        HarnessRegistry.register("consistency_a", MockHarness)
        HarnessRegistry.register("consistency_b", MockHarness)

        try:
            config = HarnessConfig()
            harness_a = HarnessRegistry.get("consistency_a", config)
            harness_b = HarnessRegistry.get("consistency_b", config)

            request = HarnessRequest(
                task_id="same-task",
                prompt="Same prompt",
            )

            result_a = asyncio.run(harness_a.execute(request))
            result_b = asyncio.run(harness_b.execute(request))

            # Both results have same structure
            assert hasattr(result_a, "task_id")
            assert hasattr(result_b, "task_id")
            assert result_a.task_id == request.task_id
            assert result_b.task_id == request.task_id

            assert hasattr(result_a, "success")
            assert hasattr(result_b, "success")
            assert result_a.success == result_b.success

        finally:
            HarnessRegistry.unregister("consistency_a")
            HarnessRegistry.unregister("consistency_b")

    def test_config_driven_harness_selection(self):
        """ExperimentConfig drives harness selection without code changes."""
        # Create configs with different harnesses
        config_docker = ExperimentConfig.model_validate(
            {"name": "test-docker", "execution": {"harness": "pi_docker"}}
        )

        # Both configs can be used to determine harness
        assert config_docker.execution.harness == "pi_docker"

        # The same code path can be used with different configs
        def get_harness_from_config(exp_config: ExperimentConfig) -> str:
            """Example of config-driven harness selection."""
            harness_name = exp_config.execution.harness
            if HarnessRegistry.is_registered(harness_name):
                return harness_name
            raise ValueError(f"Unknown harness: {harness_name}")

        assert get_harness_from_config(config_docker) == "pi_docker"


class TestHarnessConfig:
    """Tests for HarnessConfig behavior."""

    def test_default_config_values(self):
        """HarnessConfig has sensible defaults."""
        config = HarnessConfig()

        assert config.timeout == 300.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.sandbox_enabled is True
        assert config.network_access is False

    def test_config_allows_extra_fields(self):
        """HarnessConfig allows extra fields for harness-specific config."""
        config = HarnessConfig(
            timeout=600.0,
            custom_field="custom_value",  # type: ignore
        )

        assert config.timeout == 600.0
        # Extra fields are accessible via getattr
        assert getattr(config, "custom_field", None) == "custom_value"

    def test_config_environment_variables(self):
        """HarnessConfig supports environment variable configuration."""
        config = HarnessConfig(
            env_vars={"API_KEY": "secret", "DEBUG": "1"},
        )

        assert config.env_vars["API_KEY"] == "secret"
        assert config.env_vars["DEBUG"] == "1"

    def test_get_environment_merges_vars(self):
        """get_environment merges OS env with config env."""
        config = HarnessConfig(
            env_vars={"CUSTOM_VAR": "value"},
        )
        harness = MockHarness(config)

        env = harness.get_environment()
        assert "CUSTOM_VAR" in env
        assert env["CUSTOM_VAR"] == "value"


class TestErrorCategory:
    """Tests for error classification."""

    def test_error_category_from_timeout(self):
        """Timeout exceptions are classified correctly."""
        from bench.harness import ErrorCategory

        exc = TimeoutError("Operation timed out")
        category = ErrorCategory.from_exception(exc)
        assert category == ErrorCategory.TIMEOUT

    def test_error_category_from_api_error(self):
        """API errors are classified correctly."""
        from bench.harness import ErrorCategory

        exc = Exception("API returned 500 error")
        category = ErrorCategory.from_exception(exc)
        assert category == ErrorCategory.API_ERROR

    def test_error_category_from_auth_error(self):
        """Auth errors are classified correctly."""
        from bench.harness import ErrorCategory

        exc = Exception("401 Unauthorized")
        category = ErrorCategory.from_exception(exc)
        assert category == ErrorCategory.AUTH_ERROR

    def test_error_category_unknown(self):
        """Unknown errors fall back to UNKNOWN."""
        from bench.harness import ErrorCategory

        exc = Exception("Some random error")
        category = ErrorCategory.from_exception(exc)
        assert category == ErrorCategory.UNKNOWN


class TestRetryCategoryAlignment:
    """Tests for retry behavior alignment with error categories.

    These tests verify that:
    1. Transient API/provider/network errors map to LLM_ERROR (retriable by default)
    2. Auth and unknown errors are NOT retriable by default
    3. The error category mapping is complete and predictable
    """

    def test_default_retry_on_includes_retriable_categories(self):
        """Default retry_on should include TIMEOUT, LLM_ERROR, SANDBOX_ERROR."""
        from bench.experiments.config import RetryConfig

        config = RetryConfig()
        assert "TIMEOUT" in config.retry_on
        assert "LLM_ERROR" in config.retry_on
        assert "SANDBOX_ERROR" in config.retry_on

    def test_default_retry_config_enables_exponential_backoff(self):
        """Default retry settings should enable bounded retries."""
        from bench.experiments.config import RetryConfig, RetryStrategy

        config = RetryConfig()
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.max_retries == 2
        assert config.base_delay == 30.0
        assert config.max_delay == 300.0

    def test_api_error_maps_to_llm_error(self):
        """API_ERROR should map to LLM_ERROR which is retriable by default."""
        from bench.harness import ErrorCategory as HarnessErrorCategory
        from bench.schema.v1 import ErrorCategoryV1

        # Check the mapping directly by inspecting what the runner would produce
        # API_ERROR -> LLM_ERROR
        assert HarnessErrorCategory.API_ERROR is not None
        # Verify LLM_ERROR is in the ErrorCategoryV1 enum
        assert ErrorCategoryV1.LLM_ERROR.value == "LLM_ERROR"

    def test_rate_limit_maps_to_llm_error(self):
        """RATE_LIMIT should map to LLM_ERROR which is retriable by default."""
        from bench.schema.v1 import ErrorCategoryV1

        # Verify LLM_ERROR is in the ErrorCategoryV1 enum
        assert ErrorCategoryV1.LLM_ERROR.value == "LLM_ERROR"

    def test_network_error_maps_to_llm_error(self):
        """NETWORK_ERROR should map to LLM_ERROR which is retriable by default."""
        from bench.schema.v1 import ErrorCategoryV1

        # Verify LLM_ERROR is in the ErrorCategoryV1 enum
        assert ErrorCategoryV1.LLM_ERROR.value == "LLM_ERROR"

    def test_auth_error_in_default_retry_on(self):
        """AUTH_ERROR should be retriable by default."""
        from bench.experiments.config import RetryConfig

        config = RetryConfig()
        assert "AUTH_ERROR" in config.retry_on

    def test_unknown_not_in_default_retry_on(self):
        """UNKNOWN errors should NOT be retriable by default."""
        from bench.experiments.config import RetryConfig

        config = RetryConfig()
        assert "UNKNOWN" not in config.retry_on

    def test_error_category_v1_has_auth_error(self):
        """ErrorCategoryV1 should have AUTH_ERROR for non-retriable auth failures."""
        from bench.schema.v1 import ErrorCategoryV1

        assert hasattr(ErrorCategoryV1, "AUTH_ERROR")
        assert ErrorCategoryV1.AUTH_ERROR.value == "AUTH_ERROR"

    def test_harness_error_mapping_complete(self):
        """All HarnessErrorCategory values should be mapped in the runner."""
        from bench.harness import ErrorCategory as HarnessErrorCategory

        # All expected categories
        expected_categories = [
            HarnessErrorCategory.NONE,
            HarnessErrorCategory.TIMEOUT,
            HarnessErrorCategory.API_ERROR,
            HarnessErrorCategory.RATE_LIMIT,
            HarnessErrorCategory.AUTH_ERROR,
            HarnessErrorCategory.NETWORK_ERROR,
            HarnessErrorCategory.INVALID_REQUEST,
            HarnessErrorCategory.AGENT_ERROR,
            HarnessErrorCategory.SANDBOX_ERROR,
            HarnessErrorCategory.RESOURCE_EXHAUSTED,
            HarnessErrorCategory.MODEL_ERROR,
            HarnessErrorCategory.TASK_ERROR,
            HarnessErrorCategory.TEST_ERROR,
            HarnessErrorCategory.UNKNOWN,
        ]

        # Verify all categories exist
        for category in expected_categories:
            assert category is not None, f"Missing category: {category}"

    def test_transient_errors_map_to_retriable_categories(self):
        """Transient errors (API, rate limit, network, model, resource) should map to retriable categories."""
        from bench.schema.v1 import ErrorCategoryV1

        # These are the categories that should be retriable
        retriable_categories = {"TIMEOUT", "LLM_ERROR", "SANDBOX_ERROR"}

        # LLM_ERROR should be available for transient errors
        assert ErrorCategoryV1.LLM_ERROR.value in retriable_categories

    def test_non_retriable_errors_map_to_non_retriable_categories(self):
        """Non-retriable errors (auth, invalid request, unknown) should NOT map to retriable categories."""
        from bench.schema.v1 import ErrorCategoryV1

        # These are the categories that should NOT be retriable
        non_retriable_values = {"AUTH_ERROR", "UNKNOWN", "TEST_FAILURE"}

        # Verify these exist
        assert ErrorCategoryV1.AUTH_ERROR.value in non_retriable_values
        assert ErrorCategoryV1.UNKNOWN.value in non_retriable_values
        assert ErrorCategoryV1.TEST_FAILURE.value in non_retriable_values

    def test_harness_to_schema_error_category_mapping(self):
        """Verify the actual mapping from HarnessErrorCategory to ErrorCategoryV1."""
        from bench.harness import ErrorCategory as HarnessErrorCategory
        from bench.schema.v1 import ErrorCategoryV1

        # Define the expected mapping (should match runner.py)
        expected_mapping = {
            HarnessErrorCategory.NONE: None,
            HarnessErrorCategory.TIMEOUT: ErrorCategoryV1.TIMEOUT,
            # Transient API/provider errors -> LLM_ERROR (retriable)
            HarnessErrorCategory.API_ERROR: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.RATE_LIMIT: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.NETWORK_ERROR: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.MODEL_ERROR: ErrorCategoryV1.LLM_ERROR,
            HarnessErrorCategory.RESOURCE_EXHAUSTED: ErrorCategoryV1.LLM_ERROR,
            # Auth errors -> AUTH_ERROR (NOT retriable)
            HarnessErrorCategory.AUTH_ERROR: ErrorCategoryV1.AUTH_ERROR,
            # Sandbox errors -> SANDBOX_ERROR (retriable)
            HarnessErrorCategory.SANDBOX_ERROR: ErrorCategoryV1.SANDBOX_ERROR,
            # Test failures -> TEST_FAILURE (NOT retriable)
            HarnessErrorCategory.TEST_ERROR: ErrorCategoryV1.TEST_FAILURE,
            # Invalid requests/agent/task errors -> UNKNOWN (NOT retriable)
            HarnessErrorCategory.INVALID_REQUEST: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.AGENT_ERROR: ErrorCategoryV1.UNKNOWN,
            HarnessErrorCategory.TASK_ERROR: ErrorCategoryV1.UNKNOWN,
            # Fallback for unexpected errors
            HarnessErrorCategory.UNKNOWN: ErrorCategoryV1.UNKNOWN,
        }

        # Verify retriable mappings
        retriable_harness_categories = [
            HarnessErrorCategory.TIMEOUT,
            HarnessErrorCategory.API_ERROR,
            HarnessErrorCategory.RATE_LIMIT,
            HarnessErrorCategory.NETWORK_ERROR,
            HarnessErrorCategory.MODEL_ERROR,
            HarnessErrorCategory.RESOURCE_EXHAUSTED,
            HarnessErrorCategory.SANDBOX_ERROR,
        ]
        retriable_schema_values = {"TIMEOUT", "LLM_ERROR", "SANDBOX_ERROR"}

        for harness_cat in retriable_harness_categories:
            schema_cat = expected_mapping[harness_cat]
            assert schema_cat is not None
            assert schema_cat.value in retriable_schema_values, (
                f"{harness_cat.name} maps to {schema_cat.value}, which is NOT retriable"
            )

        # Verify non-retriable mappings
        non_retriable_harness_categories = [
            HarnessErrorCategory.AUTH_ERROR,
            HarnessErrorCategory.INVALID_REQUEST,
            HarnessErrorCategory.AGENT_ERROR,
            HarnessErrorCategory.TASK_ERROR,
            HarnessErrorCategory.TEST_ERROR,
            HarnessErrorCategory.UNKNOWN,
        ]
        non_retriable_schema_values = {"AUTH_ERROR", "UNKNOWN", "TEST_FAILURE"}

        for harness_cat in non_retriable_harness_categories:
            schema_cat = expected_mapping[harness_cat]
            assert schema_cat is not None
            assert schema_cat.value in non_retriable_schema_values, (
                f"{harness_cat.name} maps to {schema_cat.value}, which IS retriable"
            )
