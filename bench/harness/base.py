"""Base types and protocol for agent harness abstraction.

This module defines the core data structures and abstract protocol that all
harness implementations must follow.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from bench.telemetry import TelemetrySchema


class ErrorCategory(Enum):
    """Classification of execution errors."""

    NONE = "none"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    NETWORK_ERROR = "network_error"
    INVALID_REQUEST = "invalid_request"
    AGENT_ERROR = "agent_error"
    SANDBOX_ERROR = "sandbox_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    MODEL_ERROR = "model_error"
    TASK_ERROR = "task_error"
    TEST_ERROR = "test_error"
    UNKNOWN = "unknown"

    @classmethod
    def from_exception(cls, exc: Exception) -> "ErrorCategory":
        """Classify an exception into an error category.

        Args:
            exc: The exception to classify

        Returns:
            Appropriate ErrorCategory for the exception
        """
        import subprocess

        exc_name = type(exc).__name__.lower()
        exc_msg = str(exc).lower()

        # Timeout errors
        if isinstance(exc, subprocess.TimeoutExpired) or "timeout" in exc_name:
            return cls.TIMEOUT

        # Rate limiting
        if "rate" in exc_msg or "limit" in exc_msg or "429" in exc_msg:
            return cls.RATE_LIMIT

        # Authentication errors
        if "auth" in exc_msg or "unauthorized" in exc_msg or "401" in exc_msg or "403" in exc_msg:
            return cls.AUTH_ERROR

        # API errors
        if "api" in exc_msg or "500" in exc_msg or "502" in exc_msg or "503" in exc_msg:
            return cls.API_ERROR

        # Network errors
        if "network" in exc_msg or "connection" in exc_msg or "dns" in exc_msg:
            return cls.NETWORK_ERROR

        # Sandbox/Docker errors
        if "docker" in exc_msg or "sandbox" in exc_msg or "container" in exc_msg:
            return cls.SANDBOX_ERROR

        # Resource exhaustion
        if "memory" in exc_msg or "oom" in exc_msg or "resource" in exc_msg:
            return cls.RESOURCE_EXHAUSTED

        # Model-specific errors
        if "model" in exc_msg or "llm" in exc_msg:
            return cls.MODEL_ERROR

        # Default to unknown
        return cls.UNKNOWN


class TokenUsage(BaseModel):
    """Token usage metrics for an agent execution."""

    prompt: int = Field(default=0, ge=0, description="Tokens in the prompt")
    completion: int = Field(default=0, ge=0, description="Tokens in the completion")
    total: int = Field(default=0, ge=0, description="Total tokens used")
    cache_read: int = Field(default=0, ge=0, description="Tokens read from cache")
    cache_write: int = Field(default=0, ge=0, description="Tokens written to cache")

    def model_post_init(self, __context: Any) -> None:
        """Calculate total if not explicitly set."""
        if self.total == 0 and (self.prompt > 0 or self.completion > 0):
            self.total = self.prompt + self.completion


class TestResult(BaseModel):
    """Result of a single test case execution."""

    name: str = Field(description="Name of the test case")
    passed: bool = Field(description="Whether the test passed")
    message: str = Field(default="", description="Test output or error message")
    execution_time: float = Field(default=0.0, ge=0, description="Test execution time in seconds")


class HarnessConfig(BaseModel):
    """Configuration for a harness instance.

    This is the common configuration that all harnesses accept. Individual
    harness implementations may define additional configuration options.
    """

    # Execution settings
    timeout: float = Field(default=300.0, gt=0, description="Maximum execution time in seconds")
    max_retries: int = Field(
        default=3, ge=0, description="Maximum retry attempts on transient errors"
    )
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries in seconds")

    # Resource limits
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens for completion")
    max_turns: int | None = Field(default=None, ge=1, description="Maximum conversation turns")

    # Environment settings
    working_dir: str | None = Field(default=None, description="Working directory for execution")
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Additional environment variables"
    )

    # Sandbox settings
    sandbox_enabled: bool = Field(default=True, description="Whether to use sandboxing")
    network_access: bool = Field(default=False, description="Whether to allow network access")

    # Telemetry
    trace_enabled: bool = Field(default=False, description="Whether to enable detailed tracing")
    log_level: str = Field(default="INFO", description="Logging level")

    # Model settings (optional override)
    model: str | None = Field(default=None, description="Model identifier to use")
    api_base: str | None = Field(default=None, description="API base URL override")
    api_key: str | None = Field(default=None, description="API key (use env var preferred)")

    class Config:
        """Pydantic model configuration."""

        extra = "allow"  # Allow additional fields for harness-specific config


class HarnessRequest(BaseModel):
    """Request payload for agent execution.

    Contains all the information needed to execute an agent on a task.
    """

    # Task specification
    task_id: str = Field(description="Unique identifier for the task")
    prompt: str = Field(description="The main prompt/instruction for the agent")
    system_prompt: str | None = Field(default=None, description="System prompt override")

    # Context
    repository: str | None = Field(default=None, description="Repository URL or path")
    base_commit: str | None = Field(default=None, description="Base commit SHA to work from")
    files: dict[str, str] = Field(default_factory=dict, description="File path -> content mapping")

    # Test specification
    test_command: str | None = Field(default=None, description="Command to run tests")
    test_files: list[str] = Field(default_factory=list, description="Paths to test files")
    expected_output: str | None = Field(default=None, description="Expected output pattern")

    # Execution parameters
    timeout: float | None = Field(default=None, description="Override timeout for this request")
    max_tokens: int | None = Field(default=None, description="Override max tokens for this request")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

    class Config:
        """Pydantic model configuration."""

        extra = "allow"


class HarnessResult(BaseModel):
    """Result of an agent execution.

    Contains the output, status, and metrics from running an agent.
    """

    # Identification
    task_id: str = Field(description="Task identifier from the request")
    run_id: str = Field(description="Unique identifier for this execution run")

    # Status
    success: bool = Field(description="Whether execution completed successfully")
    error_category: ErrorCategory = Field(
        default=ErrorCategory.NONE, description="Error classification"
    )
    error_message: str | None = Field(default=None, description="Error details if failed")
    error_traceback: str | None = Field(default=None, description="Full traceback if available")

    # Output
    output: str = Field(default="", description="Agent output/response")
    patch: str | None = Field(default=None, description="Generated diff/patch")
    files_modified: dict[str, str] = Field(default_factory=dict, description="Modified files")

    # Test results
    tests_passed: bool = Field(default=False, description="Whether all tests passed")
    test_results: list[TestResult] = Field(
        default_factory=list, description="Individual test results"
    )
    test_summary: str | None = Field(default=None, description="Test execution summary")

    # Metrics (legacy fields - prefer telemetry for new code)
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Token consumption")
    execution_time: float = Field(default=0.0, ge=0, description="Total execution time in seconds")
    turns: int = Field(default=0, ge=0, description="Number of conversation turns")

    # Unified telemetry (Phase F)
    telemetry: "TelemetrySchema | None" = Field(default=None, description="Unified telemetry data")

    # Metadata
    model: str | None = Field(default=None, description="Model used for execution")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def model_post_init(self, __context: Any) -> None:
        """Sync telemetry with legacy fields if telemetry is set."""
        if self.telemetry is not None:
            # Sync token usage from telemetry
            self.token_usage = TokenUsage(
                prompt=self.telemetry.tokens.prompt,
                completion=self.telemetry.tokens.completion,
                total=self.telemetry.tokens.total,
                cache_read=self.telemetry.tokens.cache_read or 0,
                cache_write=self.telemetry.tokens.cache_write or 0,
            )
            # Sync execution time and turns from telemetry
            self.execution_time = self.telemetry.latency.total_s
            self.turns = self.telemetry.turns
            # Sync model from telemetry
            if self.model is None and self.telemetry.model:
                self.model = self.telemetry.model


class AgentHarness(ABC):
    """Abstract base class for agent execution harnesses.

    A harness provides a unified interface for executing agents across
    different backends (Pi SDK, Pi CLI, Codex, Claude Code, etc.).

    Implementations must provide:
    - execute(): Run an agent on a task
    - validate_environment(): Check that the harness can run
    - setup(): Prepare the harness for execution
    - teardown(): Clean up after execution
    """

    def __init__(self, config: HarnessConfig) -> None:
        """Initialize the harness with configuration.

        Args:
            config: Harness configuration
        """
        self.config = config

    @abstractmethod
    async def execute(self, request: HarnessRequest) -> HarnessResult:
        """Execute an agent on the given request.

        Args:
            request: The execution request

        Returns:
            HarnessResult with execution outcome

        Raises:
            HarnessExecutionError: If execution fails critically
        """
        ...

    @abstractmethod
    def validate_environment(self) -> tuple[bool, list[str]]:
        """Validate that the environment is properly configured.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        ...

    def setup(self) -> None:
        """Set up the harness before execution.

        Override this method to perform any necessary initialization
        (e.g., starting containers, setting up sandboxes).

        This is a no-op by default.
        """
        pass

    def teardown(self) -> None:
        """Tear down the harness after execution.

        Override this method to perform any necessary cleanup
        (e.g., stopping containers, removing temp files).

        This is a no-op by default.
        """
        pass

    def get_environment(self) -> dict[str, str]:
        """Get the environment variables for execution.

        Combines the current environment with harness-specific overrides.

        Returns:
            Environment variable dictionary
        """
        env = dict(os.environ)
        env.update(self.config.env_vars)
        return env

    def get_timeout(self, request: HarnessRequest | None = None) -> float:
        """Get the effective timeout for execution.

        Args:
            request: Optional request with timeout override

        Returns:
            Timeout in seconds
        """
        if request is not None and request.timeout is not None:
            return request.timeout
        return self.config.timeout

    def get_max_tokens(self, request: HarnessRequest | None = None) -> int | None:
        """Get the effective max tokens for execution.

        Args:
            request: Optional request with max_tokens override

        Returns:
            Max tokens or None if not set
        """
        if request is not None and request.max_tokens is not None:
            return request.max_tokens
        return self.config.max_tokens

    def __repr__(self) -> str:
        """Return string representation of the harness."""
        return f"{self.__class__.__name__}(config={self.config!r})"


# Resolve forward references after all classes are defined
# This is required for Pydantic to validate TelemetrySchema in HarnessResult
def _rebuild_models() -> None:
    try:
        from bench.telemetry import TelemetrySchema

        HarnessResult.model_rebuild()
    except ImportError:
        # TelemetrySchema may not be available during initial import
        # The model will be rebuilt when it's first accessed if needed
        pass


_rebuild_models()
