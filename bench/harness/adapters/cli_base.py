"""Base class for CLI-based harness adapters (Codex, Claude Code, Gemini CLI)."""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import time
import uuid
from abc import abstractmethod
from pathlib import Path

from bench.telemetry import (
    LatencyBreakdown,
    TelemetrySchema,
    ToolCallMetrics,
)
from bench.telemetry import (
    TokenUsage as TelemetryTokenUsage,
)

from ..base import (
    AgentHarness,
    ErrorCategory,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
)


def _attach_failure_telemetry(
    result: HarnessResult,
    request: HarnessRequest,
    execution_time: float,
    provider: str | None = None,
    model: str | None = None,
    harness: str = "unknown",
) -> HarnessResult:
    """Ensure telemetry is populated for failure paths.

    For failures where token/tool stats are unknown, emit zero/default
    telemetry with latency and harness/model/provider set when known.
    """
    if result.telemetry is not None:
        return result

    result.telemetry = TelemetrySchema(
        tokens=TelemetryTokenUsage(
            prompt=0,
            completion=0,
            total=0,
        ),
        turns=0,
        tools=ToolCallMetrics(),
        latency=LatencyBreakdown(total_s=execution_time, execution_s=execution_time),
        provider=provider,
        model=model,
        harness=harness,
    )
    return result


class CliHarnessBase(AgentHarness):
    """Base class for CLI-based harness adapters.

    Subclasses must implement:
    - cli_name: Name of the CLI tool
    - build_cli_args(): Build CLI arguments from request
    - parse_output(): Parse CLI output into HarnessResult
    """

    def __init__(self, config: HarnessConfig):
        super().__init__(config)
        self._cli_path: Path | None = None

    @property
    @abstractmethod
    def cli_name(self) -> str:
        """Name of the CLI tool (e.g., 'codex', 'claude', 'gemini')."""
        ...

    @property
    def cli_command(self) -> list[str]:
        """Full CLI command to invoke."""
        return [self.cli_name]

    @abstractmethod
    def build_cli_args(self, request: HarnessRequest) -> list[str]:
        """Build CLI arguments from request.

        Args:
            request: The harness request

        Returns:
            List of CLI arguments (not including the command itself)
        """
        ...

    @abstractmethod
    def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
        """Parse CLI output into HarnessResult.

        Args:
            output: Raw CLI stdout
            request: Original request for context

        Returns:
            Parsed HarnessResult
        """
        ...

    def validate_environment(self) -> tuple[bool, list[str]]:
        """Check if CLI is available."""
        errors = []
        try:
            result = subprocess.run(
                ["which", self.cli_name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"{self.cli_name} CLI not found in PATH")
                return False, errors

            self._cli_path = Path(result.stdout.strip())
            return True, []

        except Exception as e:
            errors.append(f"Error checking {self.cli_name}: {e}")
            return False, errors

    async def execute(self, request: HarnessRequest) -> HarnessResult:
        """Execute task via CLI."""
        start_time = time.time()
        run_id = str(uuid.uuid4())

        # Resolve model/provider for telemetry
        model_name = request.metadata.get("model") or self.config.model
        provider_name = model_name.split("/", 1)[0] if model_name and "/" in model_name else None

        try:
            # Build full command
            cmd = self.cli_command + self.build_cli_args(request)

            # Set up environment
            env = self._build_env(request)

            # Get working directory
            cwd = self.config.working_dir

            # Check if we should preserve container logs
            save_container_logs = request.metadata.get("save_container_logs", False)

            # Create temp file for streaming stdout (avoids memory bloat with large outputs)
            stdout_fd, stdout_path = tempfile.mkstemp(suffix=".log", prefix=f"cli_{self.cli_name}_")
            os.close(stdout_fd)  # Close the fd, we'll reopen for subprocess

            stdout_content = None
            log_path_to_preserve = None

            try:
                with open(stdout_path, "w") as stdout_file:
                    result = subprocess.run(
                        cmd,
                        stdout=stdout_file,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.get_timeout(request),
                        env=env,
                        cwd=cwd,
                    )

                execution_time = time.time() - start_time

                # Read stdout back from file
                with open(stdout_path) as f:
                    stdout_content = f.read()

                # If save_container_logs is True, preserve the log file path
                if save_container_logs:
                    log_path_to_preserve = stdout_path
                else:
                    # Clean up temp file
                    with contextlib.suppress(OSError):
                        os.unlink(stdout_path)

            except subprocess.TimeoutExpired:
                # Clean up temp file on timeout
                with contextlib.suppress(OSError):
                    os.unlink(stdout_path)
                raise
            except Exception:
                # Clean up temp file on any other exception
                with contextlib.suppress(OSError):
                    os.unlink(stdout_path)
                raise

            if result.returncode != 0:
                failure_result = HarnessResult(
                    task_id=request.task_id,
                    run_id=run_id,
                    success=False,
                    error_message=result.stderr or f"CLI exited with code {result.returncode}",
                    error_category=ErrorCategory.AGENT_ERROR,
                    execution_time=execution_time,
                    output=stdout_content,
                )
                # Preserve log path in metadata if requested
                if log_path_to_preserve:
                    failure_result.metadata = failure_result.metadata or {}
                    failure_result.metadata["container_log_path"] = log_path_to_preserve
                return _attach_failure_telemetry(
                    failure_result,
                    request,
                    execution_time,
                    provider=provider_name,
                    model=model_name,
                    harness=self.cli_name,
                )

            # Parse output
            harness_result = self.parse_output(stdout_content, request)
            harness_result.execution_time = execution_time
            harness_result.run_id = run_id
            self._ensure_telemetry(harness_result, request, execution_time)

            # Preserve raw output in metadata if not already in output
            if not harness_result.output:
                harness_result.output = stdout_content

            # Preserve log path in metadata if requested
            if log_path_to_preserve:
                harness_result.metadata = harness_result.metadata or {}
                harness_result.metadata["container_log_path"] = log_path_to_preserve

            return harness_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            timeout_result = HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=False,
                error_message=f"CLI timed out after {self.get_timeout(request)}s",
                error_category=ErrorCategory.TIMEOUT,
                execution_time=execution_time,
            )
            return _attach_failure_telemetry(
                timeout_result,
                request,
                execution_time,
                provider=provider_name,
                model=model_name,
                harness=self.cli_name,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            exception_result = HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=False,
                error_message=str(e),
                error_category=ErrorCategory.from_exception(e),
                execution_time=execution_time,
            )
            return _attach_failure_telemetry(
                exception_result,
                request,
                execution_time,
                provider=provider_name,
                model=model_name,
                harness=self.cli_name,
            )

    def _ensure_telemetry(
        self, result: HarnessResult, request: HarnessRequest, execution_time: float
    ) -> None:
        """Ensure every CLI execution emits canonical telemetry."""
        model_name = result.model or request.metadata.get("model") or self.config.model
        provider_name = model_name.split("/", 1)[0] if model_name and "/" in model_name else None

        if result.telemetry is None:
            result.telemetry = TelemetrySchema(
                tokens=TelemetryTokenUsage(
                    prompt=result.token_usage.prompt,
                    completion=result.token_usage.completion,
                    total=result.token_usage.total,
                    cache_read=result.token_usage.cache_read,
                    cache_write=result.token_usage.cache_write,
                ),
                turns=result.turns,
                tools=ToolCallMetrics(),
                latency=LatencyBreakdown(total_s=execution_time, execution_s=execution_time),
                provider=provider_name,
                model=model_name,
                harness=self.cli_name,
            )
            return

        # Keep telemetry latency in sync with measured wall-clock execution.
        result.telemetry.latency.total_s = execution_time
        if result.telemetry.latency.execution_s is None:
            result.telemetry.latency.execution_s = execution_time
        if result.telemetry.model is None:
            result.telemetry.model = model_name
        if result.telemetry.provider is None:
            result.telemetry.provider = provider_name
        if result.telemetry.harness is None:
            result.telemetry.harness = self.cli_name

    def _build_env(self, request: HarnessRequest) -> dict[str, str]:
        """Build environment variables for CLI execution."""
        env = os.environ.copy()

        # Add environment variables from config
        env.update(self.config.env_vars)

        # Add API key if configured
        if self.config.api_key:
            # Subclasses can override to set the correct env var name
            env["API_KEY"] = self.config.api_key

        # Add model from request or config
        model = request.metadata.get("model") or self.config.model
        if model:
            env["MODEL"] = model

        return env

    def _create_empty_result(
        self,
        request: HarnessRequest,
        run_id: str | None = None,
    ) -> HarnessResult:
        """Create an empty HarnessResult with required fields.

        Args:
            request: The harness request
            run_id: Optional run ID (generated if not provided)

        Returns:
            Empty HarnessResult with task_id and run_id set
        """
        return HarnessResult(
            task_id=request.task_id,
            run_id=run_id or str(uuid.uuid4()),
            success=False,
        )
