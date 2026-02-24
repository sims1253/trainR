"""Base class for CLI-based harness adapters (Codex, Claude Code, Gemini CLI)."""

from __future__ import annotations

import subprocess
import time
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any

from ..base import (
    AgentHarness,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    TokenUsage,
    TestResult,
    ErrorCategory,
)


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

        try:
            # Build full command
            cmd = self.cli_command + self.build_cli_args(request)

            # Set up environment
            env = self._build_env(request)

            # Get working directory
            cwd = self.config.working_dir

            # Run CLI
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.get_timeout(request),
                env=env,
                cwd=cwd,
            )

            execution_time = time.time() - start_time

            if result.returncode != 0:
                return HarnessResult(
                    task_id=request.task_id,
                    run_id=run_id,
                    success=False,
                    error_message=result.stderr or f"CLI exited with code {result.returncode}",
                    error_category=ErrorCategory.AGENT_ERROR,
                    execution_time=execution_time,
                    output=result.stdout,
                )

            # Parse output
            harness_result = self.parse_output(result.stdout, request)
            harness_result.execution_time = execution_time
            harness_result.run_id = run_id

            # Preserve raw output in metadata if not already in output
            if not harness_result.output:
                harness_result.output = result.stdout

            return harness_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=False,
                error_message=f"CLI timed out after {self.get_timeout(request)}s",
                error_category=ErrorCategory.TIMEOUT,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=False,
                error_message=str(e),
                error_category=ErrorCategory.from_exception(e),
                execution_time=execution_time,
            )

    def _build_env(self, request: HarnessRequest) -> dict[str, str]:
        """Build environment variables for CLI execution."""
        import os

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
        )
