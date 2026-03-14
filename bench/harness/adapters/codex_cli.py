"""Codex CLI harness adapter."""

from __future__ import annotations

import json
import uuid

from ..base import (
    ErrorCategory,
    HarnessRequest,
    HarnessResult,
    TestResult,
)
from ..registry import register_harness
from .cli_base import CliHarnessBase


@register_harness("codex_cli")
class CodexCliHarness(CliHarnessBase):
    """Harness for OpenAI Codex CLI.

    Executes tasks using the Codex CLI tool.
    """

    @property
    def cli_name(self) -> str:
        return "codex"

    def build_cli_args(self, request: HarnessRequest) -> list[str]:
        """Build Codex CLI arguments."""
        model = request.metadata.get("model") or self.config.model or "gpt-4"

        args = [
            "--model",
            model,
            "--task",
            request.prompt,
        ]

        context = request.metadata.get("context")
        if context:
            args.extend(["--context", context])

        if request.timeout:
            args.extend(["--timeout", str(request.timeout)])

        return args

    def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
        """Parse Codex CLI output."""
        run_id = str(uuid.uuid4())
        model_name = request.metadata.get("model") or self.config.model

        # Parse JSON output if available
        try:
            data = json.loads(output)
            model_name = data.get("model") or model_name

            # Extract test results
            test_results = []
            for t in data.get("tests", []):
                test_results.append(
                    TestResult(
                        name=t.get("name", "unknown"),
                        passed=t.get("passed", False),
                        message=t.get("message", ""),
                    )
                )

            # Determine success
            success = data.get("success", False)

            # Calculate score (stored in metadata)
            score = data.get("score", 1.0 if success else 0.0)

            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=success,
                output=data.get("output", ""),
                patch=data.get("code"),
                tests_passed=success and all(tr.passed for tr in test_results)
                if test_results
                else success,
                test_results=test_results,
                error_message=data.get("error"),
                error_category=ErrorCategory.NONE if success else ErrorCategory.TASK_ERROR,
                model=model_name,
                metadata={"score": score, "raw_response": data},
            )
        except json.JSONDecodeError:
            # Fallback: parse text output
            passed = "SUCCESS" in output or "PASSED" in output
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=passed,
                output=output,
                error_category=ErrorCategory.NONE if passed else ErrorCategory.TASK_ERROR,
                model=model_name,
            )
