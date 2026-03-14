"""Gemini CLI harness adapter."""

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


@register_harness("gemini_cli")
class GeminiCliHarness(CliHarnessBase):
    """Harness for Google Gemini CLI.

    Executes tasks using the Gemini CLI tool.
    """

    @property
    def cli_name(self) -> str:
        return "gemini"

    def build_cli_args(self, request: HarnessRequest) -> list[str]:
        """Build Gemini CLI arguments."""
        model = request.metadata.get("model") or self.config.model or "gemini-1.5-pro"

        args = [
            "--model",
            model,
        ]

        if request.system_prompt:
            args.extend(["--system-instruction", request.system_prompt])

        # Add the prompt
        args.append(request.prompt)

        if request.timeout:
            args.extend(["--timeout", str(request.timeout)])

        # Output format
        args.extend(["--output", "json"])

        return args

    def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
        """Parse Gemini CLI output."""
        run_id = str(uuid.uuid4())
        model_name = request.metadata.get("model") or self.config.model

        try:
            data = json.loads(output)
            model_name = data.get("model") or model_name

            # Extract test results
            test_results = []
            if "test_results" in data:
                for t in data["test_results"]:
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

            # Get generated code - could be in 'code' or 'text' field
            generated_code = data.get("code") or data.get("text")

            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=success,
                output=data.get("output", "") or data.get("text", ""),
                patch=generated_code,
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
            # Fallback parsing
            output_lower = output.lower()
            passed = "success" in output_lower or "passed" in output_lower
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=passed,
                output=output,
                error_category=ErrorCategory.NONE if passed else ErrorCategory.TASK_ERROR,
                model=model_name,
            )
