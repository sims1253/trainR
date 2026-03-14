"""Claude Code CLI harness adapter."""

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


@register_harness("claude_cli")
class ClaudeCliHarness(CliHarnessBase):
    """Harness for Anthropic Claude Code CLI.

    Executes tasks using the Claude Code CLI tool.
    """

    @property
    def cli_name(self) -> str:
        return "claude"

    def build_cli_args(self, request: HarnessRequest) -> list[str]:
        """Build Claude CLI arguments."""
        model = request.metadata.get("model") or self.config.model or "claude-3-5-sonnet-20241022"

        args = [
            "--model",
            model,
        ]

        if request.system_prompt:
            args.extend(["--system", request.system_prompt])

        # Add the prompt
        args.append(request.prompt)

        if request.timeout:
            args.extend(["--timeout", str(request.timeout)])

        # JSON output format
        args.append("--json")

        return args

    def parse_output(self, output: str, request: HarnessRequest) -> HarnessResult:
        """Parse Claude CLI output."""
        run_id = str(uuid.uuid4())
        model_name = request.metadata.get("model") or self.config.model

        try:
            data = json.loads(output)
            model_name = data.get("model") or model_name

            # Extract test results from Claude's output
            test_results = []
            if "tests" in data:
                for t in data["tests"]:
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

            # Get generated code - could be in 'code' or 'content' field
            generated_code = data.get("code") or data.get("content")

            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=success,
                output=data.get("output", "") or data.get("content", ""),
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
            # Fallback: look for success indicators
            output_lower = output.lower()
            passed = any(
                indicator in output_lower
                for indicator in ["success", "passed", "completed", "done"]
            )
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=passed,
                output=output,
                error_category=ErrorCategory.NONE if passed else ErrorCategory.TASK_ERROR,
                model=model_name,
            )
