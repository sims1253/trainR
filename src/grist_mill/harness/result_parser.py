"""Result parser for converting raw test runner output into structured TaskResult.

Accepts ExecutionOutput (stdout, stderr, exit_code, timed_out) from test
runners (pytest, testthat, generic shell commands) and produces a
TaskResult with correct status, score, and error_category from the
error taxonomy.

Pattern detection priority:
1. Timeout (timed_out flag) → TIMEOUT
2. Exit code analysis (0 = success, non-zero = failure/error)
3. Content-based error pattern matching for fine-grained categorization

Design decisions:
- Syntax errors are detected by pattern matching on stdout/stderr, using
  language-specific heuristics when available.
- Environment errors (command not found, permission denied, segfault) are
  detected by specific patterns and exit codes (126, 127, 139, 137).
- Network/API/rate-limit errors are detected by pattern matching.
- Unrecognized non-zero exits default to FAILURE/TEST_FAILURE.
- Timeout always captures partial output in the transcript field.

Validates VAL-HARNESS-01.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from grist_mill.schemas import (
    ErrorCategory,
    ExecutionOutput,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error patterns: (compiled regex, error_category, status, description)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ErrorPattern:
    """A single error-detection rule."""

    regex: re.Pattern[str]
    category: ErrorCategory
    status: TaskStatus
    description: str


# Syntax error patterns (applied first, highest priority after timeout)
_SYNTAX_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r"SyntaxError", re.IGNORECASE),
        category=ErrorCategory.SYNTAX_ERROR,
        status=TaskStatus.ERROR,
        description="Syntax error detected in output",
    ),
    _ErrorPattern(
        regex=re.compile(r"IndentationError", re.IGNORECASE),
        category=ErrorCategory.SYNTAX_ERROR,
        status=TaskStatus.ERROR,
        description="Indentation error detected in output",
    ),
    _ErrorPattern(
        regex=re.compile(r"TabError", re.IGNORECASE),
        category=ErrorCategory.SYNTAX_ERROR,
        status=TaskStatus.ERROR,
        description="Tab/indent error detected in output",
    ),
    _ErrorPattern(
        # R: "unexpected symbol", "unexpected input", "parse error"
        regex=re.compile(
            r"(?:unexpected\s+(?:symbol|input|token)|parse\s+error)",
            re.IGNORECASE,
        ),
        category=ErrorCategory.SYNTAX_ERROR,
        status=TaskStatus.ERROR,
        description="R/Shell syntax or parse error detected",
    ),
    _ErrorPattern(
        # TypeScript: "error TSxxxx"
        regex=re.compile(r"error\s+TS\d{4}:", re.IGNORECASE),
        category=ErrorCategory.SYNTAX_ERROR,
        status=TaskStatus.ERROR,
        description="TypeScript compilation error detected",
    ),
    _ErrorPattern(
        # JavaScript: "Unexpected token" syntax errors
        regex=re.compile(r"Unexpected\s+token", re.IGNORECASE),
        category=ErrorCategory.SYNTAX_ERROR,
        status=TaskStatus.ERROR,
        description="JavaScript syntax error detected",
    ),
]

# Environment error patterns
_ENVIRONMENT_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r"command not found", re.IGNORECASE),
        category=ErrorCategory.ENVIRONMENT_ERROR,
        status=TaskStatus.ERROR,
        description="Command not found in PATH",
    ),
    _ErrorPattern(
        regex=re.compile(r"Permission denied", re.IGNORECASE),
        category=ErrorCategory.ENVIRONMENT_ERROR,
        status=TaskStatus.ERROR,
        description="Permission denied",
    ),
    _ErrorPattern(
        regex=re.compile(r"MemoryError", re.IGNORECASE),
        category=ErrorCategory.ENVIRONMENT_ERROR,
        status=TaskStatus.ERROR,
        description="Out of memory",
    ),
    _ErrorPattern(
        regex=re.compile(r"Segmentation fault", re.IGNORECASE),
        category=ErrorCategory.ENVIRONMENT_ERROR,
        status=TaskStatus.ERROR,
        description="Segmentation fault",
    ),
    _ErrorPattern(
        # R: "there is no package called"
        regex=re.compile(r"there is no package called", re.IGNORECASE),
        category=ErrorCategory.ENVIRONMENT_ERROR,
        status=TaskStatus.ERROR,
        description="Missing R package",
    ),
    _ErrorPattern(
        # Python: "ModuleNotFoundError"
        regex=re.compile(r"ModuleNotFoundError", re.IGNORECASE),
        category=ErrorCategory.ENVIRONMENT_ERROR,
        status=TaskStatus.ERROR,
        description="Missing Python module",
    ),
]

# Test failure patterns (lower priority than syntax/env, checked after them)
_TEST_FAILURE_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        # pytest: "N failed", "FAILED", "AssertionError"
        regex=re.compile(r"\d+\s+failed|FAILED\s+\S+", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="Test failure detected in output",
    ),
    _ErrorPattern(
        # Generic assertion errors
        regex=re.compile(r"AssertionError|AssertionError", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="Assertion error in test output",
    ),
    _ErrorPattern(
        # Python TypeError (runtime error in test context)
        regex=re.compile(r"TypeError", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="Type error detected in output",
    ),
    _ErrorPattern(
        # Python tracebacks in test context
        regex=re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="Python traceback in output",
    ),
    _ErrorPattern(
        # R testthat: "FAIL N"
        regex=re.compile(r"\[\s*FAIL\s+\d+", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="R testthat failures detected",
    ),
    _ErrorPattern(
        # R test failure messages
        regex=re.compile(r"Failure\s*\(", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="R test failure detected",
    ),
    _ErrorPattern(
        # Generic "test.*failed" patterns
        regex=re.compile(r"\btest\b.*\bfailed\b|\bfailed\b.*\btest\b", re.IGNORECASE),
        category=ErrorCategory.TEST_FAILURE,
        status=TaskStatus.FAILURE,
        description="Test failure detected",
    ),
]

# Network / API / rate-limit patterns
_NETWORK_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(
            r"ConnectionError|Connection refused|NetworkError|URLError", re.IGNORECASE
        ),
        category=ErrorCategory.NETWORK_ERROR,
        status=TaskStatus.FAILURE,
        description="Network connection error",
    ),
    _ErrorPattern(
        regex=re.compile(r"\b429\b.*Too Many Requests|rate.?limit", re.IGNORECASE),
        category=ErrorCategory.RATE_LIMIT,
        status=TaskStatus.ERROR,
        description="Rate limit exceeded",
    ),
    _ErrorPattern(
        regex=re.compile(
            r"\bAPIError\b|\bapi_error\b|openai\.error|anthropic\.error", re.IGNORECASE
        ),
        category=ErrorCategory.API_ERROR,
        status=TaskStatus.ERROR,
        description="API error from LLM provider",
    ),
]

# Exit-code-based environment errors (before content matching)
_EXIT_CODE_PATTERNS: dict[int, tuple[ErrorCategory, str]] = {
    126: (ErrorCategory.ENVIRONMENT_ERROR, "Command cannot execute (permission denied)"),
    127: (ErrorCategory.ENVIRONMENT_ERROR, "Command not found"),
    137: (ErrorCategory.ENVIRONMENT_ERROR, "Process killed (OOM or SIGKILL)"),
    139: (ErrorCategory.ENVIRONMENT_ERROR, "Segmentation fault"),
}


class ResultParser:
    """Converts raw test runner output into structured TaskResult.

    Usage::

        parser = ResultParser()
        result = parser.parse(
            execution_output,
            task_id="task-123",
            language="python",
        )

    The parser applies a cascade of detection strategies:

    1. **Timeout** — if ``execution_output.timed_out`` is True, the result
       is always ``TIMEOUT`` with partial output captured in the transcript.
    2. **Exit-code heuristics** — known exit codes (126, 127, 137, 139) map
       directly to ``ENVIRONMENT_ERROR``.
    3. **Content-based pattern matching** — stdout and stderr are scanned
       for known error patterns (syntax, environment, network, API).
       Syntax patterns take priority over other categories.
    4. **Default** — exit code 0 → SUCCESS; non-zero → FAILURE/UNKNOWN.
    """

    def __init__(self) -> None:
        # Order matters: syntax > environment > test_failure > network > default
        self._pattern_groups: list[tuple[str, list[_ErrorPattern]]] = [
            ("syntax", _SYNTAX_PATTERNS),
            ("environment", _ENVIRONMENT_PATTERNS),
            ("test_failure", _TEST_FAILURE_PATTERNS),
            ("network", _NETWORK_PATTERNS),
        ]

    def parse(
        self,
        output: ExecutionOutput,
        *,
        task_id: str,
        language: str = "python",
    ) -> TaskResult:
        """Parse execution output into a TaskResult.

        Args:
            output: The raw execution output from the test runner.
            task_id: The ID of the task that was executed.
            language: Hint about the programming language (helps with
                language-specific error detection).

        Returns:
            A TaskResult with status, score, and error_category populated.
        """
        # --- 1. Timeout check (highest priority) ---
        if output.timed_out:
            return self._build_timeout_result(output, task_id)

        # --- 2. Exit-code based heuristics ---
        if output.exit_code in _EXIT_CODE_PATTERNS:
            category, description = _EXIT_CODE_PATTERNS[output.exit_code]
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=category,
                transcript=[self._build_transcript_entry(output, description)],
            )

        # Negative exit codes (signal-based) are environment errors
        if output.exit_code < 0:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.ERROR,
                score=0.0,
                error_category=ErrorCategory.ENVIRONMENT_ERROR,
                transcript=[
                    self._build_transcript_entry(
                        output, f"Process killed by signal {-output.exit_code}"
                    )
                ],
            )

        # --- 3. Special exit codes ---
        # pytest exit 5: no tests collected
        if output.exit_code == 5:
            combined = f"{output.stdout}\n{output.stderr}"
            if "no tests" in combined.lower() or "test" in combined.lower():
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.ERROR,
                    score=0.0,
                    error_category=ErrorCategory.TEST_FAILURE,
                    transcript=[self._build_transcript_entry(output, "No tests collected")],
                )

        # pytest exit 2: test collection errors or interrupted (only if output looks pytest-like)
        if output.exit_code == 2:
            combined = f"{output.stdout}\n{output.stderr}"
            lower = combined.lower()
            # Only match pytest-specific patterns, not generic "error" text
            if any(
                marker in lower
                for marker in ("pytest", "error collecting", "no tests ran", "interrupted")
            ):
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.ERROR,
                    score=0.0,
                    error_category=ErrorCategory.TEST_FAILURE,
                    transcript=[
                        self._build_transcript_entry(output, "Test collection or setup error")
                    ],
                )

        # --- 4. Success path ---
        if output.exit_code == 0:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                score=1.0,
                error_category=None,
            )

        # --- 5. Content-based pattern matching ---
        combined_output = f"{output.stdout}\n{output.stderr}"
        matched = self._match_patterns(combined_output, language)

        if matched is not None:
            pattern = matched
            return TaskResult(
                task_id=task_id,
                status=pattern.status,
                score=0.0,
                error_category=pattern.category,
                transcript=[self._build_transcript_entry(output, pattern.description)],
            )

        # --- 6. Default: unrecognised non-zero exit ---
        description = self._default_description(output)
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILURE,
            score=0.0,
            error_category=ErrorCategory.UNKNOWN,
            transcript=[self._build_transcript_entry(output, description)],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_timeout_result(
        self,
        output: ExecutionOutput,
        task_id: str,
    ) -> TaskResult:
        """Build a TIMEOUT TaskResult with partial output in transcript."""
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.TIMEOUT,
            score=0.0,
            error_category=ErrorCategory.UNKNOWN,
            transcript=[
                {
                    "phase": "execution",
                    "output": output.stdout,
                    "stderr": output.stderr,
                    "message": "Execution timed out",
                },
            ],
        )

    def _match_patterns(
        self,
        combined_output: str,
        language: str,
    ) -> _ErrorPattern | None:
        """Match output against all pattern groups in priority order.

        Returns the first matching pattern, or None if no pattern matches.
        """
        for _group_name, patterns in self._pattern_groups:
            for pattern in patterns:
                if pattern.regex.search(combined_output):
                    return pattern
        return None

    @staticmethod
    def _build_transcript_entry(
        output: ExecutionOutput,
        description: str,
    ) -> dict[str, Any]:
        """Build a transcript entry from execution output."""
        entry: dict[str, Any] = {
            "stdout": output.stdout,
            "stderr": output.stderr,
            "exit_code": output.exit_code,
            "message": description,
        }
        return entry

    @staticmethod
    def _default_description(output: ExecutionOutput) -> str:
        """Generate a descriptive message for unrecognized failures."""
        combined = output.stdout.strip() + output.stderr.strip()
        if not combined:
            return "Command exited with non-zero code and produced no output"
        # Truncate for readability
        preview = combined[:200]
        return f"Command exited with code {output.exit_code}: {preview}"
