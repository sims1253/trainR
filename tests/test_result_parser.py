"""Tests for the result parser module.

Validates VAL-HARNESS-01: Result parser converts test runner output
(pytest exit codes, testthat output, generic shell exit codes) into
structured TaskResult with correct status, score, and error_category.
"""

from __future__ import annotations

from grist_mill.harness.result_parser import ResultParser
from grist_mill.schemas import ErrorCategory, ExecutionOutput, TaskResult, TaskStatus

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_PARSER = ResultParser()


def _parse(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
    task_id: str = "test-task-1",
    language: str = "python",
) -> TaskResult:
    """Shortcut to parse an ExecutionOutput and return a TaskResult."""
    output = ExecutionOutput(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
    )
    return _PARSER.parse(output, task_id=task_id, language=language)


# ===========================================================================
# 1. Passing tests
# ===========================================================================


class TestPassingOutput:
    """Passing test output should produce status=SUCCESS, score=1.0."""

    def test_pytest_all_passed(self) -> None:
        result = _parse(stdout="===== 5 passed in 2.3s =====", exit_code=0)
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.error_category is None

    def test_pytest_no_tests_collected(self) -> None:
        """pytest exit 5 (no tests) is treated as an error."""
        result = _parse(
            stdout="===== no tests ran (skipped 1) in 0.01s =====",
            exit_code=5,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_generic_exit_zero(self) -> None:
        result = _parse(stdout="Done.", exit_code=0)
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.error_category is None

    def test_testthat_all_passed(self) -> None:
        """testthat with OK result."""
        result = _parse(
            stdout="[ FAIL 0 | WARN 0 | SKIP 0 | PASS 5 ]",
            stderr="OK",
            exit_code=0,
            language="r",
        )
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0
        assert result.error_category is None

    def test_shell_echo_success(self) -> None:
        result = _parse(stdout="hello world", exit_code=0, language="shell")
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0


# ===========================================================================
# 2. Failing tests
# ===========================================================================


class TestFailingOutput:
    """Failing test output should produce status=FAILURE with error_category."""

    def test_pytest_failures(self) -> None:
        result = _parse(
            stdout="===== 3 passed, 2 failed in 1.5s =====",
            exit_code=1,
        )
        assert result.status == TaskStatus.FAILURE
        assert result.score == 0.0
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_pytest_single_failure(self) -> None:
        result = _parse(
            stdout="===== FAILED test_example.py::test_something =====",
            exit_code=1,
        )
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_testthat_failures(self) -> None:
        result = _parse(
            stdout="test_a ..\ntest_b F\n[ FAIL 1 | WARN 0 | SKIP 0 | PASS 1 ]",
            stderr="Failure (test_b): ...",
            exit_code=1,
            language="r",
        )
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_generic_exit_nonzero(self) -> None:
        result = _parse(stdout="Error: something went wrong", exit_code=1)
        assert result.status == TaskStatus.FAILURE
        assert result.error_category is not None

    def test_generic_exit_two(self) -> None:
        """Exit code 2 is common in shell for usage errors."""
        result = _parse(stderr="Usage: cmd [options]", exit_code=2)
        assert result.status == TaskStatus.FAILURE
        assert result.error_category is not None


# ===========================================================================
# 3. Syntax errors
# ===========================================================================


class TestSyntaxErrorOutput:
    """Syntax error output should produce status=ERROR with SYNTAX_ERROR."""

    def test_python_syntax_error(self) -> None:
        result = _parse(
            stderr="  File 'solution.py', line 3\n    def foo(\n             ^\nSyntaxError: invalid syntax",
            exit_code=1,
            language="python",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR

    def test_python_indentation_error(self) -> None:
        result = _parse(
            stderr="IndentationError: unexpected indent",
            exit_code=1,
            language="python",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR

    def test_r_syntax_error(self) -> None:
        result = _parse(
            stderr='Error: unexpected symbol in "function"',
            exit_code=1,
            language="r",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR

    def test_javascript_syntax_error(self) -> None:
        result = _parse(
            stderr="SyntaxError: Unexpected token '{'",
            exit_code=1,
            language="typescript",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR

    def test_python_syntax_error_in_stdout(self) -> None:
        """Sometimes syntax errors appear in stdout instead of stderr."""
        result = _parse(
            stdout="Traceback (most recent call last):\n  File 'app.py', line 5\nSyntaxError: invalid syntax",
            exit_code=1,
            language="python",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR

    def test_r_parsing_error(self) -> None:
        """R uses 'parse error' for syntax issues."""
        result = _parse(
            stderr="Error in parse(file) : unexpected input",
            exit_code=1,
            language="r",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR


# ===========================================================================
# 4. Timeout
# ===========================================================================


class TestTimeoutOutput:
    """Timeout should produce status=TIMEOUT with partial output captured."""

    def test_timeout_flag(self) -> None:
        output = ExecutionOutput(
            stdout="partial output...",
            stderr="",
            exit_code=-1,
            timed_out=True,
        )
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.status == TaskStatus.TIMEOUT
        assert result.score == 0.0
        assert result.error_category is not None
        # Verify partial output is preserved in the result
        assert "partial" in result.transcript[-1]["output"] if result.transcript else True

    def test_timeout_with_stderr(self) -> None:
        output = ExecutionOutput(
            stdout="",
            stderr="timeout: killed after 30s",
            exit_code=137,
            timed_out=True,
        )
        result = _PARSER.parse(output, task_id="t2", language="python")
        assert result.status == TaskStatus.TIMEOUT

    def test_timeout_empty_output(self) -> None:
        output = ExecutionOutput(
            stdout="",
            stderr="",
            exit_code=-1,
            timed_out=True,
        )
        result = _PARSER.parse(output, task_id="t3", language="python")
        assert result.status == TaskStatus.TIMEOUT
        assert result.score == 0.0


# ===========================================================================
# 5. Empty output
# ===========================================================================


class TestEmptyOutput:
    """Empty output should produce a descriptive error."""

    def test_empty_stdout_stderr_zero_exit(self) -> None:
        """Exit 0 with empty output is still SUCCESS but score is 1.0."""
        result = _parse(stdout="", stderr="", exit_code=0)
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0

    def test_empty_stdout_stderr_nonzero_exit(self) -> None:
        """Non-zero exit with empty output is an error with descriptive message."""
        result = _parse(stdout="", stderr="", exit_code=1)
        assert result.status == TaskStatus.FAILURE
        assert result.score == 0.0
        assert result.error_category == ErrorCategory.UNKNOWN

    def test_whitespace_only_output(self) -> None:
        result = _parse(stdout="   \n  \t  ", stderr="  ", exit_code=1)
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.UNKNOWN


# ===========================================================================
# 6. Error pattern detection
# ===========================================================================


class TestErrorPatternDetection:
    """Various error patterns should map to correct categories."""

    def test_import_error(self) -> None:
        """Import errors indicate missing dependencies → ENVIRONMENT_ERROR."""
        result = _parse(
            stderr="ModuleNotFoundError: No module named 'pytest'",
            exit_code=1,
            language="python",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR

    def test_environment_error_pattern(self) -> None:
        """Environment-specific errors (Docker, permissions) map to ENVIRONMENT_ERROR."""
        result = _parse(
            stderr="Permission denied: /usr/bin/python",
            exit_code=126,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR

    def test_command_not_found(self) -> None:
        result = _parse(
            stderr="/bin/sh: pytest: command not found",
            exit_code=127,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR

    def test_network_error_pattern(self) -> None:
        result = _parse(
            stderr="requests.exceptions.ConnectionError: ...",
            exit_code=1,
        )
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.NETWORK_ERROR

    def test_rate_limit_pattern(self) -> None:
        result = _parse(
            stderr="429 Too Many Requests",
            exit_code=1,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.RATE_LIMIT

    def test_api_error_pattern(self) -> None:
        result = _parse(
            stderr="openai.error.APIError: ...",
            exit_code=1,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.API_ERROR

    def test_generic_unknown_error(self) -> None:
        """Unrecognizable non-zero exit should be FAILURE with UNKNOWN category."""
        result = _parse(
            stderr="Something unexpected happened",
            exit_code=1,
        )
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.UNKNOWN

    def test_memory_error(self) -> None:
        result = _parse(
            stderr="MemoryError: Unable to allocate array",
            exit_code=137,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR

    def test_segfault(self) -> None:
        """Segfault (exit code 139) maps to environment error."""
        result = _parse(
            stderr="Segmentation fault (core dumped)",
            exit_code=139,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR

    def test_type_error_python(self) -> None:
        """Python TypeError is a runtime error, not syntax."""
        result = _parse(
            stderr="TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            exit_code=1,
            language="python",
        )
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.TEST_FAILURE


# ===========================================================================
# 7. ResultParser.parse returns correct TaskResult
# ===========================================================================


class TestResultParserFull:
    """Verify the full TaskResult object is correctly populated."""

    def test_result_has_task_id(self) -> None:
        output = ExecutionOutput(stdout="pass", exit_code=0)
        result = _PARSER.parse(output, task_id="my-task", language="python")
        assert result.task_id == "my-task"

    def test_success_result_no_telemetry(self) -> None:
        """ResultParser doesn't set telemetry (that's the harness's job)."""
        output = ExecutionOutput(stdout="pass", exit_code=0)
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.telemetry is None

    def test_timeout_includes_partial_output(self) -> None:
        output = ExecutionOutput(
            stdout="running test 1...\nrunning test 2...",
            stderr="",
            exit_code=-1,
            timed_out=True,
        )
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.transcript is not None
        assert len(result.transcript) == 1
        assert result.transcript[0]["output"] == output.stdout

    def test_error_result_has_message(self) -> None:
        """Error results should include a descriptive transcript entry."""
        output = ExecutionOutput(
            stdout="",
            stderr="SyntaxError: invalid syntax",
            exit_code=1,
        )
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.transcript is not None
        assert len(result.transcript) == 1
        assert "SyntaxError" in result.transcript[0]["stderr"]


# ===========================================================================
# 8. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases for result parser."""

    def test_large_output_truncated_in_transcript(self) -> None:
        """Large output should still produce a valid result."""
        big_output = "x" * 100000
        output = ExecutionOutput(stdout=big_output, exit_code=0)
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0

    def test_unicode_output(self) -> None:
        output = ExecutionOutput(stdout="✓ 测试通过", exit_code=0)
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.status == TaskStatus.SUCCESS

    def test_exit_code_negative_non_timeout(self) -> None:
        """Negative exit code without timed_out flag is treated as an error."""
        result = _parse(exit_code=-1, timed_out=False)
        assert result.status == TaskStatus.ERROR

    def test_pytest_errored_tests(self) -> None:
        """pytest exit code 2 means errors during collection or setup."""
        result = _parse(
            stdout="ERROR collecting test_example.py",
            exit_code=2,
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_long_running_success(self) -> None:
        """Successful tests should still succeed regardless of output size."""
        output = ExecutionOutput(
            stdout="test_a ... ok\ntest_b ... ok\n" * 1000 + "===== 2000 passed =====",
            exit_code=0,
        )
        result = _PARSER.parse(output, task_id="t1", language="python")
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0


# ===========================================================================
# 9. Language-specific heuristics
# ===========================================================================


class TestLanguageSpecificHeuristics:
    """Test that language hint improves error categorization."""

    def test_python_traceback_maps_to_test_failure(self) -> None:
        result = _parse(
            stdout="Traceback (most recent call last):\n  File 'test_foo.py', line 10\nAssertionError",
            exit_code=1,
            language="python",
        )
        assert result.status == TaskStatus.FAILURE
        assert result.error_category == ErrorCategory.TEST_FAILURE

    def test_r_library_loading_error(self) -> None:
        result = _parse(
            stderr="Error in library(ggplot2) : there is no package called 'ggplot2'",
            exit_code=1,
            language="r",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR

    def test_typescript_compilation_error(self) -> None:
        result = _parse(
            stderr="src/index.ts:12:5 - error TS2322: Type 'string' is not assignable to type 'number'",
            exit_code=2,
            language="typescript",
        )
        assert result.status == TaskStatus.ERROR
        assert result.error_category == ErrorCategory.SYNTAX_ERROR
