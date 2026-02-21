"""Task templates for generating testing instructions."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .models import Difficulty, ExtractedPattern, TestPattern


@dataclass
class TaskTemplate(ABC):
    """Base class for task templates."""

    name: str
    description: str
    difficulty: Difficulty
    pattern_types: list[TestPattern] = field(default_factory=list)

    @abstractmethod
    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        """Generate the task instruction."""
        pass

    @abstractmethod
    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        """Generate reference test code."""
        pass

    def can_apply(self, patterns: list[ExtractedPattern]) -> bool:
        """Check if this template can be applied to the given patterns."""
        if not self.pattern_types:
            return True

        pattern_types = {p.pattern_type for p in patterns}
        return any(pt in pattern_types for pt in self.pattern_types)

    def _extract_function_name(self, context: str) -> str:
        """Extract function name from context string."""
        for line in context.split("\n"):
            if line.startswith("Function: "):
                return line.split("Function: ", 1)[1].strip()
        return "the target function"

    def _extract_package_name(self, context: str) -> str:
        """Extract package name from context string."""
        for line in context.split("\n"):
            if line.startswith("Package: "):
                return line.split("Package: ", 1)[1].split(" ")[0].strip()
        return "the package"

    def _get_real_reference_code(self, patterns: list[ExtractedPattern]) -> str:
        """Build reference test from real extracted pattern code snippets."""
        if not patterns:
            return "# No reference patterns available"

        # Use actual test code from the package
        parts = []
        for p in patterns[:3]:
            snippet = p.code_snippet.strip()
            if snippet and len(snippet) > 20:
                parts.append(snippet)

        if parts:
            return "\n\n".join(parts)

        return "# No substantive reference patterns available"


@dataclass
class WriteTestTemplate(TaskTemplate):
    """Template for writing tests from scratch."""

    name: str = "write_test"
    description: str = "Write a complete test file for a function"
    difficulty: Difficulty = Difficulty.MEDIUM
    pattern_types: list[TestPattern] = field(default_factory=lambda: [TestPattern.TEST_THAT])

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)
        pkg_name = self._extract_package_name(context)

        # Collect expectations used in existing patterns
        expect_types = set()
        for p in patterns:
            expect_types.update(p.expectations)
        expect_list = (
            ", ".join(sorted(expect_types)[:5]) if expect_types else "expect_equal, expect_error"
        )

        instruction = f"""Write a complete testthat test file for `{pkg_name}::{func_name}`.

{context}

Requirements:
1. Use test_that() blocks with descriptive names that reference `{func_name}`
2. Test basic functionality using appropriate expectations ({expect_list})
3. Test edge cases: NULL input, empty strings, missing arguments
4. Test error conditions with expect_error()
5. Each test_that() block should test one logical behavior
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        return self._get_real_reference_code(patterns)


@dataclass
class AddEdgeCaseTemplate(TaskTemplate):
    """Template for adding edge case tests."""

    name: str = "add_edge_case"
    description: str = "Add edge case tests to existing test file"
    difficulty: Difficulty = Difficulty.HARD
    pattern_types: list[TestPattern] = field(default_factory=lambda: [TestPattern.TEST_THAT])

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)

        existing_tests = ""
        if patterns:
            existing_tests = "\nExisting tests for reference:\n"
            for p in patterns[:3]:
                existing_tests += f"```r\n{p.code_snippet}\n```\n"

        instruction = f"""Add comprehensive edge case tests for `{func_name}`.

{existing_tests}

The existing tests cover basic functionality. Add tests for:
1. NULL inputs to each argument
2. Empty vectors/strings: character(0), "", integer(0)
3. Boundary values: 0, 1, -1, NA, NaN, Inf
4. Invalid input types (passing numeric where character expected, etc.)
5. Missing arguments (use expect_error with class matching)
6. Very long inputs (strings > 1000 chars, vectors > 10000 elements)
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        return self._get_real_reference_code(patterns)


@dataclass
class FixFailingTestTemplate(TaskTemplate):
    """Template for fixing failing tests."""

    name: str = "fix_failing_test"
    description: str = "Fix a failing test based on error message"
    difficulty: Difficulty = Difficulty.EASY
    pattern_types: list[TestPattern] = field(default_factory=lambda: [TestPattern.TEST_THAT])

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)

        # Show the test code that "fails"
        test_code = ""
        if patterns:
            test_code = f"\nFailing test:\n```r\n{patterns[0].code_snippet}\n```\n"

        instruction = f"""A test for `{func_name}` is failing. Diagnose and fix the issue.

{test_code}

Steps:
1. Read the function source code to understand expected behavior
2. Compare the test expectations with actual function behavior
3. Fix the test if expectations are wrong, or fix the function if it has a bug
4. Ensure the fix maintains backward compatibility
5. Run the tests to verify the fix
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        return self._get_real_reference_code(patterns)


@dataclass
class RefactorTestTemplate(TaskTemplate):
    """Template for refactoring tests."""

    name: str = "refactor_test"
    description: str = "Refactor tests to use better patterns"
    difficulty: Difficulty = Difficulty.MEDIUM
    pattern_types: list[TestPattern] = field(
        default_factory=lambda: [TestPattern.TEST_THAT, TestPattern.DESCRIBE_IT]
    )

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)

        pattern_info = ""
        if patterns:
            pattern_info = "\nCurrent test patterns to refactor:\n"
            for p in patterns[:5]:
                pattern_info += f"- {p.pattern_type!s} at {p.source_file}:{p.line_number}\n"
                if p.code_snippet:
                    # Show first 3 lines of each
                    lines = p.code_snippet.strip().split("\n")[:3]
                    pattern_info += "  ```r\n  " + "\n  ".join(lines) + "\n  ```\n"

        instruction = f"""Refactor the existing tests for `{func_name}` to follow better testing practices.

{pattern_info}

Goals:
1. Use describe()/it() blocks for better organization where appropriate
2. Reduce code duplication with helper functions or local fixtures
3. Improve test descriptions to be specific about what behavior is verified
4. Extract common setup into withr::local_* fixtures
5. Each test should have a single responsibility
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        return self._get_real_reference_code(patterns)


@dataclass
class UseFixtureTemplate(TaskTemplate):
    """Template for adding test fixtures."""

    name: str = "use_fixture"
    description: str = "Add test fixtures using withr"
    difficulty: Difficulty = Difficulty.MEDIUM
    pattern_types: list[TestPattern] = field(
        default_factory=lambda: [TestPattern.WITH_FIXTURE, TestPattern.TEST_THAT]
    )

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)

        # Find any existing fixture patterns
        fixture_examples = ""
        fixture_patterns = [p for p in patterns if p.pattern_type == TestPattern.WITH_FIXTURE]
        if fixture_patterns:
            fixture_examples = "\nExisting fixture usage in this package:\n"
            for p in fixture_patterns[:2]:
                fixture_examples += f"```r\n{p.code_snippet}\n```\n"

        instruction = f"""Add proper test fixtures to the tests for `{func_name}` using the withr package.

{fixture_examples}

Requirements:
1. Identify tests that modify global state (options, env vars, working directory)
2. Wrap state modifications with withr::local_* for automatic cleanup:
   - withr::local_options() for R options
   - withr::local_envvar() for environment variables
   - withr::local_tempdir() for temporary directories
   - withr::local_file() for temporary files
3. Ensure tests can run in any order without side effects
4. Use local_* (not with_*) inside test_that() blocks
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        # Prefer fixture patterns for reference
        fixture_patterns = [p for p in patterns if p.pattern_type == TestPattern.WITH_FIXTURE]
        if fixture_patterns:
            return self._get_real_reference_code(fixture_patterns)
        return self._get_real_reference_code(patterns)


@dataclass
class AddMockingTemplate(TaskTemplate):
    """Template for adding mocking to tests."""

    name: str = "add_mocking"
    description: str = "Add mocking for external dependencies"
    difficulty: Difficulty = Difficulty.HARD
    pattern_types: list[TestPattern] = field(
        default_factory=lambda: [TestPattern.LOCAL_MOCKED_BINDINGS]
    )

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)

        mock_examples = ""
        mock_patterns = [p for p in patterns if p.pattern_type == TestPattern.LOCAL_MOCKED_BINDINGS]
        if mock_patterns:
            mock_examples = "\nExisting mocking patterns in this package:\n"
            for p in mock_patterns[:2]:
                mock_examples += f"```r\n{p.code_snippet}\n```\n"

        instruction = f"""Add mocking to isolate `{func_name}` from its external dependencies.

{mock_examples}

Requirements:
1. Identify external dependencies in the function (system calls, file I/O, other packages)
2. Use testthat::local_mocked_bindings() to mock each dependency
3. Test both success and failure scenarios via mock return values
4. Verify the function handles mock errors gracefully
5. Ensure tests are deterministic (no network calls, no file system side effects)
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        mock_patterns = [p for p in patterns if p.pattern_type == TestPattern.LOCAL_MOCKED_BINDINGS]
        if mock_patterns:
            return self._get_real_reference_code(mock_patterns)
        return self._get_real_reference_code(patterns)


@dataclass
class SnapshotTestTemplate(TaskTemplate):
    """Template for adding snapshot tests."""

    name: str = "snapshot_test"
    description: str = "Add snapshot tests for complex outputs"
    difficulty: Difficulty = Difficulty.EASY
    pattern_types: list[TestPattern] = field(default_factory=lambda: [TestPattern.EXPECT_SNAPSHOT])

    def generate_instruction(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
        context: str,
    ) -> str:
        func_name = self._extract_function_name(context)
        pkg_name = self._extract_package_name(context)

        snapshot_examples = ""
        snap_patterns = [p for p in patterns if p.pattern_type == TestPattern.EXPECT_SNAPSHOT]
        if snap_patterns:
            snapshot_examples = "\nExisting snapshot tests in this package:\n"
            for p in snap_patterns[:2]:
                snapshot_examples += f"```r\n{p.code_snippet}\n```\n"

        instruction = f"""Add snapshot tests for `{pkg_name}::{func_name}` to capture complex output.

{snapshot_examples}

Snapshot testing is appropriate when the function produces:
- Formatted text output (messages, errors, warnings)
- Complex data structures that are hard to test with expect_equal
- Console output with ANSI formatting

Requirements:
1. Use expect_snapshot() for capturing printed output
2. Use expect_snapshot(error = TRUE) for error message snapshots
3. Use expect_snapshot(transform = ...) if output contains volatile data (timestamps, paths)
4. Test with representative inputs that exercise different output formats
"""
        return instruction

    def generate_reference_test(
        self,
        function_code: str | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        snap_patterns = [p for p in patterns if p.pattern_type == TestPattern.EXPECT_SNAPSHOT]
        if snap_patterns:
            return self._get_real_reference_code(snap_patterns)
        return self._get_real_reference_code(patterns)


class TemplateRegistry:
    """Registry of available task templates."""

    def __init__(self) -> None:
        self._templates: dict[str, TaskTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        default_templates = [
            WriteTestTemplate(),
            AddEdgeCaseTemplate(),
            FixFailingTestTemplate(),
            RefactorTestTemplate(),
            UseFixtureTemplate(),
            AddMockingTemplate(),
            SnapshotTestTemplate(),
        ]

        for template in default_templates:
            self.register(template)

    def register(self, template: TaskTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> TaskTemplate | None:
        return self._templates.get(name)

    def get_all(self) -> list[TaskTemplate]:
        return list(self._templates.values())

    def get_for_patterns(
        self, patterns: list[ExtractedPattern], difficulty: Difficulty | None = None
    ) -> list[TaskTemplate]:
        applicable = []

        for template in self._templates.values():
            if template.can_apply(patterns) and (
                difficulty is None or template.difficulty == difficulty
            ):
                applicable.append(template)

        return applicable

    def get_random(
        self, patterns: list[ExtractedPattern], difficulty: Difficulty | None = None
    ) -> TaskTemplate | None:
        applicable = self.get_for_patterns(patterns, difficulty)
        if not applicable:
            return None
        return random.choice(applicable)
