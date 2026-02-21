"""Extract testing patterns from R package source."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .ast_parser import RASTParser
from .models import ExtractedPattern, TestPattern

logger = logging.getLogger(__name__)


class PatternExtractor:
    """Extract testing patterns from R package source code."""

    def __init__(self, package_path: Path) -> None:
        """Initialize the pattern extractor.

        Args:
            package_path: Path to the R package root directory.
        """
        self.package_path = Path(package_path)
        self.parser = RASTParser()
        self._validate_package()

    def _validate_package(self) -> None:
        """Ensure package has expected structure.

        Raises:
            ValueError: If not a valid R package.
        """
        if not (self.package_path / "DESCRIPTION").exists():
            raise ValueError(f"Not an R package: {self.package_path}")

    @property
    def package_info(self) -> dict[str, Any]:
        """Get package information from DESCRIPTION file.

        Returns:
            Dictionary with package info.
        """
        return self._parse_description()

    def _parse_description(self) -> dict[str, Any]:
        """Parse the DESCRIPTION file.

        Returns:
            Dictionary with package info.
        """
        desc_path = self.package_path / "DESCRIPTION"
        content = desc_path.read_text(encoding="utf-8")

        info: dict[str, Any] = {
            "name": "",
            "version": "",
            "title": "",
            "description": "",
            "dependencies": [],
            "test_dependencies": [],
            "has_tests": False,
            "test_framework": None,
        }

        current_field = ""
        current_value: list[str] = []

        def save_field() -> None:
            if current_field and current_value:
                value = "\n".join(current_value).strip()
                if current_field == "Package":
                    info["name"] = value
                elif current_field == "Version":
                    info["version"] = value
                elif current_field == "Title":
                    info["title"] = value
                elif current_field == "Description":
                    info["description"] = value
                elif current_field == "Depends" or current_field == "Imports":
                    info["dependencies"].extend(self._parse_deps(value))
                elif current_field == "Suggests":
                    deps = self._parse_deps(value)
                    info["dependencies"].extend(deps)
                    info["test_dependencies"].extend(deps)

        for line in content.split("\n"):
            if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                save_field()
                parts = line.split(":", 1)
                current_field = parts[0].strip()
                current_value = [parts[1].strip()] if len(parts) > 1 else []
            else:
                current_value.append(line.strip())

        save_field()

        # Check for tests
        tests_dir = self.package_path / "tests"
        info["has_tests"] = tests_dir.exists()

        # Detect test framework
        if (tests_dir / "testthat.R").exists():
            info["test_framework"] = "testthat"
        elif tests_dir.exists():
            info["test_framework"] = "unknown"

        return info

    def _parse_deps(self, dep_string: str) -> list[str]:
        """Parse dependency string into list of package names.

        Args:
            dep_string: Dependency string from DESCRIPTION.

        Returns:
            List of package names.
        """
        deps = []
        for part in dep_string.split(","):
            part = part.strip()
            if not part:
                continue
            # Remove version specifications
            pkg = part.split("(")[0].strip()
            if pkg:
                deps.append(pkg)
        return deps

    def iter_test_files(self) -> Iterator[Path]:
        """Iterate over all testthat test files.

        Yields:
            Path to each test file.
        """
        tests_dir = self.package_path / "tests" / "testthat"
        if tests_dir.exists():
            yield from tests_dir.glob("test-*.R")
            # Also include helper files and setup files
            yield from tests_dir.glob("helper-*.R")
            yield from tests_dir.glob("setup-*.R")

    def iter_source_files(self) -> Iterator[Path]:
        """Iterate over all R source files in R/ directory.

        Yields:
            Path to each R source file.
        """
        r_dir = self.package_path / "R"
        if r_dir.exists():
            yield from r_dir.glob("*.R")

    def extract_all_patterns(self) -> list[ExtractedPattern]:
        """Extract all testing patterns from package tests.

        Returns:
            List of extracted patterns.
        """
        patterns = []

        for test_file in self.iter_test_files():
            try:
                file_patterns = self.extract_from_file(test_file)
                patterns.extend(file_patterns)
            except Exception as e:
                logger.warning(f"Failed to parse {test_file}: {e}")

        return patterns

    def extract_from_file(self, test_file: Path) -> list[ExtractedPattern]:
        """Extract patterns from a single test file.

        Args:
            test_file: Path to the test file.

        Returns:
            List of extracted patterns.
        """
        tree = self.parser.parse_file(test_file)
        source = test_file.read_text().encode("utf-8")

        patterns: list[ExtractedPattern] = []

        # Extract test_that blocks
        for block in self.parser.find_test_that_blocks(tree, source):
            expectations = self._extract_expectations_from_block(block["node"], source)

            patterns.append(
                ExtractedPattern(
                    pattern_type=TestPattern.TEST_THAT,
                    source_file=str(test_file.relative_to(self.package_path)),
                    line_number=block["line_number"],
                    code_snippet=block["code"],
                    function_name=self._identify_function_from_description(
                        block.get("description", "")
                    ),
                    context_before="",
                    context_after="",
                    expectations=[e["expect_type"] for e in expectations],
                )
            )

        # Extract describe/it blocks
        for block in self.parser.find_describe_blocks(tree, source):
            for it_block in block.get("it_blocks", []):
                expectations = self._extract_expectations_from_text(it_block.get("code", ""))

                patterns.append(
                    ExtractedPattern(
                        pattern_type=TestPattern.DESCRIBE_IT,
                        source_file=str(test_file.relative_to(self.package_path)),
                        line_number=it_block.get("line_number", 0),
                        code_snippet=it_block.get("code", ""),
                        function_name=self._identify_function_from_description(
                            block.get("description", "")
                        ),
                        context_before=block.get("description", ""),
                        context_after=it_block.get("description", ""),
                        expectations=expectations,
                    )
                )

        # Extract withr usage
        for call in self.parser.find_withr_calls(tree, source):
            patterns.append(
                ExtractedPattern(
                    pattern_type=TestPattern.WITH_FIXTURE,
                    source_file=str(test_file.relative_to(self.package_path)),
                    line_number=call["line_number"],
                    code_snippet=call["code"],
                    function_name=None,
                    context_before="",
                    context_after="",
                    expectations=[],
                )
            )

        # Extract mock calls
        for call in self.parser.find_mock_calls(tree, source):
            if call.get("type") == "testthat":
                pattern_type = TestPattern.LOCAL_MOCKED_BINDINGS
            else:
                # Skip mockery mocks for now since we don't have a pattern for them
                continue

            patterns.append(
                ExtractedPattern(
                    pattern_type=pattern_type,
                    source_file=str(test_file.relative_to(self.package_path)),
                    line_number=call["line_number"],
                    code_snippet=call["code"],
                    function_name=None,
                    context_before="",
                    context_after="",
                    expectations=[],
                )
            )

        # Extract snapshot calls
        for call in self.parser.find_snapshot_calls(tree, source):
            patterns.append(
                ExtractedPattern(
                    pattern_type=TestPattern.EXPECT_SNAPSHOT,
                    source_file=str(test_file.relative_to(self.package_path)),
                    line_number=call["line_number"],
                    code_snippet=call["code"],
                    function_name=None,
                    context_before="",
                    context_after="",
                    expectations=["expect_snapshot"],
                )
            )

        return patterns

    def get_source_function(self, function_name: str) -> dict[str, Any] | None:
        """Get the R function being tested from R/ directory.

        Args:
            function_name: Name of the function to find.

        Returns:
            Dictionary with function info or None if not found.
        """
        # First try: look for R/{function_name}.R
        direct_path = self.package_path / "R" / f"{function_name}.R"
        if direct_path.exists():
            return self._extract_function_from_file(direct_path, function_name)

        # Second try: search all R files
        for r_file in self.iter_source_files():
            func = self._extract_function_from_file(r_file, function_name)
            if func:
                return func

        return None

    def _extract_function_from_file(
        self, file_path: Path, function_name: str
    ) -> dict[str, Any] | None:
        """Extract a specific function from an R file.

        Args:
            file_path: Path to the R source file.
            function_name: Name of the function to extract.

        Returns:
            Dictionary with function info or None if not found.
        """
        tree = self.parser.parse_file(file_path)
        source = file_path.read_text().encode("utf-8")

        for func_def in self.parser.find_function_definitions(tree, source):
            if func_def["name"] == function_name:
                return {
                    "name": function_name,
                    "source_file": str(file_path.relative_to(self.package_path)),
                    "line_number": func_def["line"],
                    "code": func_def["code"],
                    "parameters": func_def.get("parameters", []),
                }

        return None

    def get_all_source_functions(self) -> list[dict[str, Any]]:
        """Get all exported functions from the package.

        Returns:
            List of function info dictionaries.
        """
        functions = []

        for r_file in self.iter_source_files():
            try:
                funcs = self._extract_all_functions_from_file(r_file)
                functions.extend(funcs)
            except Exception as e:
                logger.warning(f"Failed to parse {r_file}: {e}")

        return functions

    def _extract_all_functions_from_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract all functions from an R file.

        Args:
            file_path: Path to the R source file.

        Returns:
            List of function info dictionaries.
        """
        tree = self.parser.parse_file(file_path)
        source = file_path.read_text().encode("utf-8")

        functions = []
        for func_def in self.parser.find_function_definitions(tree, source):
            functions.append(
                {
                    "name": func_def["name"],
                    "source_file": str(file_path.relative_to(self.package_path)),
                    "line_number": func_def["line"],
                    "code": func_def["code"],
                    "parameters": func_def.get("parameters", []),
                }
            )

        return functions

    def _extract_expectations_from_block(self, node: Any, source: bytes) -> list[dict[str, Any]]:
        """Extract expect_* calls from a test block.

        Args:
            node: Tree-sitter node for the block.
            source: Source code as bytes.

        Returns:
            List of expectation dictionaries.
        """
        expectations = []

        # Walk the tree and find expect_ calls
        def walk(n: Any) -> None:
            if n.type == "call":
                for child in n.children:
                    if child.type == "identifier":
                        name = self.parser.extract_node_text(child, source)
                        if name.startswith("expect_"):
                            expectations.append(
                                {
                                    "expect_type": name,
                                    "code": self.parser.extract_node_text(n, source),
                                }
                            )
                            break

            for child in n.children:
                walk(child)

        walk(node)
        return expectations

    def _extract_expectations_from_text(self, code: str) -> list[str]:
        """Extract expect_* call types from code text.

        Args:
            code: R code as string.

        Returns:
            List of expectation type names.
        """
        # Find all expect_* calls
        pattern = r"expect_[a-z_]+"
        return list(set(re.findall(pattern, code)))

    def _identify_function_from_description(self, description: str) -> str | None:
        """Try to identify the function being tested from test description.

        Args:
            description: Test description string.

        Returns:
            Function name or None.
        """
        # Common patterns:
        # "function_name() does X"
        # "function_name works with Y"
        # "function_name handles Z"

        if not description:
            return None

        # Try to extract function name from description
        match = re.match(r"^([a-z][a-z0-9._]*)\s*[\(\s]", description)
        if match:
            return match.group(1)

        return None

    def get_pattern_summary(self) -> dict[str, int]:
        """Get a summary of pattern counts by type.

        Returns:
            Dictionary mapping pattern type to count.
        """
        patterns = self.extract_all_patterns()
        summary: dict[str, int] = {}

        for pattern in patterns:
            key = str(pattern.pattern_type)
            summary[key] = summary.get(key, 0) + 1

        return summary
