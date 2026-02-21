"""R AST Parser using tree-sitter-language-pack."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser, Query, Tree
from tree_sitter_language_pack import get_language, get_parser

logger = logging.getLogger(__name__)


class RASTParser:
    """Parse R code using tree-sitter to extract testing patterns."""

    def __init__(self) -> None:
        self.language: Language = get_language("r")
        self.parser: Parser = get_parser("r")

    def parse_file(self, file_path: Path) -> Tree:
        """Parse an R file and return the tree.

        Args:
            file_path: Path to the R source file.

        Returns:
            Tree-sitter Tree object.
        """
        source = file_path.read_text(encoding="utf-8")
        return self.parser.parse(source.encode("utf-8"))

    def parse_code(self, code: str) -> Tree:
        """Parse R code string and return the tree.

        Args:
            code: R source code as string.

        Returns:
            Tree-sitter Tree object.
        """
        return self.parser.parse(code.encode("utf-8"))

    def parse_string(self, source: str) -> Tree:
        """Parse R code from a string (alias for parse_code).

        Args:
            source: R source code as string.

        Returns:
            Tree-sitter Tree object.
        """
        return self.parser.parse(source.encode("utf-8"))

    def extract_node_text(self, node: Node, source: bytes) -> str:
        """Extract text for a node.

        Args:
            node: Tree-sitter Node.
            source: Original source code as bytes.

        Returns:
            Extracted text as string.
        """
        return source[node.start_byte : node.end_byte].decode("utf-8")

    def find_calls_by_name(
        self, tree: Tree, source: bytes, *names: str
    ) -> Generator[dict, None, None]:
        """Find all function calls with specific names.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.
            *names: Function names to search for.

        Yields:
            Dictionary with node, name, code, and line info.
        """

        def walk(node: Node) -> Generator[dict, None, None]:
            if node.type == "call" or node.type == "call_expression":
                # Get the function name
                func_node = node.child_by_field_name("function")
                if func_node:
                    func_name = self.extract_node_text(func_node, source)
                    # Handle namespaced calls like testthat::test_that
                    clean_name = func_name.split("::")[-1]
                    if clean_name in names or func_name in names:
                        yield {
                            "node": node,
                            "name": func_name,
                            "code": self.extract_node_text(node, source),
                            "line": node.start_point[0] + 1,
                        }
            for child in node.children:
                yield from walk(child)

        yield from walk(tree.root_node)

    def find_test_that_blocks(self, tree: Tree, source: bytes) -> list[dict]:
        """Find all test_that() blocks.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of test_that block info dictionaries.
        """
        results = []
        for call in self.find_calls_by_name(tree, source, "test_that"):
            node = call["node"]
            desc = self._extract_test_description(node, source)
            body = self._extract_call_body(node, source)
            results.append(
                {
                    "description": desc,
                    "code": body,
                    "full_call": call["code"],
                    "line_number": call["line"],
                    "node": node,
                }
            )
        return results

    def find_describe_blocks(self, tree: Tree, source: bytes) -> list[dict]:
        """Find describe() blocks.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of describe block info dictionaries.
        """
        results = []
        for call in self.find_calls_by_name(tree, source, "describe"):
            node = call["node"]
            desc = self._extract_test_description(node, source)
            it_calls = self._find_nested_calls(node, source, "it")
            results.append(
                {
                    "description": desc,
                    "code": call["code"],
                    "line_number": call["line"],
                    "it_blocks": it_calls,
                    "node": node,
                }
            )
        return results

    def find_expect_calls(self, tree: Tree, source: bytes) -> list[dict]:
        """Find all expect_* calls.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of expect call info dictionaries.
        """
        expect_names = [
            "expect_equal",
            "expect_identical",
            "expect_equivalent",
            "expect_true",
            "expect_false",
            "expect_error",
            "expect_warning",
            "expect_message",
            "expect_snapshot",
            "expect_type",
            "expect_s3_class",
            "expect_length",
            "expect_match",
            "expect_null",
            "expect_named",
            "expect_setequal",
            "expect_contains",
            "expect_in",
        ]
        results = []
        for call in self.find_calls_by_name(tree, source, *expect_names):
            node = call["node"]
            results.append(
                {
                    "expect_type": call["name"].split("::")[-1],
                    "code": call["code"],
                    "arguments": self._extract_arguments_text(node, source),
                    "line_number": call["line"],
                    "node": node,
                }
            )
        return results

    def find_withr_calls(self, tree: Tree, source: bytes) -> list[dict]:
        """Find withr::local_* calls for fixtures.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of withr call info dictionaries.
        """
        results = []
        source.decode("utf-8")

        for call in self.find_calls_by_name(
            tree,
            source,
            "local_options",
            "local_envvar",
            "local_tempfile",
            "local_tempdir",
            "local_package",
        ):
            # Check if it's a withr call
            code = call["code"]
            if "withr::" in code or "local_" in code:
                results.append(
                    {
                        "function": call["name"].split("::")[-1],
                        "code": code,
                        "line_number": call["line"],
                        "node": call["node"],
                    }
                )

        return results

    def find_mock_calls(self, tree: Tree, source: bytes) -> list[dict]:
        """Find mocking calls (mockery, testthat::local_mocked_bindings).

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of mock call info dictionaries.
        """
        results = []

        # Find namespaced calls to mockery::stub or mockery::mock
        for call in self.find_calls_by_name(tree, source, "mockery::stub", "mockery::mock"):
            func_name = call["name"].split("::")[-1]
            results.append(
                {
                    "type": "mockery",
                    "function": func_name,
                    "code": call["code"],
                    "line_number": call["line"],
                    "node": call["node"],
                }
            )

        # Find local_mocked_bindings calls
        for call in self.find_calls_by_name(tree, source, "local_mocked_bindings"):
            results.append(
                {
                    "type": "testthat",
                    "function": "local_mocked_bindings",
                    "code": call["code"],
                    "line_number": call["line"],
                    "node": call["node"],
                }
            )

        return results

    def find_snapshot_calls(self, tree: Tree, source: bytes) -> list[dict]:
        """Find snapshot testing calls.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of snapshot call info dictionaries.
        """
        snapshot_funcs = (
            "expect_snapshot",
            "expect_snapshot_value",
            "expect_snapshot_file",
            "expect_snapshot_error",
            "snapshot_accept",
            "snapshot_review",
            "snapshot_compare",
        )

        results = []
        for call in self.find_calls_by_name(tree, source, *snapshot_funcs):
            results.append(
                {
                    "function": call["name"].split("::")[-1],
                    "code": call["code"],
                    "line_number": call["line"],
                    "node": call["node"],
                }
            )
        return results

    def find_mocked_bindings(self, tree: Tree, source: bytes) -> list[dict]:
        """Find local_mocked_bindings() calls.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of mocked bindings call info dictionaries.
        """
        return list(self.find_calls_by_name(tree, source, "local_mocked_bindings"))

    def extract_function_definitions(self, tree: Tree, source: bytes) -> list[dict]:
        """Find all function definitions.

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of function definition info dictionaries.
        """
        functions = []

        def walk(node: Node) -> None:
            # R functions are typically: name <- function(...) { ... }
            if node.type == "binary_operator":
                left = node.child_by_field_name("lhs")
                right = node.child_by_field_name("rhs")
                if left and right and right.type == "function_definition":
                    func_name = self.extract_node_text(left, source)
                    func_body = self.extract_node_text(node, source)
                    params = self._extract_parameters(right, source)
                    functions.append(
                        {
                            "name": func_name,
                            "code": func_body,
                            "line": node.start_point[0] + 1,
                            "parameters": params,
                            "node": node,
                        }
                    )
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return functions

    def find_function_definitions(self, tree: Tree, source: bytes) -> list[dict]:
        """Find all function definitions (alias for extract_function_definitions).

        Args:
            tree: Parsed tree-sitter Tree.
            source: Original source code as bytes.

        Returns:
            List of function definition info dictionaries.
        """
        return self.extract_function_definitions(tree, source)

    def get_function_from_source(self, source: bytes, function_name: str) -> str | None:
        """Extract a specific function from source code.

        Args:
            source: Source code as bytes.
            function_name: Name of the function to extract.

        Returns:
            Function code or None if not found.
        """
        tree = self.parse_code(source.decode("utf-8"))
        functions = self.extract_function_definitions(tree, source)
        for func in functions:
            if func["name"] == function_name:
                return func["code"]
        return None

    def query_pattern(self, tree: Tree, query_str: str, source: bytes) -> list[dict[str, Any]]:
        """Run a tree-sitter query and return matches.

        Args:
            tree: Parsed tree-sitter Tree.
            query_str: Tree-sitter query string.
            source: Original source code as bytes.

        Returns:
            List of match dictionaries with node info and captures.
        """
        try:
            query = Query(self.language, query_str)
            # Use the correct API for tree-sitter captures
            captures = query.captures(tree.root_node)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []

        results: list[dict[str, Any]] = []
        current_match: dict[str, Any] = {}

        # Handle both old and new tree-sitter API
        for item in captures:
            # New API returns (capture_name, node) tuples
            if isinstance(item, tuple):
                capture_name, node = item
            else:
                # Old API or different format
                node = item[0] if isinstance(item, (list, tuple)) else item
                capture_name = (
                    item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else "match"
                )

            if capture_name not in current_match:
                current_match[capture_name] = []

            node_info = {
                "node": node,
                "text": self.extract_node_text(node, source),
                "start_point": node.start_point,
                "end_point": node.end_point,
            }
            current_match[capture_name].append(node_info)

            # If we have a primary capture, finalize the match
            if capture_name == "match":
                if current_match:
                    results.append(current_match)
                current_match = {}

        # Add any remaining captures
        if current_match:
            results.append(current_match)

        return results

    def _extract_parameters(self, func_node: Node, source: bytes) -> list[str]:
        """Extract parameter names from a function definition.

        Args:
            func_node: Function definition node.
            source: Source code as bytes.

        Returns:
            List of parameter names.
        """
        params = []

        # Find parameters node
        for child in func_node.children:
            if child.type == "parameters":
                for param in child.children:
                    if param.type == "identifier":
                        params.append(self.extract_node_text(param, source))
                    elif param.type == "parameter":
                        # Named parameter with default
                        for pchild in param.children:
                            if pchild.type == "identifier":
                                params.append(self.extract_node_text(pchild, source))
                                break
                    elif param.type == "default_parameter":
                        # Parameter with default value
                        for pchild in param.children:
                            if pchild.type == "identifier":
                                params.append(self.extract_node_text(pchild, source))
                                break

        return params

    def _extract_test_description(self, call_node: Node, source: bytes) -> str:
        """Extract the description string from a test_that or describe call.

        Args:
            call_node: Call node.
            source: Source code as bytes.

        Returns:
            Description string or empty string.
        """
        for child in call_node.children:
            if child.type == "arguments":
                for arg in child.children:
                    if arg.type == "argument":
                        for val in arg.children:
                            if val.type == "string":
                                text = self.extract_node_text(val, source)
                                # Remove quotes
                                if text.startswith('"') and text.endswith('"'):
                                    return text[1:-1]
                                if text.startswith("'") and text.endswith("'"):
                                    return text[1:-1]
                                return text
                    elif arg.type == "string":
                        text = self.extract_node_text(arg, source)
                        if text.startswith('"') and text.endswith('"'):
                            return text[1:-1]
                        if text.startswith("'") and text.endswith("'"):
                            return text[1:-1]
                        return text
        return ""

    def _extract_call_body(self, call_node: Node, source: bytes) -> str:
        """Extract the body/code block from a function call.

        Args:
            call_node: Call node.
            source: Source code as bytes.

        Returns:
            Body code as string.
        """
        for child in call_node.children:
            if child.type == "arguments":
                for arg in child.children:
                    if arg.type == "argument":
                        for val in arg.children:
                            if val.type == "brace_list":
                                return self.extract_node_text(val, source)
                    elif arg.type == "brace_list":
                        return self.extract_node_text(arg, source)
        return ""

    def _extract_arguments_text(self, call_node: Node, source: bytes) -> str:
        """Extract arguments text from a call node.

        Args:
            call_node: Call node.
            source: Source code as bytes.

        Returns:
            Arguments text as string.
        """
        for child in call_node.children:
            if child.type == "arguments":
                return self.extract_node_text(child, source)
        return ""

    def _find_nested_calls(
        self, parent_node: Node, source: bytes, target_name: str
    ) -> list[dict[str, Any]]:
        """Find nested calls with a specific name.

        Args:
            parent_node: Parent node to search within.
            source: Source code as bytes.
            target_name: Name of the call to find.

        Returns:
            List of nested call info dictionaries.
        """
        results = []

        def walk(node: Node) -> None:
            if node.type == "call":
                for child in node.children:
                    if child.type == "identifier":
                        name = self.extract_node_text(child, source)
                        if name == target_name:
                            results.append(
                                {
                                    "code": self.extract_node_text(node, source),
                                    "description": self._extract_test_description(node, source),
                                    "line_number": node.start_point[0] + 1,
                                }
                            )
                            break

            for child in node.children:
                walk(child)

        walk(parent_node)
        return results
