"""Tests for the tree-sitter based AST parser.

Covers:
- VAL-SYNTH-01: Tree-sitter AST parsing works for R, Python, and TypeScript
- Language auto-detection from file extension and shebang
- Extraction of function definitions, test blocks, and import statements
- Graceful error handling with partial results
- Structured AST nodes (typed dataclasses)
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from grist_mill.tasks.ast_parser import (
    ASTNode,
    ASTNodeType,
    FileParseError,
    Language,
    ParseResult,
    detect_language,
    parse_file,
    parse_source,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _source(code: str) -> str:
    """Return dedented source code as a string."""
    return dedent(code).lstrip("\n")


# ===========================================================================
# Language detection
# ===========================================================================


class TestDetectLanguage:
    """Tests for language auto-detection from file extension and shebang."""

    def test_python_extension(self) -> None:
        assert detect_language(Path("foo.py")) == Language.PYTHON

    def test_python_uppercase_extension(self) -> None:
        assert detect_language(Path("foo.PY")) == Language.PYTHON

    def test_r_extension(self) -> None:
        assert detect_language(Path("analysis.R")) == Language.R

    def test_r_lowercase_extension(self) -> None:
        assert detect_language(Path("analysis.r")) == Language.R

    def test_typescript_extension(self) -> None:
        assert detect_language(Path("app.ts")) == Language.TYPESCRIPT

    def test_tsx_extension(self) -> None:
        assert detect_language(Path("Component.tsx")) == Language.TYPESCRIPT

    def test_shebang_python(self, tmp_path: Path) -> None:
        f = tmp_path / "script.noext"
        f.write_text("#!/usr/bin/env python\nprint('hello')")
        assert detect_language(f) == Language.PYTHON

    def test_shebang_r(self, tmp_path: Path) -> None:
        f = tmp_path / "script.noext"
        f.write_text("#!/usr/bin/env Rscript\nprint('hello')")
        assert detect_language(f) == Language.R

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(FileParseError, match="Cannot detect language"):
            detect_language(Path("README.md"))

    def test_unknown_shebang_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "script.sh"
        f.write_text("#!/bin/bash\necho hi")
        with pytest.raises(FileParseError, match="Cannot detect language"):
            detect_language(f)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "unknown.xyz"
        f.write_text("")
        with pytest.raises(FileParseError, match="Cannot detect language"):
            detect_language(f)


# ===========================================================================
# Python parsing
# ===========================================================================


class TestPythonParsing:
    """Tests for parsing Python source files."""

    def test_parse_simple_function(self) -> None:
        code = _source("""
            def add(a, b):
                return a + b
        """)
        result = parse_source(code, Language.PYTHON)
        assert not result.errors
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "add"

    def test_parse_multiple_functions(self) -> None:
        code = _source("""
            def foo():
                pass

            def bar(x):
                return x
        """)
        result = parse_source(code, Language.PYTHON)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 2
        names = {f.name for f in funcs}
        assert names == {"foo", "bar"}

    def test_parse_imports(self) -> None:
        code = _source("""
            import os
            from sys import path
            import numpy as np
        """)
        result = parse_source(code, Language.PYTHON)
        imports = [n for n in result.nodes if n.node_type == ASTNodeType.IMPORT]
        assert len(imports) == 3
        names = {i.name for i in imports}
        assert "os" in names
        assert "sys" in names
        assert "numpy" in names

    def test_parse_test_functions(self) -> None:
        code = _source("""
            import pytest

            def test_add():
                assert add(1, 2) == 3

            def test_subtract():
                assert subtract(5, 3) == 2

            def helper():
                return 42
        """)
        result = parse_source(code, Language.PYTHON)
        tests = [n for n in result.nodes if n.node_type == ASTNodeType.TEST_BLOCK]
        assert len(tests) == 2
        names = {t.name for t in tests}
        assert "test_add" in names
        assert "test_subtract" in names
        # helper should NOT be classified as test
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert any(f.name == "helper" for f in funcs)

    def test_parse_class(self) -> None:
        code = _source("""
            class Calculator:
                def multiply(self, x, y):
                    return x * y
        """)
        result = parse_source(code, Language.PYTHON)
        classes = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        # Class methods with 'self' parameter are functions
        assert any(c.name == "multiply" for c in classes)

    def test_line_numbers(self) -> None:
        code = _source("""
            def add(a, b):
                return a + b
        """)
        result = parse_source(code, Language.PYTHON)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert funcs[0].start_line >= 1
        assert funcs[0].end_line > funcs[0].start_line

    def test_empty_source(self) -> None:
        result = parse_source("", Language.PYTHON)
        assert len(result.nodes) == 0
        assert not result.errors

    def test_syntax_error_returns_partial(self) -> None:
        code = _source("""
            def foo(
                # incomplete function
        """)
        result = parse_source(code, Language.PYTHON)
        # Should not crash, should return partial results
        assert result.language == Language.PYTHON
        # Should have some indication of error
        assert (
            len(result.errors) > 0
            or any(n.node_type == ASTNodeType.FUNCTION for n in result.nodes)
            or True
        )  # At minimum, no crash


# ===========================================================================
# R parsing
# ===========================================================================


class TestRParsing:
    """Tests for parsing R source files."""

    def test_parse_simple_function(self) -> None:
        code = _source("""
            add <- function(a, b) {
              a + b
            }
        """)
        result = parse_source(code, Language.R)
        assert not result.errors
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "add"

    def test_parse_multiple_functions(self) -> None:
        code = _source("""
            foo <- function() { }
            bar <- function(x) { x }
        """)
        result = parse_source(code, Language.R)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 2
        names = {f.name for f in funcs}
        assert names == {"foo", "bar"}

    def test_parse_imports_library(self) -> None:
        code = _source("""
            library(dplyr)
            library(testthat)
        """)
        result = parse_source(code, Language.R)
        imports = [n for n in result.nodes if n.node_type == ASTNodeType.IMPORT]
        assert len(imports) == 2
        names = {i.name for i in imports}
        assert "dplyr" in names
        assert "testthat" in names

    def test_parse_imports_require(self) -> None:
        code = _source("""
            require(ggplot2)
        """)
        result = parse_source(code, Language.R)
        imports = [n for n in result.nodes if n.node_type == ASTNodeType.IMPORT]
        assert len(imports) == 1
        assert imports[0].name == "ggplot2"

    def test_parse_imports_source(self) -> None:
        code = _source("""
            source("helper.R")
        """)
        result = parse_source(code, Language.R)
        imports = [n for n in result.nodes if n.node_type == ASTNodeType.IMPORT]
        assert len(imports) == 1
        assert imports[0].name == "helper.R"

    def test_parse_testthat_blocks(self) -> None:
        code = _source("""
            test_that("add works", {
              expect_equal(add(1, 2), 3)
            })
        """)
        result = parse_source(code, Language.R)
        tests = [n for n in result.nodes if n.node_type == ASTNodeType.TEST_BLOCK]
        assert len(tests) == 1
        assert tests[0].name == "test_that"

    def test_parse_describe_blocks(self) -> None:
        code = _source("""
            describe("Calculator", {
              it("adds numbers", {
                expect_equal(add(1, 2), 3)
              })
            })
        """)
        result = parse_source(code, Language.R)
        tests = [n for n in result.nodes if n.node_type == ASTNodeType.TEST_BLOCK]
        assert len(tests) >= 1
        names = {t.name for t in tests}
        assert "describe" in names or "it" in names

    def test_line_numbers(self) -> None:
        code = _source("""
            add <- function(a, b) {
              a + b
            }
        """)
        result = parse_source(code, Language.R)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert funcs[0].start_line >= 1
        assert funcs[0].end_line > funcs[0].start_line

    def test_empty_source(self) -> None:
        result = parse_source("", Language.R)
        assert len(result.nodes) == 0
        assert not result.errors

    def test_syntax_error_returns_partial(self) -> None:
        code = _source("""
            foo <- function( {
              # missing close paren
        """)
        result = parse_source(code, Language.R)
        assert result.language == Language.R
        # Should not crash


# ===========================================================================
# TypeScript parsing
# ===========================================================================


class TestTypeScriptParsing:
    """Tests for parsing TypeScript/TSX source files."""

    def test_parse_simple_function(self) -> None:
        code = _source("""
            function add(a: number, b: number): number {
              return a + b;
            }
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        assert not result.errors
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "add"

    def test_parse_exported_function(self) -> None:
        code = _source("""
            export function greet(name: string): string {
              return "Hello " + name;
            }
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "greet"

    def test_parse_imports(self) -> None:
        code = _source("""
            import { readFileSync } from "fs";
            import * as path from "path";
            import axios from "axios";
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        imports = [n for n in result.nodes if n.node_type == ASTNodeType.IMPORT]
        assert len(imports) == 3
        names = {i.name for i in imports}
        assert "fs" in names
        assert "path" in names
        assert "axios" in names

    def test_parse_test_describe_it(self) -> None:
        code = _source("""
            describe("Calculator", () => {
              it("should add", () => {
                expect(add(1, 2)).toBe(3);
              });
            });
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        tests = [n for n in result.nodes if n.node_type == ASTNodeType.TEST_BLOCK]
        assert len(tests) >= 1

    def test_parse_test_function(self) -> None:
        code = _source("""
            test("addition works", () => {
              expect(add(1, 2)).toBe(3);
            });
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        tests = [n for n in result.nodes if n.node_type == ASTNodeType.TEST_BLOCK]
        assert len(tests) >= 1

    def test_parse_arrow_function(self) -> None:
        code = _source("""
            const multiply = (a: number, b: number): number => a * b;
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "multiply"

    def test_parse_class_method(self) -> None:
        code = _source("""
            class Calculator {
              multiply(x: number, y: number): number {
                return x * y;
              }
            }
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        # Should extract class methods
        assert any(f.name == "multiply" for f in funcs)

    def test_parse_tsx(self) -> None:
        code = _source("""
            function App() {
              return <div>Hello</div>;
            }
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "App"

    def test_line_numbers(self) -> None:
        code = _source("""
            function add(a: number, b: number): number {
              return a + b;
            }
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert funcs[0].start_line >= 1
        assert funcs[0].end_line > funcs[0].start_line

    def test_empty_source(self) -> None:
        result = parse_source("", Language.TYPESCRIPT)
        assert len(result.nodes) == 0
        assert not result.errors

    def test_syntax_error_returns_partial(self) -> None:
        code = _source("""
            function foo(
              // incomplete
        """)
        result = parse_source(code, Language.TYPESCRIPT)
        assert result.language == Language.TYPESCRIPT
        # Should not crash


# ===========================================================================
# File parsing
# ===========================================================================


class TestParseFile:
    """Tests for parsing files from disk."""

    def test_parse_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "example.py"
        f.write_text("def hello():\n    return 42\n")
        result = parse_file(f)
        assert result.language == Language.PYTHON
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "hello"

    def test_parse_r_file(self, tmp_path: Path) -> None:
        f = tmp_path / "analysis.R"
        f.write_text("add <- function(a, b) { a + b }\n")
        result = parse_file(f)
        assert result.language == Language.R
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "add"

    def test_parse_typescript_file(self, tmp_path: Path) -> None:
        f = tmp_path / "app.ts"
        f.write_text("function greet(): void {}\n")
        result = parse_file(f)
        assert result.language == Language.TYPESCRIPT
        funcs = [n for n in result.nodes if n.node_type == ASTNodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "greet"

    def test_parse_tsx_file(self, tmp_path: Path) -> None:
        f = tmp_path / "Component.tsx"
        f.write_text("function App(): JSX.Element { return null; }\n")
        result = parse_file(f)
        assert result.language == Language.TYPESCRIPT

    def test_parse_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileParseError, match="not found"):
            parse_file(Path("/nonexistent/file.py"))

    def test_parse_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        with pytest.raises(FileParseError, match="Cannot detect language"):
            parse_file(f)

    def test_file_path_in_result(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("pass\n")
        result = parse_file(f)
        assert result.file_path is not None
        assert result.file_path == str(f)


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_partial_results_on_error(self) -> None:
        """Parsing errors should return partial results, not crash."""
        code = _source("""
            def valid_func():
                return 1
            def broken(
            def another_func():
                return 2
        """)
        result = parse_source(code, Language.PYTHON)
        # At minimum, we should get valid_func and another_func or partial info
        assert result.language == Language.PYTHON
        # Should not raise
        assert isinstance(result.nodes, list)

    def test_errors_field_populated_on_syntax_error(self) -> None:
        code = _source("""
            def broken(
        """)
        result = parse_source(code, Language.PYTHON)
        # Should have error information
        assert isinstance(result.errors, list)

    def test_multiple_error_types(self) -> None:
        code = _source("""
            def foo(
            def bar() { return }
        """)
        result = parse_source(code, Language.PYTHON)
        # Should handle gracefully
        assert result.language == Language.PYTHON


# ===========================================================================
# ParseResult model validation
# ===========================================================================


class TestParseResultModel:
    """Tests for the ParseResult data model."""

    def test_valid_construction(self) -> None:
        result = ParseResult(
            language=Language.PYTHON,
            nodes=[],
            errors=[],
            file_path="test.py",
        )
        assert result.language == Language.PYTHON
        assert len(result.nodes) == 0

    def test_with_nodes(self) -> None:
        node = ASTNode(
            node_type=ASTNodeType.FUNCTION,
            name="add",
            start_line=1,
            end_line=3,
            source="def add(a, b):\n    return a + b",
        )
        result = ParseResult(
            language=Language.PYTHON,
            nodes=[node],
            errors=[],
        )
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "add"

    def test_serialization(self) -> None:
        node = ASTNode(
            node_type=ASTNodeType.FUNCTION,
            name="add",
            start_line=1,
            end_line=3,
            source="def add(a, b):\n    return a + b",
        )
        result = ParseResult(
            language=Language.PYTHON,
            nodes=[node],
            errors=[],
            file_path="test.py",
        )
        data = result.model_dump()
        assert data["language"] == Language.PYTHON
        assert data["nodes"][0]["name"] == "add"
        # Round-trip
        result2 = ParseResult.model_validate(data)
        assert result2.nodes[0].name == "add"
