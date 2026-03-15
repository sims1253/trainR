"""Tree-sitter based AST parser for multi-language code analysis.

Parses R, Python, and TypeScript/TSX source files and extracts
function definitions, test blocks, and import statements.
Returns structured AST nodes as typed Pydantic models.

Handles parsing errors gracefully by returning partial results
along with any error information, rather than crashing.

Validates VAL-SYNTH-01.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FileParseError(Exception):
    """Raised when a file cannot be parsed (e.g., not found, unsupported language)."""

    pass


# ---------------------------------------------------------------------------
# Enums and Models
# ---------------------------------------------------------------------------


class Language(str, Enum):
    """Supported programming languages for AST parsing."""

    PYTHON = "python"
    R = "r"
    TYPESCRIPT = "typescript"


class ASTNodeType(str, Enum):
    """Type of AST node extracted from source code."""

    FUNCTION = "function"
    IMPORT = "import"
    TEST_BLOCK = "test_block"
    CLASS = "class"


class ASTNode(BaseModel):
    """A single AST node extracted from source code.

    Attributes:
        node_type: The kind of construct this node represents.
        name: The name/identifier of the construct (e.g., function name).
        start_line: 1-based line number where the construct starts.
        end_line: 1-based line number where the construct ends.
        source: The raw source text of the construct.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    node_type: ASTNodeType = Field(
        ...,
        description="Type of AST node (function, import, test_block, class).",
    )
    name: str = Field(
        ...,
        min_length=0,
        description="Name/identifier of the construct.",
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="1-based line number where the construct starts.",
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="1-based line number where the construct ends.",
    )
    source: str = Field(
        default="",
        description="Raw source text of the construct.",
    )


class ParseError(BaseModel):
    """Information about a parsing error encountered during AST extraction.

    Attributes:
        message: Human-readable description of the error.
        line: Line number where the error occurred, if known.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable error description.",
    )
    line: int | None = Field(
        default=None,
        ge=1,
        description="1-based line number where the error occurred.",
    )


class ParseResult(BaseModel):
    """Result of parsing a source file.

    Contains all extracted AST nodes and any errors encountered.
    Parsing errors do not prevent partial results from being returned.

    Attributes:
        language: The detected/specified language of the source.
        nodes: Extracted AST nodes (functions, imports, test blocks).
        errors: Parsing errors encountered (if any).
        file_path: Path to the file that was parsed, if known.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    language: Language = Field(
        ...,
        description="Language of the parsed source.",
    )
    nodes: list[ASTNode] = Field(
        default_factory=list,
        description="Extracted AST nodes.",
    )
    errors: list[ParseError] = Field(
        default_factory=list,
        description="Parsing errors encountered.",
    )
    file_path: str | None = Field(
        default=None,
        description="Path to the file that was parsed.",
    )


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# File extension to Language mapping
_EXTENSION_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".r": Language.R,
    ".R": Language.R,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
}

# Shebang patterns for files without extensions
_SHEBANG_MAP: list[tuple[str, Language]] = [
    (r"python", Language.PYTHON),
    (r"rscript", Language.R),
    (r"Rscript", Language.R),
]


def detect_language(file_path: Path) -> Language:
    """Auto-detect the programming language from a file's extension or shebang.

    Args:
        file_path: Path to the source file.

    Returns:
        The detected Language.

    Raises:
        FileParseError: If the language cannot be determined.
    """
    # First, try file extension (case-insensitive)
    ext = file_path.suffix
    ext_lower = ext.lower()
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext]
    if ext_lower in {k.lower() for k in _EXTENSION_MAP}:
        for key, lang in _EXTENSION_MAP.items():
            if key.lower() == ext_lower:
                return lang

    # Fall back to shebang detection
    if file_path.is_file():
        try:
            first_line = file_path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        except (OSError, IndexError):
            pass
        else:
            if first_line.startswith("#!"):
                for pattern, lang in _SHEBANG_MAP:
                    if re.search(pattern, first_line, re.IGNORECASE):
                        return lang

    msg = (
        f"Cannot detect language for file: {file_path}. "
        f"Supported extensions: {', '.join(sorted(_EXTENSION_MAP.keys()))}. "
        f"Supported shebangs: python, Rscript."
    )
    raise FileParseError(msg)


# ---------------------------------------------------------------------------
# Tree-sitter parser helper
# ---------------------------------------------------------------------------

# Map from Language to tree-sitter-language-pack parser name
_LANGUAGE_TO_PARSER_NAME: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.R: "r",
    Language.TYPESCRIPT: "typescript",
}


def _get_parser(language: Language):
    """Get a tree-sitter parser for the given language.

    Args:
        language: The language to get a parser for.

    Returns:
        A tree-sitter Parser instance configured for the language.

    Raises:
        FileParseError: If the tree-sitter-language-pack is not installed.
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:
        msg = (
            "tree-sitter-language-pack is required for AST parsing. "
            "Install it with: uv pip install tree-sitter-language-pack"
        )
        raise FileParseError(msg) from exc

    parser_name = _LANGUAGE_TO_PARSER_NAME[language]
    return get_parser(parser_name)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Node extraction utilities
# ---------------------------------------------------------------------------


def _node_text(node, source_bytes: bytes) -> str:
    """Extract the text content of a tree-sitter node."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _find_child_by_type(node, *types: str):
    """Find the first child matching one of the given types."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_child_text(node, *types: str, source_bytes: bytes = b"") -> str:
    """Find the text of the first child matching one of the given types."""
    child = _find_child_by_type(node, *types)
    if child is None:
        return ""
    return child.text.decode("utf-8", errors="replace") if child.text else ""


def _extract_identifier(node, source_bytes: bytes) -> str:
    """Extract an identifier name from a node and its children."""
    # Direct child identifier
    ident = _find_child_text(node, "identifier", "type_identifier", source_bytes=source_bytes)
    if ident:
        return ident
    # Check for named child
    if node.child_by_field_name("name") is not None:
        name_node = node.child_by_field_name("name")
        return name_node.text.decode("utf-8", errors="replace") if name_node.text else ""
    # Check children recursively for identifier
    for child in node.children:
        if child.type in ("identifier", "type_identifier"):
            return child.text.decode("utf-8", errors="replace") if child.text else ""
    return ""


def _extract_parameters_text(node) -> str:
    """Extract parameter text from function parameters."""
    params = _find_child_by_type(node, "parameters", "formal_parameters")
    if params is not None:
        return params.text.decode("utf-8", errors="replace") if params.text else ""
    return ""


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------

# Python function names that indicate test functions
_PYTHON_TEST_PREFIX = "test_"


def _extract_python_nodes(root, source_bytes: bytes) -> list[ASTNode]:
    """Extract AST nodes from a Python parse tree."""
    nodes: list[ASTNode] = []

    for child in root.children:
        if child.type == "import_statement":
            name = _find_child_text(
                child, "dotted_name", "aliased_import", source_bytes=source_bytes
            )
            # Handle: import os.path -> get first part
            if "." in name:
                name = name.split(".")[0]
            # Handle: import numpy as np -> get base module
            if " as " in name:
                name = name.split(" as ")[0].strip()
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.IMPORT,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )

        elif child.type == "import_from_statement":
            name = _find_child_text(child, "dotted_name", source_bytes=source_bytes)
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.IMPORT,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )

        elif child.type == "function_definition":
            name = _extract_identifier(child, source_bytes)
            is_test = name.startswith(_PYTHON_TEST_PREFIX)
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.TEST_BLOCK if is_test else ASTNodeType.FUNCTION,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )

        elif child.type == "class_definition":
            name = _extract_identifier(child, source_bytes)
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.CLASS,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )
            # Also extract methods from the class
            _extract_class_methods(child, source_bytes, nodes)

    return nodes


def _extract_class_methods(class_node, source_bytes: bytes, nodes: list[ASTNode]) -> None:
    """Extract function definitions from within a class body."""
    for child in class_node.children:
        if child.type == "block":
            for member in child.children:
                if member.type == "function_definition":
                    name = _extract_identifier(member, source_bytes)
                    is_test = name.startswith(_PYTHON_TEST_PREFIX)
                    nodes.append(
                        ASTNode(
                            node_type=ASTNodeType.TEST_BLOCK if is_test else ASTNodeType.FUNCTION,
                            name=name,
                            start_line=member.start_point[0] + 1,
                            end_line=member.end_point[0] + 1,
                            source=_node_text(member, source_bytes),
                        )
                    )


# ---------------------------------------------------------------------------
# R extraction
# ---------------------------------------------------------------------------

# R functions that indicate import/source statements
_R_IMPORT_FUNCTIONS = {"library", "require", "source"}

# R functions that indicate test blocks
_R_TEST_FUNCTIONS = {"test_that", "describe", "it", "context"}


def _extract_r_nodes(root, source_bytes: bytes) -> list[ASTNode]:
    """Extract AST nodes from an R parse tree."""
    nodes: list[ASTNode] = []

    for child in root.children:
        if child.type == "call":
            func_name = _extract_r_call_name(child)
            if func_name in _R_IMPORT_FUNCTIONS:
                name = _extract_r_import_arg(child, source_bytes)
                nodes.append(
                    ASTNode(
                        node_type=ASTNodeType.IMPORT,
                        name=name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=_node_text(child, source_bytes),
                    )
                )
            elif func_name in _R_TEST_FUNCTIONS:
                nodes.append(
                    ASTNode(
                        node_type=ASTNodeType.TEST_BLOCK,
                        name=func_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=_node_text(child, source_bytes),
                    )
                )

        elif child.type == "binary_operator":
            # R function definitions: name <- function(...) { }
            func_name = _extract_r_function_name(child)
            if func_name:
                nodes.append(
                    ASTNode(
                        node_type=ASTNodeType.FUNCTION,
                        name=func_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=_node_text(child, source_bytes),
                    )
                )
            # Also check for nested test_that calls inside binary_operator
            _extract_r_nested_calls(child, source_bytes, nodes)

    return nodes


def _extract_r_call_name(call_node) -> str:
    """Extract the function name from an R call expression."""
    for child in call_node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace") if child.text else ""
    return ""


def _extract_r_import_arg(call_node, source_bytes: bytes) -> str:
    """Extract the argument from an import call (library, require, source)."""
    for child in call_node.children:
        if child.type == "arguments":
            for arg in child.children:
                if arg.type in ("argument", "identifier", "string"):
                    text = arg.text.decode("utf-8", errors="replace") if arg.text else ""
                    # Strip quotes from string literals
                    text = text.strip("\"'")
                    return text
    return ""


def _extract_r_function_name(binary_node) -> str:
    """Extract the function name from an R function definition.

    R function definitions have the form: name <- function(...) { }
    The binary_operator has a left child (the name) and right child (function_definition).
    """
    func_def = _find_child_by_type(binary_node, "function_definition")
    if func_def is None:
        return ""
    # The name is the left sibling of the <- operator
    for child in binary_node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace") if child.text else ""
    return ""


def _extract_r_nested_calls(node, source_bytes: bytes, nodes: list[ASTNode]) -> None:
    """Recursively search for test_that/describe calls nested in R structures."""
    if node.type == "call":
        func_name = _extract_r_call_name(node)
        if func_name in _R_TEST_FUNCTIONS:
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.TEST_BLOCK,
                    name=func_name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    source=_node_text(node, source_bytes),
                )
            )
    for child in node.children:
        _extract_r_nested_calls(child, source_bytes, nodes)


# ---------------------------------------------------------------------------
# TypeScript extraction
# ---------------------------------------------------------------------------

# TypeScript patterns that indicate test blocks
_TS_TEST_FUNCTIONS = {"describe", "it", "test", "beforeEach", "afterEach", "beforeAll", "afterAll"}


def _extract_typescript_nodes(root, source_bytes: bytes) -> list[ASTNode]:
    """Extract AST nodes from a TypeScript/TSX parse tree."""
    nodes: list[ASTNode] = []

    for child in root.children:
        if child.type == "import_statement":
            name = _extract_ts_import_source(child, source_bytes)
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.IMPORT,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )

        elif child.type in ("function_declaration", "generator_function_declaration"):
            name = _extract_identifier(child, source_bytes)
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.FUNCTION,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )

        elif child.type == "export_statement":
            # Handle: export function foo() {}
            for sub in child.children:
                if sub.type in ("function_declaration", "generator_function_declaration"):
                    name = _extract_identifier(sub, source_bytes)
                    nodes.append(
                        ASTNode(
                            node_type=ASTNodeType.FUNCTION,
                            name=name,
                            start_line=sub.start_point[0] + 1,
                            end_line=sub.end_point[0] + 1,
                            source=_node_text(sub, source_bytes),
                        )
                    )

        elif child.type == "class_declaration":
            name = _extract_identifier(child, source_bytes)
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.CLASS,
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source=_node_text(child, source_bytes),
                )
            )
            # Extract methods
            _extract_ts_class_methods(child, source_bytes, nodes)

        elif child.type in ("variable_declaration", "lexical_declaration"):
            # Handle: const foo = (...) => ...;
            name = _extract_ts_arrow_name(child, source_bytes)
            if name:
                nodes.append(
                    ASTNode(
                        node_type=ASTNodeType.FUNCTION,
                        name=name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=_node_text(child, source_bytes),
                    )
                )

        elif child.type == "expression_statement":
            # Check for test function calls: describe("...", () => {...})
            _extract_ts_test_calls(child, source_bytes, nodes)

    return nodes


def _extract_ts_import_source(import_node, source_bytes: bytes) -> str:
    """Extract the module source from a TypeScript import statement."""
    for child in import_node.children:
        if child.type == "string":
            text = child.text.decode("utf-8", errors="replace") if child.text else ""
            # Strip quotes
            text = text.strip("\"'")
            return text
    return ""


def _extract_ts_class_methods(class_node, source_bytes: bytes, nodes: list[ASTNode]) -> None:
    """Extract methods from a TypeScript class."""
    for child in class_node.children:
        if child.type == "class_body":
            for member in child.children:
                if member.type == "method_definition":
                    name = _extract_identifier(member, source_bytes)
                    nodes.append(
                        ASTNode(
                            node_type=ASTNodeType.FUNCTION,
                            name=name,
                            start_line=member.start_point[0] + 1,
                            end_line=member.end_point[0] + 1,
                            source=_node_text(member, source_bytes),
                        )
                    )
                elif member.type == "public_field_definition":
                    # Handle: myMethod = () => {};
                    name = _extract_ts_arrow_name(member, source_bytes)
                    if name:
                        nodes.append(
                            ASTNode(
                                node_type=ASTNodeType.FUNCTION,
                                name=name,
                                start_line=member.start_point[0] + 1,
                                end_line=member.end_point[0] + 1,
                                source=_node_text(member, source_bytes),
                            )
                        )


def _extract_ts_arrow_name(node, source_bytes: bytes) -> str:
    """Extract the variable name from an arrow function assignment."""
    for child in node.children:
        if child.type == "variable_declarator":
            for sub in child.children:
                if sub.type == "identifier":
                    return sub.text.decode("utf-8", errors="replace") if sub.text else ""
    return ""


def _extract_ts_test_calls(node, source_bytes: bytes, nodes: list[ASTNode]) -> None:
    """Recursively search for test-related call expressions in TypeScript."""
    if node.type == "call_expression":
        name = _extract_ts_call_name(node)
        if name in _TS_TEST_FUNCTIONS:
            nodes.append(
                ASTNode(
                    node_type=ASTNodeType.TEST_BLOCK,
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    source=_node_text(node, source_bytes),
                )
            )
    for child in node.children:
        _extract_ts_test_calls(child, source_bytes, nodes)


def _extract_ts_call_name(call_node) -> str:
    """Extract the function name from a TypeScript call expression."""
    for child in call_node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace") if child.text else ""
    return ""


# ---------------------------------------------------------------------------
# Error collection
# ---------------------------------------------------------------------------


def _collect_errors(root, source_bytes: bytes) -> list[ParseError]:
    """Walk the tree and collect all error nodes."""
    errors: list[ParseError] = []

    def _walk(node) -> None:
        if node.has_error:
            # Try to find the specific error node
            _find_error_nodes(node, source_bytes, errors)
        for child in node.children:
            _walk(child)

    _walk(root)
    return errors


def _find_error_nodes(node, source_bytes: bytes, errors: list[ParseError]) -> None:
    """Recursively find ERROR and MISSING nodes."""
    if node.type == "ERROR" or node.is_missing:
        errors.append(
            ParseError(
                message=f"Syntax error at line {node.start_point[0] + 1}",
                line=node.start_point[0] + 1,
            )
        )
    for child in node.children:
        _find_error_nodes(child, source_bytes, errors)


# ---------------------------------------------------------------------------
# Main parsing functions
# ---------------------------------------------------------------------------

# Map from Language to extraction function
_EXTRACTION_DISPATCH = {
    Language.PYTHON: _extract_python_nodes,
    Language.R: _extract_r_nodes,
    Language.TYPESCRIPT: _extract_typescript_nodes,
}


def parse_source(source: str, language: Language) -> ParseResult:
    """Parse source code and extract structured AST nodes.

    Args:
        source: The source code to parse.
        language: The language of the source code.

    Returns:
        A ParseResult containing extracted nodes and any errors.
        Parsing errors are captured but do not prevent partial results.
    """
    source_bytes = source.encode("utf-8")
    parser = _get_parser(language)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # Extract AST nodes
    extract_fn = _EXTRACTION_DISPATCH[language]
    nodes = extract_fn(root, source_bytes)

    # Collect errors
    parse_errors = _collect_errors(root, source_bytes)

    return ParseResult(
        language=language,
        nodes=nodes,
        errors=parse_errors,
    )


def parse_file(file_path: Path | str) -> ParseResult:
    """Parse a source file and extract structured AST nodes.

    Auto-detects the language from the file extension or shebang.

    Args:
        file_path: Path to the source file.

    Returns:
        A ParseResult containing extracted nodes and any errors.

    Raises:
        FileParseError: If the file doesn't exist or language can't be detected.
    """
    path = Path(file_path)

    if not path.is_file():
        msg = f"File not found: {path}"
        raise FileParseError(msg)

    language = detect_language(path)

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read file: {path}: {exc}"
        raise FileParseError(msg) from exc

    result = parse_source(source, language)
    result.file_path = str(path.resolve())
    return result
