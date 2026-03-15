"""Test mutation pipeline for task synthesis.

Mutates source code to induce targeted test failures, generates natural-language
task descriptions from mutations, and produces varied mutation types (logic bugs,
missing imports, type errors, edge cases, wrong return values).

Mutations are revertable via clean diff/patch. Failed mutations (e.g., syntax
errors) are skipped with logging, not crashes.

Validates VAL-SYNTH-02, VAL-SYNTH-03, VAL-SYNTH-04, VAL-SYNTH-06.
"""

from __future__ import annotations

import difflib
import logging
import uuid
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from grist_mill.tasks.ast_parser import (
    ASTNode,
    ASTNodeType,
    Language,
    parse_source,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MutationError(Exception):
    """Base exception for mutation pipeline errors."""

    pass


class MutationApplyError(MutationError):
    """Raised when a mutation cannot be applied to source code."""

    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MutationType(str, Enum):
    """Categories of code mutations that can be applied."""

    LOGIC_BUG = "LOGIC_BUG"
    MISSING_IMPORT = "MISSING_IMPORT"
    TYPE_ERROR = "TYPE_ERROR"
    WRONG_RETURN_VALUE = "WRONG_RETURN_VALUE"
    EDGE_CASE = "EDGE_CASE"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Mutation(BaseModel):
    """A single code mutation.

    Attributes:
        mutation_type: Category of the mutation.
        file_path: Path to the file being mutated.
        original_code: The original source code snippet.
        mutated_code: The mutated version of the code snippet.
        start_line: 1-based start line of the mutation.
        end_line: 1-based end line of the mutation.
        description: Human-readable description of the mutation.
        diff: Unified diff between original and mutated code (computed lazily).
    """

    original_code: str = Field(
        ...,
        description="The original source code snippet.",
    )
    mutated_code: str = Field(
        default="",
        description="The mutated version of the code snippet.",
    )
    mutation_type: MutationType = Field(
        ...,
        description="Category of the mutation.",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        description="Path to the file being mutated.",
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="1-based start line of the mutation.",
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="1-based end line of the mutation.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the mutation.",
    )
    diff: str | None = Field(
        default=None,
        description="Unified diff between original and mutated code.",
    )


class MutationResult(BaseModel):
    """Result of a single mutation attempt.

    Attributes:
        success: Whether the mutation was applied and validated successfully.
        mutation: The mutation that was applied (None if failed).
        task_id: Generated task ID for this mutation.
        description: Natural-language task description for the agent.
        error: Error message if the mutation failed.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    success: bool = Field(
        ...,
        description="Whether the mutation was applied successfully.",
    )
    mutation: Mutation | None = Field(
        default=None,
        description="The applied mutation.",
    )
    task_id: str = Field(
        default="",
        description="Generated task ID for this mutation.",
    )
    description: str = Field(
        default="",
        description="Natural-language task description.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the mutation failed.",
    )


class MutationPipelineConfig(BaseModel):
    """Configuration for the mutation pipeline.

    Attributes:
        max_mutations_per_type: Maximum number of mutations to generate per type.
        target_functions: Specific function names to target (None = all functions).
        language: Target programming language for mutations.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    max_mutations_per_type: int = Field(
        default=3,
        ge=1,
        description="Maximum mutations per mutation type.",
    )
    target_functions: list[str] | None = Field(
        default=None,
        description="Specific function names to target. None means all functions.",
    )
    language: str = Field(
        default="python",
        min_length=1,
        description="Target programming language.",
    )


# ---------------------------------------------------------------------------
# Mutator function type
# ---------------------------------------------------------------------------

MutatorFn = Callable[[str, list[ASTNode], str, int], list[Mutation]]


# ---------------------------------------------------------------------------
# MutatorRegistry
# ---------------------------------------------------------------------------


class MutatorRegistry:
    """Registry for mutation strategy functions.

    Maps mutation types to their corresponding mutator functions.
    """

    def __init__(self) -> None:
        self._mutators: dict[MutationType, MutatorFn] = {}

    def register(
        self,
        mutation_type: MutationType,
        mutator_fn: MutatorFn,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a mutator function for a mutation type.

        Args:
            mutation_type: The type of mutation this function produces.
            mutator_fn: Function that takes (source, nodes, file_path, line_offset)
                       and returns a list of Mutation objects.
            overwrite: If True, allow overwriting an existing registration.

        Raises:
            ValueError: If the type is already registered and overwrite is False.
        """
        if mutation_type in self._mutators and not overwrite:
            msg = (
                f"Mutator for {mutation_type.value} is already registered. "
                f"Use overwrite=True to replace it."
            )
            raise ValueError(msg)
        self._mutators[mutation_type] = mutator_fn

    def get(self, mutation_type: MutationType) -> MutatorFn:
        """Get the mutator function for a mutation type.

        Args:
            mutation_type: The mutation type to look up.

        Raises:
            KeyError: If no mutator is registered for this type.
        """
        if mutation_type not in self._mutators:
            msg = f"No mutator registered for {mutation_type.value}"
            raise KeyError(msg)
        return self._mutators[mutation_type]

    def list_types(self) -> list[MutationType]:
        """List all registered mutation types."""
        return list(self._mutators.keys())


# ---------------------------------------------------------------------------
# Diff creation
# ---------------------------------------------------------------------------


def create_mutation_diff(original: str, mutated: str, file_path: str = "file") -> str:
    """Create a unified diff between original and mutated code.

    Args:
        original: The original source code.
        mutated: The mutated source code.
        file_path: File path to use in the diff header.

    Returns:
        Unified diff string, or empty string if no changes.
    """
    original_lines = original.splitlines(keepends=True)
    mutated_lines = mutated.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            original_lines,
            mutated_lines,
            fromfile=file_path,
            tofile=file_path,
        )
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Apply / Revert mutations
# ---------------------------------------------------------------------------


def apply_mutation(source: str | Path, mutation: Mutation) -> str:
    """Apply a mutation to a source code string or file.

    When a file path is given, the file is mutated in-place and the mutated
    content is returned as a string. When a string is given, the mutated
    string is returned.

    Args:
        source: The source code string or file path to mutate.
        mutation: The mutation to apply.

    Returns:
        The mutated source code as a string.

    Raises:
        MutationApplyError: If the original code is not found in the source.
    """
    if isinstance(source, Path):
        _apply_mutation_to_file(source, mutation)
        return source.read_text(encoding="utf-8")
    else:
        return _apply_mutation_to_string_impl(source, mutation)


def _apply_mutation_to_string_impl(source: str, mutation: Mutation) -> str:
    """Apply mutation to a string and return the result."""
    if mutation.original_code not in source:
        msg = (
            f"Cannot apply mutation: original code not found in source. "
            f"File: {mutation.file_path}, Lines {mutation.start_line}-{mutation.end_line}"
        )
        raise MutationApplyError(msg)
    # If mutated_code is empty (e.g., removing an import), remove the original code
    if not mutation.mutated_code.strip():
        return source.replace(mutation.original_code, "", 1)
    return source.replace(mutation.original_code, mutation.mutated_code, 1)


def _apply_mutation_to_file(file_path: Path, mutation: Mutation) -> None:
    """Apply mutation by replacing original_code with mutated_code in a file."""
    content = file_path.read_text(encoding="utf-8")
    if mutation.original_code not in content:
        msg = (
            f"Cannot apply mutation: original code not found in {file_path}. "
            f"Lines {mutation.start_line}-{mutation.end_line}"
        )
        raise MutationApplyError(msg)
    replacement = mutation.mutated_code if mutation.mutated_code.strip() else ""
    mutated_content = content.replace(mutation.original_code, replacement, 1)
    file_path.write_text(mutated_content, encoding="utf-8")


def revert_mutation(source: str | Path, mutation: Mutation) -> str:
    """Revert a mutation by replacing mutated_code back with original_code.

    For mutations where mutated_code is empty (e.g., removed imports),
    uses line-number based insertion to restore the original code.

    Args:
        source: The mutated source code string or file path.
        mutation: The mutation to revert.

    Returns:
        The reverted source code as a string.
    """
    if isinstance(source, Path):
        content = source.read_text(encoding="utf-8")
        reverted = _revert_string(content, mutation)
        source.write_text(reverted, encoding="utf-8")
        return reverted
    else:
        return _revert_string(source, mutation)


def _revert_string(source: str, mutation: Mutation) -> str:
    """Revert a mutation in a source string."""
    # If mutated_code is empty, we need to insert original_code at the
    # mutation's start line position.
    if not mutation.mutated_code.strip():
        return _insert_at_line(source, mutation.start_line, mutation.original_code)

    # Standard revert: replace mutated_code with original_code
    if mutation.mutated_code not in source:
        msg = (
            f"Cannot revert mutation: mutated code not found in source. File: {mutation.file_path}"
        )
        raise MutationApplyError(msg)
    return source.replace(mutation.mutated_code, mutation.original_code, 1)


def _insert_at_line(source: str, line_number: int, text: str) -> str:
    """Insert text at a specific line number in the source.

    Line numbers are 1-based. The text is inserted before the
    content at the specified line.

    Args:
        source: The source code string.
        line_number: 1-based line number to insert at.
        text: Text to insert.

    Returns:
        The source with text inserted.
    """
    lines = source.split("\n")
    # Convert to 0-based index
    insert_index = line_number - 1
    # Ensure we don't go out of bounds
    insert_index = max(0, min(insert_index, len(lines)))

    # Split the text to insert into lines
    text_lines = text.rstrip("\n").split("\n")

    # Insert the text lines
    for i, text_line in enumerate(text_lines):
        lines.insert(insert_index + i, text_line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# String-based apply (returns mutated string)
# ---------------------------------------------------------------------------


def apply_mutation_to_string(source: str, mutation: Mutation) -> str:
    """Apply a mutation to a source code string and return the result.

    This is the string-returning version for when you don't have a file.

    Args:
        source: The source code string.
        mutation: The mutation to apply.

    Returns:
        The mutated source code string.

    Raises:
        MutationApplyError: If the original code is not found.
    """
    if mutation.original_code not in source:
        msg = (
            f"Cannot apply mutation: original code not found in source. "
            f"File: {mutation.file_path}, Lines {mutation.start_line}-{mutation.end_line}"
        )
        raise MutationApplyError(msg)
    return source.replace(mutation.original_code, mutation.mutated_code, 1)


# ---------------------------------------------------------------------------
# Task description generation
# ---------------------------------------------------------------------------


def generate_task_description(mutation: Mutation, *, language: str = "python") -> str:
    """Generate a natural-language task description from a mutation.

    The description explains what was broken, which function is affected,
    and what the expected fix is.

    Args:
        mutation: The mutation that was applied.
        language: Programming language of the source.

    Returns:
        A natural-language task description.
    """
    mt = mutation.mutation_type
    fp = mutation.file_path
    desc = mutation.description

    match mt:
        case MutationType.LOGIC_BUG:
            return (
                f"In {fp}, the function has a logic bug: {desc} "
                f"Fix the logic error so that the function produces the correct result."
            )
        case MutationType.MISSING_IMPORT:
            return (
                f"In {fp}, an import statement was removed: {desc} "
                f"Restore the missing import to fix the import error."
            )
        case MutationType.TYPE_ERROR:
            return (
                f"In {fp}, a type error was introduced: {desc} "
                f"Fix the type error so that the function handles types correctly."
            )
        case MutationType.WRONG_RETURN_VALUE:
            return (
                f"In {fp}, the return value is incorrect: {desc} "
                f"Fix the return value to match the expected output."
            )
        case MutationType.EDGE_CASE:
            return (
                f"In {fp}, the function does not handle an edge case correctly: {desc} "
                f"Fix the edge case handling so the function works for all inputs."
            )
        case _:
            return (
                f"In {fp}, a bug was introduced: {desc} "
                f"Fix the issue so the function works correctly."
            )


# ---------------------------------------------------------------------------
# Mutator functions (Python-specific)
# ---------------------------------------------------------------------------


def _get_function_nodes(nodes: list[ASTNode]) -> list[ASTNode]:
    """Filter AST nodes to get only function definitions (not tests)."""
    return [n for n in nodes if n.node_type == ASTNodeType.FUNCTION]


def _get_import_nodes(nodes: list[ASTNode]) -> list[ASTNode]:
    """Filter AST nodes to get only import statements."""
    return [n for n in nodes if n.node_type == ASTNodeType.IMPORT]


def _mutate_logic_bug(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate logic bug mutations by flipping comparison and arithmetic operators."""
    mutations: list[Mutation] = []

    # Operator replacements for logic bugs
    operator_map = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
        ">": "<",
        "<": ">",
        ">=": "<=",
        "<=": ">=",
        "==": "!=",
        "!=": "==",
    }

    functions = _get_function_nodes(nodes)
    for func in functions:
        func_source = func.source
        lines = func_source.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip empty lines, comments, and function/class definitions
            if not stripped or stripped.startswith("#") or stripped.startswith("def "):
                continue
            if stripped.startswith("return "):
                for op, replacement in operator_map.items():
                    if op in stripped and len(stripped) > len("return "):
                        original_line = stripped
                        mutated_line = stripped.replace(op, replacement, 1)

                        original_code = line + "\n"
                        mutated_code = line.replace(stripped, mutated_line) + "\n"

                        desc = (
                            f"Operator '{op}' was changed to '{replacement}' "
                            f"in {func.name}(): {original_line}"
                        )
                        mutations.append(
                            Mutation(
                                mutation_type=MutationType.LOGIC_BUG,
                                file_path=file_path,
                                original_code=original_code,
                                mutated_code=mutated_code,
                                start_line=func.start_line + i,
                                end_line=func.start_line + i,
                                description=desc,
                            )
                        )

    return mutations


def _mutate_wrong_return_value(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate wrong return value mutations."""
    mutations: list[Mutation] = []

    functions = _get_function_nodes(nodes)
    for func in functions:
        func_source = func.source
        lines = func_source.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("return ")
                and stripped != "return"
                and any(op in stripped for op in ["+", "-", "*", "/"])
            ):
                original_code = line + "\n"
                # Replace the return expression with a constant
                mutated_line = line.replace(stripped, "return 0")
                mutated_code = mutated_line + "\n"

                desc = f"Return value in {func.name}() was replaced with 0 instead of: {stripped}"
                mutations.append(
                    Mutation(
                        mutation_type=MutationType.WRONG_RETURN_VALUE,
                        file_path=file_path,
                        original_code=original_code,
                        mutated_code=mutated_code,
                        start_line=func.start_line + i,
                        end_line=func.start_line + i,
                        description=desc,
                    )
                )
                # Also try returning None
                mutated_line_none = line.replace(stripped, "return None")
                mutated_code_none = mutated_line_none + "\n"
                desc_none = (
                    f"Return value in {func.name}() was replaced with None instead of: {stripped}"
                )
                mutations.append(
                    Mutation(
                        mutation_type=MutationType.WRONG_RETURN_VALUE,
                        file_path=file_path,
                        original_code=original_code,
                        mutated_code=mutated_code_none,
                        start_line=func.start_line + i,
                        end_line=func.start_line + i,
                        description=desc_none,
                    )
                )

    return mutations


def _mutate_missing_import(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate missing import mutations by removing import statements."""
    mutations: list[Mutation] = []

    imports = _get_import_nodes(nodes)
    for imp in imports:
        # Include trailing newline in original_code for clean removal
        original_code = imp.source
        if not original_code.endswith("\n"):
            original_code = original_code + "\n"
        mutated_code = ""  # Remove the import entirely

        desc = f"Import for '{imp.name}' was removed from {file_path}."
        mutations.append(
            Mutation(
                mutation_type=MutationType.MISSING_IMPORT,
                file_path=file_path,
                original_code=original_code,
                mutated_code=mutated_code,
                start_line=imp.start_line,
                end_line=imp.end_line,
                description=desc,
            )
        )

    return mutations


def _mutate_type_error(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate type error mutations by wrapping return values in str()."""
    mutations: list[Mutation] = []

    functions = _get_function_nodes(nodes)
    for func in functions:
        func_source = func.source
        lines = func_source.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("return ") and stripped != "return":
                expr = stripped[len("return ") :]
                # Wrap expression in str() to introduce a type error
                mutated_expr = f"return str({expr})"

                original_code = line + "\n"
                mutated_code = line.replace(stripped, mutated_expr) + "\n"

                desc = (
                    f"Return value in {func.name}() was wrapped in str(), "
                    f"changing type from numeric to string: {stripped}"
                )
                mutations.append(
                    Mutation(
                        mutation_type=MutationType.TYPE_ERROR,
                        file_path=file_path,
                        original_code=original_code,
                        mutated_code=mutated_code,
                        start_line=func.start_line + i,
                        end_line=func.start_line + i,
                        description=desc,
                    )
                )

    return mutations


def _mutate_edge_case(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate edge case mutations by changing comparison operators."""
    mutations: list[Mutation] = []

    edge_replacements = {
        ">": ">=",
        ">=": ">",
        "<": "<=",
        "<=": "<",
    }

    functions = _get_function_nodes(nodes)
    for func in functions:
        func_source = func.source
        lines = func_source.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("def "):
                continue

            for op, replacement in edge_replacements.items():
                if op in stripped:
                    original_code = line + "\n"
                    mutated_code = line.replace(op, replacement, 1) + "\n"

                    desc = (
                        f"Comparison operator '{op}' was changed to '{replacement}' "
                        f"in {func.name}(), affecting boundary/edge case handling."
                    )
                    mutations.append(
                        Mutation(
                            mutation_type=MutationType.EDGE_CASE,
                            file_path=file_path,
                            original_code=original_code,
                            mutated_code=mutated_code,
                            start_line=func.start_line + i,
                            end_line=func.start_line + i,
                            description=desc,
                        )
                    )

    return mutations


# ---------------------------------------------------------------------------
# R-specific mutator functions
# ---------------------------------------------------------------------------


def _mutate_logic_bug_r(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate logic bug mutations for R code."""
    mutations: list[Mutation] = []

    operator_map = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
    }

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "<- function" in stripped:
                continue
            for op, replacement in operator_map.items():
                if op in stripped:
                    original_code = line + "\n"
                    mutated_code = line.replace(op, replacement, 1) + "\n"

                    desc = (
                        f"Operator '{op}' was changed to '{replacement}' "
                        f"in {func.name}(): {stripped}"
                    )
                    mutations.append(
                        Mutation(
                            mutation_type=MutationType.LOGIC_BUG,
                            file_path=file_path,
                            original_code=original_code,
                            mutated_code=mutated_code,
                            start_line=func.start_line + i,
                            end_line=func.start_line + i,
                            description=desc,
                        )
                    )

    return mutations


def _mutate_missing_import_r(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate missing import mutations for R code."""
    mutations: list[Mutation] = []

    imports = _get_import_nodes(nodes)
    for imp in imports:
        original_code = imp.source
        if not original_code.endswith("\n"):
            original_code = original_code + "\n"
        mutated_code = ""

        desc = f"Import for '{imp.name}' was removed from {file_path}."
        mutations.append(
            Mutation(
                mutation_type=MutationType.MISSING_IMPORT,
                file_path=file_path,
                original_code=original_code,
                mutated_code=mutated_code,
                start_line=imp.start_line,
                end_line=imp.end_line,
                description=desc,
            )
        )

    return mutations


def _mutate_wrong_return_value_r(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate wrong return value mutations for R code."""
    mutations: list[Mutation] = []

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "<- function" in stripped:
                continue
            # Look for expression lines (last expression in R function is the return value)
            if any(op in stripped for op in ["+", "-", "*", "/"]):
                original_code = line + "\n"
                mutated_code = line.replace(stripped, "  0") + "\n"

                desc = f"Return value in {func.name}() was replaced with 0 instead of: {stripped}"
                mutations.append(
                    Mutation(
                        mutation_type=MutationType.WRONG_RETURN_VALUE,
                        file_path=file_path,
                        original_code=original_code,
                        mutated_code=mutated_code,
                        start_line=func.start_line + i,
                        end_line=func.start_line + i,
                        description=desc,
                    )
                )

    return mutations


def _mutate_edge_case_r(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate edge case mutations for R code."""
    mutations: list[Mutation] = []

    edge_replacements = {
        ">": ">=",
        ">=": ">",
        "<": "<=",
        "<=": "<",
    }

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "<- function" in stripped:
                continue
            for op, replacement in edge_replacements.items():
                if op in stripped:
                    original_code = line + "\n"
                    mutated_code = line.replace(op, replacement, 1) + "\n"

                    desc = (
                        f"Comparison operator '{op}' was changed to '{replacement}' "
                        f"in {func.name}(), affecting boundary/edge case handling."
                    )
                    mutations.append(
                        Mutation(
                            mutation_type=MutationType.EDGE_CASE,
                            file_path=file_path,
                            original_code=original_code,
                            mutated_code=mutated_code,
                            start_line=func.start_line + i,
                            end_line=func.start_line + i,
                            description=desc,
                        )
                    )

    return mutations


# ---------------------------------------------------------------------------
# TypeScript-specific mutator functions
# ---------------------------------------------------------------------------


def _mutate_logic_bug_ts(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate logic bug mutations for TypeScript code."""
    mutations: list[Mutation] = []

    operator_map = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
        "===": "!==",
        "!==": "===",
        ">": "<",
        "<": ">",
    }

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("function "):
                continue
            if stripped.startswith("return "):
                for op, replacement in operator_map.items():
                    if op in stripped:
                        original_code = line + "\n"
                        mutated_code = line.replace(op, replacement, 1) + "\n"

                        desc = (
                            f"Operator '{op}' was changed to '{replacement}' "
                            f"in {func.name}(): {stripped}"
                        )
                        mutations.append(
                            Mutation(
                                mutation_type=MutationType.LOGIC_BUG,
                                file_path=file_path,
                                original_code=original_code,
                                mutated_code=mutated_code,
                                start_line=func.start_line + i,
                                end_line=func.start_line + i,
                                description=desc,
                            )
                        )

    return mutations


def _mutate_missing_import_ts(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate missing import mutations for TypeScript code."""
    mutations: list[Mutation] = []

    for imp in nodes:
        if imp.node_type != ASTNodeType.IMPORT:
            continue
        original_code = imp.source
        if not original_code.endswith("\n"):
            original_code = original_code + "\n"
        mutated_code = ""

        desc = f"Import for '{imp.name}' was removed from {file_path}."
        mutations.append(
            Mutation(
                mutation_type=MutationType.MISSING_IMPORT,
                file_path=file_path,
                original_code=original_code,
                mutated_code=mutated_code,
                start_line=imp.start_line,
                end_line=imp.end_line,
                description=desc,
            )
        )

    return mutations


def _mutate_type_error_ts(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate type error mutations for TypeScript code."""
    mutations: list[Mutation] = []

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("return ") and stripped != "return":
                expr = stripped[len("return ") :]
                # Remove trailing semicolons for the wrapping
                expr_clean = expr.rstrip(";")
                mutated_expr = f"return String({expr_clean})" + (";" if expr.endswith(";") else "")

                original_code = line + "\n"
                mutated_code = line.replace(stripped, mutated_expr) + "\n"

                desc = (
                    f"Return value in {func.name}() was wrapped in String(), "
                    f"changing type to string: {stripped}"
                )
                mutations.append(
                    Mutation(
                        mutation_type=MutationType.TYPE_ERROR,
                        file_path=file_path,
                        original_code=original_code,
                        mutated_code=mutated_code,
                        start_line=func.start_line + i,
                        end_line=func.start_line + i,
                        description=desc,
                    )
                )

    return mutations


def _mutate_wrong_return_value_ts(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate wrong return value mutations for TypeScript code."""
    mutations: list[Mutation] = []

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("return ") and stripped != "return":
                expr = stripped[len("return ") :]
                has_semicolon = expr.endswith(";")
                mutated_line = "return 0" + (";" if has_semicolon else "")

                original_code = line + "\n"
                mutated_code = line.replace(stripped, mutated_line) + "\n"

                desc = f"Return value in {func.name}() was replaced with 0 instead of: {stripped}"
                mutations.append(
                    Mutation(
                        mutation_type=MutationType.WRONG_RETURN_VALUE,
                        file_path=file_path,
                        original_code=original_code,
                        mutated_code=mutated_code,
                        start_line=func.start_line + i,
                        end_line=func.start_line + i,
                        description=desc,
                    )
                )

    return mutations


def _mutate_edge_case_ts(
    source: str,
    nodes: list[ASTNode],
    file_path: str,
    line_offset: int = 0,
) -> list[Mutation]:
    """Generate edge case mutations for TypeScript code."""
    mutations: list[Mutation] = []

    edge_replacements = {
        "===": "==",
        "!==": "!=",
        ">": ">=",
        ">=": ">",
        "<": "<=",
        "<=": "<",
    }

    functions = _get_function_nodes(nodes)
    for func in functions:
        lines = func.source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("function "):
                continue
            for op, replacement in edge_replacements.items():
                if op in stripped:
                    original_code = line + "\n"
                    mutated_code = line.replace(op, replacement, 1) + "\n"

                    desc = (
                        f"Comparison operator '{op}' was changed to '{replacement}' "
                        f"in {func.name}(), affecting boundary/edge case handling."
                    )
                    mutations.append(
                        Mutation(
                            mutation_type=MutationType.EDGE_CASE,
                            file_path=file_path,
                            original_code=original_code,
                            mutated_code=mutated_code,
                            start_line=func.start_line + i,
                            end_line=func.start_line + i,
                            description=desc,
                        )
                    )

    return mutations


# ---------------------------------------------------------------------------
# Mutation validation
# ---------------------------------------------------------------------------


def _validate_mutation(source: str, mutation: Mutation) -> bool:
    """Validate that a mutation can be applied to the source code.

    Checks that the original code exists in the source and that
    the mutation produces a syntactically reasonable change.

    Args:
        source: The source code string.
        mutation: The mutation to validate.

    Returns:
        True if the mutation is valid, False otherwise.
    """
    # Check original code exists in source
    if mutation.original_code not in source:
        logger.debug(
            "Mutation validation failed: original code not found in source. File: %s, Lines %d-%d",
            mutation.file_path,
            mutation.start_line,
            mutation.end_line,
        )
        return False

    # Check that the mutation actually changes something
    if mutation.original_code == mutation.mutated_code:
        logger.debug("Mutation validation failed: no change in mutation.")
        return False

    # For non-empty mutations, verify the mutated code doesn't produce
    # obvious syntax issues (basic check: balanced parens in new code)
    if mutation.mutated_code.strip():
        open_parens = mutation.mutated_code.count("(")
        close_parens = mutation.mutated_code.count(")")
        if abs(open_parens - close_parens) > 2:
            logger.debug(
                "Mutation validation failed: unbalanced parentheses. File: %s, Lines %d-%d",
                mutation.file_path,
                mutation.start_line,
                mutation.end_line,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# MutationPipeline
# ---------------------------------------------------------------------------


def _build_default_registry(language: str) -> MutatorRegistry:
    """Build a default mutator registry with language-appropriate mutators."""
    registry = MutatorRegistry()

    if language == "python":
        registry.register(MutationType.LOGIC_BUG, _mutate_logic_bug)
        registry.register(MutationType.MISSING_IMPORT, _mutate_missing_import)
        registry.register(MutationType.TYPE_ERROR, _mutate_type_error)
        registry.register(MutationType.WRONG_RETURN_VALUE, _mutate_wrong_return_value)
        registry.register(MutationType.EDGE_CASE, _mutate_edge_case)
    elif language == "r":
        registry.register(MutationType.LOGIC_BUG, _mutate_logic_bug_r)
        registry.register(MutationType.MISSING_IMPORT, _mutate_missing_import_r)
        registry.register(MutationType.WRONG_RETURN_VALUE, _mutate_wrong_return_value_r)
        registry.register(MutationType.EDGE_CASE, _mutate_edge_case_r)
    elif language == "typescript":
        registry.register(MutationType.LOGIC_BUG, _mutate_logic_bug_ts)
        registry.register(MutationType.MISSING_IMPORT, _mutate_missing_import_ts)
        registry.register(MutationType.TYPE_ERROR, _mutate_type_error_ts)
        registry.register(MutationType.WRONG_RETURN_VALUE, _mutate_wrong_return_value_ts)
        registry.register(MutationType.EDGE_CASE, _mutate_edge_case_ts)

    return registry


def _map_language_str(language: str) -> Language:
    """Map a string to a Language enum."""
    lang_lower = language.lower().strip()
    mapping = {
        "python": Language.PYTHON,
        "r": Language.R,
        "typescript": Language.TYPESCRIPT,
        "ts": Language.TYPESCRIPT,
    }
    if lang_lower not in mapping:
        msg = f"Unsupported language for mutation: {language}. Supported: {list(mapping.keys())}"
        raise MutationError(msg)
    return mapping[lang_lower]


class MutationPipeline:
    """Orchestrates the mutation pipeline: AST analysis → mutation → description → Task.

    Takes source code, parses it with the AST parser, generates mutations
    using registered mutators, validates them, generates task descriptions,
    and produces MutationResult objects.

    Handles failed mutations gracefully by logging and continuing.
    """

    def __init__(
        self,
        config: MutationPipelineConfig | None = None,
        registry: MutatorRegistry | None = None,
    ) -> None:
        """Initialize the mutation pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
            registry: Custom mutator registry. Uses language defaults if None.
        """
        self._config = config or MutationPipelineConfig()
        self._registry = registry

    def _get_registry(self, language: str) -> MutatorRegistry:
        """Get the mutator registry, building defaults if needed."""
        if self._registry is not None:
            return self._registry
        return _build_default_registry(language)

    def generate(
        self,
        source: str,
        *,
        language: str = "python",
        file_path: str = "source.py",
    ) -> list[MutationResult]:
        """Generate mutations from source code.

        Args:
            source: The source code to mutate.
            language: Programming language of the source.
            file_path: File path to use in generated mutations.

        Returns:
            List of MutationResult objects, including both successful and failed mutations.
        """
        # Parse the source with the AST parser
        try:
            lang_enum = _map_language_str(language)
            parse_result = parse_source(source, lang_enum)
        except Exception as exc:
            logger.warning(
                "Failed to parse source for mutation. File: %s, Error: %s",
                file_path,
                exc,
            )
            return []

        if not parse_result.nodes:
            logger.info("No AST nodes extracted from source: %s", file_path)
            return []

        # Filter to target functions if specified
        nodes = parse_result.nodes
        if self._config.target_functions:
            target_set = set(self._config.target_functions)
            nodes = [n for n in nodes if n.name in target_set or n.node_type == ASTNodeType.IMPORT]

        # Get the mutator registry for this language
        try:
            registry = self._get_registry(language)
        except Exception as exc:
            logger.warning("Failed to get mutator registry: %s", exc)
            return []

        # Generate mutations from each mutator
        all_results: list[MutationResult] = []
        mutation_count = 0

        for mutation_type in registry.list_types():
            try:
                mutator_fn = registry.get(mutation_type)
                mutations = mutator_fn(source, nodes, file_path, 0)
            except Exception as exc:
                logger.warning(
                    "Mutator for %s raised an error. Skipping. Error: %s",
                    mutation_type.value,
                    exc,
                )
                all_results.append(
                    MutationResult(
                        success=False,
                        mutation=None,
                        task_id="",
                        description="",
                        error=f"Mutator {mutation_type.value} failed: {exc}",
                    )
                )
                continue

            type_count = 0
            for mutation in mutations:
                if type_count >= self._config.max_mutations_per_type:
                    break

                # Validate the mutation
                if not _validate_mutation(source, mutation):
                    logger.warning(
                        "Skipping invalid mutation. Type: %s, File: %s, Lines %d-%d",
                        mutation.mutation_type.value,
                        mutation.file_path,
                        mutation.start_line,
                        mutation.end_line,
                    )
                    all_results.append(
                        MutationResult(
                            success=False,
                            mutation=mutation,
                            task_id="",
                            description="",
                            error=(
                                f"Mutation validation failed: original code not found "
                                f"in source. File: {mutation.file_path}, "
                                f"Lines {mutation.start_line}-{mutation.end_line}"
                            ),
                        )
                    )
                    continue

                # Compute diff
                mutation.diff = create_mutation_diff(
                    mutation.original_code, mutation.mutated_code, mutation.file_path
                )

                # Generate task description
                description = generate_task_description(mutation, language=language)

                # Generate task ID
                task_id = f"mut-{uuid.uuid4().hex[:8]}"

                all_results.append(
                    MutationResult(
                        success=True,
                        mutation=mutation,
                        task_id=task_id,
                        description=description,
                    )
                )
                type_count += 1
                mutation_count += 1

        logger.info(
            "Mutation pipeline completed. File: %s, Generated: %d, Successful: %d",
            file_path,
            len(all_results),
            mutation_count,
        )

        return all_results
