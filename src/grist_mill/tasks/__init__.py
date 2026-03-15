"""Task synthesis module.

Provides AST-based code analysis for multi-language source files
(R, Python, TypeScript/TSX). Used by the task synthesis pipeline to
extract function definitions, test blocks, and import statements.
"""

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

__all__ = [
    "ASTNode",
    "ASTNodeType",
    "FileParseError",
    "Language",
    "ParseResult",
    "detect_language",
    "parse_file",
    "parse_source",
]
