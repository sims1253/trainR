"""Task synthesis module.

Provides AST-based code analysis for multi-language source files
(R, Python, TypeScript/TSX) and a mutation pipeline for generating
test tasks from source code modifications.

Submodules:
- ast_parser: Tree-sitter based AST parsing and node extraction
- mutation: Test mutation pipeline for task synthesis
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
from grist_mill.tasks.mutation import (
    Mutation,
    MutationApplyError,
    MutationError,
    MutationPipeline,
    MutationPipelineConfig,
    MutationResult,
    MutationType,
    MutatorRegistry,
    apply_mutation,
    create_mutation_diff,
    generate_task_description,
    revert_mutation,
)

__all__ = [
    # AST parser
    "ASTNode",
    "ASTNodeType",
    "FileParseError",
    "Language",
    "ParseResult",
    "detect_language",
    "parse_file",
    "parse_source",
    # Mutation pipeline
    "Mutation",
    "MutationApplyError",
    "MutationError",
    "MutationPipeline",
    "MutationPipelineConfig",
    "MutationResult",
    "MutationType",
    "MutatorRegistry",
    "apply_mutation",
    "create_mutation_diff",
    "generate_task_description",
    "revert_mutation",
]
