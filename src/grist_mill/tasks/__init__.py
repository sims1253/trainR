"""Task synthesis module.

Provides AST-based code analysis for multi-language source files
(R, Python, TypeScript/TSX), a mutation pipeline for generating
test tasks from source code modifications, and an end-to-end pipeline
that wires discovery, AST analysis, mutation, quality gates, and
dataset production.

Submodules:
- ast_parser: Tree-sitter based AST parsing and node extraction
- mutation: Test mutation pipeline for task synthesis
- pipeline: End-to-end task synthesis pipeline
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
from grist_mill.tasks.pipeline import (
    PipelineConfig,
    PipelineQualityGate,
    TaskPipeline,
    TaskPipelineResult,
    discover_source_files,
)

__all__ = [
    # AST parser
    "ASTNode",
    "ASTNodeType",
    "FileParseError",
    "Language",
    # Mutation pipeline
    "Mutation",
    "MutationApplyError",
    "MutationError",
    "MutationPipeline",
    "MutationPipelineConfig",
    "MutationResult",
    "MutationType",
    "MutatorRegistry",
    "ParseResult",
    # End-to-end pipeline
    "PipelineConfig",
    "PipelineQualityGate",
    "TaskPipeline",
    "TaskPipelineResult",
    "apply_mutation",
    "create_mutation_diff",
    "detect_language",
    "discover_source_files",
    "generate_task_description",
    "parse_file",
    "parse_source",
    "revert_mutation",
]
