"""Task Generator - Generate testing tasks from R package source code."""

from .ast_parser import RASTParser
from .generator import TaskGenerator
from .mined_task import (
    Difficulty as MinedDifficulty,
    MinedTask,
    MinedTaskSchema,
    MiningConfig,
    MiningStats,
    PRAnalysisInput,
    TaskType,
)
from .models import Difficulty, ExtractedPattern, TestingTask, TestPattern
from .pattern_extractor import PatternExtractor
from .quality_gate import QualityMetrics, TaskQualityGate
from .templates import TaskTemplate, TemplateRegistry

__all__ = [
    # From models.py
    "Difficulty",
    "ExtractedPattern",
    "TestPattern",
    "TestingTask",
    # From mined_task.py
    "MinedDifficulty",
    "MinedTask",
    "MinedTaskSchema",
    "MiningConfig",
    "MiningStats",
    "PRAnalysisInput",
    "TaskType",
    # Other modules
    "PatternExtractor",
    "QualityMetrics",
    "RASTParser",
    "TaskGenerator",
    "TaskQualityGate",
    "TaskTemplate",
    "TemplateRegistry",
]
