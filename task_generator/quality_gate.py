"""Task validation and quality checks."""

import logging
from dataclasses import dataclass, field
from typing import Any

from .ast_parser import RASTParser
from .models import TestingTask, TestPattern

logger = logging.getLogger(__name__)

# Minimum quality score for a task to be accepted
MIN_QUALITY_SCORE = 0.5


@dataclass
class QualityMetrics:
    """Quality metrics for a generated task."""

    has_clear_instruction: bool = False
    has_context: bool = False
    has_reference: bool = False
    has_function_code: bool = False
    code_compiles: bool = False
    has_valid_r_syntax: bool = False
    instruction_specificity: float = 0.0
    reference_quality: float = 0.0
    complexity_score: float = 0.0
    pattern_diversity_score: float = 0.0
    composite_score: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_clear_instruction": self.has_clear_instruction,
            "has_context": self.has_context,
            "has_reference": self.has_reference,
            "has_function_code": self.has_function_code,
            "code_compiles": self.code_compiles,
            "has_valid_r_syntax": self.has_valid_r_syntax,
            "instruction_specificity": self.instruction_specificity,
            "reference_quality": self.reference_quality,
            "complexity_score": self.complexity_score,
            "pattern_diversity_score": self.pattern_diversity_score,
            "composite_score": self.composite_score,
            "issues": self.issues,
        }

    @property
    def is_valid(self) -> bool:
        """Check if task passes quality checks."""
        return (
            self.has_clear_instruction
            and self.has_context
            and self.composite_score >= MIN_QUALITY_SCORE
            and len(self.issues) == 0
        )


class TaskQualityGate:
    """Validate generated tasks for quality."""

    def __init__(
        self,
        min_instruction_length: int = 50,
        min_context_length: int = 20,
        min_quality_score: float = MIN_QUALITY_SCORE,
        require_reference_test: bool = True,
        validate_r_syntax: bool = True,
    ) -> None:
        self.min_instruction_length = min_instruction_length
        self.min_context_length = min_context_length
        self.min_quality_score = min_quality_score
        self.require_reference_test = require_reference_test
        self.validate_r_syntax = validate_r_syntax
        self.parser = RASTParser() if validate_r_syntax else None

    def validate(self, task: TestingTask) -> QualityMetrics:
        """Validate a task and return quality metrics."""
        metrics = QualityMetrics()
        issues: list[str] = []

        # Check instruction quality
        metrics.has_clear_instruction = self._check_instruction(task.instruction, issues)

        # Check instruction specificity (does it reference specific functions?)
        metrics.instruction_specificity = self._score_instruction_specificity(task)

        # Check context
        metrics.has_context = self._check_context(task.context, issues)

        # Check for actual function code in context
        metrics.has_function_code = "```r" in task.context and "function(" in task.context

        # Check reference test
        metrics.has_reference = bool(task.reference_test)
        if self.require_reference_test and not metrics.has_reference:
            issues.append("Task has no reference test")

        # Score reference test quality
        metrics.reference_quality = self._score_reference_quality(task.reference_test)

        # Validate R syntax in reference test
        if self.validate_r_syntax and task.reference_test:
            metrics.has_valid_r_syntax = self._validate_code_syntax(task.reference_test, issues)
        elif not task.reference_test:
            metrics.has_valid_r_syntax = False

        # Calculate complexity score
        metrics.complexity_score = self._calculate_complexity(task)

        # Calculate pattern diversity score
        metrics.pattern_diversity_score = self._calculate_pattern_diversity(task.patterns)

        # Store issues
        metrics.issues = issues

        # Compute composite quality score
        metrics.composite_score = self._compute_composite_score(metrics)

        return metrics

    def _check_instruction(self, instruction: str, issues: list[str]) -> bool:
        """Check instruction quality."""
        if not instruction:
            issues.append("Instruction is empty")
            return False

        if len(instruction) < self.min_instruction_length:
            issues.append(f"Instruction too short ({len(instruction)} chars)")
            return False

        has_action = any(
            word in instruction.lower()
            for word in ["write", "add", "fix", "refactor", "create", "implement", "modify"]
        )
        if not has_action:
            issues.append("Instruction lacks clear action verb")

        return len([i for i in issues if "Instruction" in i]) == 0

    def _score_instruction_specificity(self, task: TestingTask) -> float:
        """Score how specific the instruction is (0.0 to 1.0)."""
        score = 0.0
        instruction = task.instruction.lower()

        # References a specific function name
        if task.function_name and task.function_name in task.instruction:
            score += 0.3

        # References the package name
        if task.source_package and task.source_package in task.instruction:
            score += 0.1

        # Contains backtick-quoted identifiers (e.g., `cli::cli_alert`)
        if "`" in task.instruction:
            score += 0.2

        # Contains specific test patterns (not just generic "write tests")
        specific_terms = [
            "expect_equal",
            "expect_error",
            "expect_snapshot",
            "expect_warning",
            "test_that",
            "describe",
            "local_mocked_bindings",
            "withr",
        ]
        if any(term in instruction for term in specific_terms):
            score += 0.2

        # Contains code blocks with actual code
        if "```r" in task.instruction:
            score += 0.2

        return min(score, 1.0)

    def _score_reference_quality(self, reference_test: str) -> float:
        """Score the quality of a reference test (0.0 to 1.0)."""
        if not reference_test:
            return 0.0

        score = 0.0

        # Has test_that blocks
        if "test_that(" in reference_test:
            score += 0.3

        # Has actual expect_* calls (not just placeholders)
        expect_calls = [
            "expect_equal",
            "expect_identical",
            "expect_error",
            "expect_snapshot",
            "expect_true",
            "expect_false",
            "expect_warning",
            "expect_message",
            "expect_type",
        ]
        found_expects = sum(1 for e in expect_calls if e in reference_test)
        score += min(found_expects * 0.1, 0.3)

        # Is not a placeholder/template
        placeholders = [
            "my_function(input)",
            "# Placeholder",
            "expect_true(TRUE)  # Placeholder",
            "# Add your",
            "# Test code here",
            "# Fixed test code would go here",
            "# Refactored test code would go here",
        ]
        if any(p in reference_test for p in placeholders):
            score -= 0.3

        # Has substantial code (not just comments)
        code_lines = [
            line
            for line in reference_test.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(code_lines) >= 5:
            score += 0.2
        elif len(code_lines) >= 3:
            score += 0.1

        # References real functions (not just my_function)
        if "my_function" not in reference_test and "function(" not in reference_test:
            score += 0.1

        return max(min(score, 1.0), 0.0)

    def _check_context(self, context: str, issues: list[str]) -> bool:
        """Check context quality."""
        if not context:
            issues.append("Context is empty")
            return False

        if len(context) < self.min_context_length:
            issues.append(f"Context too short ({len(context)} chars)")
            return False

        return True

    def _validate_code_syntax(self, code: str, issues: list[str]) -> bool:
        """Validate R syntax in code."""
        if not self.parser:
            return True

        try:
            tree = self.parser.parse_code(code)

            if self._has_error_nodes(tree.root_node):
                issues.append("Syntax error in reference test code")
                return False

        except Exception as e:
            issues.append(f"Failed to parse code: {e}")
            return False

        return True

    def _has_error_nodes(self, node: Any) -> bool:
        """Check if tree has any error nodes."""
        if node.type == "ERROR" or node.has_error:
            return True

        return any(self._has_error_nodes(child) for child in node.children)

    def _calculate_complexity(self, task: TestingTask) -> float:
        """Calculate complexity score for a task (0.0 to 1.0)."""
        score = 0.0

        difficulty_scores = {
            "easy": 0.2,
            "medium": 0.4,
            "hard": 0.7,
        }
        score += difficulty_scores.get(str(task.difficulty), 0.5)

        num_patterns = len(task.patterns)
        if num_patterns > 3:
            score += 0.1
        if num_patterns > 5:
            score += 0.1

        if task.context:
            context_lines = task.context.count("\n")
            if context_lines > 10:
                score += 0.1
            if context_lines > 20:
                score += 0.1

        return min(score, 1.0)

    def _calculate_pattern_diversity(self, patterns: list[TestPattern]) -> float:
        """Calculate pattern diversity score (0.0 to 1.0)."""
        if not patterns:
            return 0.0

        unique_patterns = set(patterns)
        total_patterns = len(list(TestPattern))
        type_diversity = len(unique_patterns) / total_patterns if total_patterns > 0 else 0

        score = type_diversity * 0.5

        if len(unique_patterns) > 2:
            score += 0.2
        if len(unique_patterns) > 4:
            score += 0.2
        if len(unique_patterns) > 6:
            score += 0.1

        return min(score, 1.0)

    def _compute_composite_score(self, metrics: QualityMetrics) -> float:
        """Compute a weighted composite quality score (0.0 to 1.0).

        Weights:
        - Instruction specificity: 0.25
        - Reference test quality: 0.30
        - Has function code in context: 0.20
        - Pattern diversity: 0.15
        - Valid R syntax: 0.10
        """
        score = 0.0

        score += metrics.instruction_specificity * 0.25
        score += metrics.reference_quality * 0.30
        score += (1.0 if metrics.has_function_code else 0.0) * 0.20
        score += metrics.pattern_diversity_score * 0.15
        score += (1.0 if metrics.has_valid_r_syntax else 0.0) * 0.10

        return round(score, 3)

    def filter_valid(self, tasks: list[TestingTask]) -> tuple[list[TestingTask], list[dict]]:
        """Filter tasks, returning only valid ones. Also sets quality_score on each task."""
        valid = []
        rejected = []

        for task in tasks:
            metrics = self.validate(task)

            # Set the quality score on the task
            task.quality_score = metrics.composite_score

            if metrics.is_valid:
                valid.append(task)
            else:
                rejected.append(
                    {
                        "task_id": task.task_id,
                        "quality_score": metrics.composite_score,
                        "reasons": metrics.issues,
                        "metrics": metrics.to_dict(),
                    }
                )

        return valid, rejected
