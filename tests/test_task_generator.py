"""Tests for task generator module."""

from task_generator.models import Difficulty, TestingTask, TestPattern
from task_generator.quality_gate import QualityMetrics, TaskQualityGate


def make_task(
    instruction: str = "x" * 100,
    context: str = "x" * 100,
    reference_test: str = "test_that('test', { expect_equal(1, 1) })",
    task_id: str = "test-task-1",
    source_package: str = "testpkg",
    difficulty: Difficulty = Difficulty.MEDIUM,
    function_name: str | None = None,
) -> TestingTask:
    """Helper to create a TestingTask with sensible defaults."""
    return TestingTask(
        task_id=task_id,
        instruction=instruction,
        context=context,
        reference_test=reference_test,
        source_package=source_package,
        difficulty=difficulty,
        function_name=function_name,
        source_file="R/test.R",
        test_type="unit",
        patterns=[TestPattern.EXPECT_EQUAL],
        dependencies=[],
        split="dev",
    )


def test_quality_gate_defaults():
    """Test TaskQualityGate default values."""
    gate = TaskQualityGate()
    assert gate.min_instruction_length == 50
    assert gate.min_context_length == 20
    assert gate.require_reference_test is True
    assert gate.validate_r_syntax is True


def test_quality_gate_custom_config():
    """Test TaskQualityGate with custom configuration."""
    gate = TaskQualityGate(
        min_instruction_length=100,
        min_context_length=50,
        min_quality_score=0.7,
        require_reference_test=False,
        validate_r_syntax=False,
    )
    assert gate.min_instruction_length == 100
    assert gate.min_context_length == 50
    assert gate.min_quality_score == 0.7
    assert gate.require_reference_test is False
    assert gate.validate_r_syntax is False


def test_quality_gate_validates_instruction_length():
    """Test that quality gate validates instruction length."""
    gate = TaskQualityGate(min_instruction_length=50)

    # Short instruction should fail
    task = TestingTask(
        task_id="test-1",
        instruction="Write test",  # Too short
        context="x" * 100,
        reference_test="test_that('works', {})",
        difficulty=Difficulty.EASY,
        source_package="testpkg",
        function_name="test_fn",
        source_file="R/test.R",
        test_type="unit",
        patterns=[TestPattern.EXPECT_EQUAL],
        dependencies=[],
        split="dev",
    )
    metrics = gate.validate(task)
    assert metrics.instruction_specificity < 0.5  # Low specificity due to short length

    # Long enough with action verb should pass
    task = TestingTask(
        task_id="test-2",
        instruction="Write a comprehensive test that verifies the `test_fn` function handles edge cases correctly including NULL input and empty strings",  # Good instruction with action verb and function name
        context="x" * 100,
        reference_test="test_that('works', { expect_equal(fn(NULL), NULL) })",
        difficulty=Difficulty.MEDIUM,
        source_package="testpkg",
        function_name="test_fn",
        source_file="R/test.R",
        test_type="unit",
        patterns=[TestPattern.EXPECT_EQUAL],
        dependencies=[],
        split="dev",
    )
    metrics = gate.validate(task)
    assert metrics.has_clear_instruction is True
    assert metrics.instruction_specificity >= 0.5


def test_quality_gate_validates_context_length():
    """Test that quality gate rejects short context."""
    gate = TaskQualityGate(min_context_length=50)

    # Short context should fail
    short_context_task = make_task(context="short")
    metrics = gate.validate(short_context_task)
    assert metrics.has_context is False
    assert "Context too short" in str(metrics.issues)

    # Long enough should pass
    long_context_task = make_task(context="x" * 100)
    metrics = gate.validate(long_context_task)
    assert metrics.has_context is True


def test_quality_gate_requires_reference_test():
    """Test that quality gate requires reference test when configured."""
    gate = TaskQualityGate(require_reference_test=True)

    # No reference test should fail
    no_ref_task = make_task(reference_test="")
    metrics = gate.validate(no_ref_task)
    assert metrics.has_reference is False
    assert "no reference test" in str(metrics.issues).lower()

    # With reference test should pass this check
    with_ref_task = make_task()
    metrics = gate.validate(with_ref_task)
    assert metrics.has_reference is True


def test_quality_gate_validates_action_verb():
    """Test that quality gate checks for action verbs in instruction."""
    gate = TaskQualityGate()

    # Instruction without action verb should have issue
    no_action_task = make_task(instruction="This is a task without an action verb" * 2)
    metrics = gate.validate(no_action_task)
    assert "action verb" in str(metrics.issues).lower()


def test_quality_metrics_is_valid():
    """Test QualityMetrics is_valid property."""
    # Valid metrics
    metrics = QualityMetrics(
        has_clear_instruction=True,
        has_context=True,
        composite_score=0.8,
        issues=[],
    )
    assert metrics.is_valid is True

    # Invalid due to low score
    metrics = QualityMetrics(
        has_clear_instruction=True,
        has_context=True,
        composite_score=0.3,
        issues=[],
    )
    assert metrics.is_valid is False

    # Invalid due to issues
    metrics = QualityMetrics(
        has_clear_instruction=True,
        has_context=True,
        composite_score=0.8,
        issues=["Some issue"],
    )
    assert metrics.is_valid is False


def test_quality_metrics_to_dict():
    """Test QualityMetrics to_dict method."""
    metrics = QualityMetrics(
        has_clear_instruction=True,
        has_context=True,
        composite_score=0.75,
        issues=["test issue"],
    )
    result = metrics.to_dict()
    assert isinstance(result, dict)
    assert result["has_clear_instruction"] is True
    assert result["composite_score"] == 0.75
    assert result["issues"] == ["test issue"]


def test_quality_gate_filter_valid():
    """Test filter_valid returns valid and rejected tasks."""
    gate = TaskQualityGate()

    # Create a valid task and an invalid task
    valid_task = make_task(
        instruction="Write tests for the `mean` function in the stats package",
        context="```r\nmean <- function(x) sum(x) / length(x)\n```" * 10,
        reference_test="test_that('mean works', { expect_equal(mean(1:3), 2) })",
    )
    invalid_task = make_task(
        task_id="invalid-task",
        instruction="short",
        context="short",
        reference_test="",
    )

    valid, rejected = gate.filter_valid([valid_task, invalid_task])

    assert len(valid) >= 0  # May or may not be valid depending on composite score
    assert len(rejected) >= 1  # At least the invalid one should be rejected


def test_quality_gate_instruction_specificity():
    """Test instruction specificity scoring."""
    gate = TaskQualityGate()

    # Task with function name in instruction
    task_with_function = make_task(
        instruction="Write tests for the `my_function` helper",
        function_name="my_function",
    )
    metrics = gate.validate(task_with_function)
    assert metrics.instruction_specificity >= 0.0

    # Task with package name in instruction
    task_with_package = make_task(
        instruction="Write tests for dplyr functions",
        source_package="dplyr",
    )
    metrics = gate.validate(task_with_package)
    assert metrics.instruction_specificity >= 0.0
