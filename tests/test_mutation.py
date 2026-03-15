"""Tests for the test mutation pipeline.

Covers:
- VAL-SYNTH-02: Test mutation induces controlled failures, revertable via diff/patch
- VAL-SYNTH-03: Mutation pipeline generates task descriptions automatically
- VAL-SYNTH-04: Mutation diversity produces varied tasks (3+ categories across 10+ tasks)
- VAL-SYNTH-06: Failed mutations are recovered gracefully (skip with logging, not crashes)
"""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent

import pytest

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

# ===========================================================================
# Helpers
# ===========================================================================


def _source(code: str) -> str:
    """Return dedented source code as a string."""
    return dedent(code).lstrip("\n")


SAMPLE_PYTHON_SOURCE = _source("""
    import os
    import math

    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    def is_positive(n):
        return n > 0

    def divide(a, b):
        return a / b

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    def test_add():
        assert add(1, 2) == 3
        assert add(-1, 1) == 0

    def test_multiply():
        assert multiply(2, 3) == 6
        assert multiply(0, 5) == 0

    def test_is_positive():
        assert is_positive(5) is True
        assert is_positive(-1) is False
        assert is_positive(0) is False
""")

SAMPLE_R_SOURCE = _source("""
    add <- function(a, b) {
      a + b
    }

    multiply <- function(a, b) {
      a * b
    }

    library(dplyr)

    test_that("add works", {
      expect_equal(add(1, 2), 3)
    })
""")


# ===========================================================================
# MutationType enum
# ===========================================================================


class TestMutationType:
    """Tests for the MutationType enum."""

    def test_all_categories_exist(self) -> None:
        expected = {
            "LOGIC_BUG",
            "MISSING_IMPORT",
            "TYPE_ERROR",
            "WRONG_RETURN_VALUE",
            "EDGE_CASE",
        }
        actual = {m.value for m in MutationType}
        assert expected.issubset(actual)

    def test_at_least_three_categories(self) -> None:
        assert len(MutationType) >= 3


# ===========================================================================
# Mutation model
# ===========================================================================


class TestMutationModel:
    """Tests for the Mutation Pydantic model."""

    def test_valid_construction(self) -> None:
        m = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calculator.py",
            original_code="return a + b",
            mutated_code="return a - b",
            start_line=3,
            end_line=3,
            description="Changed addition to subtraction in add function.",
        )
        assert m.mutation_type == MutationType.LOGIC_BUG
        assert m.file_path == "calculator.py"
        assert m.mutated_code == "return a - b"

    def test_serialization_round_trip(self) -> None:
        m = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="return a + b",
            mutated_code="return a - b",
            start_line=5,
            end_line=5,
            description="Broken add.",
        )
        data = m.model_dump()
        m2 = Mutation.model_validate(data)
        assert m2.mutation_type == m.mutation_type
        assert m2.file_path == m.file_path

    def test_optional_diff_field(self) -> None:
        m = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="x",
            mutated_code="y",
            start_line=1,
            end_line=1,
            description="test",
        )
        assert m.diff is None


# ===========================================================================
# Diff creation
# ===========================================================================


class TestCreateMutationDiff:
    """Tests for creating a clean diff between original and mutated code."""

    def test_creates_diff_with_changes(self) -> None:
        original = "return a + b\n"
        mutated = "return a - b\n"
        diff = create_mutation_diff(original, mutated, "calc.py")
        assert "--- calc.py" in diff or "calc.py" in diff
        assert "-" in diff
        assert "+" in diff

    def test_empty_diff_when_no_change(self) -> None:
        code = "return a + b\n"
        diff = create_mutation_diff(code, code, "calc.py")
        assert diff == ""

    def test_diff_includes_context(self) -> None:
        original = "def foo():\n    return 42\n"
        mutated = "def foo():\n    return 0\n"
        diff = create_mutation_diff(original, mutated, "foo.py")
        # Should contain the file path
        assert "foo.py" in diff


# ===========================================================================
# apply_mutation / revert_mutation
# ===========================================================================


class TestApplyAndRevertMutation:
    """Tests for applying and reverting mutations on source code."""

    def test_apply_mutation_changes_source(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="return a + b\n",
            mutated_code="return a - b\n",
            start_line=3,
            end_line=3,
            description="Broke add.",
        )
        source = "def add(a, b):\n    return a + b\n"
        result = apply_mutation(source, mutation)
        assert "return a - b" in result
        assert "return a + b" not in result

    def test_revert_mutation_restores_original(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="return a + b\n",
            mutated_code="return a - b\n",
            start_line=3,
            end_line=3,
            description="Broke add.",
        )
        source = "def add(a, b):\n    return a + b\n"
        mutated = apply_mutation(source, mutation)
        reverted = revert_mutation(mutated, mutation)
        assert reverted == source

    def test_apply_mutation_on_file(self, tmp_path: Path) -> None:
        src_file = tmp_path / "calc.py"
        src_file.write_text("def add(a, b):\n    return a + b\n")
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path=str(src_file),
            original_code="return a + b\n",
            mutated_code="return a - b\n",
            start_line=2,
            end_line=2,
            description="Broke add.",
        )
        apply_mutation(src_file, mutation)
        content = src_file.read_text()
        assert "return a - b" in content

    def test_revert_mutation_on_file(self, tmp_path: Path) -> None:
        src_file = tmp_path / "calc.py"
        original = "def add(a, b):\n    return a + b\n"
        src_file.write_text(original)
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path=str(src_file),
            original_code="return a + b\n",
            mutated_code="return a - b\n",
            start_line=2,
            end_line=2,
            description="Broke add.",
        )
        apply_mutation(src_file, mutation)
        revert_mutation(src_file, mutation)
        assert src_file.read_text() == original

    def test_apply_raises_if_original_not_found(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="return a + b\n",
            mutated_code="return a - b\n",
            start_line=3,
            end_line=3,
            description="Broke add.",
        )
        source = "def add(a, b):\n    return a * b\n"
        with pytest.raises(MutationApplyError, match="original code not found"):
            apply_mutation(source, mutation)


# ===========================================================================
# Task description generation
# ===========================================================================


class TestGenerateTaskDescription:
    """Tests for auto-generating natural-language task descriptions from mutations.

    Validates VAL-SYNTH-03.
    """

    def test_description_references_mutation(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="return a + b",
            mutated_code="return a - b",
            start_line=3,
            end_line=3,
            description="Changed + to - in add function.",
        )
        desc = generate_task_description(mutation, language="python")
        assert len(desc) > 0
        # Description should reference the mutation
        assert any(word in desc.lower() for word in ["add", "subtraction", "minus", "-"])

    def test_description_references_missing_import(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.MISSING_IMPORT,
            file_path="utils.py",
            original_code="import os\nimport math\n",
            mutated_code="import os\n",
            start_line=1,
            end_line=2,
            description="Removed math import.",
        )
        desc = generate_task_description(mutation, language="python")
        assert "math" in desc.lower()

    def test_description_references_type_error(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.TYPE_ERROR,
            file_path="calc.py",
            original_code="return a + b",
            mutated_code="return a + str(b)",
            start_line=3,
            end_line=3,
            description="Introduced type error in add function.",
        )
        desc = generate_task_description(mutation, language="python")
        assert len(desc) > 0
        # Should mention type-related terms
        assert any(word in desc.lower() for word in ["type", "string", "str"])

    def test_description_mentions_expected_fix(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.WRONG_RETURN_VALUE,
            file_path="calc.py",
            original_code="return a * b",
            mutated_code="return a + b",
            start_line=5,
            end_line=5,
            description="Changed multiply to add in multiply function.",
        )
        desc = generate_task_description(mutation, language="python")
        # Should mention what needs to be fixed
        assert len(desc) > 20  # Should be a meaningful description


# ===========================================================================
# MutatorRegistry
# ===========================================================================


class TestMutatorRegistry:
    """Tests for the mutator registry that manages mutation strategies."""

    def test_register_and_get_mutator(self) -> None:
        registry = MutatorRegistry()

        def dummy_mutator(source: str, nodes: list) -> list[Mutation]:
            return []

        registry.register(MutationType.LOGIC_BUG, dummy_mutator)
        fn = registry.get(MutationType.LOGIC_BUG)
        assert fn is dummy_mutator

    def test_register_duplicate_raises(self) -> None:
        registry = MutatorRegistry()

        def mutator1(source: str, nodes: list) -> list[Mutation]:
            return []

        def mutator2(source: str, nodes: list) -> list[Mutation]:
            return []

        registry.register(MutationType.LOGIC_BUG, mutator1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MutationType.LOGIC_BUG, mutator2)

    def test_register_duplicate_with_overwrite(self) -> None:
        registry = MutatorRegistry()

        def mutator1(source: str, nodes: list) -> list[Mutation]:
            return []

        def mutator2(source: str, nodes: list) -> list[Mutation]:
            return []

        registry.register(MutationType.LOGIC_BUG, mutator1)
        registry.register(MutationType.LOGIC_BUG, mutator2, overwrite=True)
        assert registry.get(MutationType.LOGIC_BUG) is mutator2

    def test_list_types(self) -> None:
        registry = MutatorRegistry()

        def mutator(source: str, nodes: list) -> list[Mutation]:
            return []

        registry.register(MutationType.LOGIC_BUG, mutator)
        registry.register(MutationType.MISSING_IMPORT, mutator)
        types = registry.list_types()
        assert MutationType.LOGIC_BUG in types
        assert MutationType.MISSING_IMPORT in types


# ===========================================================================
# MutationResult model
# ===========================================================================


class TestMutationResultModel:
    """Tests for the MutationResult model."""

    def test_successful_result(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="x",
            mutated_code="y",
            start_line=1,
            end_line=1,
            description="test",
        )
        result = MutationResult(
            success=True,
            mutation=mutation,
            task_id="mut-001",
            description="Fix the add function.",
        )
        assert result.success is True
        assert result.task_id == "mut-001"

    def test_failed_result(self) -> None:
        result = MutationResult(
            success=False,
            mutation=None,
            task_id="",
            description="",
            error="Failed to apply mutation: syntax error",
        )
        assert result.success is False
        assert result.error is not None

    def test_serialization_round_trip(self) -> None:
        mutation = Mutation(
            mutation_type=MutationType.LOGIC_BUG,
            file_path="calc.py",
            original_code="x",
            mutated_code="y",
            start_line=1,
            end_line=1,
            description="test",
        )
        result = MutationResult(
            success=True,
            mutation=mutation,
            task_id="mut-001",
            description="Fix the add function.",
        )
        data = result.model_dump()
        result2 = MutationResult.model_validate(data)
        assert result2.success == result.success
        assert result2.task_id == result.task_id


# ===========================================================================
# MutationPipeline - Core Pipeline
# ===========================================================================


class TestMutationPipeline:
    """Tests for the full mutation pipeline.

    Validates VAL-SYNTH-02, VAL-SYNTH-04, VAL-SYNTH-06.
    """

    def _make_pipeline(self, **kwargs) -> MutationPipeline:
        config = MutationPipelineConfig(**kwargs)
        return MutationPipeline(config)

    def test_pipeline_produces_mutations_from_python_source(self) -> None:
        pipeline = self._make_pipeline()
        results = pipeline.generate(SAMPLE_PYTHON_SOURCE, language="python", file_path="calc.py")
        assert len(results) > 0
        # At least some should succeed
        successful = [r for r in results if r.success]
        assert len(successful) > 0

    def test_pipeline_generates_task_descriptions(self) -> None:
        """VAL-SYNTH-03: Descriptions reference the specific mutation."""
        pipeline = self._make_pipeline()
        results = pipeline.generate(SAMPLE_PYTHON_SOURCE, language="python", file_path="calc.py")
        successful = [r for r in results if r.success]
        for r in successful:
            assert len(r.description) > 20, f"Description too short: {r.description!r}"
            assert r.task_id, "Task ID should be set"

    def test_mutations_are_revertable(self, tmp_path: Path) -> None:
        """VAL-SYNTH-02: Mutations are revertable via clean diff."""
        src_file = tmp_path / "calc.py"
        src_file.write_text(SAMPLE_PYTHON_SOURCE)

        pipeline = self._make_pipeline()
        results = pipeline.generate(SAMPLE_PYTHON_SOURCE, language="python", file_path="calc.py")
        successful = [r for r in results if r.success]

        for r in successful:
            assert r.mutation is not None
            # Reset file to original state
            src_file.write_text(SAMPLE_PYTHON_SOURCE)
            # Apply mutation
            apply_mutation(src_file, r.mutation)
            # Verify mutation was applied
            assert src_file.read_text() != SAMPLE_PYTHON_SOURCE
            # Revert mutation
            revert_mutation(src_file, r.mutation)
            # Should be back to original
            assert src_file.read_text() == SAMPLE_PYTHON_SOURCE

    def test_diverse_mutation_types(self) -> None:
        """VAL-SYNTH-04: At least 3 categories across generated tasks."""
        pipeline = self._make_pipeline(max_mutations_per_type=5)
        results = pipeline.generate(SAMPLE_PYTHON_SOURCE, language="python", file_path="calc.py")
        successful = [r for r in results if r.success and r.mutation is not None]

        categories = {r.mutation.mutation_type for r in successful}
        assert len(categories) >= 3, f"Expected at least 3 mutation categories, got: {categories}"

    def test_produces_10_plus_tasks(self) -> None:
        """VAL-SYNTH-04: 10+ generated tasks from a codebase."""
        pipeline = self._make_pipeline(max_mutations_per_type=5)
        results = pipeline.generate(SAMPLE_PYTHON_SOURCE, language="python", file_path="calc.py")
        successful = [r for r in results if r.success]
        assert len(successful) >= 10, f"Expected 10+ tasks, got {len(successful)}"

    def test_failed_mutations_gracefully_skipped(self, caplog) -> None:
        """VAL-SYNTH-06: Failed mutations are skipped with logging, not crashes."""
        pipeline = self._make_pipeline()

        # Source with a function that can't easily be mutated to produce valid code
        tricky_source = _source("""
            def single_line():
                return 42
        """)
        with caplog.at_level(logging.WARNING):
            results = pipeline.generate(tricky_source, language="python", file_path="tiny.py")

        # Pipeline should not crash
        assert isinstance(results, list)
        # Check that failures are logged
        failed = [r for r in results if not r.success]
        for f in failed:
            assert f.error is not None
            # The log should contain information about the failure
            log_messages = [rec.message for rec in caplog.records]
            assert any(
                "skip" in msg.lower() or "fail" in msg.lower() or "error" in msg.lower()
                for msg in log_messages
            ), f"Expected logging for failed mutation, got: {f.error}"

    def test_pipeline_with_invalid_source_does_not_crash(self, caplog) -> None:
        """VAL-SYNTH-06: Invalid source code doesn't crash the pipeline."""
        pipeline = self._make_pipeline()
        invalid_source = "def broken(\n"  # Syntax error
        with caplog.at_level(logging.WARNING):
            results = pipeline.generate(invalid_source, language="python", file_path="broken.py")
        # Should return results (possibly empty) without crashing
        assert isinstance(results, list)

    def test_pipeline_with_empty_source(self) -> None:
        pipeline = self._make_pipeline()
        results = pipeline.generate("", language="python", file_path="empty.py")
        assert isinstance(results, list)

    def test_pipeline_config_max_mutations(self) -> None:
        config = MutationPipelineConfig(max_mutations_per_type=1)
        pipeline = MutationPipeline(config)
        results = pipeline.generate(SAMPLE_PYTHON_SOURCE, language="python", file_path="calc.py")
        # Should have limited mutations
        successful = [r for r in results if r.success]
        assert len(successful) <= len(MutationType)  # At most 1 per type


# ===========================================================================
# MutationConfig
# ===========================================================================


class TestMutationPipelineConfig:
    """Tests for the pipeline configuration model."""

    def test_defaults_are_sensible(self) -> None:
        config = MutationPipelineConfig()
        assert config.max_mutations_per_type > 0
        assert config.target_functions is None  # All functions by default

    def test_custom_config(self) -> None:
        config = MutationPipelineConfig(
            max_mutations_per_type=3,
            target_functions=["add", "multiply"],
        )
        assert config.max_mutations_per_type == 3
        assert config.target_functions == ["add", "multiply"]


# ===========================================================================
# Custom Exceptions
# ===========================================================================


class TestExceptions:
    """Tests for custom exception classes."""

    def test_mutation_error_is_exception(self) -> None:
        assert issubclass(MutationError, Exception)

    def test_mutation_apply_error_is_mutation_error(self) -> None:
        assert issubclass(MutationApplyError, MutationError)

    def test_mutation_apply_error_message(self) -> None:
        err = MutationApplyError("original code not found in source")
        assert "original code not found" in str(err)
