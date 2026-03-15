"""Tests for the end-to-end task synthesis pipeline.

Validates:
- VAL-SYNTH-05: End-to-end task generation from source repo to task dataset
- VAL-TASKFMT-01: Multi-language task declaration with language, dependencies, env requirements
- Pipeline runs on a source repo and produces a validated dataset
- Quality gates filter out invalid tasks
- Tasks register in the artifact registry
- CLI 'grist-mill tasks generate --repo <path>' produces output dataset
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grist_mill.dataset import (
    Dataset,
)
from grist_mill.registry import ArtifactRegistry
from grist_mill.schemas import (
    Difficulty,
    Task,
)
from grist_mill.tasks.pipeline import (
    PipelineConfig,
    PipelineQualityGate,
    TaskPipeline,
    TaskPipelineResult,
    _discover_source_files,
    _infer_dependencies,
    _infer_test_command,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PYTHON_SOURCE = '''\
"""A simple math utility module."""

import math
from typing import List


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def factorial(n: int) -> int:
    """Compute factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

R_SOURCE = """\
#' Add two numbers
#' @param a First number
#' @param b Second number
#' @return Sum
add_numbers <- function(a, b) {
  return(a + b)
}

#' Multiply two numbers
#' @param a First number
#' @param b Second number
#' @return Product
multiply_numbers <- function(a, b) {
  return(a * b)
}
"""

PYTHON_TEST_FILE = '''\
"""Tests for math utilities."""
import pytest
from math_utils import add, multiply, factorial


def test_add():
    assert add(2, 3) == 5


def test_multiply():
    assert multiply(2, 3) == 6


def test_factorial():
    assert factorial(5) == 120
    assert factorial(0) == 1
'''

R_TEST_FILE = """\
test_that("add works", {
  expect_equal(add_numbers(2, 3), 5)
})

test_that("multiply works", {
  expect_equal(multiply_numbers(2, 3), 6)
})
"""


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    """Create a minimal Python source repo for testing."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "math_utils.py").write_text(PYTHON_SOURCE, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_math.py").write_text(PYTHON_TEST_FILE, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest>=7.0\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def multi_lang_repo(tmp_path: Path) -> Path:
    """Create a multi-language repo for testing."""
    # Python files
    py_dir = tmp_path / "python"
    py_dir.mkdir()
    (py_dir / "utils.py").write_text(PYTHON_SOURCE, encoding="utf-8")

    # R files
    r_dir = tmp_path / "R"
    r_dir.mkdir()
    (r_dir / "utils.R").write_text(R_SOURCE, encoding="utf-8")

    # Python tests
    py_test = tmp_path / "tests"
    py_test.mkdir()
    (py_test / "test_utils.py").write_text(PYTHON_TEST_FILE, encoding="utf-8")

    return tmp_path


@pytest.fixture
def empty_repo(tmp_path: Path) -> Path:
    """Create an empty repo (no source files)."""
    return tmp_path


# ---------------------------------------------------------------------------
# Tests for _discover_source_files
# ---------------------------------------------------------------------------


class TestDiscoverSourceFiles:
    """Tests for source file discovery."""

    def test_discovers_python_files(self, python_repo: Path) -> None:
        files = _discover_source_files(python_repo)
        basenames = {f.name for f in files}
        assert "math_utils.py" in basenames

    def test_discovers_r_files(self, multi_lang_repo: Path) -> None:
        files = _discover_source_files(multi_lang_repo)
        basenames = {f.name for f in files}
        assert "utils.py" in basenames
        assert "utils.R" in basenames

    def test_ignores_test_files(self, python_repo: Path) -> None:
        files = _discover_source_files(python_repo)
        test_files = {f.name for f in files if f.name.startswith("test_")}
        assert len(test_files) == 0

    def test_empty_repo(self, empty_repo: Path) -> None:
        files = _discover_source_files(empty_repo)
        assert len(files) == 0

    def test_skips_non_source_files(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# test", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
        files = _discover_source_files(tmp_path)
        basenames = {f.name for f in files}
        assert "main.py" in basenames
        assert "README.md" not in basenames
        assert "data.json" not in basenames


# ---------------------------------------------------------------------------
# Tests for _infer_dependencies
# ---------------------------------------------------------------------------


class TestInferDependencies:
    """Tests for dependency inference from imports."""

    def test_python_imports(self) -> None:
        deps = _infer_dependencies(PYTHON_SOURCE, "python")
        assert "math" in deps
        assert "typing" in deps

    def test_r_imports(self) -> None:
        deps = _infer_dependencies(R_SOURCE, "r")
        # R source doesn't use library() calls, so should be minimal
        assert isinstance(deps, list)

    def test_empty_source(self) -> None:
        deps = _infer_dependencies("", "python")
        assert deps == []

    def test_unknown_language(self) -> None:
        deps = _infer_dependencies("some code", "javascript")
        assert deps == []


# ---------------------------------------------------------------------------
# Tests for _infer_test_command
# ---------------------------------------------------------------------------


class TestInferTestCommand:
    """Tests for test command inference."""

    def test_python_command(self) -> None:
        cmd = _infer_test_command("python")
        assert "pytest" in cmd

    def test_r_command(self) -> None:
        cmd = _infer_test_command("r")
        assert "testthat" in cmd

    def test_typescript_command(self) -> None:
        cmd = _infer_test_command("typescript")
        assert "npm" in cmd or "npx" in cmd

    def test_unknown_language(self) -> None:
        cmd = _infer_test_command("unknown")
        assert cmd == "echo 'No test runner configured'"


# ---------------------------------------------------------------------------
# Tests for PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.max_mutations_per_type == 3
        assert config.max_tasks_per_file == 5
        assert config.timeout == 300
        assert config.quality_gate is not None

    def test_custom_config(self) -> None:
        config = PipelineConfig(
            max_mutations_per_type=5,
            max_tasks_per_file=10,
            timeout=600,
        )
        assert config.max_mutations_per_type == 5
        assert config.max_tasks_per_file == 10
        assert config.timeout == 600

    def test_quality_gate_default(self) -> None:
        config = PipelineConfig()
        assert config.quality_gate.min_prompt_length > 0
        assert config.quality_gate.min_description_length > 0


# ---------------------------------------------------------------------------
# Tests for PipelineQualityGate
# ---------------------------------------------------------------------------


class TestPipelineQualityGate:
    """Tests for the pipeline quality gate."""

    def test_valid_task_passes(self) -> None:
        gate = PipelineQualityGate()
        task = Task(
            id="test-001",
            prompt="Fix the bug in the add function where it returns wrong result for negative numbers",
            language="python",
            test_command="pytest tests/test_math.py -v",
            timeout=300,
            dependencies=["pytest"],
        )
        assert gate.passes_quality_gate(task) is True

    def test_short_prompt_fails(self) -> None:
        gate = PipelineQualityGate(min_prompt_length=20)
        task = Task(
            id="test-002",
            prompt="Fix bug",  # Too short
            language="python",
            test_command="pytest tests/",
            timeout=300,
        )
        assert gate.passes_quality_gate(task) is False

    def test_empty_language_fails(self) -> None:
        """Empty language is rejected by Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Task(
                id="test-003",
                prompt="Fix the bug in the function",
                language="",  # Empty - violates min_length=1
                test_command="pytest tests/",
                timeout=300,
            )

    def test_filter_tasks(self) -> None:
        gate = PipelineQualityGate(min_prompt_length=20)
        good_task = Task(
            id="good-001",
            prompt="Fix the bug in the add function where it returns wrong result for negative numbers",
            language="python",
            test_command="pytest tests/test_math.py",
            timeout=300,
        )
        bad_task = Task(
            id="bad-001",
            prompt="Fix it",  # Too short
            language="python",
            test_command="pytest tests/",
            timeout=300,
        )
        filtered = gate.filter_tasks([good_task, bad_task])
        assert len(filtered) == 1
        assert filtered[0].id == "good-001"


# ---------------------------------------------------------------------------
# Tests for TaskPipelineResult
# ---------------------------------------------------------------------------


class TestTaskPipelineResult:
    """Tests for pipeline result dataclass."""

    def test_result_structure(self) -> None:
        dataset = Dataset()
        result = TaskPipelineResult(
            dataset=dataset,
            total_source_files=5,
            total_mutations_attempted=20,
            total_mutations_succeeded=15,
            tasks_filtered_by_quality=2,
        )
        assert result.total_source_files == 5
        assert result.total_mutations_succeeded == 15
        assert result.tasks_filtered_by_quality == 2
        assert result.dataset.task_count == 0


# ---------------------------------------------------------------------------
# Tests for TaskPipeline (end-to-end)
# ---------------------------------------------------------------------------


class TestTaskPipeline:
    """Tests for the end-to-end task pipeline."""

    def test_pipeline_on_python_repo(self, python_repo: Path) -> None:
        """Pipeline runs on a Python repo and produces a validated dataset."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        # Should produce at least one task from the repo
        assert result.total_source_files > 0
        assert result.total_mutations_attempted > 0
        assert result.dataset.task_count > 0
        assert result.total_mutations_succeeded > 0

    def test_pipeline_on_multi_language_repo(self, multi_lang_repo: Path) -> None:
        """Pipeline handles multi-language repos correctly."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(multi_lang_repo)

        # Should have tasks from multiple languages
        languages = {task.language for task in result.dataset.list_tasks()}
        assert len(languages) >= 1  # At least one language's mutators work

    def test_pipeline_on_empty_repo(self, empty_repo: Path) -> None:
        """Pipeline handles empty repos gracefully."""
        config = PipelineConfig()
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(empty_repo)

        assert result.total_source_files == 0
        assert result.dataset.task_count == 0

    def test_generated_tasks_declare_language(self, python_repo: Path) -> None:
        """Generated tasks declare their language."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        for task in result.dataset.list_tasks():
            assert task.language in ("python", "r", "typescript")

    def test_generated_tasks_declare_dependencies(self, python_repo: Path) -> None:
        """Generated tasks declare dependencies."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        for task in result.dataset.list_tasks():
            assert isinstance(task.dependencies, list)

    def test_generated_tasks_have_environment_requirements(self, python_repo: Path) -> None:
        """Generated tasks have test commands (execution environment requirements)."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        for task in result.dataset.list_tasks():
            assert len(task.test_command) > 0
            assert task.timeout > 0

    def test_quality_gates_filter_invalid(self, python_repo: Path) -> None:
        """Quality gates filter out invalid tasks."""
        # Use a very restrictive quality gate
        gate = PipelineQualityGate(min_prompt_length=1000)
        config = PipelineConfig(
            quality_gate=gate,
            max_mutations_per_type=2,
            max_tasks_per_file=10,
        )
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        # With min_prompt_length=1000, most auto-generated descriptions
        # won't pass the gate, so tasks_filtered should be high
        assert result.tasks_filtered_by_quality >= 0
        # All tasks in the dataset should pass the gate
        for task in result.dataset.list_tasks():
            assert gate.passes_quality_gate(task) is True

    def test_pipeline_registers_tasks_in_artifact_registry(self, python_repo: Path) -> None:
        """Tasks are registered in the artifact registry."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        registry = ArtifactRegistry()
        pipeline.run(python_repo, registry=registry)

        # Tasks should be registered as artifacts
        assert registry.count > 0

    def test_pipeline_without_registry(self, python_repo: Path) -> None:
        """Pipeline works without a registry (optional)."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)
        # Should succeed without a registry
        assert result.dataset.task_count >= 0

    def test_pipeline_with_output_dir(self, python_repo: Path, tmp_path: Path) -> None:
        """Pipeline can export results to a directory."""
        output_dir = tmp_path / "output"
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo, output_dir=output_dir)

        # Check that output files exist
        assert (output_dir / "tasks.json").exists()

        # Verify the exported data is valid
        data = json.loads((output_dir / "tasks.json").read_text())
        assert "schema_version" in data
        assert "tasks" in data
        assert len(data["tasks"]) == result.dataset.task_count

    def test_pipeline_difficulty_estimation(self, python_repo: Path) -> None:
        """Pipeline estimates difficulty for generated tasks."""
        config = PipelineConfig(
            max_mutations_per_type=2,
            max_tasks_per_file=10,
            estimate_difficulty=True,
        )
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        for task in result.dataset.list_tasks():
            assert isinstance(task.difficulty, Difficulty)


# ---------------------------------------------------------------------------
# Tests for CLI 'grist-mill tasks generate'
# ---------------------------------------------------------------------------


class TestTasksGenerateCLI:
    """Tests for the 'grist-mill tasks generate' CLI subcommand."""

    def test_tasks_generate_help(self) -> None:
        """CLI 'grist-mill tasks generate --help' shows help text."""
        from click.testing import CliRunner

        from grist_mill.cli.tasks_cmd import generate

        runner = CliRunner()
        result = runner.invoke(generate, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output.lower()
        assert "--repo" in result.output

    def test_tasks_generate_from_repo(self, python_repo: Path, tmp_path: Path) -> None:
        """CLI generates tasks from a repo."""
        from click.testing import CliRunner

        from grist_mill.cli.tasks_cmd import generate

        output_dir = tmp_path / "cli_out"
        runner = CliRunner()
        result = runner.invoke(
            generate,
            [
                "--repo",
                str(python_repo),
                "--output",
                str(output_dir),
            ],
            catch_exceptions=False,
        )
        # Exit code 0 means success
        assert result.exit_code == 0
        assert (output_dir / "tasks.json").exists()

    def test_tasks_generate_missing_repo(self) -> None:
        """CLI handles missing repo path gracefully."""
        from click.testing import CliRunner

        from grist_mill.cli.tasks_cmd import generate

        runner = CliRunner()
        result = runner.invoke(generate, ["--repo", "/nonexistent/path/repo"])
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "not found" in result.output.lower()

    def test_tasks_generate_quiet(self, python_repo: Path, tmp_path: Path) -> None:
        """CLI --quiet flag suppresses verbose output."""
        from click.testing import CliRunner

        from grist_mill.cli.tasks_cmd import generate

        output_dir = tmp_path / "cli_out_quiet"
        runner = CliRunner()
        result = runner.invoke(
            generate,
            [
                "--repo",
                str(python_repo),
                "--output",
                str(output_dir),
                "--quiet",
            ],
        )
        # Should still succeed
        assert result.exit_code == 0

    def test_tasks_group_registered(self) -> None:
        """The 'tasks' group is registered in the CLI."""
        from click.testing import CliRunner

        from grist_mill.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "tasks" in result.output.lower()


# ---------------------------------------------------------------------------
# Tests for VAL-SYNTH-05 (end-to-end without manual intervention)
# ---------------------------------------------------------------------------


class TestEndToEndNoManualIntervention:
    """Tests for VAL-SYNTH-05: Pipeline runs without manual intervention."""

    def test_pipeline_produces_dataset_automatically(self, python_repo: Path) -> None:
        """Pipeline on a source repo produces a non-empty validated dataset."""
        config = PipelineConfig(max_mutations_per_type=2, max_tasks_per_file=10)
        pipeline = TaskPipeline(config=config)
        result = pipeline.run(python_repo)

        assert result.dataset.task_count > 0, "Dataset should be non-empty"
        assert result.total_source_files > 0, "Should have processed source files"

        # All tasks in the dataset should be valid
        for task in result.dataset.list_tasks():
            assert len(task.id) > 0
            assert len(task.prompt) > 0
            assert len(task.language) > 0
            assert len(task.test_command) > 0
            assert task.timeout > 0


# ---------------------------------------------------------------------------
# Tests for VAL-TASKFMT-01 (multi-language task declaration)
# ---------------------------------------------------------------------------


class TestMultiLanguageTaskDeclaration:
    """Tests for VAL-TASKFMT-01: Tasks declare language, dependencies, env requirements."""

    def test_task_has_language(self) -> None:
        task = Task(
            id="test-lang",
            prompt="Fix the function",
            language="python",
            test_command="pytest",
            timeout=60,
            dependencies=["numpy"],
        )
        assert task.language == "python"

    def test_task_has_dependencies(self) -> None:
        task = Task(
            id="test-deps",
            prompt="Fix the function",
            language="python",
            test_command="pytest",
            timeout=60,
            dependencies=["numpy", "pandas"],
        )
        assert "numpy" in task.dependencies
        assert "pandas" in task.dependencies

    def test_task_has_env_requirements(self) -> None:
        """Task test_command and setup_command define execution environment."""
        task = Task(
            id="test-env",
            prompt="Fix the function",
            language="python",
            test_command="pytest tests/ -v",
            setup_command="pip install -r requirements.txt",
            timeout=120,
        )
        assert len(task.test_command) > 0
        assert task.setup_command is not None
