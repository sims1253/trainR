"""End-to-end task synthesis pipeline.

Wires together source repository discovery, AST analysis, mutation,
task generation, quality gates, and dataset production. Generated tasks
declare language, dependencies, and execution environment requirements.
Tasks can be registered in the artifact registry.

Validates:
- VAL-SYNTH-05: End-to-end task generation from source repo to task dataset
- VAL-TASKFMT-01: Multi-language task declaration with language, dependencies, env requirements
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from grist_mill.dataset import (
    Dataset,
    DatasetExport,
    DatasetQualityReport,
    DifficultyEstimator,
)
from grist_mill.schemas import (
    Task,
)
from grist_mill.tasks.ast_parser import detect_language
from grist_mill.tasks.mutation import (
    MutationPipeline,
    MutationPipelineConfig,
    MutationResult,
)

if TYPE_CHECKING:
    from grist_mill.registry import ArtifactRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported source file extensions
_SOURCE_EXTENSIONS: set[str] = {".py", ".r", ".R", ".ts", ".tsx"}

# Patterns for files to exclude from source discovery
_EXCLUDE_PATTERNS: list[str] = [
    r"test_",
    r"_test\.py$",
    r"test_",
    r"__pycache__",
    r"\.pyc$",
    r"node_modules",
    r"\.git",
    r"venv",
    r"\.venv",
    r"__pypackages__",
]

# Language-to-test-command mapping
_TEST_COMMANDS: dict[str, str] = {
    "python": "pytest tests/ -v",
    "r": "Rscript -e 'testthat::test_dir(\"tests/\")'",
    "typescript": "npx jest --verbose",
}

# Language-to-setup-command mapping
_SETUP_COMMANDS: dict[str, str] = {
    "python": "pip install -r requirements.txt",
    "r": 'Rscript -e \'if (file.exists("DESCRIPTION")) pak::pkg_install(".")\'',
    "typescript": "npm install",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_exclude_pattern(name: str) -> bool:
    """Check if a filename matches any exclusion pattern."""
    return any(re.search(pattern, name) for pattern in _EXCLUDE_PATTERNS)


def discover_source_files(repo_path: Path) -> list[Path]:
    """Discover source files in a repository.

    Walks the directory tree looking for source files with supported
    extensions, excluding test files, caches, and other non-source patterns.

    Args:
        repo_path: Root path of the source repository.

    Returns:
        Sorted list of source file paths.
    """
    if not repo_path.is_dir():
        logger.warning("Repository path does not exist: %s", repo_path)
        return []

    source_files: list[Path] = []
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in _SOURCE_EXTENSIONS:
            continue
        # Check exclude patterns against relative path parts
        rel_parts = path.relative_to(repo_path).parts
        if any(_matches_exclude_pattern(part) for part in rel_parts):
            continue
        # Also check the filename itself
        if _matches_exclude_pattern(path.name):
            continue
        source_files.append(path)

    logger.info(
        "Discovered %d source files in %s",
        len(source_files),
        repo_path,
    )
    return source_files


def _infer_dependencies(source: str, language: str) -> list[str]:
    """Infer dependencies from source code imports.

    Extracts import statements and returns a list of dependency names.

    Args:
        source: The source code string.
        language: Programming language.

    Returns:
        List of inferred dependency names.
    """
    deps: list[str] = []

    if language == "python":
        # Match 'import X' and 'from X import Y'
        import_pattern = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
        from_pattern = re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE)

        for match in import_pattern.finditer(source):
            module = match.group(1).split(".")[0]
            if module not in deps:
                deps.append(module)

        for match in from_pattern.finditer(source):
            module = match.group(1).split(".")[0]
            if module not in deps:
                deps.append(module)

    elif language == "r":
        # Match library() and require() calls
        lib_pattern = re.compile(r"library\((\w+)\)", re.MULTILINE)
        require_pattern = re.compile(r"require\((\w+)\)", re.MULTILINE)

        for match in lib_pattern.finditer(source):
            pkg = match.group(1)
            if pkg not in deps:
                deps.append(pkg)

        for match in require_pattern.finditer(source):
            pkg = match.group(1)
            if pkg not in deps:
                deps.append(pkg)

    return deps


def _infer_test_command(language: str) -> str:
    """Infer the test command for a language.

    Args:
        language: Programming language.

    Returns:
        Test command string.
    """
    return _TEST_COMMANDS.get(language, "echo 'No test runner configured'")


def _infer_setup_command(language: str, repo_path: Path) -> str | None:
    """Infer the setup command, checking for config files first.

    Args:
        language: Programming language.
        repo_path: Root path of the repository.

    Returns:
        Setup command string, or None if no setup needed.
    """
    if language == "python":
        # Check for requirements.txt, setup.py, pyproject.toml
        if (repo_path / "requirements.txt").exists():
            return "pip install -r requirements.txt"
        if (repo_path / "pyproject.toml").exists():
            return "pip install -e ."
        if (repo_path / "setup.py").exists():
            return "pip install -e ."
        return None

    if language == "r":
        if (repo_path / "DESCRIPTION").exists():
            return "Rscript -e 'pak::pkg_install(\".\")'"
        return None

    if language == "typescript":
        if (repo_path / "package.json").exists():
            return "npm install"
        return None

    return _SETUP_COMMANDS.get(language)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PipelineQualityGate(BaseModel):
    """Quality gate configuration for filtering generated tasks.

    Filters out tasks that don't meet minimum quality thresholds.

    Attributes:
        min_prompt_length: Minimum character length for the task prompt.
        min_description_length: Minimum character length for descriptions.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    min_prompt_length: int = Field(
        default=10,
        ge=1,
        description="Minimum character length for task prompts.",
    )
    min_description_length: int = Field(
        default=5,
        ge=1,
        description="Minimum character length for task descriptions.",
    )

    def passes_quality_gate(self, task: Task) -> bool:
        """Check whether a task passes the quality gate.

        Args:
            task: The task to validate.

        Returns:
            True if the task passes all quality checks.
        """
        if len(task.prompt.strip()) < self.min_prompt_length:
            return False
        return bool(task.language.strip()) and bool(task.test_command.strip())

    def filter_tasks(self, tasks: list[Task]) -> list[Task]:
        """Filter a list of tasks, keeping only those that pass quality gates.

        Args:
            tasks: List of tasks to filter.

        Returns:
            List of tasks that pass all quality checks.
        """
        passed: list[Task] = []
        for task in tasks:
            if self.passes_quality_gate(task):
                passed.append(task)
            else:
                logger.debug("Task %s filtered out by quality gate.", task.id)
        return passed


class PipelineConfig(BaseModel):
    """Configuration for the end-to-end task synthesis pipeline.

    Attributes:
        max_mutations_per_type: Maximum mutations to generate per mutation type per file.
        max_tasks_per_file: Maximum tasks to produce per source file.
        timeout: Default timeout (seconds) for generated tasks.
        quality_gate: Quality gate for filtering tasks.
        estimate_difficulty: Whether to estimate difficulty for generated tasks.
        register_in_registry: Whether to register tasks in the artifact registry.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    max_mutations_per_type: int = Field(
        default=3,
        ge=1,
        description="Maximum mutations per mutation type per file.",
    )
    max_tasks_per_file: int = Field(
        default=5,
        ge=1,
        description="Maximum tasks to produce per source file.",
    )
    timeout: int = Field(
        default=300,
        gt=0,
        description="Default timeout for generated tasks.",
    )
    quality_gate: PipelineQualityGate = Field(
        default_factory=PipelineQualityGate,
        description="Quality gate configuration.",
    )
    estimate_difficulty: bool = Field(
        default=True,
        description="Whether to estimate difficulty for generated tasks.",
    )
    register_in_registry: bool = Field(
        default=True,
        description="Whether to register tasks in the artifact registry.",
    )


@dataclass
class TaskPipelineResult:
    """Result of running the end-to-end task pipeline.

    Attributes:
        dataset: The final validated dataset of generated tasks.
        total_source_files: Number of source files discovered in the repo.
        total_mutations_attempted: Total number of mutations attempted across all files.
        total_mutations_succeeded: Number of mutations that succeeded.
        tasks_filtered_by_quality: Number of tasks removed by quality gates.
    """

    dataset: Dataset
    total_source_files: int = 0
    total_mutations_attempted: int = 0
    total_mutations_succeeded: int = 0
    tasks_filtered_by_quality: int = 0


# ---------------------------------------------------------------------------
# TaskPipeline
# ---------------------------------------------------------------------------


class TaskPipeline:
    """End-to-end task synthesis pipeline.

    Orchestrates the full flow from source repository to validated task dataset:
    1. Discover source files in the repository
    2. Parse each file with the AST parser
    3. Run the mutation pipeline on each file
    4. Convert mutations to Task objects with language, dependencies, and env requirements
    5. Apply quality gates to filter invalid tasks
    6. Estimate difficulty for passing tasks
    7. Build a Dataset with the validated tasks
    8. Optionally register tasks in the artifact registry
    9. Optionally export to disk

    Usage::

        pipeline = TaskPipeline()
        result = pipeline.run(Path("/path/to/repo"))
        print(f"Generated {result.dataset.task_count} tasks")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self._config = config or PipelineConfig()

    def run(
        self,
        repo_path: Path | str,
        *,
        registry: ArtifactRegistry | None = None,
        output_dir: Path | str | None = None,
    ) -> TaskPipelineResult:
        """Run the end-to-end pipeline on a source repository.

        Args:
            repo_path: Path to the source repository.
            registry: Optional artifact registry to register tasks in.
            output_dir: Optional directory to export the dataset to.

        Returns:
            A TaskPipelineResult with the generated dataset and statistics.
        """
        repo = Path(repo_path)

        # Step 1: Discover source files
        source_files = discover_source_files(repo)

        # Step 2-4: Process each file
        mutation_config = MutationPipelineConfig(
            max_mutations_per_type=self._config.max_mutations_per_type,
            language="python",  # Will be overridden per file
        )
        mutation_pipeline = MutationPipeline(config=mutation_config)

        all_tasks: list[Task] = []
        total_attempted = 0
        total_succeeded = 0

        for source_file in source_files:
            tasks_from_file, attempted, succeeded = self._process_file(
                source_file, repo, mutation_pipeline
            )
            all_tasks.extend(tasks_from_file)
            total_attempted += attempted
            total_succeeded += succeeded

            # Limit tasks per file
            if len(tasks_from_file) > self._config.max_tasks_per_file:
                excess = len(tasks_from_file) - self._config.max_tasks_per_file
                all_tasks = all_tasks[:-excess]
                total_succeeded -= excess

        # Step 5: Quality gates
        filtered_tasks = self._config.quality_gate.filter_tasks(all_tasks)
        filtered_count = len(all_tasks) - len(filtered_tasks)

        # Step 6: Difficulty estimation
        dataset = Dataset()
        dataset.add_tasks(filtered_tasks)

        if self._config.estimate_difficulty and dataset.task_count > 0:
            estimator = DifficultyEstimator(overwrite=True)
            estimator.estimate_dataset(dataset)

        # Step 7: Register in artifact registry
        if self._config.register_in_registry and registry is not None:
            self._register_tasks(dataset, registry)

        # Step 8: Export to disk
        if output_dir is not None:
            self._export_dataset(dataset, Path(output_dir))

        result = TaskPipelineResult(
            dataset=dataset,
            total_source_files=len(source_files),
            total_mutations_attempted=total_attempted,
            total_mutations_succeeded=total_succeeded,
            tasks_filtered_by_quality=filtered_count,
        )

        logger.info(
            "Pipeline completed: %d source files, %d mutations attempted, "
            "%d succeeded, %d tasks generated, %d filtered by quality gate.",
            result.total_source_files,
            result.total_mutations_attempted,
            result.total_mutations_succeeded,
            result.dataset.task_count,
            result.tasks_filtered_by_quality,
        )

        return result

    def _process_file(
        self,
        source_file: Path,
        repo_path: Path,
        mutation_pipeline: MutationPipeline,
    ) -> tuple[list[Task], int, int]:
        """Process a single source file through the pipeline.

        Args:
            source_file: Path to the source file.
            repo_path: Root path of the repository.
            mutation_pipeline: The mutation pipeline instance.

        Returns:
            Tuple of (tasks, mutations_attempted, mutations_succeeded).
        """
        # Detect language
        try:
            language_enum = detect_language(source_file)
            language = language_enum.value
        except Exception as exc:
            logger.warning(
                "Could not detect language for %s: %s. Skipping.",
                source_file,
                exc,
            )
            return [], 0, 0

        # Read source
        try:
            source = source_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s. Skipping.", source_file, exc)
            return [], 0, 0

        # Run mutation pipeline
        file_config = MutationPipelineConfig(
            max_mutations_per_type=self._config.max_mutations_per_type,
            language=language,
        )
        file_pipeline = MutationPipeline(config=file_config)
        mutation_results = file_pipeline.generate(
            source,
            language=language,
            file_path=str(source_file.relative_to(repo_path)),
        )

        attempted = len(mutation_results)
        succeeded = sum(1 for r in mutation_results if r.success)

        # Convert to Tasks
        tasks: list[Task] = []
        for mr in mutation_results:
            if not mr.success or mr.mutation is None:
                continue

            task = self._mutation_result_to_task(mr, language, source, repo_path)
            tasks.append(task)

        return tasks, attempted, succeeded

    def _mutation_result_to_task(
        self,
        mr: MutationResult,
        language: str,
        source: str,
        repo_path: Path,
    ) -> Task:
        """Convert a MutationResult to a Task.

        The task declares language, dependencies, and execution environment
        requirements (test command, setup command, timeout).

        Args:
            mr: The mutation result.
            language: Programming language.
            source: Original source code.
            repo_path: Repository root path.

        Returns:
            A Task object.
        """
        # Infer dependencies from source
        deps = _infer_dependencies(source, language)

        # Infer test and setup commands
        test_command = _infer_test_command(language)
        setup_command = _infer_setup_command(language, repo_path)

        task = Task(
            id=mr.task_id,
            prompt=mr.description,
            language=language,
            test_command=test_command,
            setup_command=setup_command,
            timeout=self._config.timeout,
            dependencies=deps,
        )

        return task

    def _register_tasks(
        self,
        dataset: Dataset,
        registry: ArtifactRegistry,
    ) -> None:
        """Register generated tasks as artifacts in the registry.

        Each task is registered as a ToolArtifact with metadata about
        the task (language, dependencies, test command).

        Args:
            dataset: The dataset of generated tasks.
            registry: The artifact registry to register tasks in.
        """
        from grist_mill.schemas.artifact import ToolArtifact

        for task in dataset.list_tasks():
            artifact_name = f"task:{task.id}"
            try:
                artifact = ToolArtifact(
                    type="tool",
                    name=artifact_name,
                    description=f"Synthesized task: {task.prompt[:200]}",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "language": {"type": "string"},
                            "test_command": {"type": "string"},
                        },
                    },
                    command=task.test_command,
                )
                registry.register(artifact, overwrite=True)
            except Exception as exc:
                logger.warning(
                    "Failed to register task %s as artifact: %s",
                    task.id,
                    exc,
                )

        logger.info(
            "Registered %d tasks in artifact registry.",
            dataset.task_count,
        )

    def _export_dataset(
        self,
        dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Export the dataset to disk.

        Exports as JSON with full metadata.

        Args:
            dataset: The dataset to export.
            output_dir: Directory to write files to.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = DatasetExport()
        exporter.to_json_file(dataset, output_dir / "tasks.json")

        # Also generate a quality report
        report = DatasetQualityReport.generate(dataset)
        report_path = output_dir / "quality_report.json"
        report_path.write_text(
            __import__("json").dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Exported dataset to %s", output_dir)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by tests)
# ---------------------------------------------------------------------------

# Expose _discover_source_files as module-level for direct import
_discover_source_files = discover_source_files
_infer_dependencies = _infer_dependencies
_infer_test_command = _infer_test_command


__all__ = [
    "PipelineConfig",
    "PipelineQualityGate",
    "TaskPipeline",
    "TaskPipelineResult",
    "_discover_source_files",
    "_infer_dependencies",
    "_infer_test_command",
    "discover_source_files",
]
