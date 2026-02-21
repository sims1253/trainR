"""Main task generation pipeline."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from .models import Difficulty, ExtractedPattern, TestingTask, TestPattern
from .pattern_extractor import PatternExtractor
from .quality_gate import TaskQualityGate
from .templates import TaskTemplate, TemplateRegistry

logger = logging.getLogger(__name__)


class TaskGenerator:
    """Generate testing tasks from R packages."""

    def __init__(
        self,
        output_dir: Path,
        template_registry: TemplateRegistry | None = None,
        quality_gate: TaskQualityGate | None = None,
    ) -> None:
        """Initialize the task generator.

        Args:
            output_dir: Directory to save generated tasks.
            template_registry: Registry of task templates.
            quality_gate: Quality gate for task validation.
        """
        self.output_dir = Path(output_dir)
        self.templates = template_registry or TemplateRegistry()
        self.quality_gate = quality_gate or TaskQualityGate()

    def generate_from_package(
        self,
        package_path: Path,
        num_tasks: int = 20,
        split_ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
        difficulty_distribution: dict[Difficulty, float] | None = None,
    ) -> list[TestingTask]:
        """Generate tasks from an R package.

        Args:
            package_path: Path to the R package.
            num_tasks: Number of tasks to generate.
            split_ratio: Ratio for train/dev/held_out splits.
            difficulty_distribution: Optional distribution for difficulty levels.

        Returns:
            List of generated TestingTask objects.
        """
        package_path = Path(package_path)

        # Extract patterns from the package
        logger.info(f"Extracting patterns from {package_path}")
        extractor = PatternExtractor(package_path)
        patterns = extractor.extract_all_patterns()
        package_info = extractor.package_info

        if not patterns:
            logger.warning(f"No patterns found in {package_path}")
            return []

        logger.info(f"Found {len(patterns)} patterns")

        # Get all source functions
        source_functions = extractor.get_all_source_functions()
        logger.info(f"Found {len(source_functions)} source functions")

        # Map functions to their test patterns
        func_to_patterns = self._map_patterns_to_functions(patterns, source_functions)

        # Generate tasks
        tasks = self._generate_tasks(
            patterns=patterns,
            source_functions=source_functions,
            func_to_patterns=func_to_patterns,
            package_info=package_info,
            num_tasks=num_tasks,
            difficulty_distribution=difficulty_distribution,
        )

        # Assign splits
        tasks = self._assign_splits(tasks, split_ratio)

        # Validate tasks
        valid_tasks, rejected = self.quality_gate.filter_valid(tasks)
        logger.info(f"Generated {len(valid_tasks)} valid tasks, rejected {len(rejected)}")

        return valid_tasks

    def _map_patterns_to_functions(
        self,
        patterns: list[ExtractedPattern],
        source_functions: list[dict[str, Any]],
    ) -> dict[str, list[ExtractedPattern]]:
        """Map test patterns to the functions they test.

        Args:
            patterns: Extracted test patterns.
            source_functions: Source functions from the package.

        Returns:
            Dictionary mapping function names to their test patterns.
        """
        func_names = {f["name"] for f in source_functions}
        mapping: dict[str, list[ExtractedPattern]] = {name: [] for name in func_names}

        for pattern in patterns:
            # Try to identify which function is being tested
            tested_func = self._identify_tested_function(pattern, func_names)
            if tested_func:
                mapping[tested_func].append(pattern)

        return mapping

    def _identify_tested_function(
        self, pattern: ExtractedPattern, func_names: set[str]
    ) -> str | None:
        """Identify which function a pattern tests.

        Args:
            pattern: The test pattern.
            func_names: Set of known function names.

        Returns:
            Function name or None.
        """
        # Check for function name in pattern
        if pattern.function_name and pattern.function_name in func_names:
            return pattern.function_name

        # Check context_before for function references
        if pattern.context_before:
            for name in func_names:
                if name in pattern.context_before:
                    return name

        # Check code snippet for function calls
        code = pattern.code_snippet.lower()
        for name in func_names:
            if name.lower() in code:
                return name

        return None

    def _generate_tasks(
        self,
        patterns: list[ExtractedPattern],
        source_functions: list[dict[str, Any]],
        func_to_patterns: dict[str, list[ExtractedPattern]],
        package_info: dict[str, Any],
        num_tasks: int,
        difficulty_distribution: dict[Difficulty, float] | None,
    ) -> list[TestingTask]:
        """Generate tasks from patterns and functions.

        Args:
            patterns: All extracted patterns.
            source_functions: All source functions.
            func_to_patterns: Mapping of functions to patterns.
            package_info: Package information.
            num_tasks: Number of tasks to generate.
            difficulty_distribution: Optional difficulty distribution.

        Returns:
            List of generated tasks.
        """
        tasks: list[TestingTask] = []

        # Determine difficulty distribution
        if difficulty_distribution is None:
            difficulty_distribution = {
                Difficulty.EASY: 0.2,
                Difficulty.MEDIUM: 0.4,
                Difficulty.HARD: 0.4,
            }

        # Calculate tasks per difficulty
        difficulty_counts = {
            diff: int(num_tasks * ratio) for diff, ratio in difficulty_distribution.items()
        }

        # Adjust to hit target
        total_assigned = sum(difficulty_counts.values())
        if total_assigned < num_tasks:
            difficulty_counts[Difficulty.MEDIUM] += num_tasks - total_assigned

        # Generate tasks for each function with tests
        for func in source_functions:
            func_patterns = func_to_patterns.get(func["name"], [])

            if not func_patterns:
                # Function has no tests - could be a candidate for new tests
                continue

            # Get applicable templates
            for difficulty, count in difficulty_counts.items():
                if count <= 0:
                    continue

                applicable_templates = self.templates.get_for_patterns(func_patterns, difficulty)

                if not applicable_templates:
                    # Try without difficulty filter
                    applicable_templates = self.templates.get_for_patterns(func_patterns)
                    applicable_templates = [
                        t for t in applicable_templates if t.difficulty == difficulty
                    ]

                if not applicable_templates:
                    continue

                # Generate a task using a random template
                template = random.choice(applicable_templates)
                task = self._create_task(
                    source_function=func,
                    patterns=func_patterns,
                    template=template,
                    package_info=package_info,
                    difficulty=difficulty,
                )

                if task:
                    tasks.append(task)
                    difficulty_counts[difficulty] -= 1

                if len(tasks) >= num_tasks:
                    break

            if len(tasks) >= num_tasks:
                break

        # If we still need more tasks, generate without functions
        while len(tasks) < num_tasks:
            remaining_patterns = (
                random.sample(patterns, min(5, len(patterns))) if len(patterns) > 5 else patterns
            )

            template = self.templates.get_random(remaining_patterns)
            if not template:
                template = self.templates.get("write_test")

            if template:
                difficulty = template.difficulty
                if difficulty_counts.get(difficulty, 0) > 0:
                    task = self._create_task(
                        source_function=None,
                        patterns=remaining_patterns,
                        template=template,
                        package_info=package_info,
                        difficulty=difficulty,
                    )
                    if task:
                        tasks.append(task)
                        difficulty_counts[difficulty] -= 1
                    else:
                        break
                else:
                    # Pick a difficulty we need
                    for diff, count in difficulty_counts.items():
                        if count > 0:
                            difficulty = diff
                            break

                    task = self._create_task(
                        source_function=None,
                        patterns=remaining_patterns,
                        template=template,
                        package_info=package_info,
                        difficulty=difficulty,
                    )
                    if task:
                        tasks.append(task)
                        difficulty_counts[difficulty] -= 1
                    else:
                        break

        return tasks

    def _create_task(
        self,
        source_function: dict[str, Any] | None,
        patterns: list[ExtractedPattern],
        template: TaskTemplate,
        package_info: dict[str, Any],
        difficulty: Difficulty,
    ) -> TestingTask | None:
        """Create a single task.

        Args:
            source_function: The function being tested.
            patterns: Reference patterns.
            template: Task template to use.
            package_info: Package information.
            difficulty: Task difficulty.

        Returns:
            TestingTask or None if creation fails.
        """
        # Build context
        context = self._build_context(package_info, source_function, patterns)

        # Get function code
        function_code = source_function["code"] if source_function else None

        # Generate instruction
        instruction = template.generate_instruction(function_code, patterns, context)

        # Generate reference test
        reference_test = template.generate_reference_test(function_code, patterns)

        # Generate task ID
        func_name = source_function["name"] if source_function else "general"
        content = f"{package_info.get('name', 'unknown')}:{func_name}:{template.name}:{instruction}"
        task_id = self._generate_task_id(content)

        # Determine test type
        test_type = self._determine_test_type(patterns)

        # Get source file
        source_file = source_function["source_file"] if source_function else "unknown"

        # Extract unique patterns
        unique_patterns = list({p.pattern_type for p in patterns})

        # Get dependencies - trim to what's actually needed
        all_deps = package_info.get("dependencies", [])
        dependencies = self._filter_dependencies(all_deps, patterns, reference_test)

        # Create task (split will be assigned later, quality_score set by quality gate)
        task = TestingTask(
            task_id=task_id,
            source_package=package_info.get("name", "unknown"),
            source_file=source_file,
            difficulty=difficulty,
            instruction=instruction,
            context=context,
            reference_test=reference_test,
            test_type=test_type,
            patterns=unique_patterns,
            dependencies=dependencies,
            split="",  # Will be assigned in _assign_splits
            function_name=func_name if source_function else None,
            quality_score=0.0,
            constraints={},
        )

        return task

    def _build_context(
        self,
        package_info: dict[str, Any],
        source_function: dict[str, Any] | None,
        patterns: list[ExtractedPattern],
    ) -> str:
        """Build context string for a task.

        Args:
            package_info: Package information.
            source_function: Source function being tested.
            patterns: Reference patterns.

        Returns:
            Context string.
        """
        parts = []

        # Package info
        parts.append(
            f"Package: {package_info.get('name', 'unknown')} v{package_info.get('version', '0.0.0')}"
        )
        if package_info.get("title"):
            parts.append(f"Description: {package_info['title']}")

        # Function info
        if source_function:
            parts.append(f"Function: {source_function['name']}")
            if source_function.get("parameters"):
                parts.append(f"Parameters: {', '.join(source_function['parameters'])}")

        # Pattern info
        pattern_types = {str(p.pattern_type) for p in patterns}
        if pattern_types:
            parts.append(f"Test patterns used: {', '.join(sorted(pattern_types))}")

        expectation_types = set()
        for p in patterns:
            expectation_types.update(p.expectations)
        if expectation_types:
            parts.append(f"Expectations used: {', '.join(sorted(expectation_types)[:5])}")

        # Function code
        if source_function and source_function.get("code"):
            parts.append("")  # Blank line before code section
            parts.append("Function Code:")
            parts.append("```r")
            parts.append(source_function["code"])
            parts.append("```")

        return "\n".join(parts)

    def _determine_test_type(self, patterns: list[ExtractedPattern]) -> str:
        """Determine the type of test based on patterns.

        Args:
            patterns: List of extracted patterns.

        Returns:
            Test type string (unit, snapshot, integration).
        """
        pattern_types = {p.pattern_type for p in patterns}

        if TestPattern.EXPECT_SNAPSHOT in pattern_types:
            return "snapshot"
        elif TestPattern.LOCAL_MOCKED_BINDINGS in pattern_types:
            return "integration"
        else:
            return "unit"

    def _filter_dependencies(
        self,
        all_deps: list[str],
        patterns: list[ExtractedPattern],
        reference_test: str,
    ) -> list[str]:
        """Filter dependencies to only what's needed for the task.

        Args:
            all_deps: Full dependency list from package DESCRIPTION.
            patterns: Patterns used in this task.
            reference_test: Reference test code.

        Returns:
            Filtered dependency list.
        """
        # Always needed for testing
        core_deps = {"testthat"}

        # Check patterns for framework-specific deps
        pattern_types = {p.pattern_type for p in patterns}
        if TestPattern.WITH_FIXTURE in pattern_types:
            core_deps.add("withr")
        if TestPattern.LOCAL_MOCKED_BINDINGS in pattern_types:
            core_deps.add("testthat")  # Already there, but explicit

        # Check reference test for package mentions
        combined_text = reference_test + " ".join(p.code_snippet for p in patterns)
        for dep in all_deps:
            if dep in combined_text:
                core_deps.add(dep)

        # Always include the source package itself
        # (will be added via source_package field)

        return sorted(core_deps)

    def _generate_task_id(self, content: str) -> str:
        """Generate unique task ID from content.

        Args:
            content: Content to hash.

        Returns:
            Task ID string.
        """
        hash_bytes = hashlib.md5(content.encode()).digest()[:4]
        return f"task-{hash_bytes.hex()}"

    def _assign_splits(
        self,
        tasks: list[TestingTask],
        split_ratio: tuple[float, float, float],
    ) -> list[TestingTask]:
        """Assign train/dev/held_out splits to tasks.

        Args:
            tasks: List of tasks.
            split_ratio: Ratio for train/dev/held_out.

        Returns:
            Tasks with splits assigned.
        """
        if not tasks:
            return tasks

        # Shuffle tasks for random assignment
        shuffled = tasks.copy()
        random.shuffle(shuffled)

        # Calculate split sizes
        n = len(shuffled)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        # Assign splits
        for i, task in enumerate(shuffled):
            if i < train_end:
                task.split = "train"
            elif i < val_end:
                task.split = "dev"
            else:
                task.split = "held_out"

        return shuffled

    def save_task(self, task: TestingTask) -> Path:
        """Save a task to a JSON file.

        Args:
            task: The task to save.

        Returns:
            Path to the saved file.
        """
        split_dir = self.output_dir / task.split
        split_dir.mkdir(parents=True, exist_ok=True)

        task_file = split_dir / f"{task.task_id}.json"
        task_file.write_text(json.dumps(task.to_dict(), indent=2))

        return task_file

    def save_all_tasks(self, tasks: list[TestingTask]) -> list[Path]:
        """Save all tasks to JSON files.

        Args:
            tasks: List of tasks to save.

        Returns:
            List of paths to saved files.
        """
        paths = []
        for task in tasks:
            path = self.save_task(task)
            paths.append(path)

        return paths

    def generate_and_save(
        self,
        package_path: Path,
        num_tasks: int = 20,
        split_ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> tuple[list[TestingTask], list[Path]]:
        """Generate and save tasks in one call.

        Args:
            package_path: Path to the R package.
            num_tasks: Number of tasks to generate.
            split_ratio: Ratio for train/dev/held_out splits.

        Returns:
            Tuple of (tasks, file_paths).
        """
        tasks = self.generate_from_package(
            package_path=package_path,
            num_tasks=num_tasks,
            split_ratio=split_ratio,
        )

        paths = self.save_all_tasks(tasks)

        return tasks, paths

    def load_task(self, task_path: Path) -> TestingTask:
        """Load a task from a JSON file.

        Args:
            task_path: Path to the task JSON file.

        Returns:
            TestingTask object.
        """
        content = task_path.read_text()
        data = json.loads(content)
        return TestingTask.from_dict(data)

    def load_all_tasks(self, split: str | None = None) -> list[TestingTask]:
        """Load all tasks from the output directory.

        Args:
            split: Optional split to load (train/dev/held_out).

        Returns:
            List of TestingTask objects.
        """
        tasks = []

        splits_to_load = [split] if split else ["train", "dev", "held_out"]

        for s in splits_to_load:
            split_dir = self.output_dir / s
            if not split_dir.exists():
                continue

            for task_file in split_dir.glob("task-*.json"):
                try:
                    task = self.load_task(task_file)
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"Failed to load {task_file}: {e}")

        return tasks

    def get_statistics(self, tasks: list[TestingTask]) -> dict[str, Any]:
        """Get statistics about generated tasks.

        Args:
            tasks: List of tasks.

        Returns:
            Statistics dictionary.
        """
        if not tasks:
            return {"total": 0}

        stats: dict[str, Any] = {
            "total": len(tasks),
            "by_split": {},
            "by_difficulty": {},
            "by_package": {},
            "by_test_type": {},
        }

        for task in tasks:
            # By split
            split = task.split or "unknown"
            stats["by_split"][split] = stats["by_split"].get(split, 0) + 1

            # By difficulty
            diff = str(task.difficulty)
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # By package
            pkg = task.source_package
            stats["by_package"][pkg] = stats["by_package"].get(pkg, 0) + 1

            # By test type
            test_type = task.test_type
            stats["by_test_type"][test_type] = stats["by_test_type"].get(test_type, 0) + 1

        return stats
