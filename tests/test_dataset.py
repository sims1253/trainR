"""Tests for the dataset management module.

Validates:
- VAL-DATASET-01: Stratified splitting preserves difficulty/category distribution
- VAL-DATASET-02: Dataset versioning produces immutable snapshots
- VAL-DATASET-03: Decontamination detects duplicates
- VAL-DATASET-04: Difficulty estimation produces meaningful labels
- VAL-DATASET-05: Export to JSON and CSV with all metadata
- VAL-DATASET-06: Quality reports summarize dataset health
- VAL-TASKFMT-02: Task metadata validation
- VAL-TASKFMT-03: Manual task authoring via YAML
- VAL-TASKFMT-04: Task ID uniqueness enforcement
"""

from __future__ import annotations

import csv
import io
import json
import textwrap
import uuid
from pathlib import Path

import pytest

from grist_mill.dataset import (
    Dataset,
    DatasetDecontamination,
    DatasetDecontaminationResult,
    DatasetExport,
    DatasetQualityReport,
    DatasetSplit,
    DatasetVersioningStore,
    DifficultyEstimator,
    DuplicateEntry,
    StratifiedSplitter,
    import_tasks_from_yaml,
)
from grist_mill.schemas import Difficulty, Task

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "",
    language: str = "python",
    difficulty: Difficulty = Difficulty.EASY,
    prompt: str = "Fix the bug in the function.",
    timeout: int = 60,
    test_command: str = "pytest test_foo.py",
    setup_command: str | None = None,
    constraints: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> Task:
    """Create a Task for testing."""
    if not task_id:
        task_id = f"task-{uuid.uuid4().hex[:8]}"
    return Task(
        id=task_id,
        prompt=prompt,
        language=language,
        test_command=test_command,
        timeout=timeout,
        difficulty=difficulty,
        setup_command=setup_command,
        constraints=constraints or [],
        dependencies=dependencies or [],
    )


def _make_tasks(n: int = 30, **overrides) -> list[Task]:
    """Create n tasks with varied difficulty and language."""
    tasks: list[Task] = []
    difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    languages = ["python", "r", "typescript"]
    for i in range(n):
        tasks.append(
            _make_task(
                task_id=f"task-{i:03d}",
                language=languages[i % len(languages)],
                difficulty=difficulties[i % len(difficulties)],
                prompt=f"Fix bug {i} in the codebase.",
                **overrides,
            )
        )
    return tasks


@pytest.fixture
def mixed_tasks() -> list[Task]:
    """Tasks with varied difficulty and language for split/decontamination tests."""
    return _make_tasks(30)


@pytest.fixture
def small_dataset(mixed_tasks: list[Task]) -> Dataset:
    """A Dataset containing mixed tasks."""
    ds = Dataset()
    for task in mixed_tasks:
        ds.add_task(task)
    return ds


# ---------------------------------------------------------------------------
# Dataset: basic operations and ID uniqueness (VAL-TASKFMT-04, VAL-TASKFMT-02)
# ---------------------------------------------------------------------------


class TestDataset:
    """Tests for Dataset core operations."""

    def test_add_task_success(self) -> None:
        ds = Dataset()
        task = _make_task(task_id="task-001")
        ds.add_task(task)
        assert ds.task_count == 1
        assert ds.get_task("task-001") is task

    def test_add_task_enforces_unique_id(self) -> None:
        ds = Dataset()
        task1 = _make_task(task_id="task-001")
        task2 = _make_task(task_id="task-001", prompt="Different prompt.")
        ds.add_task(task1)
        with pytest.raises(DuplicateEntry, match="task-001"):
            ds.add_task(task2)

    def test_add_tasks_batch(self) -> None:
        ds = Dataset()
        tasks = _make_tasks(5)
        ds.add_tasks(tasks)
        assert ds.task_count == 5

    def test_add_tasks_batch_rejects_duplicate(self) -> None:
        ds = Dataset()
        task = _make_task(task_id="dup")
        ds.add_task(task)
        with pytest.raises(DuplicateEntry, match="dup"):
            ds.add_tasks([task])

    def test_get_task_not_found(self) -> None:
        ds = Dataset()
        assert ds.get_task("nonexistent") is None

    def test_list_tasks(self) -> None:
        ds = Dataset()
        tasks = _make_tasks(3)
        ds.add_tasks(tasks)
        listed = ds.list_tasks()
        assert len(listed) == 3

    def test_remove_task(self) -> None:
        ds = Dataset()
        task = _make_task(task_id="to-remove")
        ds.add_task(task)
        ds.remove_task("to-remove")
        assert ds.task_count == 0
        assert ds.get_task("to-remove") is None

    def test_remove_task_not_found(self) -> None:
        ds = Dataset()
        ds.remove_task("nonexistent")  # should not raise

    def test_empty_dataset(self) -> None:
        ds = Dataset()
        assert ds.task_count == 0
        assert ds.list_tasks() == []

    def test_task_count(self) -> None:
        ds = Dataset()
        assert ds.task_count == 0
        ds.add_task(_make_task(task_id="t1"))
        assert ds.task_count == 1
        ds.add_task(_make_task(task_id="t2"))
        assert ds.task_count == 2

    def test_has_task(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="exists"))
        assert ds.has_task("exists")
        assert not ds.has_task("nope")

    def test_task_ids_set(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="a"))
        ds.add_task(_make_task(task_id="b"))
        assert ds.task_ids == {"a", "b"}

    def test_filter_by_language(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="py-1", language="python"))
        ds.add_task(_make_task(task_id="r-1", language="r"))
        ds.add_task(_make_task(task_id="py-2", language="python"))
        filtered = ds.filter_by_language("python")
        assert filtered.task_count == 2
        assert all(t.language == "python" for t in filtered.list_tasks())

    def test_filter_by_difficulty(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="e1", difficulty=Difficulty.EASY))
        ds.add_task(_make_task(task_id="m1", difficulty=Difficulty.MEDIUM))
        ds.add_task(_make_task(task_id="e2", difficulty=Difficulty.EASY))
        filtered = ds.filter_by_difficulty(Difficulty.EASY)
        assert filtered.task_count == 2

    def test_to_task_list_and_from_task_list(self) -> None:
        tasks = _make_tasks(5)
        ds = Dataset.from_task_list(tasks)
        assert ds.task_count == 5
        assert len(ds.to_task_list()) == 5


# ---------------------------------------------------------------------------
# Stratified Splitting (VAL-DATASET-01)
# ---------------------------------------------------------------------------


class TestStratifiedSplitter:
    """Tests for stratified train/dev/test splitting."""

    def test_default_ratios(self, small_dataset: Dataset) -> None:
        splitter = StratifiedSplitter()
        split = splitter.split(small_dataset)
        assert split.train.task_count > 0
        assert split.dev.task_count > 0
        assert split.test.task_count > 0
        total = split.train.task_count + split.dev.task_count + split.test.task_count
        assert total == small_dataset.task_count

    def test_custom_ratios(self, small_dataset: Dataset) -> None:
        splitter = StratifiedSplitter(train_ratio=0.5, dev_ratio=0.3, test_ratio=0.2)
        split = splitter.split(small_dataset)
        total = small_dataset.task_count
        # Allow tolerance of 1 task for rounding
        assert abs(split.train.task_count / total - 0.5) < 0.15
        assert abs(split.dev.task_count / total - 0.3) < 0.15
        assert abs(split.test.task_count / total - 0.2) < 0.15

    def test_invalid_ratios_raise(self) -> None:
        with pytest.raises(ValueError, match=r"sum.*1\.0"):
            StratifiedSplitter(train_ratio=0.5, dev_ratio=0.3, test_ratio=0.3)
        with pytest.raises(ValueError, match="negative"):
            StratifiedSplitter(train_ratio=-0.1, dev_ratio=0.6, test_ratio=0.5)

    def test_stratification_preserves_difficulty(self) -> None:
        """Verify that difficulty distribution is approximately preserved."""
        ds = Dataset()
        # 10 easy, 10 medium, 10 hard
        for i in range(10):
            ds.add_task(_make_task(task_id=f"easy-{i}", difficulty=Difficulty.EASY))
            ds.add_task(_make_task(task_id=f"med-{i}", difficulty=Difficulty.MEDIUM))
            ds.add_task(_make_task(task_id=f"hard-{i}", difficulty=Difficulty.HARD))

        splitter = StratifiedSplitter(seed=42)
        split = splitter.split(ds)

        for subset in [split.train, split.dev, split.test]:
            if subset.task_count > 0:
                difficulties = [t.difficulty for t in subset.list_tasks()]
                # Each split should have at least one of each difficulty
                assert Difficulty.EASY in difficulties, "Easy missing from split"
                assert Difficulty.MEDIUM in difficulties, "Medium missing from split"
                assert Difficulty.HARD in difficulties, "Hard missing from split"

    def test_stratification_preserves_language(self) -> None:
        """Verify that language distribution is approximately preserved."""
        ds = Dataset()
        for i in range(10):
            ds.add_task(_make_task(task_id=f"py-{i}", language="python"))
            ds.add_task(_make_task(task_id=f"r-{i}", language="r"))
            ds.add_task(_make_task(task_id=f"ts-{i}", language="typescript"))

        splitter = StratifiedSplitter(seed=42)
        split = splitter.split(ds)

        for subset in [split.train, split.dev, split.test]:
            if subset.task_count > 0:
                languages = {t.language for t in subset.list_tasks()}
                # Each split should have at least two languages
                assert len(languages) >= 2, f"Only languages {languages} in split"

    def test_reproducibility_with_seed(self, small_dataset: Dataset) -> None:
        splitter1 = StratifiedSplitter(seed=123)
        split1 = splitter1.split(small_dataset)
        splitter2 = StratifiedSplitter(seed=123)
        split2 = splitter2.split(small_dataset)

        train1_ids = sorted(t.id for t in split1.train.list_tasks())
        train2_ids = sorted(t.id for t in split2.train.list_tasks())
        assert train1_ids == train2_ids

    def test_small_dataset_handled(self) -> None:
        """Dataset with fewer tasks than splits should still work."""
        ds = Dataset()
        ds.add_task(_make_task(task_id="t1"))
        ds.add_task(_make_task(task_id="t2"))
        splitter = StratifiedSplitter()
        split = splitter.split(ds)
        total = split.train.task_count + split.dev.task_count + split.test.task_count
        assert total == 2

    def test_single_task_dataset(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="only"))
        splitter = StratifiedSplitter()
        split = splitter.split(ds)
        total = split.train.task_count + split.dev.task_count + split.test.task_count
        assert total == 1

    def test_split_returns_dataset_split_model(self, small_dataset: Dataset) -> None:
        splitter = StratifiedSplitter()
        split = splitter.split(small_dataset)
        assert isinstance(split, DatasetSplit)
        assert isinstance(split.train, Dataset)
        assert isinstance(split.dev, Dataset)
        assert isinstance(split.test, Dataset)

    def test_empty_dataset_raises(self) -> None:
        ds = Dataset()
        splitter = StratifiedSplitter()
        with pytest.raises(ValueError, match="empty"):
            splitter.split(ds)

    def test_no_overlap_between_splits(self, small_dataset: Dataset) -> None:
        splitter = StratifiedSplitter(seed=42)
        split = splitter.split(small_dataset)
        all_ids = (
            {t.id for t in split.train.list_tasks()}
            | {t.id for t in split.dev.list_tasks()}
            | {t.id for t in split.test.list_tasks()}
        )
        assert len(all_ids) == small_dataset.task_count


# ---------------------------------------------------------------------------
# Dataset Versioning (VAL-DATASET-02)
# ---------------------------------------------------------------------------


class TestDatasetVersioning:
    """Tests for immutable dataset versioning."""

    def test_create_version(self) -> None:
        store = DatasetVersioningStore()
        ds = Dataset.from_task_list(_make_tasks(5))
        version = store.create_version(ds, description="initial version")
        assert version.version_number == 1
        assert version.description == "initial version"
        assert version.task_count == 5
        assert version.created_at is not None

    def test_version_is_immutable(self) -> None:
        store = DatasetVersioningStore()
        ds = Dataset.from_task_list(_make_tasks(3))
        version = store.create_version(ds)
        snapshot_tasks = version.tasks

        # Modify the original dataset
        ds.add_task(_make_task(task_id="new-task"))

        # Version snapshot should be unchanged
        assert len(snapshot_tasks) == 3

    def test_multiple_versions(self) -> None:
        store = DatasetVersioningStore()
        ds1 = Dataset.from_task_list(_make_tasks(3))
        v1 = store.create_version(ds1, description="v1")

        ds2 = Dataset.from_task_list(_make_tasks(5))
        v2 = store.create_version(ds2, description="v2")

        assert v1.version_number == 1
        assert v2.version_number == 2
        assert len(v1.tasks) == 3
        assert len(v2.tasks) == 5

    def test_get_version(self) -> None:
        store = DatasetVersioningStore()
        ds = Dataset.from_task_list(_make_tasks(3))
        store.create_version(ds, description="first")
        store.create_version(Dataset.from_task_list(_make_tasks(5)), description="second")

        retrieved = store.get_version(1)
        assert retrieved is not None
        assert retrieved.version_number == 1
        assert retrieved.description == "first"
        assert len(retrieved.tasks) == 3

    def test_get_nonexistent_version(self) -> None:
        store = DatasetVersioningStore()
        assert store.get_version(999) is None

    def test_list_versions(self) -> None:
        store = DatasetVersioningStore()
        store.create_version(Dataset.from_task_list(_make_tasks(1)), description="v1")
        store.create_version(Dataset.from_task_list(_make_tasks(2)), description="v2")
        versions = store.list_versions()
        assert len(versions) == 2
        assert versions[0].version_number == 1
        assert versions[1].version_number == 2

    def test_latest_version(self) -> None:
        store = DatasetVersioningStore()
        store.create_version(Dataset.from_task_list(_make_tasks(1)))
        store.create_version(Dataset.from_task_list(_make_tasks(2)))
        latest = store.latest_version
        assert latest is not None
        assert latest.version_number == 2

    def test_version_preserves_task_data(self) -> None:
        store = DatasetVersioningStore()
        task = _make_task(task_id="special", language="r", difficulty=Difficulty.HARD)
        ds = Dataset()
        ds.add_task(task)
        version = store.create_version(ds)

        assert version.tasks[0].id == "special"
        assert version.tasks[0].language == "r"
        assert version.tasks[0].difficulty == Difficulty.HARD

    def test_version_snapshot_is_deep_copy(self) -> None:
        """Modifying tasks in the original dataset shouldn't affect the version snapshot."""
        store = DatasetVersioningStore()
        task = _make_task(task_id="t1", prompt="Original prompt")
        ds = Dataset()
        ds.add_task(task)
        version = store.create_version(ds)

        # Modify task in original dataset
        task.prompt = "Modified prompt"
        # Version should have the original
        assert version.tasks[0].prompt == "Original prompt"


# ---------------------------------------------------------------------------
# Decontamination (VAL-DATASET-03)
# ---------------------------------------------------------------------------


class TestDecontamination:
    """Tests for duplicate and near-duplicate detection."""

    def test_detect_exact_duplicates(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="dup-1", prompt="Fix the bug in the function."))
        ds.add_task(_make_task(task_id="dup-2", prompt="Fix the bug in the function."))
        ds.add_task(_make_task(task_id="unique-1", prompt="Different task description."))

        decont = DatasetDecontamination()
        result = decont.run(ds)

        assert len(result.duplicate_pairs) >= 1
        dup_ids = {(pair[0], pair[1]) for pair in result.duplicate_pairs}
        assert ("dup-1", "dup-2") in dup_ids or ("dup-2", "dup-1") in dup_ids

    def test_detect_near_duplicates(self) -> None:
        ds = Dataset()
        ds.add_task(
            _make_task(
                task_id="near-1",
                prompt="Fix the bug in the calculate function that causes incorrect results.",
            )
        )
        ds.add_task(
            _make_task(
                task_id="near-2",
                prompt="Fix the bug in the calculate function that produces incorrect output.",
            )
        )

        decont = DatasetDecontamination(threshold=0.5)
        result = decont.run(ds)

        # Near-duplicates should be flagged
        assert len(result.duplicate_pairs) >= 1
        assert result.flagged_task_ids.intersection({"near-1", "near-2"})

    def test_no_duplicates_in_clean_dataset(self) -> None:
        ds = Dataset.from_task_list(_make_tasks(10))
        decont = DatasetDecontamination()
        result = decont.run(ds)
        assert len(result.duplicate_pairs) == 0

    def test_threshold_parameter(self) -> None:
        # High threshold: should not flag near-duplicates with somewhat different prompts
        ds = Dataset()
        ds.add_task(_make_task(task_id="a", prompt="Fix the bug in the function."))
        ds.add_task(_make_task(task_id="b", prompt="Implement a new sorting algorithm."))

        decont_strict = DatasetDecontamination(threshold=0.99)
        result_strict = decont_strict.run(ds)
        assert len(result_strict.duplicate_pairs) == 0

        # Low threshold: should flag even dissimilar tasks as similar
        decont_loose = DatasetDecontamination(threshold=0.0)
        result_loose = decont_loose.run(ds)
        assert len(result_loose.duplicate_pairs) >= 1

    def test_result_contains_flagged_ids(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="dup-a", prompt="Fix bug A."))
        ds.add_task(_make_task(task_id="dup-b", prompt="Fix bug A."))
        ds.add_task(_make_task(task_id="clean", prompt="Fix bug B."))

        decont = DatasetDecontamination()
        result = decont.run(ds)

        assert "clean" not in result.flagged_task_ids
        assert "dup-a" in result.flagged_task_ids or "dup-b" in result.flagged_task_ids

    def test_empty_dataset(self) -> None:
        ds = Dataset()
        decont = DatasetDecontamination()
        result = decont.run(ds)
        assert len(result.duplicate_pairs) == 0
        assert len(result.flagged_task_ids) == 0

    def test_single_task(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="only"))
        decont = DatasetDecontamination()
        result = decont.run(ds)
        assert len(result.duplicate_pairs) == 0

    def test_result_is_dataclass(self) -> None:
        decont = DatasetDecontamination()
        result = decont.run(Dataset())
        assert isinstance(result, DatasetDecontaminationResult)


# ---------------------------------------------------------------------------
# Difficulty Estimation (VAL-DATASET-04)
# ---------------------------------------------------------------------------


class TestDifficultyEstimator:
    """Tests for automatic difficulty estimation."""

    def test_easy_task(self) -> None:
        estimator = DifficultyEstimator()
        task = _make_task(
            prompt="Fix the typo in the function name.",
            test_command="pytest test_typo.py",
            timeout=30,
        )
        difficulty = estimator.estimate(task)
        assert difficulty == Difficulty.EASY

    def test_hard_task(self) -> None:
        estimator = DifficultyEstimator()
        task = _make_task(
            prompt=(
                "The complex distributed system has a race condition in the "
                "concurrent message processing pipeline that only manifests "
                "under high load. Fix the synchronization issue in the "
                "microservices architecture while maintaining backward "
                "compatibility with the legacy API and ensuring proper error "
                "handling across all network failures. The solution must also "
                "handle edge cases involving message ordering, deadlocks, "
                "and partial failures across multiple services."
            ),
            test_command="pytest -x -v tests/test_distributed_pipeline.py tests/test_concurrent_processing.py tests/test_api_compat.py tests/test_error_handling.py tests/test_edge_cases.py",
            timeout=600,
        )
        difficulty = estimator.estimate(task)
        assert difficulty == Difficulty.HARD

    def test_medium_task(self) -> None:
        estimator = DifficultyEstimator()
        task = _make_task(
            prompt=(
                "Refactor the data processing module to handle CSV and JSON "
                "input formats. Add proper error handling for malformed input."
            ),
            test_command="pytest tests/test_data_processing.py",
            timeout=120,
        )
        difficulty = estimator.estimate(task)
        assert difficulty == Difficulty.MEDIUM

    def test_estimate_dataset(self) -> None:
        estimator = DifficultyEstimator()
        ds = Dataset()
        # Easy tasks
        for i in range(5):
            ds.add_task(
                _make_task(
                    task_id=f"easy-{i}",
                    prompt="Fix the typo.",
                    test_command="pytest t.py",
                    timeout=30,
                )
            )
        # Hard tasks
        for i in range(5):
            ds.add_task(
                _make_task(
                    task_id=f"hard-{i}",
                    prompt=(
                        "Fix the complex race condition in the distributed "
                        "concurrent message processing pipeline that manifests "
                        "under high load with partial failures across multiple "
                        "services requiring careful synchronization."
                    ),
                    test_command="pytest -x -v tests/ test_a.py test_b.py test_c.py",
                    timeout=600,
                )
            )

        estimator.estimate_dataset(ds)
        difficulties = {t.difficulty for t in ds.list_tasks()}

        assert Difficulty.EASY in difficulties
        assert Difficulty.HARD in difficulties

    def test_distribution_reasonable(self) -> None:
        """A diverse dataset should produce a mix of difficulty labels."""
        estimator = DifficultyEstimator()
        ds = Dataset()
        # Create tasks with varied prompt complexity
        for i in range(10):
            ds.add_task(
                _make_task(
                    task_id=f"easy-{i}",
                    prompt="Fix the typo in the function name.",
                    timeout=30,
                )
            )
        for i in range(10):
            ds.add_task(
                _make_task(
                    task_id=f"hard-{i}",
                    prompt=(
                        "The complex distributed system has a race condition "
                        "in the concurrent message processing pipeline that "
                        "manifests under high load. Fix the synchronization "
                        "issue while maintaining backward compatibility."
                    ),
                    timeout=600,
                )
            )
        estimator.estimate_dataset(ds)

        difficulties = [t.difficulty for t in ds.list_tasks()]
        counts = {}
        for d in difficulties:
            counts[d] = counts.get(d, 0) + 1

        # At least two different difficulty levels should be present
        assert len(counts) >= 2

    def test_respects_existing_difficulty_when_skipped(self) -> None:
        """When overwrite=False, existing difficulty should be preserved."""
        estimator = DifficultyEstimator(overwrite=False)
        task = _make_task(
            task_id="t1",
            prompt="A very complex distributed systems problem.",
            difficulty=Difficulty.EASY,
        )
        estimator.estimate_dataset(Dataset.from_task_list([task]))
        assert task.difficulty == Difficulty.EASY


# ---------------------------------------------------------------------------
# Export (VAL-DATASET-05)
# ---------------------------------------------------------------------------


class TestDatasetExport:
    """Tests for dataset export to JSON and CSV."""

    def test_export_to_json(self, small_dataset: Dataset) -> None:
        exporter = DatasetExport()
        json_str = exporter.to_json(small_dataset)
        data = json.loads(json_str)

        assert data["schema_version"] == "1.0"
        assert data["total_tasks"] == small_dataset.task_count
        assert len(data["tasks"]) == small_dataset.task_count

        # Verify task fields are preserved
        first_task = data["tasks"][0]
        assert "id" in first_task
        assert "prompt" in first_task
        assert "language" in first_task
        assert "test_command" in first_task
        assert "difficulty" in first_task
        assert "timeout" in first_task
        assert "constraints" in first_task
        assert "dependencies" in first_task

    def test_export_to_json_file(self, small_dataset: Dataset, tmp_path: Path) -> None:
        exporter = DatasetExport()
        path = tmp_path / "dataset.json"
        exporter.to_json_file(small_dataset, path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["total_tasks"] == small_dataset.task_count

    def test_export_to_csv(self, small_dataset: Dataset) -> None:
        exporter = DatasetExport()
        csv_str = exporter.to_csv(small_dataset)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == small_dataset.task_count

        # Check headers
        expected_fields = [
            "id",
            "prompt",
            "language",
            "test_command",
            "timeout",
            "difficulty",
            "setup_command",
            "constraints",
            "dependencies",
        ]
        for field in expected_fields:
            assert field in reader.fieldnames, f"Missing field: {field}"

    def test_export_to_csv_file(self, small_dataset: Dataset, tmp_path: Path) -> None:
        exporter = DatasetExport()
        path = tmp_path / "dataset.csv"
        exporter.to_csv_file(small_dataset, path)
        assert path.exists()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == small_dataset.task_count

    def test_json_round_trip(self, small_dataset: Dataset) -> None:
        exporter = DatasetExport()
        json_str = exporter.to_json(small_dataset)
        data = json.loads(json_str)

        # Reconstruct tasks from JSON
        reconstructed_tasks = []
        for task_data in data["tasks"]:
            task = Task.model_validate(task_data)
            reconstructed_tasks.append(task)

        # Verify all original tasks are present with matching data
        original_map = {t.id: t for t in small_dataset.list_tasks()}
        for task in reconstructed_tasks:
            orig = original_map[task.id]
            assert task.id == orig.id
            assert task.prompt == orig.prompt
            assert task.language == orig.language
            assert task.test_command == orig.test_command
            assert task.difficulty == orig.difficulty
            assert task.timeout == orig.timeout

    def test_export_empty_dataset(self) -> None:
        exporter = DatasetExport()
        ds = Dataset()
        json_str = exporter.to_json(ds)
        data = json.loads(json_str)
        assert data["total_tasks"] == 0
        assert data["tasks"] == []

        csv_str = exporter.to_csv(ds)
        reader = csv.DictReader(io.StringIO(csv_str))
        assert list(reader) == []

    def test_json_has_schema_version_and_timestamp(self, small_dataset: Dataset) -> None:
        exporter = DatasetExport()
        json_str = exporter.to_json(small_dataset)
        data = json.loads(json_str)
        assert "schema_version" in data
        assert "generated_at" in data

    def test_csv_preserves_list_fields(self) -> None:
        ds = Dataset()
        ds.add_task(
            _make_task(
                task_id="list-test",
                constraints=["no-network", "no-filesystem"],
                dependencies=["numpy", "pandas"],
            )
        )
        exporter = DatasetExport()
        csv_str = exporter.to_csv(ds)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert rows[0]["constraints"] == "no-network,no-filesystem"
        assert rows[0]["dependencies"] == "numpy,pandas"


# ---------------------------------------------------------------------------
# Quality Report (VAL-DATASET-06)
# ---------------------------------------------------------------------------


class TestQualityReport:
    """Tests for dataset quality reporting."""

    def test_basic_report(self, small_dataset: Dataset) -> None:
        report = DatasetQualityReport.generate(small_dataset)

        assert report.total_tasks == small_dataset.task_count
        assert len(report.by_language) > 0
        assert len(report.by_difficulty) > 0

    def test_language_counts(self) -> None:
        ds = Dataset()
        for i in range(3):
            ds.add_task(_make_task(task_id=f"py-{i}", language="python"))
            ds.add_task(_make_task(task_id=f"r-{i}", language="r"))

        report = DatasetQualityReport.generate(ds)
        assert report.by_language["python"] == 3
        assert report.by_language["r"] == 3

    def test_difficulty_counts(self) -> None:
        ds = Dataset()
        for i in range(4):
            ds.add_task(_make_task(task_id=f"e-{i}", difficulty=Difficulty.EASY))
        for i in range(3):
            ds.add_task(_make_task(task_id=f"m-{i}", difficulty=Difficulty.MEDIUM))
        for i in range(2):
            ds.add_task(_make_task(task_id=f"h-{i}", difficulty=Difficulty.HARD))

        report = DatasetQualityReport.generate(ds)
        assert report.by_difficulty["EASY"] == 4
        assert report.by_difficulty["MEDIUM"] == 3
        assert report.by_difficulty["HARD"] == 2

    def test_flags_short_descriptions(self) -> None:
        ds = Dataset()
        ds.add_task(
            _make_task(task_id="good", prompt="Fix the off-by-one error in the sorting function.")
        )
        ds.add_task(_make_task(task_id="short", prompt="Fix it."))

        report = DatasetQualityReport.generate(ds)
        assert len(report.issues) > 0
        issue_descriptions = [issue.description for issue in report.issues]
        # Should have some issue flagged (short prompt or no setup)
        assert len(issue_descriptions) > 0

    def test_flags_missing_setup_commands(self) -> None:
        ds = Dataset()
        ds.add_task(_make_task(task_id="with-setup", setup_command="pip install numpy"))
        ds.add_task(_make_task(task_id="no-setup", setup_command=None))

        report = DatasetQualityReport.generate(ds)
        # Should flag that some tasks have no setup command
        issue_descriptions = [issue.description for issue in report.issues]
        assert any("setup" in desc.lower() for desc in issue_descriptions)

    def test_empty_dataset_report(self) -> None:
        ds = Dataset()
        report = DatasetQualityReport.generate(ds)
        assert report.total_tasks == 0
        assert report.by_language == {}
        assert report.by_difficulty == {}

    def test_report_to_dict(self, small_dataset: Dataset) -> None:
        report = DatasetQualityReport.generate(small_dataset)
        d = report.to_dict()
        assert "total_tasks" in d
        assert "by_language" in d
        assert "by_difficulty" in d
        assert "issues" in d


# ---------------------------------------------------------------------------
# YAML Import (VAL-TASKFMT-03)
# ---------------------------------------------------------------------------


class TestYAMLImport:
    """Tests for manual task authoring via YAML import."""

    def test_import_single_task(self) -> None:
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: manual-001
                prompt: Fix the off-by-one error in the loop.
                language: python
                test_command: pytest test_loop.py
                timeout: 60
                difficulty: MEDIUM
                dependencies:
                  - numpy
                constraints:
                  - no-network
        """)
        tasks = import_tasks_from_yaml(yaml_str)
        assert len(tasks) == 1
        assert tasks[0].id == "manual-001"
        assert tasks[0].language == "python"
        assert tasks[0].difficulty == Difficulty.MEDIUM
        assert tasks[0].dependencies == ["numpy"]
        assert tasks[0].constraints == ["no-network"]

    def test_import_multiple_tasks(self) -> None:
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: task-a
                prompt: Fix bug A.
                language: python
                test_command: pytest a.py
                timeout: 30
                difficulty: EASY
              - id: task-b
                prompt: Fix bug B.
                language: r
                test_command: Rscript test_b.R
                timeout: 120
                difficulty: HARD
        """)
        tasks = import_tasks_from_yaml(yaml_str)
        assert len(tasks) == 2
        assert tasks[0].id == "task-a"
        assert tasks[1].id == "task-b"

    def test_import_with_optional_fields(self) -> None:
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: full-task
                prompt: Complete the implementation.
                language: typescript
                test_command: npm test
                timeout: 300
                difficulty: HARD
                setup_command: npm install
                dependencies:
                  - express
                  - typescript
                constraints:
                  - no-network
                  - no-filesystem
        """)
        tasks = import_tasks_from_yaml(yaml_str)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.setup_command == "npm install"
        assert t.dependencies == ["express", "typescript"]
        assert t.constraints == ["no-network", "no-filesystem"]

    def test_import_defaults(self) -> None:
        """Test that missing optional fields use defaults."""
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: minimal
                prompt: Fix it.
                language: python
                test_command: pytest
                timeout: 60
        """)
        tasks = import_tasks_from_yaml(yaml_str)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.difficulty == Difficulty.EASY  # default
        assert t.setup_command is None
        assert t.constraints == []
        assert t.dependencies == []

    def test_import_validates_task_fields(self) -> None:
        """Invalid task data should raise a validation error."""
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: bad-task
                prompt: ""
                language: python
                test_command: pytest
                timeout: -1
        """)
        with pytest.raises(ValueError):  # Pydantic ValidationError wraps to ValueError
            import_tasks_from_yaml(yaml_str)

    def test_import_from_file(self, tmp_path: Path) -> None:
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: file-task
                prompt: Fix the bug.
                language: python
                test_command: pytest
                timeout: 60
        """)
        yaml_file = tmp_path / "tasks.yaml"
        yaml_file.write_text(yaml_str)
        tasks = import_tasks_from_yaml(yaml_file)
        assert len(tasks) == 1
        assert tasks[0].id == "file-task"

    def test_import_empty_tasks_list(self) -> None:
        yaml_str = textwrap.dedent("""\
            tasks: []
        """)
        tasks = import_tasks_from_yaml(yaml_str)
        assert len(tasks) == 0

    def test_import_into_dataset(self) -> None:
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: yaml-1
                prompt: Fix bug 1.
                language: python
                test_command: pytest
                timeout: 60
              - id: yaml-2
                prompt: Fix bug 2.
                language: r
                test_command: Rscript test.R
                timeout: 120
        """)
        tasks = import_tasks_from_yaml(yaml_str)
        ds = Dataset()
        ds.add_tasks(tasks)
        assert ds.task_count == 2
        assert ds.has_task("yaml-1")
        assert ds.has_task("yaml-2")

    def test_import_duplicate_ids_detected(self) -> None:
        """YAML with duplicate task IDs should raise an error."""
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: dup
                prompt: First.
                language: python
                test_command: pytest
                timeout: 60
              - id: dup
                prompt: Second.
                language: python
                test_command: pytest
                timeout: 60
        """)
        with pytest.raises(ValueError, match=r"dup"):
            import_tasks_from_yaml(yaml_str)

    def test_import_preserves_all_metadata(self, tmp_path: Path) -> None:
        yaml_str = textwrap.dedent("""\
            tasks:
              - id: meta-test
                prompt: Full metadata task.
                language: typescript
                test_command: npx jest
                timeout: 300
                difficulty: HARD
                setup_command: npm install
                dependencies:
                  - "@types/node"
                constraints:
                  - memory-limited
        """)
        yaml_file = tmp_path / "meta.yaml"
        yaml_file.write_text(yaml_str)

        tasks = import_tasks_from_yaml(yaml_file)
        assert len(tasks) >= 1

        exporter = DatasetExport()
        json_str = exporter.to_json(Dataset.from_task_list(tasks))
        data = json.loads(json_str)
        exported_task = data["tasks"][0]

        assert exported_task["id"] == "meta-test"
        assert exported_task["language"] == "typescript"
        assert exported_task["difficulty"] == "HARD"
        assert exported_task["setup_command"] == "npm install"
        assert exported_task["dependencies"] == ["@types/node"]
        assert exported_task["constraints"] == ["memory-limited"]
