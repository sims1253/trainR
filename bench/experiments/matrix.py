"""Experiment matrix generation.

Generates the full matrix of experiment combinations:
- Model x Task x Repeat combinations
- Fingerprinting for determinism
- Pre-computation of experiment plan
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.experiments.config import ExperimentConfig


@dataclass
class TaskSpec:
    """Specification for a single task in the experiment matrix."""

    task_id: str
    task_path: str
    source_package: str | None = None
    split: str = "unknown"
    difficulty: str = "unknown"
    task_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_path": self.task_path,
            "source_package": self.source_package,
            "split": self.split,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "metadata": self.metadata,
        }


@dataclass
class ModelSpec:
    """Specification for a model in the experiment matrix."""

    name: str
    litellm_model: str
    provider: str | None = None
    api_key_env: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "litellm_model": self.litellm_model,
            "provider": self.provider,
            "api_key_env": self.api_key_env,
            "capabilities": self.capabilities,
        }


@dataclass
class ExperimentRun:
    """Single run specification in the experiment matrix."""

    run_index: int
    task: TaskSpec
    model: ModelSpec
    repeat_index: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_index": self.run_index,
            "task": self.task.to_dict(),
            "model": self.model.to_dict(),
            "repeat_index": self.repeat_index,
            "fingerprint": self.fingerprint,
        }


@dataclass
class ExperimentMatrix:
    """
    Complete experiment matrix.

    Pre-computes all runs in the experiment for:
    - Deterministic execution order
    - Fingerprinting for reproducibility
    - Progress tracking
    """

    config: ExperimentConfig
    tasks: list[TaskSpec] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
    runs: list[ExperimentRun] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""
    tasks_hash: str = ""
    models_hash: str = ""

    def __len__(self) -> int:
        """Return total number of runs."""
        return len(self.runs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_hash": self.config_hash,
            "tasks_hash": self.tasks_hash,
            "models_hash": self.models_hash,
            "created_at": self.created_at,
            "total_runs": len(self.runs),
            "tasks_count": len(self.tasks),
            "models_count": len(self.models),
            "repeats": self.config.execution.repeats,
            "tasks": [t.to_dict() for t in self.tasks],
            "models": [m.to_dict() for m in self.models],
            "runs": [r.to_dict() for r in self.runs],
        }

    def compute_fingerprint(
        self, task_id: str, model_name: str, repeat_index: int, seed: int | None
    ) -> str:
        """Compute a unique fingerprint for a run."""
        data = {
            "task_id": task_id,
            "model": model_name,
            "repeat": repeat_index,
            "seed": seed,
            "config_hash": self.config_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def resolve_models(config: ExperimentConfig) -> list[ModelSpec]:
    """
    Resolve model names to full specifications.

    Uses the LLM config system to resolve model names to
    their LiteLLM equivalents and provider information.
    """
    from config import get_llm_config

    llm_config = get_llm_config()
    models = []

    for model_name in config.models.names:
        try:
            model_cfg = llm_config.get_model_config(model_name)
            litellm_model = llm_config.get_litellm_model(model_name)

            models.append(
                ModelSpec(
                    name=model_name,
                    litellm_model=litellm_model,
                    provider=model_cfg.get("provider"),
                    api_key_env=model_cfg.get("api_key_env"),
                    capabilities=model_cfg.get("capabilities", {}),
                )
            )
        except ValueError:
            # Model not in config - use as-is (assume it's a valid LiteLLM model)
            models.append(
                ModelSpec(
                    name=model_name,
                    litellm_model=model_name,
                    provider=None,
                    api_key_env=None,
                    capabilities={},
                )
            )

    return models


def load_tasks(config: ExperimentConfig) -> list[TaskSpec]:
    """
    Load tasks based on configuration.

    Supports multiple selection modes:
    - all: Load all tasks
    - splits: Load tasks from specific splits
    - categories: Filter by task metadata
    - files: Load specific task files
    - task_id: Load single task by ID
    """
    from task_generator import TaskGenerator

    tasks_dir = Path(config.tasks.dir)
    generator = TaskGenerator(tasks_dir)
    task_specs = []

    # Load tasks based on selection mode
    if config.tasks.selection.value == "all":
        raw_tasks = generator.load_all_tasks()
    elif config.tasks.selection.value == "splits":
        raw_tasks = []
        for split in config.tasks.splits:
            raw_tasks.extend(generator.load_all_tasks(split=split))
    elif config.tasks.selection.value == "categories":
        # Load all and filter
        all_tasks = generator.load_all_tasks()
        categories = config.tasks.categories
        task_types = set(categories.get("task_types", []))
        difficulties = set(categories.get("difficulties", []))

        raw_tasks = []
        for task in all_tasks:
            # Check task_type if specified
            if task_types:
                task_type = getattr(task, "task_type", None) or getattr(
                    getattr(task, "metadata", {}), "task_type", None
                )
                if task_type and str(task_type) not in task_types:
                    continue

            # Check difficulty if specified
            if difficulties:
                difficulty = getattr(task, "difficulty", None) or getattr(
                    getattr(task, "metadata", {}), "difficulty", None
                )
                if difficulty and str(difficulty) not in difficulties:
                    continue

            raw_tasks.append(task)
    elif config.tasks.selection.value == "files":
        raw_tasks = []
        for file_path in config.tasks.files:
            path = Path(file_path)
            if not path.is_absolute():
                path = tasks_dir / path
            if path.exists():
                data = json.loads(path.read_text())
                # Create a task-like object
                task = type(
                    "Task",
                    (),
                    {
                        "task_id": data.get("task_id", path.stem),
                        "instruction": data.get("task", {}).get("instruction", ""),
                        "context": data.get("task", {}).get("context", ""),
                        "source_package": data.get("source_package", ""),
                        "split": data.get("split", "custom"),
                        "difficulty": data.get("difficulty", "unknown"),
                        "task_type": data.get("task_type", "unknown"),
                        **data,
                    },
                )()
                raw_tasks.append(task)
    elif config.tasks.selection.value == "task_id":
        raw_tasks = []
        task_id = config.tasks.task_id
        if task_id:
            # Search for task file by ID
            for task_file in tasks_dir.glob("**/*.json"):
                data = json.loads(task_file.read_text())
                if data.get("task_id") == task_id:
                    task = type(
                        "Task",
                        (),
                        {
                            "task_id": data.get("task_id", task_file.stem),
                            "instruction": data.get("task", {}).get("instruction", ""),
                            "context": data.get("task", {}).get("context", ""),
                            "source_package": data.get("source_package", ""),
                            "split": data.get("split", "custom"),
                            "difficulty": data.get("difficulty", "unknown"),
                            "task_type": data.get("task_type", "unknown"),
                            **data,
                        },
                    )()
                    raw_tasks.append(task)
                    break
    else:
        raw_tasks = []

    # Convert to TaskSpecs
    for task in raw_tasks:
        task_spec = TaskSpec(
            task_id=task.task_id,
            task_path=str(getattr(task, "task_path", f"{tasks_dir}/{task.task_id}.json")),
            source_package=getattr(task, "source_package", None),
            split=getattr(task, "split", "unknown"),
            difficulty=str(getattr(task, "difficulty", "unknown")),
            task_type=str(getattr(task, "task_type", "unknown")),
            metadata={},
        )
        task_specs.append(task_spec)

    # Apply max_tasks limit
    if config.tasks.max_tasks is not None:
        task_specs = task_specs[: config.tasks.max_tasks]

    return task_specs


def compute_hash(items: list[Any]) -> str:
    """Compute a hash of a list of items."""
    content = json.dumps([str(i) for i in items], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_matrix(config: ExperimentConfig) -> ExperimentMatrix:
    """
    Generate the complete experiment matrix.

    Creates a pre-computed matrix of all runs for deterministic execution.
    """
    matrix = ExperimentMatrix(config=config)

    # Resolve models
    matrix.models = resolve_models(config)

    # Load tasks
    matrix.tasks = load_tasks(config)

    # Compute hashes for fingerprinting
    matrix.config_hash = compute_hash([config.model_dump()])
    matrix.tasks_hash = compute_hash([t.to_dict() for t in matrix.tasks])
    matrix.models_hash = compute_hash([m.to_dict() for m in matrix.models])

    # Generate all runs
    run_index = 0
    for model_spec in matrix.models:
        for task_spec in matrix.tasks:
            for repeat_index in range(config.execution.repeats):
                fingerprint = matrix.compute_fingerprint(
                    task_id=task_spec.task_id,
                    model_name=model_spec.name,
                    repeat_index=repeat_index,
                    seed=config.determinism.seed,
                )

                run = ExperimentRun(
                    run_index=run_index,
                    task=task_spec,
                    model=model_spec,
                    repeat_index=repeat_index,
                    fingerprint=fingerprint,
                )
                matrix.runs.append(run)
                run_index += 1

    return matrix
