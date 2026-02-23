"""Experiment matrix generation.

Generates the full matrix of experiment combinations:
- Model x Task x Repeat combinations
- Fingerprinting for determinism
- Pre-computation of experiment plan
- A/B Tool comparison matrix for controlled tool experiments
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
    support_profile: str | None = None
    pair_id: str | None = None
    pair_role: str | None = None  # "control" or "treatment"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_index": self.run_index,
            "task": self.task.to_dict(),
            "model": self.model.to_dict(),
            "repeat_index": self.repeat_index,
            "fingerprint": self.fingerprint,
            "support_profile": self.support_profile,
            "pair_id": self.pair_id,
            "pair_role": self.pair_role,
        }


@dataclass
class ExperimentMatrix:
    """
    Complete experiment matrix.

    Pre-computes all runs in the experiment for:
    - Deterministic execution order
    - Fingerprinting for reproducibility
    - Progress tracking
    - Paired experiment support
    """

    config: ExperimentConfig
    tasks: list[TaskSpec] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
    runs: list[ExperimentRun] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""
    tasks_hash: str = ""
    models_hash: str = ""
    # Pairing metadata
    pairing_enabled: bool = False
    support_profiles: list[str] = field(default_factory=list)

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
            "pairing_enabled": self.pairing_enabled,
            "support_profiles": self.support_profiles,
            "tasks": [t.to_dict() for t in self.tasks],
            "models": [m.to_dict() for m in self.models],
            "runs": [r.to_dict() for r in self.runs],
        }

    def compute_fingerprint(
        self,
        task_id: str,
        model_name: str,
        repeat_index: int,
        seed: int | None,
        support_profile: str | None = None,
    ) -> str:
        """Compute a unique fingerprint for a run."""
        data = {
            "task_id": task_id,
            "model": model_name,
            "repeat": repeat_index,
            "seed": seed,
            "support_profile": support_profile,
            "config_hash": self.config_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def compute_pair_id(
        self,
        task_id: str,
        model_name: str,
        repeat_index: int,
        seed: int | None,
    ) -> str:
        """Compute a unique pair ID for linking control/treatment runs."""
        data = {
            "task_id": task_id,
            "model": model_name,
            "repeat": repeat_index,
            "seed": seed,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


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
    Supports paired experiments for fair support profile comparison.
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

    # Check if pairing is enabled
    if config.pairing.is_active():
        matrix.pairing_enabled = True
        matrix.support_profiles = config.pairing.get_all_profiles()
        _generate_paired_runs(matrix, config)
    else:
        _generate_standard_runs(matrix, config)

    return matrix


def _generate_standard_runs(matrix: ExperimentMatrix, config: ExperimentConfig) -> None:
    """Generate standard (non-paired) experiment runs."""
    run_index = 0
    for model_spec in matrix.models:
        for task_spec in matrix.tasks:
            for repeat_index in range(config.execution.repeats):
                fingerprint = matrix.compute_fingerprint(
                    task_id=task_spec.task_id,
                    model_name=model_spec.name,
                    repeat_index=repeat_index,
                    seed=config.determinism.seed,
                    support_profile=None,
                )

                run = ExperimentRun(
                    run_index=run_index,
                    task=task_spec,
                    model=model_spec,
                    repeat_index=repeat_index,
                    fingerprint=fingerprint,
                    support_profile=None,
                    pair_id=None,
                    pair_role=None,
                )
                matrix.runs.append(run)
                run_index += 1


def _generate_paired_runs(matrix: ExperimentMatrix, config: ExperimentConfig) -> None:
    """
    Generate paired experiment runs for support profile comparison.

    For each task/model/repeat combination, generates runs for:
    - Control profile (the baseline)
    - Treatment profiles (the variants to compare)

    All runs in a pair share the same pair_id for linking.
    """
    run_index = 0
    control_profile = config.pairing.control_profile
    treatment_profiles = config.pairing.treatment_profiles

    for model_spec in matrix.models:
        for task_spec in matrix.tasks:
            for repeat_index in range(config.execution.repeats):
                # Compute pair ID (shared by control and treatment runs)
                pair_id = matrix.compute_pair_id(
                    task_id=task_spec.task_id,
                    model_name=model_spec.name,
                    repeat_index=repeat_index,
                    seed=config.determinism.seed,
                )

                # Generate control run
                control_fingerprint = matrix.compute_fingerprint(
                    task_id=task_spec.task_id,
                    model_name=model_spec.name,
                    repeat_index=repeat_index,
                    seed=config.determinism.seed,
                    support_profile=control_profile,
                )

                control_run = ExperimentRun(
                    run_index=run_index,
                    task=task_spec,
                    model=model_spec,
                    repeat_index=repeat_index,
                    fingerprint=control_fingerprint,
                    support_profile=control_profile,
                    pair_id=pair_id,
                    pair_role="control",
                )
                matrix.runs.append(control_run)
                run_index += 1

                # Generate treatment runs
                for treatment_profile in treatment_profiles:
                    treatment_fingerprint = matrix.compute_fingerprint(
                        task_id=task_spec.task_id,
                        model_name=model_spec.name,
                        repeat_index=repeat_index,
                        seed=config.determinism.seed,
                        support_profile=treatment_profile,
                    )

                    treatment_run = ExperimentRun(
                        run_index=run_index,
                        task=task_spec,
                        model=model_spec,
                        repeat_index=repeat_index,
                        fingerprint=treatment_fingerprint,
                        support_profile=treatment_profile,
                        pair_id=pair_id,
                        pair_role="treatment",
                    )
                    matrix.runs.append(treatment_run)
                    run_index += 1


# =============================================================================
# A/B Tool Comparison Matrix
# =============================================================================


@dataclass
class ToolSpec:
    """Specification for a tool profile in A/B experiments."""

    tool_id: str
    name: str
    version: str
    variant: str | None = None
    fingerprint: str = ""
    config_hash: str = ""
    profile_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "version": self.version,
            "variant": self.variant,
            "fingerprint": self.fingerprint,
            "config_hash": self.config_hash,
            "profile_path": self.profile_path,
        }

    def get_full_id(self) -> str:
        """Get full identifier including version and variant."""
        base = f"{self.tool_id}@{self.version}"
        if self.variant:
            return f"{base}:{self.variant}"
        return base


@dataclass
class SupportSpec:
    """Specification for support profile in A/B experiments."""

    profile_id: str
    name: str
    mode: str
    fingerprint: str = ""
    profile_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "mode": self.mode,
            "fingerprint": self.fingerprint,
            "profile_path": self.profile_path,
        }


@dataclass
class ToolABRun:
    """
    A single run in a tool A/B comparison experiment.

    For A/B experiments, all dimensions EXCEPT tool_profile are held constant:
    - Same task
    - Same model
    - Same support profile
    - Same seed
    - Only tool_profile varies between control and treatment
    """

    run_index: int
    pair_id: str  # Unique identifier for this A/B pair
    pair_position: str  # "control" or "treatment"
    task: TaskSpec
    model: ModelSpec
    support: SupportSpec
    tool: ToolSpec
    repeat_index: int
    seed: int
    fingerprint: str

    # Join keys proving only tool differs (same across pairs)
    join_key_task: str
    join_key_model: str
    join_key_support: str
    join_key_seed: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_index": self.run_index,
            "pair_id": self.pair_id,
            "pair_position": self.pair_position,
            "task": self.task.to_dict(),
            "model": self.model.to_dict(),
            "support": self.support.to_dict(),
            "tool": self.tool.to_dict(),
            "repeat_index": self.repeat_index,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "join_keys": {
                "task": self.join_key_task,
                "model": self.join_key_model,
                "support": self.join_key_support,
                "seed": self.join_key_seed,
            },
        }


@dataclass
class ToolABPair:
    """A pair of control/treatment runs for tool A/B comparison."""

    pair_id: str
    control_run: ToolABRun
    treatment_run: ToolABRun
    # Join key summary (proves only tool differs)
    join_key_summary: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair_id": self.pair_id,
            "control_run": self.control_run.to_dict(),
            "treatment_run": self.treatment_run.to_dict(),
            "join_key_summary": self.join_key_summary,
        }


@dataclass
class ToolABMatrix:
    """
    Complete A/B tool comparison experiment matrix.

    Generates paired runs where only tool_profile differs,
    keeping task/model/support/seed constant.
    """

    config: ExperimentConfig
    tasks: list[TaskSpec] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
    supports: list[SupportSpec] = field(default_factory=list)
    control_tools: list[ToolSpec] = field(default_factory=list)
    treatment_tools: list[ToolSpec] = field(default_factory=list)
    runs: list[ToolABRun] = field(default_factory=list)
    pairs: list[ToolABPair] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""

    def __len__(self) -> int:
        """Return total number of runs."""
        return len(self.runs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "total_runs": len(self.runs),
            "total_pairs": len(self.pairs),
            "tasks_count": len(self.tasks),
            "models_count": len(self.models),
            "supports_count": len(self.supports),
            "control_tools_count": len(self.control_tools),
            "treatment_tools_count": len(self.treatment_tools),
            "tasks": [t.to_dict() for t in self.tasks],
            "models": [m.to_dict() for m in self.models],
            "supports": [s.to_dict() for s in self.supports],
            "control_tools": [t.to_dict() for t in self.control_tools],
            "treatment_tools": [t.to_dict() for t in self.treatment_tools],
            "runs": [r.to_dict() for r in self.runs],
            "pairs": [p.to_dict() for p in self.pairs],
        }

    def compute_pair_fingerprint(
        self,
        task_id: str,
        model_name: str,
        support_id: str,
        control_tool_id: str,
        treatment_tool_id: str,
        repeat_index: int,
        seed: int,
    ) -> str:
        """Compute a unique fingerprint for a pair."""
        data = {
            "task_id": task_id,
            "model": model_name,
            "support": support_id,
            "control_tool": control_tool_id,
            "treatment_tool": treatment_tool_id,
            "repeat": repeat_index,
            "seed": seed,
            "config_hash": self.config_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def resolve_ab_tools(
    ab_config: dict[str, Any] | None = None,
) -> tuple[list[ToolSpec], list[ToolSpec]]:
    """
    Resolve control and treatment tool profiles for A/B experiments.

    Args:
        ab_config: A/B specific configuration from settings.ab_tools

    Returns:
        Tuple of (control_tools, treatment_tools)
    """
    from bench.profiles.tools import load_tool_profile

    control_tools: list[ToolSpec] = []
    treatment_tools: list[ToolSpec] = []

    if not ab_config:
        # Default: use default vs strict profiles
        ab_config = {
            "control": ["configs/profiles/tools/default.yaml"],
            "treatment": ["configs/profiles/tools/strict.yaml"],
        }

    # Resolve control tools
    for tool_path in ab_config.get("control", []):
        try:
            path = Path(tool_path)
            if path.exists():
                profile = load_tool_profile(path)
                control_tools.append(
                    ToolSpec(
                        tool_id=profile.tool_id,
                        name=profile.name,
                        version=profile.get_version_string(),
                        variant=profile.variant,
                        fingerprint=profile.fingerprint,
                        config_hash=profile.fingerprint[:8],
                        profile_path=str(path),
                    )
                )
        except Exception:
            # Fallback to tool_id reference
            control_tools.append(
                ToolSpec(
                    tool_id=tool_path,
                    name=tool_path,
                    version="v1",
                    variant=None,
                    fingerprint="",
                    config_hash="",
                    profile_path=tool_path,
                )
            )

    # Resolve treatment tools
    for tool_path in ab_config.get("treatment", []):
        try:
            path = Path(tool_path)
            if path.exists():
                profile = load_tool_profile(path)
                treatment_tools.append(
                    ToolSpec(
                        tool_id=profile.tool_id,
                        name=profile.name,
                        version=profile.get_version_string(),
                        variant=profile.variant,
                        fingerprint=profile.fingerprint,
                        config_hash=profile.fingerprint[:8],
                        profile_path=str(path),
                    )
                )
        except Exception:
            # Fallback to tool_id reference
            treatment_tools.append(
                ToolSpec(
                    tool_id=tool_path,
                    name=tool_path,
                    version="v1",
                    variant=None,
                    fingerprint="",
                    config_hash="",
                    profile_path=tool_path,
                )
            )

    return control_tools, treatment_tools


def resolve_ab_supports(ab_config: dict[str, Any] | None = None) -> list[SupportSpec]:
    """
    Resolve support profiles for A/B experiments.

    Args:
        ab_config: A/B specific configuration

    Returns:
        List of SupportSpec instances
    """
    from bench.profiles.support import load_support_profile

    supports: list[SupportSpec] = []

    if not ab_config or "support_profiles" not in ab_config:
        # Default: use baseline_none support
        ab_config = {"support_profiles": ["configs/profiles/support/baseline_none.yaml"]}

    for support_ref in ab_config.get("support_profiles", []):
        try:
            profile = load_support_profile(support_ref)
            fingerprint = profile.fingerprint.config_hash
            supports.append(
                SupportSpec(
                    profile_id=profile.profile_id,
                    name=profile.name,
                    mode=profile.mode.value,
                    fingerprint=fingerprint,
                    profile_path=support_ref if Path(support_ref).exists() else None,
                )
            )
        except Exception:
            # Fallback
            supports.append(
                SupportSpec(
                    profile_id=support_ref,
                    name=support_ref,
                    mode="none",
                    fingerprint="",
                    profile_path=None,
                )
            )

    return supports


def generate_tool_ab_matrix(config: ExperimentConfig) -> ToolABMatrix:
    """
    Generate an A/B tool comparison experiment matrix.

    Creates paired runs where only the tool_profile differs, keeping
    task/model/support/seed constant across pairs.

    Configuration is read from config.settings.ab_tools:
        settings:
          ab_tools:
            control:
              - "configs/profiles/tools/default.yaml"
            treatment:
              - "configs/profiles/tools/strict.yaml"
            support_profiles:
              - "configs/profiles/support/baseline_none.yaml"

    Args:
        config: Experiment configuration with A/B tool settings

    Returns:
        ToolABMatrix with paired runs
    """
    ab_config = config.settings.get("ab_tools", {})

    matrix = ToolABMatrix(config=config)

    # Resolve all components
    matrix.models = resolve_models(config)
    matrix.tasks = load_tasks(config)
    matrix.supports = resolve_ab_supports(ab_config)
    matrix.control_tools, matrix.treatment_tools = resolve_ab_tools(ab_config)

    # Compute config hash
    matrix.config_hash = compute_hash(
        [
            config.model_dump(),
            ab_config,
        ]
    )

    # Fixed seed for A/B experiments (ensures same randomness across pairs)
    seed = config.determinism.seed or 42

    run_index = 0

    # Generate paired runs
    for model_spec in matrix.models:
        for task_spec in matrix.tasks:
            for support_spec in matrix.supports:
                for repeat_index in range(config.execution.repeats):
                    for control_tool in matrix.control_tools:
                        for treatment_tool in matrix.treatment_tools:
                            # Create pair ID
                            pair_id = matrix.compute_pair_fingerprint(
                                task_id=task_spec.task_id,
                                model_name=model_spec.name,
                                support_id=support_spec.profile_id,
                                control_tool_id=control_tool.get_full_id(),
                                treatment_tool_id=treatment_tool.get_full_id(),
                                repeat_index=repeat_index,
                                seed=seed,
                            )

                            # Join keys (same for both control and treatment)
                            join_key_task = task_spec.task_id
                            join_key_model = model_spec.name
                            join_key_support = support_spec.profile_id
                            join_key_seed = str(seed)

                            # Create control run
                            control_fingerprint = hashlib.sha256(
                                json.dumps(
                                    {
                                        "pair_id": pair_id,
                                        "position": "control",
                                        "tool": control_tool.get_full_id(),
                                    },
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest()[:16]

                            control_run = ToolABRun(
                                run_index=run_index,
                                pair_id=pair_id,
                                pair_position="control",
                                task=task_spec,
                                model=model_spec,
                                support=support_spec,
                                tool=control_tool,
                                repeat_index=repeat_index,
                                seed=seed,
                                fingerprint=control_fingerprint,
                                join_key_task=join_key_task,
                                join_key_model=join_key_model,
                                join_key_support=join_key_support,
                                join_key_seed=join_key_seed,
                            )
                            matrix.runs.append(control_run)
                            run_index += 1

                            # Create treatment run
                            treatment_fingerprint = hashlib.sha256(
                                json.dumps(
                                    {
                                        "pair_id": pair_id,
                                        "position": "treatment",
                                        "tool": treatment_tool.get_full_id(),
                                    },
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest()[:16]

                            treatment_run = ToolABRun(
                                run_index=run_index,
                                pair_id=pair_id,
                                pair_position="treatment",
                                task=task_spec,
                                model=model_spec,
                                support=support_spec,
                                tool=treatment_tool,
                                repeat_index=repeat_index,
                                seed=seed,
                                fingerprint=treatment_fingerprint,
                                join_key_task=join_key_task,
                                join_key_model=join_key_model,
                                join_key_support=join_key_support,
                                join_key_seed=join_key_seed,
                            )
                            matrix.runs.append(treatment_run)
                            run_index += 1

                            # Create pair
                            pair = ToolABPair(
                                pair_id=pair_id,
                                control_run=control_run,
                                treatment_run=treatment_run,
                                join_key_summary={
                                    "task": join_key_task,
                                    "model": join_key_model,
                                    "support": join_key_support,
                                    "seed": join_key_seed,
                                    "control_tool": control_tool.get_full_id(),
                                    "treatment_tool": treatment_tool.get_full_id(),
                                },
                            )
                            matrix.pairs.append(pair)

    return matrix
