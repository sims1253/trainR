"""Configuration for evaluation runs via YAML files."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model selection configuration."""

    task: str = "glm-4.5"
    reflection: str = "glm-5"
    # Support multiple models for batch comparison
    tasks: list[str] | None = None  # If set, run all these models

    def get_models(self) -> list[str]:
        """Get list of models to evaluate."""
        if self.tasks:
            return self.tasks
        return [self.task]


@dataclass
class WorkersConfig:
    """Parallelism configuration."""

    count: int = 5
    auto_detect: bool = False


@dataclass
class SkillConfig:
    """Skill configuration."""

    path: str | None = None
    no_skill: bool = False


@dataclass
class TasksConfig:
    """Task selection configuration."""

    dir: str = "tasks"
    splits: list[str] = field(default_factory=lambda: ["dev", "held_out"])
    difficulty: list[str] | None = None
    packages: list[str] | None = None


@dataclass
class ExecutionConfig:
    """Execution settings."""

    timeout: int = 600
    docker_image: str = "posit-gskill-eval:latest"
    repeats: int = 1


@dataclass
class OutputConfig:
    """Output settings."""

    dir: str = "results/evaluations"
    save_trajectories: bool = True
    filename: str = "eval_{model}_{skill_name}_{timestamp}.json"


@dataclass
class EvaluationConfig:
    """Top-level evaluation configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationConfig":
        """Create configuration from dictionary."""
        model = ModelConfig(**data.get("model", {}))
        workers = WorkersConfig(**data.get("workers", {}))
        skill = SkillConfig(**data.get("skill", {}))
        tasks_data = data.get("tasks", {})
        tasks = TasksConfig(
            dir=tasks_data.get("dir", "tasks"),
            splits=tasks_data.get("splits", ["dev", "held_out"]),
            difficulty=tasks_data.get("difficulty"),
            packages=tasks_data.get("packages"),
        )
        execution = ExecutionConfig(**data.get("execution", {}))
        output = OutputConfig(**data.get("output", {}))

        return cls(
            model=model,
            workers=workers,
            skill=skill,
            tasks=tasks,
            execution=execution,
            output=output,
        )

    def get_skill_name(self) -> str:
        """Get a readable skill name for output files."""
        if self.skill.no_skill:
            return "no_skill"
        if self.skill.path:
            return Path(self.skill.path).stem
        return "unknown"
