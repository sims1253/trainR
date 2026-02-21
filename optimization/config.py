"""Configuration for skill optimization."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OptimizationConfig:
    """Configuration for GEPA-based skill optimization."""

    # Paths
    packages_dir: Path = field(default_factory=lambda: Path("packages"))
    tasks_dir: Path = field(default_factory=lambda: Path("tasks"))
    skills_dir: Path = field(default_factory=lambda: Path("skills"))
    output_dir: Path = field(default_factory=lambda: Path("results"))

    # Docker settings
    docker_image: str = "posit-gskill-eval:latest"
    evaluation_timeout: int = 600

    # GEPA settings
    max_metric_calls: int = 100
    population_size: int = 6
    reflection_lm: str = "openai/glm-5"

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        for attr in ["packages_dir", "tasks_dir", "skills_dir", "output_dir"]:
            value = getattr(self, attr)
            if isinstance(value, str):
                setattr(self, attr, Path(value))
