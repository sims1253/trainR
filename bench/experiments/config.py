"""Experiment configuration schema.

Defines the ExperimentConfig which specifies:
- Models to evaluate
- Tasks to run
- Profiles/support levels
- Execution settings
- Output configuration

This is the single source of truth for experiment configuration.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from bench.provider import AuthPolicy
from bench.sandbox import SandboxProfile


# Known harness types for execution
HarnessType = Literal[
    "pi_docker",
    "pi_sdk",
    "pi_cli",
    "codex_cli",
    "claude_cli",
    "gemini_cli",
    "swe_agent",
]


class TaskSelectionMode(str, Enum):
    """How to select tasks for the experiment."""

    ALL = "all"
    SPLITS = "splits"
    CATEGORIES = "categories"
    FILES = "files"
    TASK_ID = "task_id"


class RetryStrategy(str, Enum):
    """Retry strategy for failed evaluations."""

    NONE = "none"  # No retries
    FIXED = "fixed"  # Fixed number of retries
    EXPONENTIAL = "exponential"  # Exponential backoff


class TasksConfig(BaseModel):
    """Task selection configuration."""

    selection: TaskSelectionMode = Field(
        default=TaskSelectionMode.SPLITS,
        description="How to select tasks",
    )
    splits: list[str] = Field(
        default_factory=lambda: ["dev"],
        description="Splits to include (train, dev, held_out, mined)",
    )
    categories: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Category filters (task_types, difficulties)",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Specific task files (when selection=files)",
    )
    task_id: str | None = Field(
        default=None,
        description="Single task ID (when selection=task_id)",
    )
    dir: str = Field(
        default="tasks",
        description="Directory containing task JSON files",
    )
    max_tasks: int | None = Field(
        default=None,
        description="Maximum number of tasks to run (for testing)",
    )


class ModelsConfig(BaseModel):
    """Model configuration for the experiment."""

    names: list[str] = Field(
        default_factory=list,
        description="Model names to evaluate (references llm.yaml)",
    )
    provider_fallback: bool = Field(
        default=True,
        description="Allow fallback to alternate providers",
    )


class SkillConfig(BaseModel):
    """Skill configuration."""

    path: str | None = Field(
        default=None,
        description="Path to skill file",
    )
    no_skill: bool = Field(
        default=False,
        description="Disable skill (baseline mode)",
    )
    content: str | None = Field(
        default=None,
        description="Inline skill content (overrides path)",
    )

    def is_active(self) -> bool:
        """Check if skill is active."""
        return not self.no_skill and (self.path is not None or self.content is not None)

    def get_name(self) -> str:
        """Get a readable skill name."""
        if self.no_skill:
            return "no_skill"
        if self.path:
            return Path(self.path).stem
        if self.content:
            return "inline"
        return "none"


class ExecutionConfig(BaseModel):
    """Execution environment configuration.

    Attributes:
        harness: The harness adapter to use for running tasks.
            - pi_docker: Docker-based execution with Pi SDK (default)
            - pi_sdk: Native Pi SDK execution
            - pi_cli: Pi CLI-based execution
            - codex_cli: OpenAI Codex CLI
            - claude_cli: Anthropic Claude CLI
            - gemini_cli: Google Gemini CLI
            - swe_agent: SWE-agent execution
        sandbox_profile: Security sandbox profile for execution.
            - strict: Non-root, readonly FS, no network (default for CI/benchmarks)
            - networked: Explicit outbound network access
            - developer: Relaxed local-debug settings
        timeout: Maximum time per task in seconds
        docker_image: Docker image for evaluation containers
        repeats: Number of times to repeat each task
        parallel_workers: Number of parallel workers
        save_trajectories: Whether to save agent trajectories
        save_traces: Whether to save LLM request/response traces
    """

    harness: HarnessType = Field(
        default="pi_docker",
        description="Harness name: pi_docker, pi_sdk, pi_cli, codex_cli, claude_cli, gemini_cli, swe_agent",
    )
    sandbox_profile: SandboxProfile = Field(
        default=SandboxProfile.STRICT,
        description="Sandbox security profile: strict, networked, or developer",
    )
    timeout: int = Field(
        default=600,
        ge=1,
        description="Timeout per task in seconds",
    )
    docker_image: str = Field(
        default="posit-gskill-eval:latest",
        description="Docker image for evaluation",
    )
    repeats: int = Field(
        default=1,
        ge=1,
        description="Number of times to repeat each task",
    )
    parallel_workers: int = Field(
        default=1,
        ge=1,
        description="Number of parallel workers",
    )
    save_trajectories: bool = Field(
        default=True,
        description="Save agent trajectories",
    )
    save_traces: bool = Field(
        default=False,
        description="Save LLM request/response traces",
    )
    auth_policy: AuthPolicy = Field(
        default=AuthPolicy.ENV,
        description="Authentication policy: env or mounted_file",
    )

    @field_validator("sandbox_profile", mode="before")
    @classmethod
    def validate_sandbox_profile(cls, v: str | SandboxProfile) -> SandboxProfile:
        """Validate and convert sandbox_profile to SandboxProfile enum."""
        if isinstance(v, SandboxProfile):
            return v
        if isinstance(v, str):
            try:
                return SandboxProfile(v.lower())
            except ValueError:
                allowed = [p.value for p in SandboxProfile]
                raise ValueError(f"Invalid sandbox_profile: {v}. Must be one of: {allowed}")
        raise ValueError(f"Invalid sandbox_profile type: {type(v)}")

    @field_validator("auth_policy", mode="before")
    @classmethod
    def validate_auth_policy(cls, v: str | AuthPolicy) -> AuthPolicy:
        """Validate and convert auth_policy to AuthPolicy enum."""
        if isinstance(v, AuthPolicy):
            return v
        if isinstance(v, str):
            try:
                return AuthPolicy(v.lower())
            except ValueError:
                allowed = [p.value for p in AuthPolicy]
                raise ValueError(f"Invalid auth_policy: {v}. Must be one of: {allowed}")
        raise ValueError(f"Invalid auth_policy type: {type(v)}")


class RetryConfig(BaseModel):
    """Retry configuration for failed evaluations."""

    strategy: RetryStrategy = Field(
        default=RetryStrategy.NONE,
        description="Retry strategy",
    )
    max_retries: int = Field(
        default=0,
        ge=0,
        description="Maximum number of retries",
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Base delay in seconds (for exponential backoff)",
    )
    max_delay: float = Field(
        default=60.0,
        ge=0.0,
        description="Maximum delay in seconds",
    )
    retry_on: list[str] = Field(
        default_factory=lambda: ["TIMEOUT", "LLM_ERROR", "SANDBOX_ERROR"],
        description="Error categories to retry on",
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: str = Field(
        default="results/experiments",
        description="Output directory",
    )
    run_id: str | None = Field(
        default=None,
        description="Run ID (auto-generated if not set)",
    )
    save_intermediate: bool = Field(
        default=True,
        description="Save intermediate results after each task",
    )


class DeterminismConfig(BaseModel):
    """Determinism configuration for reproducible runs."""

    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    freeze_model_resolution: bool = Field(
        default=True,
        description="Snapshot model resolution at start",
    )
    freeze_config: bool = Field(
        default=True,
        description="Snapshot config hashes in manifest",
    )


class PairingConfig(BaseModel):
    """Configuration for paired experiment comparisons.

    Paired experiments run the same task/model/tool/seed with different
    support profiles to enable fair comparisons of support effectiveness.
    """

    enabled: bool = Field(
        default=False,
        description="Enable paired experiment mode",
    )
    dimensions: list[str] = Field(
        default_factory=lambda: ["support"],
        description="Dimensions to pair on (currently only 'support' is supported)",
    )
    control_profile: str = Field(
        default="",
        description="Control support profile name or path",
    )
    treatment_profiles: list[str] = Field(
        default_factory=list,
        description="Treatment support profile names or paths to compare against control",
    )

    def is_active(self) -> bool:
        """Check if pairing is enabled and properly configured."""
        return self.enabled and bool(self.control_profile) and bool(self.treatment_profiles)

    def get_all_profiles(self) -> list[str]:
        """Get all profiles (control first, then treatments)."""
        return [self.control_profile, *list(self.treatment_profiles)]


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    This is the single source of truth for experiment configuration,
    used by the unified experiment runner.
    """

    # Metadata
    name: str = Field(
        default="experiment",
        description="Experiment name",
    )
    description: str = Field(
        default="",
        description="Experiment description",
    )
    schema_version: str = Field(
        default="1.0",
        description="Configuration schema version",
    )

    # Components
    tasks: TasksConfig = Field(
        default_factory=TasksConfig,
        description="Task selection configuration",
    )
    models: ModelsConfig = Field(
        default_factory=ModelsConfig,
        description="Model configuration",
    )
    skill: SkillConfig = Field(
        default_factory=SkillConfig,
        description="Skill configuration",
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution settings",
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
    )
    determinism: DeterminismConfig = Field(
        default_factory=DeterminismConfig,
        description="Determinism settings",
    )
    pairing: PairingConfig = Field(
        default_factory=PairingConfig,
        description="Paired experiment configuration",
    )

    # Additional settings
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional experiment settings",
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Ensure schema version is valid."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported schema version: {v}. Expected 1.x")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        tasks_data = data.get("tasks", {})
        if isinstance(tasks_data.get("selection"), str):
            tasks_data["selection"] = TaskSelectionMode(tasks_data["selection"])

        models_data = data.get("models", {})
        if "names" not in models_data and "models" in data:
            # Support top-level 'models' list
            model_names = data.get("models", [])
            if isinstance(model_names, list) and all(isinstance(m, str) for m in model_names):
                models_data["names"] = model_names

        retry_data = data.get("retry", {})
        if isinstance(retry_data.get("strategy"), str):
            retry_data["strategy"] = RetryStrategy(retry_data["strategy"])

        return cls(
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            schema_version=data.get("schema_version", "1.0"),
            tasks=TasksConfig(**tasks_data),
            models=ModelsConfig(**models_data),
            skill=SkillConfig(**data.get("skill", {})),
            execution=ExecutionConfig(**data.get("execution", {})),
            retry=RetryConfig(**retry_data),
            output=OutputConfig(**data.get("output", {})),
            determinism=DeterminismConfig(**data.get("determinism", {})),
            pairing=PairingConfig(**data.get("pairing", {})),
            settings=data.get("settings", {}),
        )

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        if self.output.run_id:
            return self.output.run_id

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        seed_suffix = f"_{self.determinism.seed}" if self.determinism.seed else ""
        return f"{self.name}_{timestamp}{seed_suffix}"

    def get_output_dir(self) -> Path:
        """Get the output directory path."""
        return Path(self.output.dir) / self.generate_run_id()

    def get_skill_content(self) -> str | None:
        """Load skill content from path or use inline content."""
        if self.skill.no_skill:
            return None
        if self.skill.content:
            return self.skill.content
        if self.skill.path:
            skill_path = Path(self.skill.path)
            if skill_path.exists():
                return skill_path.read_text()
        return None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load experiment configuration from file.

    Args:
        path: Path to YAML config file

    Returns:
        ExperimentConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
    """
    return ExperimentConfig.from_yaml(path)
