"""Profile schemas for agent/tool support configurations.

Profiles define:
- Agent capabilities (models, skills, prompts)
- Tool configurations (sandbox, execution environment)
- Evaluation settings (judges, scoring)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProfileTypeV1(str, Enum):
    """Types of profiles."""

    AGENT = "agent"  # Agent capability profile
    TOOL = "tool"  # Tool/sandbox configuration
    EVALUATION = "evaluation"  # Evaluation settings


class JudgeModeV1(str, Enum):
    """Judge evaluation modes."""

    SINGLE = "single"  # Single judge
    ENSEMBLE = "ensemble"  # Multiple judges with voting


class VotingStrategyV1(str, Enum):
    """Voting strategies for ensemble judges."""

    MAJORITY = "majority"
    AVERAGE = "average"
    WEIGHTED = "weighted"


class ModelCapabilityV1(BaseModel):
    """Capabilities of a model."""

    reasoning: bool = Field(default=False, description="Supports reasoning output")
    json_mode: str = Field(default="prompt", description="JSON mode: native, prompt, or none")
    max_tokens_default: int = Field(default=2000, description="Default max tokens")
    drop_params: list[str] = Field(default_factory=list, description="Params not supported")


class ModelConfigV1(BaseModel):
    """Configuration for a model reference."""

    name: str = Field(description="Model name as defined in llm.yaml")
    provider: str | None = Field(default=None, description="Provider name")
    model_id: str | None = Field(default=None, description="Full model ID for API calls")
    capabilities: ModelCapabilityV1 | None = Field(default=None, description="Model capabilities")


class JudgeConfigV1(BaseModel):
    """Judge configuration for evaluation."""

    mode: JudgeModeV1 = Field(default=JudgeModeV1.SINGLE, description="Judge mode")
    single: str | None = Field(default=None, description="Single judge model name")
    ensemble_judges: list[str] = Field(
        default_factory=list, description="Ensemble judge model names"
    )
    voting: VotingStrategyV1 = Field(
        default=VotingStrategyV1.MAJORITY, description="Voting strategy"
    )

    def get_judge_models(self) -> list[str]:
        """Get list of judge model names."""
        if self.mode == JudgeModeV1.SINGLE:
            return [self.single] if self.single else []
        return self.ensemble_judges


class SkillConfigV1(BaseModel):
    """Skill configuration for agent behavior."""

    path: str | None = Field(default=None, description="Path to skill file")
    no_skill: bool = Field(default=False, description="Disable skill (baseline mode)")
    content: str | None = Field(default=None, description="Inline skill content")

    @field_validator("no_skill", mode="after")
    @classmethod
    def validate_skill_config(cls, v: bool, info: Any) -> bool:
        """Ensure valid skill configuration."""
        # If no_skill is True, path and content should be None
        if v and (info.data.get("path") or info.data.get("content")):
            # Log warning but allow it
            pass
        return v

    def is_active(self) -> bool:
        """Check if skill is active (not disabled)."""
        return not self.no_skill and (self.path is not None or self.content is not None)


class ExecutionConfigV1(BaseModel):
    """Execution environment configuration."""

    timeout: int = Field(default=600, description="Timeout in seconds")
    docker_image: str = Field(default="posit-gskill-eval:latest", description="Docker image")
    repeats: int = Field(default=1, ge=1, description="Number of repeats per task")
    save_trajectories: bool = Field(default=True, description="Save agent trajectories")
    save_traces: bool = Field(default=False, description="Save LLM request/response traces")


class WorkerConfigV1(BaseModel):
    """Worker/parallelism configuration."""

    count: int = Field(default=5, ge=1, description="Number of parallel workers")
    auto_detect: bool = Field(default=False, description="Auto-detect worker count")


class ProfileV1(BaseModel):
    """
    Canonical profile schema for benchmark configuration.

    Profiles combine agent settings, tool configurations, and evaluation
    settings into a single configuration object.
    """

    # Schema versioning
    schema_version: str = Field(default="1.0", description="Schema version")

    # Profile identification
    profile_id: str = Field(description="Unique identifier for the profile")
    profile_type: ProfileTypeV1 = Field(
        default=ProfileTypeV1.AGENT,
        description="Type of profile",
    )
    name: str = Field(default="", description="Human-readable name")
    description: str = Field(default="", description="Profile description")

    # Model configuration
    model: ModelConfigV1 | None = Field(default=None, description="Primary model configuration")
    models: list[ModelConfigV1] = Field(
        default_factory=list, description="Multiple model configurations"
    )

    # Skill configuration
    skill: SkillConfigV1 = Field(default_factory=SkillConfigV1, description="Skill configuration")

    # Judge configuration
    judge: JudgeConfigV1 | None = Field(default=None, description="Judge configuration")

    # Execution settings
    execution: ExecutionConfigV1 = Field(
        default_factory=ExecutionConfigV1,
        description="Execution settings",
    )

    # Worker settings
    workers: WorkerConfigV1 = Field(
        default_factory=WorkerConfigV1,
        description="Worker/parallelism settings",
    )

    # Additional settings
    settings: dict[str, Any] = Field(default_factory=dict, description="Additional settings")

    # Metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the profile was created",
    )
    updated_at: str | None = Field(default=None, description="When the profile was last updated")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Ensure schema version is valid."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported schema version: {v}. Expected 1.x")
        return v

    def to_json_schema(self) -> dict[str, Any]:
        """Export this model's JSON schema."""
        return ProfileV1.model_json_schema()

    @classmethod
    def from_evaluation_config(cls, data: dict[str, Any], profile_id: str = "") -> "ProfileV1":
        """
        Convert from legacy EvaluationConfig format to ProfileV1.

        Args:
            data: Dictionary from EvaluationConfig
            profile_id: Optional profile ID

        Returns:
            ProfileV1 instance
        """
        model_data = data.get("model", {})
        model_config = ModelConfigV1(
            name=model_data.get("task", ""),
            provider=None,
            model_id=None,
        )

        skill_data = data.get("skill", {})
        skill_config = SkillConfigV1(
            path=skill_data.get("path"),
            no_skill=skill_data.get("no_skill", False),
        )

        execution_data = data.get("execution", {})
        execution_config = ExecutionConfigV1(
            timeout=execution_data.get("timeout", 600),
            docker_image=execution_data.get("docker_image", "posit-gskill-eval:latest"),
            repeats=execution_data.get("repeats", 1),
        )

        workers_data = data.get("workers", {})
        workers_config = WorkerConfigV1(
            count=workers_data.get("count", 5),
            auto_detect=workers_data.get("auto_detect", False),
        )

        return cls(
            profile_id=profile_id or "default",
            profile_type=ProfileTypeV1.EVALUATION,
            model=model_config,
            skill=skill_config,
            execution=execution_config,
            workers=workers_config,
        )

    @classmethod
    def from_benchmark_config(cls, data: dict[str, Any], profile_id: str = "") -> "ProfileV1":
        """
        Convert from legacy benchmark.yaml format to ProfileV1.

        Args:
            data: Dictionary from benchmark.yaml
            profile_id: Optional profile ID

        Returns:
            ProfileV1 instance
        """
        # Extract models
        model_names = data.get("models", [])
        models = [ModelConfigV1(name=name) for name in model_names]

        # Skill
        skill_path = data.get("skill")
        skill_config = SkillConfigV1(
            path=skill_path,
            no_skill=skill_path is None,
        )

        # Judge
        judge_data = data.get("judge", {})
        judge_mode = JudgeModeV1(judge_data.get("mode", "single"))
        judge_config = JudgeConfigV1(
            mode=judge_mode,
            single=judge_data.get("single"),
            ensemble_judges=judge_data.get("ensemble", {}).get("judges", []),
            voting=VotingStrategyV1(judge_data.get("ensemble", {}).get("voting", "majority")),
        )

        # Settings
        settings_data = data.get("settings", {})
        execution_config = ExecutionConfigV1(
            timeout=settings_data.get("timeout", 600),
            docker_image=settings_data.get("docker_image", "posit-gskill-eval:latest"),
            repeats=settings_data.get("repeats", 1),
            save_trajectories=settings_data.get("save_trajectories", True),
            save_traces=settings_data.get("save_traces", False),
        )

        return cls(
            profile_id=profile_id or "benchmark",
            profile_type=ProfileTypeV1.EVALUATION,
            models=models,
            skill=skill_config,
            judge=judge_config,
            execution=execution_config,
            settings={"results_dir": settings_data.get("results_dir", "results/benchmarks")},
        )


def validate_profile(data: dict[str, Any]) -> ProfileV1:
    """
    Validate profile data and return a ProfileV1 instance.

    Args:
        data: Raw profile data dictionary

    Returns:
        Validated ProfileV1 instance

    Raises:
        ValidationError: If data doesn't conform to ProfileV1 schema
    """
    return ProfileV1.model_validate(data)
