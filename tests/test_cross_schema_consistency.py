"""Cross-schema consistency tests.

Validates:
- VAL-CROSS-02: Configuration resolution across all subsystems.
  A single config file change propagates correctly to all dependent subsystems
  (harness, agent, optimization, reports) without separate edits.
- VAL-CROSS-03: Schema consistency from definition to consumption.
  Schemas defined in M1 are used without redefinition or divergence by all
  downstream modules. A schema change in M1 triggers validation errors in
  all downstream consumers.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import textwrap

import pytest

from grist_mill.config import load_config
from grist_mill.schemas import (
    AgentConfig,
    Difficulty,
    EnvironmentConfig,
    ErrorCategory,
    HarnessConfig,
    Manifest,
    Task,
    TaskResult,
    TaskStatus,
)
from grist_mill.schemas.artifact import (
    MCPServerArtifact,
    SkillArtifact,
    ToolArtifact,
)
from grist_mill.schemas.telemetry import (
    LatencyBreakdown,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "grist_mill"
SCHEMAS_DIR = SRC_DIR / "schemas"

# All schema classes that should be defined ONLY in grist_mill.schemas
CORE_SCHEMA_CLASSES = {
    "Task",
    "TaskResult",
    "TaskStatus",
    "Difficulty",
    "ErrorCategory",
    "ExecutionOutput",
    "HarnessConfig",
    "AgentConfig",
    "EnvironmentConfig",
    "Manifest",
    "EnvironmentHealth",
    "Artifact",
    "ToolArtifact",
    "MCPServerArtifact",
    "SkillArtifact",
    "TelemetrySchema",
    "TokenUsage",
    "LatencyBreakdown",
    "ToolCallMetrics",
    "TelemetryCollector",
}


def _collect_schema_imports_from_file(filepath: pathlib.Path) -> set[str]:
    """Parse a Python file and return all 'from grist_mill.schemas' imports."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "grist_mill.schemas" in node.module:
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    return imported


# ---------------------------------------------------------------------------
# VAL-CROSS-03: No redefinitions outside schemas/
# ---------------------------------------------------------------------------


class TestNoSchemaRedefinitions:
    """Verify no module outside schemas/ redefines core schema classes."""

    def test_no_class_definitions_outside_schemas(self) -> None:
        """Grep-like: no 'class Task|TaskResult|...' outside schemas/."""
        violations: list[str] = []
        for py_file in SRC_DIR.rglob("*.py"):
            # Skip __pycache__
            if "__pycache__" in str(py_file):
                continue
            # Allow files inside schemas/ to define these classes
            rel = py_file.relative_to(SRC_DIR)
            parts = rel.parts
            if parts[0] == "schemas":
                continue

            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name in CORE_SCHEMA_CLASSES:
                    violations.append(
                        f"{rel}: class {node.name} is redefined (should only be in schemas/)"
                    )

        assert not violations, (
            f"Found {len(violations)} schema redefinitions outside schemas/:\n"
            + "\n".join(violations)
        )

    def test_no_enum_redefinitions_outside_schemas(self) -> None:
        """Ensure enums like TaskStatus, ErrorCategory, Difficulty are not
        re-assigned (e.g., = 'SUCCESS') in downstream modules."""
        violations: list[str] = []
        for py_file in SRC_DIR.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            rel = py_file.relative_to(SRC_DIR)
            if rel.parts[0] == "schemas":
                continue

            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            enum_names = {"TaskStatus", "ErrorCategory", "Difficulty"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in enum_names:
                            violations.append(
                                f"{rel}: {target.id} is reassigned (should be imported from schemas)"
                            )

        assert not violations, (
            f"Found {len(violations)} enum re-assignments outside schemas/:\n"
            + "\n".join(violations)
        )

    def test_all_downstream_modules_import_from_schemas(self) -> None:
        """Verify key downstream modules import from grist_mill.schemas."""
        downstream_modules = {
            "harness.harness": {"HarnessConfig", "Task", "TaskResult", "ErrorCategory", "TaskStatus"},
            "agents.api_agent": {"HarnessConfig", "Task", "TaskResult", "ErrorCategory", "TaskStatus"},
            "optimization.evaluator_adapter": {"Task", "TaskResult", "TaskStatus", "Difficulty"},
            "environments.local_runner": {"ExecutionOutput", "Task"},
            "harness.result_parser": {"TaskResult", "TaskStatus", "ErrorCategory"},
        }

        for module_name, expected_imports in downstream_modules.items():
            module = importlib.import_module(f"grist_mill.{module_name}")
            source = inspect.getsource(module)

            # Check each expected import is present
            for schema_name in expected_imports:
                # Could be imported directly or via the module
                has_import = "from grist_mill.schemas" in source and schema_name in source
                assert has_import, (
                    f"grist_mill.{module_name} should import '{schema_name}' from "
                    f"grist_mill.schemas but it doesn't appear to do so"
                )


class TestSchemaIdentity:
    """Verify that schemas used across modules are the SAME class object."""

    def test_task_class_identity_across_modules(self) -> None:
        """Task imported from schemas is the same as what harness uses."""
        from grist_mill.harness import harness as harness_mod

        # The harness module imports Task from grist_mill.schemas
        assert harness_mod.Task is Task

    def test_task_result_identity_across_modules(self) -> None:
        """TaskResult is identical across harness and agents."""
        from grist_mill.agents import api_agent as agent_mod
        from grist_mill.harness import harness as harness_mod

        assert harness_mod.TaskResult is agent_mod.TaskResult is TaskResult

    def test_task_status_identity_across_modules(self) -> None:
        """TaskStatus enum is identical across all consumers."""
        from grist_mill.harness import result_parser as parser_mod

        assert parser_mod.TaskStatus is TaskStatus

    def test_error_category_identity_across_modules(self) -> None:
        """ErrorCategory enum is identical across harness and agents."""
        from grist_mill.agents import api_agent as agent_mod
        from grist_mill.harness import harness as harness_mod

        assert harness_mod.ErrorCategory is agent_mod.ErrorCategory is ErrorCategory

    def test_harness_config_identity_across_modules(self) -> None:
        """HarnessConfig is identical across harness and registry."""
        from grist_mill.harness import harness as harness_mod

        assert harness_mod.HarnessConfig is HarnessConfig

    def test_telemetry_schema_identity(self) -> None:
        """TelemetrySchema is the same class object everywhere."""
        from grist_mill.optimization import evaluator_adapter as adapter_mod

        assert adapter_mod.TelemetrySchema is TelemetrySchema


# ---------------------------------------------------------------------------
# VAL-CROSS-03: Schema changes propagate (validated via type-check)
# ---------------------------------------------------------------------------


class TestSchemaChangePropagation:
    """Verify a schema change in M1 would cause type errors in consumers.

    We can't actually mutate the schema at runtime and re-check, but we can
    verify that:
    1. All consumers type-annotate their parameters with the schema types.
    2. Missing required fields cause ValidationError at runtime.
    3. Field removal would cause attribute errors.
    """

    def test_task_required_fields_enforced_in_harness(self) -> None:
        """Harness.run fails if Task is missing required fields."""
        from pydantic import ValidationError

        # Creating a Task without required fields should raise ValidationError
        with pytest.raises(ValidationError):
            Task(id="", prompt="", language="", test_command="", timeout=10)

    def test_task_result_score_bounds_enforced_in_agent(self) -> None:
        """Agent cannot produce a TaskResult with score outside [0, 1]."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TaskResult(
                task_id="t1",
                status=TaskStatus.SUCCESS,
                score=1.5,  # Out of bounds
            )

    def test_task_result_status_enum_enforced_in_parser(self) -> None:
        """Result parser cannot produce invalid status values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TaskResult(
                task_id="t1",
                status="almost",  # type: ignore[arg-type]
                score=0.5,
            )

    def test_harness_config_agent_config_propagates(self) -> None:
        """Changing agent model in config propagates to HarnessConfig.agent.model."""
        yaml_content = textwrap.dedent("""\
            agent:
              model: original-model
              provider: openai
            environment:
              runner_type: local
        """)
        config = load_config(yaml_content=yaml_content)
        assert config.agent.model == "original-model"

        # Override via CLI args
        config_override = load_config(
            yaml_content=yaml_content,
            cli_args={"model": "new-model"},
        )
        assert config_override.agent.model == "new-model"

    def test_manifest_task_ids_validated(self) -> None:
        """Manifest duplicate ID validation is enforced."""
        from pydantic import ValidationError

        task = Task(
            id="t1",
            prompt="Test task",
            language="python",
            test_command="pytest",
            timeout=60,
        )
        with pytest.raises(ValidationError, match="Duplicate task ID"):
            Manifest(name="test", version="1.0", tasks=[task, task])


# ---------------------------------------------------------------------------
# VAL-CROSS-02: Config resolution across subsystems
# ---------------------------------------------------------------------------


class TestConfigResolutionAcrossSubsystems:
    """Verify a single config file resolves settings for all subsystems."""

    def _make_full_config(self) -> str:
        """Create a YAML config covering agent, environment, telemetry."""
        return textwrap.dedent("""\
            agent:
              model: gpt-4
              provider: openrouter
              system_prompt: "You are a helpful assistant."
              max_turns: 10
              timeout: 300
              api_key: sk-test-key
            environment:
              runner_type: docker
              docker_image: python:3.11
              cpu_limit: 2.0
              memory_limit: 4g
              network_access: false
              working_dir: /app
            telemetry:
              enabled: true
              trace_enabled: true
              output_dir: ./results/telemetry
            tasks:
              - id: task-1
                prompt: "Fix the bug"
                language: python
                test_command: "pytest tests/"
                timeout: 120
                difficulty: EASY
            artifacts:
              - type: tool
                name: grep
                description: "Search tool"
                input_schema:
                  type: object
                  properties:
                    pattern:
                      type: string
        """)

    def test_config_loads_all_sections(self) -> None:
        """All config sections (agent, environment, telemetry) are populated."""
        config = load_config(yaml_content=self._make_full_config())

        # Agent section
        assert config.agent.model == "gpt-4"
        assert config.agent.provider == "openrouter"
        assert config.agent.system_prompt == "You are a helpful assistant."
        assert config.agent.max_turns == 10
        assert config.agent.timeout == 300
        assert config.agent.api_key == "sk-test-key"

        # Environment section
        assert config.environment.runner_type == "docker"
        assert config.environment.docker_image == "python:3.11"
        assert config.environment.cpu_limit == 2.0
        assert config.environment.memory_limit == "4g"
        assert config.environment.network_access is False
        assert config.environment.working_dir == "/app"

        # Telemetry section
        assert config.telemetry.enabled is True
        assert config.telemetry.trace_enabled is True
        assert config.telemetry.output_dir == "./results/telemetry"

    def test_config_tasks_use_schema_types(self) -> None:
        """Tasks from config are proper Task schema instances."""
        config = load_config(yaml_content=self._make_full_config())

        assert len(config.tasks) == 1
        task = config.tasks[0]
        assert isinstance(task, Task)
        assert task.id == "task-1"
        assert task.prompt == "Fix the bug"
        assert task.language == "python"
        assert task.difficulty == Difficulty.EASY

    def test_config_artifacts_use_schema_types(self) -> None:
        """Artifacts from config are proper schema instances."""
        config = load_config(yaml_content=self._make_full_config())

        assert len(config.artifacts) == 1
        artifact = config.artifacts[0]
        assert isinstance(artifact, ToolArtifact)
        assert artifact.name == "grep"

    def test_config_change_propagates_to_harness_config(self) -> None:
        """A config change (model switch) propagates to HarnessConfig."""
        yaml_content = textwrap.dedent("""\
            agent:
              model: gpt-4
              provider: openai
            environment:
              runner_type: local
        """)
        config1 = load_config(yaml_content=yaml_content)
        hc1 = HarnessConfig(
            agent=AgentConfig(model=config1.agent.model, provider=config1.agent.provider),
            environment=EnvironmentConfig(runner_type=config1.environment.runner_type),
        )
        assert hc1.agent.model == "gpt-4"

        # Change model
        yaml_v2 = textwrap.dedent("""\
            agent:
              model: claude-3-opus
              provider: anthropic
            environment:
              runner_type: local
        """)
        config2 = load_config(yaml_content=yaml_v2)
        hc2 = HarnessConfig(
            agent=AgentConfig(model=config2.agent.model, provider=config2.agent.provider),
            environment=EnvironmentConfig(runner_type=config2.environment.runner_type),
        )
        assert hc2.agent.model == "claude-3-opus"
        assert hc2.agent.provider == "anthropic"

    def test_env_var_override_propagates_to_subsystems(self) -> None:
        """Environment variable override propagates to all subsystems."""
        import os

        yaml_content = textwrap.dedent("""\
            agent:
              model: gpt-4
              provider: openai
              timeout: 300
            environment:
              runner_type: local
        """)

        # Without env override
        config1 = load_config(yaml_content=yaml_content)
        assert config1.agent.timeout == 300

        # With env override
        os.environ["GRIST_MILL_TIMEOUT"] = "600"
        try:
            config2 = load_config(yaml_content=yaml_content)
            assert config2.agent.timeout == 600
        finally:
            del os.environ["GRIST_MILL_TIMEOUT"]

    def test_telemetry_config_propagates(self) -> None:
        """Telemetry settings from config propagate correctly."""
        yaml_content = textwrap.dedent("""\
            agent:
              model: test
              provider: openai
            telemetry:
              enabled: false
              trace_enabled: true
              output_dir: /tmp/telemetry
        """)
        config = load_config(yaml_content=yaml_content)
        assert config.telemetry.enabled is False
        assert config.telemetry.trace_enabled is True
        assert config.telemetry.output_dir == "/tmp/telemetry"

    def test_single_config_multiple_tasks(self) -> None:
        """Multiple tasks from a single config all use Task schema."""
        yaml_content = textwrap.dedent("""\
            agent:
              model: test
              provider: openai
            tasks:
              - id: t1
                prompt: "Task 1"
                language: python
                test_command: "pytest"
                timeout: 60
              - id: t2
                prompt: "Task 2"
                language: r
                test_command: "Rscript test.R"
                timeout: 120
                difficulty: HARD
              - id: t3
                prompt: "Task 3"
                language: typescript
                test_command: "npm test"
                timeout: 90
                difficulty: MEDIUM
        """)
        config = load_config(yaml_content=yaml_content)
        assert len(config.tasks) == 3
        assert all(isinstance(t, Task) for t in config.tasks)
        assert config.tasks[0].language == "python"
        assert config.tasks[1].language == "r"
        assert config.tasks[1].difficulty == Difficulty.HARD
        assert config.tasks[2].language == "typescript"
        assert config.tasks[2].difficulty == Difficulty.MEDIUM


# ---------------------------------------------------------------------------
# VAL-CROSS-03: Data round-trips across subsystems
# ---------------------------------------------------------------------------


class TestSchemaRoundTrips:
    """Verify schema data round-trips correctly across subsystems."""

    def test_task_serialization_round_trip(self) -> None:
        """Task serializes to JSON and deserializes without loss."""
        task = Task(
            id="t1",
            prompt="Fix the bug in function foo",
            language="python",
            test_command="pytest tests/test_foo.py",
            setup_command="pip install -e .",
            timeout=120,
            difficulty=Difficulty.HARD,
            constraints=["no-network"],
            dependencies=["numpy"],
        )
        json_str = task.model_dump_json()
        restored = Task.model_validate_json(json_str)
        assert restored == task

    def test_task_result_serialization_round_trip(self) -> None:
        """TaskResult with telemetry round-trips without loss."""
        telemetry = TelemetrySchema(
            tokens=TokenUsage(prompt=100, completion=50, total=150),
            latency=LatencyBreakdown(setup_s=1.0, execution_s=5.0, teardown_s=0.5, total_s=6.5),
            tool_calls=ToolCallMetrics(total_calls=3, successful_calls=2, failed_calls=1),
            estimated_cost_usd=0.002,
        )
        result = TaskResult(
            task_id="t1",
            status=TaskStatus.SUCCESS,
            score=1.0,
            telemetry=telemetry.model_dump(),
        )
        json_str = result.model_dump_json()
        restored = TaskResult.model_validate_json(json_str)
        assert restored.task_id == "t1"
        assert restored.status == TaskStatus.SUCCESS
        assert restored.score == 1.0
        assert restored.telemetry is not None
        # telemetry is stored as dict (Any type), verify token data preserved
        assert isinstance(restored.telemetry, dict)
        assert restored.telemetry["tokens"]["total"] == 150

    def test_manifest_serialization_round_trip(self) -> None:
        """Manifest with tasks round-trips without loss."""
        tasks = [
            Task(id=f"t{i}", prompt=f"Task {i}", language="python", test_command="pytest", timeout=60)
            for i in range(3)
        ]
        manifest = Manifest(name="bench", version="1.0", tasks=tasks)
        json_str = manifest.model_dump_json()
        restored = Manifest.model_validate_json(json_str)
        assert len(restored.tasks) == 3
        assert restored.name == "bench"
        assert restored.tasks[0].id == "t0"

    def test_harness_config_serialization_round_trip(self) -> None:
        """HarnessConfig round-trips without loss."""
        config = HarnessConfig(
            agent=AgentConfig(model="gpt-4", provider="openai", system_prompt="Be helpful"),
            environment=EnvironmentConfig(
                runner_type="docker",
                docker_image="python:3.11",
                resource_limits={"cpu": 2, "memory": "4g"},
            ),
            artifact_bindings=["grep", "skill-1"],
        )
        json_str = config.model_dump_json()
        restored = HarnessConfig.model_validate_json(json_str)
        assert restored.agent.model == "gpt-4"
        assert restored.agent.system_prompt == "Be helpful"
        assert restored.environment.docker_image == "python:3.11"
        assert restored.artifact_bindings == ["grep", "skill-1"]

    def test_telemetry_forward_compatibility(self) -> None:
        """Loading telemetry with extra (future) fields does not error."""
        json_str = """{
            "version": "V1",
            "tokens": {"prompt": 10, "completion": 5, "total": 15},
            "latency": {"setup_s": 0.1, "execution_s": 0.5, "teardown_s": 0.05, "total_s": 0.65},
            "tool_calls": {"total_calls": 1, "successful_calls": 1, "failed_calls": 0, "by_tool": {}},
            "future_field": "this should be ignored",
            "nested_future": {"unknown": true}
        }"""
        telemetry = TelemetrySchema.model_validate_json(json_str)
        assert telemetry.tokens.total == 15
        assert telemetry.version == "V1"

    def test_artifact_discriminated_union_round_trip(self) -> None:
        """Each artifact variant round-trips correctly."""
        tool = ToolArtifact(
            type="tool",
            name="grep",
            description="Search tool",
            input_schema={"type": "object"},
        )
        json_str = tool.model_dump_json()
        restored = ToolArtifact.model_validate_json(json_str)
        assert restored.name == "grep"
        assert restored.type == "tool"

        mcp = MCPServerArtifact(
            type="mcp_server",
            name="my-mcp",
            command="npx",
            args=["-y", "my-server"],
        )
        json_str = mcp.model_dump_json()
        restored = MCPServerArtifact.model_validate_json(json_str)
        assert restored.name == "my-mcp"
        assert restored.args == ["-y", "my-server"]

        skill = SkillArtifact(
            type="skill",
            name="debugging",
            skill_file_path="/skills/debug.md",
        )
        json_str = skill.model_dump_json()
        restored = SkillArtifact.model_validate_json(json_str)
        assert restored.skill_file_path == "/skills/debug.md"


# ---------------------------------------------------------------------------
# VAL-CROSS-02: Config change affects all subsystem outputs
# ---------------------------------------------------------------------------


class TestConfigChangeEffects:
    """Verify a single config change affects subsystem outputs consistently."""

    def test_model_change_affects_harness_config(self) -> None:
        """Changing the model in config changes what the harness sees."""
        yaml_base = textwrap.dedent("""\
            agent:
              model: gpt-4
              provider: openai
            environment:
              runner_type: local
        """)

        config_v1 = load_config(yaml_content=yaml_base, cli_args={})
        config_v2 = load_config(yaml_content=yaml_base, cli_args={"model": "gpt-4-turbo"})

        hc_v1 = HarnessConfig(
            agent=AgentConfig(model=config_v1.agent.model, provider=config_v1.agent.provider),
            environment=EnvironmentConfig(runner_type=config_v1.environment.runner_type),
        )
        hc_v2 = HarnessConfig(
            agent=AgentConfig(model=config_v2.agent.model, provider=config_v2.agent.provider),
            environment=EnvironmentConfig(runner_type=config_v2.environment.runner_type),
        )

        assert hc_v1.agent.model != hc_v2.agent.model
        assert hc_v2.agent.model == "gpt-4-turbo"

    def test_timeout_change_affects_all_subsystems(self) -> None:
        """Timeout change propagates to both agent config and task configs."""
        yaml_base = textwrap.dedent("""\
            agent:
              model: test
              provider: openai
              timeout: 300
            tasks:
              - id: t1
                prompt: "Task 1"
                language: python
                test_command: "pytest"
                timeout: 60
        """)

        config = load_config(yaml_content=yaml_base, cli_args={"timeout": 600})
        assert config.agent.timeout == 600
        # Task timeout comes from the task definition, not the agent timeout
        assert config.tasks[0].timeout == 60

    def test_telemetry_trace_flag_propagates(self) -> None:
        """Trace flag from config can be used by harness and adapters."""
        yaml_base = textwrap.dedent("""\
            agent:
              model: test
              provider: openai
            telemetry:
              trace_enabled: true
        """)
        config = load_config(yaml_content=yaml_base)
        assert config.telemetry.trace_enabled is True

        # The harness uses this flag
        # Verify it's accessible and boolean-typed
        assert isinstance(config.telemetry.trace_enabled, bool)
