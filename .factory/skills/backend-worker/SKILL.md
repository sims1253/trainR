---
name: backend-worker
description: General-purpose Python worker for grist-mill framework implementation. Handles schemas, interfaces, registries, configs, runners, agents, tools, optimization, reports, and all framework components.
---

# Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for all grist-mill implementation features:
- Core schemas and Pydantic models
- Abstract interfaces and base classes
- Configuration system
- CLI subcommands
- Evaluation harness and runners (Docker, local)
- Agent implementations
- Tool orchestration and registries
- Task synthesis pipeline
- Optimization (GEPA integration)
- Reporting and export
- Package scaffolding and metadata

## Work Procedure

1. **Read mission context.** Read `mission.md` and `AGENTS.md` from the mission directory. Understand the milestone and feature you're implementing.

2. **Read existing code.** Use `Glob` and `Read` to understand the current state of `src/grist_mill/`. Check what modules already exist and follow established patterns.

3. **Write tests FIRST (TDD).** Create test file(s) for the feature. Tests go in `tests/`. Use pytest markers (`@pytest.mark.integration_local` for Docker tests, `@pytest.mark.integration_provider` for API tests). Write comprehensive tests covering:
   - Happy path behavior
   - Edge cases (empty inputs, invalid inputs, boundary values)
   - Error handling (missing fields, wrong types, resource failures)
   - Integration with existing modules where applicable

4. **Run tests to confirm they FAIL.** Execute `uv run pytest tests/<your_test_file>.py -x -q` and verify the tests fail (red phase). If tests pass immediately, they're not testing the right thing or the feature already exists.

5. **Implement the feature.** Write the minimal code to make the tests pass (green phase). Follow these conventions:
   - Use `pydantic` v2 for all data models
   - Use `abc.ABC` + `@abstractmethod` for interfaces
   - Use `pydantic-settings` `BaseSettings` for configuration
   - Type-annotate everything — no bare `Any`
   - Use Python `logging` module, never `print` in library code
   - Place code under `src/grist_mill/` in the appropriate module
   - Follow existing naming conventions (PascalCase classes, snake_case functions)

6. **Run tests to confirm they PASS.** Execute the full test suite: `uv run pytest tests/ -m "not integration_local and not integration_provider" -x -q`. Fix any failures.

7. **Run type checking.** Execute `uv run ty check .` and fix any type errors. All new code must pass strict type checking.

8. **Run linting.** Execute `uv run ruff check --fix && uv run ruff format` and fix any issues.

9. **Manual verification.** If the feature has a CLI surface, test it manually:
   - Run `grist-mill --help` to verify the command appears
   - Run the specific subcommand with `--help`
   - Run a smoke test (e.g., `grist-mill validate --config configs/examples/smoke.yaml`)
   - Test error cases (missing config, invalid input)

10. **Commit the work.** Stage and commit with a descriptive message following conventional commits format (e.g., `feat(schemas): add Task and TaskResult Pydantic models`).

## Verification Commands

Always run these before completing:
```bash
# Unit tests
uv run pytest tests/ -m "not integration_local and not integration_provider" -x -q

# Type checking
uv run ty check .

# Linting
uv run ruff check --fix && uv run ruff format

# Verify CLI installs
uv sync && grist-mill --version
```

## Example Handoff

```json
{
  "salientSummary": "Implemented the Pydantic v2 schema layer with Task, TaskResult, and Manifest models including discriminated unions for Artifact types and comprehensive validation. All 24 unit tests pass, ty check clean, ruff clean.",
  "whatWasImplemented": "Created src/grist_mill/schemas/ with task.py (Task, TaskResult, TaskStatus enum), manifest.py (Manifest with dedup validation), artifact.py (Artifact discriminated union with ToolArtifact, MCPServerArtifact, SkillArtifact variants), and telemetry.py (TelemetrySchema, TokenUsage, LatencyBreakdown). Added conftest.py with shared fixtures.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/ -m 'not integration_local and not integration_provider' -x -q", "exitCode": 0, "observation": "24 tests passed"},
      {"command": "uv run ty check .", "exitCode": 0, "observation": "No type errors"},
      {"command": "uv run ruff check --fix && uv run ruff format", "exitCode": 0, "observation": "No lint issues"}
    ],
    "interactiveChecks": [
      {"action": "grist-mill --help", "observed": "Help text displayed with subcommands"}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_schemas.py", "cases": [
        {"name": "test_task_valid_construction", "verifies": "Task model accepts valid inputs"},
        {"name": "test_task_rejects_negative_timeout", "verifies": "ValidationError on timeout <= 0"},
        {"name": "test_task_result_score_bounds", "verifies": "Score constrained to [0.0, 1.0]"},
        {"name": "test_artifact_discriminated_union", "verifies": "Correct variant construction"},
        {"name": "test_manifest_dedup_validation", "verifies": "Duplicate task IDs rejected"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on an interface or module that hasn't been implemented yet (check the milestone order)
- Requirements in the feature description are ambiguous or contradictory
- Existing code has bugs that block this feature
- Docker is required but unavailable and the feature cannot use LocalRunner instead
- A dependency needs to be added to pyproject.toml (confirm with orchestrator first)
