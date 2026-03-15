# Contributing to grist-mill

Thank you for your interest in contributing to **grist-mill**! This guide covers everything you need to get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Running Tests](#running-tests)
- [Linting & Formatting](#linting--formating)
- [Type Checking](#type-checking)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Writing Tests](#writing-tests)
- [Configuration System](#configuration-system)

## Development Setup

### Prerequisites

- **Python 3.10+** (3.14 recommended for development)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Docker** (optional, required for integration tests)
- **Git**

### Clone and Install

```bash
git clone https://github.com/grist-mill/grist-mill.git
cd grist-mill

# Install all dependencies
uv sync

# Verify installation
uv run grist-mill --version
```

### Optional Dependencies

```bash
# GEPA optimization support
uv sync --extra optimization

# (tree-sitter-language-pack is now a core dependency)
```

## Project Structure

```
grist-mill/
├── src/grist_mill/         # Main package
│   ├── schemas/            # Pydantic v2 data models
│   │   ├── artifact.py     # Artifact discriminated union (Tool, MCP, Skill)
│   │   ├── task.py         # Task, TaskResult, Manifest
│   │   └── telemetry.py    # TelemetrySchema, TokenUsage, LatencyBreakdown
│   ├── interfaces.py       # Abstract base classes (BaseAgent, BaseBenchmark, etc.)
│   ├── config.py           # Configuration loading (pydantic-settings)
│   ├── registry/           # Artifact and agent registries
│   ├── harness/            # Harness implementation (task → env → agent → result)
│   ├── environments/       # Docker and local process runners
│   ├── agents/             # Agent implementations (API-backed)
│   ├── tools/              # Tool orchestration and registry
│   ├── providers/          # LLM provider abstraction (OpenRouter, OpenAI, Anthropic)
│   ├── optimization/       # GEPA evaluator adapter and optimization runtime
│   ├── tasks/              # Task synthesis (AST parsing, mutation pipeline)
│   ├── dataset/            # Dataset management (splitting, versioning, quality)
│   ├── reports/            # Result analysis and comparison
│   ├── export/             # Export to JSON, CSV, HTML
│   └── cli/                # CLI entrypoint and subcommands
├── configs/examples/       # Example configuration files
├── tests/                  # Test suite
└── pyproject.toml          # Package metadata and tool configuration
```

## Architecture Overview

grist-mill follows a **BenchFlow-style decoupled architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (grist-mill)                         │
│  run  │  validate  │  list  │  optimize  │  report  │  export  │
└───────┼─────────────┼────────┼────────────┼──────────┼─────────┘
        │             │        │            │          │
┌───────▼─────────────▼────────▼────────────▼──────────▼─────────┐
│                     Configuration Layer                         │
│         YAML + env vars (GRIST_MILL_*) + CLI args               │
│         Precedence: CLI > env vars > YAML defaults             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       Harness Layer                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Agent    │───▶│  Environment │───▶│ Result Parser│          │
│  │ (LLM API) │    │ (Docker/Local)│    │ (TaskResult) │          │
│  └──────────┘    └──────────────┘    └──────────────┘          │
│         │              │                     │                  │
│         ▼              ▼                     ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │              Artifact Registry                    │          │
│  │  Tools  │  MCP Servers  │  Skills               │          │
│  └──────────────────────────────────────────────────┘          │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Tool     │    │ Telemetry    │    │ Reports &    │          │
│  │ Registry │    │ Collector    │    │ Export       │          │
│  └──────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

- **Registry pattern**: `ArtifactRegistry`, `AgentRegistry`, `ProviderRegistry` — decorator-based registration
- **Strategy pattern**: Interchangeable runners (Docker, local) and providers (OpenRouter, OpenAI, Anthropic)
- **Observer pattern**: Telemetry collectors observe execution phases and record metrics
- **Pydantic v2 throughout**: All data models use Pydantic v2 with discriminated unions for polymorphic types
- **Artifact-first**: Tools, MCP servers, and skills are first-class objects registered centrally

### Core Evaluation Loop

```
Task → Harness → env.prepare() → agent.run() → env.execute(test_command) → result.parse() → TaskResult
```

## Running Tests

### Quick Test Run

```bash
# Run all non-integration tests
uv run pytest -m 'not integration_local and not integration_provider' -q

# Run a specific test file
uv run pytest tests/test_schemas.py -v

# Run with verbose output
uv run pytest tests/test_harness.py -v -s
```

### Test Markers

| Marker | Description | When it runs |
|--------|-------------|-------------|
| *(default)* | Unit tests | Always |
| `integration_local` | Tests requiring Docker | Only when Docker is available |
| `integration_provider` | Tests requiring real API keys | Only when credentials are configured |

### Running Integration Tests

```bash
# Docker integration tests (requires running Docker daemon)
uv run pytest -m integration_local -v

# Provider integration tests (requires API keys)
uv run pytest -m integration_provider -v

# All tests
uv run pytest -v
```

## Linting & Formatting

grist-mill uses **[ruff](https://docs.astral.sh/ruff/)** for linting and formatting.

```bash
# Auto-fix lint issues and format
uv run ruff check --fix && uv run ruff format

# Check without fixing
uv run ruff check .
uv run ruff format --check .
```

### Ruff Configuration

- Line length: 100 characters
- Rules: E, F, I, N, W, UP, B, C4, SIM, TCH, RUF
- Ignored: E501 (line too long), TC001, TC003 (type-checking imports)

## Type Checking

grist-mill uses **[ty](https://docs.astral.sh/ty/)** for strict type checking.

```bash
# Check the entire project
uv run ty check .

# Check only the main package
uv run ty check src/grist_mill/
```

### Type Checking Rules

Most rules are configured as warnings for test code flexibility. The source code (`src/grist_mill/`) should have zero type errors.

### Conventions

- **Avoid `Any`** — use specific types or `Unknown` from `typing-extensions`
- All new code must be type-annotated
- Use `from __future__ import annotations` for forward references

## Code Style

### Naming

| Element | Style | Example |
|---------|-------|---------|
| Classes | `PascalCase` | `TaskResult`, `DockerRunner` |
| Functions/methods | `snake_case` | `run_evaluation`, `parse_result` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_TURNS`, `DEFAULT_TIMEOUT` |
| Module files | `snake_case.py` | `result_parser.py` |
| Package | `grist-mill` (PyPI), `grist_mill` (import) |

### Error Handling

- Never catch bare `Exception` — catch specific exceptions
- All framework errors should be custom exception classes
- Error messages must be actionable (tell the user what to fix, not just what broke)
- Use Python `logging` module — no `print` statements in library code

### Pydantic Models

- Use Pydantic v2 with `model_config = ConfigDict(...)` (not `class Config`)
- Use discriminated unions for polymorphic types
- All models must be serializable to/from JSON
- Validation errors must include field name, expected constraint, and actual value

### Configuration

- Use `pydantic-settings` `BaseSettings` with `env_prefix="grist_mill_"`
- Sensitive fields (API keys) must have masking in `__str__`/`__repr__`
- Precedence: CLI args > environment variables > YAML defaults

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Make changes** following the code style guidelines above
3. **Run quality gates** before pushing:
   ```bash
   uv run ruff check --fix && uv run ruff format
   uv run ty check .
   uv run pytest -m 'not integration_local and not integration_provider' -q
   ```
4. **Write tests** for any new functionality (TDD preferred)
5. **Update documentation** if you change public APIs or add features
6. **Commit** with clear, descriptive messages
7. **Open a Pull Request** with a description of the changes

### Quality Gates

All PRs must pass:

- ✅ `uv run ruff check .` — zero errors
- ✅ `uv run ruff format --check .` — no unformatted files
- ✅ `uv run ty check .` — zero errors
- ✅ `uv run pytest -m 'not integration_local and not integration_provider'` — all pass

## Writing Tests

Tests live in `tests/` and follow the naming convention `test_<module>.py`.

```python
"""Tests for the result parser module."""

from __future__ import annotations

import pytest

from grist_mill.schemas import TaskStatus


class TestResultParser:
    """Tests for result parsing logic."""

    def test_passing_output_produces_success(self) -> None:
        """A passing test produces status=SUCCESS, score=1.0."""
        # Arrange
        parser = ResultParser()

        # Act
        result = parser.parse(stdout="1 passed", exit_code=0)

        # Assert
        assert result.status == TaskStatus.SUCCESS
        assert result.score == 1.0

    @pytest.mark.integration_local
    def test_docker_execution(self) -> None:
        """Integration test requiring Docker."""
        ...
```

### Fixtures

Shared fixtures are in `tests/conftest.py`. Use `@pytest.fixture` for reusable test setup.

## Configuration System

grist-mill uses a unified configuration system with three layers:

```yaml
# config.yaml
agent:
  model: "gpt-4o"
  provider: "openai"
  max_turns: 10
  timeout: 300

environment:
  runner_type: "local"  # or "docker"

tasks:
  - id: "example-task"
    prompt: "Fix the bug in the calculator module."
    language: "python"
    test_command: "pytest tests/test_calc.py"
    timeout: 60
    difficulty: "MEDIUM"
```

Environment variables override YAML values:

```bash
GRIST_MILL_AGENT_MODEL=claude-3-haiku grist-mill run --config config.yaml
```

CLI arguments take highest precedence:

```bash
grist-mill run --config config.yaml --dry-run --output-format json
```

## Example Configs

See `configs/examples/` for ready-to-use configurations:

| Config | Use Case |
|--------|----------|
| `smoke.yaml` | Quick smoke test (local runner, echo tasks) |
| `single_model.yaml` | Single-model benchmark evaluation |
| `multi_model.yaml` | Multi-model comparison experiment |
| `provider_setup.yaml` | LLM provider configuration examples |
| `optimize_smoke.yaml` | Optimization loop smoke test |

## Getting Help

- **CLI help**: `grist-mill --help` or `grist-mill <command> --help`
- **Issues**: [GitHub Issues](https://github.com/grist-mill/grist-mill/issues)
- **Docs**: [README.md](README.md)
