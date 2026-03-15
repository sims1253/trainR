# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GRIST_MILL_DEFAULT_PROVIDER` | Default LLM provider (openrouter, openai, etc.) | Optional (default: openrouter) |
| `OPENROUTER_API_KEY` | OpenRouter API key | For provider resolution |
| `OPENAI_API_KEY` | OpenAI API key | For provider resolution |
| `ANTHROPIC_API_KEY` | Anthropic API key | For provider resolution |

## Python Tooling

- **Package manager:** `uv` — all commands use `uv run`, `uv sync`, `uv add`
- **Type checker:** `ty` with strict settings. **Important:** ty uses 0.0.x versioning (e.g., 0.0.23, NOT 0.1+). The python-version must go under `[tool.ty]` → `environment.python-version`, NOT at the top level of `[tool.ty]`.
- **Linter/Formatter:** `ruff check --fix && ruff format`
- **Test runner:** `pytest` with markers
- **Click:** `CliRunner(mix_stderr=False)` is NOT compatible with the installed Click version. Use `CliRunner()` without `mix_stderr` parameter.

## Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `optimization` | `gepa` | GEPA-based skill/artifact optimization |
| `providers` | `openai` | Additional LLM provider support |

## Core Dependencies

| Package | Use Case |
|---------|----------|
| `tree-sitter-language-pack` | Task synthesis via AST analysis (installed as core dependency, not optional) |

> **Note:** Although the `synthesis` extra exists in pyproject.toml, `tree-sitter-language-pack` is listed as a core dependency. It is always available after `uv sync`. The synthesis features (AST parser, mutation pipeline, task pipeline) require it.

## Docker

Docker must be running for Docker-based evaluation. The `LocalRunner` works without Docker.
