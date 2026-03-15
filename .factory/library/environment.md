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
- **Type checker:** `ty` with strict settings
- **Linter/Formatter:** `ruff check --fix && ruff format`
- **Test runner:** `pytest` with markers

## Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `optimization` | `gepa` | GEPA-based skill/artifact optimization |
| `synthesis` | `tree-sitter-language-pack` | Task synthesis via AST analysis |
| `providers` | `openai` | Additional LLM provider support |

## Docker

Docker must be running for Docker-based evaluation. The `LocalRunner` works without Docker.
