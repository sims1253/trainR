# grist-mill

> Language-agnostic benchmarking framework for evaluating autonomous coding agents.

grist-mill evaluates AI coding agents across their full toolchain — tools, MCP servers, skills, and harness configurations — providing extensible task synthesis, multi-provider evaluation, and optimization feedback loops.

## Quick Start

Get a result in under 5 minutes:

```bash
# 1. Clone and install
git clone https://github.com/grist-mill/grist-mill.git
cd grist-mill
uv sync

# 2. Run the smoke test (no API keys needed)
uv run grist-mill run --config configs/examples/smoke.yaml

# 3. Validate a configuration
uv run grist-mill validate --config configs/examples/single_model.yaml
```

That's it — you've just run a benchmark evaluation!

## Architecture

```text
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
│                                                                  │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │  Agent    │────▶│  Environment │────▶│ Result Parser│        │
│  │ (LLM API) │     │ (Docker/Local)│     │ (TaskResult) │        │
│  └────┬─────┘     └──────────────┘     └──────────────┘        │
│       │                                                      │
│  ┌────▼────────────────────────────────────────────────────┐  │
│  │              Artifact Registry                            │  │
│  │     Tools  │  MCP Servers  │  Skills                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │              │              │                          │
│       ▼              ▼              ▼                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Tool    │  │Telemetry │  │ Reports  │                   │
│  │ Registry │  │Collector │  │ & Export │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Evaluation Loop

```
Task → Harness → env.prepare() → agent.run() → env.execute(test_command) → parse → TaskResult
```

### Module Structure

```
src/grist_mill/
├── schemas/          # Pydantic v2 data models (Task, TaskResult, Artifact, Telemetry)
├── interfaces.py     # Abstract base classes (BaseAgent, BaseBenchmark, BaseEnvironment)
├── config.py         # Configuration loading (pydantic-settings, YAML, env vars, CLI)
├── registry/         # Artifact and agent registries (decorator-based registration)
├── harness/          # Harness implementation (wires task → env → agent → result)
├── environments/     # Docker and local-process runners
├── agents/           # API-backed agent with multi-turn conversation
├── tools/            # Tool orchestration and registry with capability advertisement
├── providers/        # Multi-provider LLM resolution (OpenRouter, OpenAI, Anthropic)
├── optimization/     # GEPA evaluator adapter and optimization runtime
├── tasks/            # Task synthesis (tree-sitter AST analysis, test mutation)
├── dataset/          # Dataset management (splitting, versioning, quality gates)
├── reports/          # Result analysis and comparison
├── export/           # Export to JSON, CSV, HTML
└── cli/              # CLI entrypoint and subcommands
```

## Features

### 🧪 Benchmarking
- **Multi-provider support**: OpenRouter, OpenAI, Anthropic, and custom providers via a single config change
- **Flexible environments**: Local-process runner for fast iteration, Docker runner for isolated execution
- **Artifact system**: Tools, MCP servers, and skills as first-class pluggable artifacts

### 🔬 Task Synthesis
- **AST-based analysis**: tree-sitter parsing for Python, R, and TypeScript
- **Test mutation pipeline**: Automatically generates tasks by introducing controlled bugs
- **Manual authoring**: Import tasks from YAML/TOML with full metadata validation

### ⚡ Optimization
- **GEPA integration**: Evolve skills, system prompts, and tool policies through evaluation feedback
- **Budget management**: Composable stop conditions (max calls, timeout, no-improvement)
- **Checkpoint/Resume**: Persist and restore optimization state

### 📊 Reporting & Export
- **Experiment comparison**: Per-task deltas with statistical significance
- **Telemetry aggregation**: Per-model, per-tool, per-experiment summaries
- **Multiple formats**: JSON (self-describing), CSV (pandas-compatible), HTML (standalone)

## CLI Usage

```bash
# Run a benchmark evaluation
grist-mill run --config experiment.yaml

# Preview without executing
grist-mill run --config experiment.yaml --dry-run

# Validate a configuration
grist-mill validate --config experiment.yaml

# Generate tasks from source code
grist-mill tasks generate --repo /path/to/source

# Optimize a skill
grist-mill optimize --config optimize.yaml

# Compare two experiments
grist-mill report --type comparison --results exp-a.json --compare-with exp-b.json

# Export results
grist-mill export --results results.json --format html --output report.html
```

## Configuration

### Example Config

```yaml
agent:
  model: "gpt-4o"
  provider: "openai"
  max_turns: 10
  timeout: 300

environment:
  runner_type: "local"

tasks:
  - id: "fix-bug-001"
    prompt: "Fix the null pointer exception in the parser."
    language: "python"
    test_command: "pytest tests/ -q"
    timeout: 60
    difficulty: "MEDIUM"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GRIST_MILL_AGENT_MODEL` | Override the LLM model |
| `GRIST_MILL_AGENT_PROVIDER` | Override the LLM provider |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |

### Config Precedence

`CLI arguments > Environment variables > YAML defaults`

## Example Configs

| Config | Use Case |
|--------|----------|
| `configs/examples/smoke.yaml` | Quick smoke test (no API keys needed) |
| `configs/examples/single_model.yaml` | Single-model benchmark evaluation |
| `configs/examples/multi_model.yaml` | Multi-model comparison experiment |
| `configs/examples/provider_setup.yaml` | LLM provider configuration guide |
| `configs/examples/skill_optimization.yaml` | Skill prompt optimization workflow |
| `configs/examples/optimize_smoke.yaml` | Optimization loop smoke test |

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -m 'not integration_local and not integration_provider' -q

# Lint and format
uv run ruff check --fix && uv run ruff format

# Type check
uv run ty check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (optional, for containerized evaluation)

## License

MIT License — see [LICENSE](LICENSE) for details.
