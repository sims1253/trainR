# grist-mill

> Language-agnostic benchmarking framework for evaluating autonomous coding agents.

grist-mill evaluates AI coding agents across their full toolchain -- tools, MCP servers, skills, and harness configurations -- providing extensible task synthesis, multi-provider evaluation, and evolutionary optimization feedback loops.

The framework powers **posit-gskill**, a system for automatically improving R package testing skills for Claude Code using GEPA (Gradient-free Evolutionary Prompt Algorithm).

## Quick Start

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
└────────────────────────────────────────────────────────────────┘
```

### Core Evaluation Loop

```
Task -> Harness -> env.prepare() -> agent.run() -> env.execute(test_command) -> parse -> TaskResult
```

### Core Library (`src/grist_mill/`)

```
src/grist_mill/
├── schemas/          # Pydantic v2 data models (Artifact, Telemetry, Manifest)
├── interfaces.py     # Abstract base classes (BaseAgent, BaseBenchmark, BaseEnvironment)
├── config.py         # Configuration loading (pydantic-settings, YAML, env vars, CLI)
├── registry/         # Artifact and agent registries (decorator-based registration)
├── harness/          # Harness implementation (wires task -> env -> agent -> result)
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

### Application Modules

| Module | Purpose |
|--------|---------|
| `bench/` | Canonical benchmark runner, experiment config, Docker sandbox, provider management, telemetry |
| `evaluation/` | DockerPiRunner for sandboxed task evaluation, scoring, and result collection |
| `task_generator/` | R task synthesis via tree-sitter AST parsing, templates, quality gates |
| `optimization/` | GEPA skill adapter for evolutionary skill optimization |
| `scripts/` | CLI entrypoints for benchmarking, optimization, task generation, PR mining, and CI |
| `visualizer/` | Next.js dashboard for experiment visualization (bun) |

## Features

### Benchmarking
- **Multi-provider support**: OpenRouter, OpenAI, Anthropic, Gemini, and custom providers via a single config change
- **Flexible environments**: Local-process runner for fast iteration, Docker runner for isolated execution
- **Artifact system**: Tools, MCP servers, and skills as first-class pluggable artifacts
- **Telemetry**: Per-task token usage, latency breakdowns, and provider cost estimation

### Task Synthesis
- **AST-based analysis**: tree-sitter parsing for R source code
- **207 tasks across 20 R packages**: cli, dplyr, ggplot2, tidyr, stringr, purrr, rlang, vctrs, tibble, withr, glue, posterior, bayesplot, officer, flextable, farver, testthat, httr2, lubridate, sf
- **PR mining**: Automated GitHub PR collection with LLM-based quality judging (bug_fix, feature_impl, write_test)
- **Dataset splits**: train/dev/held_out with quality gates and difficulty calibration
- **Kaggle tasks**: Support for Kaggle competition benchmark tasks

### Optimization (GEPA)
- **Evolutionary prompt optimization**: Evolve skill prompts through evaluation feedback
- **Budget management**: Composable stop conditions (max calls, timeout, no-improvement)
- **Multi-model evaluation**: Validate optimized skills across multiple LLM providers

### Reporting & Export
- **Experiment comparison**: Per-task deltas with statistical significance
- **Telemetry aggregation**: Per-model, per-tool, per-experiment summaries
- **Multiple formats**: JSON, CSV, HTML
- **Visualizer dashboard**: Next.js web UI for experiment exploration

## CLI Usage

### grist-mill (core library)

```bash
# Run a benchmark evaluation
uv run grist-mill run --config configs/examples/smoke.yaml

# Preview without executing
uv run grist-mill run --config configs/examples/smoke.yaml --dry-run

# Validate a configuration
uv run grist-mill validate --config configs/examples/single_model.yaml

# List registered artifacts
uv run grist-mill list

# Optimize a skill
uv run grist-mill optimize --config configs/examples/optimize_smoke.yaml

# Generate reports
uv run grist-mill report --results results.json

# Export results
uv run grist-mill export --results results.json --format html
```

### Makefile workflows

```bash
# Run a benchmark experiment
make benchmark CONFIG=configs/experiments/r_bench_smoke.yaml

# Run GEPA optimization
make optimize MODELS=openai MAX_CALLS=30

# Mine tasks from GitHub PRs
make mine REPO=tidyverse/dplyr

# Run tests
make test

# Lint and typecheck
make lint

# CI suite
make ci
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
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GEMINI_API_KEY` | Gemini API key |
| `Z_AI_API_KEY` | Z.AI API key |

See [`.env.example`](.env.example) for the full list of supported environment variables.

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

## Dataset

207 tasks from 20 R packages, split into train/dev/held_out sets:

| Split | Tasks |
|-------|-------|
| train | 100 |
| dev | 33 |
| held_out | 33 |
| kaggle | 13 |
| mined | 29 |

See [PACKAGES.md](PACKAGES.md) for package selection rationale and generation methodology.

## Development

```bash
# Install dependencies
uv sync

# Run tests (skip Docker and provider integration tests)
uv run pytest -m 'not integration_local and not integration_provider' -q

# Run all tests
uv run pytest tests/ -v

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
- R (optional, for task generation and evaluation)

## License

MIT License
