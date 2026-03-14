# trainR

> Evolutionary optimization of AI coding skills for R package testing using GEPA

TrainR automatically improves AI coding skills for R package testing by generating tasks, running agents in Docker, and evolving skill prompts. It also serves as a benchmark harness for comparing model performance on R testing tasks.

## Status: v0.1.0-alpha

Early development release. Core functionality works, but APIs may change.

| Component | Status |
|-----------|--------|
| Task Generator | ✅ Done |
| Evaluation (DockerPiRunner) | ✅ Done |
| GEPA Integration | ✅ Done |
| Baseline Comparisons | ✅ Done |
| PR Mining | ✅ Done |
| Multi-package Support | ✅ Done (20 packages, 138 tasks) |

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Docker
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [gh CLI](https://cli.github.com/) authenticated (`gh auth login`)
- API key for at least one provider (OpenRouter, OpenCode, or z.ai)

### 2. Setup

```bash
# Clone
git clone <repo-url>
cd trainR

# Create local env file (single source of truth for project secrets)
cp .env.example .env

# Edit .env and set at least one of:
# OPENROUTER_API_KEY=...
# OPENCODE_API_KEY=...
# Z_AI_API_KEY=...

# Install dependencies
uv sync

# Build Docker image
make docker-build
```

### 3. Generate Tasks

```bash
# Generate from a single package
uv run python scripts/generate_tasks.py --package dplyr --num-tasks 10

# Or use pre-generated tasks (138 tasks from 20 packages)
ls tasks/train/ tasks/dev/ tasks/held_out/
```

### 4. Run Baselines

```bash
# Run baseline evaluation with canonical runner
make benchmark

# Quick smoke test (verifies pipeline works)
make benchmark-smoke

# Run first benchmark (1 task x 3 providers)
uv run python scripts/run_experiment.py --config configs/experiments/first_benchmark.yaml

# Run with a specific experiment config
uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml

# Run with custom output directory and seed
uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml \
    --output-dir results/my_run --seed 42

# Capture raw Docker stdout/stderr logs per run (for debugging)
uv run python scripts/run_experiment.py --config configs/experiments/first_benchmark.yaml \
    --save-container-logs
```

### 5. Run Optimization

```bash
# Quick test (3 metric calls)
make optimize-test

# Full optimization (30+ calls)
make optimize-fresh OPT_MAX_CALLS=30
```

### 6. Mine Tasks from GitHub PRs

```bash
# Mine from a single repo
uv run python scripts/mine_prs.py --repo tidyverse/dplyr --since-days 30

# Mine from configured repos
uv run python scripts/mine_prs.py --repos-file configs/repos_to_mine.yaml
```

## Configuration

### Environment Variables

trainR automatically loads project-local `.env` values.
Supported aliases are normalized automatically (for example `ZAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `OPENCODE_API_TOKEN`).

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | One provider key required |
| `OPENCODE_API_KEY` | OpenCode API key | One provider key required |
| `Z_AI_API_KEY` | z.ai API key (canonical) | One provider key required |
| `ZAI_API_KEY` | z.ai alias (auto-normalized) | Optional |
| `GITHUB_TOKEN` | GitHub PAT (for PR mining) | Optional (or use gh CLI) |
| `LLM_MODEL_REFLECTION` | Model for GEPA reflection | Default: from `configs/llm.yaml` |

### Model Selection

| Purpose | Default Model | Alternative |
|---------|---------------|-------------|
| **Task Agent** | `openrouter/openai/gpt-oss-120b:free` | Any OpenRouter model |
| **Reflection/Judge** | Same as task agent | Stronger model recommended |

## Project Structure

```
trainR/
├── bench/                # Canonical benchmark infrastructure
│   ├── runner.py         # Canonical execution API
│   ├── experiments/      # Experiment runner and config
│   ├── schema/v1/        # Canonical schemas (task, manifest, results)
│   ├── dataset/          # Dataset management
│   ├── eval/             # Evaluation utilities
│   ├── optimize/         # GEPA optimization
│   └── reports/          # Result reporting
│
├── evaluation/           # Docker-based evaluation
│   ├── pi_runner.py      # DockerPiRunner
│   └── sandbox.py        # EvaluationSandbox
│
├── scripts/              # CLI tools
│   ├── run_experiment.py # CANONICAL benchmark runner
│   ├── run_optimize.py   # Optimization runner
│   ├── generate_tasks.py # Task generation
│   └── mine_prs.py       # PR mining
│
├── configs/              # YAML configs
│   └── experiments/      # Experiment configs (*.yaml)
│
├── tasks/                # Generated tasks (60/20/20 split)
│   ├── train/            # 83 tasks
│   ├── dev/              # 28 tasks
│   ├── held_out/         # 27 tasks
│   └── kaggle/           # 12 tasks (from 12 competitions)
│
├── skills/               # Skill definitions
├── tests/                # pytest suite
├── packages/             # Cloned R packages
├── PACKAGES.md           # Package documentation
└── PLAN.md               # Project roadmap
```

## Commands

```bash
# Setup
make docker-build       # Build evaluation Docker image

# Tasks
make generate-tasks     # Generate from package (PACKAGES=dplyr)

# Benchmark (canonical runner)
make benchmark          # Run: scripts/run_experiment.py --config configs/experiments/first_benchmark.yaml
make benchmark-smoke    # Quick smoke test (verifies pipeline)

# Direct experiment runner
uv run python scripts/run_experiment.py --config configs/experiments/first_benchmark.yaml
uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml
uv run python scripts/run_experiment.py --config configs/experiments/support_pair_smoke.yaml

# Optimization
make optimize-test      # Quick test (3 calls)
make optimize-fresh     # Full optimization (uses scripts/run_optimize.py)

# Testing & Linting
make test               # Run tests
make lint               # Ruff linter
make format             # Ruff format
uv run ty check .       # Type checking

# PR Mining
uv run python scripts/mine_prs.py --repo tidyverse/dplyr
./scripts/scheduled_mine.sh  # Scheduled mining
```

## Supported Entrypoints

All benchmark execution must go through the canonical runner API:

| Entrypoint | Type | Use Case |
|------------|------|----------|
| `bench.runner.run()` | Library API | Python code, notebooks, tests |
| `scripts/run_experiment.py` | CLI | Shell commands, CI pipelines |

Both entrypoints delegate to the same execution path, ensuring consistent behavior.

```python
# Library API example
from bench.runner import run

manifest = run("configs/experiments/smoke.yaml")
print(f"Pass rate: {manifest.summary.pass_rate:.1%}")

# With overrides
manifest = run("config.yaml", output_dir="results/custom", seed=42, workers=2)
```

```bash
# CLI equivalent
uv run python scripts/run_experiment.py --config config.yaml \
    --output-dir results/custom --seed 42 --workers 2
```

**Important**: Do not create new entrypoints that bypass the canonical runner.
All execution paths must delegate to `bench.runner.run()` to ensure:

- Consistent artifact generation (manifest.json, results.jsonl, summary.json)
- Unified telemetry and logging
- Proper sandbox and credential handling
- Reproducible fingerprinting and configuration snapshots

## Architecture

```
  R Packages (20 packages)
           │
           ▼
   Task Generator (tree-sitter) → Tasks
           │                        │
           │                        ▼
           │      ┌────────────────────────────────────┐
           │      │         EVALUATION LOOP            │
           │      │                                    │
           │      │  Skill → pi CLI → Tests (Docker)   │
           │      │                │                   │
           │      │          Pass/Fail + Score         │
           │      │                │                   │
            │      │     GEPA Evolution (Gateway)      │
           │      └────────────────────────────────────┘
           │                        │
           │                        ▼
           │                  Evolved Skill
           │
           └── Baseline Comparison → Model Report
```

## Packages

See [PACKAGES.md](PACKAGES.md) for the full list of 20 R packages and rationale for each.

## Development Roadmap

See [PLAN.md](PLAN.md) for the full project roadmap.

## Acknowledgments

- [GEPA](https://github.com/gepa-ai/gepa) - Evolutionary prompt optimization
- [OpenRouter](https://openrouter.ai) - Multi-provider LLM access
- [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) - R parsing
- [SWE-bench](https://github.com/swe-bench/SWE-bench) - Task collection methodology
- All R package authors (see PACKAGES.md)

### Historical

- [LiteLLM](https://github.com/BerriAI/litellm) - Previous unified LLM interface (superseded by provider-native inference gateway)
