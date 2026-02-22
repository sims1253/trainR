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
| Multi-package Support | ✅ Done (17 packages, 125 tasks) |

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Docker
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [gh CLI](https://cli.github.com/) authenticated (`gh auth login`)
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))

### 2. Setup

```bash
# Clone
git clone <repo-url>
cd trainR

# Set API key in shell config
# Fish: add to ~/.config/fish/config.fish
#   set -gx OPENROUTER_API_KEY "your-key"
# Bash: add to ~/.bashrc
#   export OPENROUTER_API_KEY="your-key"

# Install dependencies
uv sync

# Build Docker image
make docker-build
```

### 3. Generate Tasks

```bash
# Generate from a single package
uv run python scripts/generate_tasks.py --package dplyr --num-tasks 10

# Or use pre-generated tasks (125 tasks from 17 packages)
ls tasks/train/ tasks/dev/ tasks/held_out/
```

### 4. Run Baselines

```bash
# Run baseline with a specific model
make baseline-openai-no-skill
make baseline-openai-skill

# Compare results
uv run python scripts/compare_results.py
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

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `GITHUB_TOKEN` | GitHub PAT (for PR mining) | Via gh CLI |
| `LLM_MODEL_REFLECTION` | Model for GEPA reflection | Default: openrouter/openai/gpt-oss-120b:free |

### Model Selection

| Purpose | Default Model | Alternative |
|---------|---------------|-------------|
| **Task Agent** | `openrouter/openai/gpt-oss-120b:free` | Any OpenRouter model |
| **Reflection/Judge** | Same as task agent | Stronger model recommended |

## Project Structure

```
trainR/
├── task_generator/      # Generate tasks from R packages
│   ├── ast_parser.py    # R code parsing (tree-sitter)
│   ├── templates.py     # Task templates (7 types)
│   ├── quality_gate.py  # Quality scoring
│   └── mined_task.py    # PR mining schemas
│
├── evaluation/          # Evaluate skills on tasks
│   ├── pi_runner.py     # DockerPiRunner (pi CLI)
│   ├── sandbox.py       # EvaluationSandbox
│   └── models.py        # Result types
│
├── optimization/        # GEPA integration
│   ├── adapter.py       # SkillEvaluator
│   └── config.py        # OptimizationConfig
│
├── scripts/             # CLI tools
│   ├── generate_tasks.py
│   ├── evaluate_batch.py
│   ├── mine_prs.py
│   └── compare_results.py
│
├── configs/             # YAML configs
│   ├── baseline_*.yaml  # Baseline configs per model
│   └── repos_to_mine.yaml
│
├── tasks/               # Generated tasks (60/20/20 split)
│   ├── train/           # 75 tasks
│   ├── dev/             # 25 tasks
│   └── held_out/        # 25 tasks
│
├── skills/              # Skill definitions
├── tests/               # pytest suite
├── packages/            # Cloned R packages
├── PACKAGES.md          # Package documentation
└── PLAN.md              # Project roadmap
```

## Commands

```bash
# Setup
make docker-build       # Build evaluation Docker image

# Tasks
make generate-tasks     # Generate from package (PACKAGES=dplyr)

# Evaluation
make evaluate           # Single task evaluation
make baseline-all-free  # Run all free model baselines

# Optimization
make optimize-test      # Quick test (3 calls)
make optimize-fresh     # Full optimization

# Testing & Linting
make test               # Run tests
make lint               # Ruff linter
make format             # Ruff format
uv run ty check .       # Type checking

# PR Mining
uv run python scripts/mine_prs.py --repo tidyverse/dplyr
./scripts/scheduled_mine.sh  # Scheduled mining
```

## Architecture

```
  R Packages (17 packages)
           │
           ▼
  Task Generator (tree-sitter) → 125 Tasks
           │                        │
           │                        ▼
           │    ┌─────────────────────────────────────┐
           │    │         EVALUATION LOOP              │
           │    │                                      │
           │    │  Skill → pi CLI → Tests (Docker)     │
           │    │                │                     │
           │    │          Pass/Fail + Score           │
           │    │                │                     │
           │    │     GEPA Evolution (LiteLLM)         │
           │    └─────────────────────────────────────┘
           │                    │
           │                    ▼
           │           Evolved Skill
           │
           └── Baseline Comparison → Model Report
```

## Packages

See [PACKAGES.md](PACKAGES.md) for the full list of 17 R packages and rationale for each.

## Development Roadmap

See [PLAN.md](PLAN.md) for the full project roadmap.

## Acknowledgments

- [GEPA](https://github.com/gepa-ai/gepa) - Evolutionary prompt optimization
- [OpenRouter](https://openrouter.ai) - Multi-provider LLM access
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM interface
- [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) - R parsing
- [SWE-bench](https://github.com/swe-bench/SWE-bench) - Task collection methodology
- All R package authors (see PACKAGES.md)
