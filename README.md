# trainR

> Evolutionary optimization of Claude Skills for R package testing using GEPA

Automatically improve Claude Code skills for R package testing by generating synthetic tasks, running agents in Docker, and evolving skill prompts based on failures. Also serves as a benchmark harness for comparing model performance on R testing tasks.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- z.ai API key (get one at [z.ai/model-api](https://z.ai/model-api))

### 2. Setup

```bash
# Clone and enter
git clone <repo-url>
cd trainR

# Set your API key in shell config (not in project files)
# Fish: add to ~/.config/fish/conf.d/secrets.fish
#   set -gx Z_AI_API_KEY "your-zai-api-key"
# Bash/Zsh: add to ~/.bashrc or ~/.zshrc
#   export Z_AI_API_KEY="your-zai-api-key"

# Install Python dependencies
uv sync

# Build Docker image for evaluation
make docker-build
```

### 3. Generate Tasks

```bash
# Generate 15 testing tasks from cli package
uv run python scripts/generate_tasks.py --package cli --num-tasks 15

# Generate tasks from all packages (cli, withr, rlang, vctrs)
make generate-all-tasks

# View generated tasks
ls tasks/train/ tasks/dev/ tasks/held_out/
```

### 4. Run a Single Evaluation

```bash
uv run python scripts/run_evaluation.py \
  --task tasks/train/task-19305555.json \
  --skill skills/testing-r-packages-orig.md \
  --verbose
```

### 5. Run Optimization

```bash
# Short optimization test (5 generations)
make optimize-test

# Full optimization (50 generations)
make optimize
```

### 6. Run Benchmark

```bash
# Compare models across the task set
make benchmark

# Generate report from results
make benchmark-report
```

## Testing the System

### Verify setup (no API key needed)

```bash
# 1. Python imports work
uv run python -c "from evaluation import EvaluationSandbox; from optimization import optimize_skill; from benchmark import BenchmarkRun; print('OK')"

# 2. Run unit tests
uv run pytest tests/ -v

# 3. Lint checks pass
make lint
```

### Verify Docker (no API key needed for build)

```bash
# Build the image
make docker-build

# Test R packages are installed
make docker-test
```

### Verify task generation (no API key needed)

```bash
# Clone cli package and generate tasks
uv run python scripts/generate_tasks.py --package cli --num-tasks 10 --verbose

# Check quality scores (should be >= 0.5)
uv run python -c "
from task_generator import TaskGenerator
gen = TaskGenerator('tasks')
tasks = gen.load_all_tasks()
for t in sorted(tasks, key=lambda x: x.quality_score):
    print(f'{t.task_id}: score={t.quality_score:.2f} difficulty={t.difficulty} type={t.test_type}')
"
```

### End-to-end evaluation (requires Z_AI_API_KEY)

```bash
# Pick any task and run it
uv run python scripts/run_evaluation.py \
  --task tasks/dev/$(ls tasks/dev/ | head -1) \
  --skill skills/testing-r-packages-orig.md \
  --verbose
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `Z_AI_API_KEY` | z.ai API key (set in shell config, not .env) | - |
| `LLM_MODEL_REFLECTION` | Model for GEPA reflection | `openai/glm-5` |
| `LLM_API_BASE` | API endpoint | `https://api.z.ai` |

### Model Selection

| Purpose | Model | Why |
|---------|-------|-----|
| **Task Agent** | `glm-4.7-flash` | Fast, efficient for test generation |
| **Reflection/Judge** | `glm-5` | Stronger reasoning for optimization |

### Benchmark Model Config

Edit `configs/benchmark_models.yaml` to define which models to compare:

```yaml
models:
  - name: glm-4.7-flash
    provider: zai
    cc_mirror_provider: zai
    env:
      ANTHROPIC_BASE_URL: "https://api.z.ai/api/anthropic"
```

## Project Structure

```
trainR/
├── task_generator/      # Generate tasks from R packages
│   ├── ast_parser.py    # R code parsing (tree-sitter)
│   ├── pattern_extractor.py  # Extract test patterns
│   ├── templates.py     # Task templates (7 types)
│   ├── quality_gate.py  # Composite quality scoring (min 0.5)
│   └── generator.py     # Main pipeline
│
├── evaluation/          # Evaluate skills on tasks
│   ├── test_runner.py   # Docker test execution
│   ├── sandbox.py       # Orchestration
│   └── models.py        # Result types
│
├── optimization/        # GEPA integration
│   ├── adapter.py       # SkillEvaluator + optimize_skill
│   └── config.py        # OptimizationConfig
│
├── benchmark/           # Benchmark infrastructure
│   └── schema.py        # BenchmarkResult, BenchmarkRun
│
├── tasks/               # Generated tasks (60/20/20 split)
│   ├── train/
│   ├── dev/
│   └── held_out/
│
├── skills/              # Skill definitions
├── configs/             # YAML configs (GEPA, benchmark models)
├── docker/              # Dockerfile + entrypoint
├── scripts/             # CLI tools
├── tests/               # pytest suite
└── packages/            # Cloned R packages (cli, withr, rlang, vctrs)
```

## Commands

```bash
# Setup
make setup              # Full setup (Python + R)
make docker-build       # Build evaluation Docker image
make docker-test        # Test Docker environment

# Tasks
make clone-packages     # Clone all R packages
make generate-tasks     # Generate from single package (PACKAGES=cli)
make generate-all-tasks # Generate from all packages
make validate-tasks     # Validate task quality

# Evaluation & Optimization
make evaluate           # Single task evaluation
make optimize           # Full GEPA optimization
make optimize-test      # Short optimization (5 generations)

# Benchmarking
make benchmark          # Run all models across task set
make benchmark-report   # Generate comparison report

# Testing & Linting
make test               # Run all tests (lint + format + pytest)
make lint               # Run ruff linter
make format             # Format with ruff
make check              # Lint + typecheck
```

## Architecture

```
  R Packages (cli, withr, rlang, vctrs)
           │
           ▼
  Task Generator (tree-sitter AST) → Task Dataset
           │                           │
           │                           ▼
           │    ┌─────────────────────────────────────┐
           │    │         EVALUATION LOOP              │
           │    │                                      │
           │    │  Skill → Agent (cc-mirror) → Tests   │
           │    │                    │                  │
           │    │              Docker Runner            │
           │    │                    │                  │
           │    │              Pass/Fail + Score        │
           │    │                    │                  │
           │    │         Trajectory → Reflection       │
           │    │                         │             │
           │    │     GEPA Evolution ←────┘             │
           │    └─────────────────────────────────────┘
           │                    │
           │                    ▼
           │           Evolved Skill
           │
           ├── Benchmark Runner → Model Comparison Report
           └── Trajectory Logs → Failure Analysis
```

## Development Status

| Component | Status |
|-----------|--------|
| Task Generator | Done |
| Evaluation Sandbox | Done |
| Docker Environment | Done |
| GEPA Adapter | Done |
| Quality Gate (0.5 threshold) | Done |
| Benchmark Infrastructure | Done |
| Multi-package Support | Done |
| Optimization Loop | Needs end-to-end testing |

## Acknowledgments

- [GEPA](https://github.com/gepa-ai/gepa) - Evolutionary prompt optimization
- [z.ai](https://z.ai) - GLM models and Claude Code integration
- [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) - R parsing
- [r-lib/cli](https://github.com/r-lib/cli), [r-lib/withr](https://github.com/r-lib/withr), [r-lib/rlang](https://github.com/r-lib/rlang), [r-lib/vctrs](https://github.com/r-lib/vctrs) - Source packages
