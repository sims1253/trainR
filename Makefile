# Posit-gskill Makefile
# Common commands for development and operations
# Uses uv for Python tooling

.PHONY: setup setup-r setup-python docker docker-build docker-run tasks generate-tasks generate-all-tasks clone-packages validate-tasks evaluate optimize test test-python test-r clean clean-docker help install-claude-cli lock-deps experiment-init lint format typecheck check benchmark benchmark-report evaluate-batch baseline-no-skill baseline-skill quick-test compare-baselines baseline-46-no-skill baseline-46-skill baseline-47-no-skill baseline-47-skill baseline-all-models optimize-multi optimize-fresh optimize-full optimize-safe install-pi test-pi eval-pi test-docker-pi baseline-stepfun-no-skill baseline-stepfun-skill baseline-openai-no-skill baseline-openai-skill baseline-nvidia-no-skill baseline-nvidia-skill baseline-minimax-no-skill baseline-minimax-skill baseline-all-free compare-free-models

# Default target
help:
	@echo "Posit-gskill - Evolutionary optimization of Claude Skills for R packages"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Full setup (Python + R + Docker)"
	@echo "  make setup-python    - Install Python dependencies using uv"
	@echo "  make setup-r         - Install R dependencies with renv"
	@echo "  make lock-deps       - Lock all dependencies (uv.lock + renv.lock)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build evaluation Docker image"
	@echo "  make docker-run      - Run interactive Docker container"
	@echo "  make docker-test     - Test Docker environment"
	@echo "  make install-claude-cli - Install Claude CLI (version-pinned)"
	@echo ""
	@echo "Tasks:"
	@echo "  make clone-packages  - Clone all R packages (cli, withr, rlang, vctrs)"
	@echo "  make generate-tasks  - Generate tasks from single package (PACKAGES=cli)"
	@echo "  make generate-all-tasks - Generate tasks from all packages"
	@echo "  make validate-tasks  - Validate all tasks pass quality gates"
	@echo ""
	@echo "Evaluation & Optimization:"
	@echo "  make evaluate        - Run single task evaluation (dry run)"
	@echo "  make optimize        - Run full GEPA optimization"
	@echo "  make optimize-test   - Run short optimization test (5 generations)"
	@echo "  make optimize-multi   - Multi-model optimization (glm-4.5,4.6,4.7)"
	@echo "  make optimize-fresh   - Fresh optimization (no seed skill)"
	@echo "  make optimize-full    - Full optimization (100 metric calls)"
	@echo "  make optimize-safe    - Safe single-model optimization (no rate limits)"
	@echo "  make evaluate-batch  - Run batch evaluation with parallel workers"
	@echo "  make baseline-no-skill - Run no-skill baseline"
	@echo "  make baseline-skill   - Run current skill baseline"
	@echo "  make quick-test       - Run quick test (dev split only)"
	@echo "  make compare-baselines - Compare baseline results"
	@echo "  make baseline-46-no-skill - Run glm-4.6 no-skill baseline"
	@echo "  make baseline-46-skill   - Run glm-4.6 skill baseline"
	@echo "  make baseline-47-no-skill - Run glm-4.7 no-skill baseline"
	@echo "  make baseline-47-skill   - Run glm-4.7 skill baseline"
	@echo "  make baseline-all-models - Run all model baselines"
	@echo "  make compare-all-models  - Compare all model results"
	@echo "  make baseline-all-free   - Run all free model baselines"
	@echo "  make compare-free-models - Compare free model results"
	@echo ""
	@echo "Pi SDK Evaluation (alternative to Docker):"
	@echo "  make install-pi       - Install Pi CLI globally"
	@echo "  make test-pi          - Test Pi runner with a task"
	@echo "  make eval-pi          - Quick Pi evaluation"
	@echo "  make test-docker-pi   - Test Docker Pi runner (sandboxed)"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark       - Run benchmark across all configured models"
	@echo "  make benchmark-report - Generate report from latest benchmark"
	@echo ""
	@echo "Experiment Tracking:"
	@echo "  make experiment-init - Initialize experiment tracking (mlflow)"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all tests (lint + format check + pytest)"
	@echo "  make test-python     - Run Python tests only"
	@echo "  make test-r          - Run R tests only"
	@echo ""
	@echo "Linting & Formatting:"
	@echo "  make lint            - Run ruff linter"
	@echo "  make format          - Format code with ruff"
	@echo "  make typecheck       - Type check with ty"
	@echo "  make check           - Run lint + typecheck"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           - Clean generated files"
	@echo "  make clean-docker    - Remove Docker images and containers"

# =============================================================================
# Setup
# =============================================================================

setup: setup-python setup-r
	@echo "✓ Full setup complete!"

setup-python:
	@echo "Installing Python dependencies with uv..."
	uv sync
	@echo "✓ Python dependencies installed"

setup-r:
	@echo "Installing R dependencies with renv..."
	Rscript requirements.R
	@echo "✓ R dependencies installed"

lock-deps:
	@echo "Locking Python dependencies..."
	uv lock
	@echo "Locking R dependencies..."
	Rscript -e "renv::snapshot()"
	@echo "✓ All dependencies locked"

# =============================================================================
# Docker
# =============================================================================

DOCKER_IMAGE := posit-gskill-eval
DOCKER_TAG := latest

docker-build:
	@echo "Building Docker image..."
	docker build \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		-f docker/Dockerfile.evaluation .
	@echo "✓ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

docker-run:
	docker run -it --rm \
		-v $(PWD):/workspace \
		--env-file .env \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		bash

docker-test:
	@echo "Testing Docker environment..."
	@echo "Checking R packages..."
	docker run --rm --entrypoint Rscript $(DOCKER_IMAGE):$(DOCKER_TAG) -e "library(testthat); library(devtools); library(cli); library(withr); cat('✓ R packages OK\n')"
	@echo "Checking cc-mirror..."
	docker run --rm --entrypoint which $(DOCKER_IMAGE):$(DOCKER_TAG) cc-mirror || echo "⚠ cc-mirror not found"
	@echo "✓ Docker environment OK"

install-claude-cli:
	@echo "Installing Claude CLI version $(CLAUDE_CLI_VERSION)..."
	./scripts/install_claude_cli.sh $(CLAUDE_CLI_VERSION)
	@echo "✓ Claude CLI installed"

# =============================================================================
# Task Generation
# =============================================================================

PACKAGES ?= cli
ALL_PACKAGES := cli withr rlang vctrs
TASKS_DIR := tasks

clone-packages:
	@echo "Cloning R packages..."
	@for pkg in $(ALL_PACKAGES); do \
		if [ ! -d "packages/$$pkg" ]; then \
			echo "Cloning r-lib/$$pkg..."; \
			git clone --depth 1 https://github.com/r-lib/$$pkg.git packages/$$pkg; \
		else \
			echo "$$pkg already exists"; \
		fi; \
	done
	@echo "All packages cloned"

generate-tasks:
	@echo "Generating tasks from package: $(PACKAGES)"
	uv run python scripts/generate_tasks.py \
		--package $(PACKAGES) \
		--output $(TASKS_DIR)/

generate-all-tasks: clone-packages
	@echo "Generating tasks from all packages..."
	@for pkg in $(ALL_PACKAGES); do \
		echo "Generating tasks from $$pkg..."; \
		uv run python scripts/generate_tasks.py \
			--package $$pkg \
			--output $(TASKS_DIR)/ \
			--num-tasks 25; \
	done
	@echo "All tasks generated"

validate-tasks:
	@echo "Validating tasks..."
	uv run python scripts/validate_tasks.py \
		--tasks $(TASKS_DIR)/

# =============================================================================
# Evaluation & Optimization
# =============================================================================

TASK_FILE ?= tasks/train/task-001.json
SKILL_FILE ?= skills/testing-r-packages-orig.md
CONFIG_FILE ?= configs/gepa_config.yaml

# Batch evaluation config
EVAL_CONFIG ?= configs/evaluation.yaml

# Optimization settings
OPT_MODELS ?= glm-4.5,glm-4.6,glm-4.7
OPT_AGGREGATION ?= min
OPT_MAX_CALLS ?= 50

evaluate:
	@echo "Running evaluation for task: $(TASK_FILE)"
	uv run python scripts/run_evaluation.py \
		--task $(TASK_FILE) \
		--skill $(SKILL_FILE) \
		--verbose

optimize:
	@echo "Running full optimization..."
	uv run python scripts/run_optimization.py \
		--seed-skill $(SKILL_FILE) \
		--tasks-dir $(TASKS_DIR)

optimize-test:
	@echo "Running optimization test (5 metric calls)..."
	uv run python scripts/run_optimization.py \
		--seed-skill $(SKILL_FILE) \
		--tasks-dir $(TASKS_DIR) \
		--max-metric-calls 5 \
		--verbose

# Multi-model optimization (optimize for all models)
optimize-multi:
	@echo "Running multi-model optimization..."
	@echo "Models: $(OPT_MODELS)"
	@echo "Aggregation: $(OPT_AGGREGATION)"
	uv run python scripts/run_optimization.py \
		--models "$(OPT_MODELS)" \
		--aggregation $(OPT_AGGREGATION) \
		--max-metric-calls $(OPT_MAX_CALLS) \
		--verbose

# Fresh optimization (start from empty skill)
optimize-fresh:
	@echo "Running fresh optimization (no seed skill)..."
	uv run python scripts/run_optimization.py \
		--no-skill \
		--models "$(OPT_MODELS)" \
		--aggregation $(OPT_AGGREGATION) \
		--max-metric-calls $(OPT_MAX_CALLS) \
		--verbose

# Full optimization (more iterations)
optimize-full:
	@echo "Running full optimization..."
	uv run python scripts/run_optimization.py \
		--no-skill \
		--models "$(OPT_MODELS)" \
		--aggregation $(OPT_AGGREGATION) \
		--max-metric-calls 100 \
		--verbose

# Safe optimization (single model, sequential - avoids rate limits)
optimize-safe:
	@echo "Running safe single-model optimization..."
	uv run python scripts/run_optimization.py \
		--seed-skill $(SKILL_FILE) \
		--model glm-4.5 \
		--parallel false \
		--max-metric-calls 20 \
		--verbose

# Batch evaluation with parallel execution
evaluate-batch:
	@echo "Running batch evaluation..."
	uv run python scripts/evaluate_batch.py --config $(EVAL_CONFIG)

# Baseline comparisons
baseline-no-skill:
	@echo "Running no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill.yaml

baseline-skill:
	@echo "Running skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill.yaml

# Quick test for development
quick-test:
	@echo "Running quick test..."
	uv run python scripts/evaluate_batch.py --config configs/quick_test.yaml

# Compare baseline results
compare-baselines:
	@echo "Comparing baseline results..."
	@latest_no_skill=$$(ls -t results/baselines/eval_*_no_skill_*.json 2>/dev/null | head -1); \
	latest_skill=$$(ls -t results/baselines/eval_*_testing-r-packages-orig_*.json 2>/dev/null | head -1); \
	if [ -z "$$latest_no_skill" ] || [ -z "$$latest_skill" ]; then \
		echo "Run 'make baseline-no-skill' and 'make baseline-skill' first."; \
		exit 1; \
	fi; \
	echo "No-skill: $$latest_no_skill"; \
	echo "With skill: $$latest_skill"; \
	uv run python -c "\
import json; \
ns = json.load(open('$$latest_no_skill')); \
sk = json.load(open('$$latest_skill')); \
print(f\"No-skill pass rate: {ns['summary']['pass_rate']:.1%}\"); \
print(f\"With skill pass rate: {sk['summary']['pass_rate']:.1%}\"); \
print(f\"Improvement: {(sk['summary']['pass_rate'] - ns['summary']['pass_rate'])*100:.1f}pp\")\
"

# Model-specific baselines
baseline-46-no-skill:
	@echo "Running glm-4.6 no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill_46.yaml

baseline-46-skill:
	@echo "Running glm-4.6 skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill_46.yaml

baseline-47-no-skill:
	@echo "Running glm-4.7 no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill_47.yaml

baseline-47-skill:
	@echo "Running glm-4.7 skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill_47.yaml

# Run all model baselines (glm-4.5, 4.6, 4.7)
baseline-all-models: baseline-no-skill baseline-skill baseline-46-no-skill baseline-46-skill baseline-47-no-skill baseline-47-skill
	@echo "All model baselines complete!"

# Compare all models
compare-all-models:
	@echo "Comparing all model baselines..."
	@uv run python -c "\
import json, glob; \
results = {}; \
for f in sorted(glob.glob('results/baselines/eval_*.json'))[-6:]: \
    d = json.load(open(f)); \
    model = d['config']['model']; \
    skill = d['config']['skill']; \
    rate = d['summary']['pass_rate']; \
    key = f'{model}/{skill}'; \
    results[key] = rate; \
print('\\n| Model | Skill | No-Skill | Delta |'); \
print('|-------|-------|----------|-------|'); \
for m in ['glm-4.5', 'glm-4.6', 'glm-4.7']: \
    sk = results.get(f'{m}/testing-r-packages-orig', 0); \
    ns = results.get(f'{m}/no_skill', 0); \
    delta = (sk - ns) * 100; \
    print(f'| {m} | {sk:.1%} | {ns:.1%} | {delta:+.1f}pp |')\
"

# === Free Model Baselines (DockerPiRunner) ===

# StepFun Step-3.5-Flash
baseline-stepfun-no-skill:
	@echo "Running StepFun no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill_stepfun.yaml

baseline-stepfun-skill:
	@echo "Running StepFun skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill_stepfun.yaml

# OpenAI GPT-OSS-120B
baseline-openai-no-skill:
	@echo "Running OpenAI no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill_openai.yaml

baseline-openai-skill:
	@echo "Running OpenAI skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill_openai.yaml

# NVIDIA Nemotron
baseline-nvidia-no-skill:
	@echo "Running NVIDIA no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill_nvidia.yaml

baseline-nvidia-skill:
	@echo "Running NVIDIA skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill_nvidia.yaml

# Minimax M2.5
baseline-minimax-no-skill:
	@echo "Running Minimax no-skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_no_skill_minimax.yaml

baseline-minimax-skill:
	@echo "Running Minimax skill baseline..."
	uv run python scripts/evaluate_batch.py --config configs/baseline_skill_minimax.yaml

# Run all free model baselines
baseline-all-free: baseline-stepfun-no-skill baseline-stepfun-skill baseline-openai-no-skill baseline-openai-skill baseline-nvidia-no-skill baseline-nvidia-skill baseline-minimax-no-skill baseline-minimax-skill
	@echo "All free model baselines complete!"

# Compare all free model baselines
compare-free-models:
	@echo "Comparing all free model baselines..."
	@uv run python -c "\
import json, glob; \
results = {}; \
for f in sorted(glob.glob('results/baselines/eval_*.json')): \
    d = json.load(open(f)); \
    model = d['config']['model']; \
    skill = d['config']['skill']; \
    rate = d['summary']['pass_rate']; \
    results[(model, skill)] = rate; \
print('\\n| Model | Skill | No-Skill | Delta |'); \
print('|-------|-------|----------|-------|'); \
models = ['openrouter/stepfun/step-3.5-flash:free', 'openrouter/openai/gpt-oss-120b:free', 'openrouter/nvidia/nemotron-3-nano-30b-a3b:free', 'opencode/minimax-m2.5-free']; \
for m in models: \
    sk = results.get((m, 'testing-r-packages-orig'), 0); \
    ns = results.get((m, 'no_skill'), 0); \
    delta = (sk - ns) * 100; \
    name = m.split('/')[-1].replace(':free', ''); \
    print(f'| {name} | {sk:.1%} | {ns:.1%} | {delta:+.1f}pp |')\
"

# === Pi SDK Evaluation ===

install-pi:
	@echo "Installing Pi CLI..."
	npm install -g @mariozechner/pi-coding-agent
	@echo "Pi installed. Run 'pi auth login' to configure providers."

test-pi:
	@echo "Testing Pi runner with a simple task..."
	uv run python scripts/test_pi_runner.py

eval-pi:
	@echo "Running evaluation with Pi (no Docker)..."
	uv run python -c "\
from evaluation.pi_runner import PiRunner, PiRunnerConfig; \
from pathlib import Path; \
from task_generator import TaskGenerator; \
gen = TaskGenerator(Path('tasks')); \
tasks = gen.load_all_tasks(split='dev'); \
config = PiRunnerConfig(model='openrouter/google/gemini-2.0-flash', max_turns=30); \
runner = PiRunner(config); \
skill = Path('skills/testing-r-packages-orig.md').read_text() if Path('skills/testing-r-packages-orig.md').exists() else ''; \
result = runner.run_evaluation(skill, tasks[0].instruction, tasks[0].context, Path('packages') / tasks[0].source_package); \
print(f'Success: {result["success"]}, Time: {result["execution_time"]:.1f}s')\
"

test-docker-pi:
	@echo "Testing Docker Pi runner (sandboxed)..."
	uv run python scripts/test_docker_pi_runner.py

# =============================================================================
# Benchmarking
# =============================================================================

BENCHMARK_CONFIG ?= configs/benchmark_models.yaml

benchmark:
	@echo "Running benchmark..."
	uv run python scripts/run_benchmark.py \
		--config $(BENCHMARK_CONFIG) \
		--tasks-dir $(TASKS_DIR) \
		--skill $(SKILL_FILE) \
		--verbose

benchmark-report:
	@echo "Generating benchmark report..."
	@latest=$$(ls -td results/benchmarks/*/benchmark_results.json 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then \
		echo "No benchmark results found. Run 'make benchmark' first."; \
		exit 1; \
	fi; \
	uv run python scripts/benchmark_report.py "$$latest" -o results/benchmarks/latest_report.md

# =============================================================================
# Experiment Tracking
# =============================================================================

experiment-init:
	@echo "Initializing experiment tracking..."
	uv run mlflow server \
		--backend-store-uri sqlite:///experiments/mlflow.db \
		--default-artifact-root ./experiments/artifacts \
		--host 0.0.0.0 \
		--port 5000 &

# =============================================================================
# Testing
# =============================================================================

lint:
	@echo "Running ruff linter..."
	uv run ruff check .

format:
	@echo "Formatting with ruff..."
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	@echo "Type checking with ty..."
	uv run ty check task_generator/ evaluation/ optimization/ benchmark/ scripts/ tests/

check: lint typecheck
	@echo "All checks passed!"

test:
	@echo "Running lint check..."
	uv run ruff check .
	@echo "Running format check..."
	uv run ruff format --check .
	@echo "Running Python tests..."
	uv run pytest tests/ -v

test-python:
	@echo "Running Python tests with uv..."
	uv run pytest tests/ -v --tb=short

test-r:
	@echo "Running R tests..."
	Rscript -e "testthat::test_dir('tests/')"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/
	@echo "✓ Clean complete"

clean-docker:
	@echo "Removing Docker artifacts..."
	docker container prune -f
	docker image prune -f
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	@echo "✓ Docker cleanup complete"
