# Posit-gskill Makefile
# Common commands for development and operations
# Uses uv for Python tooling

.PHONY: setup setup-r setup-python docker docker-build docker-run tasks generate-tasks generate-all-tasks clone-packages validate-tasks evaluate optimize test test-python test-r clean clean-docker help install-claude-cli lock-deps experiment-init lint format typecheck check benchmark benchmark-report

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

evaluate:
	@echo "Running evaluation for task: $(TASK_FILE)"
	uv run python scripts/run_evaluation.py \
		--task $(TASK_FILE) \
		--skill $(SKILL_FILE) \
		--verbose

optimize:
	@echo "Running full optimization..."
	uv run python scripts/run_optimization.py \
		--config $(CONFIG_FILE) \
		--tasks $(TASKS_DIR)/

optimize-test:
	@echo "Running optimization test (5 generations)..."
	uv run python scripts/run_optimization.py \
		--config configs/gepa_config_v1.1.yaml \
		--tasks $(TASKS_DIR)/ \
		--generations 5 \
		--test

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
