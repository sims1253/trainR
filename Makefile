# Posit-gskill Makefile
# Three main workflows: benchmark, optimize, mine
# Uses uv for Python tooling

.PHONY: benchmark benchmark-smoke benchmark-all optimize optimize-test mine mine-all compare test lint visualizer-lint visualizer-build validate-contracts ci ci-quick

# =============================================================================
# 1. BENCHMARK: Evaluate a skill on tasks
# =============================================================================

# Usage: make benchmark [CONFIG=configs/experiments/r_bench_smoke.yaml]
#       make benchmark-smoke  # Quick test with smoke config
#       make benchmark-all    # Full benchmark run
#
# The canonical runner (run_experiment.py) supports:
#   --config FILE    Experiment config (required)
#   --output-dir DIR Override output directory
#   --seed N         Random seed for reproducibility
#   --workers N      Number of parallel workers
#   --validate       Validate config without running
#   --dry-run        Show experiment matrix without running
#   --verbose        Enable verbose output
CONFIG ?= configs/experiments/r_bench_smoke.yaml

benchmark:
	@echo "=== BENCHMARK: $(CONFIG) ==="
	@uv run python scripts/run_experiment.py --config $(CONFIG)

# Smoke test: Quick pipeline verification with minimal config
benchmark-smoke:
	@echo "=== BENCHMARK SMOKE TEST ==="
	@uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml

# Full benchmark: Run with full experiment config
# Note: Create a full experiment config in configs/experiments/ for production use
benchmark-all:
	@echo "=== BENCHMARK ALL MODELS ==="
	@uv run python scripts/run_experiment.py --config $(CONFIG)

# =============================================================================
# 2. OPTIMIZE: Iteratively improve skill with GEPA
# =============================================================================

# Usage: make optimize MODELS=openai MAX_CALLS=30
MODELS ?= openai
MAX_CALLS ?= 30

optimize:
	@echo "=== OPTIMIZE: $(MODELS) ($(MAX_CALLS) calls) ==="
	@uv run python scripts/run_optimize.py \
		--models "$(MODELS)" \
		--max-metric-calls $(MAX_CALLS) \
		--output-dir results/optimization/$(shell date +%Y%m%d_%H%M%S)

optimize-test:
	@$(MAKE) optimize MODELS=openai MAX_CALLS=3

# =============================================================================
# 3. MINE: Collect tasks from GitHub PRs
# =============================================================================

# Usage: make mine REPO=tidyverse/dplyr
REPO ?= 
DAYS ?= 30

mine:
	@echo "=== MINE: $(REPO) (last $(DAYS) days) ==="
	@uv run python scripts/mine_prs.py \
		--repo $(REPO) \
		--since-days $(DAYS) \
		--output tasks/mined/

mine-all:
	@uv run python scripts/mine_prs.py \
		--repos-file configs/repos_to_mine.yaml \
		--output tasks/mined/

# =============================================================================
# UTILITIES
# =============================================================================

validate-contracts:
	@uv run python scripts/validate_contracts.py

# =============================================================================
# CI: Continuous Integration
# =============================================================================

# Full CI suite (all checks + smoke experiment dry-run)
ci:
	@echo "=== CI SMOKE SUITE ==="
	@bash scripts/ci_smoke.sh

# Quick CI (skip smoke experiment, ~2 min)
ci-quick:
	@echo "=== CI SMOKE SUITE (quick) ==="
	@bash scripts/ci_smoke.sh --quick

compare:
	@uv run python scripts/compare_results.py --output results/COMPARISON.md

test:
	@uv run pytest tests/ -v

lint:
	@uv run ruff check . --fix
	@uv run ruff format .
	@uv run ty check .

visualizer-lint:
	@echo "=== VISUALIZER LINT ==="
	cd visualizer && bun run lint

visualizer-build:
	@echo "=== VISUALIZER BUILD ==="
	cd visualizer && bun run build

# =============================================================================
# SETUP (retained for development)
# =============================================================================

.PHONY: setup setup-python setup-r docker-build docker-run

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
