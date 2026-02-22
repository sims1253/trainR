# Posit-gskill Makefile
# Three main workflows: benchmark, optimize, mine
# Uses uv for Python tooling

.PHONY: benchmark benchmark-all optimize optimize-test mine mine-all compare test lint

# =============================================================================
# 1. BENCHMARK: Evaluate a skill on tasks
# =============================================================================

# Usage: make benchmark MODEL=openai SKILL=skill
MODEL ?= openai
SKILL ?= no_skill

benchmark:
	@echo "=== BENCHMARK: $(MODEL) / $(SKILL) ==="
	@uv run python scripts/evaluate_batch.py \
		--model $(MODEL) \
		$(if $(filter no_skill,$(SKILL)),--no-skill,--skill skills/testing-r-packages-orig.md) \
		--output results/benchmark/$(MODEL)_$(SKILL).json

benchmark-all:
	@for model in openai minimax nvidia stepfun; do \
		$(MAKE) benchmark MODEL=$$model SKILL=no_skill; \
		$(MAKE) benchmark MODEL=$$model SKILL=skill; \
	done

# =============================================================================
# 2. OPTIMIZE: Iteratively improve skill with GEPA
# =============================================================================

# Usage: make optimize MODELS=openai MAX_CALLS=30
MODELS ?= openai
MAX_CALLS ?= 30

optimize:
	@echo "=== OPTIMIZE: $(MODELS) ($(MAX_CALLS) calls) ==="
	@uv run python scripts/run_optimization.py \
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

compare:
	@uv run python scripts/compare_results.py --output results/COMPARISON.md

test:
	@uv run pytest tests/ -v

lint:
	@uv run ruff check . --fix
	@uv run ruff format .
	@uv run ty check .

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
