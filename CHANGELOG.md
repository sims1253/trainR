# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0-alpha] - 2026-02-22

### Added

#### Core Infrastructure
- DockerPiRunner for sandboxed evaluation using `pi` CLI
- Task generator with tree-sitter R AST parsing
- 7 task templates (write_test, add_edge_case, fix_failing_test, etc.)
- Quality gate with composite scoring
- GEPA integration for skill optimization

#### Task Expansion (Phase 5)
- 125 tasks from 17 R packages
- Difficulty distribution: 40% hard, 40% medium, 20% easy
- Split distribution: 75 train, 25 dev, 25 held_out
- Packages: cli, dplyr, ggplot2, tidyr, stringr, purrr, rlang, vctrs, tibble, withr, glue, posterior, bayesplot, officer, flextable, farver, testthat

#### Baseline System (Phase 4)
- Baseline configs for 4 free OpenRouter models:
  - StepFun Step-3.5-Flash
  - OpenAI GPT-OSS-120B
  - NVIDIA Nemotron-3-Nano
  - Minimax M2.5
- No-skill and skill variants for each model
- Comparison script and markdown reports

#### PR Mining System
- GitHub PR mining with `gh` CLI
- LLM judge for task quality evaluation (LiteLLM)
- Structured output with Pydantic models
- Scheduled mining script for cron
- Support for bug_fix, feature_impl, test_writing task types

#### Documentation
- PACKAGES.md with rationale for each package
- PLAN.md with project roadmap
- Updated README.md

### Changed

- Migrated from TestRunner to DockerPiRunner
- Changed API key from Z_AI_API_KEY to OPENROUTER_API_KEY
- Updated reflection model from glm-4.5 to glm-5
- Unified LLM interface to LiteLLM

### Baseline Results

| Model | No-Skill | With Skill | Delta |
|-------|----------|------------|-------|
| Minimax M2.5 | 44.4% | 100.0% | +55.6pp |
| OpenAI GPT-OSS-120B | 33.3% | 83.3% | +50.0pp |
| StepFun Step-3.5-Flash | 50.0% | 50.0% | 0pp |
| NVIDIA Nemotron-3-Nano | 83.3% | 72.2% | -11.1pp |

### Technical Details

- Python 3.12+
- Uses tree-sitter for R AST parsing
- Docker-based evaluation sandbox
- LiteLLM for multi-provider LLM access
- Pydantic for structured data validation
