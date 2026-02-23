# trainR Bench+ Work Packages (Execution Tickets)

This document converts `ORCHESTRATOR_IMPLEMENTATION_PLAN.md` into implementable tickets for an agent orchestrator.

## Usage Contract
- Execute tickets in order unless marked parallelizable.
- After each ticket: run listed checks and attach output.
- Do not move to next phase gate until all acceptance criteria pass.

---

## Phase 0 Gate: Foundation Hardening

## WP-01A: Makefile/CLI Contract Repair

### Goal
Make baseline benchmark commands runnable and internally consistent.

### Files
- `Makefile`
- `scripts/evaluate_batch.py`
- `README.md`

### Tasks
1. Update `make benchmark` to pass `--config` and valid flags accepted by `scripts/evaluate_batch.py`.
2. Remove invalid/legacy flags from Makefile targets.
3. Add `benchmark-smoke` target:
   - 1 model
   - 1 task split
   - `--max-tasks 1`
4. Update README commands to match actual targets.

### Commands
- `make benchmark-smoke`
- `uv run python scripts/evaluate_batch.py --help`

### Acceptance tests
- `make benchmark-smoke` exits 0.
- No argparse error about missing required flags.
- README “Run Baselines” section references real targets.

---

## WP-01B: Packaging and Entrypoint Integrity

### Goal
Ensure installable CLI entrypoints map to existing modules and wheel config is valid.

### Files
- `pyproject.toml`
- (create if needed) `src/posit_gskill/cli.py` or adjust to existing CLI module

### Tasks
1. Fix `[project.scripts]` entrypoint to valid module path.
2. Remove non-existent package entries from wheel config.
3. Ensure package discovery/build backend includes actual package directories.

### Commands
- `uv run python -m build`
- `uv run posit-gskill --help` (or updated script name)

### Acceptance tests
- Build succeeds.
- Script entrypoint runs without `ModuleNotFoundError`.

---

## WP-01C: Config Contract Validator

### Goal
Fail fast on model/config drift.

### Files
- `scripts/validate_contracts.py` (new)
- `configs/llm.yaml`
- `configs/benchmark.yaml`

### Tasks
1. Add validator script that checks:
   - benchmark model names resolve in llm config
   - referenced profile/config files exist
2. Add Makefile target `validate-contracts`.
3. Integrate into CI command list (if CI config exists, update there).

### Commands
- `uv run python scripts/validate_contracts.py`
- `make validate-contracts`

### Acceptance tests
- Validator returns non-zero on mismatch and prints actionable diff.
- Returns zero on current corrected config.

---

## WP-01D: Provider-Aware Preflight + Error Taxonomy

### Goal
Require only relevant credentials and classify infra/config failures correctly.

### Files
- `scripts/run_benchmark.py`
- `scripts/evaluate_batch.py`
- `evaluation/sandbox.py`
- `evaluation/models.py`

### Tasks
1. Replace global key checks with selected-provider checks.
2. Add/normalize failure categories:
   - `CONFIG_ERROR`
   - `ENVIRONMENT_ERROR`
   - `PACKAGE_NOT_FOUND`
   - `TIMEOUT`
   - `TEST_FAILURE`
3. Update classification logic in `EvaluationSandbox`.
4. Add tests for key-missing/package-missing classification.

### Commands
- `uv run pytest -q tests/test_evaluation_config.py tests/test_optimization.py`
- `uv run python scripts/run_benchmark.py --task tasks/dev/task-0b71dca5.json --worker-model <valid_model> --output-dir /tmp/trainr_smoke`

### Acceptance tests
- Missing irrelevant provider key does not block run.
- Missing relevant key yields `CONFIG_ERROR`/`ENVIRONMENT_ERROR` (not syntax).

---

## Phase 1 Gate: R SWE-Bench Core

## WP-02A: Canonical Schema v1

### Goal
Implement canonical task/profile/result schemas.

### Files
- `bench/schema/v1/task.py` (new)
- `bench/schema/v1/profiles.py` (new)
- `bench/schema/v1/results.py` (new)
- `bench/schema/v1/*.schema.json` (new)

### Tasks
1. Define Pydantic models + JSON schemas.
2. Support `task_type` values: `swe`, `test_gen`, `skill`.
3. Add schema validation helpers.

### Commands
- `uv run python -c "from bench.schema.v1.task import TaskV1; print('ok')"`
- `uv run pytest -q tests`

### Acceptance tests
- Schema models import cleanly.
- Example tasks validate for each task type.

---

## WP-02B: Task Migration Pipeline

### Goal
Migrate legacy tasks to canonical v1 without data loss.

### Files
- `scripts/migrate_tasks_to_v1.py` (new)
- `bench/dataset/migrate.py` (new)
- output: `tasks_v1/` (or dual-write location)

### Tasks
1. Build migration for all legacy task formats.
2. Emit migration report:
   - migrated count
   - failed count + reasons
3. Preserve original tasks untouched.

### Commands
- `uv run python scripts/migrate_tasks_to_v1.py --in tasks --out tasks_v1 --report results/migration_report.json`

### Acceptance tests
- Report produced.
- >=95% legacy tasks migrate or are explicitly flagged with reasons.

---

## WP-02C: Dataset Manager + Split/Decontamination Metadata

### Goal
Add dataset lifecycle manager for ingest/validate/split/export.

### Files
- `bench/dataset/manager.py` (new)
- `bench/dataset/decontam.py` (new)
- `configs/dataset/*.yaml` (new)

### Tasks
1. Implement split tooling (`train/dev/test`).
2. Add metadata fields for freshness/decontamination.
3. Export dataset manifest with fingerprint.

### Commands
- `uv run python -m bench.dataset.manager split --config configs/dataset/default.yaml`

### Acceptance tests
- Deterministic split with fixed seed.
- Manifest includes version/fingerprint and split counts.

---

## WP-03A: Unified Experiment Runner (Core)

### Goal
Create single execution entrypoint.

### Files
- `scripts/run_experiment.py` (new)
- `bench/experiments/runner.py` (new)
- `bench/experiments/config.py` (new)
- `configs/experiments/r_bench_smoke.yaml` (new)

### Tasks
1. Validate experiment config schema.
2. Generate run manifest before execution.
3. Execute experiment matrix for baseline profiles.
4. Write `results.jsonl`, `summary.json`, `manifest.json`.

### Commands
- `uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml`

### Acceptance tests
- Single command runs end-to-end.
- Output artifacts exist and validate against schema.

---

## WP-03B: Legacy Compatibility Wrappers

### Goal
Keep existing scripts functional while migrating.

### Files
- `scripts/run_benchmark.py`
- `scripts/run_parallel_benchmark.py`
- `scripts/evaluate_batch.py`

### Tasks
1. Route legacy scripts through `run_experiment.py` where feasible.
2. Preserve key CLI args with translation layer.
3. Print deprecation warnings with replacement command.

### Commands
- `uv run python scripts/run_benchmark.py --help`
- `uv run python scripts/evaluate_batch.py --help`

### Acceptance tests
- Existing scripts run or fail with clear migration guidance.

---

## Phase 2 Gate: Support Structures

## WP-04A: Support Profile Schema + Resolver

### Goal
Make support structures first-class and reproducible.

### Files
- `bench/profiles/support.py` (new)
- `configs/profiles/support/*.yaml` (new)
- `bench/eval/prompt_builder.py` (new or refactor existing prompt composition)

### Tasks
1. Implement support modes:
   - `none`
   - `system_only`
   - `agents_only`
   - `system_plus_agents`
   - `single_skill`
   - `collection_forced`
   - `collection_selective`
2. Add support fingerprinting.
3. Persist composed support artifact metadata.

### Commands
- `uv run python -c "from bench.profiles.support import SupportProfile; print('ok')"`

### Acceptance tests
- Each mode resolves deterministically.
- Fingerprint stable for same content/order.

---

## WP-04B: Paired Support Evaluation

### Goal
Generate paired runs for fair support comparison.

### Files
- `bench/experiments/matrix.py` (new/refactor)
- `bench/reports/paired.py` (new)

### Tasks
1. Add pairing directives in experiment config.
2. Generate paired rows with same task/model/tool/seed.
3. Emit `paired_deltas.jsonl`.

### Commands
- `uv run python scripts/run_experiment.py --config configs/experiments/support_pair_smoke.yaml`

### Acceptance tests
- Paired records generated with valid join keys.
- Delta metrics computed for pass rate/cost/latency.

---

## WP-04C: Policy-Driven Selective Skill Access

### Goal
Support skill collections without ad hoc prompt commands.

### Files
- `bench/profiles/support.py`
- `bench/eval/skill_policy.py` (new)

### Tasks
1. Implement selective policy decision path.
2. Log selected skill IDs and selection rationale metadata.
3. Ensure selection is reproducible with fixed seed/config.

### Commands
- `uv run pytest -q tests/test_optimization.py tests/test_task_generator.py`

### Acceptance tests
- Selection logs persisted.
- Same input profile+seed yields same selected set (deterministic mode).

---

## Phase 3 Gate: Tool Benchmarking

## WP-05A: Tool Profile + Registry

### Goal
Define toolsets and variants as versioned profiles.

### Files
- `bench/profiles/tools.py` (new)
- `bench/eval/tool_registry.py` (new)
- `configs/profiles/tools/*.yaml` (new)

### Tasks
1. Implement base tool profile and variants.
2. Add versioned tool definitions (`patch_v1`, `patch_v2`, etc.).
3. Add tool profile fingerprinting.

### Commands
- `uv run python -c "from bench.profiles.tools import ToolProfile; print('ok')"`

### Acceptance tests
- Two tool profiles load and validate.
- Profile hash changes when tool definition changes.

---

## WP-05B: A/B Tool Comparison Engine

### Goal
Run controlled tool experiments with constant non-tool dimensions.

### Files
- `bench/experiments/matrix.py`
- `bench/reports/paired.py`

### Tasks
1. Add A/B generator that fixes task/model/support/seed.
2. Vary only `tool_profile`.
3. Emit paired tool deltas and summary significance stats (basic bootstrap optional).

### Commands
- `uv run python scripts/run_experiment.py --config configs/experiments/tool_ab_smoke.yaml`

### Acceptance tests
- Paired tool deltas produced.
- Join keys prove only tool profile differs.

---

## WP-05C: Tool Telemetry and Failure Attribution

### Goal
Track per-tool impact on outcomes.

### Files
- `bench/eval/telemetry.py` (new)
- `bench/schema/v1/results.py`

### Tasks
1. Capture per-tool calls/errors/time.
2. Add normalized failure attribution fields.
3. Include telemetry in result rows.

### Commands
- `uv run python scripts/run_experiment.py --config configs/experiments/tool_ab_smoke.yaml`

### Acceptance tests
- Telemetry fields present in every result row.
- Tool-level errors bucketed distinctly from model/task/verifier errors.

---

## Phase 4 Gate: Optimize-Anything

## WP-06A: Optimizable Target Interface

### Goal
Generalize optimization beyond skills.

### Files
- `bench/optimize/targets/base.py` (new)
- `bench/optimize/targets/skill.py` (new)
- `bench/optimize/targets/system_prompt.py` (new)
- `bench/optimize/targets/tool_policy.py` (new)

### Tasks
1. Define interface:
   - serialize candidate
   - deserialize candidate
   - apply candidate to experiment config
2. Implement first target: skill text.
3. Implement second target: system prompt.

### Commands
- `uv run python -c "from bench.optimize.targets.base import OptimizableTarget; print('ok')"`

### Acceptance tests
- Targets can round-trip serialize/deserialize.
- Applying target mutates only intended profile dimension.

---

## WP-06B: GEPA Objective Adapter + Runner

### Goal
Run optimize-anything against benchmark objective.

### Files
- `bench/optimize/gepa_adapter.py` (new/refactor from `optimization/adapter.py`)
- `scripts/run_optimize.py` (new)
- `configs/optimize/*.yaml` (new)

### Tasks
1. Implement evaluator wrapper over `run_experiment`.
2. Add objectives:
   - pass-rate maximize
   - cost-adjusted
   - weighted by difficulty
3. Save optimization artifacts: best candidate, trajectory, holdout summary.

### Commands
- `uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml`

### Acceptance tests
- Optimization run completes on smoke dataset.
- Best candidate and holdout artifacts emitted.

---

## WP-06C: Resume/Budget/Safety Controls

### Goal
Make optimization robust for long-running experiments.

### Files
- `bench/optimize/runtime.py` (new)
- `scripts/run_optimize.py`

### Tasks
1. Add resume from run directory.
2. Add budget caps (time, calls, tokens).
3. Add graceful interruption and checkpoint persistence.

### Commands
- `uv run python scripts/run_optimize.py --config configs/optimize/skill_smoke.yaml --resume`

### Acceptance tests
- Interrupted run resumes correctly.
- Budget limits hard-stop with clear status.

---

## Cross-Phase CI and Quality Gate

## WP-07A: CI Contract + Smoke Suite

### Goal
Prevent regressions in schemas/configs/CLI/runtime.

### Files
- CI config (project-specific, e.g. `.github/workflows/*.yml`)
- `scripts/ci_smoke.sh` (new)

### Tasks
1. Add CI steps:
   - `uv run pytest -q`
   - `uv run ruff check .`
   - `uv run ty check .`
   - `uv run python scripts/validate_contracts.py`
2. Add smoke experiment configs and execute at least one in CI.
3. Enforce schema validation on generated artifacts.

### Commands
- `bash scripts/ci_smoke.sh`

### Acceptance tests
- CI fails on schema/config drift.
- CI includes at least one end-to-end smoke execution.

---

## Dependency Graph (Execution Scheduling)

Use this graph as the source of truth for orchestration order and parallelism.

### Node legend
- `WP-*`: backend work packages in this file.
- `VW-*`: visualizer work packages in `ORCHESTRATOR_VISUALIZER_WORK_PACKAGES.md`.

### Adjacency list (hard dependencies)
- `WP-01A -> WP-01B, WP-01C`
- `WP-01B -> WP-01D`
- `WP-01C -> WP-01D, WP-02A`
- `WP-01D -> WP-02B`
- `WP-02A -> WP-02B`
- `WP-02B -> WP-02C`
- `WP-02C -> WP-03A`
- `WP-03A -> WP-03B, WP-04A, WP-05A, VW-02`
- `WP-03B -> WP-04A, WP-05A`
- `WP-04A -> WP-04B, WP-04C, VW-03`
- `WP-04B -> VW-03`
- `WP-04C -> WP-06A`
- `WP-05A -> WP-05B, WP-05C, VW-03`
- `WP-05B -> VW-03`
- `WP-05C -> WP-06A, VW-04`
- `WP-06A -> WP-06B, WP-06C, VW-04`
- `WP-06B -> WP-07A, VW-05`
- `WP-06C -> WP-07A`
- `WP-07A -> (phase completion)`
- `VW-01 -> VW-02`
- `VW-02 -> VW-03`
- `VW-03 -> VW-04`
- `VW-04 -> VW-05`

### Parallel lanes by gate
1. Gate G0 complete (`WP-01C` done):
- Run `WP-01D` and `WP-02A` in parallel.

2. Gate G1 complete (`WP-03A` done):
- Run `WP-03B`, `WP-04A`, `WP-05A`, `VW-02` in parallel.

3. Gate G2 complete (`WP-04A` + `WP-05A` done):
- Run `WP-04B`, `WP-04C`, `WP-05B`, `WP-05C`, `VW-03` in parallel where deps allow.

4. Gate G3 complete (`WP-06A` done):
- Run `WP-06B`, `WP-06C`, `VW-04` in parallel.

5. Gate G4 complete (`WP-06B` done):
- Run `WP-07A` and `VW-05` in parallel.

### Critical path
`WP-01A -> WP-01C -> WP-01D -> WP-02B -> WP-02C -> WP-03A -> WP-04A -> WP-04C -> WP-06A -> WP-06B -> WP-07A`

---

## Suggested Delivery Order and Parallelism

### Strict order
1. WP-01A → WP-01D
2. WP-02A → WP-02C
3. WP-03A → WP-03B
4. WP-04A → WP-04C
5. WP-05A → WP-05C
6. WP-06A → WP-06C
7. WP-07A

### Safe parallel groups
- Parallel Group 1 (after WP-01C): WP-02A + WP-01D
- Parallel Group 2 (after WP-03A): WP-04A + WP-05A
- Parallel Group 3 (after WP-06A): WP-06B + WP-07A setup

---

## Global Exit Criteria

All phases are complete when:
1. Canonical schemas and profile fingerprints are enforced across all outputs.
2. One unified experiment command runs baseline/support/tool experiments.
3. Paired deltas exist for support and tool A/B comparisons.
4. Optimization can target skill and at least one non-skill component.
5. CI enforces contracts and runs smoke end-to-end.
