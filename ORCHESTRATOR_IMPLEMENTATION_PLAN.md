# trainR Bench+ Implementation Plan (Orchestrator-Ready)

## 1. Objective

Build `trainR` into a staged benchmark platform for R coding agents with four prioritized capabilities:

1. R-focused SWE-Bench core: collect tasks and benchmark model/agent performance on R package tasks.
2. SkillsBench-style support structures: benchmark with/without system prompt, `agents.md`, single skills, and skill collections.
3. Tooling experiments: benchmark the same tasks with different toolsets/tool implementations.
4. Optimization layer: use GEPA/optimize-anything to optimize any benchmark component (starting with R skills).

This plan is decision-complete and intended for direct execution by an agent orchestrator.

---

## 2. Scope, Constraints, and Principles

### In scope
- Backend Python code, configs, schemas, scripts, evaluation harnesses, dataset pipeline, CI checks.
- R task collection and evaluation using Docker-based reproducible environments.
- Structured experiment dimensions: model, support profile, tool profile.

### Out of scope
- `visualizer/` implementation details.
- UI/dashboard development.

### Constraints
- Preserve existing task assets where possible.
- Favor backward-compatible migration path for current scripts/configs.
- Every benchmark run must be reproducible via stored config fingerprints.

### Design principles
- Separate concerns strictly:
  - `model_profile`
  - `support_profile`
  - `tool_profile`
- Single canonical task schema with explicit versioning.
- Deterministic and auditable evaluation contracts.
- Prefer incremental refactors from current modules over big-bang rewrites.

---

## 2.1 Incorporated Decisions from Alternate Roadmap

The following ideas are explicitly adopted:

1. Dual task paradigms in one benchmark:
- SWE-style issue/patch tasks.
- R test-generation/skill tasks.

2. Explicit support-structure matrix:
- system prompt mode
- `AGENTS.md` mode
- skill mode (`none`, `single`, `collection_forced`, `collection_selective`)

3. Tool variant benchmarking:
- controlled A/B tool profiles (e.g., patch tool v1 vs v2).

4. General optimization abstraction:
- optimize-anything targets extend beyond skill text.

The following decisions override parts of the alternate draft:

1. Use one canonical schema/version path (`bench/schema/v1`), not parallel dataclass-only contracts.
2. Use one main experiment entrypoint (`scripts/run_experiment.py`) rather than separate benchmark executables per phase.
3. Implement on-demand skill access as `support_profile` policy and telemetry, not custom ad hoc prompt commands.
4. Keep a minimal fixed baseline toolset even in Phase 1; \"no tools\" means no experimental tool variants.

---

## 3. Target Architecture

## 3.1 Top-level modules

- `bench/schema/`
  - Canonical schemas and validators.
- `bench/dataset/`
  - Task ingestion, validation, splitting, decontamination.
- `bench/collector/`
  - PR mining and transformation into canonical tasks.
- `bench/eval/`
  - Runner orchestration, execution sandbox, verifier adapters.
- `bench/profiles/`
  - Model, support, and tool profile models + fingerprinting.
- `bench/experiments/`
  - Experiment matrix expansion, scheduling, and result collation.
- `bench/optimize/`
  - Optimize-anything integration for arbitrary targets.
- `bench/reports/`
  - Aggregations, metrics, and reproducibility manifests.

## 3.2 Canonical profile dimensions

- `model_profile`
  - model name
  - provider
  - endpoint/env mapping
  - generation parameters
- `support_profile`
  - system prompt content/ref
  - agents.md mode/content/ref
  - skill mode (`none`, `single`, `collection_forced`, `collection_selective`)
  - selected skill IDs/refs
- `tool_profile`
  - enabled tools list
  - tool implementation versions
  - tool limits/timeouts/policies

Each profile must produce a stable SHA256 fingerprint.

## 3.3 Primary run key

Every result row must include this unique logical key:

`task_id + model_profile_fingerprint + support_profile_fingerprint + tool_profile_fingerprint + run_seed`

---

## 4. Canonical Schemas (Versioned)

Create `bench/schema/v1/` and define JSON-schema + Pydantic models.

## 4.1 Task schema (`task.schema.json`)

Required fields:
- `schema_version` (e.g. `task-v1`)
- `task_id`
- `task_type` (`swe`, `test_gen`, `skill`)
- `instance_id`
- `repo`
- `base_commit`
- `patch`
- `test_patch`
- `fail_to_pass` (array)
- `pass_to_pass` (array)
- `instruction`
- `context`
- `source_package`
- `difficulty`
- `split`
- `metadata` (quality score, mined timestamp, miner version)

Notes:
- `task_type` controls validation rules, but all tasks stay in one canonical dataset format.
- SWE-style tasks use `problem_statement` and patch/test fields.
- test-gen/skill tasks use `instruction/context/reference_test` style fields (within canonical envelope).

## 4.2 Profile schemas

- `model_profile.schema.json`
- `support_profile.schema.json`
- `tool_profile.schema.json`

## 4.3 Run/result schema

- `run_manifest.schema.json`
- `result_row.schema.json`

Required result fields:
- profile fingerprints
- run timing/cost/token usage
- pass/fail status
- verifier outputs
- error category (infra/config/test/task/agent)

---

## 5. Phase Plan (Priority-Ordered)

## Phase 0: Foundation Hardening (must complete first)

### Goals
- Fix current contract breakages to make baseline execution reliable.

### Tasks
1. Fix CLI/Makefile wiring:
- Ensure `make benchmark` passes required config and valid flags.
- Add lightweight smoke target (`make benchmark-smoke`) for 1 task/1 model.

2. Fix packaging entrypoint integrity:
- Align `pyproject.toml` script entrypoint to real module.
- Remove non-existent package entries from wheel config.

3. Fix model config contract drift:
- Enforce benchmark model names must resolve from `configs/llm.yaml`.
- Add explicit startup validation and fail-fast messaging.

4. Fix provider-specific auth checks:
- Replace global ZAI key checks with per-selected-provider preflight.

5. Improve failure taxonomy:
- Introduce error categories for missing API key, missing package, infra timeout, verifier failure.

### Acceptance criteria
- `make benchmark-smoke` works with one configured model and one task.
- CLI entrypoints execute without `ModuleNotFoundError`.
- Invalid config/model mismatch fails with precise validation errors.
- Provider key checks only require relevant credentials.

---

## Phase 1: R SWE-Bench Core

### Goals
- Build stable R benchmark dataset/eval core.
- Benchmark raw model/agent baseline capability with fixed baseline support/tool profiles.

### Tasks
1. Implement canonical task schema and migration tooling:
- `scripts/migrate_tasks_to_v1.py`
- Validate all tasks under `tasks/`.

2. Build dataset manager:
- ingest, validate, split (`train/dev/test`), export manifests.
- add freshness/decontamination metadata fields.

3. Build core benchmark runner:
- deterministic execution plan from experiment config.
- store run manifests + per-row results using canonical schema.
- baseline profile defaults:
  - support profile: minimal/default
  - tool profile: fixed baseline tools only (no experimental variants)

4. Add reproducibility artifacts:
- docker image digest
- env lock fingerprints
- config/profile fingerprints

### Acceptance criteria
- All existing tasks migrated or flagged with actionable schema errors.
- Single command can run `train/dev/test` benchmark and emit valid v1 results.
- Re-running same manifest yields same run key structure and comparable outputs.

---

## Phase 2: Support Structures (SkillsBench-style)

### Goals
- Benchmark support structure impact independent of model/tooling.

### Tasks
1. Implement support profile model and resolver:
- System prompt only.
- Agents file only.
- System + agents.
- Single skill.
- Skill collection forced read.
- Skill collection selective read.

2. Add paired evaluation mode:
- auto-generate paired configs (e.g., `no_skill` vs `with_skill`) with identical task/model/tool seeds.

3. Add skill collection policies:
- `forced_all`: always inject all skills.
- `selective`: agent/runtime chooses subset according to policy; log selected set.
- no custom in-band command protocol required; selection is mediated by runtime/profile policy.

4. Persist support telemetry:
- prompt composition length
- selected skills
- support artifacts hashes

### Acceptance criteria
- Same task/model/tool can be run across all support modes with identical runner.
- Paired output table includes delta metrics by support profile.

---

## Phase 3: Tooling Experiments

### Goals
- Measure impact of tool availability and tool implementation changes.

### Tasks
1. Implement tool profile schema and execution policy.
2. Add tool adapters registry:
- patch tool variants
- file search/edit variants
- sandbox policy variants
3. Add controlled A/B matrix generator:
- hold task/model/support constant while varying tool profile.
- emit paired comparison rows for each tool experiment.

4. Add tool usage telemetry:
- call counts
- failures/timeouts
- per-tool latency

5. Add benchmark matrix support:
- Cartesian expansion over model x support x tool profiles (with budget guardrails).

### Acceptance criteria
- Runner can execute identical task/model/support across at least 2 tool profiles.
- Result rows include tool profile fingerprint and tool telemetry.

---

## Phase 4: Optimize-Anything Layer

### Goals
- Optimize benchmark components with GEPA/optimize-anything.

### Tasks
1. Define optimization target interface:
- target type (`skill_text`, `system_prompt`, `skill_routing_policy`, `tool_policy`)
- mutation operators
- serialization

2. Implement objective adapters:
- pass-rate objective
- cost-adjusted objective
- weighted objective by difficulty/package

3. Implement optimization run orchestrator:
- seed candidate
- train set, val set, holdout evaluation
- parallel budget controls

4. Add safety controls:
- max metric calls
- budget/time caps
- run resume/restart

### Acceptance criteria
- End-to-end optimization run for R skill produces best candidate + evaluation report.
- Holdout evaluation emitted in canonical result schema.

---

## 6. Concrete Deliverables by File/Module

### New paths
- `bench/schema/v1/*.py|*.json`
- `bench/profiles/*.py`
- `bench/dataset/*.py`
- `bench/experiments/*.py`
- `bench/reports/*.py`
- `bench/optimize/*.py`
- `configs/profiles/{models,support,tools}/*.yaml`
- `configs/experiments/*.yaml`
- `scripts/run_experiment.py`

### Refactor targets
- `scripts/run_benchmark.py` -> use canonical profiles and run manifest.
- `scripts/run_parallel_benchmark.py` -> typed model/provider contract + profile fingerprints.
- `scripts/evaluate_batch.py` -> align CLI/config contracts and schema output.
- `scripts/mine_prs.py` + `task_generator/mined_task.py` -> emit canonical task schema.
- `evaluation/*` -> standardized error taxonomy and verifier result structure.

### Incremental migration rule
- Keep existing entry scripts operational during migration.
- Add compatibility wrappers that delegate to `scripts/run_experiment.py`.
- Remove legacy paths only after schema + CI + smoke parity is achieved.

### Remove/replace technical debt
- Remove ambiguous model naming conventions (`*-free` alias drift) unless mapped centrally.
- Eliminate hardcoded provider key assumptions in unrelated paths.

---

## 7. Experiment Configuration Contract

Create a single experiment config schema:

- dataset selector
- model profile set
- support profile set
- tool profile set
- sampling/repeats/seeds
- timeout/budget
- output location
- pairing directives (`paired_support`, `paired_tools`)
- baseline profiles (`baseline_support_profile`, `baseline_tool_profile`)

Add command:

`uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml`

Required behavior:
- validate config against schema
- materialize run manifest
- execute matrix
- emit `results.jsonl` + `summary.json` + `manifest.json`
- emit `paired_deltas.jsonl` when pairing directives are enabled

---

## 8. Metrics and Reporting

Required summary outputs:
- pass rate overall and by split
- pass rate by package/difficulty
- paired deltas (support/tool comparisons)
- token and cost metrics
- timeout and infra error rates

Required artifacts:
- `summary.json`
- `results.jsonl`
- `manifest.json`
- `failures.jsonl` (normalized failure records)

---

## 9. Testing and CI Gates

## 9.1 Unit tests
- schema validation tests
- profile fingerprint stability tests
- runner preflight tests
- provider-specific auth tests
- failure classification tests

## 9.2 Integration smoke tests
- 1 task x 1 model x 1 support x 1 tool
- 1 paired support run
- 1 dual-tool profile run

## 9.3 CI required checks
- `pytest`
- `ruff check`
- `ty check`
- config contract validator
- CLI smoke commands

---

## 10. Migration Plan

1. Add v1 schema and validators without breaking existing scripts.
2. Add migration script for historical tasks and results.
3. Run dual-write mode (legacy + v1 outputs) for one milestone.
4. Flip default to v1; keep legacy reader compatibility for one release.
5. Remove legacy-only paths after stability period.

---

## 11. Risk Register and Mitigations

1. Risk: dataset contamination / stale tasks.
- Mitigation: freshness metadata + decontamination checks + strict heldout policy.

2. Risk: nondeterministic verifier/runtime drift.
- Mitigation: image digests, lockfile fingerprints, deterministic verifier policies.

3. Risk: matrix explosion and cost blow-up.
- Mitigation: budget caps, stratified sampling, smoke configs, staged promotions.

4. Risk: profile contract complexity.
- Mitigation: strict schema validation + fingerprints + manifest-first execution.

---

## 12. Execution Backlog (Orchestrator Work Packages)

## WP-01 (P0): Contract and Entry Fixes
- Fix Makefile/CLI wiring.
- Fix pyproject entrypoint and packaging list.
- Add model/config contract check command.
- Add provider-aware auth preflight.

## WP-02 (P1): Schema and Dataset Core
- Implement `bench/schema/v1`.
- Implement task migration + validation report.
- Implement dataset manager split/export.

## WP-03 (P1): Core Runner + Manifest
- Implement run manifest generation.
- Implement canonical results writer.
- Add reproducibility metadata capture.
- Implement fixed baseline profile defaults used by Phase-1 sweeps.

## WP-04 (P2): Support Profiles
- Implement support profile schema/resolver.
- Implement paired support evaluation.
- Persist support telemetry.
- Implement policy-driven selective skill loading (no custom prompt command protocol).

## WP-05 (P3): Tool Profiles
- Implement tool profile schema + registry.
- Add tool telemetry and matrix support.
- Implement controlled A/B paired tool comparisons.

## WP-06 (P4): Optimization Integration
- Implement optimize target interface.
- Implement GEPA objective adapters.
- Implement optimize run report artifacts.

## WP-07 (Cross-cutting): CI and Quality Gates
- Add smoke configs and CI jobs.
- Add contract/schema tests and regression tests.

---

## 13. Definition of Done (Global)

The implementation is done when:

1. A smoke experiment can run end-to-end via one command with canonical manifests/results.
2. Core benchmark (Phase 1) is reproducible and schema-valid.
3. Support profiles and tool profiles are first-class experiment dimensions.
4. Optimization can target at least one component (R skill text) and evaluate on holdout.
5. CI enforces schema, config, typing/linting, and smoke execution contracts.

---

## 14. Immediate Next Commands (for the implementing agent)

1. `uv run python scripts/validate_contracts.py` (to be created in WP-01)
2. `uv run pytest -q`
3. `uv run ruff check .`
4. `uv run ty check .`
5. `uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml` (after WP-03)
