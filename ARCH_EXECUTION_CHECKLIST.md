# Architecture Remediation - Execution Checklist (Greenfield)

Reference: `ARCH_REMEDIATION_PLAN.md`

## How to read this checklist

- Items within each section are ordered by dependency. Complete them top-to-bottom.
- Cross-section dependencies are noted with `Depends:` annotations.
- Items marked `[DEFERRED]` are not scheduled for the current phase.

---

## A) Canonical Runner Enforcement

Phase gate: None (start here).

- [x] A-00: Add `bench/runner.py` with `run()` as the canonical library API.
- [x] A-01: Confirm `scripts/run_experiment.py` delegates to `bench.runner.run()` as the canonical API.
- [x] A-02: Remove or hard-fail legacy executors (`run_benchmark.py`, `evaluate_batch.py`, `mini_benchmark.py`) with canonical pointer.
- [x] A-03: Rewire `posit_gskill evaluate` to canonical runner API.
- [x] A-04: Remove duplicated benchmark/evaluation business logic outside canonical core.
- [x] A-05: Add regression tests asserting all supported entrypoints delegate to canonical path.
- [x] A-06: Add guard test confirming no bypass path executes direct Docker/provider logic.
- [x] A-07: Remove dead code, orphaned test fixtures, and obsolete config files left behind by legacy paths.
- [x] A-08: Run dead-code scan and confirm no legacy artifacts remain.

## B) Harness Adapter Layer

Phase gate: **A complete** (all A items checked).

Items are ordered by dependency -- B-01 through B-04 define the contracts that B-05 through B-11 consume.

- [x] B-01: Create `bench/harness/` module.
- [x] B-02: Define `HarnessRequest` and `HarnessResult` typed contracts.
- [x] B-03: Define `AgentHarness` protocol/interface.
- [x] B-04: Add `HarnessRegistry`/`HarnessFactory` for config-driven selection.
- [x] B-05: Refactor runner to use injected harness (no direct Pi runner calls). *Depends: B-01..B-04.*
- [x] B-06: Implement `PiSdkHarness` adapter (primary). *Depends: B-01..B-04.* (combined with B-07 as PiDockerHarness)
- [x] B-07: Implement `PiCliHarness` adapter (fallback). *Depends: B-01..B-04.* (combined with B-06 as PiDockerHarness)
- [x] B-08: Add `CliHarnessBase` for Codex/Claude/Gemini adapters. *Depends: B-01..B-04.*
- [x] B-09: Wire harness selector in runtime (`execution.harness`). *Depends: B-04, H-01.*
- [x] B-10: Add harness capability metadata (json mode, tools, usage support). *Depends: B-03.* (partially done with config changes)
- [x] B-11: Add integration tests proving harness swap via config only. *Depends: B-05, B-06, B-07.*

## C) Polyglot Boundary (Python Control Plane, TS Edge)

Phase gate: **B-06 complete** (`PiSdkHarness` is implemented).

- [x] C-01: Implement minimal unversioned JSON IPC contract for Python <-> TS Pi SDK worker.
- [x] C-02: Implement TS worker for Pi SDK adapter path.
- [x] C-03: Add cross-language contract smoke test for request/response shape.
- [x] C-04: Ensure raw TS adapter events are returned for audit/debug.
- [ ] C-05: `[DEFERRED]` Add versioned JSON IPC schema and compatibility matrix tests.
- [ ] C-06: `[DEFERRED]` Enforce strict bidirectional schema validation across adapter versions.

## D) Provider Resolution + Credential Policy

Phase gate: **B-04 and B-06 complete** (harness contracts stabilized).

- [x] D-01: Create central provider resolver module (model -> provider -> env aliases -> runtime identifier).
- [x] D-02: Remove duplicate provider/env mapping tables from scattered modules. *Depends: D-01.*
- [x] D-03: Normalize key naming and aliases (including ZAI alias handling). *Depends: D-01.*
- [x] D-04: Add startup preflight validation for model/provider/key requirements. *Depends: D-01..D-03.*
- [x] D-05: Introduce `AuthPolicy` values (`env`, `mounted_auth_file`).
- [x] D-06: Implement credential resolver chain honoring auth policy. *Depends: D-05.*
- [x] D-07: Support explicit read-only auth file mounts (opt-in only). *Depends: D-06.*
- [x] D-08: Add redaction for token values and credential file paths in logs. *Depends: D-06.*
- [x] D-09: Record auth source and policy in manifest metadata. *Depends: D-06.*
- [x] D-10: Add negative tests for missing key and mismatched model/provider configs. *Depends: D-04.*

## E) Sandboxing Hardening

Phase gate: **A complete** (independent of B; can run in parallel with B).

- [x] E-01: Centralize Docker command construction in one module.
- [x] E-02: Add sandbox profiles (`strict`, `networked`, `developer`). *Depends: E-01.*
- [x] E-03: Make `strict` default for benchmarks/CI. *Depends: E-02.*
- [x] E-04: Enforce non-root execution for benchmark container paths. *Depends: E-01.*
- [x] E-05: Add filesystem hardening flags where compatible (`read-only`, temp writable dirs). *Depends: E-01.*
- [x] E-06: Add resource limits (CPU/memory/pids) with configurable defaults. *Depends: E-01.*
- [x] E-07: Add explicit network behavior per profile. *Depends: E-02.*
- [x] E-08: Ensure no business logic path assembles raw `docker run` flags. *Depends: E-01.*
- [x] E-09: Persist active sandbox profile/flags into manifest. *Depends: E-02.*
- [x] E-10: Add tests asserting profile -> Docker flags mapping. *Depends: E-02.*

## F) Unified Telemetry

Phase gate: **B-02 (HarnessResult contract) defined.**

- [ ] F-01: Define canonical telemetry schema for token/cost/turn/tool metrics.
- [ ] F-02: Map Pi SDK/CLI native usage events into canonical schema. *Depends: F-01, B-06.*
- [ ] F-03: Preserve raw adapter events for audit/debug. *Depends: F-01.*
- [ ] F-04: Ensure unknown usage fields are `null`/unknown, not forced zero. *Depends: F-01.*
- [ ] F-05: Add optional cost estimation layer with provider pricing table. *Depends: F-01.*
- [ ] F-06: Surface normalized telemetry in result artifacts and summaries. *Depends: F-01..F-04.*
- [ ] F-07: Add telemetry contract tests for each harness adapter. *Depends: F-01.*
- [ ] F-08: Version the telemetry schema; add schema compatibility test in visualizer CI job. *Depends: F-01.*

## G) CI + Integration Reliability

Phase gate: **B, C, D, E complete and F-01 defined.**

- [x] G-00: Set up test fixtures, mock provider stubs, and CI secrets management. *(prerequisite for G-03..G-07)*
- [x] G-01: Split CI into `fast`, `integration-local`, `integration-provider`, `visualizer` (TanStack Start + Bun + Vite).
- [x] G-02: Keep `fast` for lint/type/unit/schema/contracts only.
- [x] G-03: Add `integration-local` test using Docker + stubbed model output. *Depends: G-00, G-01.*
- [ ] G-04: Add env-gated provider smoke test (1 task x 1 model x selected harness). *Depends: G-00, G-01.*
- [ ] G-05: Add optimization smoke integration test (1-2 iterations + checkpoint/resume/budget stop). *Depends: G-00, G-01.*
- [ ] G-06: Add synthetic task generation smoke test. *Depends: G-00, G-01.*
- [ ] G-07: Add PR-mined task pipeline smoke test (fixture-driven). *Depends: G-00, G-01.*
- [ ] G-08: Update `ci-quick` docs/output with explicit covered vs skipped checks.
- [ ] G-09: Add failure triage hints in CI output by layer (resolver/harness/sandbox/pipeline).

## H) Config/Schema Cutover

Phase gate: **A complete** (schema fields must land before runtime wiring).

- [x] H-01: Extend experiment config schema with `execution.harness`.
- [x] H-02: Add `execution.sandbox_profile`.
- [x] H-03: Add `execution.auth_policy` + optional auth mounts. *Depends: D-05.*
- [ ] H-04: Add schema validation for new fields. *Depends: H-01..H-03.*
- [ ] H-05: Remove backward-compat translation paths from runtime code. *Depends: H-04.*
- [ ] H-06: Add one-time config migration script (`scripts/migrate_config.py`) with sensible defaults. *Depends: H-01..H-03.*

## I) Additional Harness Integrations

Phase gate: **G complete; J-02 and J-05 published.**

- [ ] I-01: Implement Codex CLI adapter on top of `CliHarnessBase`.
- [ ] I-02: Implement Claude Code adapter on top of `CliHarnessBase`.
- [ ] I-03: Implement Gemini CLI adapter on top of `CliHarnessBase`.
- [ ] I-04: Implement SWE-agent adapter (if needed).
- [ ] I-05: Add adapter contract/integration tests for each enabled harness. *Depends: I-01..I-04.*
- [ ] I-06: Document harness capability matrix and known limitations. *Depends: I-01..I-04.*

## J) Documentation + Guardrails

J-01 through J-02 and J-05 are **prerequisites for Phase I** and should be written during or immediately after Phase B. The remaining items can follow later.

- [x] J-01: Update README to show canonical commands only, including frontend stack (`TanStack Start` + `Bun` + `Vite`). *(write during Phase A)*
- [x] J-02: Add architecture doc for harness/provider/sandbox layering. *(write during Phase B)*
- [ ] J-03: Add security doc for credential policies and auth mount caveats. *(write during Phase D)*
- [ ] J-04: Add reproducibility doc defining required metadata capture. *(write during Phase E)*
- [x] J-05: Add contributor guide for adding new harness adapters. *(write during Phase B; required before Phase I)*

## K) Final Verification Gate

Phase gate: **All prior sections complete** (excluding items explicitly marked `[DEFERRED]`).

- [ ] K-01: Run full CI suite with new architecture.
- [ ] K-02: Run one real provider smoke benchmark and verify end-to-end artifacts.
- [ ] K-03: Run one mini optimization and verify checkpoint/resume/budget behavior.
- [ ] K-04: Verify visualizer consumes updated artifacts without contract breaks.
- [ ] K-05: Confirm no legacy execution path bypasses canonical runner.
