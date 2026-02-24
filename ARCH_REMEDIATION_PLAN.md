# Architecture Remediation Plan (Greenfield-First)

Status: Proposed  
Owner: GLM-5 implementation track  
Scope: Orchestrator benchmark stack, harness adapters, sandboxing, telemetry, CI

## 1) Guiding Constraints

- Treat architecture decisions as greenfield (optimize for final system quality).
- No backward compatibility requirement with legacy scripts/configs/artifacts.
- Python is the control plane for orchestration, schema, policy, and evaluation lifecycle.
- TypeScript is allowed at harness edges when an SDK is TS-native (e.g., Pi SDK).

## 2) Goals

Build a single, extensible benchmark architecture where:

- one canonical execution engine runs all experiments;
- harnesses (Pi SDK/CLI, Codex CLI, Claude Code, Gemini CLI, SWE-agent) are pluggable adapters;
- sandbox and credential handling are policy-driven and auditable;
- token/cost/tool telemetry is normalized across harnesses/providers;
- CI validates at least one true end-to-end execution path.

## 3) Non-Goals

- Supporting every historical script/CLI path.
- Preserving old config formats beyond a one-time cutover converter.
- Perfect provider parity in the first release.
- Full multi-tenant production isolation in this phase.

## 4) Target Architecture

### 4.1 Canonical execution

The goal is a single supported execution path so that every harness, sandbox profile,
and telemetry hook is guaranteed to fire. A library API (`bench.runner.run()`) is the
canonical entry point; `scripts/run_experiment.py` is a thin CLI wrapper around it.
This means future entrypoints (web UI, programmatic tests, notebook usage) get the same
guarantees without duplicating orchestration logic.

- `bench.runner.run()` is the canonical execution API.
- `scripts/run_experiment.py` is a CLI wrapper that delegates to it.
- Core logic lives in `bench/experiments/*`.
- Any legacy scripts are removed or fail-fast with a canonical command pointer.

### 4.2 Harness adapter layer

Add `bench/harness/` with:

- `HarnessRequest`
- `HarnessResult`
- `AgentHarness` protocol (`run(request) -> HarnessResult`)
- `HarnessRegistry` and `HarnessFactory`

Initial adapters:

1. `PiSdkHarness` (preferred)
2. `PiCliHarness` (fallback)
3. `CliHarnessBase` (Codex/Claude/Gemini wrappers)
4. optional `SweAgentHarness`

### 4.3 Polyglot boundary

- Keep runner/policy/telemetry in Python.
- For TS-native SDKs, use a narrow JSON IPC boundary.
- For `PiSdkHarness`, ship a minimal unversioned JSON boundary first.
- Promote to a versioned IPC contract once at least two TS-native adapters exist.
- Contract:
  - Python sends normalized `HarnessRequest`.
  - TS worker returns normalized `HarnessResult` + raw events.
- No provider logic in runner; provider specifics stay in resolver + harness adapter.

### 4.4 Provider and auth policy

Central policy modules:

- `ProviderResolver` (model reference -> provider -> runtime model id -> required env aliases)
- `AuthPolicy` (`env`, `mounted_auth_file`)
- `CredentialResolver` chain with explicit source metadata

All runs record provider resolution and auth source in manifest metadata.

### 4.5 Sandboxing policy

Central `SandboxPolicy` with profiles:

- `strict` (default): non-root, readonly FS where possible, dropped caps, resource limits, no network by default
- `networked`: explicit outbound network allowance
- `developer`: relaxed local-debug profile

All Docker command construction is centralized in one builder module.

### 4.6 Unified telemetry contract

Every harness emits normalized telemetry:

- prompt/completion/total tokens
- cache read/write tokens (nullable)
- estimated cost (nullable)
- turns used
- tool call counts/errors/durations
- latency breakdown
- raw adapter events for audit

Unknown fields are represented as `null`/unknown, never synthetic zero.

### 4.7 Frontend stack

- Standardize frontend/visualizer work on `TanStack Start` (latest), with `Vite` and `Bun`.
- Keep frontend contract tests focused on artifact schema compatibility and route/build health.
- Frontend framework choice must not alter canonical backend artifact schemas.

## 5) Implementation Phases

### Phase dependency graph

```
Phase A ─────┬──> Phase B ──> Phase C ──> Phase E
             │                   ^
             └──> Phase D ───────┘
                                          Phase E ──> Phase F
                                            ^
Documentation (J-01, J-02, J-05) ───────────┘
```

- **A** is the foundation; nothing starts before it.
- **B** (harness abstraction) and **D** (sandbox hardening) are independent of each other and can proceed in parallel after A.
- **C** (provider/credential) depends on B (harness contracts define what the resolver must supply) but not on D.
- **E** (CI) depends on B, C, and D all being complete (integration tests exercise the full stack).
- **F** (ecosystem expansion) depends on E (new adapters need CI coverage) and on key documentation (architecture doc, contributor guide).
- **Versioned IPC hardening** is deferred — see Section 5.1.

### Phase gates

Each phase has an explicit entry condition. Do not start a phase until its gate is met.

| Phase | Entry gate |
|-------|-----------|
| A | None (start here) |
| B | A acceptance criteria met; all legacy paths removed or hard-failed |
| C | B-04 (registry/factory) and B-06 (PiSdkHarness) operational |
| D | A acceptance criteria met (independent of B) |
| E | B, C, D acceptance criteria all met |
| F | E acceptance criteria met; J-02 and J-05 published |

### 5.1 Deferred hardening: Versioned IPC boundary

`PiSdkHarness` still ships with a minimal TS worker and unversioned JSON contract in Phase B.
Only the hardening layer is deferred:

- versioned JSON IPC schema,
- compatibility matrix tests across schema versions,
- strict bidirectional schema enforcement for multiple TS adapters.

Promote this deferred work when a second TS-native adapter is introduced.

## Phase A - Canonical Runner Enforcement

Tasks:

- Introduce `bench/runner.py` with `run()` as the canonical library API.
- Remove or disable non-canonical benchmark execution paths.
- Ensure all entrypoints call canonical runner API (`bench.runner.run()`).
- Add tests proving no bypass path exists.
- Remove dead code, old configs, and obsolete test fixtures left behind by legacy paths.

Acceptance criteria:

- Only one supported runtime path for benchmark execution.
- No dead legacy code remains in the tree.

## Phase B - Harness Abstraction + Pi SDK First

Depends on: **Phase A complete.**

Tasks:

- Implement `bench/harness/base.py` contracts.
- Refactor runner to inject harness from config (`execution.harness`).
- Implement `PiSdkHarness` as primary and `PiCliHarness` as fallback.
- Implement minimal TS worker + unversioned JSON boundary for `PiSdkHarness`.
- Add a cross-language contract smoke test for Python <-> TS request/response shape.
- Add `CliHarnessBase` and one additional harness proof adapter.

Acceptance criteria:

- Same task/model/skill runs by swapping harness id only.

## Phase C - Provider/Credential Integrity

Depends on: **B-04 and B-06 operational.**

Tasks:

- Implement central provider resolver and preflight validation.
- Remove duplicated provider/env maps from runner/sandbox/scripts.
- Implement credential resolver chain with redaction and source tracking.

Acceptance criteria:

- Missing key/model mismatch is caught before first task.
- Auth source and policy are explicit in metadata.

## Phase D - Sandbox Hardening

Depends on: **Phase A complete** (independent of B).

Tasks:

- Centralize Docker command assembly.
- Implement strict/networked/developer profiles.
- Make `strict` default in benchmarks and CI.

Acceptance criteria:

- No raw `docker run` assembly in business logic paths.
- Active sandbox profile/flags appear in manifest.

## Phase E - CI Reliability

Depends on: **Phases B, C, D all complete.**

Prerequisites (new):

- Test fixtures, mock provider stubs, and CI secrets configuration must be set up before writing integration tests.

Jobs:

- `fast`: lint/type/unit/schema/contracts
- `integration-local`: docker + harness stub
- `integration-provider`: scheduled/manual + secrets-gated real provider
- `visualizer`: TanStack Start (`Bun` + `Vite`) lint/build/schema-compat contract

Acceptance criteria:

- At least one non-dry end-to-end path in CI.
- Failures localize by layer (resolver/harness/sandbox/pipeline).

## Phase F - Ecosystem Expansion

Depends on: **Phase E complete; J-02 (architecture doc) and J-05 (contributor guide) published.**

Tasks:

- Add Codex/Claude/Gemini adapters on `CliHarnessBase`.
- Add SWE-agent adapter if needed.
- Publish capability matrix per harness.

Acceptance criteria:

- New harness requires adapter + tests + config only; no runner edits.

## 6) Config and Schema (Target)

Required execution fields:

- `execution.harness: pi_sdk | pi_cli | codex_cli | claude_cli | gemini_cli | swe_agent`
- `execution.sandbox_profile: strict | networked | developer`
- `execution.auth_policy: env | mounted_auth_file`
- `execution.auth_mounts[]` (path, read_only, purpose)

Result/manifest additions:

- normalized telemetry fields
- provider resolution metadata
- auth source/policy metadata
- active sandbox profile/flags

### Config migration path

Existing experiment configs that lack the new `execution.*` fields will fail schema
validation at startup with a clear error message listing missing fields and example
values. A one-time offline migration script (`scripts/migrate_config.py`) converts
old configs by adding sensible defaults (`harness: pi_sdk`, `sandbox_profile: strict`,
`auth_policy: env`). The script is idempotent and safe to re-run.

## 7) Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | TS boundary drift between Python contracts and worker payloads | Medium | High | Contract tests for harness JSON payloads; defer full IPC schema until second TS adapter (see 5.1) |
| R2 | Strict sandbox defaults break harness assumptions | High | Medium | Capability matrix per harness; explicit `strict` vs `developer` profile behavior; integration test per profile |
| R3 | Provider SDK breaking changes require adapter maintenance | Medium | Medium | Pin SDK versions; adapter-specific CI smoke tests catch regressions early |
| R4 | Cost telemetry partial without maintained pricing tables | High | Low | Keep unknown telemetry explicit (`null`) rather than inferred; cost estimation is opt-in |
| R5 | IPC serialization slow under large payloads (many tool calls, long outputs) | Low | Medium | Benchmark IPC round-trip in integration tests; set payload size limits with clear error |
| R6 | Provider rate-limiting differences cause inconsistent benchmark results | Medium | Medium | Add per-provider retry/backoff config in harness adapter; record rate-limit events in telemetry |
| R7 | Sandbox profile misconfiguration fails silently (wrong profile applied) | Medium | High | Log active profile + Docker flags at run start; add assertion in manifest that profile matches config |
| R8 | Telemetry schema evolution breaks downstream consumers (visualizer, reports) | Low | High | Version the telemetry schema; add schema compatibility test in visualizer CI job |
| R9 | Legacy code removal leaves orphaned test fixtures or config files | High | Low | Add cleanup checklist item (A-07); run dead-code scan after Phase A |

## 8) Milestones

- M1: Canonical path enforcement + resolver foundation
- M2: Harness abstraction + Pi SDK adapter operational
- M3: Auth/sandbox policy shipping with metadata capture
- M4: Integration CI (local + provider-gated) passing
- M5: Additional harnesses + full telemetry normalization

## 9) Definition of Done

- One supported execution engine.
- Harnesses are config-pluggable without runner code changes.
- Provider/auth/sandbox behavior is policy-driven and auditable.
- Telemetry is normalized across harnesses/providers.
- CI proves non-dry end-to-end execution.
