# Architecture Remediation Plan

Status: Proposed  
Owner: GLM-5 implementation track  
Scope: Orchestrator benchmark stack, harness adapters, sandboxing, telemetry, CI

## 1) Goals

Build a single, extensible benchmark architecture where:

- one canonical execution engine runs all experiments;
- agent harnesses (Pi SDK/CLI, Codex CLI, Claude Code, Gemini CLI, SWE-agent) are pluggable adapters;
- sandboxing and credential handling are policy-driven and auditable;
- token/cost/tool telemetry is normalized across harnesses/providers;
- CI validates real integration paths, not only dry-run/config paths.

## 2) Non-Goals

- Perfect provider parity in one pass.
- Migrating every historical result artifact.
- Full production-grade multi-tenant isolation in this phase.

## 3) Current Gaps (Why This Exists)

- Execution path fragmentation (canonical and legacy scripts both active).
- Legacy CLI evaluation path wired to outdated runtime contract.
- Provider/env mapping duplication and inconsistencies (notably ZAI env naming pathing).
- CI smoke is mostly static/dry-run and misses true end-to-end eval.
- Sandboxing is basic Docker isolation without centralized hardening policy.
- No first-class harness abstraction for multi-agent ecosystem support.

## 4) Target Architecture

### 4.1 Canonical execution

- `scripts/run_experiment.py` is the single canonical entrypoint.
- All old scripts become thin wrappers that translate args/config only.
- Core logic lives in `bench/experiments/*` + harness interfaces.

### 4.2 Harness adapter layer

Add `bench/harness/` with:

- `HarnessRequest`
- `HarnessResult`
- `AgentHarness` protocol (`run(request) -> HarnessResult`)

Initial adapters:

1. `PiSdkHarness` (preferred primary)
2. `PiCliHarness` (compat fallback)
3. `CliHarnessBase` (Codex/Claude/Gemini wrappers)
4. optional `SweAgentHarness`

### 4.3 Unified telemetry contract

Every harness must emit:

- prompt/completion/total tokens
- cache read/write tokens (if available)
- estimated cost (if pricing map available)
- turns used
- tool call counts/errors/durations
- latency breakdown
- raw adapter events (for audit)

### 4.4 Sandboxing and auth policy

Central policy objects:

- `SandboxPolicy` (strict/networked/developer profiles)
- `AuthPolicy` (env keys, mounted auth file, future bridge)

All runs record active policy in manifest.

## 5) Implementation Phases

## Phase A - Canonical Runner Consolidation

### A1. Make one execution path

Tasks:

- Route all benchmark execution through canonical runner API.
- Convert deprecated scripts (`run_benchmark.py`, `evaluate_batch.py`, `mini_benchmark.py`) into wrappers only.
- Rewire `posit_gskill evaluate` to canonical runner flow.

Acceptance criteria:

- No duplicated benchmark/evaluation business logic outside canonical core.
- Wrapper scripts have no direct Docker/provider logic.

## Phase B - Harness Abstraction and Adapterization

### B1. Introduce harness interface

Tasks:

- Add `bench/harness/base.py` with typed contracts.
- Replace direct Pi calls in evaluation path with injected harness instance.

Acceptance criteria:

- Harness is selected from config only.
- Core runner has no provider-specific branches.

### B2. Implement adapters

Tasks:

- Implement `PiSdkHarness` (preferred).
- Keep `PiCliHarness` as fallback.
- Add generic CLI adapter foundation for Codex/Claude/Gemini.

Acceptance criteria:

- Same task/model/skill can run by swapping harness ID in config.

## Phase C - Provider Resolution and Credential Integrity

### C1. Single source of truth for provider mapping

Tasks:

- Create central resolver module for:
  - model reference resolution,
  - provider prefixing,
  - required env var mapping.
- Remove duplicated mapping logic from runner/sandbox/scripts.
- Fix env var naming consistency (include explicit compatibility aliases where needed).

Acceptance criteria:

- Startup preflight catches missing key/model mismatch before first task.
- No duplicated provider map tables remain.

### C2. Credential policy

Tasks:

- Implement `CredentialResolver` chain:
  1) env api key (default),
  2) explicit read-only auth file mount (opt-in),
  3) reserved future bridge mode.
- Add log redaction for token/file path leakage.

Acceptance criteria:

- Auth source is explicit in run metadata.
- Personal OAuth/session mode requires explicit flag and is marked non-reproducible.

## Phase D - Sandboxing Hardening

### D1. Central Docker run builder

Tasks:

- Consolidate container invocation flags in one module.
- Add profile presets:
  - `strict` (default): non-root, read-only filesystem where possible, reduced caps, resource limits.
  - `networked`: explicit outbound network allowance.
  - `developer`: local debug relaxed mode.

Acceptance criteria:

- No scattered raw `docker run` construction in business logic paths.
- Active sandbox profile appears in manifest and summary.

## Phase E - CI and Integration Reliability

### E1. Test pyramid upgrades

Tasks:

- Keep unit + contract tests.
- Add adapter contract tests with fixture events.
- Add local Docker integration test with stubbed model output.
- Add env-gated provider smoke (1 task x 1 model x selected harness).
- Add optimization mini integration test (1-2 iterations, checkpoint/resume/budget stop).
- Add synthetic + mined pipeline smoke tests.

Acceptance criteria:

- CI includes at least one true non-dry end-to-end path.
- Failures localize by layer (resolver/harness/sandbox/pipeline).

### E2. CI job decomposition

Jobs:

- `fast`: lint/type/unit/schema/contracts
- `integration-local`: docker + harness stub
- `integration-provider` (scheduled/manual/secrets gated): real provider
- `visualizer`: bun lint/build/export contract

Acceptance criteria:

- `ci-quick` explicitly documents scope limits.

## Phase F - Ecosystem Expansion

### F1. Additional harnesses

Tasks:

- Add Codex/Claude/Gemini CLI adapters using common CLI base.
- Add SWE-agent adapter if required.
- Optionally add AI SDK-backed adapter where it simplifies provider glue.

Acceptance criteria:

- New harness integrations require adapter + tests + config only (no core runner edits).

## 6) Config and Schema Changes

Additions (illustrative):

- `execution.harness: pi_sdk | pi_cli | codex_cli | claude_cli | gemini_cli | swe_agent`
- `execution.sandbox_profile: strict | networked | developer`
- `execution.auth_policy: env | mounted_auth_file`
- `execution.auth_mounts[]` (path, read_only, purpose)
- telemetry fields for normalized token/cost/tool metrics

Migration:

- Preserve backward compatibility for one cycle via wrapper translation.
- Emit deprecation warnings with exact replacement commands.

## 7) Risk Register

- OAuth/session-file auth may break due to provider/keychain/device coupling.
- Tight sandbox defaults may initially reduce compatibility for some agents.
- Provider SDK/API changes may require adapter-specific maintenance.
- Cost telemetry may be partially unknown for some providers unless price metadata is maintained.

Mitigations:

- Keep adapter capability matrix.
- Separate strict and developer profiles.
- Record unknown metrics explicitly; do not fake zeros.

## 8) Milestones

- M1: Canonical consolidation + provider resolver cleanup.
- M2: Harness abstraction + Pi SDK adapter operational.
- M3: Sandbox policy + credential policy shipping.
- M4: Integration CI (local + provider-gated) passing.
- M5: Additional harness adapters and optimization/task pipeline smoke coverage.

## 9) Definition of Done (Architecture)

- One canonical execution engine in practice.
- Harnesses pluggable via config without core edits.
- Provider/auth/sandbox behavior policy-driven and auditable.
- Telemetry normalized across harnesses/providers.
- CI includes non-dry end-to-end validation.

## Appendix A - Sandbox Options (Pragmatic Guidance)

For this project, the right sequence is:

1. Harden local Docker policy first (fastest path, lowest migration cost).
2. Add optional hosted Docker sandbox execution later (if parallel scale or ops burden requires it).
3. Consider microVM-based isolation only if threat model requires stronger tenant isolation.

Notes:

- Many "agent sandbox" services are orchestration layers around containers.
- They can improve ops ergonomics, quotas, observability, and remote isolation.
- They do not remove the need for explicit auth policy, telemetry normalization, and adapter contracts.
