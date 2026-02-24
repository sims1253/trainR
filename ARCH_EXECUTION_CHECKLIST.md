# Architecture Remediation - Execution Checklist

Reference: `ARCH_REMEDIATION_PLAN.md`

## A) Canonical Runner Consolidation

- [ ] A-01: Confirm `scripts/run_experiment.py` is the only canonical benchmark executor.
- [ ] A-02: Convert `scripts/run_benchmark.py` to wrapper-only (arg/config translation + call canonical runner).
- [ ] A-03: Convert `scripts/evaluate_batch.py` to wrapper-only.
- [ ] A-04: Convert `scripts/mini_benchmark.py` to wrapper-only or replace with canonical config invocation.
- [ ] A-05: Rewire `posit_gskill evaluate` to canonical runner API (remove legacy runtime path dependency).
- [ ] A-06: Ensure wrappers emit clear deprecation warning + replacement command.
- [ ] A-07: Remove duplicated Docker/provider execution logic from legacy scripts.
- [ ] A-08: Add regression test asserting wrappers delegate to canonical path.

## B) Harness Adapter Layer

- [ ] B-01: Create `bench/harness/` module.
- [ ] B-02: Define `HarnessRequest` and `HarnessResult` typed contracts.
- [ ] B-03: Define `AgentHarness` protocol/interface.
- [ ] B-04: Refactor evaluation path to use injected harness, not direct Pi runner calls.
- [ ] B-05: Implement `PiSdkHarness` adapter.
- [ ] B-06: Implement `PiCliHarness` adapter as fallback.
- [ ] B-07: Add `CliHarnessBase` for Codex/Claude/Gemini-style adapters.
- [ ] B-08: Add harness selector in experiment config (`execution.harness`).
- [ ] B-09: Add harness capability metadata (json mode, tool use, usage support).
- [ ] B-10: Add integration tests proving harness can be swapped via config only.

## C) Provider Resolution + Credential Policy

- [ ] C-01: Create central provider resolver module (model -> provider -> env var -> runtime identifier).
- [ ] C-02: Remove duplicate provider/env mapping tables from scattered modules.
- [ ] C-03: Fix key naming consistency (including ZAI aliases where required).
- [ ] C-04: Add startup preflight validation for model/provider/key requirements.
- [ ] C-05: Introduce `AuthPolicy` values (`env`, `mounted_auth_file`).
- [ ] C-06: Implement credential resolver chain honoring policy.
- [ ] C-07: Support explicit read-only auth file mounts (opt-in only).
- [ ] C-08: Add redaction for token values and credential file paths in logs.
- [ ] C-09: Record auth source and policy in manifest metadata.
- [ ] C-10: Add negative tests for missing key and mismatched model/provider configs.

## D) Sandboxing Hardening

- [ ] D-01: Centralize Docker command construction in one module.
- [ ] D-02: Add sandbox profiles (`strict`, `networked`, `developer`).
- [ ] D-03: Make `strict` default for benchmarks/CI.
- [ ] D-04: Enforce non-root execution in all benchmark container paths.
- [ ] D-05: Add filesystem hardening flags where compatible (`read-only`, temp writable dirs).
- [ ] D-06: Add resource limits (CPU/memory/pids) with configurable defaults.
- [ ] D-07: Add explicit network mode behavior per profile.
- [ ] D-08: Ensure no benchmark path assembles raw `docker run` flags outside central builder.
- [ ] D-09: Persist active sandbox profile/flags into manifest.
- [ ] D-10: Add tests asserting profile -> Docker flags mapping.

## E) Unified Telemetry

- [ ] E-01: Define canonical telemetry schema for token/cost/turn/tool metrics.
- [ ] E-02: Map Pi SDK/CLI native usage events into canonical schema.
- [ ] E-03: Preserve raw adapter events for audit/debug.
- [ ] E-04: Ensure unknown usage fields are represented as unknown/null, not forced zero.
- [ ] E-05: Add optional cost estimation layer with provider pricing table.
- [ ] E-06: Surface normalized telemetry in result artifacts + summary.
- [ ] E-07: Add telemetry contract tests for each harness adapter.

## F) CI + Integration Reliability

- [ ] F-01: Split CI into `fast`, `integration-local`, `integration-provider`, `visualizer`.
- [ ] F-02: Keep `fast` for lint/type/unit/schema/contracts only.
- [ ] F-03: Add `integration-local` test using Docker + mocked/stubbed model output.
- [ ] F-04: Add env-gated provider smoke test (1 task x 1 model x selected harness).
- [ ] F-05: Add optimization smoke integration test (1-2 iterations + checkpoint/resume/budget stop).
- [ ] F-06: Add synthetic task generation smoke test.
- [ ] F-07: Add PR-mined task pipeline smoke test (fixture-driven).
- [ ] F-08: Update `ci-quick` docs/output to clearly state covered vs skipped checks.
- [ ] F-09: Add failure triage hints in CI output by layer (resolver/harness/sandbox/pipeline).

## G) Config/Schema Migration

- [ ] G-01: Extend experiment config schema with `execution.harness`.
- [ ] G-02: Add `execution.sandbox_profile`.
- [ ] G-03: Add `execution.auth_policy` + optional auth mount list.
- [ ] G-04: Add schema validation for new fields.
- [ ] G-05: Provide backward-compatible translation for legacy configs.
- [ ] G-06: Add migration tests from legacy script args/configs to canonical schema.

## H) Additional Harness Integrations (Optional Wave)

- [ ] H-01: Implement Codex CLI adapter on top of `CliHarnessBase`.
- [ ] H-02: Implement Claude Code adapter on top of `CliHarnessBase`.
- [ ] H-03: Implement Gemini CLI adapter on top of `CliHarnessBase`.
- [ ] H-04: Implement SWE-agent adapter (if needed).
- [ ] H-05: Add adapter contract/integration tests for each enabled harness.
- [ ] H-06: Document harness capability matrix and known limitations.

## I) Documentation + Guardrails

- [ ] I-01: Update README to show canonical commands only.
- [ ] I-02: Add architecture doc for harness/provider/sandbox layering.
- [ ] I-03: Add security doc for credential policies and OAuth mount caveats.
- [ ] I-04: Add reproducibility doc defining what metadata must be captured.
- [ ] I-05: Add contributor guide for implementing new harness adapters.

## J) Final Verification Gate

- [ ] J-01: Run full CI suite with new architecture.
- [ ] J-02: Run one real provider smoke benchmark and verify end-to-end artifacts.
- [ ] J-03: Run one mini optimization and verify checkpoint/resume/budget behavior.
- [ ] J-04: Verify visualizer consumes updated artifacts without contract breaks.
- [ ] J-05: Confirm no legacy code path bypasses canonical runner.
