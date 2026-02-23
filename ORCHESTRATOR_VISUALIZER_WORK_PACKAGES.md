# trainR Visualizer Work Packages (Parallel Track)

This document defines visualizer execution tickets (`VW-*`) intended to run in parallel with backend work packages where dependencies permit.

## Usage Contract
- Visualizer must consume canonical backend artifacts through a stable adapter contract.
- Do not bind UI directly to raw backend internals beyond the adapter output.
- Every VW ticket must include smoke verification in `visualizer/`.

---

## VW-01: UI Data Contract Freeze

### Goal
Freeze a stable frontend-facing schema to avoid drift while backend evolves.

### Depends on
- none (can start immediately)

### Files
- `visualizer/src/lib/schema.ts` (new)
- `visualizer/src/lib/types.ts` (refactor to align)
- `visualizer/README.md` (contract section)

### Tasks
1. Define `VisualizerDataV1` interface and validation helper.
2. Version the schema (`visualizer_data_version`).
3. Add strict runtime guard for malformed data.

### Commands
- `cd visualizer && bun run lint`
- `cd visualizer && bun run build`

### Acceptance tests
- Build passes with schema guard enabled.
- Invalid JSON shape produces explicit UI error state.

---

## VW-02: Adapter-Backed Data Ingestion

### Goal
Consume canonical backend outputs (`manifest/results/summary`) via one adapter export.

### Depends on
- `VW-01`
- `WP-03A` (unified experiment runner outputs)

### Files
- `scripts/export_visualizer_data.py` (new, backend)
- `visualizer/src/data/benchmark-results.json` (generated)
- `visualizer/src/app/page.tsx` (consume adapted shape only)

### Tasks
1. Create export adapter from canonical artifacts to `VisualizerDataV1`.
2. Replace implicit aggregation assumptions with explicit mapping.
3. Support backward-compatible fallback until v1 outputs are stable.

### Commands
- `uv run python scripts/export_visualizer_data.py --input results --output visualizer/src/data/benchmark-results.json`
- `cd visualizer && bun run build`

### Acceptance tests
- Visualizer renders from adapter-generated file.
- Adapter fails fast with actionable error on missing required backend fields.

---

## VW-03: Support/Tool Dimension Views

### Goal
Expose Bench+ dimensions in the UI for analysis.

### Depends on
- `VW-02`
- `WP-04A` (support profiles)
- `WP-05A` (tool profiles)
- better insights unlocked by `WP-04B` and `WP-05B`

### Files
- `visualizer/src/app/page.tsx`
- `visualizer/src/components/charts/*`
- `visualizer/src/components/ui/*` (as needed)
- `visualizer/src/lib/types.ts`

### Tasks
1. Add filters for `support_profile` and `tool_profile`.
2. Add paired delta tables/charts for support and tool A/B.
3. Add matrix slice controls (task split, package, difficulty, profile).

### Commands
- `cd visualizer && bun run lint`
- `cd visualizer && bun run build`

### Acceptance tests
- User can isolate one dimension while holding others fixed.
- Paired deltas render correctly when `paired_deltas.jsonl` exists.

---

## VW-04: Reproducibility and Provenance Panel

### Goal
Make every chart traceable to exact run artifacts.

### Depends on
- `VW-03`
- `WP-05C` (telemetry fields)
- `WP-06A` (optimization target metadata standardization)

### Files
- `visualizer/src/components/provenance-panel.tsx` (new)
- `visualizer/src/components/footer.tsx`
- `visualizer/src/lib/types.ts`

### Tasks
1. Display run manifest fingerprint, schema version, dataset fingerprint.
2. Show environment metadata (image digest, lock hashes) if present.
3. Add downloadable links/paths for manifest and summary artifacts.

### Commands
- `cd visualizer && bun run build`

### Acceptance tests
- Every dashboard state references source run metadata.
- Missing provenance fields degrade gracefully with explicit "not available".

---

## VW-05: Optimization Insights View

### Goal
Visualize optimization trajectories and holdout impact.

### Depends on
- `VW-04`
- `WP-06B` (optimization outputs)

### Files
- `visualizer/src/app/optimization/page.tsx` (new)
- `visualizer/src/components/charts/optimization-trajectory.tsx` (new)
- `visualizer/src/lib/types.ts`

### Tasks
1. Plot candidate score over iterations.
2. Show best candidate metadata and profile deltas vs seed.
3. Show holdout validation summary and confidence indicators.

### Commands
- `cd visualizer && bun run build`

### Acceptance tests
- Optimization page renders with real run artifacts.
- Seed vs best candidate comparison visible and reproducible.

---

## Visualizer Dependency Graph

### Adjacency list
- `VW-01 -> VW-02`
- `VW-02 -> VW-03`
- `VW-03 -> VW-04`
- `VW-04 -> VW-05`

### Cross-track dependencies
- `WP-03A -> VW-02`
- `WP-04A -> VW-03`
- `WP-05A -> VW-03`
- `WP-04B -> VW-03` (for paired support deltas)
- `WP-05B -> VW-03` (for paired tool deltas)
- `WP-05C -> VW-04`
- `WP-06A -> VW-04`
- `WP-06B -> VW-05`

### Recommended orchestration
1. Start `VW-01` immediately in parallel with `WP-01*` and `WP-02*`.
2. Start `VW-02` as soon as `WP-03A` is merged.
3. Start `VW-03` after `WP-04A` + `WP-05A`; enrich after `WP-04B` + `WP-05B`.
4. Start `VW-04` after telemetry/provenance fields land (`WP-05C`, `WP-06A`).
5. Start `VW-05` after optimization outputs are stable (`WP-06B`).

---

## Exit Criteria

Visualizer parallel track is complete when:
1. It consumes only adapter-backed stable data contracts.
2. Support/tool dimensions and paired deltas are explorable.
3. Provenance metadata is visible for reproducibility.
4. Optimization trajectory and holdout outcome are visualized.

