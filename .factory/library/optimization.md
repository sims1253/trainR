# Optimization

GEPA integration, optimization runtime, and target types for grist-mill.

---

## GEPA Evaluator Protocol

The GEPA (`gepa` PyPI package) evaluator expects a callable conforming to this protocol:

```python
def __call__(
    candidate: Any,
    example: Any,
    **kwargs
) -> tuple[float, dict[str, Any]]
```

- Returns `(score, side_info_dict)` where score is a scalar and side_info contains actionable metadata.
- `side_info` is forwarded to the reflection model that proposes the next candidate.
- Proposals should reference specific errors from side_info.

Discovered via runtime introspection: `import gepa.optimize_anything; inspect.signature(...)`.

## Module Structure: optimization/

```
src/grist_mill/optimization/
├── __init__.py           # Module exports
├── evaluator_adapter.py  # GEPA evaluator adapter (813 lines)
└── runtime.py            # Optimization runner (1096 lines)
```

## evaluator_adapter.py — GEPA Evaluator Adapter

### Key Classes
- **`ObjectiveFunction`** (ABC): Base for scoring functions. Subclasses: `PassRateObjective`, `CostAdjustedObjective`, `DifficultyWeightedObjective`.
- **`EvaluatorAdapterConfig`** (Pydantic v2): Configuration for the adapter with `objective` field as a Literal type.
- **`EvaluatorAdapter`**: Wraps the evaluation harness as a GEPA evaluator. Returns `(float_score, dict_side_info)`.

### Side Info Structure
The side_info dict captures per-evaluation:
- `traces`: Execution trace data
- `errors`: Error messages and error_category
- `duration_s`: Wall-clock timing
- `token_usage`: `{"prompt": N, "completion": N, "total": N}`
- `task_id`, `status`: Task identification
- `tool_calls`: Tool invocation details
- `difficulty`: Task difficulty label (for reflection model)
- `raw_events`: Trace events when `trace_enabled=True`

### Custom Evaluators
- `load_custom_evaluator(path)` loads any callable from a Python file.
- `create_evaluator_adapter(config, harness)` factory function creates adapters from config.

### Known Design Note
`_run_harness` uses `inspect.signature()` + try/except to discover the harness.run() signature. The real `Harness.run(*, task, agent, env, collector)` requires agent/env params. The adapter works with mock harnesses in tests; integrating with the real Harness requires a wrapper that provides agent/env from config.

## runtime.py — Optimization Runtime

### Key Classes
- **`BudgetConfig`** (Pydantic v2): Budget settings with `max_calls`, `timeout_s`, `no_improvement_patience`.
- **`StopCondition`** (ABC): Composable stop conditions. Subclasses: `MaxCallsCondition`, `TimeoutCondition`, `NoImprovementCondition`. `should_stop()` returns `(stop: bool, reason: str | None)`.
- **`ParetoFront`**: Maintains accepted candidates. `_dominates()` implements correct Pareto dominance — no accepted candidate degrades any objective.
- **`TargetType`**: Enum with `skill`, `system_prompt`, `tool_policy`.
- **`TargetConfig`** (Pydantic v2): Target configuration with validation.
- **`BaseProposer`** (ABC): Abstract proposer interface. Subclasses implement `propose(best, side_info) -> candidate`.
- **`OptimizationCheckpoint`** (Pydantic v2): Persisted state — candidate pool, Pareto front, iteration count, scores, budget counters.
- **`OptimizationRunner`**: Main runner loop with signal handling (SIGTERM → graceful checkpoint).

### Checkpoint Format
```python
class OptimizationCheckpoint:
    candidates: list[dict]
    pareto_front: list[dict]
    iteration: int
    scores: list[float]
    budget_used: dict  # {"calls": int, "elapsed_s": float, "no_improvement": int}
    best_candidate: dict | None
    best_score: float
    target_type: str
    target_config: dict
```

### CLI Subcommand
`grist-mill optimize --config optimize.yaml` runs end-to-end. Produces `best_candidate.json` and `trajectory.jsonl`.

### Signal Handling
SIGTERM triggers graceful shutdown: saves checkpoint, records termination reason, cleans up. No orphaned processes.

## Legacy Optimization Reference

The legacy optimization code lives at `optimization/adapter.py` and `optimization/config.py` (project root, not in src/). The new implementation (M5) was informed by these patterns but uses clean Pydantic v2 models and the GEPA evaluator protocol.
