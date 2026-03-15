# Architecture

Architectural decisions, patterns, and conventions for grist-mill.

---

## Core Design Principles

1. **BenchFlow-style decoupling**: `BaseAgent`, `BaseBenchmark`, `BaseEnvironment`, `BaseHarness` are abstract interfaces. Concrete implementations are pluggable via registries.

2. **Artifact-first**: Tools, MCP servers, skills, and other pluggable artifacts are first-class objects registered in a central `ArtifactRegistry`. The harness wires artifacts into the execution environment.

3. **Pydantic v2 throughout**: All data models use Pydantic v2 with discriminated unions for polymorphic types.

4. **Config precedence**: CLI args > environment variables (`GRIST_MILL_*`) > YAML defaults.

5. **Telemetry everywhere**: Every execution produces telemetry with token usage, latency breakdown, and tool call metrics. The harness merges two telemetry sources: agent-level telemetry (tool calls, tokens, cost from the agent/provider) and harness-level telemetry (setup/execution/teardown phase timings). The `_merge_harness_and_agent_telemetry` function in `harness.py` combines these sources. Note: the TIMEOUT/SKIPPED code path replaces agent telemetry entirely with harness telemetry (known limitation).

## Module Structure

```
src/grist_mill/
├── schemas/          # Pydantic v2 models
├── interfaces/       # Abstract base classes
├── registry/         # Artifact and agent registries
├── config/           # Configuration loading
├── harness/          # Harness implementation
├── environments/     # Docker and local runners
├── agents/           # Agent implementations
├── tools/            # Tool orchestration
├── providers/        # LLM provider abstraction
├── telemetry/        # Telemetry collection
├── optimization/     # GEPA integration
├── tasks/            # Task synthesis
├── dataset/          # Dataset management
├── reports/          # Result analysis and export
└── cli/              # CLI entrypoint
```

## Synthesis Module Details (M4)

### tasks/ — Task Synthesis
- `ast_parser.py`: Tree-sitter based AST parser for Python, R, TypeScript/TSX. Auto-detects language from file extension or shebang. Returns structured AST nodes (functions, test blocks, imports, classes). Graceful error handling returns partial results.
- `mutation.py`: Test mutation pipeline (1392 lines). 5 mutation types: logic_bug, missing_import, type_error, wrong_return_value, edge_case. Supports apply/revert via diff/patch. Generates natural-language task descriptions. Pluggable mutator registry.
- `pipeline.py`: End-to-end `TaskPipeline` class orchestrating: source discovery → AST analysis → mutation → quality gates → difficulty estimation → dataset building → registry registration → export. CLI subcommand: `grist-mill tasks generate --repo <path>`.

### dataset/ — Dataset Management
- `core.py`: `Dataset` is a **plain Python class** (not Pydantic) by design — it represents mutable working state. `DatasetVersion` is a **frozen dataclass** for immutable snapshots. Versioning uses `copy.deepcopy`.
- `splitting.py`: Stratified splitting by difficulty, with implicit language preservation (shuffled within difficulty groups).
- `difficulty.py`: Heuristic scoring with thresholds: `_EASY_THRESHOLD=3`, `_HARD_THRESHOLD=7`. Factors: prompt complexity keywords, timeout ranges, test command complexity, dependency count, constraint count.
- `decontamination.py`: O(n²) pairwise similarity comparison. Adequate for datasets < 1000 tasks; LSH/MinHash recommended for larger datasets.
- `versioning.py`: Immutable version snapshots via frozen dataclasses.
- `export.py`: JSON (with schema_version + generated_at metadata) and CSV export. CSV does not include metadata headers.
- `quality.py`: Quality report with task counts by language/difficulty/category. Truncates issue descriptions for readability (full task_ids in DatasetQualityIssue.task_ids).
- `yaml_import.py`: Manual task authoring via YAML import with Task model validation.

## Key Patterns

- **Registry pattern**: `ArtifactRegistry`, `AgentRegistry`, `ProviderRegistry` — all use decorator-based registration
- **Strategy pattern**: Different runners (Docker, local), different providers (OpenRouter, OpenAI) are interchangeable strategies
- **Observer pattern**: Telemetry collectors observe execution phases and record metrics
- **Builder pattern**: `HarnessConfig.builder()` for constructing complex configurations
