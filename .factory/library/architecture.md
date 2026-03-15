# Architecture

Architectural decisions, patterns, and conventions for grist-mill.

---

## Core Design Principles

1. **BenchFlow-style decoupling**: `BaseAgent`, `BaseBenchmark`, `BaseEnvironment`, `BaseHarness` are abstract interfaces. Concrete implementations are pluggable via registries.

2. **Artifact-first**: Tools, MCP servers, skills, and other pluggable artifacts are first-class objects registered in a central `ArtifactRegistry`. The harness wires artifacts into the execution environment.

3. **Pydantic v2 throughout**: All data models use Pydantic v2 with discriminated unions for polymorphic types.

4. **Config precedence**: CLI args > environment variables (`GRIST_MILL_*`) > YAML defaults.

5. **Telemetry everywhere**: Every execution produces telemetry with token usage, latency breakdown, and tool call metrics.

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

## Key Patterns

- **Registry pattern**: `ArtifactRegistry`, `AgentRegistry`, `ProviderRegistry` — all use decorator-based registration
- **Strategy pattern**: Different runners (Docker, local), different providers (OpenRouter, OpenAI) are interchangeable strategies
- **Observer pattern**: Telemetry collectors observe execution phases and record metrics
- **Builder pattern**: `HarnessConfig.builder()` for constructing complex configurations
