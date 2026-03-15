# User Testing

Testing surface, tools, and resource cost classification.

---

## Validation Surfaces

### CLI (Primary)
- **Tool:** tuistory
- **Commands:** `grist-mill run`, `grist-mill validate`, `grist-mill list`, `grist-mill optimize`, `grist-mill report`, `grist-mill export`
- **Setup:** `uv sync` must be run first
- **Coverage:** All subcommands, help text, error handling, dry-run mode

### Programmatic API (Secondary)
- **Tool:** pytest (no special tooling)
- **Surface:** `from grist_mill import ...` — import and use as library
- **Coverage:** Public API surface matches documented interfaces

### Docker Evaluation (Integration)
- **Tool:** Docker CLI (via subprocess)
- **Surface:** End-to-end task execution in Docker container
- **Setup:** Docker daemon must be running
- **Coverage:** Container lifecycle, artifact injection, result capture, cleanup

## Validation Concurrency

**Machine specs:** 31 GB RAM, 24 CPU cores, ~2.8 GB baseline usage.

### CLI (tuistory)
- Each tuistory instance: ~50 MB RAM
- No shared infrastructure between instances
- Max concurrent: **5** (250 MB total, well within budget)

### Docker Evaluation
- Each Docker container: ~200-500 MB RAM depending on image
- Shared Docker daemon
- Max concurrent: **3** (conservative, avoids Docker daemon strain)

## Known Constraints

- Integration tests requiring Docker skip gracefully when Docker is unavailable
- Integration tests requiring API keys skip gracefully without credentials
- No database or persistent state between test runs

## Setup Notes

- **Synthesis milestone** requires `uv sync --extra synthesis` to install `tree-sitter-language-pack`. Without this optional dependency, AST parsing and the end-to-end pipeline produce 0 tasks (graceful degradation per VAL-SYNTH-06). Always run `uv sync --extra synthesis` before testing synthesis milestone assertions.
- First subagent installed the dependency and all 206 synthesis tests passed. Second subagent (group-dataset) didn't need it. Third subagent (group-pipeline) needed it but stayed within isolation boundary.

## Flow Validator Guidance: harness-milestone

### Isolation Rules
- Each assertion group runs independently via pytest; no shared state between groups.
- Docker tests create/remove their own containers; no interference between groups.
- CLI tests write to stdout/stderr only; no file system side effects.
- All test files use their own fixtures with no shared mutable state.

### Testing Approach
- **Primary method:** Run pytest on specific test files that validate each assertion.
- **CLI verification:** Run `grist-mill run --config configs/examples/smoke.yaml` and `--output-format` flags.
- **Docker availability:** Docker is running; all Docker tests pass (44 tests).
- **Environment:** `uv sync` complete; `grist-mill --version` returns 0.1.0.

### Assertion-to-Test Mapping
| Assertion | Test File | Key Test Classes/Functions |
|-----------|-----------|---------------------------|
| VAL-HARNESS-01 | tests/test_result_parser.py | TestPassingOutput, TestFailingOutput, TestErrorOutput, TestTimeoutOutput |
| VAL-HARNESS-06, VAL-HARNESS-07 | tests/test_local_runner.py | TestLocalRunnerExecute, TestLocalRunnerTimeout |
| VAL-HARNESS-02..05, VAL-ENV-05, VAL-ENV-07 | tests/test_docker_runner.py | TestDockerRunnerCreateExecute, TestDockerArtifactInjection, TestDockerResourceLimits, TestDockerCleanup |
| VAL-ENV-01..04, VAL-ENV-06, VAL-ENV-08 | tests/test_environments.py | TestLanguageImageConfig, TestWorkspaceIsolation, TestNetworkAccess, TestHealthCheck |
| VAL-HARNESS-08, VAL-HARNESS-09, VAL-TELEM-02..03, VAL-TELEM-06..07 | tests/test_harness.py | TestHarnessWiring, TestRetryLogic, TestTelemetryCapture |
| VAL-CLI-02, VAL-CLI-06 | CLI: `grist-mill run` | Manual verification of exit code, JSON/YAML output |
| VAL-CROSS-01 | tests/test_cross_eval_export.py | TestEvalExportRoundTrip |
| VAL-CROSS-09, VAL-CROSS-12 | tests/test_cross_tool_binding.py | TestRegistryUnification, TestToolBindingIsolation |
