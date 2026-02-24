# Architecture Documentation

This document describes the layered architecture of the trainR benchmark execution system, focusing on the harness/provider/sandbox abstraction layers.

## 1. Overview

### Purpose

The trainR architecture provides a unified, extensible framework for running AI agent benchmarks on R package development tasks. The system is designed around three core principles:

### Key Design Principles

1. **Canonical Execution**
   - Single entry point (`bench.runner.run()`) for all benchmark execution
   - Consistent artifact generation (manifest, results, summaries)
   - Unified telemetry and logging
   - Deterministic outputs with configurable seeds

2. **Pluggable Adapters**
   - Harness abstraction allows swapping execution backends (Pi SDK, CLI tools, etc.)
   - Registry pattern for dynamic discovery and instantiation
   - Clean contracts via `HarnessRequest`/`HarnessResult` data structures

3. **Policy-Driven Configuration**
   - Sandbox profiles for security isolation
   - Authentication policies for credential management
   - Declarative experiment configuration via YAML

---

## 2. Layer Diagram

```
+---------------------------------------------------------+
|                   Entry Points                          |
|   scripts/run_experiment.py  |  bench.runner.run()      |
+---------------------------------------------------------+
                          |
                          v
+---------------------------------------------------------+
|              Canonical Runner (bench.runner)            |
|   - Preflight validation                                |
|   - Harness selection                                   |
|   - Result aggregation                                  |
+---------------------------------------------------------+
                          |
          +---------------+---------------+
          v               v               v
+-----------------+ +-----------------+ +-----------------+
|   Provider      | |     Harness     | |    Sandbox      |
|   Resolver      | |    Registry     | |    Policy       |
| (bench.provider)| | (bench.harness) | | (bench.sandbox) |
+-----------------+ +-----------------+ +-----------------+
          |               |               |
          +---------------+---------------+
                          v
+---------------------------------------------------------+
|              Execution Layer                            |
|   DockerPiRunner  |  CliHarnessBase  |  Future adapters |
+---------------------------------------------------------+
```

### Data Flow

1. **Entry Point** loads configuration and calls `bench.runner.run()`
2. **Canonical Runner** validates config, generates experiment matrix, runs preflight
3. **Provider Resolver** resolves model-to-provider mappings and validates credentials
4. **Harness Registry** instantiates the appropriate execution adapter
5. **Sandbox Policy** configures security settings for containerized execution
6. **Execution Layer** runs the actual agent evaluation

---

## 3. Component Details

### bench.runner

The canonical execution API - the single supported runtime path for running benchmarks.

#### Location
`bench/runner.py`

#### Function Signature

```python
def run(
    config: Union[str, Path, ExperimentConfig],
    *,
    output_dir: str | None = None,
    seed: int | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    **kwargs: Any,
) -> ManifestV1:
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str \| Path \| ExperimentConfig` | Config file path or object |
| `output_dir` | `str \| None` | Override output directory |
| `seed` | `int \| None` | Random seed for reproducibility |
| `workers` | `int \| None` | Parallel worker count override |
| `dry_run` | `bool` | Show matrix without executing |
| `validate_only` | `bool` | Validate config only |

#### Returns

`ManifestV1` - Contains run metadata, fingerprints, and result summaries.

#### Lifecycle

```
setup -> preflight -> execute -> aggregate -> teardown
```

1. **Setup**: Create output directory, generate experiment matrix
2. **Preflight**: Validate credentials, check harness environment
3. **Execute**: Run all task/model combinations via harness
4. **Aggregate**: Compute summaries, finalize manifest
5. **Teardown**: Clean up resources

#### Usage Example

```python
from bench.runner import run

# From config path
manifest = run("configs/experiments/smoke.yaml")

# With overrides
manifest = run("config.yaml", output_dir="results/custom", seed=42)

# Dry run to preview
manifest = run("config.yaml", dry_run=True)
print(f"Total runs: {manifest.task_count}")
```

---

### bench.harness

The harness abstraction layer provides a unified interface for different agent execution backends.

#### Location
`bench/harness/`

#### Core Types

##### AgentHarness (Protocol)

Abstract base class that all harness implementations must extend:

```python
class AgentHarness(ABC):
    def __init__(self, config: HarnessConfig) -> None: ...
    
    @abstractmethod
    async def execute(self, request: HarnessRequest) -> HarnessResult: ...
    
    @abstractmethod
    def validate_environment(self) -> tuple[bool, list[str]]: ...
    
    def setup(self) -> None: ...      # Optional lifecycle hook
    def teardown(self) -> None: ...   # Optional lifecycle hook
```

##### HarnessRequest

Request payload containing all information needed to execute an agent:

```python
class HarnessRequest(BaseModel):
    task_id: str                    # Unique task identifier
    prompt: str                     # Main instruction for the agent
    system_prompt: str | None       # System prompt override
    repository: str | None          # Repository URL or path
    base_commit: str | None         # Base commit SHA
    files: dict[str, str]           # File path -> content mapping
    test_command: str | None        # Command to run tests
    test_files: list[str]           # Paths to test files
    timeout: float | None           # Override timeout
    max_tokens: int | None          # Override max tokens
    metadata: dict[str, Any]        # Additional context
```

##### HarnessResult

Result of an agent execution:

```python
class HarnessResult(BaseModel):
    task_id: str                    # Task identifier
    run_id: str                     # Execution run ID
    success: bool                   # Whether execution succeeded
    error_category: ErrorCategory   # Error classification
    error_message: str | None       # Error details
    output: str                     # Agent output
    patch: str | None               # Generated diff/patch
    tests_passed: bool              # Whether tests passed
    test_results: list[TestResult]  # Individual test results
    token_usage: TokenUsage         # Token consumption
    execution_time: float           # Total execution time
    model: str | None               # Model used
```

##### HarnessConfig

Configuration for harness instances:

```python
class HarnessConfig(BaseModel):
    timeout: float = 300.0          # Max execution time (seconds)
    max_retries: int = 3            # Retry attempts
    retry_delay: float = 1.0        # Delay between retries
    max_tokens: int | None          # Max completion tokens
    max_turns: int | None           # Max conversation turns
    working_dir: str | None         # Working directory
    env_vars: dict[str, str]        # Environment variables
    sandbox_enabled: bool = True    # Use sandboxing
    network_access: bool = False    # Allow network access
    model: str | None               # Model identifier
```

#### HarnessRegistry

Registry for dynamic harness discovery and instantiation:

```python
# Register a harness
@register_harness("my_harness")
class MyHarness(AgentHarness):
    ...

# Or manually register
HarnessRegistry.register("another_harness", AnotherHarness)

# Get a harness instance
harness = HarnessRegistry.get("my_harness", config)

# List available harnesses
available = HarnessRegistry.list_available()
# Returns: ['pi_docker', 'cli_base', ...]
```

#### Available Adapters

| Adapter | Name | Description |
|---------|------|-------------|
| `PiDockerHarness` | `pi_docker` | Docker-based execution with Pi SDK (primary) |
| `CliHarnessBase` | `cli_base` | Base class for CLI-based adapters |

##### PiDockerHarness

The primary harness for R package testing. Runs the Pi CLI inside a Docker container with sandboxing.

**Features:**
- Full isolation via Docker containers
- Configurable sandbox profiles
- Automatic test execution and result parsing
- Token usage tracking

##### CliHarnessBase

Abstract base for CLI-based harness adapters. Subclasses implement:
- `cli_name`: Name of the CLI tool
- `build_cli_args()`: Build CLI arguments from request
- `parse_output()`: Parse CLI output into HarnessResult

---

### bench.provider

Provider resolution and credential management - the single source of truth for model-to-provider mappings.

#### Location
`bench/provider/`

#### ProviderResolver

Resolves providers and models from `configs/llm.yaml`:

```python
from bench.provider import ProviderResolver, get_provider_resolver

# Get singleton instance
resolver = get_provider_resolver()

# Resolve provider for a model
provider = resolver.resolve_provider("glm-5")  # Returns: "opencode"

# Get API key environment variable
api_key_env = resolver.get_api_key_env("opencode")  # Returns: "OPENCODE_API_KEY"

# Get LiteLLM prefix
prefix = resolver.get_litellm_prefix("opencode")  # Returns: "openai/"

# Get full LiteLLM model string
litellm_model = resolver.get_litellm_model("glm-5")  # Returns: "openai/glm-5-free"
```

#### CredentialResolver

Handles credential loading based on authentication policy:

```python
from bench.provider import CredentialResolver, AuthPolicy

# Create resolver with policy
resolver = CredentialResolver(AuthPolicy.ENV)

# Resolve credential
info = resolver.resolve("OPENROUTER_API_KEY")
print(info.is_valid)     # True if set
print(info.value)        # The actual key (handle carefully!)
print(info.redacted())   # Safe for logging: "sk-o...1234"
```

#### AuthPolicy

```python
class AuthPolicy(Enum):
    ENV = "env"                  # Load from environment variables
    MOUNTED_FILE = "mounted_file"  # Load from /run/secrets/ (Kubernetes)
```

#### Preflight Validation

Validates credentials before execution:

```python
from bench.provider import run_preflight, AuthPolicy

result = run_preflight(
    models=["glm-5", "gpt-oss-120b"],
    auth_policy=AuthPolicy.ENV,
    strict=True,  # Fail on missing credentials
)

if not result.is_valid:
    print("Errors:", result.errors)
print("Warnings:", result.warnings)
```

#### Provider/Model Info

```python
@dataclass
class ProviderInfo:
    name: str                    # Provider name
    litellm_prefix: str          # LiteLLM prefix
    api_key_env: str             # API key env var name
    base_url: str | None         # Custom API URL
    supports_response_format: bool

@dataclass
class ModelInfo:
    id: str                      # Model identifier
    provider: str                # Provider name
    name: str                    # Short name
    litellm_model: str           # Full LiteLLM string
    capabilities: dict[str, Any] # Model capabilities
```

---

### bench.sandbox

Sandbox security policy management for containerized execution.

#### Location
`bench/sandbox/`

#### SandboxProfile

Named security profiles:

```python
class SandboxProfile(str, Enum):
    STRICT = "strict"        # Default: non-root, readonly FS, no network
    NETWORKED = "networked"  # Explicit outbound network access
    DEVELOPER = "developer"  # Relaxed local-debug settings
```

#### SandboxPolicy

Detailed security configuration:

```python
@dataclass
class SandboxPolicy:
    profile: SandboxProfile = SandboxProfile.STRICT
    
    # Security settings
    run_as_non_root: bool = True
    read_only_root_fs: bool = True
    drop_capabilities: list[str] = ["ALL"]
    
    # Resource limits
    cpu_limit: str = "2"          # 2 CPUs
    memory_limit: str = "4g"      # 4GB
    pids_limit: int = 256
    
    # Network settings
    network_enabled: bool = False
    
    # Writable paths (as tmpfs)
    writable_paths: list[str] = ["/tmp", "/var/tmp"]
```

#### Profile Details

| Profile | Network | Root FS | User | Capabilities | Use Case |
|---------|---------|---------|------|--------------|----------|
| `strict` | None | Read-only | nobody | Dropped | CI/benchmarks |
| `networked` | Enabled | Read-only | nobody | Dropped | Package installation |
| `developer` | Enabled | Writable | Root | Kept | Local debugging |

#### DockerCommandBuilder

Builds Docker run commands from policy:

```python
from bench.sandbox import DockerCommandBuilder, get_sandbox_policy

policy = get_sandbox_policy("networked")
builder = DockerCommandBuilder(policy)

cmd = builder.build_run_command(
    image="posit-gskill-eval:latest",
    command=["pi", "run", "--task", "task.json"],
    env_vars={"MODEL": "glm-5"},
    volumes=[("/host/path", "/container/path", "ro")],
)
# Returns: ["docker", "run", "--rm", "--user", "nobody", ...]
```

---

## 4. Configuration Schema

Experiment configuration is defined in `bench/experiments/config.py` and loaded from YAML files.

### execution.harness

Specifies which harness adapter to use:

```yaml
execution:
  harness: pi_docker  # Options: pi_docker, pi_sdk, pi_cli, codex_cli, claude_cli, gemini_cli, swe_agent
```

### execution.sandbox_profile

Security profile for containerized execution:

```yaml
execution:
  sandbox_profile: strict  # Options: strict, networked, developer
```

### execution.auth_policy

How credentials should be loaded:

```yaml
execution:
  auth_policy: env  # Options: env, mounted_file
```

### Full Configuration Example

```yaml
name: my_experiment
description: Example experiment configuration

tasks:
  selection: splits
  splits: [dev, held_out]
  max_tasks: 100

models:
  names: [glm-5, gpt-oss-120b]

skill:
  path: skills/r_package_expert.md

execution:
  harness: pi_docker
  sandbox_profile: networked
  timeout: 600
  docker_image: posit-gskill-eval:latest
  repeats: 1
  parallel_workers: 2
  save_trajectories: true
  auth_policy: env

retry:
  strategy: exponential
  max_retries: 2
  base_delay: 1.0

output:
  dir: results/experiments
  save_intermediate: true

determinism:
  seed: 42
```

---

## 5. Adding New Components

### How to Add a New Harness Adapter

1. **Create the adapter class** in `bench/harness/adapters/`:

```python
# bench/harness/adapters/my_harness.py

from ..base import (
    AgentHarness,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    ErrorCategory,
)
from ..registry import register_harness

@register_harness("my_harness")
class MyHarness(AgentHarness):
    """My custom harness implementation."""
    
    def __init__(self, config: HarnessConfig):
        super().__init__(config)
    
    async def execute(self, request: HarnessRequest) -> HarnessResult:
        # Implement execution logic
        return HarnessResult(
            task_id=request.task_id,
            run_id="...",
            success=True,
            # ...
        )
    
    def validate_environment(self) -> tuple[bool, list[str]]:
        # Check prerequisites
        return True, []
    
    def setup(self) -> None:
        # Optional: Initialize resources
        pass
    
    def teardown(self) -> None:
        # Optional: Clean up resources
        pass
```

2. **Export from adapters `__init__.py`**:

```python
# bench/harness/adapters/__init__.py
from .my_harness import MyHarness

__all__ = ["CliHarnessBase", "PiDockerHarness", "MyHarness"]
```

3. **Add to harness `__init__.py`**:

```python
# bench/harness/__init__.py
from .adapters import PiDockerHarness, MyHarness

__all__ = [
    # ...existing exports...
    "MyHarness",
]
```

4. **Add to HarnessType literal** in `bench/experiments/config.py`:

```python
HarnessType = Literal[
    "pi_docker",
    "my_harness",  # Add here
    # ...
]
```

### How to Add a New Provider

1. **Add to `configs/llm.yaml`**:

```yaml
providers:
  my_provider:
    api_key_env: MY_PROVIDER_API_KEY
    litellm_prefix: my_provider/
    base_url: https://api.myprovider.com/v1

models:
  my_model:
    id: my-model-v1
    provider: my_provider
    capabilities:
      reasoning: true
      json_mode: native
```

2. **Update canonical mappings** (optional) in `bench/provider/resolver.py`:

```python
PROVIDER_API_KEY_MAP = {
    # ...existing mappings...
    "my_provider": "MY_PROVIDER_API_KEY",
}

PROVIDER_LITELLM_PREFIX = {
    # ...existing mappings...
    "my_provider": "my_provider/",
}
```

### How to Add a New Sandbox Profile

1. **Add to `SandboxProfile` enum** in `bench/sandbox/policy.py`:

```python
class SandboxProfile(str, Enum):
    STRICT = "strict"
    NETWORKED = "networked"
    DEVELOPER = "developer"
    CUSTOM = "custom"  # Add new profile
```

2. **Add profile configuration** in `SandboxPolicy.from_profile()`:

```python
@classmethod
def from_profile(cls, profile: SandboxProfile) -> "SandboxPolicy":
    # ...existing cases...
    elif profile == SandboxProfile.CUSTOM:
        return cls(
            profile=profile,
            run_as_non_root=True,
            read_only_root_fs=False,
            network_enabled=True,
            drop_capabilities=["NET_RAW"],
        )
```

---

## 6. Migration Guide

### From Legacy Scripts to Canonical Runner

**Before (legacy):**

```python
# Old pattern - direct script execution
from evaluation.pi_runner import DockerPiRunner

runner = DockerPiRunner(config)
result = runner.run_evaluation(...)
```

**After (canonical):**

```python
# New pattern - use canonical runner
from bench.runner import run

manifest = run(
    "configs/experiments/my_experiment.yaml",
    seed=42,
    workers=2,
)
```

### Config Migration

**Before (scattered config):**

```python
# Multiple config sources
MODEL = os.environ.get("MODEL", "gpt-4")
TIMEOUT = int(os.environ.get("TIMEOUT", "300"))
DOCKER_IMAGE = "my-image:latest"
```

**After (unified YAML):**

```yaml
# configs/experiments/my_experiment.yaml
models:
  names: [glm-5]

execution:
  harness: pi_docker
  timeout: 300
  docker_image: posit-gskill-eval:latest
```

### Entry Point Migration

| Old Entry Point | New Entry Point |
|-----------------|-----------------|
| `scripts/run_benchmark.py` | `scripts/run_experiment.py` |
| `scripts/run_evaluation.py` | `bench.runner.run()` |
| Direct `DockerPiRunner` usage | Via `HarnessRegistry.get()` |

### Artifact Location

| Artifact | Old Location | New Location |
|----------|--------------|--------------|
| Results | `results/` | `{output_dir}/{run_id}/results.jsonl` |
| Summary | Various | `{output_dir}/{run_id}/summary.json` |
| Manifest | None | `{output_dir}/{run_id}/manifest.json` |
| Matrix | None | `{output_dir}/{run_id}/matrix.json` |
| Trajectories | `trajectories/` | `{output_dir}/{run_id}/trajectories/` |

---

## Quick Reference

### Module Locations

| Component | Module |
|-----------|--------|
| Canonical Runner | `bench.runner` |
| Harness Base | `bench.harness.base` |
| Harness Registry | `bench.harness.registry` |
| Pi Docker Harness | `bench.harness.adapters.pi_docker` |
| CLI Base Harness | `bench.harness.adapters.cli_base` |
| Provider Resolver | `bench.provider.resolver` |
| Credential Resolver | `bench.provider.auth` |
| Preflight | `bench.provider.preflight` |
| Sandbox Policy | `bench.sandbox.policy` |
| Docker Builder | `bench.sandbox.docker` |
| Experiment Config | `bench.experiments.config` |
| Experiment Runner | `bench.experiments.runner` |

### Common Imports

```python
# Run experiments
from bench.runner import run

# Harness abstraction
from bench.harness import (
    HarnessRegistry,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    AgentHarness,
)

# Provider resolution
from bench.provider import (
    ProviderResolver,
    CredentialResolver,
    AuthPolicy,
    run_preflight,
)

# Sandbox policies
from bench.sandbox import (
    SandboxPolicy,
    SandboxProfile,
    DockerCommandBuilder,
)

# Configuration
from bench.experiments import ExperimentConfig, load_experiment_config
```
