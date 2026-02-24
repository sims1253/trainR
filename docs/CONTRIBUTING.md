# Contributing: Adding New Harness Adapters

This guide explains how to add a new harness adapter to trainR. A harness adapter provides a unified interface for executing AI agents across different backends (Docker, CLI tools, SDKs, etc.).

## 1. Introduction

### Purpose

trainR uses a harness abstraction layer to support multiple agent execution backends. By adding a new harness adapter, you enable the benchmark system to work with your preferred agent framework or tool.

### Prerequisites

- **Python 3.10+**
- **uv** - Python package manager (`pip install uv`)
- **Docker** - Required for sandboxed execution (if using Docker-based harnesses)
- Basic familiarity with async Python and Pydantic

---

## 2. Development Setup

```bash
# Clone the repository
git clone https://github.com/posit-dev/trainR.git
cd trainR

# Install dependencies with uv
uv sync

# Verify setup by running tests
uv run pytest tests/ -v

# Optional: Build Docker image for Docker-based harnesses
make docker-build
```

---

## 3. Architecture Overview

For a complete architectural overview, see [docs/ARCHITECTURE.md](./ARCHITECTURE.md).

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Canonical Runner** | Single entry point (`bench.runner.run()`) for all benchmark execution |
| **Harness Abstraction** | `AgentHarness` base class that all adapters must extend |
| **Registry** | `HarnessRegistry` for dynamic discovery and instantiation of harnesses |
| **Request/Result** | `HarnessRequest` and `HarnessResult` define the input/output contract |

### Data Flow

```
Experiment Config
       |
       v
  Canonical Runner
       |
       v
  HarnessRegistry.get("my_harness", config)
       |
       v
  MyHarness.execute(request)
       |
       v
  HarnessResult
```

---

## 4. Adding a New Harness Adapter

### Step 1: Create the Adapter Class

Create a new file in `bench/harness/adapters/`:

```python
# bench/harness/adapters/my_harness.py

from __future__ import annotations

import time
import uuid
from pathlib import Path

from bench.harness.base import (
    AgentHarness,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    TokenUsage,
    TestResult,
    ErrorCategory,
)
from bench.harness.registry import register_harness


@register_harness("my_harness")
class MyHarness(AgentHarness):
    """My custom harness adapter.
    
    This harness [describe what it does and when to use it].
    """
    
    def __init__(self, config: HarnessConfig):
        super().__init__(config)
        # Initialize any harness-specific state
    
    def validate_environment(self) -> tuple[bool, list[str]]:
        """Check if the harness environment is ready.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Example: Check for required CLI tool
        # try:
        #     result = subprocess.run(
        #         ["my-cli", "--version"],
        #         capture_output=True,
        #         text=True,
        #     )
        #     if result.returncode != 0:
        #         errors.append("my-cli is not available")
        # except FileNotFoundError:
        #     errors.append("my-cli not found in PATH")
        
        return len(errors) == 0, errors
    
    def setup(self) -> None:
        """Set up the harness before execution.
        
        Override this to initialize resources (connections, containers, etc.).
        Called once before running tasks.
        """
        pass
    
    def teardown(self) -> None:
        """Clean up after execution.
        
        Override this to release resources. Called once after all tasks complete.
        """
        pass
    
    async def execute(self, request: HarnessRequest) -> HarnessResult:
        """Execute a task and return results.
        
        Args:
            request: The execution request containing task details
            
        Returns:
            HarnessResult with execution outcome
        """
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            # 1. Prepare the execution environment
            # 2. Run the agent with the prompt
            # 3. Capture output and test results
            # 4. Return the result
            
            # Example implementation:
            output = await self._run_agent(request)
            test_results = await self._run_tests(request)
            
            execution_time = time.time() - start_time
            
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=all(tr.passed for tr in test_results),
                output=output,
                tests_passed=all(tr.passed for tr in test_results),
                test_results=test_results,
                execution_time=execution_time,
                token_usage=TokenUsage(
                    prompt=100,  # Track actual usage
                    completion=200,
                    total=300,
                ),
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return HarnessResult(
                task_id=request.task_id,
                run_id=run_id,
                success=False,
                error_message=str(e),
                error_category=ErrorCategory.from_exception(e),
                execution_time=execution_time,
            )
    
    async def _run_agent(self, request: HarnessRequest) -> str:
        """Run the agent and return its output."""
        # Implement your agent execution logic
        return ""
    
    async def _run_tests(self, request: HarnessRequest) -> list[TestResult]:
        """Run tests and return results."""
        # Implement test execution logic
        return []
```

### Step 2: Register the Adapter

Update `bench/harness/adapters/__init__.py` to export your harness:

```python
# bench/harness/adapters/__init__.py

"""Harness adapter implementations."""

from .cli_base import CliHarnessBase
from .pi_docker import PiDockerHarness
from .my_harness import MyHarness  # Add import

__all__ = ["CliHarnessBase", "PiDockerHarness", "MyHarness"]  # Add to __all__
```

### Step 3: Add Config Schema

Update `bench/experiments/config.py` to add the harness type:

```python
# bench/experiments/config.py

HarnessType = Literal[
    "pi_docker",
    "pi_sdk",
    "pi_cli",
    "codex_cli",
    "claude_cli",
    "gemini_cli",
    "swe_agent",
    "my_harness",  # Add your harness here
]
```

### Step 4: Write Tests

Create tests for your harness:

```python
# tests/test_my_harness.py

import pytest

from bench.harness import HarnessRegistry, HarnessConfig, HarnessRequest


class TestMyHarness:
    """Tests for MyHarness adapter."""
    
    def test_harness_registered(self):
        """Verify the harness is registered."""
        assert "my_harness" in HarnessRegistry.list_available()
    
    def test_can_instantiate(self):
        """Verify the harness can be created."""
        config = HarnessConfig(timeout=60.0)
        harness = HarnessRegistry.get("my_harness", config)
        assert harness is not None
    
    def test_validate_environment(self):
        """Test environment validation."""
        config = HarnessConfig()
        harness = HarnessRegistry.get("my_harness", config)
        is_valid, errors = harness.validate_environment()
        # Adjust assertion based on expected environment
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        """Test that execute returns a valid HarnessResult."""
        config = HarnessConfig(timeout=60.0)
        harness = HarnessRegistry.get("my_harness", config)
        harness.setup()
        
        try:
            request = HarnessRequest(
                task_id="test-task-001",
                prompt="Write a function that adds two numbers",
            )
            result = await harness.execute(request)
            
            assert result.task_id == "test-task-001"
            assert result.run_id is not None
            assert isinstance(result.success, bool)
        finally:
            harness.teardown()
    
    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        """Test that the harness handles timeouts gracefully."""
        config = HarnessConfig(timeout=0.1)  # Very short timeout
        harness = HarnessRegistry.get("my_harness", config)
        
        request = HarnessRequest(
            task_id="timeout-test",
            prompt="This should timeout",
        )
        result = await harness.execute(request)
        
        assert result.success is False
        assert result.error_category in [
            ErrorCategory.TIMEOUT,
            ErrorCategory.UNKNOWN,
        ]
```

### Step 5: Update Documentation

1. Add your harness to the Available Adapters table in `docs/ARCHITECTURE.md`:

```markdown
| Adapter | Name | Description |
|---------|------|-------------|
| `PiDockerHarness` | `pi_docker` | Docker-based execution with Pi SDK (primary) |
| `MyHarness` | `my_harness` | Brief description of your harness |
```

2. Update `README.md` if your harness affects user-facing functionality.

---

## 5. Testing Guidelines

### Unit Tests

Test individual components in isolation:

```python
def test_error_classification():
    """Test that errors are properly classified."""
    from bench.harness.base import ErrorCategory
    
    # Test timeout classification
    exc = TimeoutError("Operation timed out")
    assert ErrorCategory.from_exception(exc) == ErrorCategory.TIMEOUT
```

### Integration Tests

Test with mock providers to avoid real API calls:

```python
@pytest.fixture
def mock_agent_response():
    """Mock agent response for testing."""
    return {"output": "def add(a, b): return a + b", "tokens": 50}

@pytest.mark.asyncio
async def test_integration_with_mock(mock_agent_response):
    """Test harness with mocked agent responses."""
    # Use unittest.mock or similar to mock external calls
    pass
```

### Contract Tests

Verify your harness adheres to the `AgentHarness` contract:

```python
def test_harness_contract():
    """Verify harness implements the required interface."""
    from bench.harness.base import AgentHarness
    
    config = HarnessConfig()
    harness = HarnessRegistry.get("my_harness", config)
    
    # Must be a subclass of AgentHarness
    assert isinstance(harness, AgentHarness)
    
    # Must implement required methods
    assert hasattr(harness, "execute")
    assert hasattr(harness, "validate_environment")
    assert callable(harness.execute)
    assert callable(harness.validate_environment)
```

---

## 6. Code Style

trainR uses **ruff** for linting and formatting, and **ty** for type checking.

```bash
# Run linter with auto-fix
uv run ruff check . --fix

# Format code
uv run ruff format .

# Type check
uv run ty check .
```

### Style Requirements

- **Type hints required** on all public functions and methods
- Use **async/await** for the `execute` method
- Follow **Pydantic v2** patterns for data models
- Keep methods focused and well-documented

---

## 7. Pull Request Process

### Before Submitting

1. **Fork and create a branch**:
   ```bash
   git checkout -b feature/my-harness-adapter
   ```

2. **Make your changes** following the steps above

3. **Run the full CI suite locally**:
   ```bash
   make ci
   ```
   
   Or for a quick check (~2 min):
   ```bash
   make ci-quick
   ```

4. **Ensure all tests pass**:
   ```bash
   uv run pytest tests/ -v
   ```

### Submitting

1. Push your branch to your fork
2. Open a pull request against `main`
3. Include in your PR description:
   - What harness you're adding and why
   - Any new dependencies introduced
   - Testing approach
   - Breaking changes (if any)

### Review Criteria

PRs are reviewed for:

- [ ] Correct implementation of `AgentHarness` interface
- [ ] Proper error handling and classification
- [ ] Adequate test coverage
- [ ] Documentation updates
- [ ] Code style compliance

---

## 8. Getting Help

- **GitHub Issues**: Open an issue for bugs, questions, or feature discussions
- **Architecture Doc**: See [docs/ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design
- **Existing Adapters**: Reference `bench/harness/adapters/pi_docker.py` as a complete example

---

## Quick Reference

### File Locations

| Component | Location |
|-----------|----------|
| Base classes | `bench/harness/base.py` |
| Registry | `bench/harness/registry.py` |
| Adapter implementations | `bench/harness/adapters/` |
| Config types | `bench/experiments/config.py` |
| Example adapter | `bench/harness/adapters/pi_docker.py` |

### Key Imports

```python
# Base types
from bench.harness.base import (
    AgentHarness,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
    TokenUsage,
    TestResult,
    ErrorCategory,
)

# Registry
from bench.harness.registry import register_harness, HarnessRegistry

# Convenience imports
from bench.harness import (
    HarnessRegistry,
    HarnessConfig,
    HarnessRequest,
    HarnessResult,
)
```

### Common Patterns

**Environment validation with Docker:**
```python
def validate_environment(self) -> tuple[bool, list[str]]:
    issues = []
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            issues.append("Docker is not available")
    except FileNotFoundError:
        issues.append("Docker not found")
    return len(issues) == 0, issues
```

**Error classification:**
```python
# Automatic classification from exceptions
error_category = ErrorCategory.from_exception(exc)

# Manual classification
if "rate limit" in str(exc).lower():
    error_category = ErrorCategory.RATE_LIMIT
```

**Accessing request metadata:**
```python
# HarnessRequest has a metadata field for extra context
skill_content = request.metadata.get("skill_content")
package_dir = request.metadata.get("package_dir", "packages/default")
```
