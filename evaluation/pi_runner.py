"""Pi SDK-based evaluation harness for R package testing tasks.

Replaces cc-mirror + Docker with direct Pi CLI calls.
Simpler, faster, and supports free models via OpenRouter.
"""

import contextlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Canonical Pi model prefixes by provider.
PI_MODEL_PREFIX_BY_PROVIDER: dict[str, str] = {
    "openrouter": "openrouter",
    "opencode": "opencode",
    "zai": "zai",
    "zai_coding_plan": "zai",
    "openai": "openai",
    "kimi": "kimi",
    "kimi_coding_plan": "kimi",
}


def _bootstrap_env() -> None:
    """Bootstrap environment with lazy import to avoid circular imports."""
    try:
        from bench.provider import bootstrap_environment

        bootstrap_environment()
    except ImportError:
        pass


def _get_env_var(name: str, default: str | None = None) -> str | None:
    """Read env var through provider bootstrap when available."""
    _bootstrap_env()
    try:
        from bench.provider import get_env_var

        return get_env_var(name, default)
    except ImportError:
        import os

        return os.environ.get(name, default)


def _get_api_key_aliases() -> dict[str, tuple[str, ...]]:
    """Get API key aliases from provider env module if available."""
    try:
        from bench.provider import api_key_aliases

        return api_key_aliases()
    except ImportError:
        return {}


def _provider_api_key_map() -> dict[str, str]:
    """Get provider->API-key mapping with canonical defaults."""
    try:
        from bench.provider.resolver import PROVIDER_API_KEY_MAP

        return dict(PROVIDER_API_KEY_MAP)
    except ImportError:
        return {
            "openrouter": "OPENROUTER_API_KEY",
            "zai": "Z_AI_API_KEY",
            "zai_coding_plan": "Z_AI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "opencode": "OPENCODE_API_KEY",
            "kimi": "KIMI_API_KEY",
        }


def _build_pi_model(provider: str, model_id: str) -> str:
    """Build Pi model string from provider + model_id."""
    prefix = PI_MODEL_PREFIX_BY_PROVIDER.get(provider)
    if prefix:
        return f"{prefix}/{model_id}"
    return model_id


def _resolve_pi_model(llm_config: Any, model: str) -> tuple[str, str]:
    """Resolve provider and Pi model string from model input.

    Supports:
    - short model names from llm.yaml (e.g., "glm-5-free")
    - legacy LiteLLM aliases (e.g., "openai/glm-5-free")
    - explicit provider-prefixed IDs (e.g., "openrouter/openai/gpt-oss-120b:free")

    Returns:
        Tuple of (provider_name, pi_model). Provider is empty when unknown.
    """
    # 1) Direct llm.yaml lookup by model name.
    try:
        model_cfg = llm_config.get_model_config(model)
        provider = model_cfg.get("provider", "")
        model_id = model_cfg.get("id", model)
        return provider, _build_pi_model(provider, model_id)
    except ValueError:
        pass

    # 2) Reverse lookup for legacy LiteLLM aliases.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for model_name in llm_config.list_models():
            with contextlib.suppress(ValueError):
                if llm_config.get_litellm_model(model_name) != model:
                    continue
                model_cfg = llm_config.get_model_config(model_name)
                provider = model_cfg.get("provider", "")
                model_id = model_cfg.get("id", model)
                return provider, _build_pi_model(provider, model_id)

    # 3) Provider-prefixed model passthrough.
    if "/" in model:
        provider, model_id = model.split("/", 1)
        if provider in PI_MODEL_PREFIX_BY_PROVIDER:
            return provider, _build_pi_model(provider, model_id)

    return "", model


def _resolve_api_key_env_name(provider_name: str) -> str:
    """Resolve canonical API key env name for provider."""
    from bench.provider import resolve_api_key_env

    return resolve_api_key_env(provider_name)


def _resolve_credential_value(env_var: str, auth_policy: str = "env") -> str | None:
    """Resolve credential value using auth policy, with env fallback."""
    try:
        from bench.provider import AuthPolicy, get_credentials

        return get_credentials(env_var, policy=AuthPolicy(auth_policy))
    except (ImportError, ValueError):
        return _get_env_var(env_var)


def _get_all_provider_api_keys() -> list[str]:
    """Get list of all known provider API key environment variables.

    Uses the central resolver if available, falls back to hardcoded list.
    """
    _bootstrap_env()
    try:
        from bench.provider.resolver import get_provider_resolver

        resolver = get_provider_resolver()
        keys = set()
        # Add keys from resolver
        for provider_name in resolver.providers:
            with contextlib.suppress(KeyError):
                keys.add(resolver.get_api_key_env(provider_name))
        # Also include canonical keys
        keys.update(_provider_api_key_map().values())
        # Include alias keys for compatibility with provider SDK/CLI expectations.
        for canonical, aliases in _get_api_key_aliases().items():
            if canonical in keys:
                keys.update(aliases)
        return list(keys)
    except (ImportError, FileNotFoundError):
        pass

    # Fallback to hardcoded list (deprecated)
    warnings.warn(
        "Using fallback API key list in pi_runner. Prefer bench.provider for provider resolution.",
        DeprecationWarning,
        stacklevel=3,
    )
    keys = set(_provider_api_key_map().values())
    for canonical, aliases in _get_api_key_aliases().items():
        if canonical in keys:
            keys.update(aliases)
    return sorted(keys)


@dataclass
class DockerPiRunnerConfig:
    """Configuration for Docker-based Pi runner."""

    model: str = "openrouter/openai/gpt-oss-120b:free"
    max_turns: int = 50
    timeout: int = 600
    docker_image: str = "posit-gskill-eval:latest"
    # API keys to pass to container (only if set in environment)
    api_keys: list[str] | None = None
    # Opt-in forwarding for GitHub token. Disabled by default to avoid leaking secrets.
    forward_github_token: bool = False
    # Sandbox profile for security settings (string: "strict", "networked", or "developer")
    sandbox_profile: str = "networked"
    # Credential resolution policy: "env" or "mounted_file"
    auth_policy: str = "env"
    # Retain workspace on failure for debugging
    keep_workspace_on_failure: bool = False
    # Save tool traces for debugging
    save_traces: bool = False
    # Directory to save tool traces
    trace_dir: Path = Path("container_logs")

    def __post_init__(self):
        if self.api_keys is None:
            # Default: forward all known API keys if set
            self.api_keys = _get_all_provider_api_keys()
        # Validate sandbox profile
        valid_profiles = ("strict", "networked", "developer")
        if self.sandbox_profile not in valid_profiles:
            raise ValueError(
                f"Invalid sandbox_profile '{self.sandbox_profile}'. "
                f"Must be one of: {valid_profiles}"
            )
        valid_policies = ("env", "mounted_file")
        if self.auth_policy not in valid_policies:
            raise ValueError(
                f"Invalid auth_policy '{self.auth_policy}'. Must be one of: {valid_policies}"
            )


class DockerPiRunner:
    """Run Pi inside Docker container for sandboxing.

    Combines Docker isolation with Pi CLI simplicity.
    No cc-mirror variant creation overhead.
    """

    _MAX_OUTPUT_CHARS = 100_000  # Increased from 20k to avoid truncation
    _MAX_OUTPUT_LINE_CHARS = 2_000
    _MAX_ERROR_CHARS = 1000
    _REPEATED_READ_THRESHOLD = 10  # Increased threshold to be less aggressive
    _REPEATED_PATH_THRESHOLD = 6  # Increased threshold to be less aggressive

    def __init__(self, config: DockerPiRunnerConfig | None = None):
        self.config = config or DockerPiRunnerConfig()
        self._check_docker()

    def _check_docker(self) -> None:
        """Check that Docker is available and image exists."""
        result = subprocess.run(
            ["docker", "images", "-q", self.config.docker_image],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            raise RuntimeError(
                f"Docker image '{self.config.docker_image}' not found. "
                f"Run 'make docker-build' first."
            )

    @property
    def workspace_dir(self) -> Path:
        """Base directory for workspaces (uses temp directory)."""
        temp_dir = Path(tempfile.gettempdir()) / "trainr_workspaces"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def run_evaluation(
        self,
        skill_content: str | None,
        task_instruction: str,
        task_context: str,
        package_dir: Path,
        model: str | None = None,
        task: Any = None,
    ) -> dict[str, Any]:
        """Run evaluation in Docker container using Pi CLI batch mode."""
        import base64

        from config import get_llm_config

        start_time = time.time()
        model = model or self.config.model

        # Resolve model from llm.yaml if needed
        llm_config = get_llm_config()
        provider, pi_model = _resolve_pi_model(llm_config, model)

        # Extract repo info from task for cloning
        source = getattr(task, "source", {}) or {}
        if isinstance(source, dict):
            repo = source.get("repo", "")
            base_commit = source.get("base_commit", "")
            source_type = source.get("type", "")
        else:
            repo = getattr(source, "repo", "")
            base_commit = getattr(source, "base_commit", "")
            source_type = getattr(source, "type", "")

        # Check for Kaggle kernel tasks (notebooks/scripts, not R packages)
        is_kaggle_task = source_type == "kaggle_kernel"

        # Check for synthetic task (has source_package but no repo)
        source_package = getattr(task, "source_package", "") or ""

        if is_kaggle_task:
            # Kaggle task - no repo, uses reference solution code
            repo = ""
            base_commit = ""
            test_patch = ""
            gold_patch = ""
            fail_to_pass = []
            pass_to_pass = []
            # Extract reference solution code for Kaggle tasks
            ref_solution = {}
            if hasattr(task, "reference_solution"):
                ref_solution = task.reference_solution or {}
            elif isinstance(getattr(task, "__dict__", {}), dict):
                ref_solution = task.__dict__.get("reference_solution", {})
            kaggle_code = ref_solution.get("code", "") if isinstance(ref_solution, dict) else ""
        elif not repo and source_package:
            # Synthetic task - look up repo from package name
            from config import get_package_repo

            repo = get_package_repo(source_package) or ""
            if not repo:
                raise ValueError(f"No repo mapping for package: {source_package}")
            # Synthetic tasks don't have these fields
            base_commit = ""
            test_patch = ""
            gold_patch = ""
            fail_to_pass = []
            pass_to_pass = []
            kaggle_code = ""
        else:
            # Mined task - use git repo with patches
            test_patch = getattr(task, "test_patch", "") or ""
            gold_patch = getattr(task, "patch", "") or ""
            tests = getattr(task, "tests", {}) or {}
            fail_to_pass = tests.get("fail_to_pass", []) if isinstance(tests, dict) else []
            pass_to_pass = tests.get("pass_to_pass", []) if isinstance(tests, dict) else []
            kaggle_code = ""

        skill_b64 = base64.b64encode(skill_content.encode()).decode() if skill_content else ""

        # Build the prompt with explicit instructions based on task type
        if is_kaggle_task:
            # Kaggle task: agent works on a solution script
            prompt_parts = []
            if skill_content and skill_content.strip():
                prompt_parts.append(f"# Skill Context\n\n{skill_content}\n\n---")
            prompt_parts.append(
                f"""# Kaggle Competition Task

{task_instruction}

## Context

{task_context}

## Important Instructions

1. Your solution should be written to: `notebook/solution.R`
2. You can reference `notebook/reference_solution.R` for approach ideas
3. Make sure your solution handles the data preprocessing and model training
4. After writing the solution, test it with: `Rscript notebook/solution.R`
5. If there are errors, fix them and re-run until it works

Write the solution now."""
            )
            prompt = "\n\n".join(prompt_parts)
        else:
            # R package task: agent writes tests
            prompt_parts = []
            if skill_content and skill_content.strip():
                prompt_parts.append(f"# Skill Context\n\n{skill_content}\n\n---")
            prompt_parts.append(
                f"""# Task

{task_instruction}

## Context

The function to test:

```
{task_context}
```

## Important Instructions

1. Create your test file at: `tests/testthat/test-generated.R`
2. DO NOT modify any existing R source files in the R/ directory
3. DO NOT modify existing test files
4. ONLY write to tests/testthat/test-generated.R
5. After writing the test, run it with: `testthat::test_file('tests/testthat/test-generated.R')`
6. If tests fail, fix them and re-run until they pass

Write the tests now."""
            )
            prompt = "\n\n".join(prompt_parts)

        # Build docker command - container runs everything in batch mode
        env_vars = {
            "PI_MODEL": pi_model,
            "SKILL_B64": skill_b64,
            "PROMPT_B64": base64.b64encode(prompt.encode()).decode(),
            "TASK_TYPE": "kaggle_kernel" if is_kaggle_task else "r_package",
        }

        # Add task-type-specific environment variables
        if is_kaggle_task:
            # Kaggle task: pass reference solution code
            if kaggle_code:
                env_vars["KAGGLE_CODE_B64"] = base64.b64encode(kaggle_code.encode()).decode()
            # Note: Kaggle data path can be set if needed in the future
        else:
            # R package task: pass repo and patch info
            env_vars["REPO"] = repo
            env_vars["BASE_COMMIT"] = base_commit
            env_vars["TEST_PATCH"] = (
                base64.b64encode(test_patch.encode()).decode() if test_patch else ""
            )
            env_vars["GOLD_PATCH"] = (
                base64.b64encode(gold_patch.encode()).decode() if gold_patch else ""
            )
            env_vars["FAIL_TO_PASS"] = json.dumps(fail_to_pass)
            env_vars["PASS_TO_PASS"] = json.dumps(pass_to_pass)

        # Add API keys based on provider
        # Use central resolver for API key lookup
        def _get_api_key_for_provider(provider_name: str) -> str | None:
            """Get API key for a provider using resolver with fallback."""
            try:
                key_name = _resolve_api_key_env_name(provider_name)
                return _resolve_credential_value(key_name, self.config.auth_policy)
            except (ImportError, KeyError):
                # Fallback to direct env var lookup
                fallback_keys = _provider_api_key_map()
                key_name = fallback_keys.get(provider_name)
                if key_name:
                    return _resolve_credential_value(key_name, self.config.auth_policy)
            return None

        if provider:
            api_key = _get_api_key_for_provider(provider)
            if api_key:
                try:
                    key_name = _resolve_api_key_env_name(provider)
                    env_vars[key_name] = api_key
                    for alias in _get_api_key_aliases().get(key_name, ()):
                        env_vars[alias] = api_key
                except (ImportError, KeyError):
                    # Fallback: use provider-specific env var names
                    fallback_key_names = {
                        "opencode": "OPENCODE_API_KEY",
                        "openrouter": "OPENROUTER_API_KEY",
                        "zai": "Z_AI_API_KEY",
                        "zai_coding_plan": "Z_AI_API_KEY",
                        "openai": "OPENAI_API_KEY",
                        "kimi": "KIMI_API_KEY",
                        "kimi_coding_plan": "KIMI_API_KEY",
                    }
                    if provider in fallback_key_names:
                        env_vars[fallback_key_names[provider]] = api_key
        elif pi_model.startswith("openrouter/"):
            # Fallback for models not in llm.yaml
            api_key = _resolve_credential_value("OPENROUTER_API_KEY", self.config.auth_policy)
            if api_key:
                env_vars["OPENROUTER_API_KEY"] = api_key
        elif pi_model.startswith("opencode/"):
            api_key = _resolve_credential_value("OPENCODE_API_KEY", self.config.auth_policy)
            if api_key:
                env_vars["OPENCODE_API_KEY"] = api_key
        elif pi_model.startswith("zai/"):
            api_key = _resolve_credential_value("Z_AI_API_KEY", self.config.auth_policy)
            if api_key:
                env_vars["Z_AI_API_KEY"] = api_key
                env_vars["ZAI_API_KEY"] = api_key
        elif pi_model.startswith("openai/"):
            # Fallback for models not in llm.yaml
            api_key = _resolve_credential_value("OPENAI_API_KEY", self.config.auth_policy)
            if api_key:
                env_vars["OPENAI_API_KEY"] = api_key
        elif pi_model.startswith("kimi/"):
            api_key = _resolve_credential_value("KIMI_API_KEY", self.config.auth_policy)
            if api_key:
                env_vars["KIMI_API_KEY"] = api_key

        if self.config.forward_github_token:
            github_token = _get_env_var("GITHUB_TOKEN") or _get_env_var("GITHUB_PAT")
            if github_token:
                env_vars["GITHUB_TOKEN"] = github_token

        # Create unique workspace directory for this run to avoid parallel conflicts
        safe_repo = repo.replace("/", "_").replace(":", "_") if repo else "unknown"
        safe_model_name = model.replace("/", "_").replace(":", "_")
        unique_workspace = Path(
            tempfile.mkdtemp(
                prefix=f"trainr_{safe_repo}_{safe_model_name}_{uuid.uuid4().hex[:8]}_",
                dir=tempfile.gettempdir(),
            )
        )

        # Build docker run command using sandbox policy
        from bench.sandbox import DockerCommandBuilder, SandboxPolicy, SandboxProfile

        profile = SandboxProfile(self.config.sandbox_profile)
        policy = SandboxPolicy.from_profile(profile)
        builder = DockerCommandBuilder(policy)

        # Prepare volumes as (source, destination, mode) tuples
        volumes = [
            (str(unique_workspace), "/workspace", "rw"),
        ]

        # Build the docker command
        # Note: container uses its default entrypoint, so command is empty
        docker_cmd = builder.build_run_command(
            image=self.config.docker_image,
            command=[],  # Container uses default entrypoint
            env_vars=env_vars,
            volumes=volumes,
        )

        console.print(f"[dim]Running Docker+Pi batch mode with model: {pi_model}[/dim]")
        console.print(f"[dim]Sandbox profile: {self.config.sandbox_profile}[/dim]")

        # Track success for cleanup decision
        success = False
        payload: dict[str, Any] = {}

        try:
            # Run container and wait for completion
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            execution_time = time.time() - start_time
            stdout = result.stdout
            stderr = result.stderr
            combined_output = stdout
            if stderr and stderr.strip():
                combined_output = f"{stdout}\n\n[STDERR]\n{stderr}"

            # Parse JSON events from stdout for token usage
            token_usage = self._parse_token_usage(stdout)
            agent_error = self._parse_agent_error(f"{stdout}\n{stderr}")

            # Extract tool call telemetry
            tool_call_counts = self._extract_tool_call_counts(stdout)
            tool_errors = self._extract_tool_errors(stdout)

            # Extract and save tool traces BEFORE loop detection for debugging
            if self.config.save_traces:
                task_id = getattr(task, "id", None) or getattr(task, "task_id", "unknown")
                self._extract_and_save_tool_traces(stdout, pi_model, task_id)

            read_loop_error = self._detect_repetitive_read_loop(stdout)
            if read_loop_error and self._looks_like_noisy_agent_error(agent_error):
                agent_error = read_loop_error
            agent_error = self._sanitize_agent_error(agent_error)
            combined_output = self._sanitize_output_for_artifacts(combined_output)

            # If Pi produced no usage events at all, treat it as an agent-level failure
            # instead of scoring against the repo's pre-existing test state.
            if (
                token_usage.get("total_tokens", 0) == 0
                and token_usage.get("turn_count", 0) == 0
                and not agent_error
            ):
                agent_error = "Agent produced no output (0 tokens generated)"

            # Parse testthat results from combined output
            test_result = self._parse_testthat_output(stdout + stderr)
            effective_agent_error = agent_error
            non_fatal_agent_warning = None
            if self._is_non_fatal_agent_error(
                agent_error,
                returncode=result.returncode,
                test_result=test_result,
            ):
                non_fatal_agent_warning = agent_error
                effective_agent_error = None

            # Build error message from container stderr and/or agent-level failures.
            error_msg = ""
            if result.returncode != 0:
                error_msg = (
                    stderr.strip()
                    if stderr.strip()
                    else f"Container exited with code {result.returncode}"
                )
            if effective_agent_error:
                if error_msg:
                    error_msg = f"{error_msg}\n\nAgent error: {effective_agent_error}"
                else:
                    error_msg = f"Agent error: {effective_agent_error}"

            payload = {
                "success": result.returncode == 0
                and not effective_agent_error
                and test_result.get("passed", False),
                "score": (
                    1.0
                    if (
                        result.returncode == 0
                        and not effective_agent_error
                        and test_result.get("passed", False)
                    )
                    else 0.0
                ),
                "output": combined_output,
                "error": error_msg,
                "test_results": test_result,
                "execution_time": execution_time,
                "model": pi_model,
                "token_usage": token_usage,
                "agent_error": agent_error,
                "tool_call_counts": tool_call_counts,
                "tool_errors": tool_errors,
                "tool_total_time_ms": {},  # Optional: can calculate from timestamps if needed
            }
            if non_fatal_agent_warning:
                payload["agent_warning"] = non_fatal_agent_warning
            success = payload["success"]
            return payload

        except subprocess.TimeoutExpired:
            payload = {
                "success": False,
                "score": 0.0,
                "output": "",
                "error": f"Evaluation timed out after {self.config.timeout}s",
                "test_results": {},
                "execution_time": self.config.timeout,
                "model": pi_model,
                "token_usage": {},
                "tool_call_counts": {},
                "tool_errors": {},
                "tool_total_time_ms": {},
            }
            success = False
            return payload

        except Exception as e:
            payload = {
                "success": False,
                "score": 0.0,
                "output": "",
                "error": str(e),
                "test_results": {},
                "execution_time": time.time() - start_time,
                "model": pi_model,
                "token_usage": {},
                "tool_call_counts": {},
                "tool_errors": {},
                "tool_total_time_ms": {},
            }
            success = False
            return payload

        finally:
            import logging

            logger = logging.getLogger(__name__)
            should_keep = (not success) and self.config.keep_workspace_on_failure
            if not should_keep:
                shutil.rmtree(unique_workspace, ignore_errors=True)
            else:
                logger.warning("Retaining failed workspace: %s", unique_workspace)

    def _parse_testthat_output(self, output: str) -> dict[str, Any]:
        """Parse testthat results from combined output."""
        import re

        # Look for testthat summary patterns
        pass_match = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
        fail_match = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)
        skip_match = re.search(r"(\d+)\s+skipped", output, re.IGNORECASE)
        warn_match = re.search(r"(\d+)\s+warning", output, re.IGNORECASE)

        num_passed = int(pass_match.group(1)) if pass_match else 0
        num_failed = int(fail_match.group(1)) if fail_match else 0
        num_skipped = int(skip_match.group(1)) if skip_match else 0
        num_warnings = int(warn_match.group(1)) if warn_match else 0

        # Also count test_that blocks if summary not found
        if num_passed == 0 and num_failed == 0:
            # Count test_that blocks as a proxy
            test_count = len(re.findall(r"test_that\s*\(", output))
            # Check for Error patterns which indicate failures
            error_count = len(re.findall(r"^──.*Error", output, re.MULTILINE))
            if test_count > 0 and error_count == 0:
                num_passed = test_count
                num_failed = 0
            elif error_count > 0:
                num_failed = error_count

        # Success if no failures and at least some tests ran
        passed = num_failed == 0 and num_passed > 0

        return {
            "passed": passed,
            "num_passed": num_passed,
            "num_failed": num_failed,
            "num_skipped": num_skipped,
            "num_warnings": num_warnings,
        }

    def _parse_token_usage(self, output: str) -> dict:
        """Parse token usage from Pi CLI JSON output.

        Pi CLI outputs JSON lines with 'turn_end' or 'message_end' events
        containing usage stats. Aggregate all token counts.
        """
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        turn_count = 0

        for line in output.split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                # Check for turn_end or message_end with usage
                if event_type in ("turn_end", "message_end"):
                    # Usage may be at top level (message_end) or nested in message (turn_end)
                    usage = event.get("usage") or event.get("message", {}).get("usage", {})

                    if usage:
                        total_input += usage.get("input", 0)
                        total_output += usage.get("output", 0)
                        total_cache_read += usage.get("cacheRead", 0)
                        total_cache_write += usage.get("cacheWrite", 0)
                        turn_count += 1
            except json.JSONDecodeError:
                continue

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "total_tokens": total_input + total_output,
            "turn_count": turn_count,
        }

    def _parse_agent_error(self, output: str) -> str | None:
        """Extract provider/agent error text from Pi CLI JSON events."""
        import re

        latest_error: str | None = None

        for line in output.split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Errors may be at top-level or nested in message payload.
            top_level_error = event.get("errorMessage")
            nested_error = event.get("message", {}).get("errorMessage")
            stop_reason = event.get("stopReason") or event.get("message", {}).get("stopReason")

            if isinstance(top_level_error, str) and top_level_error.strip():
                latest_error = top_level_error.strip()
            if isinstance(nested_error, str) and nested_error.strip():
                latest_error = nested_error.strip()

            if stop_reason == "error" and latest_error:
                return latest_error
        if latest_error:
            return latest_error

        # Fallback for non-JSON errors emitted directly to stderr.
        plain_patterns = [
            r"No API key found for [^\n.]+\.?",
            r"Error verifying OIDC token",
            r"(?:status|code|http|error|response)[^\n]*\b401\b[^\n]*",
            r"\b401\b[^\n]*(?:unauthorized|forbidden|authentication|auth)[^\n]*",
            r"(?:status|code|http|error|response)[^\n]*\b403\b[^\n]*",
            r"\b403\b[^\n]*(?:forbidden|denied|access|permission)[^\n]*",
            r"(?:status|code|http|error|response)[^\n]*\b429\b[^\n]*",
            r"\b429\b[^\n]*(?:rate|limit|throttl|too many|retry)[^\n]*",
            r"Insufficient balance[^\n]*",
            r"model\b[^\n]*not found[^\n]*",
            r"no resource package[^\n]*",
            r"User not found[^\n.]*\.?",
            r"Detected repetitive[^\n]*",
        ]
        for pattern in plain_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return None

    @classmethod
    def _truncate_text(cls, text: str, max_chars: int) -> str:
        """Truncate long text with a visible suffix."""
        if len(text) <= max_chars:
            return text
        suffix = "... [truncated]"
        head = max(0, max_chars - len(suffix))
        return text[:head].rstrip() + suffix

    @classmethod
    def _sanitize_output_for_artifacts(cls, output: str) -> str:
        """Clamp oversized container output before storing it in artifacts."""
        if not output:
            return output

        sanitized_lines = [
            cls._truncate_text(line, cls._MAX_OUTPUT_LINE_CHARS) for line in output.splitlines()
        ]
        sanitized = "\n".join(sanitized_lines)
        return cls._truncate_text(sanitized, cls._MAX_OUTPUT_CHARS)

    @classmethod
    def _looks_like_noisy_agent_error(cls, error_message: str | None) -> bool:
        """Detect malformed or oversized error payloads that need summarizing."""
        if not error_message:
            return True
        if len(error_message) > cls._MAX_ERROR_CHARS:
            return True
        lowered = error_message.lower()
        noisy_markers = (
            '"toolcall"',
            '"toolresults"',
            '"toolcallid"',
            '"thinking"',
            "[98]",
            "[100]",
            "let's get more context",
            "maybe there are more lines after what we saw",
        )
        return any(marker in lowered for marker in noisy_markers)

    @classmethod
    def _sanitize_agent_error(cls, error_message: str | None) -> str | None:
        """Normalize noisy agent errors into short, parseable messages."""
        if not error_message:
            return error_message

        compact = " ".join(line.strip() for line in error_message.splitlines() if line.strip())
        compact = re.sub(r"\s+", " ", compact).strip()
        lowered = compact.lower()

        if '"toolname":"read"' in lowered or '"name":"read"' in lowered:
            return "Detected repetitive file-read loop in agent output"
        if '"toolresults"' in lowered or '"toolcallid"' in lowered or '"thinking"' in lowered:
            return "Agent emitted malformed tool payload in error stream"
        if re.search(r"\[\d+\]\s+\"", compact):
            return "Agent emitted oversized structured error payload"

        return cls._truncate_text(compact, cls._MAX_ERROR_CHARS)

    @staticmethod
    def _is_non_fatal_agent_error(
        error_message: str | None,
        *,
        returncode: int,
        test_result: dict[str, Any],
    ) -> bool:
        """Allow known end-of-stream artifacts when execution otherwise succeeded."""
        if error_message != "Agent emitted malformed tool payload in error stream":
            return False
        if returncode != 0:
            return False
        return bool(test_result.get("passed", False))

    @classmethod
    def _detect_repetitive_read_loop(cls, output: str) -> str | None:
        """Detect obvious read-tool loops that inflate context without progress."""
        if not output:
            return None

        unique_reads: dict[str, str | None] = {}
        anonymous_reads: list[str | None] = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            cls._record_read_tool_call(
                unique_reads,
                anonymous_reads,
                tool_name=event.get("toolName"),
                tool_call_id=event.get("toolCallId"),
                payload=event,
            )

            message = event.get("message")
            if isinstance(message, dict):
                for item in message.get("content", []):
                    if not isinstance(item, dict):
                        continue
                    cls._record_read_tool_call(
                        unique_reads,
                        anonymous_reads,
                        tool_name=item.get("name"),
                        tool_call_id=item.get("id"),
                        payload=item,
                    )

        read_calls = len(unique_reads) + len(anonymous_reads)
        if read_calls < cls._REPEATED_READ_THRESHOLD:
            return None

        paths = [path for path in unique_reads.values() if path]
        paths.extend(path for path in anonymous_reads if path)

        if not paths:
            # If we have many read calls but no paths (unlikely), it might be a malformed loop
            return "Detected excessive file-read activity"

        path_counts = Counter(paths)
        path, count = path_counts.most_common(1)[0]

        # FIX: Only return error if threshold is MET
        if count < cls._REPEATED_PATH_THRESHOLD:
            return None

        display_path = cls._truncate_text(path, 120)
        return f"Detected repetitive file-read loop on {display_path}"

    @classmethod
    def _record_read_tool_call(
        cls,
        unique_reads: dict[str, str | None],
        anonymous_reads: list[str | None],
        *,
        tool_name: Any,
        tool_call_id: Any,
        payload: dict[str, Any],
    ) -> None:
        """Record one read-tool invocation, deduplicated by toolCallId when present."""
        if tool_name != "read":
            return

        path = cls._extract_read_path(payload)
        normalized_id = str(tool_call_id).strip() if isinstance(tool_call_id, str) else ""

        if normalized_id:
            if normalized_id not in unique_reads or (
                unique_reads[normalized_id] is None and path is not None
            ):
                unique_reads[normalized_id] = path
            return

        anonymous_reads.append(path)

    @staticmethod
    def _extract_read_path(payload: dict[str, Any]) -> str | None:
        """Extract a read-tool path from event payload shapes."""
        for key in ("args", "arguments"):
            value = payload.get(key)
            if isinstance(value, dict):
                path = value.get("path")
                if isinstance(path, str) and path.strip():
                    return path.strip()
        return None

    def _extract_and_save_tool_traces(
        self,
        output: str,
        model: str,
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Extract tool call traces from stdout and save to JSONL file.

        Parses JSON events from container stdout, extracts tool execution events,
        and saves them to container_logs/{model}/{task_id}_trace.jsonl for debugging.

        Args:
            output: Raw stdout from container containing JSON events
            model: Model identifier for log directory naming
            task_id: Task identifier for trace filename

        Returns:
            List of extracted tool call trace entries
        """
        traces: list[dict[str, Any]] = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract tool execution events at top level
            trace_entry = self._build_trace_entry(event)
            if trace_entry:
                traces.append(trace_entry)

            # Also check nested message content for tool calls
            message = event.get("message")
            if isinstance(message, dict):
                for item in message.get("content", []):
                    if not isinstance(item, dict):
                        continue
                    nested_trace = self._build_trace_entry(item)
                    if nested_trace:
                        traces.append(nested_trace)

        # Save traces to file
        if traces:
            self._save_traces_to_file(traces, model, task_id)

        return traces

    @classmethod
    def _extract_tool_call_counts(cls, output: str) -> dict[str, int]:
        """Extract tool call counts from stdout.

        Parses stdout for tool events from the extension and counts calls per tool name.
        Looks for lines matching: {"type":"tool_call","toolName":"...",...} or
        {"type":"tool_execution_start","toolName":"...",...}

        Args:
            output: Raw stdout from container containing JSON events

        Returns:
            Dict mapping tool names to call counts, e.g. {"read": 5, "bash": 3, "edit": 2}
        """
        counts: Counter = Counter()
        seen_call_ids: set[str] = set()

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Count tool calls at top level
            cls._count_tool_call(event, counts, seen_call_ids)

            # Also check nested message content for tool calls
            message = event.get("message")
            if isinstance(message, dict):
                for item in message.get("content", []):
                    if not isinstance(item, dict):
                        continue
                    cls._count_tool_call(item, counts, seen_call_ids)

        return dict(counts)

    @classmethod
    def _count_tool_call(
        cls,
        event: dict[str, Any],
        counts: Counter,
        seen_call_ids: set[str],
    ) -> None:
        """Count a tool call from an event, deduplicating by toolCallId."""
        event_type = event.get("type", "")

        # Only count tool call start events, not results
        if event_type not in ("tool_call", "tool_execution_start"):
            # Also check for tool calls in message content blocks (name field)
            tool_name = event.get("name")
            if tool_name and event.get("id"):
                # This is a tool call in content block format
                tool_call_id = event.get("id", "")
                if tool_call_id and tool_call_id in seen_call_ids:
                    return
                if tool_call_id:
                    seen_call_ids.add(tool_call_id)
                counts[tool_name] += 1
            return

        tool_name = event.get("toolName")
        if not tool_name:
            return

        # Deduplicate by toolCallId to avoid counting same call multiple times
        tool_call_id = event.get("toolCallId", "")
        if tool_call_id and tool_call_id in seen_call_ids:
            return
        if tool_call_id:
            seen_call_ids.add(tool_call_id)

        counts[tool_name] += 1

    @classmethod
    def _extract_tool_errors(cls, output: str) -> dict[str, int]:
        """Extract tool execution error counts from stdout."""
        error_counts: Counter = Counter()

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "tool_execution_end" and event.get("isError"):
                tool_name = event.get("toolName")
                if tool_name:
                    error_counts[tool_name] += 1

        return dict(error_counts)

    @classmethod
    def _build_trace_entry(cls, event: dict[str, Any]) -> dict[str, Any] | None:
        """Build a trace entry from a tool execution event.

        Looks for tool execution events with toolName/toolCallId fields.
        Returns None if not a tool execution event.
        """
        # Check for tool execution at top level (tool_execution_start, etc.)
        tool_name = event.get("toolName") or event.get("name")
        tool_call_id = event.get("toolCallId") or event.get("id")

        if not tool_name:
            return None

        # Extract args/arguments
        args = event.get("args") or event.get("arguments") or {}

        return {
            "timestamp": event.get("timestamp")
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": event.get("type", "tool_execution_start"),
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args if isinstance(args, dict) else {},
        }

    def _save_traces_to_file(
        self,
        traces: list[dict[str, Any]],
        model: str,
        task_id: str,
    ) -> None:
        """Save tool traces to a JSONL file.

        Saves to container_logs/{model}/{task_id}_trace.jsonl (or configured trace_dir)
        """
        # Sanitize model name for filesystem
        safe_model = model.replace("/", "_").replace(":", "_")
        safe_task_id = task_id.replace("/", "_").replace(":", "_")

        log_dir = Path(self.config.trace_dir) / safe_model
        log_dir.mkdir(parents=True, exist_ok=True)

        trace_file = log_dir / f"{safe_task_id}_trace.jsonl"

        try:
            with open(trace_file, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace) + "\n")
            console.print(f"[dim]Saved {len(traces)} tool traces to {trace_file}[/dim]")
        except OSError as e:
            console.print(f"[yellow]Warning: Failed to save traces: {e}[/yellow]")
