"""Pi SDK-based evaluation harness for R package testing tasks.

Replaces cc-mirror + Docker with direct Pi CLI calls.
Simpler, faster, and supports free models via OpenRouter.
"""

import contextlib
import json
import os
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


def _get_all_provider_api_keys() -> list[str]:
    """Get list of all known provider API key environment variables.

    Uses the central resolver if available, falls back to hardcoded list.
    """
    try:
        from bench.provider.resolver import PROVIDER_API_KEY_MAP, get_provider_resolver

        resolver = get_provider_resolver()
        keys = set()
        # Add keys from resolver
        for provider_name in resolver.providers:
            try:
                keys.add(resolver.get_api_key_env(provider_name))
            except KeyError:
                pass
        # Also include canonical keys
        keys.update(PROVIDER_API_KEY_MAP.values())
        return list(keys)
    except (ImportError, FileNotFoundError):
        pass

    # Fallback to hardcoded list (deprecated)
    warnings.warn(
        "Using fallback API key list in pi_runner. Prefer bench.provider for provider resolution.",
        DeprecationWarning,
        stacklevel=3,
    )
    return [
        "OPENROUTER_API_KEY",
        "Z_AI_API_KEY",
        "KIMI_API_KEY",
        "GEMINI_API_KEY",
        "OPENCODE_API_KEY",
    ]


@dataclass
class DockerPiRunnerConfig:
    """Configuration for Docker-based Pi runner."""

    model: str = "openrouter/openai/gpt-oss-120b:free"
    max_turns: int = 50
    timeout: int = 600
    docker_image: str = "posit-gskill-eval:latest"
    # API keys to pass to container (only if set in environment)
    api_keys: list[str] | None = None

    def __post_init__(self):
        if self.api_keys is None:
            # Default: forward all known API keys if set
            self.api_keys = _get_all_provider_api_keys()


class DockerPiRunner:
    """Run Pi inside Docker container for sandboxing.

    Combines Docker isolation with Pi CLI simplicity.
    No cc-mirror variant creation overhead.
    """

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
        model_cfg = {}  # Initialize to avoid unbound issues
        provider = ""  # Initialize to avoid unbound issues
        try:
            model_cfg = llm_config.get_model_config(model)
            provider = model_cfg.get("provider", "")
            model_id = model_cfg.get("id", model)
            # Pi CLI supports: openrouter/, openai/, zai/, opencode/ (via OPENCODE_API_KEY)
            if provider == "opencode":
                pi_model = f"opencode/{model_id}"
            elif provider == "openrouter":
                pi_model = f"openrouter/{model_id}"
            else:
                pi_model = model_id
        except ValueError:
            pi_model = model

        # Extract repo info from task for cloning
        source = getattr(task, "source", {}) or {}
        if isinstance(source, dict):
            repo = source.get("repo", "")
            base_commit = source.get("base_commit", "")
        else:
            repo = getattr(source, "repo", "")
            base_commit = getattr(source, "base_commit", "")

        # Check for synthetic task (has source_package but no repo)
        source_package = getattr(task, "source_package", "") or ""

        if not repo and source_package:
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
        else:
            # Mined task - use git repo with patches
            test_patch = getattr(task, "test_patch", "") or ""
            gold_patch = getattr(task, "patch", "") or ""
            tests = getattr(task, "tests", {}) or {}
            fail_to_pass = tests.get("fail_to_pass", []) if isinstance(tests, dict) else []
            pass_to_pass = tests.get("pass_to_pass", []) if isinstance(tests, dict) else []

        skill_b64 = base64.b64encode(skill_content.encode()).decode() if skill_content else ""

        # Build the prompt with explicit instructions
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
            "REPO": repo,
            "BASE_COMMIT": base_commit,
            "TEST_PATCH": base64.b64encode(test_patch.encode()).decode() if test_patch else "",
            "GOLD_PATCH": base64.b64encode(gold_patch.encode()).decode() if gold_patch else "",
            "FAIL_TO_PASS": json.dumps(fail_to_pass),
            "PASS_TO_PASS": json.dumps(pass_to_pass),
        }

        # Add API keys based on provider
        # Use central resolver for API key lookup
        def _get_api_key_for_provider(provider_name: str) -> str | None:
            """Get API key for a provider using resolver with fallback."""
            try:
                from bench.provider import resolve_api_key_env

                key_name = resolve_api_key_env(provider_name)
                return os.environ.get(key_name)
            except (ImportError, KeyError):
                # Fallback to direct env var lookup
                fallback_keys = {
                    "opencode": "OPENCODE_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "zai": "Z_AI_API_KEY",
                    "gemini": "GEMINI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                }
                key_name = fallback_keys.get(provider_name)
                if key_name:
                    return os.environ.get(key_name)
            return None

        if provider:
            api_key = _get_api_key_for_provider(provider)
            if api_key:
                try:
                    from bench.provider import resolve_api_key_env

                    key_name = resolve_api_key_env(provider)
                    env_vars[key_name] = api_key
                except (ImportError, KeyError):
                    # Fallback: use provider-specific env var names
                    fallback_key_names = {
                        "opencode": "OPENCODE_API_KEY",
                        "openrouter": "OPENROUTER_API_KEY",
                        "openai": "OPENAI_API_KEY",
                    }
                    if provider in fallback_key_names:
                        env_vars[fallback_key_names[provider]] = api_key
        elif pi_model.startswith("openrouter/"):
            # Fallback for models not in llm.yaml
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                env_vars["OPENROUTER_API_KEY"] = api_key
        elif pi_model.startswith("openai/"):
            # Fallback for models not in llm.yaml
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                env_vars["OPENAI_API_KEY"] = api_key

        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        if github_token:
            env_vars["GITHUB_TOKEN"] = github_token

        # Create unique workspace directory for this run to avoid parallel conflicts
        task_name = repo.replace("/", "_") if repo else "unknown"
        safe_model_name = model.replace("/", "_").replace(":", "_")
        unique_workspace = self.workspace_dir / f"{task_name}_{safe_model_name}"
        unique_workspace.mkdir(parents=True, exist_ok=True)

        # Build docker run command (no -i needed for batch mode)
        docker_cmd = ["docker", "run", "--rm"]
        for key, value in env_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.extend(
            [
                "-v",
                f"{unique_workspace}:/workspace",
            ]
        )

        docker_cmd.append(self.config.docker_image)

        console.print(f"[dim]Running Docker+Pi batch mode with model: {model}[/dim]")

        import shutil

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

            # Clean up unique workspace directory to save disk space
            try:
                if unique_workspace.exists():
                    shutil.rmtree(unique_workspace)
            except Exception:
                # Don't fail the run if cleanup fails
                pass

            # Parse JSON events from stdout for token usage
            token_usage = self._parse_token_usage(stdout)

            # Parse testthat results from combined output
            test_result = self._parse_testthat_output(stdout + stderr)

            # Build error message from stderr if container failed
            error_msg = ""
            if result.returncode != 0:
                error_msg = (
                    stderr.strip()
                    if stderr.strip()
                    else f"Container exited with code {result.returncode}"
                )

            return {
                "success": test_result.get("passed", False),
                "score": 1.0 if test_result.get("passed", False) else 0.0,
                "output": stdout,
                "error": error_msg,
                "test_results": test_result,
                "execution_time": execution_time,
                "model": model,
                "token_usage": token_usage,
            }

        except subprocess.TimeoutExpired:
            # Clean up workspace on timeout
            try:
                if unique_workspace.exists():
                    shutil.rmtree(unique_workspace)
            except Exception:
                pass
            return {
                "success": False,
                "score": 0.0,
                "output": "",
                "error": f"Evaluation timed out after {self.config.timeout}s",
                "test_results": {},
                "execution_time": self.config.timeout,
                "model": model,
                "token_usage": {},
            }
        except Exception as e:
            # Clean up workspace on error
            try:
                if unique_workspace.exists():
                    shutil.rmtree(unique_workspace)
            except Exception:
                pass
            return {
                "success": False,
                "score": 0.0,
                "output": "",
                "error": str(e),
                "test_results": {},
                "execution_time": time.time() - start_time,
                "model": model,
                "token_usage": {},
            }

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
