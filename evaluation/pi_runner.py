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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


@dataclass
class PiRunnerConfig:
    """Configuration for Pi runner."""

    # Model to use (e.g., "openrouter/google/gemini-2.0-flash", "zai/glm-4.5")
    model: str = "openrouter/openai/gpt-oss-120b:free"
    # Max turns for the agent
    max_turns: int = 50
    # Timeout in seconds
    timeout: int = 600
    # Working directory for the agent
    cwd: str | None = None
    # Path to Pi binary (default: check local node_modules, then PATH)
    pi_binary: str = ""

    def __post_init__(self):
        if not self.pi_binary:
            # Check for local bun/npm install
            import shutil

            local_pi = Path("node_modules/.bin/pi")
            if local_pi.exists():
                self.pi_binary = str(local_pi.resolve())  # Use absolute path
            elif pi_path := shutil.which("pi"):
                self.pi_binary = pi_path  # Use full path
            else:
                self.pi_binary = "bun"  # Will use 'bun run pi'


@dataclass
class PiEvaluationResult:
    """Result from Pi evaluation."""

    success: bool
    score: float
    output: str
    error: str | None = None
    test_results: dict[str, Any] | None = None
    execution_time: float = 0.0
    model: str = ""
    turns_used: int = 0


class PiRunner:
    """Run evaluations using Pi CLI instead of cc-mirror + Docker."""

    def __init__(self, config: PiRunnerConfig | None = None):
        self.config = config or PiRunnerConfig()
        self._check_pi_installed()

    def _check_pi_installed(self) -> None:
        """Check that Pi CLI is installed."""
        # Handle 'bun' case - we'll use 'bun x pi'
        if self.config.pi_binary == "bun":
            self._use_bun = True
            return

        self._use_bun = False
        try:
            result = subprocess.run(
                [self.config.pi_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Pi CLI not working: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Pi CLI not found at '{self.config.pi_binary}'. "
                "Install with: bun add @mariozechner/pi-coding-agent"
            ) from None

    def _build_prompt(self, skill_content: str, task_instruction: str, task_context: str) -> str:
        """Build the full prompt for Pi agent."""
        parts = []

        # Add skill as context (if provided)
        if skill_content and skill_content.strip():
            parts.append(f"""# Skill Context

You are following this skill guide:

{skill_content}

---""")

        # Add the task with explicit instructions
        parts.append(f"""# Task

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

Write the tests now.
""")

        return "\n\n".join(parts)

    def run_evaluation(
        self,
        skill_content: str | None,
        task_instruction: str,
        task_context: str,
        package_dir: Path,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run a single evaluation using Pi CLI.

        Args:
            skill_content: The skill markdown content (or empty string for no-skill)
            task_instruction: What the agent should do
            task_context: The code/function context for the task
            package_dir: Path to the R package
            model: Model override (uses config default if None)

        Returns:
            Dict with success, score, output, error, etc.
        """
        start_time = time.time()
        model = model or self.config.model
        cwd = Path(package_dir).resolve()

        # Build the prompt
        prompt = self._build_prompt("", task_instruction, task_context)

        skill_file = None
        try:
            # Write skill to temp file if provided
            if skill_content and skill_content.strip():
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                    f.write(skill_content)
                    skill_file = f.name

            # Build Pi command
            if getattr(self, "_use_bun", False):
                cmd = ["bun", "run", "pi"]
            else:
                cmd = [self.config.pi_binary]

            cmd.extend(
                [
                    "--print",  # Non-interactive mode
                    "--mode",
                    "json",  # JSON output
                    "--model",
                    model,
                    "--no-session",  # Don't save session
                ]
            )

            # Add skill file if provided
            if skill_file:
                cmd.extend(["--skill", skill_file])

            # Add the prompt as positional argument
            cmd.append(prompt)

            console.print(f"[dim]Running Pi with model: {model}[/dim]")
            console.print(f"[dim]CWD: {cwd}[/dim]")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=str(cwd),  # Pass cwd to subprocess.run instead
                env={**os.environ, "NO_COLOR": "1"},
            )

            execution_time = time.time() - start_time
            output = result.stdout
            error_output = result.stderr

            # Parse JSON output from Pi
            agent_output = ""
            turns_used = 0

            if output.strip():
                try:
                    # Pi outputs JSONL (one JSON object per line)
                    for line in output.strip().split("\n"):
                        if line.strip():
                            event = json.loads(line)
                            if event.get("type") == "message_update":
                                msg_event = event.get("assistantMessageEvent", {})
                                if msg_event.get("type") == "text_delta":
                                    agent_output += msg_event.get("delta", "")
                            elif event.get("type") == "agent_end":
                                turns_used = event.get("turns", 0)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use raw output
                    agent_output = output

            # Run testthat to verify
            test_result = self._run_testthat(cwd)

            success = test_result.get("passed", False)
            score = 1.0 if success else 0.0

            # Parse token usage from Pi CLI JSON output
            token_usage = self._parse_token_usage(output)

            return {
                "success": success,
                "score": score,
                "output": agent_output,
                "error": error_output if result.returncode != 0 else None,
                "test_results": test_result,
                "execution_time": execution_time,
                "model": model,
                "turns_used": turns_used,
                "token_usage": token_usage,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "score": 0.0,
                "output": "",
                "error": f"Evaluation timed out after {self.config.timeout}s",
                "execution_time": self.config.timeout,
                "model": model,
                "turns_used": 0,
                "token_usage": {},
            }
        finally:
            # Cleanup temp files
            if skill_file:
                with contextlib.suppress(OSError):
                    os.unlink(skill_file)

    def _run_testthat(self, package_dir: Path) -> dict[str, Any]:
        """Run testthat tests in the package directory."""
        try:
            result = subprocess.run(
                ["Rscript", "-e", "testthat::test_dir('tests/testthat', reporter = 'summary')"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(package_dir),
            )

            output = result.stdout + result.stderr

            # Parse testthat output
            import re

            pass_match = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
            fail_match = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)

            num_passed = int(pass_match.group(1)) if pass_match else 0
            num_failed = int(fail_match.group(1)) if fail_match else 0

            return {
                "passed": num_failed == 0 and num_passed > 0,
                "num_passed": num_passed,
                "num_failed": num_failed,
                "output": output,
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "output": "testthat timed out",
            }
        except Exception as e:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "output": str(e),
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
            self.api_keys = [
                "OPENROUTER_API_KEY",
                "ZAI_API_KEY",
                "KIMI_API_KEY",
                "GEMINI_API_KEY",
                "OPENCODE_API_KEY",
            ]


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
        """Get the workspace directory for Docker mounts."""
        return Path.cwd()

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

        # Get patches and tests
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
        if provider == "opencode":
            # opencode.ai uses OPENCODE_API_KEY
            api_key = os.environ.get("OPENCODE_API_KEY")
            if api_key:
                env_vars["OPENCODE_API_KEY"] = api_key
            # Pi handles opencode base URL internally
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                env_vars["OPENROUTER_API_KEY"] = api_key
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                env_vars["OPENAI_API_KEY"] = api_key
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
                self.config.docker_image,
            ]
        )

        console.print(f"[dim]Running Docker+Pi batch mode with model: {model}[/dim]")

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
