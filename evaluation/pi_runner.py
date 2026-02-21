"""Pi SDK-based evaluation harness for R package testing tasks.

Replaces cc-mirror + Docker with direct Pi CLI calls.
Simpler, faster, and supports free models via OpenRouter.
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class PiRunnerConfig:
    """Configuration for Pi runner."""

    # Model to use (e.g., "openrouter/google/gemini-2.0-flash", "zai/glm-4.5")
    model: str = "openrouter/google/gemini-2.0-flash"
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
            )

    def _build_prompt(self, skill_content: str, task_instruction: str, task_context: str) -> str:
        """Build the full prompt for Pi agent."""
        parts = []

        # Add skill as context (if provided)
        if skill_content and skill_content.strip():
            parts.append(f"""# Skill Context

You are following this skill guide:

{skill_content}

---""")

        # Add the task
        parts.append(f"""# Task

{task_instruction}

## Context

The function to test:

```
{task_context}
```

Write the tests in the appropriate test file under tests/testthat/. Run the tests to verify they pass.
""")

        return "\n\n".join(parts)

    def run_evaluation(
        self,
        skill_content: str,
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

            return {
                "success": success,
                "score": score,
                "output": agent_output,
                "error": error_output if result.returncode != 0 else None,
                "test_results": test_result,
                "execution_time": execution_time,
                "model": model,
                "turns_used": turns_used,
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
            }
        finally:
            # Cleanup temp files
            if skill_file:
                try:
                    os.unlink(skill_file)
                except OSError:
                    pass

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
            # Look for patterns like "X passed", "Y failed"
            passed = "passed" in output.lower() and "failed" not in output.lower()

            # More nuanced parsing
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


@dataclass
class DockerPiRunnerConfig:
    """Configuration for Docker-based Pi runner."""

    model: str = "openrouter/openai/gpt-oss-20b:free"
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

    def run_evaluation(
        self,
        skill_content: str,
        task_instruction: str,
        task_context: str,
        package_dir: Path,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run evaluation inside Docker container.

        Args:
            skill_content: Skill markdown (or empty for no-skill)
            task_instruction: What the agent should do
            task_context: Code/function context
            package_dir: Path to R package
            model: Model override

        Returns:
            Dict with success, score, output, error, etc.
        """
        import base64

        start_time = time.time()
        model = model or self.config.model
        cwd = Path(package_dir).resolve()

        # Build the prompt
        prompt_parts = []
        if skill_content and skill_content.strip():
            prompt_parts.append(f"# Skill Context\n\n{skill_content}\n\n---")
        prompt_parts.append(
            f"# Task\n\n{task_instruction}\n\n## Context\n\n```\n{task_context}\n```"
        )
        prompt = "\n\n".join(prompt_parts)

        # Encode skill for passing to container
        skill_b64 = base64.b64encode(skill_content.encode()).decode()
        prompt_b64 = base64.b64encode(prompt.encode()).decode()

        # Build Docker command
        # Use --entrypoint to bypass default entrypoint.sh
        cmd = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",  # Bypass entrypoint.sh
            "-v",
            f"{cwd}:/workspace",
            "-e",
            "NO_COLOR=1",
            "-e",
            f"SKILL_B64={skill_b64}",
            "-e",
            f"PROMPT_B64={prompt_b64}",
            "-e",
            f"PI_MODEL={model}",
        ]

        # Add API keys
        for key in self.config.api_keys or []:
            if os.environ.get(key):
                cmd.extend(["-e", key])

        cmd.append(self.config.docker_image)

        # The container entrypoint should run Pi
        # We'll pass a command that decodes and runs Pi
        cmd.extend(
            [
                "bash",
                "-c",
                """
            set -e
            echo "[docker-pi] Starting evaluation with model: $PI_MODEL"
            
            # Decode prompt
            PROMPT=$(echo "$PROMPT_B64" | base64 -d)
            
            # Write skill to temp file if provided
            if [ -n "$SKILL_B64" ]; then
                SKILL_FILE=/tmp/skill.md
                echo "$SKILL_B64" | base64 -d > "$SKILL_FILE"
                SKILL_FLAG="--skill $SKILL_FILE"
            else
                SKILL_FLAG=""
            fi
            
            # Run Pi
            cd /workspace
            pi --print --mode json --no-session --model "$PI_MODEL" $SKILL_FLAG "$PROMPT" 2>&1 || true
            
            echo ""
            echo "[docker-pi] Running testthat..."
            Rscript -e "testthat::test_dir('tests/testthat', reporter = 'summary')" 2>&1 || true
            
            echo "[docker-pi] Done"
            """,
            ]
        )

        console.print(f"[dim]Running Docker+Pi with model: {model}[/dim]")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            execution_time = time.time() - start_time
            output = result.stdout
            error_output = result.stderr if result.returncode != 0 else None

            # Parse testthat results from output
            test_result = self._parse_testthat_output(output)

            return {
                "success": test_result.get("passed", False),
                "score": 1.0 if test_result.get("passed", False) else 0.0,
                "output": output,
                "error": error_output,
                "test_results": test_result,
                "execution_time": execution_time,
                "model": model,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "score": 0.0,
                "output": "",
                "error": f"Evaluation timed out after {self.config.timeout}s",
                "execution_time": self.config.timeout,
                "model": model,
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

        # Success if no failures and at least some tests ran
        passed = num_failed == 0 and num_passed > 0

        return {
            "passed": passed,
            "num_passed": num_passed,
            "num_failed": num_failed,
            "num_skipped": num_skipped,
            "num_warnings": num_warnings,
        }
