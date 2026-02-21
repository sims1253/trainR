"""Run evaluations in Docker container with cc-mirror agent."""

import base64
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TestRunnerConfig:
    """Configuration for Docker test runner."""

    docker_image: str = "posit-gskill-eval:latest"
    timeout: int = 600  # 10 minutes for agentic evaluation
    package_path: str = "/workspace/packages/cli"


@dataclass
class EvaluationJob:
    """Single evaluation job for batch processing."""

    task_id: str
    skill_content: str
    task_instruction: str
    task_context: str
    package_dir: Path


class DockerTestRunner:
    """Run evaluations in Docker with cc-mirror agent."""

    def __init__(self, config: TestRunnerConfig | None = None) -> None:
        self.config = config or TestRunnerConfig()

    def run_evaluation(
        self,
        skill_content: str,
        task_instruction: str,
        task_context: str,
        package_dir: Path,
        custom_prompt: str | None = None,
        model: str = "glm-4.5",
    ) -> dict[str, Any]:
        """Run a full evaluation cycle in Docker.

        Args:
            skill_content: The SKILL.md content
            task_instruction: The task instruction
            task_context: Additional context (function code, etc.)
            package_dir: Path to the R package
            custom_prompt: Optional custom prompt for the agent
            model: LLM model to use for evaluation

        Returns:
            Dict with evaluation results
        """
        # Convert to absolute path for Docker volume mount
        package_dir = package_dir.resolve()

        if not package_dir.exists():
            logger.error(f"Package directory does not exist: {package_dir}")
            return {
                "success": False,
                "error": f"Package directory does not exist: {package_dir}",
                "tests_passed": 0,
                "tests_failed": 0,
            }

        # Combine instruction and context
        full_task = f"# Task\n\n{task_instruction}\n\n# Context\n\n{task_context}"

        # Base64 encode to handle special characters
        skill_b64 = base64.b64encode(skill_content.encode()).decode()
        task_b64 = base64.b64encode(full_task.encode()).decode()

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{package_dir}:/workspace/packages/{package_dir.name}",
            "-e",
            f"PACKAGE_PATH=/workspace/packages/{package_dir.name}",
            "-e",
            "Z_AI_API_KEY",
            "-e",
            f"SKILL_CONTENT={skill_b64}",
            "-e",
            f"TASK_CONTENT={task_b64}",
            "-e",
            f"LLM_MODEL={model}",
        ]

        if custom_prompt:
            # Base64 encode custom prompt
            prompt_b64 = base64.b64encode(custom_prompt.encode()).decode()
            cmd.extend(["-e", f"CUSTOM_PROMPT={prompt_b64}"])

        cmd.append(self.config.docker_image)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            # Log output for debugging
            if result.stdout:
                logger.debug(f"Docker stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"Docker stderr:\n{result.stderr}")

            # Parse JSON output - try stdout first, then stderr
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            return self._parse_output(output, error_output, result.returncode)

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Evaluation timed out",
                "tests_passed": 0,
                "tests_failed": 0,
            }
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tests_passed": 0,
                "tests_failed": 0,
            }

    def _parse_output(self, output: str, stderr: str, returncode: int) -> dict[str, Any]:
        """Parse Docker output to extract JSON results."""
        # Try to find JSON in the output (search stdout first, then combined)
        for text in [output, f"{output}\n{stderr}"]:
            text = text.strip()
            try:
                # Find the last JSON object in output (the evaluation result)
                last_brace = text.rfind("}")
                if last_brace < 0:
                    continue
                # Find the matching opening brace by scanning backwards
                depth = 0
                json_start = -1
                for i in range(last_brace, -1, -1):
                    if text[i] == "}":
                        depth += 1
                    elif text[i] == "{":
                        depth -= 1
                    if depth == 0:
                        json_start = i
                        break
                if json_start >= 0:
                    json_text = text[json_start : last_brace + 1]
                    return json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")

        # Fallback: return raw output with error details
        return {
            "success": returncode == 0,
            "raw_output": output,
            "error": stderr if stderr else "No JSON output found",
            "tests_passed": 0,
            "tests_failed": 0 if returncode == 0 else 1,
        }

    def run_evaluations_batch(
        self,
        jobs: list[EvaluationJob],
        max_workers: int = 5,
        model: str = "glm-4.5",
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run multiple evaluations in parallel using ThreadPoolExecutor.

        Args:
            jobs: List of evaluation jobs to run
            max_workers: Maximum parallel Docker containers (default 5)
            model: Model to use for all evaluations
            progress_callback: Optional callback(task_id, status) for progress

        Returns:
            List of results in the same order as input jobs
        """

        def run_single(job: EvaluationJob) -> tuple[str, dict[str, Any]]:
            if progress_callback:
                progress_callback(job.task_id, "running")
            result = self.run_evaluation(
                skill_content=job.skill_content,
                task_instruction=job.task_instruction,
                task_context=job.task_context,
                package_dir=job.package_dir,
                model=model,
            )
            if progress_callback:
                status = "done" if result.get("success") else "failed"
                progress_callback(job.task_id, status)
            return job.task_id, result

        # Map task_id to index for ordering
        task_to_idx = {job.task_id: i for i, job in enumerate(jobs)}
        results: list[dict[str, Any] | None] = [None] * len(jobs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single, job): job for job in jobs}

            for future in as_completed(futures):
                task_id, result = future.result()
                results[task_to_idx[task_id]] = result

        return results  # type: ignore
