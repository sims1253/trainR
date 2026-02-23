#!/usr/bin/env python
"""Run benchmarks in parallel with progress tracking and status display.

DEPRECATED: This script is transitional. Use run_experiment.py instead.

Migration Guide:
    OLD: uv run python scripts/run_parallel_benchmark.py --models all --tasks "tasks/mined/*.json"
    NEW: uv run python scripts/run_experiment.py --config configs/experiments/your_config.yaml

This script orchestrates parallel execution of benchmark runs across multiple
models, respecting provider-level concurrency limits.

Example usage:
    # Run all models on all tasks (default)
    uv run python scripts/run_parallel_benchmark.py --models all

    # Run specific models on all tasks
    uv run python scripts/run_parallel_benchmark.py --models glm-5-free minimax-m2.5-free

    # Run specific tasks
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks tasks/mined/tidyverse_readr_1615.json \\
        --models all

    # Custom concurrency
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks "tasks/mined/*.json" \\
        --models all \\
        --max-per-provider 3 \\
        --output results/benchmarks/parallel_{timestamp}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# Setup logging
log = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from config import get_llm_config
from rich.console import Console

console = Console()


# ============================================================================
# DEPRECATION LAYER
# ============================================================================

DEPRECATION_WARNING = """
WARNING: run_parallel_benchmark.py is DEPRECATED and will be removed in a future version.

Use the unified experiment runner instead:
    uv run python scripts/run_experiment.py --config configs/experiments/your_config.yaml

To create a config file, see: configs/experiments/r_bench_smoke.yaml for an example.

For parallel execution, set 'execution.parallel_workers' in your config.
"""


def print_deprecation_warning(replacement_hint: str | None = None) -> None:
    """Print deprecation warning with migration guidance."""
    console.print(f"\n[yellow]{DEPRECATION_WARNING}[/yellow]")
    if replacement_hint:
        console.print(f"[cyan]Suggested replacement:[/cyan]\n  {replacement_hint}\n")


def translate_args_to_experiment_config(args: argparse.Namespace) -> dict:
    """Translate legacy run_parallel_benchmark.py args to experiment config format.

    Args:
        args: Parsed argparse namespace from legacy CLI

    Returns:
        Dict representation of experiment config
    """
    # Resolve task paths
    import glob

    if args.tasks is None or args.tasks == ["all"]:
        task_paths = []
        for task_dir in ["tasks/mined", "tasks/train", "tasks/dev", "tasks/held_out"]:
            task_paths.extend(glob.glob(f"{task_dir}/*.json"))
        task_paths = sorted(task_paths)
    else:
        task_paths = []
        for pattern in args.tasks:
            path = Path(pattern)
            if path.exists():
                task_paths.append(str(path))
            else:
                matched = glob.glob(pattern)
                task_paths.extend(matched)

    # Resolve model names
    model_names = args.models if args.models and "all" not in args.models else []
    if not model_names or "all" in args.models:
        # Get all available models
        llm_config = get_llm_config()
        model_names = llm_config.list_models()

    # Build output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output.replace("{timestamp}", timestamp)

    return {
        "name": f"parallel_benchmark_{timestamp}",
        "description": "Migrated from run_parallel_benchmark.py",
        "schema_version": "1.0",
        "models": {"names": model_names},
        "tasks": {
            "selection": "files",
            "files": task_paths[:10]
            if len(task_paths) > 10
            else task_paths,  # Limit for config readability
            "dir": "tasks",
        },
        "skill": {"no_skill": True},
        "execution": {
            "timeout": args.timeout,
            "docker_image": "posit-gskill-eval:latest",
            "repeats": 1,
            "parallel_workers": args.max_per_provider,
            "save_trajectories": True,
            "save_traces": args.verbose,
        },
        "output": {"dir": output_dir},
        "determinism": {"seed": None},
    }


def run_via_unified_runner(args: argparse.Namespace) -> int:
    """Attempt to run via the unified experiment runner.

    Args:
        args: Parsed argparse namespace

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    print_deprecation_warning(
        "uv run python scripts/run_experiment.py --config <config.yaml>\n"
        "    # Note: Set execution.parallel_workers in config for parallelism"
    )

    # Translate args to experiment config
    exp_config = translate_args_to_experiment_config(args)

    # Write to temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(exp_config, f)
        temp_config_path = f.name

    try:
        console.print(f"[dim]Generated temporary experiment config: {temp_config_path}[/dim]")
        console.print("[dim]Running via unified experiment runner...[/dim]\n")

        # Import and run the unified runner
        from run_experiment import main as run_experiment_main

        # Monkey-patch sys.argv to pass our config
        original_argv = sys.argv
        sys.argv = [
            "run_experiment.py",
            "--config",
            temp_config_path,
        ]
        if args.verbose:
            sys.argv.append("--verbose")

        try:
            run_experiment_main()
            return 0
        except SystemExit as e:
            code = e.code
            return code if isinstance(code, int) else 0
        finally:
            sys.argv = original_argv

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_config_path)
        except Exception:
            pass


def get_migration_command(args: argparse.Namespace) -> str:
    """Generate a migration command suggestion based on args."""
    parts = ["uv run python scripts/run_experiment.py --config <your_config.yaml>"]

    parts.append("\n# Config file should include:")
    if args.models and "all" not in args.models:
        parts.append(f"#   models.names: {args.models}")
    if args.max_per_provider:
        parts.append(f"#   execution.parallel_workers: {args.max_per_provider}")
    if args.timeout:
        parts.append(f"#   execution.timeout: {args.timeout}")
    if args.output:
        parts.append(f"#   output.dir: {args.output}")

    return "\n".join(parts)


# ============================================================================
# ORIGINAL IMPLEMENTATION (with deprecation hooks)
# ============================================================================


class ProgressDisplay:
    """Clean in-place progress display using ANSI escape codes."""

    # Unicode box-drawing characters for progress bar
    FILLED = "█"
    EMPTY = "░"
    # Alternative ASCII characters
    # FILLED = "#"
    # EMPTY = "-"

    def __init__(self, terminal_width: int = 80):
        self.terminal_width = terminal_width or 80
        self._last_update = 0.0
        self._update_interval = 1.5  # Update every 1.5 seconds
        self._initialized = False

    def _get_terminal_width(self) -> int:
        """Get terminal width, with fallback."""
        try:
            return shutil.get_terminal_size((80, 24)).columns
        except Exception:
            return 80

    def _move_cursor_top(self) -> None:
        """Move cursor to top of screen."""
        sys.stdout.write("\033[H")

    def _clear_from_cursor(self) -> None:
        """Clear screen from cursor to end."""
        sys.stdout.write("\033[J")

    def _clear_line(self) -> None:
        """Clear current line."""
        sys.stdout.write("\033[2K")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 0 or seconds == float("inf"):
            return "calculating..."
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _truncate_string(self, s: str, max_len: int) -> str:
        """Truncate string to max length with ellipsis."""
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    def _render_progress_bar(self, progress: float, width: int) -> str:
        """Render a progress bar."""
        if width <= 0:
            return ""
        filled_count = int(progress / 100 * width)
        filled_count = max(0, min(width, filled_count))
        empty_count = width - filled_count
        return self.FILLED * filled_count + self.EMPTY * empty_count

    def _format_worker_line(self, worker: "WorkerInfo", max_width: int) -> str:
        """Format a single worker line with truncation."""
        elapsed = worker.format_elapsed()
        # Format: "provider/model → task_name (elapsed)"
        provider_model = f"{worker.provider}/{worker.model}"
        task_part = f"→ {worker.task_id}"

        # Calculate available space
        elapsed_part = f" ({elapsed})"
        min_prefix = 10  # Minimum for provider/model

        available = max_width - len(elapsed_part) - 3  # 3 for spacing and arrow

        if available < 20:
            # Very narrow terminal, just show truncated whole line
            full = f"{provider_model} {task_part} {elapsed_part}"
            return self._truncate_string(full, max_width)

        # Truncate components if needed
        if len(provider_model) + len(task_part) > available:
            # Need to truncate - prefer showing full model, truncate task
            task_avail = available - len(provider_model) - 3
            if task_avail < 10:
                # Not enough for task, truncate model instead
                provider_model = self._truncate_string(provider_model, min_prefix)
                task_avail = available - len(provider_model) - 3
            task_part = f"→ {self._truncate_string(worker.task_id, task_avail - 2)}"

        return f"{provider_model} {task_part}{elapsed_part}"

    def should_update(self, force: bool = False) -> bool:
        """Check if enough time has passed for an update."""
        now = time.time()
        if force or not self._initialized:
            self._last_update = now
            self._initialized = True
            return True
        if now - self._last_update >= self._update_interval:
            self._last_update = now
            return True
        return False

    def render(self, benchmark: "ParallelBenchmark") -> str:
        """Render the full progress display."""
        width = self._get_terminal_width()
        lines = []

        # Top border
        border = "━" * width
        lines.append(border)

        # Header line
        task_count = len(set(w.task_id for w in benchmark.workers))
        model_count = sum(len(m) for m in benchmark.models_by_provider.values())
        header = f"BENCHMARK: {benchmark.total_runs} runs ({task_count} tasks × {model_count} models) | Workers: {benchmark.max_per_provider} per provider"
        lines.append(self._center_or_truncate(header, width))
        lines.append(border)

        # Progress bar
        progress = benchmark.progress_pct
        completed = benchmark.completed_count + benchmark.failed_count + benchmark.timeout_count
        bar_width = width - 30  # Leave room for stats
        bar_width = max(20, bar_width)
        bar = self._render_progress_bar(progress, bar_width)

        eta_str = self._format_duration(benchmark.eta_seconds)
        progress_line = (
            f"[{bar}] {completed}/{benchmark.total_runs} ({progress:.0f}%) | ETA: {eta_str}"
        )
        lines.append(progress_line)
        lines.append("")  # Blank line

        # Running workers
        running = benchmark.running
        if running:
            lines.append(f"RUNNING ({len(running)}):")
            # Show max 6 running workers
            max_display = 6
            sorted_running = sorted(running, key=lambda x: x.start_time or 0)
            for w in sorted_running[:max_display]:
                line = self._format_worker_line(w, width - 2)
                lines.append(f"  {line}")
            if len(running) > max_display:
                lines.append(f"  ... and {len(running) - max_display} more")
        else:
            lines.append("RUNNING: (none)")

        lines.append("")  # Blank line

        # Stats line
        passed_count = len([r for r in benchmark.completed if r.passed])
        failed_count = len([r for r in benchmark.completed if not r.passed])
        worker_failed = benchmark.failed_count
        timeout_count = benchmark.timeout_count

        total_completed = benchmark.completed_count
        total_done = total_completed + worker_failed + timeout_count
        if total_done > 0:
            pass_pct = passed_count / total_done * 100
            fail_pct = (failed_count + worker_failed) / total_done * 100
        else:
            pass_pct = 0
            fail_pct = 0

        stats_line = f"COMPLETED: {total_completed} | PASSED: {passed_count} ({pass_pct:.0f}%) | FAILED: {failed_count + worker_failed} ({fail_pct:.0f}%) | TIMEOUT: {timeout_count}"
        lines.append(stats_line)

        # Bottom border
        lines.append(border)

        return "\n".join(lines)

    def _center_or_truncate(self, text: str, width: int) -> str:
        """Center text or truncate if too long."""
        if len(text) <= width:
            # Center within width
            padding = (width - len(text)) // 2
            return " " * padding + text + " " * (width - len(text) - padding)
        return text[:width]

    def display(self, benchmark: "ParallelBenchmark") -> None:
        """Display progress in-place."""
        output = self.render(benchmark)

        # Move to top and clear
        self._move_cursor_top()
        self._clear_from_cursor()

        # Write output
        sys.stdout.write(output)
        sys.stdout.flush()

    def clear(self) -> None:
        """Clear the progress display area."""
        self._move_cursor_top()
        self._clear_from_cursor()
        sys.stdout.flush()

    def render_final_summary(self, results: dict, benchmark: "ParallelBenchmark") -> str:
        """Render final summary with top performers."""
        width = self._get_terminal_width()
        border = "━" * width
        lines = []

        lines.append(border)
        lines.append(self._center_or_truncate("BENCHMARK COMPLETE", width))
        lines.append(border)
        lines.append("")

        # Overall stats
        duration = results.get("duration_s", 0)
        lines.append(f"Total Duration: {self._format_duration(duration)}")
        lines.append(f"Total Runs: {results.get('total_runs', 0)}")
        lines.append(f"Completed: {results.get('completed', 0)}")
        lines.append(f"Failed: {results.get('failed', 0)}")
        lines.append(f"Timeouts: {results.get('timeouts', 0)}")
        lines.append("")

        # Token usage
        total_input = sum(r.get("input_tokens", 0) for r in results.get("results", []))
        total_output = sum(r.get("output_tokens", 0) for r in results.get("results", []))
        lines.append(
            f"Total Tokens: {self._format_token_count(total_input)} in / {self._format_token_count(total_output)} out"
        )
        lines.append("")

        # Top performers by pass rate
        lines.append(border)
        lines.append(self._center_or_truncate("TOP PERFORMERS BY PASS RATE", width))
        lines.append(border)

        # Group by model
        model_stats: dict[str, dict] = {}
        for r in results.get("results", []):
            model = r.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = {"total": 0, "passed": 0, "total_latency": 0}
            model_stats[model]["total"] += 1
            if r.get("passed"):
                model_stats[model]["passed"] += 1
            model_stats[model]["total_latency"] += r.get("latency_s", 0)

        # Sort by pass rate
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: (x[1]["passed"] / max(x[1]["total"], 1), x[1]["passed"]),
            reverse=True,
        )

        # Header
        lines.append(f"{'Model':<35} {'Pass Rate':>12} {'Avg Time':>12} {'Tests':>8}")
        lines.append("─" * min(width, 70))

        for model, stats in sorted_models[:10]:  # Top 10
            if stats["total"] > 0:
                pass_rate = stats["passed"] / stats["total"]
                avg_lat = stats["total_latency"] / stats["total"]
            else:
                pass_rate = 0
                avg_lat = 0

            # Color based on pass rate
            if pass_rate >= 0.7:
                color = "\033[32m"  # Green
            elif pass_rate >= 0.3:
                color = "\033[33m"  # Yellow
            else:
                color = "\033[31m"  # Red
            reset = "\033[0m"

            model_display = self._truncate_string(model, 33)
            lines.append(
                f"{model_display:<35} {color}{pass_rate:>11.1%}{reset} "
                f"{self._format_duration(avg_lat):>12} {stats['total']:>8}"
            )

        if len(sorted_models) > 10:
            lines.append(f"... and {len(sorted_models) - 10} more models")

        lines.append("")
        lines.append(border)

        return "\n".join(lines)

    def _format_token_count(self, n: int) -> str:
        """Format token count with K/M suffix."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.0f}K"
        return str(n)


class WorkerState(Enum):
    """State of a benchmark worker."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class WorkItem:
    """A work item for scheduling with provider preferences."""

    model: str
    task_id: str
    task_path: str
    providers: list[str]  # List of available providers for this model
    preferred_provider: str | None = None  # Set during scheduling


@dataclass
class WorkerInfo:
    """Information about a benchmark worker."""

    model: str
    provider: str
    task_id: str
    task_path: str
    available_providers: list[str] = field(default_factory=list)
    state: WorkerState = WorkerState.QUEUED
    start_time: float | None = None
    end_time: float | None = None
    result: dict | None = None
    error: str | None = None
    output_file: Path | None = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def format_elapsed(self) -> str:
        """Format elapsed time for display."""
        elapsed = self.elapsed
        if elapsed < 60:
            return f"{int(elapsed)}s"
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m{seconds}s"


@dataclass
class BenchmarkResult:
    """Result from a completed benchmark run."""

    model: str
    task_id: str
    passed: bool
    score: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        """Create from result dictionary."""
        token_usage = data.get("token_usage", {})
        return cls(
            model=data.get("model", "unknown"),
            task_id=data.get("task_id", "unknown"),
            passed=data.get("passed", False),
            score=data.get("score", 0.0),
            latency_s=data.get("latency_s", 0.0),
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0),
            error=data.get("error_message"),
        )

    def format_tokens(self) -> str:
        """Format token counts for display."""

        def abbreviate(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n / 1_000:.0f}K"
            return str(n)

        return f"{abbreviate(self.input_tokens)} in / {abbreviate(self.output_tokens)} out"


def load_models_from_config(config_path: str = "configs/llm.yaml") -> dict[str, dict]:
    """Load models from llm.yaml, grouped by provider.

    Returns:
        Dict mapping provider name to list of model configs.
    """
    llm_config = get_llm_config()

    # Get all model names
    model_names = llm_config.list_models()

    # Group by provider
    providers: dict[str, list[dict]] = {}

    for name in model_names:
        try:
            cfg = llm_config.get_model_config(name)
            provider = cfg.get("provider", "unknown")

            if provider not in providers:
                providers[provider] = []

            providers[provider].append(
                {
                    "name": name,
                    "provider": provider,
                    "id": cfg.get("id"),
                    "litellm_model": llm_config.get_litellm_model(name),
                    "api_key": llm_config.get_model_api_key(name),
                    "api_key_env": cfg.get("api_key_env"),
                }
            )
        except ValueError:
            continue

    return providers


def filter_models(
    models_by_provider: dict[str, list[dict]], model_filter: list[str] | None
) -> dict[str, list[dict]]:
    """Filter models based on user selection.

    Args:
        models_by_provider: All models grouped by provider
        model_filter: List of model names to include, or None for all

    Returns:
        Filtered models grouped by provider
    """
    if not model_filter or "all" in model_filter:
        # Return a copy to avoid modifying the original
        return {k: list(v) for k, v in models_by_provider.items()}

    filtered: dict[str, list[dict]] = {}

    for provider, models in models_by_provider.items():
        for model in models:
            if model["name"] in model_filter:
                if provider not in filtered:
                    filtered[provider] = []
                filtered[provider].append(model)

    return filtered


class BenchmarkWorker:
    """Manages a single benchmark run as a subprocess."""

    def __init__(self, model: str, provider: str, task_path: str, output_dir: Path):
        self.model = model
        self.provider = provider
        self.task_path = task_path
        self.output_dir = output_dir
        self.info = WorkerInfo(
            model=model,
            provider=provider,
            task_id=Path(task_path).stem,
            task_path=task_path,
        )

    async def run(self, timeout: int = 900) -> WorkerInfo:
        """Execute the benchmark and return result info.

        Args:
            timeout: Maximum time in seconds to wait for the benchmark (default: 900 = 15 min)
        """
        # Note: state and start_time are set by the coordinator in _run_worker

        # Create output subdirectory for this run
        run_output_dir = self.output_dir / f"{self.info.task_id}_{self.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "scripts/run_benchmark.py",
            "--task",
            self.task_path,
            "--worker-model",
            self.model,
            "--output-dir",
            str(run_output_dir),
            "--no-resume",  # Don't skip based on previous runs
        ]

        try:
            # Run subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                # Graceful shutdown - send SIGTERM first
                try:
                    process.terminate()
                    # Wait up to 30 seconds for graceful shutdown
                    try:
                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
                    except asyncio.TimeoutError:
                        # Force kill if graceful shutdown didn't work
                        process.kill()
                        await process.wait()
                        stdout, stderr = b"", b""
                except Exception:
                    pass

                self.info.state = WorkerState.TIMEOUT
                self.info.error = f"Timeout after {timeout}s"
                self.info.end_time = time.time()

                # Try to collect partial results even after timeout
                await self._collect_partial_results(run_output_dir)
                return self.info

            self.info.end_time = time.time()

            if process.returncode != 0:
                self.info.state = WorkerState.FAILED
                self.info.error = stderr.decode() if stderr else "Unknown error"
                return self.info

            # Find and read the result file
            result_files = list(run_output_dir.glob("**/benchmark_results.json"))
            if result_files:
                self.info.output_file = result_files[0]
                with open(result_files[0]) as f:
                    data = json.load(f)
                    results = data.get("results", [])
                    if results:
                        # Get the first (and should be only) result
                        self.info.result = results[0]

            self.info.state = WorkerState.COMPLETED

        except Exception as e:
            self.info.end_time = time.time()
            self.info.state = WorkerState.FAILED
            self.info.error = str(e)

        return self.info

    async def _collect_partial_results(self, run_output_dir: Path) -> None:
        """Try to collect partial results after timeout."""
        try:
            result_files = list(run_output_dir.glob("**/benchmark_results.json"))
            if result_files:
                with open(result_files[0]) as f:
                    result = json.load(f)
                if "results" in result and result["results"]:
                    r = result["results"][0]
                    self.info.result = {
                        "passed": False,
                        "token_usage": r.get("token_usage", {}),
                        "latency_s": r.get("latency_s", 0),
                        "partial": True,
                        "timeout": True,
                    }
        except Exception:
            pass


class ParallelBenchmark:
    """Coordinates parallel benchmark execution with worker pool pattern."""

    def __init__(
        self,
        tasks: list[str],
        models_by_provider: dict[str, list[dict]],
        max_per_provider: int,
        output_dir: Path,
        timeout: int = 900,
    ):
        self.tasks = tasks
        self.models_by_provider = models_by_provider
        self.max_per_provider = max_per_provider
        self.output_dir = output_dir
        self.timeout = timeout

        # Create all workers (will be populated with load-balancing in run())
        self.workers: list[WorkerInfo] = []

        # Per-provider queues for worker pools (will be populated in run())
        self._queues: dict[str, asyncio.Queue] = {}

        # Results tracking
        self.completed: list[BenchmarkResult] = []
        self.failed: list[WorkerInfo] = []

        # Progress tracking
        self._start_time: float | None = None
        self._lock = asyncio.Lock()

    def _strip_thinking_from_trajectory(self, trajectory_path: Path) -> None:
        """Strip encrypted thinking content from trajectory file to save disk space.

        Reasoning models like gpt-5-nano output encrypted thinking content that can
        cause trajectory files to balloon to 70+ MB. This removes thinking_content
        fields while preserving the rest of the trajectory.

        Args:
            trajectory_path: Path to the trajectory JSONL file
        """
        if not trajectory_path.exists():
            return

        temp_file = trajectory_path.with_suffix(".tmp")
        bytes_saved = 0

        try:
            with open(trajectory_path, "r") as infile, open(temp_file, "w") as outfile:
                for line in infile:
                    try:
                        if line.strip().startswith("{"):
                            event = json.loads(line)
                            # Remove thinking_content if present
                            if "thinking_content" in event:
                                bytes_saved += len(event["thinking_content"])
                                del event["thinking_content"]
                            # Also check in nested message
                            if "message" in event and isinstance(event["message"], dict):
                                if "thinking_content" in event["message"]:
                                    bytes_saved += len(event["message"]["thinking_content"])
                                    del event["message"]["thinking_content"]
                            outfile.write(json.dumps(event) + "\n")
                        else:
                            outfile.write(line)
                    except json.JSONDecodeError:
                        outfile.write(line)

            # Replace original with stripped version
            shutil.move(str(temp_file), str(trajectory_path))

            if bytes_saved > 0:
                log.debug(
                    f"Stripped {bytes_saved / 1024 / 1024:.1f} MB of thinking content from {trajectory_path}"
                )

        except Exception as e:
            log.warning(f"Failed to strip thinking content: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _strip_thinking_from_run_dir(self, run_dir: Path) -> None:
        """Strip thinking content from all trajectory files in a run directory.

        Args:
            run_dir: Directory containing benchmark run outputs
        """
        # Find trajectory files (typically named trajectory.jsonl or similar)
        for trajectory_path in run_dir.glob("**/trajectory*.jsonl"):
            self._strip_thinking_from_trajectory(trajectory_path)

    def _get_model_providers(self, model_name: str) -> list[str]:
        """Get list of available providers for a model.

        Args:
            model_name: The model name from llm.yaml

        Returns:
            List of provider names that support this model
        """
        llm_config = get_llm_config()
        providers = llm_config.get_providers(model_name)
        # Extract just the provider names
        return [p.get("provider", "unknown") for p in providers if p.get("provider")]

    def _create_workers_with_load_balancing(self) -> None:
        """Create workers using intelligent load balancing across providers.

        Strategy:
        1. Build work items with all available providers for each model/task
        2. Sort: exclusive models (1 provider) first, then flexible (2+ providers)
        3. Schedule exclusive models to their only provider
        4. Use flexible models to balance load across providers
        """
        # Build work items with provider options
        work_items: list[WorkItem] = []

        # Get all unique models from the provider-grouped dict
        all_models: set[str] = set()
        for provider_models in self.models_by_provider.values():
            for model in provider_models:
                all_models.add(model["name"])

        for model_name in all_models:
            providers = self._get_model_providers(model_name)
            for task_path in self.tasks:
                work_items.append(
                    WorkItem(
                        model=model_name,
                        task_id=Path(task_path).stem,
                        task_path=task_path,
                        providers=providers,
                    )
                )

        # Sort: exclusive models first (fewer providers = higher priority for assignment)
        work_items.sort(key=lambda w: len(w.providers))

        # Track provider load for balancing
        provider_load: dict[str, int] = {p: 0 for p in self.models_by_provider.keys()}

        # Schedule work items
        for item in work_items:
            if len(item.providers) == 1:
                # Exclusive - must use this provider
                item.preferred_provider = item.providers[0]
            else:
                # Flexible - assign to least loaded available provider
                # Only consider providers that are in our models_by_provider
                available = [p for p in item.providers if p in self.models_by_provider]
                if not available:
                    # Fallback to first provider if none match
                    available = item.providers
                least_loaded = min(available, key=lambda p: provider_load.get(p, 0))
                item.preferred_provider = least_loaded

            # Update load tracking
            if item.preferred_provider:
                provider_load[item.preferred_provider] = (
                    provider_load.get(item.preferred_provider, 0) + 1
                )

            # Create WorkerInfo
            worker = WorkerInfo(
                model=item.model,
                provider=item.preferred_provider or item.providers[0],
                task_id=item.task_id,
                task_path=item.task_path,
                available_providers=item.providers,
            )
            self.workers.append(worker)

    @property
    def total_runs(self) -> int:
        """Total number of benchmark runs."""
        return len(self.workers)

    @property
    def running(self) -> list[WorkerInfo]:
        """Currently running workers."""
        return [w for w in self.workers if w.state == WorkerState.RUNNING]

    @property
    def queued(self) -> list[WorkerInfo]:
        """Queued workers."""
        return [w for w in self.workers if w.state == WorkerState.QUEUED]

    @property
    def queue_size(self) -> int:
        """Number of items remaining across all provider queues."""
        return sum(q.qsize() for q in self._queues.values())

    @property
    def completed_count(self) -> int:
        """Number of completed workers."""
        return len([w for w in self.workers if w.state == WorkerState.COMPLETED])

    @property
    def failed_count(self) -> int:
        """Number of failed workers."""
        return len([w for w in self.workers if w.state == WorkerState.FAILED])

    @property
    def timeout_count(self) -> int:
        """Number of timed-out workers."""
        return len([w for w in self.workers if w.state == WorkerState.TIMEOUT])

    @property
    def progress_pct(self) -> float:
        """Progress percentage."""
        done = self.completed_count + self.failed_count + self.timeout_count
        return (done / self.total_runs * 100) if self.total_runs > 0 else 0

    @property
    def avg_latency(self) -> float:
        """Average latency of completed runs."""
        latencies = [
            w.elapsed
            for w in self.workers
            if w.state in (WorkerState.COMPLETED, WorkerState.FAILED, WorkerState.TIMEOUT)
        ]
        return statistics.mean(latencies) if latencies else 0

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        remaining = self.queue_size + len(self.running)
        if remaining == 0:
            return 0

        avg_lat = self.avg_latency
        if avg_lat == 0:
            # No completed runs yet, estimate based on running count
            return remaining * 120  # Assume 2 min average

        # Factor in parallelism (max_per_provider workers per provider)
        effective_parallelism = max(1, self.max_per_provider * len(self.models_by_provider))

        return (remaining * avg_lat) / effective_parallelism

    def format_eta(self) -> str:
        """Format ETA for display."""
        eta = self.eta_seconds
        if eta < 60:
            return f"~{int(eta)}s"
        elif eta < 3600:
            return f"~{int(eta // 60)} min"
        else:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            return f"~{hours}h {minutes}m"

    def _get_task_models(self, task_path: str) -> list[str]:
        """Get list of model names for a task."""
        models = []
        for provider_models in self.models_by_provider.values():
            for model in provider_models:
                models.append(model["name"])
        return models

    async def _worker(self, worker_id: str, queue: asyncio.Queue, results_list: list) -> None:
        """Worker that pulls tasks from its provider's queue.

        Args:
            worker_id: Identifier for this worker (e.g., "openai_0")
            queue: Provider-specific queue containing WorkerInfo items
            results_list: Shared list to append results to
        """
        while True:
            try:
                worker_info = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Update state to running
            worker_info.state = WorkerState.RUNNING
            worker_info.start_time = time.time()

            try:
                # Create and run the benchmark worker
                benchmark_worker = BenchmarkWorker(
                    model=worker_info.model,
                    provider=worker_info.provider,
                    task_path=worker_info.task_path,
                    output_dir=self.output_dir,
                )
                result_info = await benchmark_worker.run(timeout=self.timeout)

                # Copy result data to worker_info
                worker_info.end_time = result_info.end_time
                worker_info.result = result_info.result
                worker_info.error = result_info.error

                # Update state based on result
                if result_info.state == WorkerState.COMPLETED:
                    worker_info.state = WorkerState.COMPLETED
                    # Strip thinking content from trajectory files to save disk space
                    run_output_dir = self.output_dir / f"{worker_info.task_id}_{worker_info.model}"
                    self._strip_thinking_from_run_dir(run_output_dir)
                    async with self._lock:
                        if result_info.result:
                            bench_result = BenchmarkResult.from_dict(result_info.result)
                            self.completed.append(bench_result)
                        results_list.append({"worker_info": worker_info, "success": True})
                elif result_info.state == WorkerState.TIMEOUT:
                    worker_info.state = WorkerState.TIMEOUT
                    # Strip thinking content from partial trajectory files
                    run_output_dir = self.output_dir / f"{worker_info.task_id}_{worker_info.model}"
                    self._strip_thinking_from_run_dir(run_output_dir)
                    async with self._lock:
                        # Add partial result if available
                        if result_info.result:
                            bench_result = BenchmarkResult.from_dict(result_info.result)
                            self.completed.append(bench_result)
                        self.failed.append(worker_info)  # Track in failed for summary
                        results_list.append(
                            {"worker_info": worker_info, "success": False, "timeout": True}
                        )
                else:
                    worker_info.state = WorkerState.FAILED
                    async with self._lock:
                        self.failed.append(worker_info)
                        results_list.append({"worker_info": worker_info, "success": False})

            except Exception as e:
                worker_info.state = WorkerState.FAILED
                worker_info.end_time = time.time()
                worker_info.error = f"Worker exception: {e}"
                async with self._lock:
                    self.failed.append(worker_info)
                    results_list.append({"worker_info": worker_info, "success": False})
            finally:
                queue.task_done()

    async def run(self, progress_callback: Callable | None = None) -> dict:
        """Run all benchmarks with intelligent load balancing.

        Uses load-aware scheduling:
        1. Exclusive models (1 provider) are assigned to their only provider
        2. Flexible models (2+ providers) are used to balance load across providers

        This ensures optimal utilization by allowing faster providers to pick up
        more work from flexible models.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with results summary
        """
        from collections import defaultdict

        self._start_time = time.time()

        # Create workers with load balancing (if not already created)
        if not self.workers:
            self._create_workers_with_load_balancing()

        # Group workers by their assigned provider
        workers_by_provider: dict[str, list[WorkerInfo]] = defaultdict(list)
        for worker_info in self.workers:
            workers_by_provider[worker_info.provider].append(worker_info)

        # Create one queue per provider, populated with that provider's tasks
        self._queues = {}
        for provider, worker_list in workers_by_provider.items():
            queue = asyncio.Queue()
            for worker_info in worker_list:
                await queue.put(worker_info)
            self._queues[provider] = queue

        # Shared results list for tracking
        results_list: list[dict] = []
        all_worker_tasks: list[asyncio.Task] = []

        # Create worker pool for each provider
        for provider, queue in self._queues.items():
            num_provider_workers = min(
                self.max_per_provider, queue.qsize()
            )  # Don't create more workers than tasks
            for i in range(num_provider_workers):
                worker_id = f"{provider}_{i}"
                worker_task = asyncio.create_task(self._worker(worker_id, queue, results_list))
                all_worker_tasks.append(worker_task)

        # Wait for completion with progress updates
        while any(not w.done() for w in all_worker_tasks):
            if progress_callback:
                await progress_callback(self)
            await asyncio.sleep(0.5)

        # Wait for all workers to finish (should already be done)
        await asyncio.gather(*all_worker_tasks)

        # Collect final results
        results = {
            "total_runs": self.total_runs,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "timeouts": self.timeout_count,
            "results": [r.__dict__ for r in self.completed],
            "failed_workers": [
                {
                    "model": w.model,
                    "task_id": w.task_id,
                    "error": w.error,
                    "timeout": w.state == WorkerState.TIMEOUT,
                }
                for w in self.failed
            ],
            "duration_s": time.time() - self._start_time,
        }

        return results


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def display_progress(
    benchmark: ParallelBenchmark, display_obj: ProgressDisplay | None = None
) -> None:
    """Print the progress display."""
    if display_obj is None:
        display_obj = ProgressDisplay()
    display_obj.display(benchmark)


async def progress_updater(benchmark: ParallelBenchmark, display_obj: ProgressDisplay) -> None:
    """Async callback for progress updates."""
    if display_obj.should_update():
        display_obj.display(benchmark)


def save_final_results(results: dict, output_dir: Path) -> Path:
    """Save final results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"parallel_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results_file


def print_summary(results: dict) -> None:
    """Print final summary table."""
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()

    # Group by model
    model_stats: dict[str, dict] = {}
    for r in results.get("results", []):
        model = r.get("model", "unknown")
        if model not in model_stats:
            model_stats[model] = {
                "total": 0,
                "passed": 0,
                "total_latency": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        model_stats[model]["total"] += 1
        if r.get("passed"):
            model_stats[model]["passed"] += 1
        model_stats[model]["total_latency"] += r.get("latency_s", 0)
        model_stats[model]["total_input_tokens"] += r.get("input_tokens", 0)
        model_stats[model]["total_output_tokens"] += r.get("output_tokens", 0)

    # Print table
    print(f"{'Model':<30} {'Pass Rate':>10} {'Avg Latency':>12} {'Input':>12} {'Output':>12}")
    print("-" * 80)

    for model, stats in sorted(model_stats.items()):
        if stats["total"] > 0:
            pass_rate = stats["passed"] / stats["total"]
            avg_lat = stats["total_latency"] / stats["total"]
        else:
            pass_rate = 0
            avg_lat = 0

        def fmt_tokens(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n / 1_000:.0f}K"
            return str(n)

        # Color based on pass rate
        if pass_rate >= 0.7:
            color = "\033[32m"  # Green
        elif pass_rate >= 0.3:
            color = "\033[33m"  # Yellow
        else:
            color = "\033[31m"  # Red
        reset = "\033[0m"

        print(
            f"{model:<30} {color}{pass_rate:>9.1%}{reset} {avg_lat:>11.1f}s "
            f"{fmt_tokens(stats['total_input_tokens']):>12} "
            f"{fmt_tokens(stats['total_output_tokens']):>12}"
        )

    print()
    print(f"Total runs: {results.get('total_runs', 0)}")
    print(f"Completed: {results.get('completed', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Timeouts: {results.get('timeouts', 0)}")
    print(f"Duration: {results.get('duration_s', 0):.1f}s")


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    import glob

    # Resolve task paths
    if args.tasks is None or args.tasks == ["all"]:
        # Load tasks from all directories
        task_paths = []
        for task_dir in ["tasks/mined", "tasks/train", "tasks/dev", "tasks/held_out"]:
            task_paths.extend(glob.glob(f"{task_dir}/*.json"))
        task_paths = sorted(task_paths)
    else:
        # Use specified task patterns
        task_paths = []
        for pattern in args.tasks:
            path = Path(pattern)
            if path.exists():
                task_paths.append(str(path))
            else:
                # Treat as glob pattern
                matched = glob.glob(pattern)
                task_paths.extend(matched)

    if not task_paths:
        print("Error: No task files found")
        return 1

    # Determine if progress should be shown
    show_progress = args.progress and not args.no_progress

    if show_progress:
        print(f"Found {len(task_paths)} task(s)")
    else:
        # Programmatic mode: just print basic info as JSON-compatible
        print(json.dumps({"status": "starting", "tasks": len(task_paths)}))

    # Load models
    models_by_provider = load_models_from_config()

    # Filter models if specified
    if args.models and args.models != ["all"]:
        models_by_provider = filter_models(models_by_provider, args.models)

    if not models_by_provider:
        print("Error: No models found matching filter")
        return 1

    total_models = sum(len(m) for m in models_by_provider.values())

    if show_progress:
        print(f"Using {total_models} model(s) across {len(models_by_provider)} provider(s)")
        for provider, models in models_by_provider.items():
            print(f"  {provider}: {', '.join(m['name'] for m in models)}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output.replace("{timestamp}", timestamp))

    # Create benchmark coordinator
    benchmark = ParallelBenchmark(
        tasks=task_paths,
        models_by_provider=models_by_provider,
        max_per_provider=args.max_per_provider,
        output_dir=output_dir,
        timeout=args.timeout,
    )

    if show_progress:
        print(f"\nTotal benchmark runs: {benchmark.total_runs}")
        print(f"Max workers per provider: {args.max_per_provider}")
        print(f"Output directory: {output_dir}")
        print("\nStarting parallel benchmark...")

    # Create progress display
    progress_display = ProgressDisplay() if show_progress else None

    # Clear screen and initialize display area before starting
    if show_progress and progress_display:
        # Print enough newlines to reserve space for the display
        # Then move cursor up to start position
        print("\n" * 15, end="")
        sys.stdout.write("\033[15A")  # Move cursor back up
        sys.stdout.flush()

    # Run with progress display callback
    async def progress_callback(bench: ParallelBenchmark) -> None:
        if progress_display and progress_display.should_update():
            progress_display.display(bench)

    results = await benchmark.run(progress_callback=progress_callback if show_progress else None)

    # Final display
    if show_progress and progress_display:
        progress_display.clear()
        final_summary = progress_display.render_final_summary(results, benchmark)
        print(final_summary)
    else:
        # Programmatic mode: print JSON summary
        summary = {
            "status": "complete",
            "total_runs": results.get("total_runs", 0),
            "completed": results.get("completed", 0),
            "failed": results.get("failed", 0),
            "timeouts": results.get("timeouts", 0),
            "duration_s": results.get("duration_s", 0),
            "output_dir": str(output_dir),
        }
        print(json.dumps(summary))

    # Save results
    results_file = save_final_results(results, output_dir)
    if show_progress:
        print(f"\nResults saved to: {results_file}")

    # Return appropriate exit code (consider both failures and timeouts as non-success)
    if results.get("failed", 0) > 0 or results.get("timeouts", 0) > 0:
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for run_parallel_benchmark.py."""
    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Run benchmarks in parallel. Use run_experiment.py instead.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DEPRECATION NOTICE:
    This script is deprecated. Please use the unified experiment runner:
    
    uv run python scripts/run_experiment.py --config configs/experiments/your_config.yaml

    For parallel execution, set 'execution.parallel_workers' in your config.

Examples:
    # Run all models on all tasks (default)
    uv run python scripts/run_parallel_benchmark.py --models all

    # Run specific models on all tasks
    uv run python scripts/run_parallel_benchmark.py --models glm-5-free minimax-m2.5-free

    # Run specific tasks
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks tasks/mined/tidyverse_readr_1615.json \\
        --models all

    # Custom concurrency and output
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks "tasks/mined/*.json" \\
        --models all \\
        --max-per-provider 3 \\
        --output results/my_benchmark
        """,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task file paths or glob patterns (e.g., 'tasks/mined/*.json'). Default: all task directories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names to run, or 'all' for all available models (default: all)",
    )
    parser.add_argument(
        "--max-per-provider",
        type=int,
        default=2,
        help="Maximum concurrent workers per provider (default: 2)",
    )
    parser.add_argument(
        "--output",
        default="results/benchmarks/parallel_{timestamp}",
        help="Output directory (supports {timestamp} placeholder)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display for programmatic use",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show live progress display (default)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds per benchmark run (default: 900 = 15 min)",
    )

    # Migration option
    parser.add_argument(
        "--use-unified-runner",
        action="store_true",
        help="Route through the unified run_experiment.py (experimental)",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Always show deprecation warning
    migration_cmd = get_migration_command(args)
    print_deprecation_warning(migration_cmd)

    # If explicitly requested, use unified runner
    if args.use_unified_runner:
        return run_via_unified_runner(args)

    # Run async main
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
