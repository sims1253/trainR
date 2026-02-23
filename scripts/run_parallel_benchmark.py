#!/usr/bin/env python
"""Run benchmarks in parallel with progress tracking and status display.

This script orchestrates parallel execution of benchmark runs across multiple
models, respecting provider-level concurrency limits.

Example usage:
    # Run all models on all tasks
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks tasks/mined/*.json \\
        --models all

    # Run specific models
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks tasks/mined/tidyverse_readr_1615.json \\
        --models glm-5-free minimax-m2.5-free

    # Custom concurrency
    uv run python scripts/run_parallel_benchmark.py \\
        --tasks tasks/mined/*.json \\
        --models all \\
        --max-per-provider 3 \\
        --output results/benchmarks/parallel_{timestamp}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from config import get_llm_config


class WorkerState(Enum):
    """State of a benchmark worker."""

    QUEUED = "queued"
    WAITING = "waiting"  # Waiting for semaphore slot
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerInfo:
    """Information about a benchmark worker."""

    model: str
    provider: str
    task_id: str
    task_path: str
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
        return models_by_provider

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

    async def run(self) -> WorkerInfo:
        """Execute the benchmark and return result info."""
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
            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent,
            )

            stdout, stderr = await process.communicate()

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


class ParallelBenchmark:
    """Coordinates parallel benchmark execution with provider-level concurrency."""

    def __init__(
        self,
        tasks: list[str],
        models_by_provider: dict[str, list[dict]],
        max_per_provider: int,
        output_dir: Path,
    ):
        self.tasks = tasks
        self.models_by_provider = models_by_provider
        self.max_per_provider = max_per_provider
        self.output_dir = output_dir

        # Create all workers
        self.workers: list[WorkerInfo] = []
        self._create_workers()

        # Semaphores per provider
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        for provider in models_by_provider:
            self._semaphores[provider] = asyncio.Semaphore(max_per_provider)

        # Results tracking
        self.completed: list[BenchmarkResult] = []
        self.failed: list[WorkerInfo] = []

        # Progress tracking
        self._start_time: float | None = None
        self._lock = asyncio.Lock()

    def _create_workers(self) -> None:
        """Create worker info for all model/task combinations."""
        for provider, models in self.models_by_provider.items():
            for model in models:
                for task_path in self.tasks:
                    worker = WorkerInfo(
                        model=model["name"],
                        provider=provider,
                        task_id=Path(task_path).stem,
                        task_path=task_path,
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
    def waiting(self) -> list[WorkerInfo]:
        """Workers waiting for semaphore slot."""
        return [w for w in self.workers if w.state == WorkerState.WAITING]

    @property
    def queued(self) -> list[WorkerInfo]:
        """Queued workers."""
        return [w for w in self.workers if w.state == WorkerState.QUEUED]

    @property
    def completed_count(self) -> int:
        """Number of completed workers."""
        return len([w for w in self.workers if w.state == WorkerState.COMPLETED])

    @property
    def failed_count(self) -> int:
        """Number of failed workers."""
        return len([w for w in self.workers if w.state == WorkerState.FAILED])

    @property
    def progress_pct(self) -> float:
        """Progress percentage."""
        done = self.completed_count + self.failed_count
        return (done / self.total_runs * 100) if self.total_runs > 0 else 0

    @property
    def avg_latency(self) -> float:
        """Average latency of completed runs."""
        latencies = [
            w.elapsed
            for w in self.workers
            if w.state in (WorkerState.COMPLETED, WorkerState.FAILED)
        ]
        return statistics.mean(latencies) if latencies else 0

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        remaining = len(self.queued) + len(self.running)
        if remaining == 0:
            return 0

        avg_lat = self.avg_latency
        if avg_lat == 0:
            # No completed runs yet, estimate based on running count
            return remaining * 120  # Assume 2 min average

        # Factor in parallelism
        total_workers = sum(
            min(len(m), self.max_per_provider) for m in self.models_by_provider.values()
        )
        effective_parallelism = max(1, total_workers)

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

    async def _run_worker(self, worker_info: WorkerInfo) -> WorkerInfo:
        """Run a single worker with semaphore control."""
        provider = worker_info.provider
        semaphore = self._semaphores.get(provider)

        if semaphore is None:
            worker_info.state = WorkerState.FAILED
            worker_info.error = f"Unknown provider: {provider}"
            return worker_info

        # Set state to WAITING before semaphore acquisition
        worker_info.state = WorkerState.WAITING
        worker_info.start_time = time.time()

        async with semaphore:
            # Now actually running (have semaphore slot)
            worker_info.state = WorkerState.RUNNING

            try:
                worker = BenchmarkWorker(
                    model=worker_info.model,
                    provider=worker_info.provider,
                    task_path=worker_info.task_path,
                    output_dir=self.output_dir,
                )
                result = await worker.run()

                async with self._lock:
                    if result.state == WorkerState.COMPLETED and result.result:
                        bench_result = BenchmarkResult.from_dict(result.result)
                        self.completed.append(bench_result)
                    elif result.state == WorkerState.FAILED:
                        self.failed.append(result)

                return result

            except Exception as e:
                worker_info.state = WorkerState.FAILED
                worker_info.end_time = time.time()
                worker_info.error = f"Worker exception: {e}"
                async with self._lock:
                    self.failed.append(worker_info)
                return worker_info

    async def run(self, progress_callback: callable | None = None) -> dict:
        """Run all benchmarks in parallel.

        Args:
            progress_callback: Optional async callback for progress updates

        Returns:
            Dict with results summary
        """
        self._start_time = time.time()

        # Create tasks for all workers
        tasks = []
        for worker_info in self.workers:
            task = asyncio.create_task(self._run_worker(worker_info))
            tasks.append(task)

        # Wait for all tasks with periodic progress updates
        if progress_callback:
            done, pending = set(), set(tasks)
            while pending:
                # Wait for any task to complete or timeout
                done_subset, pending = await asyncio.wait(
                    pending,
                    timeout=0.5,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                done.update(done_subset)
                await progress_callback(self)
        else:
            await asyncio.gather(*tasks)

        # Collect final results
        results = {
            "total_runs": self.total_runs,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "results": [r.__dict__ for r in self.completed],
            "failed_workers": [
                {
                    "model": w.model,
                    "task_id": w.task_id,
                    "error": w.error,
                }
                for w in self.failed
            ],
            "duration_s": time.time() - self._start_time,
        }

        return results


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def display_progress(benchmark: ParallelBenchmark, verbose: bool = False) -> None:
    """Print the progress display."""
    clear_screen()

    # Header
    print("=" * 80)
    print("BENCHMARK PROGRESS")
    print("=" * 80)

    # Summary stats
    task_count = len(set(w.task_id for w in benchmark.workers))
    model_count = sum(len(m) for m in benchmark.models_by_provider.values())

    print(f"Tasks: {task_count} | Models: {model_count} | Total runs: {benchmark.total_runs}")
    print(
        f"Completed: {benchmark.completed_count} | Failed: {benchmark.failed_count} | "
        f"Running: {len(benchmark.running)} | Waiting: {len(benchmark.waiting)} | Queued: {len(benchmark.queued)}"
    )
    print(f"Progress: {benchmark.progress_pct:.1f}% | ETA: {benchmark.format_eta()}")
    print()

    # Running workers
    running = benchmark.running
    if running:
        print("RUNNING:")
        for w in sorted(running, key=lambda x: x.start_time or 0):
            elapsed = w.format_elapsed()
            print(f"  [{w.provider}] {w.model:25s} → {w.task_id:30s} ({elapsed})")
        print()

    # Waiting workers (waiting for semaphore slot)
    waiting = benchmark.waiting
    if waiting:
        print(f"WAITING ({len(waiting)} for semaphore slot):")
        for w in sorted(waiting, key=lambda x: x.start_time or 0)[:5]:  # Show max 5
            elapsed = w.format_elapsed()
            print(f"  [{w.provider}] {w.model:25s} → {w.task_id:30s} ({elapsed})")
        if len(waiting) > 5:
            print(f"  ... and {len(waiting) - 5} more")
        print()

    # Completed workers (show last 5)
    completed = [w for w in benchmark.workers if w.state == WorkerState.COMPLETED]
    if completed:
        print("COMPLETED:")
        for w in completed[-5:]:
            if w.result:
                passed = w.result.get("passed", False)
                symbol = "✓" if passed else "✗"
                token_usage = w.result.get("token_usage", {})
                inp = token_usage.get("input_tokens", 0)
                out = token_usage.get("output_tokens", 0)

                def abbrev(n: int) -> str:
                    if n >= 1_000_000:
                        return f"{n / 1_000_000:.1f}M"
                    elif n >= 1_000:
                        return f"{n / 1_000:.0f}K"
                    return str(n)

                latency = w.elapsed
                color_start = "\033[32m" if passed else "\033[31m"
                color_end = "\033[0m"
                print(
                    f"  {color_start}{symbol}{color_end} {w.model}: "
                    f"passed={passed}, {abbrev(inp)} in / {abbrev(out)} out, {latency:.1f}s"
                )
            else:
                print(f"  ✓ {w.model}: completed (no result data)")

    # Failed workers
    failed = [w for w in benchmark.workers if w.state == WorkerState.FAILED]
    if failed:
        print()
        print("FAILED:")
        for w in failed[-3:]:
            error_preview = (w.error or "Unknown error")[:60]
            print(f"  ✗ {w.model}: {error_preview}")

    print("=" * 80)


async def progress_updater(benchmark: ParallelBenchmark) -> None:
    """Async callback for progress updates."""
    display_progress(benchmark)


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
    print(f"Duration: {results.get('duration_s', 0):.1f}s")


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    # Resolve task paths
    task_paths = []
    for pattern in args.tasks:
        path = Path(pattern)
        if path.exists():
            task_paths.append(str(path))
        else:
            # Treat as glob pattern
            import glob as glob_module

            matched = glob_module.glob(pattern)
            task_paths.extend(matched)

    if not task_paths:
        print("Error: No task files found")
        return 1

    print(f"Found {len(task_paths)} task(s)")

    # Load models
    models_by_provider = load_models_from_config()

    # Filter models if specified
    if args.models and args.models != ["all"]:
        models_by_provider = filter_models(models_by_provider, args.models)

    if not models_by_provider:
        print("Error: No models found matching filter")
        return 1

    total_models = sum(len(m) for m in models_by_provider.values())
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
    )

    print(f"\nTotal benchmark runs: {benchmark.total_runs}")
    print(f"Max workers per provider: {args.max_per_provider}")
    print(f"Output directory: {output_dir}")
    print("\nStarting parallel benchmark...")

    # Run with progress display
    results = await benchmark.run(progress_callback=progress_updater)

    # Final display
    clear_screen()
    display_progress(benchmark)

    # Save results
    results_file = save_final_results(results, output_dir)
    print(f"\nResults saved to: {results_file}")

    # Print summary
    print_summary(results)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks in parallel with progress tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models on all mined tasks
  uv run python scripts/run_parallel_benchmark.py --tasks "tasks/mined/*.json" --models all

  # Run specific models on a single task
  uv run python scripts/run_parallel_benchmark.py \\
      --tasks tasks/mined/tidyverse_readr_1615.json \\
      --models glm-5-free minimax-m2.5-free

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
        required=True,
        help="Task file paths or glob patterns (e.g., 'tasks/mined/*.json')",
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

    args = parser.parse_args()

    # Run async main
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
