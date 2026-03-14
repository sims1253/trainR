#!/usr/bin/env python
"""Unified experiment runner CLI.

This is the canonical entry point for running experiments.
It produces deterministic outputs and proper artifact structure.

Usage:
    uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml

Output artifacts (in output_dir/run_id/):
    - results.jsonl: Individual task results (one JSON per line)
    - summary.json: Aggregated statistics
    - manifest.json: Run metadata, fingerprints, and configuration snapshot
    - matrix.json: Pre-computed experiment matrix
    - trajectories/: Generated code outputs (if enabled)
"""

import argparse
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import bench.runner
from bench.experiments import ExperimentConfig, load_experiment_config
from bench.schema.v1 import ManifestV1

console = Console()
logger = logging.getLogger(__name__)


def validate_config(config_path: Path) -> ExperimentConfig:
    """Validate and load experiment configuration."""
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    try:
        config = load_experiment_config(config_path)
        return config
    except Exception as e:
        console.print(f"[red]Invalid configuration: {e}[/red]")
        sys.exit(1)


def print_config_summary(config: ExperimentConfig) -> None:
    """Print a summary of the experiment configuration."""
    skill_name = config.skill.get_name()
    models = ", ".join(config.models.names) if config.models.names else "(none)"

    console.print(
        Panel(
            f"Name: {config.name}\n"
            f"Models: {models}\n"
            f"Tasks: {config.tasks.selection.value}\n"
            f"Skill: {skill_name}\n"
            f"Repeats: {config.execution.repeats}\n"
            f"Timeout: {config.execution.timeout}s\n"
            f"Workers: {config.execution.parallel_workers}\n"
            f"Provider Limits: "
            f"{config.execution.provider_parallel_limits or '(none)'}\n"
            f"Provider Min Interval: {config.execution.provider_min_interval_s or '(none)'}\n"
            "Provider RPS Cap: "
            f"{config.execution.provider_max_requests_per_second or '(none)'}\n"
            f"Seed: {config.determinism.seed or 'none'}",
            title="Experiment Configuration",
            border_style="blue",
        )
    )


def parse_provider_limits(raw_limits: list[str]) -> dict[str, int]:
    """Parse repeated --provider-limit PROVIDER=N options."""
    provider_aliases = {
        # Backward-compatible alias after introducing dedicated coding-plan provider.
        "zai": "zai_coding_plan",
    }
    parsed: dict[str, int] = {}
    for raw in raw_limits:
        if "=" not in raw:
            raise ValueError(f"Invalid provider limit '{raw}'. Expected PROVIDER=N.")
        provider, raw_limit = raw.split("=", 1)
        provider_name = provider.strip().lower()
        if not provider_name:
            raise ValueError(f"Invalid provider limit '{raw}'. Provider cannot be empty.")
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"Invalid provider limit '{raw}'. N must be an integer.") from exc
        if limit < 1:
            raise ValueError(f"Invalid provider limit '{raw}'. N must be >= 1.")
        canonical_provider = provider_aliases.get(provider_name, provider_name)
        parsed[canonical_provider] = limit
    return parsed


def parse_provider_float_limits(raw_limits: list[str], *, arg_name: str) -> dict[str, float]:
    """Parse repeated --<arg> PROVIDER=VALUE options where VALUE is float > 0."""
    provider_aliases = {
        # Backward-compatible alias after introducing dedicated coding-plan provider.
        "zai": "zai_coding_plan",
    }
    parsed: dict[str, float] = {}
    for raw in raw_limits:
        if "=" not in raw:
            raise ValueError(f"Invalid {arg_name} '{raw}'. Expected PROVIDER=VALUE.")
        provider, raw_value = raw.split("=", 1)
        provider_name = provider.strip().lower()
        if not provider_name:
            raise ValueError(f"Invalid {arg_name} '{raw}'. Provider cannot be empty.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid {arg_name} '{raw}'. VALUE must be numeric.") from exc
        if value <= 0:
            raise ValueError(f"Invalid {arg_name} '{raw}'. VALUE must be > 0.")
        canonical_provider = provider_aliases.get(provider_name, provider_name)
        parsed[canonical_provider] = value
    return parsed


def print_results_summary(manifest: ManifestV1, model_costs: dict[str, tuple[float, float]] | None = None) -> None:
    """Print a summary table of results."""
    table = Table(title="Experiment Results", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")

    total_cost = 0.0
    for model_summary in getattr(manifest, "model_summaries", []):
        pass_rate = model_summary.pass_rate
        color = "green" if pass_rate > 0.7 else "yellow" if pass_rate > 0.3 else "red"

        cost = 0.0
        if model_costs and model_summary.model in model_costs:
            in_rate, out_rate = model_costs[model_summary.model]
            # Approximate: split total_tokens 50/50 if we don't have the breakdown
            cost = model_summary.total_tokens * (in_rate + out_rate) / 2 / 1_000_000
        total_cost += cost

        table.add_row(
            model_summary.model,
            f"[{color}]{pass_rate:.1%}[/{color}]",
            str(model_summary.passed),
            str(model_summary.total),
            f"{model_summary.avg_latency_s:.1f}s",
            f"{model_summary.total_tokens:,}",
            _format_cost(cost),
        )

    console.print(table)

    # Overall summary
    cost_line = f"\nEst. Cost: {_format_cost(total_cost)}" if total_cost > 0 else ""
    console.print(
        Panel(
            f"Total: {manifest.summary.completed}/{manifest.summary.total_tasks} tasks\n"
            f"Pass Rate: {manifest.summary.pass_rate:.1%}\n"
            f"Avg Score: {manifest.summary.avg_score:.2f}\n"
            f"Avg Latency: {manifest.summary.avg_latency_s:.1f}s\n"
            f"Total Tokens: {manifest.summary.total_tokens:,}\n"
            + (
                f"Duration: {manifest.duration_s:.1f}s"
                if manifest.duration_s
                else "Duration: N/A"
            )
            + cost_line,
            title="Summary",
            border_style="green" if manifest.summary.pass_rate > 0.5 else "red",
        )
    )


def _format_cost(cost: float) -> str:
    """Format a cost value with adaptive precision.

    Returns '-' for zero, '$X.XX' for >= $0.01, '$0.00X' for smaller values.
    """
    if cost <= 0:
        return "-"
    if cost >= 0.01:
        return f"${cost:.2f}"
    if cost >= 0.001:
        return f"${cost:.3f}"
    return "<$0.01"


def _parse_cost(v: Any) -> float:
    """Parse a cost value from llm.yaml, handling 'NA' and other non-numeric values."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _load_model_costs() -> dict[str, tuple[float, float]]:
    """Load per-model cost rates from llm.yaml.

    Returns:
        Dict mapping model name -> (input_cost_per_M_tokens, output_cost_per_M_tokens).
    """
    try:
        from bench.provider.resolver import ProviderResolver

        resolver = ProviderResolver()
        costs: dict[str, tuple[float, float]] = {}
        for name, info in resolver.models.items():
            caps = info.capabilities
            if not isinstance(caps, dict):
                continue
            raw = caps.get("cost", [])
            # YAML `cost: [1 3.2 0.2 NA]` parses as ['1 3.2 0.2 NA'] (single string)
            # Split it into individual values if needed
            if len(raw) == 1 and isinstance(raw[0], str) and " " in raw[0]:
                raw = raw[0].split()
            input_cost = _parse_cost(raw[0]) if len(raw) > 0 else 0.0
            output_cost = _parse_cost(raw[1]) if len(raw) > 1 else 0.0
            costs[name] = (input_cost, output_cost)
        return costs
    except Exception:
        return {}


class ExperimentTUI:
    """Live experiment progress dashboard."""

    MAX_ACTIVITY_LINES = 6

    def __init__(self, console: Console, config: "ExperimentConfig | None" = None):
        self.console = console
        self.stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "completed": 0,
                "passed": 0,
                "tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": 0.0,
                "runtime": 0.0,
                "score": 0.0,
                "cost": 0.0,
            }
        )
        self.models: list[str] = []
        self.total_tasks = 0
        self.total_runs = 0
        self.model_totals: dict[str, int] = {}
        self.model_status: dict[str, str] = {}
        self.active_runs: dict[str, int] = defaultdict(int)
        self.sleeping_models: set[str] = set()
        self._lock = threading.Lock()
        self.manifest = None
        self.started_at_monotonic: float | None = None

        # Activity feed: last N completed results
        self.recent_results: list[dict] = []

        # Tool usage aggregation
        self.tool_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "time_ms": 0.0}
        )

        # Cost rates from llm.yaml
        self.model_costs = _load_model_costs()

        # Config metadata for header
        self.config_name = ""
        self.config_chips: str = ""

        if config:
            self.models = config.models.names or []
            self.model_status = dict.fromkeys(self.models, "waiting")
            self.config_name = config.name or ""

            n_models = len(self.models)
            try:
                skill_name = str(config.skill.get_name())
            except Exception:
                skill_name = "?"
            seed = config.determinism.seed or "none"
            workers = config.execution.parallel_workers
            try:
                sandbox = config.execution.sandbox_profile.value if hasattr(config.execution.sandbox_profile, "value") else str(config.execution.sandbox_profile)
            except Exception:
                sandbox = "?"
            chips = [
                f"{n_models} models",
                f"{workers} workers",
                f"seed:{seed}",
                str(skill_name),
                str(sandbox),
            ]
            self.config_chips = "  ".join(chips)

            # Try to get task count from config
            files = getattr(config.tasks, "files", None)
            if files:
                self.total_tasks = len(files)
            elif hasattr(config.tasks, "selection") and config.tasks.selection:
                sel = config.tasks.selection
                if hasattr(sel, "files") and sel.files:
                    self.total_tasks = len(sel.files)
                elif isinstance(sel, dict) and "files" in sel:
                    self.total_tasks = len(sel["files"])

    def callback(self, event: dict) -> None:
        """Progress callback for the runner."""
        with self._lock:
            etype = event.get("type")
            if etype == "start":
                self.models = event.get("models", [])
                self.total_tasks = event.get("task_count", 0)
                self.total_runs = event.get("total_runs", 0)
                self.model_totals = event.get("model_totals", {})
                self.model_status = dict.fromkeys(self.models, "waiting")
                self.active_runs = defaultdict(int)
                self.sleeping_models.clear()
                self.started_at_monotonic = time.monotonic()
                # Update chips with task count now that we know it
                if self.config_chips:
                    self.config_chips = (
                        f"{len(self.models)} models  "
                        f"{self.total_tasks} tasks  "
                        + "  ".join(self.config_chips.split("  ")[1:])
                    )
            elif etype == "run_start":
                model = event.get("model", "unknown")
                self.active_runs[model] += 1
                if model not in self.sleeping_models:
                    self.model_status[model] = "running"
            elif etype == "model_sleep":
                model = event.get("model", "unknown")
                self.sleeping_models.add(model)
                self.model_status[model] = "sleeping"
            elif etype == "model_resume":
                model = event.get("model", "unknown")
                self.sleeping_models.discard(model)
                completed = self.stats.get(model, {}).get("completed", 0)
                target = self.model_totals.get(model, self.total_tasks)
                if target > 0 and completed >= target:
                    self.model_status[model] = "done"
                elif self.active_runs.get(model, 0) > 0:
                    self.model_status[model] = "running"
                else:
                    self.model_status[model] = "waiting"
            elif etype == "result":
                model = event.get("model", "unknown")
                if self.active_runs.get(model, 0) > 0:
                    self.active_runs[model] -= 1
                s = self.stats[model]
                s["completed"] += 1
                if event.get("passed"):
                    s["passed"] += 1
                s["tokens"] += event.get("tokens", 0)
                s["latency"] += event.get("latency_s", 0)
                s["runtime"] += event.get("runtime_s", event.get("latency_s", 0))
                s["score"] += event.get("score", 0.0)

                # Token breakdown for cost
                prompt_tok = event.get("prompt_tokens", 0)
                comp_tok = event.get("completion_tokens", 0)
                total_tok = event.get("tokens", 0)
                s["prompt_tokens"] += prompt_tok
                s["completion_tokens"] += comp_tok

                # Compute cost for this result
                in_rate, out_rate = self.model_costs.get(model, (0.0, 0.0))
                if prompt_tok > 0 or comp_tok > 0:
                    result_cost = prompt_tok * in_rate / 1_000_000 + comp_tok * out_rate / 1_000_000
                elif total_tok > 0:
                    # Fallback: use average of input/output rates
                    result_cost = total_tok * (in_rate + out_rate) / 2 / 1_000_000
                else:
                    result_cost = 0.0
                s["cost"] += result_cost

                # Merge tool stats
                for tool, count in event.get("tool_call_counts", {}).items():
                    self.tool_stats[tool]["calls"] += count
                for tool, errs in event.get("tool_errors", {}).items():
                    self.tool_stats[tool]["errors"] += errs
                for tool, ms in event.get("tool_total_time_ms", {}).items():
                    self.tool_stats[tool]["time_ms"] += ms

                # Activity feed
                self.recent_results.append({
                    "model": model,
                    "task_id": event.get("task_id", "?"),
                    "passed": event.get("passed", False),
                    "score": event.get("score", 0.0),
                    "cost": result_cost,
                    "runtime_s": event.get("runtime_s", event.get("latency_s", 0)),
                    "tokens": event.get("tokens", 0),
                    "error_category": event.get("error_category"),
                    "error_message": event.get("error_message"),
                    "tests_passed": event.get("tests_passed", 0),
                    "tests_total": event.get("tests_total", 0),
                })
                if len(self.recent_results) > self.MAX_ACTIVITY_LINES:
                    self.recent_results = self.recent_results[-self.MAX_ACTIVITY_LINES:]

                # Update model status
                completed = s["completed"]
                target = self.model_totals.get(model, self.total_tasks)
                if target > 0 and completed >= target:
                    self.model_status[model] = "done"
                    self.sleeping_models.discard(model)
                elif model in self.sleeping_models:
                    self.model_status[model] = "sleeping"
                elif self.active_runs.get(model, 0) > 0:
                    self.model_status[model] = "running"
                else:
                    self.model_status[model] = "waiting"

    def format_tokens(self, tokens: int | float) -> str:
        tokens = int(tokens)
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.0f}K"
        return str(tokens)

    def format_duration(self, seconds: int | float) -> str:
        total_seconds = float(seconds)
        if total_seconds >= 3600:
            return f"{total_seconds / 3600:.1f}h"
        if total_seconds >= 60:
            return f"{total_seconds / 60:.1f}m"
        return f"{total_seconds:.1f}s"

    def _elapsed_str(self) -> str:
        if self.started_at_monotonic is None:
            return "0s elapsed"
        elapsed = time.monotonic() - self.started_at_monotonic
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s elapsed"
        return f"{m}m {s:02d}s elapsed"

    def _failure_reason(self, r: dict) -> str:
        """Build a compact failure reason string from a result dict."""
        parts: list[str] = []
        err_cat = r.get("error_category")
        if err_cat:
            # Shorten common categories for display
            short = {
                "TEST_FAILURE": "test fail",
                "SYNTAX_ERROR": "syntax",
                "TIMEOUT": "timeout",
                "MISSING_IMPORT": "missing import",
                "WRONG_ASSERTION": "wrong assert",
                "INCOMPLETE_SOLUTION": "incomplete",
                "OVERLY_COMPLEX": "too complex",
                "WRONG_FIXTURE_USAGE": "bad fixture",
                "SNAPSHOT_MISMATCH": "snapshot",
                "LLM_ERROR": "llm error",
                "SANDBOX_ERROR": "sandbox",
                "AUTH_ERROR": "auth error",
            }.get(err_cat, err_cat.lower().replace("_", " "))
            parts.append(short)

        tests_passed = r.get("tests_passed", 0)
        tests_total = r.get("tests_total", 0)
        if tests_total > 0:
            parts.append(f"{tests_passed}/{tests_total} tests")

        err_msg = r.get("error_message")
        if err_msg and not parts:
            # Truncate long messages
            msg = err_msg.strip().split("\n")[0]
            if len(msg) > 40:
                msg = msg[:37] + "..."
            parts.append(msg)

        return " | ".join(parts) if parts else ""

    def build_header(self) -> Panel:
        """Build the header panel with experiment name, elapsed time, and config chips."""
        body = Text(self.config_chips or "Initializing...", style="dim")
        return Panel(
            body,
            title=f"trainR: {self.config_name}" if self.config_name else "trainR",
            subtitle=self._elapsed_str(),
            subtitle_align="right",
            border_style="blue",
        )

    def build_progress_bar(self) -> Panel:
        """Build the overall progress bar panel."""
        total_completed = sum(
            self.stats.get(m, {}).get("completed", 0) for m in self.models
        )
        total_expected = (
            sum(self.model_totals.values())
            if self.model_totals
            else len(self.models) * self.total_tasks
        )
        pct = total_completed / total_expected if total_expected > 0 else 0
        bar_width = 40
        filled = int(bar_width * pct)
        # Color transitions: blue at start, green when mostly done
        bar_color = "green" if pct > 0.7 else "blue"
        bar_str = (
            f"[{bar_color}]{'█' * filled}[/{bar_color}]"
            f"[dim]{'─' * (bar_width - filled)}[/dim]"
            f" {total_completed}/{total_expected} runs   {int(pct * 100)}%"
        )
        return Panel(
            Text.from_markup(f"Overall {bar_str}"),
            border_style="dim",
        )

    def build_table(self) -> Table:
        """Build the model progress table."""
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Model", width=22)
        table.add_column("St", width=4)
        table.add_column("Progress", width=20)
        table.add_column("Pass", width=8, justify="right")
        table.add_column("Rate", width=6, justify="right")
        table.add_column("Score", width=6, justify="right")
        table.add_column("$Cost", width=7, justify="right")
        table.add_column("Time", width=6, justify="right")
        table.add_column("Tok", width=6, justify="right")

        total_completed = 0
        total_passed = 0
        total_score = 0.0
        total_runtime = 0.0
        total_tokens = 0
        total_cost = 0.0

        status_map = {
            "waiting": ("---", "dim"),
            "running": ("RUN", "cyan"),
            "sleeping": ("SLP", "magenta"),
            "done": ("OK", "green"),
        }

        for model in self.models:
            s = self.stats.get(
                model,
                {
                    "completed": 0, "passed": 0, "tokens": 0,
                    "latency": 0.0, "runtime": 0.0, "score": 0.0, "cost": 0.0,
                },
            )
            completed = s["completed"]
            passed = s["passed"]
            tokens = s["tokens"]
            runtime = s["runtime"]
            score_sum = s["score"]
            cost = s.get("cost", 0.0)
            target = self.model_totals.get(model, self.total_tasks)

            total_completed += completed
            total_passed += passed
            total_score += score_sum
            total_runtime += runtime
            total_tokens += tokens
            total_cost += cost

            # Progress bar with half-block chars
            pct = completed / target if target > 0 else 0
            bar_width = 12
            filled_full = int(bar_width * pct)
            half = (bar_width * pct) - filled_full >= 0.5
            bar_chars = "█" * filled_full + ("▌" if half else "") + "░" * (bar_width - filled_full - (1 if half else 0))
            bar = f"[green]{bar_chars}[/green] {int(pct * 100):3d}%"

            status = self.model_status.get(model, "waiting")
            label, color = status_map.get(status, ("---", "dim"))
            status_display = f"[{color}]{label}[/{color}]"

            rate = f"{100 * passed / completed:.0f}%" if completed > 0 else "-"
            avg_score = f"{score_sum / completed:.2f}" if completed > 0 else "-"
            cost_str = _format_cost(cost)
            model_display = model if len(model) <= 22 else model[:19] + "..."

            table.add_row(
                model_display,
                status_display,
                bar,
                f"{passed}/{completed}",
                rate,
                avg_score,
                cost_str,
                self.format_duration(runtime),
                self.format_tokens(tokens),
            )

        # TOTAL row
        overall_rate = (
            f"{100 * total_passed / total_completed:.1f}%" if total_completed > 0 else "-"
        )
        overall_score = f"{total_score / total_completed:.2f}" if total_completed > 0 else "-"
        total_cost_str = _format_cost(total_cost)
        table.add_section()
        total_expected = (
            sum(self.model_totals.values())
            if self.model_totals
            else len(self.models) * self.total_tasks
        )
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            f"[bold]{total_completed}/{total_expected}[/bold]",
            f"[bold]{total_passed}[/bold]",
            f"[bold]{overall_rate}[/bold]",
            f"[bold]{overall_score}[/bold]",
            f"[bold]{total_cost_str}[/bold]",
            f"[bold]{self.format_duration(total_runtime)}[/bold]",
            f"[bold]{self.format_tokens(total_tokens)}[/bold]",
        )

        return table

    def build_tools_strip(self) -> Panel | None:
        """Build compact tool usage strip. Returns None if no tool data yet."""
        if not self.tool_stats:
            return None
        # Top 6 tools by call count
        sorted_tools = sorted(
            self.tool_stats.items(), key=lambda kv: kv[1]["calls"], reverse=True
        )[:6]
        parts = [f"{name}:{int(info['calls'])}" for name, info in sorted_tools]
        return Panel(
            Text.from_markup(f"[bold]Tools[/bold]  {'  '.join(parts)}"),
            border_style="dim",
        )

    def build_activity_panel(self) -> Panel:
        """Build the activity feed panel showing recent completions."""
        lines: list[Text] = []
        with self._lock:
            for r in self.recent_results:
                passed = r.get("passed", False)
                badge_style = "bold green" if passed else "bold red"
                badge = "PASS" if passed else "FAIL"
                model = r.get("model", "?")
                if len(model) > 14:
                    model = model[:11] + "..."
                task_id = r.get("task_id", "?")
                if len(task_id) > 14:
                    task_id = task_id[:14]
                score = r.get("score", 0.0)
                cost = r.get("cost", 0.0)
                runtime = r.get("runtime_s", 0)
                tokens = r.get("tokens", 0)

                line = Text()
                line.append(f" {badge} ", style=badge_style)
                line.append(f" {model:<14s}", style="")
                line.append(f" {task_id:<14s}", style="dim")
                line.append(f" {score:.2f}", style="")
                line.append(f"  {self.format_duration(runtime):>5s}", style="dim")
                cost_str = _format_cost(cost)
                if cost_str == "-":
                    line.append("       -", style="dim")
                else:
                    line.append(f"  {cost_str:>6s}", style="")
                line.append(f"  {self.format_tokens(tokens):>5s}", style="dim")

                # Show failure reason for FAIL results
                if not passed:
                    reason = self._failure_reason(r)
                    if reason:
                        line.append(f"  {reason}", style="red")

                lines.append(line)

            # Show currently running tasks in dim at bottom
            running = [
                m for m in self.models
                if self.active_runs.get(m, 0) > 0
            ]
            if running:
                run_line = Text()
                run_line.append(" ...  ", style="dim")
                run_line.append(
                    ", ".join(running[:3]) + (" ..." if len(running) > 3 else ""),
                    style="dim italic",
                )
                run_line.append("  running", style="dim italic")
                lines.append(run_line)

        if not lines:
            return Panel("[dim]Waiting for results...[/dim]", title="Activity", border_style="dim")

        from rich.console import Group

        return Panel(Group(*lines), title="Activity", border_style="dim")

    def build_layout(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()

        # Calculate activity panel size
        activity_size = min(len(self.recent_results), self.MAX_ACTIVITY_LINES) + 2
        # Add 1 for running line if there are active runs
        if any(self.active_runs.get(m, 0) > 0 for m in self.models):
            activity_size += 1
        activity_size = max(activity_size, 4)  # minimum 4 lines

        tools_panel = self.build_tools_strip()
        if tools_panel:
            layout.split(
                Layout(name="header", size=4),
                Layout(name="progress", size=3),
                Layout(name="table", ratio=1),
                Layout(name="tools", size=3),
                Layout(name="activity", size=activity_size),
            )
            layout["tools"].update(tools_panel)
        else:
            layout.split(
                Layout(name="header", size=4),
                Layout(name="progress", size=3),
                Layout(name="table", ratio=1),
                Layout(name="activity", size=activity_size),
            )

        layout["header"].update(self.build_header())
        layout["progress"].update(self.build_progress_bar())
        layout["table"].update(self.build_table())
        layout["activity"].update(self.build_activity_panel())
        return layout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments with unified configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run smoke test
    uv run python scripts/run_experiment.py --config configs/experiments/r_bench_smoke.yaml

    # Run with custom output directory
    uv run python scripts/run_experiment.py --config my_config.yaml --output-dir results/my_run

    # Run with specific seed for reproducibility
    uv run python scripts/run_experiment.py --config my_config.yaml --seed 42

    # Validate config without running
    uv run python scripts/run_experiment.py --config my_config.yaml --validate
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Override random seed for reproducibility",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Override number of parallel workers",
    )
    parser.add_argument(
        "--provider-limit",
        action="append",
        default=[],
        metavar="PROVIDER=N",
        help=(
            "Cap concurrent runs for a provider. Repeatable, e.g. "
            "--provider-limit openrouter=1 --provider-limit zai=2"
        ),
    )
    parser.add_argument(
        "--provider-min-interval",
        action="append",
        default=[],
        metavar="PROVIDER=SECONDS",
        help=(
            "Minimum spacing between run starts for a provider. Repeatable, e.g. "
            "--provider-min-interval zai=1.0"
        ),
    )
    parser.add_argument(
        "--provider-rps",
        action="append",
        default=[],
        metavar="PROVIDER=RPS",
        help=(
            "Maximum run starts per second for a provider. Repeatable, e.g. "
            "--provider-rps zai=0.5"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration without running",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment matrix without running",
    )
    parser.add_argument(
        "--save-container-logs",
        action="store_true",
        help="Persist raw Docker container stdout/stderr logs per run",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Load and validate config
    config_path = Path(args.config)
    config = validate_config(config_path)

    # Build override kwargs early so we can show actual values in summary.
    run_kwargs: dict[str, Any] = {}
    if args.output_dir:
        run_kwargs["output_dir"] = args.output_dir
    if args.seed is not None:
        run_kwargs["seed"] = args.seed
    if args.workers is not None:
        run_kwargs["workers"] = args.workers
    if args.provider_limit:
        try:
            run_kwargs["provider_parallel_limits"] = parse_provider_limits(args.provider_limit)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(2)
    if args.provider_min_interval:
        try:
            run_kwargs["provider_min_interval_s"] = parse_provider_float_limits(
                args.provider_min_interval,
                arg_name="provider-min-interval",
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(2)
    if args.provider_rps:
        try:
            run_kwargs["provider_max_requests_per_second"] = parse_provider_float_limits(
                args.provider_rps,
                arg_name="provider-rps",
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(2)
    if args.save_container_logs:
        run_kwargs["save_container_logs"] = True

    # Apply overrides for display config
    display_config = config.model_copy(deep=True)
    if run_kwargs.get("workers") is not None:
        display_config.execution.parallel_workers = run_kwargs["workers"]
    if run_kwargs.get("provider_parallel_limits"):
        display_config.execution.provider_parallel_limits = run_kwargs["provider_parallel_limits"]
    if run_kwargs.get("provider_min_interval_s"):
        display_config.execution.provider_min_interval_s = run_kwargs["provider_min_interval_s"]
    if run_kwargs.get("provider_max_requests_per_second"):
        display_config.execution.provider_max_requests_per_second = run_kwargs[
            "provider_max_requests_per_second"
        ]
    if run_kwargs.get("seed") is not None:
        display_config.determinism.seed = run_kwargs["seed"]

    # Validate only
    if args.validate:
        print_config_summary(display_config)
        bench.runner.run(config, validate_only=True, **run_kwargs)
        console.print("[green]Configuration is valid[/green]")
        return

    # Dry run
    if args.dry_run:
        print_config_summary(display_config)
        manifest = bench.runner.run(config, dry_run=True, **run_kwargs)
        console.print("\n[blue]Experiment Matrix[/blue]")
        console.print(f"  Tasks: {manifest.task_count}")
        console.print(f"  Models: {len(manifest.models)}")
        console.print(f"  Total runs: {manifest.config.get('total_runs', 'N/A')}")
        if manifest.config.get("task_ids"):
            console.print(f"\n[dim]Task IDs: {manifest.config['task_ids'][:5]}...[/dim]")
        console.print(f"[dim]Models: {manifest.models}[/dim]")
        return

    # --- Live experiment run ---
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    live_console = Console(
        file=original_stdout,
        force_terminal=bool(getattr(original_stdout, "isatty", lambda: False)()),
    )

    tui = ExperimentTUI(live_console, config=display_config)

    manifest = None
    experiment_error = None
    experiment_done = threading.Event()

    def _bg_runner():
        nonlocal manifest, experiment_error
        try:
            manifest = bench.runner.run(
                config,
                progress_callback=tui.callback,
                **run_kwargs,
            )
        except Exception as e:
            experiment_error = e
        finally:
            experiment_done.set()

    experiment_thread = threading.Thread(target=_bg_runner, daemon=True)
    experiment_thread.start()

    # Redirect stdout/stderr to /dev/null during Live to suppress noise
    devnull = open(os.devnull, "w")  # noqa: SIM115
    sys.stdout = devnull
    sys.stderr = devnull
    # Silence loggers during Live display
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    root_logger.handlers = []
    root_logger.setLevel(logging.CRITICAL)

    live_failed = False
    try:
        with Live(
            tui.build_layout(),
            console=live_console,
            refresh_per_second=2,
            screen=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            while not experiment_done.is_set():
                try:
                    live.update(tui.build_layout())
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break
            live.update(tui.build_layout())
    except Exception:
        live_failed = True

    # Restore streams and logging
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    devnull.close()
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)

    if live_failed:
        console.print(
            "[yellow]Live dashboard failed; showing final summary when done.[/yellow]"
        )

    # Wait for thread to finish
    try:
        experiment_thread.join(timeout=5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user[/yellow]")
        sys.exit(130)

    # Handle results/errors
    if experiment_error:
        console.print(f"\n[red]Experiment failed: {experiment_error}[/red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    if manifest is None:
        console.print("\n[red]Experiment produced no results[/red]")
        sys.exit(1)

    # Print final summary with cost data
    try:
        print_results_summary(manifest, model_costs=tui.model_costs)
    except (AttributeError, TypeError):
        # Manifest may lack expected attributes (e.g. in test mocks)
        console.print("[dim]Results summary unavailable[/dim]")

    # Print output location
    output_dir = Path(manifest.results_path).parent if manifest.results_path else None
    if output_dir:
        console.print("\n[dim]Output artifacts:[/dim]")
        console.print(f"[dim]  - {output_dir}/manifest.json[/dim]")
        console.print(f"[dim]  - {output_dir}/results.jsonl[/dim]")
        console.print(f"[dim]  - {output_dir}/summary.json[/dim]")
        console.print(f"[dim]  - {output_dir}/matrix.json[/dim]")


if __name__ == "__main__":
    main()
