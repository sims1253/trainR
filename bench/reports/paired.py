"""Paired experiment analysis for A/B tool comparisons.

This module provides analysis tools for paired experiments where
only the tool profile differs while task/model/support/seed are held constant.

Key features:
- Paired tool deltas (pass rate, cost, latency)
- Join key verification (proves only tool differs)
- Bootstrap significance statistics
- Confidence intervals for deltas
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ToolPairDelta:
    """
    Delta metrics for a single tool A/B pair.

    Represents the difference between treatment and control for a single pair.
    All join keys must match to ensure only tool differs.
    """

    pair_id: str
    control_tool: str  # Full tool ID (e.g., "r-eval@v1")
    treatment_tool: str  # Full tool ID (e.g., "r-eval@v1:strict")

    # Join keys (proving only tool differs)
    join_key_task: str
    join_key_model: str
    join_key_support: str
    join_key_seed: str

    # Delta metrics
    delta_pass_rate: float = 0.0  # treatment - control (1 = pass, 0 = fail)
    delta_cost: float = 0.0  # treatment - control (cost in tokens or $)
    delta_latency_s: float = 0.0  # treatment - control (latency in seconds)

    # Raw values for reference
    control_pass: bool = False
    treatment_pass: bool = False
    control_cost: float = 0.0
    treatment_cost: float = 0.0
    control_latency_s: float = 0.0
    treatment_latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair_id": self.pair_id,
            "control_tool": self.control_tool,
            "treatment_tool": self.treatment_tool,
            "join_keys": {
                "task": self.join_key_task,
                "model": self.join_key_model,
                "support": self.join_key_support,
                "seed": self.join_key_seed,
            },
            "deltas": {
                "pass_rate": self.delta_pass_rate,
                "cost": self.delta_cost,
                "latency_s": self.delta_latency_s,
            },
            "raw_values": {
                "control": {
                    "pass": self.control_pass,
                    "cost": self.control_cost,
                    "latency_s": self.control_latency_s,
                },
                "treatment": {
                    "pass": self.treatment_pass,
                    "cost": self.treatment_cost,
                    "latency_s": self.treatment_latency_s,
                },
            },
        }

    def verify_join_keys(self) -> bool:
        """Verify that all join keys are set and match expected format."""
        return all(
            [
                self.join_key_task,
                self.join_key_model,
                self.join_key_support,
                self.join_key_seed,
            ]
        )


@dataclass
class BootstrapStats:
    """Bootstrap statistics for a metric delta."""

    mean: float = 0.0
    std_error: float = 0.0
    ci_lower_95: float = 0.0
    ci_upper_95: float = 0.0
    p_value_two_sided: float = 1.0
    bootstrap_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": round(self.mean, 6),
            "std_error": round(self.std_error, 6),
            "ci_95": [round(self.ci_lower_95, 6), round(self.ci_upper_95, 6)],
            "p_value_two_sided": round(self.p_value_two_sided, 6),
            "bootstrap_samples": self.bootstrap_samples,
        }

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the result is statistically significant."""
        return self.p_value_two_sided < alpha


@dataclass
class PairedToolReport:
    """
    Complete report for a paired tool A/B comparison.

    Aggregates all pair deltas and provides bootstrap statistics
    for significance testing.
    """

    control_tool: str
    treatment_tool: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # All pair deltas
    pair_deltas: list[ToolPairDelta] = field(default_factory=list)

    # Aggregated deltas
    mean_delta_pass_rate: float = 0.0
    mean_delta_cost: float = 0.0
    mean_delta_latency_s: float = 0.0

    # Bootstrap statistics
    pass_rate_stats: BootstrapStats = field(default_factory=BootstrapStats)
    cost_stats: BootstrapStats = field(default_factory=BootstrapStats)
    latency_stats: BootstrapStats = field(default_factory=BootstrapStats)

    # Summary counts
    total_pairs: int = 0
    control_wins: int = 0  # Pairs where control outperformed treatment
    treatment_wins: int = 0  # Pairs where treatment outperformed control
    ties: int = 0  # Pairs with no difference

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "control_tool": self.control_tool,
            "treatment_tool": self.treatment_tool,
            "created_at": self.created_at,
            "total_pairs": self.total_pairs,
            "wins": {
                "control": self.control_wins,
                "treatment": self.treatment_wins,
                "ties": self.ties,
            },
            "mean_deltas": {
                "pass_rate": round(self.mean_delta_pass_rate, 4),
                "cost": round(self.mean_delta_cost, 4),
                "latency_s": round(self.mean_delta_latency_s, 4),
            },
            "bootstrap_stats": {
                "pass_rate": self.pass_rate_stats.to_dict(),
                "cost": self.cost_stats.to_dict(),
                "latency_s": self.latency_stats.to_dict(),
            },
            "pair_deltas": [d.to_dict() for d in self.pair_deltas],
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


def compute_paired_deltas(
    control_results: list[dict[str, Any]],
    treatment_results: list[dict[str, Any]],
    join_keys: list[str] = None,
) -> list[ToolPairDelta]:
    """
    Compute paired deltas between control and treatment results.

    Results are matched by pair_id. Join keys are verified to ensure
    only tool differs between paired results.

    Args:
        control_results: List of control run results
        treatment_results: List of treatment run results
        join_keys: Keys to verify match (default: task, model, support, seed)

    Returns:
        List of ToolPairDelta instances
    """
    if join_keys is None:
        join_keys = ["task", "model", "support", "seed"]

    # Index by pair_id
    control_by_pair = {r.get("pair_id"): r for r in control_results if r.get("pair_id")}
    treatment_by_pair = {r.get("pair_id"): r for r in treatment_results if r.get("pair_id")}

    deltas: list[ToolPairDelta] = []

    for pair_id, control in control_by_pair.items():
        treatment = treatment_by_pair.get(pair_id)
        if not treatment:
            continue

        # Extract join keys
        join_key_task = _extract_join_key(control, "task")
        join_key_model = _extract_join_key(control, "model")
        join_key_support = _extract_join_key(control, "support")
        join_key_seed = str(control.get("seed", ""))

        # Verify join keys match
        if not _verify_join_keys_match(control, treatment, join_keys):
            continue

        # Extract metrics
        control_pass = bool(control.get("passed", control.get("pass_rate", 0) > 0.5))
        treatment_pass = bool(treatment.get("passed", treatment.get("pass_rate", 0) > 0.5))

        control_cost = float(control.get("cost", control.get("total_tokens", 0)))
        treatment_cost = float(treatment.get("cost", treatment.get("total_tokens", 0)))

        control_latency = float(control.get("latency_s", control.get("duration_s", 0)))
        treatment_latency = float(treatment.get("latency_s", treatment.get("duration_s", 0)))

        # Compute deltas
        delta = ToolPairDelta(
            pair_id=pair_id,
            control_tool=control.get("tool_profile", control.get("tool", "unknown")),
            treatment_tool=treatment.get("tool_profile", treatment.get("tool", "unknown")),
            join_key_task=join_key_task,
            join_key_model=join_key_model,
            join_key_support=join_key_support,
            join_key_seed=join_key_seed,
            delta_pass_rate=float(treatment_pass) - float(control_pass),
            delta_cost=treatment_cost - control_cost,
            delta_latency_s=treatment_latency - control_latency,
            control_pass=control_pass,
            treatment_pass=treatment_pass,
            control_cost=control_cost,
            treatment_cost=treatment_cost,
            control_latency_s=control_latency,
            treatment_latency_s=treatment_latency,
        )

        deltas.append(delta)

    return deltas


def _extract_join_key(result: dict[str, Any], key: str) -> str:
    """Extract a join key from a result dict."""
    # Try nested structure first
    if key in result.get("join_keys", {}):
        return str(result["join_keys"][key])

    # Try direct key
    if key in result:
        val = result[key]
        if isinstance(val, dict):
            return val.get("task_id", val.get("name", val.get("profile_id", str(val))))
        return str(val)

    # Try nested task/model objects
    for prefix in ["task", "model", "support"]:
        obj = result.get(prefix, {})
        if isinstance(obj, dict):
            if key in obj:
                return str(obj[key])
            if f"{prefix}_id" in obj:
                return str(obj[f"{prefix}_id"])

    return ""


def _verify_join_keys_match(
    control: dict[str, Any], treatment: dict[str, Any], join_keys: list[str]
) -> bool:
    """Verify that join keys match between control and treatment."""
    for key in join_keys:
        control_val = _extract_join_key(control, key)
        treatment_val = _extract_join_key(treatment, key)
        if control_val != treatment_val:
            return False
    return True


def bootstrap_significance_test(
    deltas: list[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapStats:
    """
    Perform bootstrap significance test on a list of deltas.

    Uses bootstrap resampling to estimate:
    - Mean and standard error
    - Confidence intervals
    - Two-sided p-value (proportion of bootstrap means with opposite sign)

    Args:
        deltas: List of delta values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI (default 0.95)
        seed: Random seed for reproducibility

    Returns:
        BootstrapStats with computed statistics
    """
    if not deltas:
        return BootstrapStats(bootstrap_samples=0)

    if seed is not None:
        random.seed(seed)

    n = len(deltas)
    alpha = 1 - confidence_level

    # Bootstrap resampling
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = random.choices(deltas, k=n)
        bootstrap_means.append(sum(sample) / n)

    # Sort for CI calculation
    bootstrap_means.sort()

    # Compute statistics
    mean = sum(deltas) / n
    std_error = (sum((m - mean) ** 2 for m in bootstrap_means) / n_bootstrap) ** 0.5

    # Confidence interval (percentile method)
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[upper_idx]

    # P-value: proportion of bootstrap means with opposite sign to observed
    # Using two-sided test
    if mean >= 0:
        p_value = 2 * sum(1 for m in bootstrap_means if m <= 0) / n_bootstrap
    else:
        p_value = 2 * sum(1 for m in bootstrap_means if m >= 0) / n_bootstrap

    # Cap p-value at 1.0
    p_value = min(p_value, 1.0)

    return BootstrapStats(
        mean=mean,
        std_error=std_error,
        ci_lower_95=ci_lower,
        ci_upper_95=ci_upper,
        p_value_two_sided=p_value,
        bootstrap_samples=n_bootstrap,
    )


def generate_paired_tool_report(
    pair_deltas: list[ToolPairDelta],
    control_tool: str | None = None,
    treatment_tool: str | None = None,
    bootstrap_samples: int = 10000,
    seed: int | None = None,
) -> PairedToolReport:
    """
    Generate a complete paired tool comparison report.

    Args:
        pair_deltas: List of ToolPairDelta instances
        control_tool: Control tool identifier (extracted from deltas if not provided)
        treatment_tool: Treatment tool identifier (extracted from deltas if not provided)
        bootstrap_samples: Number of bootstrap samples for significance testing
        seed: Random seed for reproducibility

    Returns:
        PairedToolReport with aggregated metrics and bootstrap statistics
    """
    if not pair_deltas:
        return PairedToolReport(
            control_tool=control_tool or "unknown",
            treatment_tool=treatment_tool or "unknown",
            total_pairs=0,
        )

    # Extract tool names from first delta if not provided
    if not control_tool:
        control_tool = pair_deltas[0].control_tool
    if not treatment_tool:
        treatment_tool = pair_deltas[0].treatment_tool

    # Compute aggregated deltas
    pass_deltas = [d.delta_pass_rate for d in pair_deltas]
    cost_deltas = [d.delta_cost for d in pair_deltas]
    latency_deltas = [d.delta_latency_s for d in pair_deltas]

    mean_pass = sum(pass_deltas) / len(pass_deltas)
    mean_cost = sum(cost_deltas) / len(cost_deltas)
    mean_latency = sum(latency_deltas) / len(latency_deltas)

    # Count wins/ties
    control_wins = sum(1 for d in pass_deltas if d < 0)
    treatment_wins = sum(1 for d in pass_deltas if d > 0)
    ties = sum(1 for d in pass_deltas if d == 0)

    # Bootstrap statistics
    pass_stats = bootstrap_significance_test(pass_deltas, bootstrap_samples, seed=seed)
    cost_stats = bootstrap_significance_test(cost_deltas, bootstrap_samples, seed=seed)
    latency_stats = bootstrap_significance_test(latency_deltas, bootstrap_samples, seed=seed)

    return PairedToolReport(
        control_tool=control_tool,
        treatment_tool=treatment_tool,
        pair_deltas=pair_deltas,
        mean_delta_pass_rate=mean_pass,
        mean_delta_cost=mean_cost,
        mean_delta_latency_s=mean_latency,
        pass_rate_stats=pass_stats,
        cost_stats=cost_stats,
        latency_stats=latency_stats,
        total_pairs=len(pair_deltas),
        control_wins=control_wins,
        treatment_wins=treatment_wins,
        ties=ties,
    )


def analyze_tool_ab_matrix(matrix_path: str | Path) -> PairedToolReport:
    """
    Analyze a ToolABMatrix from a saved JSON file.

    This is a convenience function that loads a matrix file and
    generates a paired report (simulated for planning purposes).

    Args:
        matrix_path: Path to the ToolABMatrix JSON file

    Returns:
        PairedToolReport with analysis results
    """
    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    with open(matrix_path) as f:
        data = json.load(f)

    # Extract pairs from matrix
    pairs = data.get("pairs", [])
    if not pairs:
        # Try to extract from runs
        runs = data.get("runs", [])
        pairs = _group_runs_into_pairs(runs)

    # Create simulated deltas (for demonstration - real implementation
    # would use actual run results)
    pair_deltas: list[ToolPairDelta] = []

    for pair in pairs[:10]:  # Limit for demo
        control = pair.get("control_run", {})
        treatment = pair.get("treatment_run", {})

        join_keys = pair.get("join_key_summary", {})

        delta = ToolPairDelta(
            pair_id=pair.get("pair_id", "unknown"),
            control_tool=join_keys.get("control_tool", "unknown"),
            treatment_tool=join_keys.get("treatment_tool", "unknown"),
            join_key_task=join_keys.get("task", ""),
            join_key_model=join_keys.get("model", ""),
            join_key_support=join_keys.get("support", ""),
            join_key_seed=join_keys.get("seed", ""),
        )
        pair_deltas.append(delta)

    if not pair_deltas:
        # Create a placeholder
        pair_deltas = [
            ToolPairDelta(
                pair_id="placeholder",
                control_tool=data.get("control_tools", [{}])[0].get("tool_id", "unknown"),
                treatment_tool=data.get("treatment_tools", [{}])[0].get("tool_id", "unknown"),
                join_key_task="placeholder",
                join_key_model="placeholder",
                join_key_support="placeholder",
                join_key_seed="42",
            )
        ]

    return generate_paired_tool_report(
        pair_deltas,
        control_tool=pair_deltas[0].control_tool,
        treatment_tool=pair_deltas[0].treatment_tool,
    )


def _group_runs_into_pairs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group runs into pairs by pair_id."""
    pairs_by_id: dict[str, dict[str, Any]] = {}

    for run in runs:
        pair_id = run.get("pair_id")
        if not pair_id:
            continue

        if pair_id not in pairs_by_id:
            pairs_by_id[pair_id] = {"pair_id": pair_id}

        position = run.get("pair_position", run.get("pair_role", ""))
        if position == "control":
            pairs_by_id[pair_id]["control_run"] = run
        elif position == "treatment":
            pairs_by_id[pair_id]["treatment_run"] = run

    # Only return complete pairs
    return [p for p in pairs_by_id.values() if "control_run" in p and "treatment_run" in p]


def save_paired_report(report: PairedToolReport, output_path: str | Path) -> None:
    """Save a paired report to a file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if output_path.suffix in (".yaml", ".yml"):
            f.write(report.to_yaml())
        else:
            json.dump(report.to_dict(), f, indent=2)


# =============================================================================
# Support Profile Pairing (for WP-04B)
# =============================================================================


@dataclass
class SupportPairDelta:
    """
    Delta metrics for a support profile A/B pair.

    Unlike ToolPairDelta, this compares support profiles while holding
    task/model/tool/seed constant.
    """

    pair_id: str
    control: str  # control support profile
    treatment: str  # treatment support profile
    task_id: str
    model: str
    repeat_index: int

    # Delta metrics (treatment - control)
    delta_pass_rate: float = 0.0  # 1.0, 0.0, or -1.0 for individual pairs
    delta_score: float = 0.0
    delta_cost: int = 0  # token cost
    delta_latency: float = 0.0  # seconds

    # Raw values for aggregation
    control_passed: bool = False
    treatment_passed: bool = False
    control_score: float = 0.0
    treatment_score: float = 0.0
    control_tokens: int = 0
    treatment_tokens: int = 0
    control_latency: float = 0.0
    treatment_latency: float = 0.0

    # Statistical significance (computed during aggregation)
    p_value: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "pair_id": self.pair_id,
            "control": self.control,
            "treatment": self.treatment,
            "task_id": self.task_id,
            "model": self.model,
            "repeat_index": self.repeat_index,
            "delta_pass_rate": self.delta_pass_rate,
            "delta_score": self.delta_score,
            "delta_cost": self.delta_cost,
            "delta_latency": self.delta_latency,
            "control_passed": self.control_passed,
            "treatment_passed": self.treatment_passed,
        }

        if self.p_value is not None:
            result["statistical_significance"] = {
                "p_value": round(self.p_value, 4),
                "confidence": round(self.confidence or 0.0, 4),
            }

        return result


@dataclass
class SupportPairReport:
    """Aggregated report of support profile delta metrics."""

    control_profile: str
    treatment_profile: str
    total_pairs: int = 0

    # Aggregated deltas
    mean_delta_pass_rate: float = 0.0
    mean_delta_score: float = 0.0
    mean_delta_cost: float = 0.0
    mean_delta_latency: float = 0.0

    # Win/loss/tie counts
    control_wins: int = 0
    treatment_wins: int = 0
    ties: int = 0

    # Statistical significance
    p_value: float | None = None
    confidence_level: float = 0.95
    is_significant: bool = False

    # Detailed deltas
    deltas: list[SupportPairDelta] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "control_profile": self.control_profile,
            "treatment_profile": self.treatment_profile,
            "total_pairs": self.total_pairs,
            "aggregated_deltas": {
                "mean_delta_pass_rate": round(self.mean_delta_pass_rate, 4),
                "mean_delta_score": round(self.mean_delta_score, 4),
                "mean_delta_cost": round(self.mean_delta_cost, 2),
                "mean_delta_latency": round(self.mean_delta_latency, 4),
            },
            "win_loss_summary": {
                "control_wins": self.control_wins,
                "treatment_wins": self.treatment_wins,
                "ties": self.ties,
            },
            "statistical_significance": {
                "p_value": round(self.p_value, 4) if self.p_value is not None else None,
                "confidence_level": self.confidence_level,
                "is_significant": self.is_significant,
            },
            "created_at": self.created_at,
            "deltas_count": len(self.deltas),
        }


def load_paired_support_results(results_path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Load results from a JSONL file and group by pair_id for support pairing.

    Args:
        results_path: Path to results.jsonl file

    Returns:
        Dict mapping pair_id -> list of results (control + treatment)
    """
    pairs: dict[str, list[dict[str, Any]]] = {}

    if not results_path.exists():
        return pairs

    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                pair_id = data.get("metadata", {}).get("pair_id") or data.get("pair_id")
                if pair_id:
                    if pair_id not in pairs:
                        pairs[pair_id] = []
                    pairs[pair_id].append(data)
            except Exception:
                continue

    return pairs


def compute_support_pair_deltas(
    paired_results: dict[str, list[dict[str, Any]]],
    control_profile: str,
    treatment_profile: str,
) -> list[SupportPairDelta]:
    """
    Compute delta metrics for all control/treatment support profile pairs.

    Args:
        paired_results: Dict mapping pair_id -> list of results
        control_profile: Name of control support profile
        treatment_profile: Name of treatment support profile

    Returns:
        List of SupportPairDelta instances
    """
    deltas: list[SupportPairDelta] = []

    for pair_id, results in paired_results.items():
        control_result = None
        treatment_result = None

        for result in results:
            profile_id = result.get("profile_id", "")
            pair_role = result.get("metadata", {}).get("pair_role", "")

            if pair_role == "control" or profile_id == control_profile:
                control_result = result
            elif pair_role == "treatment" or profile_id == treatment_profile:
                treatment_result = result

        if not control_result or not treatment_result:
            continue

        # Extract metrics
        control_passed = bool(control_result.get("passed", False))
        treatment_passed = bool(treatment_result.get("passed", False))

        control_score = float(control_result.get("score", 0.0))
        treatment_score = float(treatment_result.get("score", 0.0))

        token_usage = control_result.get("token_usage", {})
        control_tokens = int(token_usage.get("total", 0))
        token_usage = treatment_result.get("token_usage", {})
        treatment_tokens = int(token_usage.get("total", 0))

        control_latency = float(control_result.get("latency_s", 0.0))
        treatment_latency = float(treatment_result.get("latency_s", 0.0))

        task_id = control_result.get("task_id", "")
        model = control_result.get("model", "")
        repeat_index = int(control_result.get("repeat_index", 0))

        delta = SupportPairDelta(
            pair_id=pair_id,
            control=control_profile,
            treatment=treatment_profile,
            task_id=task_id,
            model=model,
            repeat_index=repeat_index,
            delta_pass_rate=(1.0 if treatment_passed else 0.0) - (1.0 if control_passed else 0.0),
            delta_score=treatment_score - control_score,
            delta_cost=treatment_tokens - control_tokens,
            delta_latency=treatment_latency - control_latency,
            control_passed=control_passed,
            treatment_passed=treatment_passed,
            control_score=control_score,
            treatment_score=treatment_score,
            control_tokens=control_tokens,
            treatment_tokens=treatment_tokens,
            control_latency=control_latency,
            treatment_latency=treatment_latency,
        )

        deltas.append(delta)

    return deltas


def compute_mcnemar_test(control_wins: int, treatment_wins: int) -> float:
    """
    Compute McNemar's test for paired binary outcomes.

    Args:
        control_wins: Pairs where control passed but treatment failed
        treatment_wins: Pairs where treatment passed but control failed

    Returns:
        p-value for the two-sided test
    """
    import math

    b = control_wins
    c = treatment_wins

    if b + c == 0:
        return 1.0

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    try:
        p_value = math.erfc(math.sqrt(chi2) / math.sqrt(2))
    except Exception:
        p_value = math.exp(-chi2 / 2)

    return p_value


def compute_paired_t_test(deltas: list[float]) -> tuple[float, float]:
    """
    Compute paired t-test for continuous outcomes.

    Args:
        deltas: List of delta values (treatment - control)

    Returns:
        Tuple of (t_statistic, p_value)
    """
    import math

    if len(deltas) < 2:
        return 0.0, 1.0

    n = len(deltas)
    mean_delta = sum(deltas) / n

    variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0:
        return 0.0, 1.0

    t_stat = mean_delta / (std_dev / math.sqrt(n))

    if n >= 30:
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    else:
        df = n - 1
        x = abs(t_stat) / math.sqrt(df)
        p_value = 2 * (1 - x / math.sqrt(x * x + df))

    return t_stat, p_value


def generate_support_pair_report(
    deltas: list[SupportPairDelta],
    control_profile: str,
    treatment_profile: str,
    confidence_level: float = 0.95,
) -> SupportPairReport:
    """
    Generate an aggregated report from support profile paired deltas.

    Args:
        deltas: List of SupportPairDelta instances
        control_profile: Name of control support profile
        treatment_profile: Name of treatment support profile
        confidence_level: Confidence level for significance testing

    Returns:
        SupportPairReport with aggregated metrics
    """
    report = SupportPairReport(
        control_profile=control_profile,
        treatment_profile=treatment_profile,
        confidence_level=confidence_level,
        deltas=deltas,
    )

    if not deltas:
        return report

    report.total_pairs = len(deltas)

    # Compute aggregated deltas
    report.mean_delta_pass_rate = sum(d.delta_pass_rate for d in deltas) / len(deltas)
    report.mean_delta_score = sum(d.delta_score for d in deltas) / len(deltas)
    report.mean_delta_cost = sum(d.delta_cost for d in deltas) / len(deltas)
    report.mean_delta_latency = sum(d.delta_latency for d in deltas) / len(deltas)

    # Count wins/losses/ties
    for delta in deltas:
        if delta.control_passed and not delta.treatment_passed:
            report.control_wins += 1
        elif not delta.control_passed and delta.treatment_passed:
            report.treatment_wins += 1
        else:
            report.ties += 1

    # Compute statistical significance using McNemar's test
    report.p_value = compute_mcnemar_test(report.control_wins, report.treatment_wins)
    report.is_significant = report.p_value is not None and report.p_value < (1 - confidence_level)

    # Also compute t-test for continuous scores
    if len(deltas) >= 2:
        score_deltas = [d.delta_score for d in deltas]
        _, score_p_value = compute_paired_t_test(score_deltas)

        if report.p_value is None or score_p_value < report.p_value:
            report.p_value = score_p_value
            report.is_significant = report.p_value < (1 - confidence_level)

    return report


def emit_support_pair_deltas(deltas: list[SupportPairDelta], output_path: Path) -> None:
    """
    Emit support pair deltas to a JSONL file.

    Args:
        deltas: List of SupportPairDelta instances
        output_path: Path to write paired_deltas.jsonl
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for delta in deltas:
            f.write(json.dumps(delta.to_dict()) + "\n")


def emit_support_pair_report(report: SupportPairReport, output_path: Path) -> None:
    """
    Emit support pair report to a JSON file.

    Args:
        report: SupportPairReport instance
        output_path: Path to write paired_report.json
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))


def process_support_pair_experiment(
    results_path: Path,
    output_dir: Path,
    control_profile: str,
    treatment_profile: str,
) -> SupportPairReport:
    """
    Process a support profile paired experiment and generate delta reports.

    This is the main entry point for processing paired support experiment results.

    Args:
        results_path: Path to results.jsonl
        output_dir: Directory to write output files
        control_profile: Name of control support profile
        treatment_profile: Name of treatment support profile

    Returns:
        SupportPairReport with aggregated metrics
    """
    # Load paired results
    paired_results = load_paired_support_results(results_path)

    # Compute deltas
    deltas = compute_support_pair_deltas(paired_results, control_profile, treatment_profile)

    # Generate report
    report = generate_support_pair_report(deltas, control_profile, treatment_profile)

    # Emit outputs
    emit_support_pair_deltas(deltas, output_dir / "paired_deltas.jsonl")
    emit_support_pair_report(report, output_dir / "paired_report.json")

    return report
