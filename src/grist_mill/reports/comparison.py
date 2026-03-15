"""Experiment comparison with statistical significance.

Compares two experiments (must use the same task set) and produces:
- Per-task deltas (score change, status change)
- Aggregate pass-rate delta with confidence interval
- Statistical significance test (McNemar's test for paired proportions)

Validates:
- VAL-REPORT-01: Experiment comparison produces paired results
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def compare_experiments(
    experiment_a: list[dict[str, Any]],
    experiment_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two experiments that share the same task set.

    Args:
        experiment_a: Results from experiment A. Each dict must have
            ``task_id``, ``status``, ``score`` fields.
        experiment_b: Results from experiment B. Same structure as A.

    Returns:
        A dict with:
        - ``per_task``: List of per-task comparison dicts with
          ``task_id``, ``delta_score``, ``status_a``, ``status_b``.
        - ``aggregate``: Dict with ``pass_rate_a``, ``pass_rate_b``,
          ``delta_pass_rate``, ``confidence_interval``, ``is_significant``,
          ``p_value``, ``n_tasks``, ``n_agree``, ``n_a_only``, ``n_b_only``,
          ``n_neither``.

    Raises:
        ValueError: If the experiments do not cover the same task set.
    """
    _validate_same_tasks(experiment_a, experiment_b)

    tasks_a = {r["task_id"]: r for r in experiment_a}
    tasks_b = {r["task_id"]: r for r in experiment_b}

    all_task_ids = sorted(tasks_a.keys())

    # Per-task comparison
    per_task: list[dict[str, Any]] = []
    for tid in all_task_ids:
        ra = tasks_a[tid]
        rb = tasks_b[tid]
        delta_score = rb["score"] - ra["score"]
        per_task.append(
            {
                "task_id": tid,
                "score_a": ra["score"],
                "score_b": rb["score"],
                "delta_score": delta_score,
                "status_a": ra["status"].value
                if hasattr(ra["status"], "value")
                else str(ra["status"]),
                "status_b": rb["status"].value
                if hasattr(rb["status"], "value")
                else str(rb["status"]),
            }
        )

    # Aggregate metrics
    n = len(all_task_ids)
    pass_rate_a = _pass_rate(experiment_a)
    pass_rate_b = _pass_rate(experiment_b)
    delta = pass_rate_b - pass_rate_a

    # McNemar's test for paired proportions
    # Count concordant and discordant pairs
    n_a_only = 0  # A pass, B fail
    n_b_only = 0  # A fail, B pass
    n_agree_pass = 0
    n_agree_fail = 0

    for tid in all_task_ids:
        a_pass = _is_pass(tasks_a[tid])
        b_pass = _is_pass(tasks_b[tid])
        if a_pass and b_pass:
            n_agree_pass += 1
        elif not a_pass and not b_pass:
            n_agree_fail += 1
        elif a_pass and not b_pass:
            n_a_only += 1
        else:
            n_b_only += 1

    # Confidence interval for the delta using normal approximation
    # (simplified Wald interval for the difference of proportions)
    se = _standard_error(pass_rate_a, pass_rate_b, n)
    z = 1.96  # 95% CI
    ci_lower = delta - z * se
    ci_upper = delta + z * se

    # McNemar's chi-squared test
    p_value, is_significant = _mcnemar_test(n_a_only, n_b_only)

    aggregate: dict[str, Any] = {
        "pass_rate_a": pass_rate_a,
        "pass_rate_b": pass_rate_b,
        "delta_pass_rate": delta,
        "confidence_interval": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "is_significant": is_significant,
        "p_value": p_value,
        "n_tasks": n,
        "n_agree": n_agree_pass + n_agree_fail,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_neither": n_agree_fail,
    }

    logger.info(
        "Comparison: A=%.2f%% B=%.2f%% delta=%.2f%% significant=%s p=%.4f",
        pass_rate_a * 100,
        pass_rate_b * 100,
        delta * 100,
        is_significant,
        p_value,
    )

    return {
        "per_task": per_task,
        "aggregate": aggregate,
    }


def _validate_same_tasks(
    experiment_a: list[dict[str, Any]],
    experiment_b: list[dict[str, Any]],
) -> None:
    """Validate that both experiments cover the same task set.

    Args:
        experiment_a: Results from experiment A.
        experiment_b: Results from experiment B.

    Raises:
        ValueError: If task sets differ.
    """
    ids_a = {r["task_id"] for r in experiment_a}
    ids_b = {r["task_id"] for r in experiment_b}

    if ids_a != ids_b:
        only_in_a = ids_a - ids_b
        only_in_b = ids_b - ids_a
        msg = (
            f"Experiments must have the same task set. "
            f"Only in A: {len(only_in_a)}, only in B: {len(only_in_b)}."
        )
        raise ValueError(msg)


def _is_pass(result: dict[str, Any]) -> bool:
    """Check if a result represents a passing task.

    A task passes if score >= 1.0 or status is SUCCESS.
    """
    if result.get("score", 0.0) >= 1.0:
        return True
    status = result.get("status")
    if hasattr(status, "value"):
        return str(status.value) == "SUCCESS"
    return str(status) == "SUCCESS"


def _pass_rate(results: list[dict[str, Any]]) -> float:
    """Calculate the pass rate for a list of results.

    Args:
        results: List of result dicts.

    Returns:
        Fraction of tasks that passed (0.0 to 1.0).
    """
    if not results:
        return 0.0
    n_pass = sum(1 for r in results if _is_pass(r))
    return n_pass / len(results)


def _standard_error(p1: float, p2: float, n: int) -> float:
    """Compute standard error for the difference of two proportions.

    Uses the Wald approximation: SE = sqrt(p1*(1-p1)/n + p2*(1-p2)/n).

    Args:
        p1: Proportion from experiment A.
        p2: Proportion from experiment B.
        n: Number of paired observations.

    Returns:
        Standard error of the difference.
    """
    if n <= 0:
        return 0.0
    var = p1 * (1 - p1) / n + p2 * (1 - p2) / n
    return math.sqrt(max(0.0, var))


def _mcnemar_test(b: int, c: int) -> tuple[float, bool]:
    """Perform McNemar's test for paired proportions.

    Tests whether the proportion of discordant pairs is symmetric.

    Uses the exact binomial test for small samples and McNemar's
    chi-squared test with continuity correction for larger samples.

    Args:
        b: Count of pairs where A passed but B failed.
        c: Count of pairs where A failed but B passed.

    Returns:
        A tuple of (p_value, is_significant_at_0.05).
    """
    total_discordant = b + c
    if total_discordant == 0:
        # No discordant pairs — no evidence of difference
        return 1.0, False

    if total_discordant < 25:
        # Use exact binomial test (exact McNemar)
        # Under H0 (no difference), discordant pairs are equally likely
        # to go either way: X ~ Binomial(n=b+c, p=0.5)
        # p-value = P(X <= min(b,c)) where X counts the "less frequent" direction
        from math import comb

        n = b + c
        k = min(b, c)
        # One-sided p-value: probability of seeing this few or fewer in one direction
        p_one_sided = sum(comb(n, i) * (0.5**n) for i in range(k + 1))
        # Two-sided: double the one-sided
        p_value = min(1.0, 2.0 * p_one_sided)
        return p_value, p_value < 0.05

    # McNemar's chi-squared test with continuity correction
    chi_sq = ((abs(b - c) - 1) ** 2) / total_discordant if total_discordant > 0 else 0.0

    # Approximate p-value from chi-squared distribution (df=1)
    p_value = _chi_sq_p_value(chi_sq, df=1)
    return p_value, p_value < 0.05


def _chi_sq_p_value(x: float, df: int = 1) -> float:
    """Approximate the p-value for a chi-squared statistic.

    Uses the regularized incomplete gamma function approximation
    (Abramowitz and Stegun).

    Args:
        x: Chi-squared test statistic.
        df: Degrees of freedom (must be positive).

    Returns:
        The p-value (upper tail probability).
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 1.0

    # For df=1, use the simpler error function approximation
    if df == 1:
        # P(chi^2 > x) = P(Z > sqrt(x)) * 2 = erfc(sqrt(x/2))
        z = math.sqrt(x / 2.0)
        return _erfc(z)

    # For general df, use the incomplete gamma function
    # P = GammaInc(df/2, x/2) / Gamma(df/2)
    k = df / 2.0
    t = x / 2.0
    return _regularized_gamma_q(k, t)


def _erfc(x: float) -> float:
    """Complementary error function approximation.

    Uses Abramowitz and Stegun approximation 7.1.26.
    """
    if x < 0:
        x = -x

    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return y


def _regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized incomplete gamma function Q(a, x).

    Uses a series expansion for small x and continued fraction for large x.

    Args:
        a: Shape parameter (> 0).
        x: Integration upper limit (>= 0).

    Returns:
        Q(a, x) = 1 - P(a, x).
    """
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1:
        # Use series expansion for P(a, x), then Q = 1 - P
        p = _gamma_series(a, x)
        return max(0.0, min(1.0, 1.0 - p))
    else:
        # Use continued fraction for Q(a, x)
        return max(0.0, min(1.0, _gamma_cf(a, x)))


def _gamma_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) via series expansion."""
    max_iter = 200
    eps = 1e-12

    gln = _lngamma(a)
    ap = a
    s = 1.0 / a
    delta = s

    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * eps:
            break

    return s * math.exp(-x + a * math.log(x) - gln)


def _gamma_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) via continued fraction."""
    max_iter = 200
    eps = 1e-12
    tiny = 1e-30

    gln = _lngamma(a)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d

    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return math.exp(-x + a * math.log(x) - gln) * h


def _lngamma(x: float) -> float:
    """Log-gamma function approximation (Stirling's for large x)."""
    if x <= 0:
        return 0.0

    # Lanczos approximation (g=7, n=9)
    coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if x < 0.5:
        # Use reflection formula
        return math.log(math.pi / math.sin(math.pi * x)) - _lngamma(1.0 - x)

    x -= 1.0
    a = coef[0]
    t = x + 7.5  # g + 0.5

    for i in range(1, len(coef)):
        a += coef[i] / (x + i)

    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


__all__ = ["compare_experiments"]
