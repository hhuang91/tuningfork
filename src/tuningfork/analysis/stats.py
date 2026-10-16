"""Statistics for small samples.

The premise of the project is that agent-skill tuning is usually done on sample sizes far
too small to support the conclusions drawn from them. So nothing here ever returns a bare
proportion: every rate comes with the interval it actually implies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: 95% two-sided normal quantile.
Z95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Rate:
    successes: int
    total: int
    low: float
    high: float

    @property
    def point(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def to_json(self) -> dict:
        return {
            "successes": self.successes,
            "total": self.total,
            "point": self.point,
            "low": self.low,
            "high": self.high,
        }

    def format(self) -> str:
        """`3/5  60%  [23-88]` — the count is never omitted."""
        return (
            f"{self.successes}/{self.total}  {self.point:.0%}  "
            f"[{self.low:.0%}-{self.high:.0%}]"
        )


def wilson_interval(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval.

    Preferred over the normal approximation because it stays sensible at the small counts
    and extreme proportions this tool deals with — 0/5 and 5/5 included, where the textbook
    interval collapses to a point and claims certainty it has not earned.
    """
    if total <= 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def rate(successes: int, total: int, z: float = Z95) -> Rate:
    low, high = wilson_interval(successes, total, z)
    return Rate(successes=successes, total=total, low=low, high=high)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table ``[[a, b], [c, d]]``.

    Rows are the two arms, columns are (target behaviour, anything else). The two-sided
    p-value sums the probability of every table with the observed margins that is no more
    likely than the observed one — the convention R's `fisher.test` and scipy use.

    Exact rather than chi-square because the counts here are small by design: a checkpoint
    A/B is a *screen*, run at twenty or thirty samples per arm, where the chi-square
    approximation is unreliable. Probabilities are summed in log space to stay stable.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("cell counts must be non-negative")
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if total == 0:
        return 1.0

    log_denominator = _log_comb(total, col1)

    def log_probability(x: int) -> float:
        return _log_comb(row1, x) + _log_comb(row2, col1 - x) - log_denominator

    # Relative tolerance, as in R: tables that tie with the observed one up to rounding
    # must count as "as extreme", or a symmetric table can report p < 1.
    threshold = log_probability(a) + math.log1p(1e-7)
    p = sum(
        math.exp(log_probability(x))
        for x in range(max(0, col1 - row2), min(row1, col1) + 1)
        if log_probability(x) <= threshold
    )
    return min(1.0, p)


def p_short(p: float) -> str:
    """A p-value for a sentence: two decimals from 0.01 up ("0.07"), and below that its first
    significant figure ("0.005", "2e-05") — never "0.00", which reads as a certainty no test of
    a few runs has."""
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1g}"


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Controls the family-wise error rate as Bonferroni does — the chance of *any* false
    positive among the tests stays under alpha — but is never less powerful: the smallest
    p-value is multiplied by m, the next by m-1, and so on, each kept at least as large as
    the one before it. Compare every adjusted value against alpha, as with Bonferroni. The
    same as R's `p.adjust(p, "holm")`.
    """
    m = len(p_values)
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def samples_needed(baseline: float, target: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Approximate per-arm sample size to distinguish two rates.

    A normal approximation for two proportions — good enough for its only job, which is
    telling someone that separating 60% from 80% takes tens of runs per arm, not five.
    """
    if not 0 < baseline < 1 or not 0 < target < 1 or baseline == target:
        return 0
    z_alpha = _z_for_tail(alpha / 2)
    z_beta = _z_for_tail(1 - power)
    pooled = (baseline + target) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(baseline * (1 - baseline) + target * (1 - target))
    ) ** 2
    return math.ceil(numerator / (target - baseline) ** 2)


def _z_for_tail(tail: float) -> float:
    """Inverse normal CDF for an upper-tail probability, via bisection.

    Implemented directly so the package keeps no numerical dependency; scipy is optional.
    """
    if not 0 < tail < 1:
        return 0.0
    low, high = -10.0, 10.0
    target = 1 - tail
    for _ in range(200):
        mid = (low + high) / 2
        if _normal_cdf(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
