"""Statistical math utilities implemented with standard library precision."""

from __future__ import annotations

import math
import random
import statistics
from typing import Sequence


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def normal_p_value_two_tailed(z: float) -> float:
    """Two-tailed p-value for standard normal distribution."""
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def normal_p_value_one_tailed(z: float, alternative: str = "less") -> float:
    """One-tailed p-value (alternative='less' tests if candidate < baseline)."""
    if alternative == "less":
        return normal_cdf(z)
    else:
        return 1.0 - normal_cdf(z)


def t_distribution_cdf_approx(t: float, df: int) -> float:
    """Approximation of Student's t-distribution CDF using Cornish-Fisher expansion."""
    if df <= 0:
        raise ValueError("Degrees of freedom must be strictly positive.")
    if df >= 30:
        # Normal approximation is asymptotically valid for df >= 30
        return normal_cdf(t)
    
    # For small df, use Peizer-Pratt / Hill transform approximation
    a = df - 0.5
    b = 48.0 * a * a
    z2 = a * math.log(1.0 + (t * t) / df)
    z = math.sqrt(max(0.0, z2))
    if t < 0:
        z = -z
    return normal_cdf(z)


def student_t_p_value(t: float, df: int) -> float:
    """Two-tailed p-value for Student's t-distribution."""
    cdf_val = t_distribution_cdf_approx(abs(t), df)
    return max(0.0, min(1.0, 2.0 * (1.0 - cdf_val)))


def bootstrap_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute empirical bootstrap confidence interval for the sample mean."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        resample = [rng.choice(values) for _ in range(n)]
        means.append(sum(resample) / n)

    means.sort()
    lower_idx = int(((1.0 - confidence) / 2.0) * n_resamples)
    upper_idx = int((1.0 - (1.0 - confidence) / 2.0) * n_resamples)
    upper_idx = min(upper_idx, n_resamples - 1)

    return (round(means[lower_idx], 4), round(means[upper_idx], 4))


def percentile(data: Sequence[float], p: float) -> float:
    """Calculate the p-th percentile (0 <= p <= 100) using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1
