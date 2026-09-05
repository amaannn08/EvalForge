"""Statistical regression detection and CI gating engine for LLM evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
from typing import Any, Optional, Sequence

from evalforge.stats.math_utils import (
    bootstrap_confidence_interval,
    normal_p_value_one_tailed,
    percentile,
    student_t_p_value,
)


class RegressionStatus(str, Enum):
    PASS = "PASS"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    IMPROVEMENT = "IMPROVEMENT"


@dataclass
class PairedMetricComparison:
    """Detailed paired difference metrics between candidate and baseline."""
    metric_name: str
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    relative_change_pct: float
    standard_error: float
    t_statistic: float
    p_value: float
    is_statistically_significant: bool
    confidence_interval_95: tuple[float, float]
    cohens_d: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_mean": round(self.baseline_mean, 4),
            "candidate_mean": round(self.candidate_mean, 4),
            "mean_delta": round(self.mean_delta, 4),
            "relative_change_pct": round(self.relative_change_pct, 2),
            "standard_error": round(self.standard_error, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 5),
            "is_statistically_significant": self.is_statistically_significant,
            "confidence_interval_95": self.confidence_interval_95,
            "cohens_d": round(self.cohens_d, 4),
        }


@dataclass
class RegressionReport:
    """Comprehensive statistical regression report for evaluation comparisons."""
    status: RegressionStatus
    sample_size: int
    baseline_run_id: str
    candidate_run_id: str
    score_comparison: PairedMetricComparison
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    baseline_latency_p50: float
    candidate_latency_p50: float
    baseline_latency_p95: float
    candidate_latency_p95: float
    latency_p95_change_pct: float
    gate_reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "score_comparison": self.score_comparison.to_dict(),
            "pass_rates": {
                "baseline": round(self.baseline_pass_rate, 4),
                "candidate": round(self.candidate_pass_rate, 4),
                "delta": round(self.pass_rate_delta, 4),
            },
            "latencies_ms": {
                "baseline_p50": round(self.baseline_latency_p50, 2),
                "candidate_p50": round(self.candidate_latency_p50, 2),
                "baseline_p95": round(self.baseline_latency_p95, 2),
                "candidate_p95": round(self.candidate_latency_p95, 2),
                "p95_change_pct": round(self.latency_p95_change_pct, 2),
            },
            "gate_reason": self.gate_reason,
            "details": self.details,
        }

    def to_markdown(self) -> str:
        """Render a GitHub PR-ready Markdown comment summary."""
        status_icon = "✅" if self.status == RegressionStatus.PASS else ("🚀" if self.status == RegressionStatus.IMPROVEMENT else "❌")
        lines = [
            f"## {status_icon} EvalForge Statistical Regression Report",
            f"",
            f"- **Status**: `{self.status.value}`",
            f"- **Verdict**: {self.gate_reason}",
            f"- **Sample Size**: `{self.sample_size}` paired test cases",
            f"",
            f"### Metric Comparison",
            f"| Metric | Baseline | Candidate | Delta | p-value | Significant (α=0.05)? |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| Quality Score | `{self.score_comparison.baseline_mean:.4f}` | `{self.score_comparison.candidate_mean:.4f}` | `{self.score_comparison.mean_delta:+.4f}` (`{self.score_comparison.relative_change_pct:+.1f}%`) | `{self.score_comparison.p_value:.4f}` | `{'YES' if self.score_comparison.is_statistically_significant else 'NO'}` |",
            f"| Pass Rate | `{self.baseline_pass_rate * 100:.1f}%` | `{self.candidate_pass_rate * 100:.1f}%` | `{self.pass_rate_delta * 100:+.1f}%` | - | - |",
            f"| Latency (p50) | `{self.baseline_latency_p50:.1f}ms` | `{self.candidate_latency_p50:.1f}ms` | `{self.candidate_latency_p50 - self.baseline_latency_p50:+.1f}ms` | - | - |",
            f"| Latency (p95) | `{self.baseline_latency_p95:.1f}ms` | `{self.candidate_latency_p95:.1f}ms` | `{self.latency_p95_change_pct:+.1f}%` | - | - |",
            f"",
            f"> **95% Bootstrap Confidence Interval on Quality Delta**: `[{self.score_comparison.confidence_interval_95[0]:+.4f}, {self.score_comparison.confidence_interval_95[1]:+.4f}]`  ",
            f"> **Cohen's d Effect Size**: `{self.score_comparison.cohens_d:.3f}`",
        ]
        return "\n".join(lines)


class StatisticalRegressionDetector:
    """Detects statistically significant regressions between paired LLM evaluation runs."""

    def __init__(
        self,
        alpha: float = 0.05,
        regression_threshold_delta: float = 0.02,  # 2% drop is flagged as degradation
        max_latency_regression_pct: float = 25.0,  # 25% p95 increase is flagged
        min_samples: int = 5,
    ):
        self.alpha = alpha
        self.regression_threshold_delta = regression_threshold_delta
        self.max_latency_regression_pct = max_latency_regression_pct
        self.min_samples = min_samples

    def compare(
        self,
        baseline_scores: Sequence[float],
        candidate_scores: Sequence[float],
        baseline_latencies: Optional[Sequence[float]] = None,
        candidate_latencies: Optional[Sequence[float]] = None,
        baseline_passed: Optional[Sequence[bool]] = None,
        candidate_passed: Optional[Sequence[bool]] = None,
        baseline_run_id: str = "baseline",
        candidate_run_id: str = "candidate",
    ) -> RegressionReport:
        n = min(len(baseline_scores), len(candidate_scores))
        if n < self.min_samples:
            return RegressionReport(
                status=RegressionStatus.INCONCLUSIVE,
                sample_size=n,
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
                score_comparison=PairedMetricComparison(
                    metric_name="score",
                    baseline_mean=sum(baseline_scores) / max(1, len(baseline_scores)),
                    candidate_mean=sum(candidate_scores) / max(1, len(candidate_scores)),
                    mean_delta=0.0,
                    relative_change_pct=0.0,
                    standard_error=0.0,
                    t_statistic=0.0,
                    p_value=1.0,
                    is_statistically_significant=False,
                    confidence_interval_95=(0.0, 0.0),
                    cohens_d=0.0,
                ),
                baseline_pass_rate=0.0,
                candidate_pass_rate=0.0,
                pass_rate_delta=0.0,
                baseline_latency_p50=0.0,
                candidate_latency_p50=0.0,
                baseline_latency_p95=0.0,
                candidate_latency_p95=0.0,
                latency_p95_change_pct=0.0,
                gate_reason=f"Insufficient paired samples ({n} < {self.min_samples}) for statistical inference.",
            )

        # Slice to paired length
        b_scores = list(baseline_scores[:n])
        c_scores = list(candidate_scores[:n])
        deltas = [c - b for b, c in zip(b_scores, c_scores)]

        base_mean = statistics.mean(b_scores)
        cand_mean = statistics.mean(c_scores)
        mean_delta = statistics.mean(deltas)
        rel_change = (mean_delta / base_mean * 100.0) if base_mean > 0 else 0.0

        std_delta = statistics.stdev(deltas) if n > 1 else 0.0
        se = (std_delta / math.sqrt(n)) if n > 1 else 0.0
        t_stat = (mean_delta / se) if se > 0 else 0.0
        p_val = student_t_p_value(t_stat, df=n - 1) if n > 1 else 1.0
        is_sig = p_val < self.alpha

        cohens_d = (mean_delta / std_delta) if std_delta > 0 else 0.0
        ci = bootstrap_confidence_interval(deltas, confidence=0.95)

        paired_comparison = PairedMetricComparison(
            metric_name="composite_score",
            baseline_mean=base_mean,
            candidate_mean=cand_mean,
            mean_delta=mean_delta,
            relative_change_pct=rel_change,
            standard_error=se,
            t_statistic=t_stat,
            p_value=p_val,
            is_statistically_significant=is_sig,
            confidence_interval_95=ci,
            cohens_d=cohens_d,
        )

        # Pass rates
        b_pass = sum(1 for p in (baseline_passed[:n] if baseline_passed else [] if not b_scores else [s >= 0.7 for s in b_scores])) / n
        c_pass = sum(1 for p in (candidate_passed[:n] if candidate_passed else [] if not c_scores else [s >= 0.7 for s in c_scores])) / n
        pass_delta = c_pass - b_pass

        # Latencies
        b_lat = list(baseline_latencies[:n]) if baseline_latencies else [100.0] * n
        c_lat = list(candidate_latencies[:n]) if candidate_latencies else [100.0] * n
        b_p50 = percentile(b_lat, 50)
        c_p50 = percentile(c_lat, 50)
        b_p95 = percentile(b_lat, 95)
        c_p95 = percentile(c_lat, 95)
        lat_p95_pct = ((c_p95 - b_p95) / b_p95 * 100.0) if b_p95 > 0 else 0.0

        # Decision Policy Logic
        is_regression = False
        reasons = []

        # Check 1: Quality score drop
        if mean_delta < -self.regression_threshold_delta and is_sig:
            is_regression = True
            reasons.append(f"Statistically significant score regression of {mean_delta:.4f} (p={p_val:.4f} < {self.alpha})")
        elif mean_delta < -self.regression_threshold_delta:
            reasons.append(f"Score dropped by {mean_delta:.4f} (below threshold {self.regression_threshold_delta}), but not statistically significant (p={p_val:.4f})")

        # Check 2: Pass rate drop
        if pass_delta < -0.05:
            is_regression = True
            reasons.append(f"Pass rate dropped sharply by {pass_delta * 100:.1f}%")

        # Check 3: Latency regression
        if lat_p95_pct > self.max_latency_regression_pct:
            is_regression = True
            reasons.append(f"Latency p95 degraded by +{lat_p95_pct:.1f}% (exceeds limit +{self.max_latency_regression_pct}%)")

        if is_regression:
            status = RegressionStatus.REGRESSION_DETECTED
            gate_reason = "REJECTED: " + "; ".join(reasons)
        elif mean_delta > self.regression_threshold_delta and is_sig:
            status = RegressionStatus.IMPROVEMENT
            gate_reason = f"PASSED WITH IMPROVEMENT: Score increased significantly by +{mean_delta:.4f} (p={p_val:.4f})"
        else:
            status = RegressionStatus.PASS
            gate_reason = "PASSED: Candidate meets quality, latency, and consistency thresholds."

        return RegressionReport(
            status=status,
            sample_size=n,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            score_comparison=paired_comparison,
            baseline_pass_rate=b_pass,
            candidate_pass_rate=c_pass,
            pass_rate_delta=pass_delta,
            baseline_latency_p50=b_p50,
            candidate_latency_p50=c_p50,
            baseline_latency_p95=b_p95,
            candidate_latency_p95=c_p95,
            latency_p95_change_pct=lat_p95_pct,
            gate_reason=gate_reason,
            details={"deltas": [round(d, 4) for d in deltas]},
        )
