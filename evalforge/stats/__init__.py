"""Statistical evaluation and regression detection package for EvalForge."""

from evalforge.stats.math_utils import (
    normal_cdf,
    student_t_p_value,
    bootstrap_confidence_interval,
    percentile,
)
from evalforge.stats.regression import (
    RegressionStatus,
    PairedMetricComparison,
    RegressionReport,
    StatisticalRegressionDetector,
)

__all__ = [
    "normal_cdf",
    "student_t_p_value",
    "bootstrap_confidence_interval",
    "percentile",
    "RegressionStatus",
    "PairedMetricComparison",
    "RegressionReport",
    "StatisticalRegressionDetector",
]
