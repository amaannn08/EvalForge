"""Unit tests for statistical regression detection and CI gating."""

from evalforge.stats import (
    RegressionStatus,
    StatisticalRegressionDetector,
    bootstrap_confidence_interval,
    normal_cdf,
    percentile,
    student_t_p_value,
)


def test_normal_and_t_math():
    assert round(normal_cdf(0.0), 3) == 0.500
    assert normal_cdf(1.96) > 0.97
    assert normal_cdf(-1.96) < 0.03

    p_val = student_t_p_value(0.0, df=10)
    assert round(p_val, 2) == 1.00

    p_val_sig = student_t_p_value(3.5, df=15)
    assert p_val_sig < 0.01


def test_percentile_and_bootstrap():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50 = percentile(data, 50)
    assert round(p50, 1) == 30.0

    ci = bootstrap_confidence_interval(data, confidence=0.95, n_resamples=500)
    assert ci[0] <= 30.0 <= ci[1]


def test_regression_detection_flagging():
    detector = StatisticalRegressionDetector(alpha=0.05, regression_threshold_delta=0.02)

    baseline = [0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.97, 0.95, 0.93, 0.96]
    candidate_regressed = [0.82, 0.80, 0.83, 0.81, 0.82, 0.79, 0.84, 0.81, 0.80, 0.82]

    report = detector.compare(baseline, candidate_regressed)
    assert report.status == RegressionStatus.REGRESSION_DETECTED
    assert report.score_comparison.is_statistically_significant is True
    assert report.score_comparison.mean_delta < -0.10
    assert "REJECTED" in report.gate_reason


def test_pass_and_improvement_detection():
    detector = StatisticalRegressionDetector(alpha=0.05, regression_threshold_delta=0.02)

    baseline = [0.80, 0.81, 0.79, 0.82, 0.80, 0.78, 0.81, 0.80, 0.79, 0.82]
    candidate_improved = [0.92, 0.93, 0.91, 0.94, 0.92, 0.90, 0.93, 0.92, 0.91, 0.94]

    report = detector.compare(baseline, candidate_improved)
    assert report.status == RegressionStatus.IMPROVEMENT
    assert report.score_comparison.mean_delta > 0.10
    assert "IMPROVEMENT" in report.gate_reason


def test_insufficient_samples_inconclusive():
    detector = StatisticalRegressionDetector(min_samples=10)
    report = detector.compare([0.9, 0.8], [0.85, 0.75])
    assert report.status == RegressionStatus.INCONCLUSIVE
