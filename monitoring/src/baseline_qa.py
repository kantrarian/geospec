"""
baseline_qa.py - Quality Assurance for THD Baselines

Provides QA checks and drift detection for THD baseline calibration.
Ensures baselines meet acceptance criteria before use in production.

QA Checks:
- Coverage: At least 80% of requested days have valid THD values
- Stability: Coefficient of variation (CV) should be reasonable
- Drift: Detect if baseline has shifted significantly over time
- MAD Inflation: Detect if variability has increased

Author: R.J. Mathews / Claude
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaselineQA:
    """Quality assessment for a station baseline."""
    station: str
    n_days_requested: int
    n_days_valid: int
    coverage_pct: float
    cv_ratio: float                    # std/mean, flag if < 0.10 (too stable = suspicious)
    cv_flag: str                       # "ok", "low" (< 0.10), "high" (> 0.50)
    drift_detected: bool               # Median shift > 2σ between halves
    drift_sigma: float                 # Size of drift in σ units
    mad_inflation_detected: bool       # MAD grew > 50% in recent period
    mad_inflation_pct: float           # Percent change in MAD
    sample_rate_hz: int                # Native sample rate
    quality_grade: str                 # "good", "acceptable", "poor", "fail"
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'station': self.station,
            'n_days_requested': self.n_days_requested,
            'n_days_valid': self.n_days_valid,
            'coverage_pct': round(self.coverage_pct, 2),
            'cv_ratio': round(self.cv_ratio, 4),
            'cv_flag': self.cv_flag,
            'drift_detected': self.drift_detected,
            'drift_sigma': round(self.drift_sigma, 2),
            'mad_inflation_detected': self.mad_inflation_detected,
            'mad_inflation_pct': round(self.mad_inflation_pct, 1),
            'sample_rate_hz': self.sample_rate_hz,
            'quality_grade': self.quality_grade,
            'issues': self.issues,
        }


# QA Thresholds
QA_THRESHOLDS = {
    'min_coverage_pct': 80.0,          # Minimum 80% coverage required
    'min_days': 60,                     # Minimum 60 days for valid baseline
    'cv_low_threshold': 0.10,           # CV < 0.10 is suspiciously stable
    'cv_high_threshold': 0.50,          # CV > 0.50 is too variable
    'drift_sigma_threshold': 2.0,       # Flag if median shifts > 2σ
    'mad_inflation_threshold': 50.0,    # Flag if MAD increases > 50%
}


def compute_baseline_qa(
    station: str,
    daily_values: List[Tuple[str, float]],
    n_days_requested: int,
    sample_rate_hz: int = 40,
) -> BaselineQA:
    """
    Compute QA metrics for a baseline calibration.

    Args:
        station: Station identifier (e.g., "IU.ANTO")
        daily_values: List of (date_str, thd_value) tuples
        n_days_requested: Number of days that were requested
        sample_rate_hz: Native sample rate of the station

    Returns:
        BaselineQA object with all QA metrics
    """
    issues = []
    n_valid = len(daily_values)

    # Coverage check
    coverage_pct = (n_valid / n_days_requested) * 100 if n_days_requested > 0 else 0
    if coverage_pct < QA_THRESHOLDS['min_coverage_pct']:
        issues.append(f"Low coverage: {coverage_pct:.1f}% (min {QA_THRESHOLDS['min_coverage_pct']}%)")

    if n_valid < QA_THRESHOLDS['min_days']:
        issues.append(f"Insufficient days: {n_valid} (min {QA_THRESHOLDS['min_days']})")

    # Can't compute other metrics without enough data
    if n_valid < 10:
        return BaselineQA(
            station=station,
            n_days_requested=n_days_requested,
            n_days_valid=n_valid,
            coverage_pct=coverage_pct,
            cv_ratio=0.0,
            cv_flag="unknown",
            drift_detected=False,
            drift_sigma=0.0,
            mad_inflation_detected=False,
            mad_inflation_pct=0.0,
            sample_rate_hz=sample_rate_hz,
            quality_grade="fail",
            issues=issues + ["Too few samples for QA analysis"],
        )

    # Extract THD values
    thd_values = np.array([v[1] for v in daily_values])

    # Coefficient of variation
    mean_thd = np.mean(thd_values)
    std_thd = np.std(thd_values)
    cv_ratio = std_thd / mean_thd if mean_thd > 0 else 0.0

    if cv_ratio < QA_THRESHOLDS['cv_low_threshold']:
        cv_flag = "low"
        issues.append(f"CV too low: {cv_ratio:.3f} (suspiciously stable)")
    elif cv_ratio > QA_THRESHOLDS['cv_high_threshold']:
        cv_flag = "high"
        issues.append(f"CV too high: {cv_ratio:.3f} (highly variable)")
    else:
        cv_flag = "ok"

    # Drift detection (compare first half vs second half)
    mid = n_valid // 2
    first_half = thd_values[:mid]
    second_half = thd_values[mid:]

    median_first = np.median(first_half)
    median_second = np.median(second_half)

    # Use MAD of full dataset for sigma estimate
    full_mad = np.median(np.abs(thd_values - np.median(thd_values)))
    full_sigma = full_mad * 1.4826 if full_mad > 0 else std_thd

    drift_sigma = abs(median_second - median_first) / full_sigma if full_sigma > 0 else 0.0
    drift_detected = drift_sigma > QA_THRESHOLDS['drift_sigma_threshold']

    if drift_detected:
        direction = "increased" if median_second > median_first else "decreased"
        issues.append(f"Drift detected: median {direction} by {drift_sigma:.1f}σ")

    # MAD inflation detection
    mad_first = np.median(np.abs(first_half - np.median(first_half)))
    mad_second = np.median(np.abs(second_half - np.median(second_half)))

    if mad_first > 0:
        mad_inflation_pct = ((mad_second - mad_first) / mad_first) * 100
    else:
        mad_inflation_pct = 0.0

    mad_inflation_detected = mad_inflation_pct > QA_THRESHOLDS['mad_inflation_threshold']

    if mad_inflation_detected:
        issues.append(f"MAD inflation: variability increased {mad_inflation_pct:.1f}%")

    # Determine overall quality grade
    quality_grade = _compute_quality_grade(
        coverage_pct=coverage_pct,
        n_valid=n_valid,
        cv_flag=cv_flag,
        drift_detected=drift_detected,
        mad_inflation_detected=mad_inflation_detected,
    )

    return BaselineQA(
        station=station,
        n_days_requested=n_days_requested,
        n_days_valid=n_valid,
        coverage_pct=coverage_pct,
        cv_ratio=cv_ratio,
        cv_flag=cv_flag,
        drift_detected=drift_detected,
        drift_sigma=drift_sigma,
        mad_inflation_detected=mad_inflation_detected,
        mad_inflation_pct=mad_inflation_pct,
        sample_rate_hz=sample_rate_hz,
        quality_grade=quality_grade,
        issues=issues,
    )


def _compute_quality_grade(
    coverage_pct: float,
    n_valid: int,
    cv_flag: str,
    drift_detected: bool,
    mad_inflation_detected: bool,
) -> str:
    """Compute overall quality grade from individual metrics."""

    # Hard failures
    if coverage_pct < 50 or n_valid < 30:
        return "fail"

    # Count issues
    n_issues = 0
    if coverage_pct < QA_THRESHOLDS['min_coverage_pct']:
        n_issues += 1
    if n_valid < QA_THRESHOLDS['min_days']:
        n_issues += 1
    if cv_flag != "ok":
        n_issues += 1
    if drift_detected:
        n_issues += 2  # Drift is a serious issue
    if mad_inflation_detected:
        n_issues += 1

    # Grade based on issue count
    if n_issues == 0:
        return "good"
    elif n_issues <= 2:
        return "acceptable"
    elif n_issues <= 4:
        return "poor"
    else:
        return "fail"


def detect_recent_drift(
    daily_values: List[Tuple[str, float]],
    recent_days: int = 30,
    prior_days: int = 30,
) -> Tuple[bool, float, str]:
    """
    Detect drift between recent window and prior window.

    Used to check if baseline has shifted since last calibration.

    Args:
        daily_values: List of (date_str, thd_value) tuples, sorted by date
        recent_days: Number of recent days to compare
        prior_days: Number of prior days to compare against

    Returns:
        Tuple of (drift_detected, drift_sigma, description)
    """
    n_values = len(daily_values)

    if n_values < recent_days + prior_days:
        return False, 0.0, "Insufficient data for drift detection"

    # Get recent and prior windows
    thd_values = np.array([v[1] for v in daily_values])
    recent_values = thd_values[-recent_days:]
    prior_values = thd_values[-(recent_days + prior_days):-recent_days]

    # Compute medians
    median_recent = np.median(recent_values)
    median_prior = np.median(prior_values)

    # Use MAD from prior period as baseline sigma
    mad_prior = np.median(np.abs(prior_values - median_prior))
    sigma_prior = mad_prior * 1.4826 if mad_prior > 0 else np.std(prior_values)

    if sigma_prior == 0:
        return False, 0.0, "Cannot compute drift (zero variance)"

    drift_sigma = (median_recent - median_prior) / sigma_prior
    drift_detected = abs(drift_sigma) > QA_THRESHOLDS['drift_sigma_threshold']

    if drift_detected:
        direction = "higher" if drift_sigma > 0 else "lower"
        description = f"Recent {recent_days}d median is {abs(drift_sigma):.1f}sigma {direction} than prior {prior_days}d"
    else:
        description = f"No significant drift (delta = {drift_sigma:.2f}sigma)"

    return drift_detected, drift_sigma, description


def generate_qa_report(qa_results: List[BaselineQA]) -> str:
    """Generate a human-readable QA report."""
    lines = [
        "=" * 80,
        "BASELINE QA REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
        "",
        f"{'Station':<12} {'Coverage':>10} {'N':>5} {'CV':>8} {'Drift':>8} {'MAD%':>8} {'Grade':<12}",
        "-" * 80,
    ]

    for qa in qa_results:
        drift_str = f"{qa.drift_sigma:.1f}s" if qa.drift_detected else "ok"
        mad_str = f"+{qa.mad_inflation_pct:.0f}%" if qa.mad_inflation_detected else "ok"

        lines.append(
            f"{qa.station:<12} {qa.coverage_pct:>9.1f}% {qa.n_days_valid:>5} "
            f"{qa.cv_ratio:>8.3f} {drift_str:>8} {mad_str:>8} {qa.quality_grade:<12}"
        )

    # Summary
    lines.append("-" * 80)
    grades = [qa.quality_grade for qa in qa_results]
    n_good = grades.count("good")
    n_acceptable = grades.count("acceptable")
    n_poor = grades.count("poor")
    n_fail = grades.count("fail")

    lines.append(f"Summary: {n_good} good, {n_acceptable} acceptable, {n_poor} poor, {n_fail} fail")

    # List issues
    all_issues = []
    for qa in qa_results:
        for issue in qa.issues:
            all_issues.append(f"  {qa.station}: {issue}")

    if all_issues:
        lines.append("")
        lines.append("Issues:")
        lines.extend(all_issues)

    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic data
    print("Baseline QA Module Test")
    print("=" * 60)

    # Generate synthetic daily values
    np.random.seed(42)
    dates = [(datetime(2025, 11, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range(90)]

    # Normal case
    thd_normal = np.random.normal(0.15, 0.04, 90)
    daily_normal = list(zip(dates, thd_normal))

    qa_normal = compute_baseline_qa("IU.TEST", daily_normal, 90, 40)
    print(f"\nNormal case: {qa_normal.quality_grade}")
    print(f"  Coverage: {qa_normal.coverage_pct:.1f}%")
    print(f"  CV: {qa_normal.cv_ratio:.3f}")
    print(f"  Drift: {qa_normal.drift_sigma:.2f}σ")

    # Drift case
    thd_drift = np.concatenate([
        np.random.normal(0.15, 0.04, 45),
        np.random.normal(0.25, 0.04, 45),  # 2.5σ shift!
    ])
    daily_drift = list(zip(dates, thd_drift))

    qa_drift = compute_baseline_qa("IU.DRIFT", daily_drift, 90, 40)
    print(f"\nDrift case: {qa_drift.quality_grade}")
    print(f"  Issues: {qa_drift.issues}")

    from datetime import timedelta
