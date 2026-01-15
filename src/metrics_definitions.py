#!/usr/bin/env python3
"""
metrics_definitions.py
Canonical metric definitions for Λ_geo validation.

SINGLE SOURCE OF TRUTH for all metrics used in validation and documentation.

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import json

# =============================================================================
# CANONICAL METRIC DEFINITIONS
# =============================================================================

"""
BASELINE DEFINITION:
-------------------
The baseline Λ_geo is computed as the MEDIAN of the spatial maximum Λ_geo
values over the first 7 days of the analysis window.

Why median? Robust to outliers.
Why spatial max? We care about the peak strain instability, not average.
Why 7 days? Provides stable reference before precursor onset.

Formula:
    baseline = median(max_spatial(Λ_geo[t]) for t in days 0-6)
"""

BASELINE_DAYS = 7  # First N days used for baseline
BASELINE_STAT = 'median'  # 'median' or 'mean'

"""
FIRST DETECTION DEFINITION:
--------------------------
First detection is the FIRST TIME the spatial maximum Λ_geo exceeds 
2× the baseline value AND remains elevated for at least 2 consecutive days.

Why 2× baseline? Distinguishes signal from noise fluctuations.
Why 2 consecutive days? Prevents single-day spikes from triggering.

Formula:
    first_detection = min(t) where:
        max_spatial(Λ_geo[t]) > 2 * baseline AND
        max_spatial(Λ_geo[t+1]) > 2 * baseline
"""

DETECTION_THRESHOLD_FACTOR = 2.0  # Signal must exceed N× baseline
DETECTION_SUSTAINED_DAYS = 2  # Must remain elevated for N days

"""
AMPLIFICATION DEFINITION:
------------------------
Amplification is the ratio of PEAK spatial maximum Λ_geo (in the 72h 
precursor window) to the baseline.

Why peak in 72h window? Focus on the critical pre-event period.
Why not detection-day? Peak represents the maximum signal strength.

Formula:
    amplification = max(max_spatial(Λ_geo[t]) for t in [-72h, 0h]) / baseline
"""

AMPLIFICATION_WINDOW_HOURS = 72  # Look at final N hours before event

"""
Z-SCORE DEFINITION:
------------------
Z-score is computed as: (peak - baseline_mean) / baseline_std

Where baseline_mean and baseline_std are computed over the first 7 days.

Formula:
    z = (peak - mean(baseline_window)) / std(baseline_window)
"""

"""
SUCCESS CRITERIA:
----------------
An earthquake detection is considered SUCCESSFUL if ALL of the following:
1. First detection occurs at least 24 hours before mainshock
2. Amplification > 5× baseline
3. Z-score > 2.0 (statistically significant)

Note: We report BOTH "first detection" (earliest signal) AND 
"72h window metrics" (operational relevance) separately.
"""

MIN_LEAD_TIME_HOURS = 24
MIN_AMPLIFICATION = 5.0
MIN_ZSCORE = 2.0


@dataclass
class CanonicalMetrics:
    """Canonical metrics for a single earthquake validation."""
    
    # Event identification
    earthquake_key: str
    mainshock_time: datetime
    
    # Baseline metrics (first 7 days)
    baseline_median: float
    baseline_mean: float
    baseline_std: float
    
    # Peak metrics (entire window)
    peak_lambda_geo: float
    peak_time: datetime
    
    # Detection metrics
    first_detection_time: Optional[datetime]
    first_detection_hours_before: Optional[float]
    detection_sustained: bool  # Was detection sustained for required days?
    
    # 72h window metrics
    peak_72h: float
    amplification_72h: float
    
    # Statistical metrics
    z_score: float
    
    # Success determination
    success: bool
    success_reasons: List[str]
    
    def to_dict(self) -> dict:
        return {
            'earthquake_key': self.earthquake_key,
            'mainshock_time': self.mainshock_time.isoformat(),
            'baseline': {
                'median': self.baseline_median,
                'mean': self.baseline_mean,
                'std': self.baseline_std,
                'days': BASELINE_DAYS,
            },
            'peak': {
                'value': self.peak_lambda_geo,
                'time': self.peak_time.isoformat() if self.peak_time else None,
            },
            'detection': {
                'first_time': self.first_detection_time.isoformat() if self.first_detection_time else None,
                'hours_before': self.first_detection_hours_before,
                'sustained': self.detection_sustained,
                'threshold_factor': DETECTION_THRESHOLD_FACTOR,
                'sustained_days_required': DETECTION_SUSTAINED_DAYS,
            },
            'amplification': {
                'value': self.amplification_72h,
                'window_hours': AMPLIFICATION_WINDOW_HOURS,
                'definition': f'peak_72h / baseline_median',
            },
            'z_score': self.z_score,
            'success': {
                'status': self.success,
                'criteria': {
                    'min_lead_time_hours': MIN_LEAD_TIME_HOURS,
                    'min_amplification': MIN_AMPLIFICATION,
                    'min_zscore': MIN_ZSCORE,
                },
                'reasons': self.success_reasons,
            }
        }


def compute_canonical_metrics(
    times: np.ndarray,
    lambda_geo: np.ndarray,
    mainshock_time: datetime,
    earthquake_key: str
) -> CanonicalMetrics:
    """
    Compute canonical metrics using standardized definitions.
    
    Args:
        times: Array of datetime objects for each time step
        lambda_geo: Array of shape (n_times, n_spatial) with Λ_geo values
        mainshock_time: Datetime of the mainshock
        earthquake_key: String identifier for the earthquake
        
    Returns:
        CanonicalMetrics object with all standardized metrics
    """
    
    # Convert times to datetime if needed
    times_dt = []
    for t in times:
        if isinstance(t, datetime):
            times_dt.append(t)
        else:
            times_dt.append(datetime.fromisoformat(str(t).replace('T', ' ').split('.')[0]))
    
    # Compute spatial maximum at each time
    lambda_max = np.nanmax(lambda_geo, axis=1)
    n_times = len(times_dt)
    
    # =========================================================================
    # BASELINE COMPUTATION (first 7 days)
    # =========================================================================
    baseline_indices = min(BASELINE_DAYS, n_times // 2)
    baseline_values = lambda_max[:baseline_indices]
    
    baseline_median = float(np.nanmedian(baseline_values))
    baseline_mean = float(np.nanmean(baseline_values))
    baseline_std = float(np.nanstd(baseline_values))
    
    # Ensure non-zero baseline
    if baseline_median <= 0:
        baseline_median = 1e-10
    if baseline_std <= 0:
        baseline_std = baseline_median * 0.1
    
    # =========================================================================
    # PEAK COMPUTATION (entire window)
    # =========================================================================
    peak_idx = np.nanargmax(lambda_max)
    peak_lambda_geo = float(lambda_max[peak_idx])
    peak_time = times_dt[peak_idx]
    
    # =========================================================================
    # FIRST DETECTION (2× baseline, sustained for 2 days)
    # =========================================================================
    threshold = DETECTION_THRESHOLD_FACTOR * baseline_median
    
    first_detection_time = None
    first_detection_hours_before = None
    detection_sustained = False
    
    for i in range(n_times - DETECTION_SUSTAINED_DAYS + 1):
        # Check if this and next N-1 days exceed threshold
        sustained = True
        for j in range(DETECTION_SUSTAINED_DAYS):
            if i + j >= n_times or lambda_max[i + j] <= threshold:
                sustained = False
                break
        
        if sustained:
            first_detection_time = times_dt[i]
            first_detection_hours_before = (mainshock_time - first_detection_time).total_seconds() / 3600
            detection_sustained = True
            break
    
    # =========================================================================
    # 72-HOUR WINDOW METRICS
    # =========================================================================
    # Find indices within 72 hours of mainshock
    window_72h_start = mainshock_time - timedelta(hours=AMPLIFICATION_WINDOW_HOURS)
    
    in_72h_window = []
    for i, t in enumerate(times_dt):
        if window_72h_start <= t <= mainshock_time:
            in_72h_window.append(i)
    
    if in_72h_window:
        peak_72h = float(np.nanmax(lambda_max[in_72h_window]))
    else:
        # If no data in 72h window, use last 3 days (for daily data)
        peak_72h = float(np.nanmax(lambda_max[-3:]))
    
    amplification_72h = peak_72h / baseline_median
    
    # =========================================================================
    # Z-SCORE
    # =========================================================================
    z_score = (peak_72h - baseline_mean) / baseline_std if baseline_std > 0 else 0
    
    # =========================================================================
    # SUCCESS DETERMINATION
    # =========================================================================
    success_reasons = []
    
    # Check lead time
    lead_time_ok = (first_detection_hours_before is not None and 
                   first_detection_hours_before >= MIN_LEAD_TIME_HOURS)
    if lead_time_ok:
        success_reasons.append(f"Lead time {first_detection_hours_before:.1f}h >= {MIN_LEAD_TIME_HOURS}h")
    else:
        success_reasons.append(f"FAIL: Lead time insufficient or no detection")
    
    # Check amplification
    amp_ok = amplification_72h >= MIN_AMPLIFICATION
    if amp_ok:
        success_reasons.append(f"Amplification {amplification_72h:.1f}× >= {MIN_AMPLIFICATION}×")
    else:
        success_reasons.append(f"FAIL: Amplification {amplification_72h:.1f}× < {MIN_AMPLIFICATION}×")
    
    # Check Z-score
    z_ok = z_score >= MIN_ZSCORE
    if z_ok:
        success_reasons.append(f"Z-score {z_score:.2f} >= {MIN_ZSCORE}")
    else:
        success_reasons.append(f"FAIL: Z-score {z_score:.2f} < {MIN_ZSCORE}")
    
    success = lead_time_ok and amp_ok and z_ok
    
    return CanonicalMetrics(
        earthquake_key=earthquake_key,
        mainshock_time=mainshock_time,
        baseline_median=baseline_median,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        peak_lambda_geo=peak_lambda_geo,
        peak_time=peak_time,
        first_detection_time=first_detection_time,
        first_detection_hours_before=first_detection_hours_before,
        detection_sustained=detection_sustained,
        peak_72h=peak_72h,
        amplification_72h=amplification_72h,
        z_score=z_score,
        success=success,
        success_reasons=success_reasons,
    )


def run_null_test(
    times: np.ndarray,
    lambda_geo: np.ndarray,
    n_windows: int = 100,
    window_days: int = 14
) -> dict:
    """
    Run null hypothesis test: how often does Λ_geo exceed thresholds
    in random windows with no known earthquake?
    
    Args:
        times: Full time series of datetimes
        lambda_geo: Full Λ_geo array (n_times, n_spatial)
        n_windows: Number of random windows to test
        window_days: Size of each test window in days
        
    Returns:
        dict with false positive rates and threshold statistics
    """
    
    lambda_max = np.nanmax(lambda_geo, axis=1)
    n_times = len(times)
    
    if n_times < window_days * 2:
        return {'error': 'Insufficient data for null test'}
    
    # Track threshold crossings
    fp_2x = 0  # False positives at 2× baseline
    fp_5x = 0  # False positives at 5× baseline
    fp_10x = 0  # False positives at 10× baseline
    
    valid_windows = 0
    
    for _ in range(n_windows):
        # Random start point
        start_idx = np.random.randint(0, n_times - window_days)
        end_idx = start_idx + window_days
        
        window_data = lambda_max[start_idx:end_idx]
        
        # Compute baseline (first 7 days of window)
        baseline_days = min(7, len(window_data) // 2)
        baseline = np.nanmedian(window_data[:baseline_days])
        
        if baseline <= 0:
            continue
        
        valid_windows += 1
        
        # Check for threshold crossings in latter half of window
        test_data = window_data[baseline_days:]
        peak = np.nanmax(test_data)
        
        amp = peak / baseline
        
        if amp >= 2.0:
            fp_2x += 1
        if amp >= 5.0:
            fp_5x += 1
        if amp >= 10.0:
            fp_10x += 1
    
    if valid_windows == 0:
        return {'error': 'No valid windows'}
    
    return {
        'n_windows_tested': valid_windows,
        'false_positive_rate': {
            '2x_threshold': fp_2x / valid_windows,
            '5x_threshold': fp_5x / valid_windows,
            '10x_threshold': fp_10x / valid_windows,
        },
        'expected_alerts_per_year': {
            '2x_threshold': fp_2x / valid_windows * 365 / window_days,
            '5x_threshold': fp_5x / valid_windows * 365 / window_days,
            '10x_threshold': fp_10x / valid_windows * 365 / window_days,
        }
    }


if __name__ == "__main__":
    print("Canonical Metric Definitions")
    print("=" * 60)
    print(f"Baseline: {BASELINE_STAT} of first {BASELINE_DAYS} days (spatial max)")
    print(f"Detection threshold: {DETECTION_THRESHOLD_FACTOR}× baseline")
    print(f"Detection sustained: {DETECTION_SUSTAINED_DAYS} consecutive days")
    print(f"Amplification window: final {AMPLIFICATION_WINDOW_HOURS} hours")
    print(f"Success criteria:")
    print(f"  - Lead time >= {MIN_LEAD_TIME_HOURS} hours")
    print(f"  - Amplification >= {MIN_AMPLIFICATION}×")
    print(f"  - Z-score >= {MIN_ZSCORE}")
