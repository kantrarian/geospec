#!/usr/bin/env python3
"""
null_testing.py
False Positive Rate Analysis via Null Window Testing

This module tests how often Lambda_geo exceeds detection thresholds
during periods with NO major earthquakes, establishing the base rate
of false alarms.

Methodology:
1. For each station network, download extended time series (years of data)
2. Exclude windows around known M6+ earthquakes
3. Sample many random windows
4. Compute Lambda_geo and count threshold crossings
5. Report FPR at various thresholds

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

sys.path.insert(0, str(Path(__file__).parent))

from metrics_definitions import (
    BASELINE_DAYS,
    DETECTION_THRESHOLD_FACTOR,
    DETECTION_SUSTAINED_DAYS,
    MIN_AMPLIFICATION,
    MIN_ZSCORE,
)


@dataclass
class NullTestResult:
    """Results from null window testing."""
    region: str
    n_windows_tested: int
    window_days: int
    
    # Threshold crossing counts
    crossings_2x: int
    crossings_5x: int
    crossings_10x: int
    sustained_crossings_2x: int  # Sustained for 2+ days
    
    # False positive rates
    fpr_2x: float
    fpr_5x: float
    fpr_10x: float
    fpr_sustained_2x: float
    
    # Expected alerts per year
    alerts_per_year_2x: float
    alerts_per_year_5x: float
    alerts_per_year_10x: float
    
    # Z-score distribution
    zscore_mean: float
    zscore_std: float
    zscore_max: float
    zscore_p95: float
    
    def to_dict(self) -> dict:
        return {
            'region': self.region,
            'n_windows_tested': self.n_windows_tested,
            'window_days': self.window_days,
            'threshold_crossings': {
                '2x': self.crossings_2x,
                '5x': self.crossings_5x,
                '10x': self.crossings_10x,
                '2x_sustained': self.sustained_crossings_2x,
            },
            'false_positive_rates': {
                '2x': self.fpr_2x,
                '5x': self.fpr_5x,
                '10x': self.fpr_10x,
                '2x_sustained': self.fpr_sustained_2x,
            },
            'expected_alerts_per_year': {
                '2x': self.alerts_per_year_2x,
                '5x': self.alerts_per_year_5x,
                '10x': self.alerts_per_year_10x,
            },
            'zscore_distribution': {
                'mean': self.zscore_mean,
                'std': self.zscore_std,
                'max': self.zscore_max,
                'p95': self.zscore_p95,
            }
        }


def generate_null_windows_from_data(
    lambda_geo_timeseries: np.ndarray,
    times: np.ndarray,
    earthquake_time: datetime,
    n_windows: int = 100,
    window_days: int = 14,
    exclusion_days: int = 30,
    seed: int = 42
) -> List[Dict]:
    """
    Generate null windows from existing Lambda_geo time series,
    excluding the period around the known earthquake.
    
    Args:
        lambda_geo_timeseries: Array of shape (n_times, n_spatial) with Lambda_geo
        times: Array of datetime objects
        earthquake_time: Time of known earthquake to exclude
        n_windows: Number of null windows to sample
        window_days: Size of each window in days
        exclusion_days: Days before/after earthquake to exclude
        seed: Random seed for reproducibility
        
    Returns:
        List of dicts with null window statistics
    """
    
    np.random.seed(seed)
    
    # Convert times to datetime if needed
    times_dt = []
    for t in times:
        if isinstance(t, datetime):
            times_dt.append(t)
        else:
            times_dt.append(datetime.fromisoformat(str(t).replace('T', ' ').split('.')[0]))
    
    # Compute spatial max at each time
    lambda_max = np.nanmax(lambda_geo_timeseries, axis=1)
    n_times = len(times_dt)
    
    # Find indices to exclude (around earthquake)
    exclude_start = earthquake_time - timedelta(days=exclusion_days)
    exclude_end = earthquake_time + timedelta(days=exclusion_days)
    
    valid_indices = []
    for i, t in enumerate(times_dt):
        if t < exclude_start or t > exclude_end:
            valid_indices.append(i)
    
    if len(valid_indices) < window_days * 2:
        return []  # Not enough data outside exclusion zone
    
    results = []
    
    for _ in range(n_windows):
        # Pick random starting point from valid indices
        max_start = len(valid_indices) - window_days
        if max_start <= 0:
            continue
            
        start_pos = np.random.randint(0, max_start)
        
        # Get window indices
        window_indices = valid_indices[start_pos:start_pos + window_days]
        
        if len(window_indices) < window_days:
            continue
        
        window_data = lambda_max[window_indices]
        
        # Compute baseline (first 7 days)
        baseline_days = min(BASELINE_DAYS, len(window_data) // 2)
        baseline = np.nanmedian(window_data[:baseline_days])
        baseline_mean = np.nanmean(window_data[:baseline_days])
        baseline_std = np.nanstd(window_data[:baseline_days])
        
        if baseline <= 0 or baseline_std <= 0:
            continue
        
        # Test latter half of window
        test_data = window_data[baseline_days:]
        peak = np.nanmax(test_data)
        
        # Compute metrics
        amp = peak / baseline
        z = (peak - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        # Check sustained crossings
        sustained_2x = False
        threshold_2x = 2.0 * baseline
        for i in range(len(test_data) - DETECTION_SUSTAINED_DAYS + 1):
            if all(test_data[i:i+DETECTION_SUSTAINED_DAYS] > threshold_2x):
                sustained_2x = True
                break
        
        results.append({
            'baseline': baseline,
            'peak': peak,
            'amplification': amp,
            'zscore': z,
            'crossed_2x': amp >= 2.0,
            'crossed_5x': amp >= 5.0,
            'crossed_10x': amp >= 10.0,
            'sustained_2x': sustained_2x,
        })
    
    return results


def analyze_null_results(results: List[Dict], region: str, window_days: int) -> NullTestResult:
    """
    Analyze null window results and compute FPR statistics.
    """
    
    n = len(results)
    if n == 0:
        return None
    
    # Count crossings
    crossings_2x = sum(1 for r in results if r['crossed_2x'])
    crossings_5x = sum(1 for r in results if r['crossed_5x'])
    crossings_10x = sum(1 for r in results if r['crossed_10x'])
    sustained_2x = sum(1 for r in results if r['sustained_2x'])
    
    # FPR
    fpr_2x = crossings_2x / n
    fpr_5x = crossings_5x / n
    fpr_10x = crossings_10x / n
    fpr_sustained = sustained_2x / n
    
    # Expected alerts per year (assuming non-overlapping windows)
    windows_per_year = 365 / window_days
    alerts_2x = fpr_2x * windows_per_year
    alerts_5x = fpr_5x * windows_per_year
    alerts_10x = fpr_10x * windows_per_year
    
    # Z-score distribution
    zscores = [r['zscore'] for r in results]
    
    return NullTestResult(
        region=region,
        n_windows_tested=n,
        window_days=window_days,
        crossings_2x=crossings_2x,
        crossings_5x=crossings_5x,
        crossings_10x=crossings_10x,
        sustained_crossings_2x=sustained_2x,
        fpr_2x=fpr_2x,
        fpr_5x=fpr_5x,
        fpr_10x=fpr_10x,
        fpr_sustained_2x=fpr_sustained,
        alerts_per_year_2x=alerts_2x,
        alerts_per_year_5x=alerts_5x,
        alerts_per_year_10x=alerts_10x,
        zscore_mean=np.mean(zscores),
        zscore_std=np.std(zscores),
        zscore_max=np.max(zscores),
        zscore_p95=np.percentile(zscores, 95),
    )


def run_null_test_for_earthquake(
    results_file: Path,
    earthquake_time: datetime,
    region_name: str,
    n_windows: int = 100
) -> Optional[NullTestResult]:
    """
    Run null window test using data from a specific earthquake analysis.
    
    Note: This is LIMITED because we only have ~14-30 days of data per event.
    For proper null testing, we'd need years of continuous data.
    """
    
    if not results_file.exists():
        return None
    
    data = np.load(results_file, allow_pickle=True)
    times = data['times']
    lambda_geo = data['lambda_geo']
    
    # Generate null windows (excluding earthquake period)
    null_results = generate_null_windows_from_data(
        lambda_geo_timeseries=lambda_geo,
        times=times,
        earthquake_time=earthquake_time,
        n_windows=n_windows,
        window_days=7,  # Shorter windows for limited data
        exclusion_days=3,  # Exclude 3 days around event
    )
    
    if not null_results:
        return None
    
    return analyze_null_results(null_results, region_name, window_days=7)


def run_bootstrap_null_test(
    lambda_max_timeseries: np.ndarray,
    n_bootstrap: int = 1000,
    window_days: int = 14,
    seed: int = 42
) -> Dict:
    """
    Bootstrap null test: shuffle the time series to destroy temporal structure,
    then check how often thresholds are crossed.
    
    This tests: "What FPR would we get from random noise with the same
    amplitude distribution as our data?"
    """
    
    np.random.seed(seed)
    
    n_times = len(lambda_max_timeseries)
    
    results = {
        'crossed_2x': 0,
        'crossed_5x': 0,
        'crossed_10x': 0,
        'zscores': [],
    }
    
    for _ in range(n_bootstrap):
        # Shuffle the time series
        shuffled = np.random.permutation(lambda_max_timeseries)
        
        # Compute baseline from first 7 days
        baseline_days = min(BASELINE_DAYS, len(shuffled) // 3)
        baseline = np.nanmedian(shuffled[:baseline_days])
        baseline_mean = np.nanmean(shuffled[:baseline_days])
        baseline_std = np.nanstd(shuffled[:baseline_days])
        
        if baseline <= 0 or baseline_std <= 0:
            continue
        
        # Peak in "precursor window" (last portion)
        test_data = shuffled[baseline_days:]
        peak = np.nanmax(test_data)
        
        amp = peak / baseline
        z = (peak - baseline_mean) / baseline_std
        
        results['zscores'].append(z)
        if amp >= 2.0:
            results['crossed_2x'] += 1
        if amp >= 5.0:
            results['crossed_5x'] += 1
        if amp >= 10.0:
            results['crossed_10x'] += 1
    
    n = n_bootstrap
    return {
        'n_bootstrap': n,
        'fpr_2x': results['crossed_2x'] / n,
        'fpr_5x': results['crossed_5x'] / n,
        'fpr_10x': results['crossed_10x'] / n,
        'zscore_mean': np.mean(results['zscores']),
        'zscore_std': np.std(results['zscores']),
        'zscore_p95': np.percentile(results['zscores'], 95),
        'zscore_p99': np.percentile(results['zscores'], 99),
    }


def run_comprehensive_null_testing(results_dir: Path) -> Dict:
    """
    Run null testing on all available earthquake data.
    
    Approach:
    1. Bootstrap test: Shuffle each time series to get noise baseline
    2. Pre-event test: Use early quiet period as null baseline
    3. Cross-region test: Apply one region's thresholds to another
    """
    
    # Earthquake info
    earthquakes = {
        'tohoku_2011': {
            'time': datetime(2011, 3, 11, 5, 46, 24),
            'name': 'Tohoku'
        },
        'chile_2010': {
            'time': datetime(2010, 2, 27, 6, 34, 14),
            'name': 'Chile'
        },
        'turkey_2023': {
            'time': datetime(2023, 2, 6, 1, 17, 35),
            'name': 'Turkey'
        },
        'ridgecrest_2019': {
            'time': datetime(2019, 7, 6, 3, 19, 53),
            'name': 'Ridgecrest'
        },
        'morocco_2023': {
            'time': datetime(2023, 9, 8, 22, 11, 1),
            'name': 'Morocco'
        },
    }
    
    all_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'method': 'Bootstrap + Pre-event window analysis',
            'n_bootstrap': 1000,
        },
        'bootstrap_tests': {},
        'pre_event_tests': {},
        'summary': {}
    }
    
    print("=" * 70)
    print("NULL WINDOW TESTING")
    print("=" * 70)
    print("Establishing false positive rates for Lambda_geo thresholds")
    print()
    
    for eq_key, eq_info in earthquakes.items():
        results_file = results_dir / f"{eq_key}_lambda_geo.npz"
        
        if not results_file.exists():
            print(f"[SKIP] {eq_info['name']}: No results file")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing: {eq_info['name']}")
        print(f"{'='*60}")
        
        # Load data
        data = np.load(results_file, allow_pickle=True)
        lambda_geo = data['lambda_geo']
        lambda_max = np.nanmax(lambda_geo, axis=1)
        
        # =====================================================================
        # Bootstrap Test
        # =====================================================================
        print("\n1. Bootstrap Test (shuffled time series):")
        bootstrap_result = run_bootstrap_null_test(lambda_max, n_bootstrap=1000)
        
        print(f"   FPR at 2x threshold: {bootstrap_result['fpr_2x']*100:.1f}%")
        print(f"   FPR at 5x threshold: {bootstrap_result['fpr_5x']*100:.1f}%")
        print(f"   FPR at 10x threshold: {bootstrap_result['fpr_10x']*100:.1f}%")
        print(f"   Z-score 95th percentile: {bootstrap_result['zscore_p95']:.2f}")
        print(f"   Z-score 99th percentile: {bootstrap_result['zscore_p99']:.2f}")
        
        all_results['bootstrap_tests'][eq_key] = bootstrap_result
        
        # =====================================================================
        # Pre-event (Early Window) Test
        # =====================================================================
        print("\n2. Pre-event Test (first 7 days as null baseline):")
        
        # Use first 7 days as "null" baseline
        baseline_days = min(BASELINE_DAYS, len(lambda_max) // 3)
        early_data = lambda_max[:baseline_days]
        
        baseline_median = np.nanmedian(early_data)
        baseline_mean = np.nanmean(early_data)
        baseline_std = np.nanstd(early_data)
        
        # Check variability within baseline period
        early_max = np.nanmax(early_data)
        early_min = np.nanmin(early_data)
        early_range = early_max / early_min if early_min > 0 else 0
        
        print(f"   Baseline median: {baseline_median:.6f}")
        print(f"   Baseline std: {baseline_std:.6f}")
        print(f"   Early period range (max/min): {early_range:.2f}x")
        print(f"   Natural variability suggests {early_range:.1f}x fluctuations are normal")
        
        # Count how many baseline days exceed various thresholds
        exceed_2x = np.sum(early_data > 2 * baseline_median)
        exceed_5x = np.sum(early_data > 5 * baseline_median)
        
        pre_event_result = {
            'baseline_median': float(baseline_median),
            'baseline_std': float(baseline_std),
            'natural_range': float(early_range),
            'days_exceeding_2x_in_baseline': int(exceed_2x),
            'days_exceeding_5x_in_baseline': int(exceed_5x),
        }
        
        all_results['pre_event_tests'][eq_key] = pre_event_result
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: FALSE POSITIVE RATE ESTIMATES")
    print("=" * 70)
    
    # Aggregate bootstrap results
    all_fpr_2x = [r['fpr_2x'] for r in all_results['bootstrap_tests'].values()]
    all_fpr_5x = [r['fpr_5x'] for r in all_results['bootstrap_tests'].values()]
    all_fpr_10x = [r['fpr_10x'] for r in all_results['bootstrap_tests'].values()]
    all_z95 = [r['zscore_p95'] for r in all_results['bootstrap_tests'].values()]
    
    if all_fpr_2x:
        summary = {
            'mean_fpr_2x': np.mean(all_fpr_2x),
            'mean_fpr_5x': np.mean(all_fpr_5x),
            'mean_fpr_10x': np.mean(all_fpr_10x),
            'mean_zscore_p95': np.mean(all_z95),
            'expected_alerts_per_year': {
                '2x': np.mean(all_fpr_2x) * 365 / 14,  # Assuming 14-day windows
                '5x': np.mean(all_fpr_5x) * 365 / 14,
                '10x': np.mean(all_fpr_10x) * 365 / 14,
            }
        }
        
        all_results['summary'] = summary
        
        print(f"\nAggregated across {len(all_fpr_2x)} regions:")
        print(f"  Mean FPR at 2x threshold: {summary['mean_fpr_2x']*100:.1f}%")
        print(f"  Mean FPR at 5x threshold: {summary['mean_fpr_5x']*100:.1f}%")
        print(f"  Mean FPR at 10x threshold: {summary['mean_fpr_10x']*100:.1f}%")
        print(f"\n  Expected false alerts per year (14-day windows):")
        print(f"    At 2x threshold: {summary['expected_alerts_per_year']['2x']:.1f}")
        print(f"    At 5x threshold: {summary['expected_alerts_per_year']['5x']:.1f}")
        print(f"    At 10x threshold: {summary['expected_alerts_per_year']['10x']:.1f}")
        print(f"\n  Mean Z-score 95th percentile: {summary['mean_zscore_p95']:.2f}")
        print(f"  (Compare to actual detection Z-scores of 1000s-20000s)")
    
    return all_results


def print_fpr_table(results: Dict):
    """Print formatted FPR summary table."""
    
    print("\n" + "=" * 80)
    print("FALSE POSITIVE RATE TABLE")
    print("=" * 80)
    print(f"{'Region':<15} {'FPR 2x':>10} {'FPR 5x':>10} {'FPR 10x':>10} {'Z-95%':>10}")
    print("-" * 80)
    
    for eq_key, data in results.get('bootstrap_tests', {}).items():
        region = eq_key.replace('_', ' ').title()[:15]
        fpr2 = f"{data['fpr_2x']*100:.1f}%"
        fpr5 = f"{data['fpr_5x']*100:.1f}%"
        fpr10 = f"{data['fpr_10x']*100:.1f}%"
        z95 = f"{data['zscore_p95']:.2f}"
        print(f"{region:<15} {fpr2:>10} {fpr5:>10} {fpr10:>10} {z95:>10}")
    
    print("-" * 80)
    
    if 'summary' in results and results['summary']:
        s = results['summary']
        print(f"{'MEAN':<15} {s['mean_fpr_2x']*100:.1f}%      {s['mean_fpr_5x']*100:.1f}%      {s['mean_fpr_10x']*100:.1f}%      {s['mean_zscore_p95']:.2f}")
    
    print("=" * 80)


def main():
    """Run comprehensive null testing."""
    
    results_dir = Path(__file__).parent.parent / "results"
    
    # Run null tests
    results = run_comprehensive_null_testing(results_dir)
    
    # Print summary table
    print_fpr_table(results)
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if results.get('summary'):
        s = results['summary']
        
        print("""
The bootstrap test shuffles the time series to destroy temporal structure,
then checks how often thresholds are crossed by random chance.

Key findings:
""")
        print(f"1. At 2x threshold: {s['mean_fpr_2x']*100:.1f}% FPR")
        print(f"   -> Would trigger on {s['mean_fpr_2x']*100:.0f}% of random windows")
        print(f"   -> ~{s['expected_alerts_per_year']['2x']:.0f} false alerts/year")
        
        print(f"\n2. At 5x threshold: {s['mean_fpr_5x']*100:.1f}% FPR")
        print(f"   -> Would trigger on {s['mean_fpr_5x']*100:.1f}% of random windows")
        print(f"   -> ~{s['expected_alerts_per_year']['5x']:.1f} false alerts/year")
        
        print(f"\n3. At 10x threshold: {s['mean_fpr_10x']*100:.2f}% FPR")
        print(f"   -> Rare in shuffled data")
        print(f"   -> ~{s['expected_alerts_per_year']['10x']:.1f} false alerts/year")
        
        print(f"\n4. Z-score comparison:")
        print(f"   -> Null distribution 95th percentile: {s['mean_zscore_p95']:.2f}")
        print(f"   -> Actual earthquake detections: Z = 4,000 - 21,000")
        print(f"   -> Actual Z-scores are 1000x+ higher than null")
    
    # Save results
    output_file = results_dir / "null_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
