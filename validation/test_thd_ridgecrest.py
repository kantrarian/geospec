#!/usr/bin/env python3
"""
test_thd_ridgecrest.py
Validate Seismic THD on 2019 Ridgecrest earthquake sequence.

Physical Hypothesis:
- Rocks respond linearly to Earth tides under normal stress
- Near failure, nonlinear behavior creates harmonics
- THD (Total Harmonic Distortion) INCREASES before failure

Test:
- Fetch continuous seismic data for June 15 - July 6, 2019
- Compute THD time series using 24-hour windows, 6-hour steps
- Show THD evolution leading up to M6.4 and M7.1

Expected Pattern:
- Baseline THD < 0.05 (June 15-25)
- Elevated THD > 0.10 (approaching failure)
- Critical THD > 0.15 (imminent failure)

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monitoring', 'src'))

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json

from seismic_thd import (
    SeismicTHDAnalyzer,
    THDResult,
    fetch_continuous_data_for_thd,
    THD_THRESHOLDS
)


def fetch_ridgecrest_data(
    station: str = 'WBS',
    start: datetime = None,
    end: datetime = None
) -> tuple:
    """
    Fetch continuous seismic data for Ridgecrest validation.

    Args:
        station: Station code (WBS, CCC, SLA)
        start: Start datetime
        end: End datetime

    Returns:
        Tuple of (data, sample_rate, actual_start)
    """
    start = start or datetime(2019, 6, 15, 0, 0)
    end = end or datetime(2019, 7, 6, 12, 0)

    print(f"Fetching CI.{station} data from {start.date()} to {end.date()}")
    print("(This may take several minutes...)")

    data, sample_rate = fetch_continuous_data_for_thd(
        station_network='CI',
        station_code=station,
        start=start,
        end=end,
        channel='BHZ'
    )

    return data, sample_rate, start


def run_thd_validation(
    station: str = 'WBS',
    window_hours: int = 24,
    step_hours: int = 6
):
    """
    Run full THD validation on Ridgecrest data.

    Args:
        station: Station code
        window_hours: Analysis window size
        step_hours: Step between windows
    """
    print("=" * 70)
    print("RIDGECREST THD VALIDATION")
    print("=" * 70)
    print()
    print("2019 Ridgecrest Earthquake Sequence:")
    print("  M6.4 Foreshock: 2019-07-04 17:33:49 UTC")
    print("  M7.1 Mainshock: 2019-07-06 03:19:53 UTC")
    print()
    print("Hypothesis: THD INCREASES before failure (nonlinear rock behavior)")
    print("-" * 70)

    # Key event times
    foreshock_time = datetime(2019, 7, 4, 17, 33, 49)
    mainshock_time = datetime(2019, 7, 6, 3, 19, 53)

    # Fetch data for the analysis period
    # Start from June 20 to capture baseline, end after mainshock
    start_time = datetime(2019, 6, 20, 0, 0)
    end_time = datetime(2019, 7, 6, 6, 0)  # Shortly after M7.1

    data, sample_rate, actual_start = fetch_ridgecrest_data(
        station=station,
        start=start_time,
        end=end_time
    )

    if data is None:
        print(f"\nERROR: Could not fetch data for CI.{station}")
        print("Trying alternative station CI.CCC...")

        data, sample_rate, actual_start = fetch_ridgecrest_data(
            station='CCC',
            start=start_time,
            end=end_time
        )

        if data is None:
            print("ERROR: Could not fetch data for any Ridgecrest station")
            return None

        station = 'CCC'

    print(f"\nRetrieved {len(data)} samples at {sample_rate} Hz")
    print(f"Duration: {len(data)/sample_rate/3600:.1f} hours")

    # Decimate to 1 Hz if needed (THD analysis doesn't need high sample rate)
    if sample_rate > 1.5:
        decimate_factor = int(sample_rate)
        # Use scipy decimate for proper anti-aliasing
        from scipy.signal import decimate
        data = decimate(data, decimate_factor, zero_phase=True)
        sample_rate = sample_rate / decimate_factor
        print(f"Decimated to {sample_rate} Hz ({len(data)} samples)")

    # Initialize THD analyzer
    analyzer = SeismicTHDAnalyzer(
        n_harmonics=5,
        window_hours=window_hours
    )

    # Compute THD time series
    print(f"\nComputing THD time series (window={window_hours}h, step={step_hours}h)...")
    results = analyzer.compute_thd_timeseries(
        data=data,
        sample_rate=sample_rate,
        station=f"CI.{station}",
        start_time=actual_start,
        window_hours=window_hours,
        step_hours=step_hours
    )

    if not results:
        print("ERROR: No THD results computed")
        return None

    # Analyze results
    print(f"\nComputed {len(results)} THD values")

    # Group by period
    baseline_results = [r for r in results if r.date < datetime(2019, 6, 30)]
    pre_foreshock_results = [r for r in results
                            if datetime(2019, 7, 1) <= r.date < foreshock_time]
    post_foreshock_results = [r for r in results
                             if foreshock_time <= r.date < mainshock_time]

    # Calculate statistics
    if baseline_results:
        baseline_thd = np.mean([r.thd_value for r in baseline_results])
        baseline_std = np.std([r.thd_value for r in baseline_results])
    else:
        baseline_thd, baseline_std = 0.0, 0.0

    if pre_foreshock_results:
        pre_foreshock_thd = np.mean([r.thd_value for r in pre_foreshock_results])
        pre_foreshock_max = max(r.thd_value for r in pre_foreshock_results)
    else:
        pre_foreshock_thd, pre_foreshock_max = 0.0, 0.0

    if post_foreshock_results:
        post_foreshock_thd = np.mean([r.thd_value for r in post_foreshock_results])
        post_foreshock_max = max(r.thd_value for r in post_foreshock_results)
    else:
        post_foreshock_thd, post_foreshock_max = 0.0, 0.0

    # Print summary table
    print("\n" + "=" * 70)
    print("THD EVOLUTION SUMMARY")
    print("=" * 70)
    print(f"{'Period':<30} {'Mean THD':>12} {'Max THD':>12} {'Status':>12}")
    print("-" * 70)
    print(f"{'Baseline (Jun 20-30)':<30} {baseline_thd:>12.4f} {'-':>12} "
          f"{'NORMAL' if baseline_thd < 0.05 else 'ELEVATED':>12}")
    print(f"{'Pre-Foreshock (Jul 1-4)':<30} {pre_foreshock_thd:>12.4f} "
          f"{pre_foreshock_max:>12.4f} "
          f"{'NORMAL' if pre_foreshock_thd < 0.05 else 'ELEVATED':>12}")
    print(f"{'Post-Foreshock (Jul 4-6)':<30} {post_foreshock_thd:>12.4f} "
          f"{post_foreshock_max:>12.4f} "
          f"{'CRITICAL' if post_foreshock_thd > 0.15 else 'ELEVATED':>12}")
    print("-" * 70)

    # Detailed time series
    print("\n" + "-" * 70)
    print("THD Time Series (Last 10 Days)")
    print("-" * 70)
    print(f"{'DateTime':<20} {'THD':>10} {'SNR':>10} {'Status':>12}")
    print("-" * 70)

    # Show last ~40 values (last 10 days at 6h steps)
    for r in results[-40:]:
        status = 'CRITICAL' if r.is_critical else ('ELEVATED' if r.is_elevated else 'normal')
        # Mark earthquake times
        marker = ''
        if abs((r.date - foreshock_time).total_seconds()) < step_hours * 3600:
            marker = ' ** M6.4'
        elif abs((r.date - mainshock_time).total_seconds()) < step_hours * 3600:
            marker = ' ** M7.1'

        print(f"{r.date.strftime('%Y-%m-%d %H:%M'):<20} {r.thd_value:>10.4f} "
              f"{r.snr:>10.1f} {status:>12}{marker}")

    # Validation assessment
    print("\n" + "=" * 70)
    print("VALIDATION ASSESSMENT")
    print("=" * 70)

    passed = 0
    total = 0

    # Test 1: THD should increase from baseline to pre-foreshock
    total += 1
    if pre_foreshock_thd > baseline_thd:
        print("[PASS] THD increased from baseline to pre-foreshock period")
        passed += 1
    else:
        print("[FAIL] THD did not increase before foreshock")

    # Test 2: THD should be elevated or critical after foreshock
    total += 1
    if post_foreshock_thd > 0.05:
        print("[PASS] THD elevated in post-foreshock / pre-mainshock period")
        passed += 1
    else:
        print("[INFO] THD not significantly elevated after foreshock")

    # Test 3: Maximum THD should exceed threshold before major event
    total += 1
    max_pre_mainshock = max(r.thd_value for r in results if r.date < mainshock_time)
    if max_pre_mainshock > THD_THRESHOLDS['elevated']:
        print(f"[PASS] Maximum pre-mainshock THD ({max_pre_mainshock:.4f}) exceeded threshold")
        passed += 1
    else:
        print(f"[INFO] Maximum pre-mainshock THD ({max_pre_mainshock:.4f}) below threshold")

    print(f"\nResult: {passed}/{total} tests passed")

    # Calculate lead times
    elevated_results = [r for r in results if r.is_elevated and r.date < foreshock_time]
    if elevated_results:
        first_elevated = min(r.date for r in elevated_results)
        lead_time = foreshock_time - first_elevated
        print(f"\nFirst elevated THD: {first_elevated}")
        print(f"Lead time before M6.4: {lead_time.total_seconds()/3600:.1f} hours")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_json = {
        'validation': 'ridgecrest_thd',
        'station': f'CI.{station}',
        'date_run': datetime.now().isoformat(),
        'parameters': {
            'window_hours': window_hours,
            'step_hours': step_hours,
        },
        'events': {
            'foreshock': {'time': foreshock_time.isoformat(), 'magnitude': 6.4},
            'mainshock': {'time': mainshock_time.isoformat(), 'magnitude': 7.1},
        },
        'summary': {
            'baseline_thd_mean': float(baseline_thd),
            'baseline_thd_std': float(baseline_std),
            'pre_foreshock_thd_mean': float(pre_foreshock_thd),
            'pre_foreshock_thd_max': float(pre_foreshock_max),
            'post_foreshock_thd_mean': float(post_foreshock_thd),
            'post_foreshock_thd_max': float(post_foreshock_max),
        },
        'thd_timeseries': [r.to_dict() for r in results],
        'tests_passed': passed,
        'tests_total': total,
    }

    output_file = output_dir / 'ridgecrest_thd_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def run_quick_validation():
    """
    Run a quick validation with shorter time window for testing.
    """
    print("=" * 70)
    print("QUICK THD VALIDATION (Shortened Period)")
    print("=" * 70)

    # Just test around the foreshock
    start_time = datetime(2019, 7, 1, 0, 0)
    end_time = datetime(2019, 7, 5, 0, 0)

    data, sample_rate, actual_start = fetch_ridgecrest_data(
        station='WBS',
        start=start_time,
        end=end_time
    )

    if data is None:
        print("ERROR: Could not fetch data")
        return

    print(f"Retrieved {len(data)/sample_rate/3600:.1f} hours of data")

    # Decimate
    if sample_rate > 1.5:
        from scipy.signal import decimate
        factor = int(sample_rate)
        data = decimate(data, factor, zero_phase=True)
        sample_rate = sample_rate / factor

    # Analyze
    analyzer = SeismicTHDAnalyzer(window_hours=24)
    results = analyzer.compute_thd_timeseries(
        data, sample_rate, 'CI.WBS', actual_start,
        window_hours=24, step_hours=6
    )

    print(f"\nComputed {len(results)} THD values")
    print("-" * 50)
    for r in results:
        status = 'CRITICAL' if r.is_critical else ('ELEVATED' if r.is_elevated else 'normal')
        print(f"{r.date.strftime('%Y-%m-%d %H:%M')} THD={r.thd_value:.4f} {status}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ridgecrest THD Validation')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation with shorter period')
    parser.add_argument('--station', default='WBS',
                       help='Station code (WBS, CCC, SLA)')
    parser.add_argument('--window', type=int, default=24,
                       help='Window size in hours (default 24)')
    parser.add_argument('--step', type=int, default=6,
                       help='Step size in hours (default 6)')

    args = parser.parse_args()

    if args.quick:
        run_quick_validation()
    else:
        run_thd_validation(
            station=args.station,
            window_hours=args.window,
            step_hours=args.step
        )
