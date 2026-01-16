#!/usr/bin/env python3
"""
test_fault_correlation_ridgecrest.py
Validate fault correlation dynamics on 2019 Ridgecrest earthquake sequence.

2019 Ridgecrest Sequence:
- M6.4 foreshock: 2019-07-04 17:33:49 UTC
- M7.1 mainshock: 2019-07-06 03:19:53 UTC

Hypothesis: Fault segments DECOUPLE before rupture, observable as:
- L2/L1 eigenvalue ratio DROPS
- Participation ratio approaches 1.0 (stress concentration)

Test Windows:
1. Baseline: June 2019 (quiet period, normal correlation)
2. Pre-foreshock: July 4, 2019 12:00 UTC (5 hours before M6.4)
3. Between shocks: July 5, 2019 12:00 UTC (15 hours before M7.1)

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monitoring', 'src'))

from datetime import datetime, timedelta
from fault_correlation import FaultCorrelationMonitor, CorrelationResult
import numpy as np
import json
from pathlib import Path


def run_ridgecrest_validation():
    """
    Run comprehensive Ridgecrest fault correlation validation.

    Returns results for multiple time windows to demonstrate
    precursor detection capability.
    """
    print("=" * 70)
    print("RIDGECREST FAULT CORRELATION VALIDATION")
    print("=" * 70)
    print()
    print("2019 Ridgecrest Earthquake Sequence:")
    print("  M6.4 Foreshock: 2019-07-04 17:33:49 UTC")
    print("  M7.1 Mainshock: 2019-07-06 03:19:53 UTC")
    print()
    print("Hypothesis: L2/L1 eigenvalue ratio DROPS before rupture")
    print("-" * 70)

    # Initialize monitor with appropriate parameters
    monitor = FaultCorrelationMonitor(
        window_hours=24,
        step_hours=6,
        decorrelation_threshold=0.3
    )

    results = {}

    # Test Window 1: Baseline (quiet period)
    print("\n[1] BASELINE PERIOD (June 20, 2019)")
    print("    Expected: Normal correlation (L2/L1 > 0.3)")
    baseline_date = datetime(2019, 6, 20, 12, 0)
    try:
        baseline_result = monitor.analyze_region('ridgecrest', baseline_date)
        results['baseline'] = baseline_result
        if baseline_result.segment_names:
            print(f"    Segments: {len(baseline_result.segment_names)}")
            if len(baseline_result.eigenvalue_ratios) > 0:
                print(f"    L2/L1 ratio: {baseline_result.eigenvalue_ratios[0]:.4f}")
            print(f"    Participation ratio: {baseline_result.participation_ratio:.2f}")
            print(f"    Decorrelated: {baseline_result.is_decorrelated}")
        else:
            print("    SKIPPED: Insufficient data for baseline period")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Test Window 2: Pre-foreshock (5 hours before M6.4)
    print("\n[2] PRE-FORESHOCK (July 4, 2019 12:00 UTC - 5 hrs before M6.4)")
    print("    Expected: Decorrelation (L2/L1 < 0.3)")
    pre_foreshock_date = datetime(2019, 7, 4, 12, 0)
    try:
        pre_foreshock_result = monitor.analyze_region('ridgecrest', pre_foreshock_date)
        results['pre_foreshock'] = pre_foreshock_result
        if pre_foreshock_result.segment_names:
            print(f"    Segments: {len(pre_foreshock_result.segment_names)}")
            if len(pre_foreshock_result.eigenvalue_ratios) > 0:
                print(f"    L2/L1 ratio: {pre_foreshock_result.eigenvalue_ratios[0]:.4f}")
            print(f"    Participation ratio: {pre_foreshock_result.participation_ratio:.2f}")
            print(f"    Decorrelated: {pre_foreshock_result.is_decorrelated}")
        else:
            print("    SKIPPED: Insufficient data")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Test Window 3: Between shocks (15 hours before M7.1)
    print("\n[3] BETWEEN SHOCKS (July 5, 2019 12:00 UTC - 15 hrs before M7.1)")
    print("    Expected: Strong decorrelation (stress transfer)")
    between_date = datetime(2019, 7, 5, 12, 0)
    try:
        between_result = monitor.analyze_region('ridgecrest', between_date)
        results['between_shocks'] = between_result
        if between_result.segment_names:
            print(f"    Segments: {len(between_result.segment_names)}")
            if len(between_result.eigenvalue_ratios) > 0:
                print(f"    L2/L1 ratio: {between_result.eigenvalue_ratios[0]:.4f}")
            print(f"    Participation ratio: {between_result.participation_ratio:.2f}")
            print(f"    Decorrelated: {between_result.is_decorrelated}")
        else:
            print("    SKIPPED: Insufficient data")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    success_count = 0
    total_tests = 0

    if 'pre_foreshock' in results and results['pre_foreshock'].segment_names:
        total_tests += 1
        if results['pre_foreshock'].is_decorrelated:
            print("[PASS] Pre-foreshock decorrelation detected")
            success_count += 1
        else:
            print("[FAIL] Pre-foreshock should show decorrelation")

    if 'between_shocks' in results and results['between_shocks'].segment_names:
        total_tests += 1
        if results['between_shocks'].is_decorrelated:
            print("[PASS] Between-shocks decorrelation detected")
            success_count += 1
        else:
            print("[FAIL] Between-shocks should show decorrelation")

    if total_tests > 0:
        print(f"\nResult: {success_count}/{total_tests} tests passed")

        # Calculate lead times if detected
        if 'pre_foreshock' in results and results['pre_foreshock'].is_decorrelated:
            foreshock_time = datetime(2019, 7, 4, 17, 33, 49)
            lead_time = foreshock_time - pre_foreshock_date
            print(f"\nPre-foreshock lead time: {lead_time.total_seconds() / 3600:.1f} hours")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_json = {
        'validation': 'ridgecrest_fault_correlation',
        'date_run': datetime.now().isoformat(),
        'events': {
            'foreshock': {'time': '2019-07-04T17:33:49', 'magnitude': 6.4},
            'mainshock': {'time': '2019-07-06T03:19:53', 'magnitude': 7.1},
        },
        'results': {}
    }

    for name, result in results.items():
        if result.segment_names:
            results_json['results'][name] = result.to_dict()

    with open(output_dir / 'ridgecrest_correlation_validation.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'ridgecrest_correlation_validation.json'}")

    return results


def run_evolution_analysis():
    """
    Run time evolution analysis showing L2/L1 ratio dropping before earthquakes.
    """
    print("\n" + "=" * 70)
    print("TIME EVOLUTION ANALYSIS")
    print("=" * 70)

    monitor = FaultCorrelationMonitor(
        window_hours=24,
        step_hours=12,  # Every 12 hours
        decorrelation_threshold=0.3
    )

    # Analyze from June 30 to July 6 (covers both earthquakes)
    start_date = datetime(2019, 7, 1, 0, 0)
    end_date = datetime(2019, 7, 5, 0, 0)

    print(f"\nAnalyzing {start_date.date()} to {end_date.date()}")
    print("(This may take a few minutes...)")

    results = monitor.compute_evolution('ridgecrest', start_date, end_date)

    print("\n" + "-" * 70)
    print(f"{'Date/Time':<20} {'L2/L1':>10} {'Part.Ratio':>12} {'Decorr':>10}")
    print("-" * 70)

    for r in results:
        if r.segment_names and len(r.eigenvalue_ratios) > 0:
            decorr = "YES" if r.is_decorrelated else "no"
            print(f"{r.date.strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{r.eigenvalue_ratios[0]:>10.4f} "
                  f"{r.participation_ratio:>12.2f} "
                  f"{decorr:>10}")

    # Mark earthquake times
    print("-" * 70)
    print("Earthquake times:")
    print("  M6.4: 2019-07-04 17:33 UTC")
    print("  M7.1: 2019-07-06 03:19 UTC")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ridgecrest Fault Correlation Validation')
    parser.add_argument('--evolution', action='store_true',
                       help='Run time evolution analysis')

    args = parser.parse_args()

    results = run_ridgecrest_validation()

    if args.evolution:
        evolution = run_evolution_analysis()
