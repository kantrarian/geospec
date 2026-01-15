#!/usr/bin/env python3
"""
canonical_analysis.py
Re-analyze all earthquakes with canonical metric definitions.

This produces consistent numbers across all documentation.

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from metrics_definitions import (
    compute_canonical_metrics, 
    run_null_test,
    CanonicalMetrics,
    BASELINE_DAYS,
    DETECTION_THRESHOLD_FACTOR,
    DETECTION_SUSTAINED_DAYS,
    AMPLIFICATION_WINDOW_HOURS,
    MIN_LEAD_TIME_HOURS,
    MIN_AMPLIFICATION,
    MIN_ZSCORE,
)

# Earthquake mainshock times (UTC)
MAINSHOCK_TIMES = {
    'tohoku_2011': datetime(2011, 3, 11, 5, 46, 24),
    'chile_2010': datetime(2010, 2, 27, 6, 34, 14),
    'turkey_2023': datetime(2023, 2, 6, 1, 17, 35),
    'ridgecrest_2019': datetime(2019, 7, 6, 3, 19, 53),
    'morocco_2023': datetime(2023, 9, 8, 22, 11, 1),
}

# Foreshock info
FORESHOCK_INFO = {
    'tohoku_2011': {'had_foreshock': True, 'foreshock_mag': 7.2, 'hours_before': 51},
    'chile_2010': {'had_foreshock': False},
    'turkey_2023': {'had_foreshock': False},
    'ridgecrest_2019': {'had_foreshock': True, 'foreshock_mag': 6.4, 'hours_before': 34},
    'morocco_2023': {'had_foreshock': False},
}

# Event metadata
EVENT_INFO = {
    'tohoku_2011': {'name': '2011 Tohoku', 'magnitude': 9.0, 'setting': 'Subduction'},
    'chile_2010': {'name': '2010 Chile', 'magnitude': 8.8, 'setting': 'Subduction'},
    'turkey_2023': {'name': '2023 Turkey', 'magnitude': 7.8, 'setting': 'Continental'},
    'ridgecrest_2019': {'name': '2019 Ridgecrest', 'magnitude': 7.1, 'setting': 'Transform'},
    'morocco_2023': {'name': '2023 Morocco', 'magnitude': 6.8, 'setting': 'Intracontinental'},
}


def analyze_all_earthquakes(results_dir: Path) -> dict:
    """
    Analyze all earthquakes with canonical metrics.
    
    Returns dict with all metrics in standardized format.
    """
    
    all_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'definitions': {
                'baseline': f'{BASELINE_DAYS} days, median of spatial max',
                'detection_threshold': f'{DETECTION_THRESHOLD_FACTOR}× baseline',
                'detection_sustained': f'{DETECTION_SUSTAINED_DAYS} consecutive days',
                'amplification_window': f'final {AMPLIFICATION_WINDOW_HOURS} hours',
                'success_criteria': {
                    'min_lead_time_hours': MIN_LEAD_TIME_HOURS,
                    'min_amplification': MIN_AMPLIFICATION,
                    'min_zscore': MIN_ZSCORE,
                }
            }
        },
        'earthquakes': {}
    }
    
    print("=" * 70)
    print("CANONICAL METRIC ANALYSIS")
    print("=" * 70)
    print(f"Baseline: median of first {BASELINE_DAYS} days (spatial max)")
    print(f"Detection: {DETECTION_THRESHOLD_FACTOR}x baseline, sustained {DETECTION_SUSTAINED_DAYS} days")
    print(f"Amplification: peak in final {AMPLIFICATION_WINDOW_HOURS}h / baseline")
    print(f"Success: lead>={MIN_LEAD_TIME_HOURS}h, amp>={MIN_AMPLIFICATION}x, Z>={MIN_ZSCORE}")
    print("=" * 70)
    print()
    
    for eq_key in MAINSHOCK_TIMES.keys():
        results_file = results_dir / f"{eq_key}_lambda_geo.npz"
        
        if not results_file.exists():
            print(f"[SKIP] {eq_key}: No results file")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {EVENT_INFO[eq_key]['name']} (M{EVENT_INFO[eq_key]['magnitude']})")
        print(f"{'='*60}")
        
        # Load results
        data = np.load(results_file, allow_pickle=True)
        times = data['times']
        lambda_geo = data['lambda_geo']
        
        # Compute canonical metrics
        metrics = compute_canonical_metrics(
            times=times,
            lambda_geo=lambda_geo,
            mainshock_time=MAINSHOCK_TIMES[eq_key],
            earthquake_key=eq_key
        )
        
        # Print results
        print(f"\nBaseline (first {BASELINE_DAYS} days):")
        print(f"  Median: {metrics.baseline_median:.6f}")
        print(f"  Mean:   {metrics.baseline_mean:.6f}")
        print(f"  Std:    {metrics.baseline_std:.6f}")
        
        print(f"\nPeak (entire window):")
        print(f"  Value: {metrics.peak_lambda_geo:.4f}")
        print(f"  Time:  {metrics.peak_time}")
        
        print(f"\nFirst Detection ({DETECTION_THRESHOLD_FACTOR}x baseline, {DETECTION_SUSTAINED_DAYS}+ days):")
        if metrics.first_detection_time:
            print(f"  Time:  {metrics.first_detection_time}")
            print(f"  Lead:  {metrics.first_detection_hours_before:.1f} hours ({metrics.first_detection_hours_before/24:.1f} days)")
            print(f"  Sustained: {metrics.detection_sustained}")
        else:
            print(f"  No sustained detection above threshold")
        
        print(f"\n72-hour Window Metrics:")
        print(f"  Peak in 72h window: {metrics.peak_72h:.4f}")
        print(f"  Amplification: {metrics.amplification_72h:.1f}x")
        
        print(f"\nStatistical:")
        print(f"  Z-score: {metrics.z_score:.2f}")
        
        print(f"\nSuccess Determination:")
        for reason in metrics.success_reasons:
            status = "[OK]" if not reason.startswith("FAIL") else "[X]"
            print(f"  {status} {reason}")
        print(f"\n  OVERALL: {'SUCCESS' if metrics.success else 'FAILURE'}")
        
        # Store results
        result_dict = metrics.to_dict()
        result_dict['event_info'] = EVENT_INFO[eq_key]
        result_dict['foreshock_info'] = FORESHOCK_INFO[eq_key]
        all_results['earthquakes'][eq_key] = result_dict
    
    return all_results


def print_summary_table(results: dict):
    """Print a summary table of all results."""
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (Canonical Metrics)")
    print("=" * 100)
    
    header = f"{'Event':<20} {'Mag':>4} {'Baseline':>10} {'Peak72h':>10} {'Amp':>8} {'Z-score':>8} {'Lead(h)':>8} {'Success':>8}"
    print(header)
    print("-" * 100)
    
    successes = 0
    
    for eq_key, data in results['earthquakes'].items():
        name = data['event_info']['name']
        mag = data['event_info']['magnitude']
        baseline = data['baseline']['median']
        peak72h = data['amplification']['value'] * baseline  # Reconstruct peak
        amp = data['amplification']['value']
        z = data['z_score']
        lead = data['detection']['hours_before']
        lead_str = f"{lead:.1f}" if lead else "N/A"
        success = "YES" if data['success']['status'] else "NO"
        
        if data['success']['status']:
            successes += 1
        
        print(f"{name:<20} {mag:>4.1f} {baseline:>10.4f} {peak72h:>10.4f} {amp:>7.1f}x {z:>8.1f} {lead_str:>8} {success:>8}")
    
    print("-" * 100)
    total = len(results['earthquakes'])
    print(f"Success Rate: {successes}/{total} ({100*successes/total:.0f}%)")
    print("=" * 100)


def print_foreshock_analysis(results: dict):
    """Analyze detection relative to foreshocks where applicable."""
    
    print("\n" + "=" * 80)
    print("FORESHOCK TIMING ANALYSIS")
    print("=" * 80)
    
    for eq_key, data in results['earthquakes'].items():
        foreshock = data['foreshock_info']
        
        if foreshock.get('had_foreshock'):
            lead = data['detection']['hours_before']
            fs_hours = foreshock['hours_before']
            fs_mag = foreshock['foreshock_mag']
            
            if lead and lead > fs_hours:
                before_fs = lead - fs_hours
                print(f"\n{data['event_info']['name']}:")
                print(f"  M{fs_mag} foreshock: {fs_hours}h before mainshock")
                print(f"  First Lambda_geo detection: {lead:.1f}h before mainshock")
                print(f"  -> Detection was {before_fs:.1f}h BEFORE foreshock")
                print(f"  -> Lambda_geo detected pre-seismic strain buildup")
            else:
                print(f"\n{data['event_info']['name']}:")
                print(f"  M{fs_mag} foreshock: {fs_hours}h before mainshock")
                print(f"  First Lambda_geo detection: {lead:.1f}h before mainshock" if lead else "  No detection")
                print(f"  -> Detection was after or concurrent with foreshock")
        else:
            lead = data['detection']['hours_before']
            print(f"\n{data['event_info']['name']}:")
            print(f"  No foreshocks detected by seismology")
            if lead:
                print(f"  First Lambda_geo detection: {lead:.1f}h ({lead/24:.1f} days) before mainshock")
                print(f"  -> Lambda_geo detected INVISIBLE precursor")
            else:
                print(f"  No Lambda_geo detection above threshold")


def main():
    """Run canonical analysis on all earthquakes."""
    
    results_dir = Path(__file__).parent.parent / "results"
    
    # Run analysis
    results = analyze_all_earthquakes(results_dir)
    
    # Print summary
    print_summary_table(results)
    
    # Print foreshock analysis
    print_foreshock_analysis(results)
    
    # Save canonical results
    output_file = results_dir / "canonical_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nCanonical metrics saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
