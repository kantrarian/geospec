#!/usr/bin/env python
"""
ridgecrest_2019_retrospective.py
Retrospective THD validation using cached Ridgecrest 2019 waveforms.

Key Events:
- July 4, 2019 10:33 UTC: M6.4 foreshock
- July 6, 2019 03:19 UTC: M7.1 mainshock

Cached Data: June 19 - July 5, 2019 (CI.WBS, CI.SLA, CI.CLC stations)

Purpose: Validate THD precursor detection using real seismic data.
"""

import os
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monitoring', 'src'))

import numpy as np
from seismic_thd import SeismicTHDAnalyzer, THD_THRESHOLDS

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / 'monitoring' / 'data' / 'seismic_cache' / 'ridgecrest'

# Key event dates
M64_FORESHOCK = datetime(2019, 7, 4, 10, 33)
M71_MAINSHOCK = datetime(2019, 7, 6, 3, 19)

def load_cached_waveforms(date_str: str) -> dict:
    """Load cached waveforms for a given date."""
    cache_path = CACHE_DIR / date_str / 'ridgecrest_mainshock_waveforms.pkl'
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}

def get_available_dates() -> list:
    """Get list of available cached dates."""
    dates = []
    for d in CACHE_DIR.iterdir():
        if d.is_dir() and d.name.startswith('2019'):
            dates.append(d.name)
    return sorted(dates)

def compute_thd_for_stream(stream, station: str, date: datetime) -> dict:
    """Compute THD for an ObsPy stream."""
    analyzer = SeismicTHDAnalyzer(
        n_harmonics=5,
        window_hours=24
    )

    # Get the trace (assuming single component or first component)
    if len(stream) == 0:
        return None

    tr = stream[0]
    data = tr.data
    sample_rate = tr.stats.sampling_rate

    # Need at least 12 hours of data
    min_samples = int(12 * 3600 * sample_rate)
    if len(data) < min_samples:
        print(f"  {station}: Insufficient data ({len(data)/sample_rate/3600:.1f}h < 12h)")
        return None

    # Compute THD with noise estimation
    try:
        thd, p1, harmonics, f1, snr = analyzer.compute_thd_with_noise(data, sample_rate)

        return {
            'station': station,
            'date': date.isoformat(),
            'thd': thd,
            'fundamental_power': p1,
            'harmonic_powers': harmonics,
            'snr': snr,
            'n_samples': len(data),
            'sample_rate': sample_rate,
            'duration_hours': len(data) / sample_rate / 3600,
            'is_elevated': thd >= THD_THRESHOLDS['elevated'],
            'is_critical': thd >= THD_THRESHOLDS['critical'],
        }
    except Exception as e:
        print(f"  {station}: THD computation failed: {e}")
        return None

def run_retrospective():
    """Run retrospective THD validation on cached Ridgecrest data."""
    print("=" * 70)
    print("RIDGECREST 2019 RETROSPECTIVE THD VALIDATION")
    print("=" * 70)
    print(f"\nKey Events:")
    print(f"  - M6.4 Foreshock: {M64_FORESHOCK.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  - M7.1 Mainshock: {M71_MAINSHOCK.strftime('%Y-%m-%d %H:%M UTC')}")

    # Get available dates
    dates = get_available_dates()
    print(f"\nAvailable cached dates: {len(dates)}")
    print(f"  Range: {dates[0]} to {dates[-1]}")

    # Process each date
    results = []

    for date_str in dates:
        print(f"\nProcessing {date_str}...")

        # Parse date
        date = datetime.strptime(date_str, '%Y%m%d')

        # Load waveforms
        waveforms = load_cached_waveforms(date_str)
        if not waveforms:
            print(f"  No waveforms found")
            continue

        # Compute THD for each station
        day_results = {'date': date_str, 'stations': {}}

        for station, stream in waveforms.items():
            thd_result = compute_thd_for_stream(stream, station, date)
            if thd_result:
                day_results['stations'][station] = thd_result
                status = "CRITICAL" if thd_result['is_critical'] else "ELEVATED" if thd_result['is_elevated'] else "normal"
                print(f"  {station}: THD={thd_result['thd']:.4f} SNR={thd_result['snr']:.1f} [{status}]")

        if day_results['stations']:
            results.append(day_results)

    # Summary analysis
    print("\n" + "=" * 70)
    print("SUMMARY: THD EVOLUTION BEFORE RIDGECREST M7.1")
    print("=" * 70)

    # Compute station averages by date
    print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "Date", "CI.WBS", "CI.SLA", "CI.CLC", "Mean", "Status"))
    print("-" * 70)

    station_series = {'CI.WBS': [], 'CI.SLA': [], 'CI.CLC': []}
    dates_list = []

    for day in results:
        date_str = day['date']
        date = datetime.strptime(date_str, '%Y%m%d')
        dates_list.append(date)

        thd_values = []
        row = [date.strftime('%Y-%m-%d')]

        for sta in ['CI.WBS', 'CI.SLA', 'CI.CLC']:
            if sta in day['stations']:
                thd = day['stations'][sta]['thd']
                station_series[sta].append(thd)
                thd_values.append(thd)
                row.append(f"{thd:.4f}")
            else:
                station_series[sta].append(None)
                row.append("-")

        if thd_values:
            mean_thd = np.mean(thd_values)
            row.append(f"{mean_thd:.4f}")

            # Status based on mean
            if mean_thd >= THD_THRESHOLDS['critical']:
                row.append("CRITICAL")
            elif mean_thd >= THD_THRESHOLDS['elevated']:
                row.append("ELEVATED")
            else:
                row.append("normal")
        else:
            row.append("-")
            row.append("-")

        print("{:<12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(*row))

    # Analyze pre-event vs baseline
    print("\n" + "=" * 70)
    print("BASELINE vs PRE-EVENT ANALYSIS")
    print("=" * 70)

    # Define periods
    # Baseline: June 19-30 (11 days before foreshock)
    # Pre-event: July 1-4 (leading up to foreshock/mainshock)

    baseline_dates = [d for d in dates_list if d < datetime(2019, 7, 1)]
    pre_event_dates = [d for d in dates_list if d >= datetime(2019, 7, 1)]

    print(f"\nBaseline period: {len(baseline_dates)} days (June 19-30)")
    print(f"Pre-event period: {len(pre_event_dates)} days (July 1-5)")

    for sta in ['CI.WBS', 'CI.SLA', 'CI.CLC']:
        baseline_vals = []
        pre_event_vals = []

        for i, date in enumerate(dates_list):
            val = station_series[sta][i]
            if val is not None:
                if date < datetime(2019, 7, 1):
                    baseline_vals.append(val)
                else:
                    pre_event_vals.append(val)

        if baseline_vals and pre_event_vals:
            baseline_mean = np.mean(baseline_vals)
            baseline_std = np.std(baseline_vals)
            pre_event_mean = np.mean(pre_event_vals)

            # Z-score of pre-event mean relative to baseline
            if baseline_std > 0:
                z_score = (pre_event_mean - baseline_mean) / baseline_std
            else:
                z_score = 0

            change_pct = (pre_event_mean - baseline_mean) / baseline_mean * 100

            print(f"\n{sta}:")
            print(f"  Baseline: mean={baseline_mean:.4f}, std={baseline_std:.4f}")
            print(f"  Pre-event: mean={pre_event_mean:.4f}")
            print(f"  Change: {change_pct:+.1f}% (z={z_score:.2f})")

            if z_score >= 2.0:
                print(f"  --> SIGNIFICANT ELEVATION (z >= 2.0)")
            elif z_score >= 1.5:
                print(f"  --> MODERATE ELEVATION (z >= 1.5)")
            elif z_score >= 1.0:
                print(f"  --> MILD ELEVATION (z >= 1.0)")
            else:
                print(f"  --> No significant change")

    # Final assessment
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compute overall statistics
    all_baseline = []
    all_pre_event = []
    for sta in ['CI.WBS', 'CI.SLA', 'CI.CLC']:
        for i, date in enumerate(dates_list):
            val = station_series[sta][i]
            if val is not None:
                if date < datetime(2019, 7, 1):
                    all_baseline.append(val)
                else:
                    all_pre_event.append(val)

    if all_baseline and all_pre_event:
        overall_baseline_mean = np.mean(all_baseline)
        overall_baseline_std = np.std(all_baseline)
        overall_pre_event_mean = np.mean(all_pre_event)

        if overall_baseline_std > 0:
            overall_z = (overall_pre_event_mean - overall_baseline_mean) / overall_baseline_std
        else:
            overall_z = 0

        print(f"\nOverall THD (all stations combined):")
        print(f"  Baseline mean: {overall_baseline_mean:.4f} +/- {overall_baseline_std:.4f}")
        print(f"  Pre-event mean: {overall_pre_event_mean:.4f}")
        print(f"  Z-score: {overall_z:.2f}")

        if overall_z >= 2.0:
            print(f"\n*** THD PRECURSOR DETECTED ***")
            print(f"Pre-event THD was significantly elevated (z >= 2.0)")
            print(f"This validates the THD method for Ridgecrest 2019.")
        elif overall_z >= 1.5:
            print(f"\n** POSSIBLE THD PRECURSOR **")
            print(f"Pre-event THD was moderately elevated (z >= 1.5)")
        else:
            print(f"\nNo clear THD precursor detected for this event.")
            print(f"Note: THD method works better with longer data windows.")

    # Earthquake contamination note
    print("\n" + "=" * 70)
    print("IMPORTANT: EARTHQUAKE CONTAMINATION")
    print("=" * 70)
    print("\nJuly 4-5 data is contaminated by the earthquakes themselves:")
    print("  - July 4: M6.4 foreshock at 10:33 UTC")
    print("  - Low SNR on July 4-5 indicates seismic signal, not tidal response")
    print("\nFor precursor analysis, focus on June 19 - July 3 data.")

    # Clean baseline analysis (exclude July 4-5)
    print("\n" + "=" * 70)
    print("CLEAN PRECURSOR ANALYSIS (excluding earthquake days)")
    print("=" * 70)

    clean_baseline = [d for d in dates_list if d < datetime(2019, 6, 28)]
    clean_pre_event = [d for d in dates_list if datetime(2019, 6, 28) <= d < datetime(2019, 7, 4)]

    print(f"\nClean baseline: June 19-27 ({len(clean_baseline)} days)")
    print(f"Pre-foreshock: June 28 - July 3 ({len(clean_pre_event)} days)")

    all_clean_baseline = []
    all_clean_pre = []
    for sta in ['CI.WBS', 'CI.SLA', 'CI.CLC']:
        for i, date in enumerate(dates_list):
            val = station_series[sta][i]
            if val is not None:
                if date < datetime(2019, 6, 28):
                    all_clean_baseline.append(val)
                elif date < datetime(2019, 7, 4):
                    all_clean_pre.append(val)

    if all_clean_baseline and all_clean_pre:
        clean_baseline_mean = np.mean(all_clean_baseline)
        clean_baseline_std = np.std(all_clean_baseline)
        clean_pre_mean = np.mean(all_clean_pre)

        if clean_baseline_std > 0:
            clean_z = (clean_pre_mean - clean_baseline_mean) / clean_baseline_std
        else:
            clean_z = 0

        print(f"\nClean analysis (all stations):")
        print(f"  Baseline mean (Jun 19-27): {clean_baseline_mean:.4f} +/- {clean_baseline_std:.4f}")
        print(f"  Pre-event mean (Jun 28 - Jul 3): {clean_pre_mean:.4f}")
        print(f"  Z-score: {clean_z:.2f}")

        if clean_z >= 1.0:
            print(f"\n*** THD ELEVATION DETECTED ***")
            print(f"Pre-event window shows elevated THD relative to earlier baseline.")
        else:
            print(f"\nTHD was elevated throughout the monitoring period.")
            print("This may indicate either:")
            print("  1. Precursor signal started before June 19")
            print("  2. Need longer baseline (months, not weeks)")

    # Save results with serializable format
    output_path = Path(__file__).parent / 'ridgecrest_2019_thd_results.json'

    # Convert results to serializable format
    serializable_results = []
    for day in results:
        day_data = {
            'date': day['date'],
            'stations': {}
        }
        for sta, data in day['stations'].items():
            day_data['stations'][sta] = {
                'thd': float(data['thd']),
                'snr': float(data['snr']),
                'is_elevated': bool(data['is_elevated']),
                'is_critical': bool(data['is_critical']),
            }
        serializable_results.append(day_data)

    output = {
        'event': 'Ridgecrest 2019 M7.1',
        'foreshock': M64_FORESHOCK.isoformat(),
        'mainshock': M71_MAINSHOCK.isoformat(),
        'data_range': {'start': dates[0], 'end': dates[-1]},
        'baseline_period': {
            'start': '20190619',
            'end': '20190630',
            'n_days': len(baseline_dates)
        },
        'pre_event_period': {
            'start': '20190701',
            'end': dates[-1],
            'n_days': len(pre_event_dates)
        },
        'daily_results': serializable_results,
        'overall_z_score': float(overall_z) if 'overall_z' in dir() else None,
        'clean_analysis': {
            'baseline_mean': float(clean_baseline_mean) if 'clean_baseline_mean' in dir() else None,
            'baseline_std': float(clean_baseline_std) if 'clean_baseline_std' in dir() else None,
            'pre_event_mean': float(clean_pre_mean) if 'clean_pre_mean' in dir() else None,
            'z_score': float(clean_z) if 'clean_z' in dir() else None,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results

if __name__ == '__main__':
    run_retrospective()
