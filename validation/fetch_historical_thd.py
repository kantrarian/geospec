#!/usr/bin/env python3
"""
fetch_historical_thd.py
Fetch THD (Total Harmonic Distortion) data for historical earthquake backtests.

This script fetches seismic waveform data from IRIS/FDSN and computes THD time-series
for historical earthquakes to populate the dashboard backtest visualizations.

Usage:
    python fetch_historical_thd.py --event tohoku_2011
    python fetch_historical_thd.py --event turkey_2023
    python fetch_historical_thd.py --all
    python fetch_historical_thd.py --list

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monitoring', 'src'))

import numpy as np

# Import THD analyzer
from seismic_thd import (
    SeismicTHDAnalyzer,
    THDResult,
    fetch_continuous_data_for_thd,
    THD_THRESHOLDS
)


# Historical earthquake events configuration
HISTORICAL_EVENTS = {
    'ridgecrest_2019': {
        'name': 'Ridgecrest 2019',
        'date': datetime(2019, 7, 6, 3, 19, 53),  # UTC
        'magnitude': 7.1,
        'location': (35.77, -117.60),
        'region': 'california_mojave',
        'lead_days': 14,
        'stations': [
            {'network': 'CI', 'code': 'WBS', 'channel': 'BHZ', 'priority': 1},
            {'network': 'CI', 'code': 'CCC', 'channel': 'BHZ', 'priority': 2},
            {'network': 'CI', 'code': 'SLA', 'channel': 'BHZ', 'priority': 3},
        ],
        'notes': 'Eastern California Shear Zone M7.1 - preceded by M6.4 foreshock on July 4'
    },
    'tohoku_2011': {
        'name': 'Tohoku 2011',
        'date': datetime(2011, 3, 11, 5, 46, 24),  # UTC
        'magnitude': 9.0,
        'location': (38.30, 142.37),
        'region': 'tokyo_kanto',
        'lead_days': 14,
        'stations': [
            {'network': 'IU', 'code': 'MAJO', 'channel': 'BHZ', 'priority': 1},
            {'network': 'II', 'code': 'ERM', 'channel': 'BHZ', 'priority': 2},
        ],
        'notes': 'Japan Trench M9.0 - largest recorded Japan earthquake'
    },
    'turkey_2023': {
        'name': 'Turkey 2023',
        'date': datetime(2023, 2, 6, 1, 17, 35),  # UTC
        'magnitude': 7.8,
        'location': (37.17, 37.03),
        'region': 'turkey_kahramanmaras',
        'lead_days': 14,
        'stations': [
            {'network': 'IU', 'code': 'ANTO', 'channel': 'BHZ', 'priority': 1},
            {'network': 'GE', 'code': 'ISP', 'channel': 'BHZ', 'priority': 2},
        ],
        'notes': 'East Anatolian Fault M7.8 - deadliest 2023 earthquake'
    },
    'chile_2010': {
        'name': 'Chile 2010',
        'date': datetime(2010, 2, 27, 6, 34, 14),  # UTC
        'magnitude': 8.8,
        'location': (-35.85, -72.72),
        'region': 'chile_maule',
        'lead_days': 14,
        'stations': [
            {'network': 'II', 'code': 'LCO', 'channel': 'BHZ', 'priority': 1},
            {'network': 'IU', 'code': 'LVC', 'channel': 'BHZ', 'priority': 2},
        ],
        'notes': 'Maule M8.8 - sixth largest recorded earthquake'
    },
    'morocco_2023': {
        'name': 'Morocco 2023',
        'date': datetime(2023, 9, 8, 22, 11, 1),  # UTC
        'magnitude': 6.8,
        'location': (31.12, -8.43),
        'region': 'morocco_atlas',
        'lead_days': 14,
        'stations': [
            # Morocco network may be restricted, try nearby global stations
            {'network': 'GE', 'code': 'MARJ', 'channel': 'BHZ', 'priority': 1},
            {'network': 'II', 'code': 'TAM', 'channel': 'BHZ', 'priority': 2},
            {'network': 'IU', 'code': 'PAB', 'channel': 'BHZ', 'priority': 3},  # Spain
        ],
        'notes': 'High Atlas M6.8 - largest Morocco earthquake since 1960'
    },
    'kaikoura_2016': {
        'name': 'Kaikoura 2016',
        'date': datetime(2016, 11, 13, 11, 2, 56),  # UTC
        'magnitude': 7.8,
        'location': (-42.69, 173.02),
        'region': 'new_zealand_south',
        'lead_days': 14,
        'stations': [
            {'network': 'IU', 'code': 'SNZO', 'channel': 'BHZ', 'priority': 1},  # South Karori, NZ
            {'network': 'IU', 'code': 'CTAO', 'channel': 'BHZ', 'priority': 2},  # Australia
            {'network': 'II', 'code': 'TAU', 'channel': 'BHZ', 'priority': 3},   # Tasmania
        ],
        'notes': 'South Island M7.8 - Complex multi-fault rupture'
    },
    'anchorage_2018': {
        'name': 'Anchorage 2018',
        'date': datetime(2018, 11, 30, 17, 29, 29),  # UTC
        'magnitude': 7.1,
        'location': (61.35, -149.96),
        'region': 'alaska_cook_inlet',
        'lead_days': 14,
        'stations': [
            {'network': 'IU', 'code': 'COLA', 'channel': 'BHZ', 'priority': 1},  # College, Alaska
            {'network': 'II', 'code': 'KDAK', 'channel': 'BHZ', 'priority': 2},  # Kodiak Island
        ],
        'notes': 'Cook Inlet M7.1 - Deep intraslab event (47km depth)'
    },
    'kumamoto_2016': {
        'name': 'Kumamoto 2016',
        'date': datetime(2016, 4, 16, 1, 25, 6),  # UTC (mainshock)
        'magnitude': 7.0,
        'location': (32.78, 130.73),
        'region': 'japan_kyushu',
        'lead_days': 14,
        'stations': [
            {'network': 'IU', 'code': 'MAJO', 'channel': 'BHZ', 'priority': 1},  # Matsushiro
            {'network': 'II', 'code': 'ERM', 'channel': 'BHZ', 'priority': 2},   # Erimo
            {'network': 'IU', 'code': 'TATO', 'channel': 'BHZ', 'priority': 3},  # Taiwan
        ],
        'notes': 'Kyushu M7.0 - Preceded by M6.5 foreshock on Apr 14'
    },
    'hualien_2024': {
        'name': 'Hualien 2024',
        'date': datetime(2024, 4, 3, 7, 58, 11),  # UTC
        'magnitude': 7.4,
        'location': (23.82, 121.56),
        'region': 'taiwan_east_coast',
        'lead_days': 14,
        'stations': [
            {'network': 'IU', 'code': 'TATO', 'channel': 'BHZ', 'priority': 1},  # Taipei
            {'network': 'II', 'code': 'TLY', 'channel': 'BHZ', 'priority': 2},   # Russia
            {'network': 'IU', 'code': 'GUMO', 'channel': 'BHZ', 'priority': 3},  # Guam
        ],
        'notes': 'East coast Taiwan M7.4 - Strongest Taiwan quake since 1999'
    },
}


def fetch_event_thd(
    event_key: str,
    window_hours: int = 24,
    step_hours: int = 6,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Fetch and compute THD time-series for a historical event.

    Args:
        event_key: Event identifier (e.g., 'tohoku_2011')
        window_hours: THD analysis window size in hours
        step_hours: Step between analysis windows
        verbose: Print progress messages

    Returns:
        Dictionary with event data and THD time-series, or None on failure
    """
    if event_key not in HISTORICAL_EVENTS:
        print(f"ERROR: Unknown event '{event_key}'")
        print(f"Available events: {list(HISTORICAL_EVENTS.keys())}")
        return None

    event = HISTORICAL_EVENTS[event_key]

    if verbose:
        print("=" * 70)
        print(f"FETCHING THD DATA: {event['name']}")
        print("=" * 70)
        print(f"Event: M{event['magnitude']} on {event['date'].strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"Location: {event['location']}")
        print(f"Region: {event['region']}")
        print("-" * 70)

    # Calculate time window
    event_date = event['date']
    lead_days = event['lead_days']
    start_time = event_date - timedelta(days=lead_days)
    end_time = event_date + timedelta(days=4)  # Include 4 days post-event data (3 full days after window calc)

    if verbose:
        print(f"Analysis window: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")

    # Try stations in priority order
    data = None
    sample_rate = 0.0
    station_used = None

    for station_info in event['stations']:
        network = station_info['network']
        code = station_info['code']
        channel = station_info['channel']

        if verbose:
            print(f"\nTrying {network}.{code}.{channel}...")

        data, sample_rate = fetch_continuous_data_for_thd(
            station_network=network,
            station_code=code,
            start=start_time,
            end=end_time,
            channel=channel
        )

        if data is not None and len(data) > 0:
            station_used = f"{network}.{code}"
            if verbose:
                duration_hours = len(data) / sample_rate / 3600
                print(f"SUCCESS: Retrieved {len(data)} samples ({duration_hours:.1f} hours)")
            break
        else:
            if verbose:
                print(f"FAILED: No data from {network}.{code}")

    if data is None:
        print(f"\nERROR: Could not fetch data for {event['name']} from any station")
        return None

    # Decimate to 1 Hz if needed (THD analysis doesn't need high sample rate)
    if sample_rate > 1.5:
        from scipy.signal import decimate
        decimate_factor = int(sample_rate)
        data = decimate(data, decimate_factor, zero_phase=True)
        sample_rate = sample_rate / decimate_factor
        if verbose:
            print(f"Decimated to {sample_rate} Hz ({len(data)} samples)")

    # Initialize THD analyzer
    analyzer = SeismicTHDAnalyzer(
        n_harmonics=5,
        window_hours=window_hours
    )

    # Compute THD time series
    if verbose:
        print(f"\nComputing THD time series (window={window_hours}h, step={step_hours}h)...")

    try:
        results = analyzer.compute_thd_timeseries(
            data=data,
            sample_rate=sample_rate,
            station=station_used,
            start_time=start_time,
            window_hours=window_hours,
            step_hours=step_hours
        )
    except Exception as e:
        print(f"ERROR computing THD: {e}")
        return None

    if not results:
        print("ERROR: No THD results computed")
        return None

    if verbose:
        print(f"Computed {len(results)} THD values")

    # Calculate baseline statistics (first 7 days)
    baseline_cutoff = start_time + timedelta(days=7)
    baseline_results = [r for r in results if r.date < baseline_cutoff]

    if baseline_results:
        baseline_mean = np.mean([r.thd_value for r in baseline_results])
        baseline_std = np.std([r.thd_value for r in baseline_results])
    else:
        baseline_mean = 0.0
        baseline_std = 0.0

    # Find peak THD before event
    pre_event_results = [r for r in results if r.date < event_date]
    if pre_event_results:
        peak_thd = max(r.thd_value for r in pre_event_results)
        peak_result = max(pre_event_results, key=lambda r: r.thd_value)
        peak_date = peak_result.date
    else:
        peak_thd = 0.0
        peak_date = None

    # Convert results to time-series format for dashboard
    timeseries = []
    for r in results:
        days_before = (event_date - r.date).total_seconds() / 86400
        timeseries.append({
            'date': r.date.strftime('%Y-%m-%d'),
            'days_before_event': round(days_before, 2),
            'thd': round(r.thd_value, 4),
            'z_score': round((r.thd_value - baseline_mean) / baseline_std, 2) if baseline_std > 0 else 0.0,
            'tier': 'CRITICAL' if r.is_critical else ('ELEVATED' if r.is_elevated else 'NORMAL')
        })

    # Build output structure
    output = {
        'event_key': event_key,
        'event': {
            'name': event['name'],
            'date': event_date.isoformat(),
            'magnitude': event['magnitude'],
            'location': list(event['location']),
            'region': event['region'],
        },
        'data_source': {
            'station': station_used,
            'fetch_date': datetime.now().isoformat(),
            'window_hours': window_hours,
            'step_hours': step_hours,
        },
        'statistics': {
            'baseline_mean': round(baseline_mean, 4),
            'baseline_std': round(baseline_std, 4),
            'peak_thd': round(peak_thd, 4),
            'peak_date': peak_date.isoformat() if peak_date else None,
            'n_elevated': sum(1 for r in results if r.is_elevated),
            'n_critical': sum(1 for r in results if r.is_critical),
        },
        'timeseries': timeseries,
        'notes': event['notes'],
    }

    if verbose:
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"Station used: {station_used}")
        print(f"Baseline THD: {baseline_mean:.4f} +/- {baseline_std:.4f}")
        print(f"Peak THD (pre-event): {peak_thd:.4f}")
        if peak_date:
            lead_hours = (event_date - peak_date).total_seconds() / 3600
            print(f"Peak date: {peak_date.strftime('%Y-%m-%d %H:%M')} ({lead_hours:.0f} hours before event)")
        print(f"Elevated readings: {output['statistics']['n_elevated']}")
        print(f"Critical readings: {output['statistics']['n_critical']}")

    return output


def fetch_all_events(verbose: bool = True) -> Dict[str, Dict]:
    """
    Fetch THD data for all historical events.

    Returns:
        Dictionary mapping event keys to their THD data
    """
    all_data = {}

    for event_key in HISTORICAL_EVENTS:
        print(f"\n{'#' * 70}")
        print(f"# Processing: {event_key}")
        print(f"{'#' * 70}\n")

        result = fetch_event_thd(event_key, verbose=verbose)
        if result:
            all_data[event_key] = result
        else:
            print(f"WARNING: Could not fetch data for {event_key}")

    return all_data


def update_backtest_timeseries(thd_data: Dict[str, Dict], output_file: Path):
    """
    Update the backtest_timeseries.json file with new THD data.

    Args:
        thd_data: Dictionary of THD data by event key
        output_file: Path to backtest_timeseries.json
    """
    # Load existing backtest data
    if output_file.exists():
        with open(output_file, 'r') as f:
            backtest = json.load(f)
    else:
        # Create new structure
        backtest = {
            'method_descriptions': {},
            'events': {}
        }

    # Update events with THD data
    for event_key, thd in thd_data.items():
        if event_key in backtest.get('events', {}):
            # Update existing event
            event = backtest['events'][event_key]

            # Add THD to methods_available if not already there
            if 'THD' not in event.get('methods_available', []):
                event['methods_available'].append('THD')

            # Merge THD time-series into existing time-series
            existing_dates = {ts['date'] for ts in event.get('timeseries', [])}

            for ts in thd['timeseries']:
                date = ts['date']
                if date in existing_dates:
                    # Update existing entry with THD
                    for existing_ts in event['timeseries']:
                        if existing_ts['date'] == date:
                            existing_ts['thd'] = ts['thd']
                            existing_ts['thd_tier'] = ts['tier']
                            break
                # Note: We don't add new dates that only have THD

            # Update statistics
            event['thd_peak'] = thd['statistics']['peak_thd']
            event['thd_baseline'] = thd['statistics']['baseline_mean']

            print(f"Updated {event_key} with THD data")
        else:
            print(f"WARNING: Event {event_key} not found in backtest file, skipping")

    # Save updated backtest
    with open(output_file, 'w') as f:
        json.dump(backtest, f, indent=2)

    print(f"\nSaved updated backtest data to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch THD data for historical earthquake backtests'
    )
    parser.add_argument(
        '--event', type=str,
        help='Specific event to fetch (e.g., tohoku_2011)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Fetch data for all events'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available events'
    )
    parser.add_argument(
        '--window', type=int, default=24,
        help='Analysis window in hours (default: 24)'
    )
    parser.add_argument(
        '--step', type=int, default=6,
        help='Step between windows in hours (default: 6)'
    )
    parser.add_argument(
        '--update-backtest', action='store_true',
        help='Update backtest_timeseries.json with fetched data'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )

    args = parser.parse_args()

    # List events
    if args.list:
        print("Available historical events:")
        print("-" * 50)
        for key, event in HISTORICAL_EVENTS.items():
            print(f"  {key:20s} M{event['magnitude']} {event['name']}")
        return

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'results' / 'thd_historical'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Fetch specific event
    if args.event:
        result = fetch_event_thd(args.event, window_hours=args.window, step_hours=args.step)
        if result:
            results[args.event] = result

            # Save individual result
            output_file = output_dir / f'{args.event}_thd.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {output_file}")

    # Fetch all events
    elif args.all:
        results = fetch_all_events()

        # Save all results
        output_file = output_dir / 'all_historical_thd.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved all results: {output_file}")

    else:
        parser.print_help()
        return

    # Update backtest file if requested
    if args.update_backtest and results:
        backtest_file = Path(__file__).parent.parent / 'docs' / 'backtest_timeseries.json'
        update_backtest_timeseries(results, backtest_file)

    # Print summary
    print("\n" + "=" * 70)
    print("FETCH SUMMARY")
    print("=" * 70)
    print(f"Events processed: {len(results)}")
    for key, data in results.items():
        peak = data['statistics']['peak_thd']
        station = data['data_source']['station']
        print(f"  {key}: Peak THD={peak:.4f} (from {station})")


if __name__ == '__main__':
    main()
