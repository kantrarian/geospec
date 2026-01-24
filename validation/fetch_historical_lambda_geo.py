#!/usr/bin/env python3
"""
fetch_historical_lambda_geo.py
Fetch and compute Lambda_geo from NGL GPS .tenv3 files for historical events.

This script computes Lambda_geo from actual GPS station data including 3 days
post-event data for each historical earthquake.

Methodology:
1. Load NGL .tenv3 GPS files for event region
2. Compute velocity gradients via Delaunay triangulation (3+ stations)
3. Build strain tensor E = (∇v + ∇vᵀ) / 2
4. Compute strain rate tensor Ė
5. Calculate commutator [E, Ė] = E·Ė - Ė·E
6. Lambda_geo = ||[E, Ė]||_F (Frobenius norm)
7. Output daily ratios including post-event

Usage:
    python fetch_historical_lambda_geo.py --event ridgecrest_2019
    python fetch_historical_lambda_geo.py --all
    python fetch_historical_lambda_geo.py --list

Author: R.J. Mathews
Date: January 2026
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from scipy.spatial import Delaunay

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data' / 'raw'
RESULTS_DIR = SCRIPT_DIR / 'results' / 'lg_historical'

# Historical events configuration with post-event window
HISTORICAL_EVENTS = {
    'ridgecrest_2019': {
        'name': 'Ridgecrest 2019 M7.1',
        'date': datetime(2019, 7, 6, 3, 19, 53),
        'magnitude': 7.1,
        'lat': 35.77,
        'lon': -117.60,
        'gps_dir': 'ridgecrest_2019',
        'lead_days': 14,
        'post_days': 3,
    },
    'tohoku_2011': {
        'name': 'Tohoku 2011 M9.0',
        'date': datetime(2011, 3, 11, 5, 46, 24),
        'magnitude': 9.0,
        'lat': 38.30,
        'lon': 142.37,
        'gps_dir': 'tohoku_2011',
        'lead_days': 14,
        'post_days': 3,
    },
    'turkey_2023': {
        'name': 'Turkey 2023 M7.8',
        'date': datetime(2023, 2, 6, 1, 17, 35),
        'magnitude': 7.8,
        'lat': 37.22,
        'lon': 37.02,
        'gps_dir': 'turkey_2023',
        'lead_days': 14,
        'post_days': 3,
    },
    'chile_2010': {
        'name': 'Chile 2010 M8.8',
        'date': datetime(2010, 2, 27, 6, 34, 14),
        'magnitude': 8.8,
        'lat': -35.85,
        'lon': -72.72,
        'gps_dir': 'chile_2010',
        'lead_days': 14,
        'post_days': 3,
    },
    'morocco_2023': {
        'name': 'Morocco 2023 M6.8',
        'date': datetime(2023, 9, 8, 22, 11, 1),
        'magnitude': 6.8,
        'lat': 31.06,
        'lon': -8.39,
        'gps_dir': 'morocco_2023',
        'lead_days': 14,
        'post_days': 3,
    },
    'kaikoura_2016': {
        'name': 'Kaikoura 2016 M7.8',
        'date': datetime(2016, 11, 13, 11, 2, 56),
        'magnitude': 7.8,
        'lat': -42.69,
        'lon': 173.02,
        'gps_dir': 'kaikoura_2016',
        'lead_days': 14,
        'post_days': 3,
    },
    'anchorage_2018': {
        'name': 'Anchorage 2018 M7.1',
        'date': datetime(2018, 11, 30, 17, 29, 29),
        'magnitude': 7.1,
        'lat': 61.35,
        'lon': -149.93,
        'gps_dir': 'anchorage_2018',
        'lead_days': 14,
        'post_days': 3,
    },
    'kumamoto_2016': {
        'name': 'Kumamoto 2016 M7.0',
        'date': datetime(2016, 4, 16, 1, 25, 6),
        'magnitude': 7.0,
        'lat': 32.75,
        'lon': 130.76,
        'gps_dir': 'kumamoto_2016',
        'lead_days': 14,
        'post_days': 3,
    },
    'hualien_2024': {
        'name': 'Hualien 2024 M7.4',
        'date': datetime(2024, 4, 3, 7, 58, 11),
        'magnitude': 7.4,
        'lat': 23.77,
        'lon': 121.67,
        'gps_dir': 'hualien_2024',
        'lead_days': 14,
        'post_days': 3,
    },
}


@dataclass
class StationData:
    """GPS station time series."""
    code: str
    lat: float
    lon: float
    times: List[datetime]
    east_mm: np.ndarray
    north_mm: np.ndarray
    up_mm: np.ndarray


def parse_tenv3_file(filepath: Path) -> Optional[StationData]:
    """
    Parse NGL .tenv3 format GPS file.

    Format columns:
    0: site
    1: YYMMMDD
    2: yyyy.yyyy (decimal year)
    3: __MJD
    4: week
    5: d (day of week)
    6: reflon
    7: _e0(m)
    8: __east(m) - east displacement from e0
    9: ____n0(m)
    10: _north(m) - north displacement from n0
    11: u0(m)
    12: ____up(m) - up displacement from u0
    13: _ant(m)
    14: sig_e(m)
    15: sig_n(m)
    16: sig_u(m)
    17-19: correlations
    20: _latitude(deg)
    21: _longitude(deg)
    22: __height(m)
    """
    try:
        times = []
        east = []
        north = []
        up = []
        lat = lon = None
        code = filepath.stem

        with open(filepath, 'r') as f:
            for line in f:
                # Skip header
                if line.startswith('site') or line.startswith('#'):
                    continue

                parts = line.strip().split()
                if len(parts) < 22:
                    continue

                try:
                    # Get lat/lon from last columns
                    if lat is None:
                        lat = float(parts[20])
                        lon = float(parts[21])

                    # Epoch is decimal year (column 2)
                    dec_year = float(parts[2])
                    year = int(dec_year)
                    frac = dec_year - year

                    # Convert to datetime
                    jan1 = datetime(year, 1, 1)
                    days = frac * (366 if year % 4 == 0 else 365)
                    dt = jan1 + timedelta(days=days)

                    times.append(dt)
                    # Columns 8, 10, 12 are displacements in meters -> mm
                    east.append(float(parts[8]) * 1000)
                    north.append(float(parts[10]) * 1000)
                    up.append(float(parts[12]) * 1000)

                except (ValueError, IndexError):
                    continue

        if len(times) < 10 or lat is None:
            return None

        return StationData(
            code=code,
            lat=lat,
            lon=lon,
            times=times,
            east_mm=np.array(east),
            north_mm=np.array(north),
            up_mm=np.array(up)
        )

    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None


def load_gps_data(event_key: str) -> List[StationData]:
    """Load all GPS stations for an event."""
    event = HISTORICAL_EVENTS[event_key]
    gps_dir = DATA_DIR / event['gps_dir'] / 'gps'

    if not gps_dir.exists():
        print(f"  GPS directory not found: {gps_dir}")
        return []

    stations = []
    for tenv3_file in gps_dir.glob('*.tenv3'):
        station = parse_tenv3_file(tenv3_file)
        if station:
            stations.append(station)

    return stations


def compute_strain_tensor(
    stations: List[StationData],
    target_date: datetime,
    window_days: int = 7
) -> Optional[np.ndarray]:
    """
    Compute 2D strain tensor from GPS velocities using Delaunay triangulation.

    Returns 2x2 strain rate tensor or None if insufficient data.
    """
    if len(stations) < 3:
        return None

    # Filter data to window around target date
    start = target_date - timedelta(days=window_days)
    end = target_date

    positions = []
    velocities = []

    for sta in stations:
        # Find data points in window
        mask = [(start <= t <= end) for t in sta.times]
        if sum(mask) < 3:
            continue

        times_window = [t for t, m in zip(sta.times, mask) if m]
        east_window = sta.east_mm[mask]
        north_window = sta.north_mm[mask]

        # Compute velocity via linear regression
        t_days = np.array([(t - start).total_seconds() / 86400 for t in times_window])
        if len(t_days) < 3:
            continue

        try:
            ve = np.polyfit(t_days, east_window, 1)[0]  # mm/day
            vn = np.polyfit(t_days, north_window, 1)[0]

            positions.append([sta.lon, sta.lat])
            velocities.append([ve, vn])
        except Exception:
            continue

    if len(positions) < 3:
        return None

    positions = np.array(positions)
    velocities = np.array(velocities)

    # Delaunay triangulation
    try:
        tri = Delaunay(positions)
    except Exception:
        return None

    # Compute strain for each triangle
    strain_tensors = []

    for simplex in tri.simplices:
        p = positions[simplex]
        v = velocities[simplex]

        try:
            # Local coordinates (km)
            x = (p[:, 0] - p[:, 0].mean()) * 111 * np.cos(np.radians(p[:, 1].mean()))
            y = (p[:, 1] - p[:, 1].mean()) * 111

            # Form design matrix for strain computation
            A = np.column_stack([np.ones(3), x, y])

            # Solve for velocity gradients
            coeffs_e = np.linalg.lstsq(A, v[:, 0], rcond=None)[0]
            coeffs_n = np.linalg.lstsq(A, v[:, 1], rcond=None)[0]

            # Strain tensor components (1/day -> 1/year)
            dudx = coeffs_e[1] * 365 / 1000  # strain/year
            dudy = coeffs_e[2] * 365 / 1000
            dvdx = coeffs_n[1] * 365 / 1000
            dvdy = coeffs_n[2] * 365 / 1000

            # Symmetric strain tensor
            E = np.array([
                [dudx, 0.5 * (dudy + dvdx)],
                [0.5 * (dudy + dvdx), dvdy]
            ])

            strain_tensors.append(E)

        except Exception:
            continue

    if not strain_tensors:
        return None

    # Average strain tensor
    return np.mean(strain_tensors, axis=0)


def compute_lambda_geo(E1: np.ndarray, E2: np.ndarray) -> float:
    """
    Compute Lambda_geo = ||[E1, E2]||_F (Frobenius norm of commutator).

    This measures the rotation of the strain eigenframe between two times.
    """
    commutator = E1 @ E2 - E2 @ E1
    return np.linalg.norm(commutator, 'fro')


def compute_lg_timeseries(event_key: str, verbose: bool = True) -> Optional[Dict]:
    """
    Compute Lambda_geo time series for a historical event, including post-event data.
    """
    if event_key not in HISTORICAL_EVENTS:
        print(f"ERROR: Unknown event '{event_key}'")
        return None

    event = HISTORICAL_EVENTS[event_key]
    event_date = event['date']
    lead_days = event['lead_days']
    post_days = event['post_days']

    if verbose:
        print("=" * 70)
        print(f"COMPUTING LAMBDA_GEO: {event['name']}")
        print("=" * 70)
        print(f"Event: M{event['magnitude']} on {event_date.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"Analysis: {lead_days} days before + {post_days} days after")
        print("-" * 70)

    # Load GPS data
    print(f"Loading GPS data from {event['gps_dir']}...")
    stations = load_gps_data(event_key)
    print(f"  Loaded {len(stations)} stations")

    if len(stations) < 4:
        print(f"  INSUFFICIENT DATA: Need >= 4 stations, have {len(stations)}")
        return {
            'event_key': event_key,
            'computed': False,
            'error': f'Insufficient stations: {len(stations)} < 4 required',
            'data_source': {
                'type': 'NGL_GPS',
                'stations': [s.code for s in stations],
                'fetch_date': datetime.now().isoformat(),
            },
            'timeseries': [],
        }

    # Time window: lead_days before to post_days after event
    start_date = event_date - timedelta(days=lead_days)
    end_date = event_date + timedelta(days=post_days)
    event_date_str = event_date.strftime('%Y-%m-%d')

    print(f"  Analysis window: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Compute Lambda_geo time series
    timeseries = []
    current = start_date + timedelta(days=7)  # Need 7-day window

    while current <= end_date:
        # Compute strain at current time and 1 day earlier
        E_now = compute_strain_tensor(stations, current, window_days=7)
        E_prev = compute_strain_tensor(stations, current - timedelta(days=1), window_days=7)

        if E_now is not None and E_prev is not None:
            lg = compute_lambda_geo(E_now, E_prev)
            date_str = current.strftime('%Y-%m-%d')

            entry = {
                'date': date_str,
                'lambda_geo': round(float(lg), 8),
            }

            # Mark post-event entries (>= because event day is co-seismic, not precursor)
            if date_str >= event_date_str:
                entry['post_event'] = True

            timeseries.append(entry)

        current += timedelta(days=1)

    if len(timeseries) < 3:
        print(f"  INSUFFICIENT TEMPORAL COVERAGE: Only {len(timeseries)} valid points")
        return {
            'event_key': event_key,
            'computed': False,
            'error': f'Insufficient temporal coverage: {len(timeseries)} points',
            'data_source': {
                'type': 'NGL_GPS',
                'stations': [s.code for s in stations],
                'fetch_date': datetime.now().isoformat(),
            },
            'timeseries': [],
        }

    # Compute statistics from pre-event data only
    pre_event = [e for e in timeseries if not e.get('post_event', False)]
    lg_values = [e['lambda_geo'] for e in pre_event]

    if len(lg_values) > 0:
        # Baseline: first half of pre-event data
        n_baseline = max(3, len(lg_values) // 2)
        baseline = lg_values[:n_baseline]
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline) if len(baseline) > 1 else baseline_mean * 0.1

        # Peak detection
        max_value = max(lg_values)
        max_idx = lg_values.index(max_value)
        peak_date = pre_event[max_idx]['date']

        # Ratio to baseline
        if baseline_mean > 0:
            peak_ratio = max_value / baseline_mean
        else:
            peak_ratio = 0.0
    else:
        baseline_mean = baseline_std = peak_ratio = 0.0
        peak_date = None

    # Add ratio to each entry
    if baseline_mean > 0:
        for entry in timeseries:
            entry['ratio'] = round(entry['lambda_geo'] / baseline_mean, 2)
    else:
        for entry in timeseries:
            entry['ratio'] = 0.0

    # Estimate triangles
    try:
        positions = [[s.lon, s.lat] for s in stations]
        tri = Delaunay(np.array(positions))
        n_triangles = len(tri.simplices)
    except Exception:
        n_triangles = 0

    output = {
        'event_key': event_key,
        'computed': True,
        'event': {
            'name': event['name'],
            'date': event_date.isoformat(),
            'magnitude': event['magnitude'],
            'location': {'lat': event['lat'], 'lon': event['lon']},
        },
        'data_source': {
            'type': 'NGL_GPS',
            'stations': [s.code for s in stations],
            'n_stations': len(stations),
            'n_triangles': n_triangles,
            'fetch_date': datetime.now().isoformat(),
        },
        'statistics': {
            'baseline_mean': round(float(baseline_mean), 8),
            'baseline_std': round(float(baseline_std), 8),
            'peak_lambda_geo': round(float(max_value), 8) if lg_values else 0.0,
            'peak_ratio': round(float(peak_ratio), 2),
            'peak_date': peak_date,
            'n_pre_event': len(pre_event),
            'n_post_event': len([e for e in timeseries if e.get('post_event', False)]),
        },
        'timeseries': timeseries,
    }

    if verbose:
        print(f"\n  Computed {len(timeseries)} Lambda_geo values")
        print(f"  Pre-event: {output['statistics']['n_pre_event']} days")
        print(f"  Post-event: {output['statistics']['n_post_event']} days")
        print(f"  Baseline: mean={baseline_mean:.8f}, std={baseline_std:.8f}")
        if peak_date:
            print(f"  Peak: {max_value:.8f} on {peak_date} ({peak_ratio:.1f}x baseline)")
        print("-" * 70)

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Compute Lambda_geo from NGL GPS data for historical earthquakes'
    )
    parser.add_argument('--event', type=str, help='Specific event to process')
    parser.add_argument('--all', action='store_true', help='Process all events')
    parser.add_argument('--list', action='store_true', help='List available events')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    if args.list:
        print("Available events for Lambda_geo computation:")
        print("-" * 60)
        for key, config in HISTORICAL_EVENTS.items():
            print(f"  {key:20s} M{config['magnitude']} {config['name']}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.event:
        result = compute_lg_timeseries(args.event)
        if result:
            results[args.event] = result
            output_file = output_dir / f'{args.event}_lg.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {output_file}")

    elif args.all:
        for event_key in HISTORICAL_EVENTS:
            print(f"\n{'#' * 70}")
            print(f"# Processing: {event_key}")
            print(f"{'#' * 70}\n")

            result = compute_lg_timeseries(event_key)
            if result:
                results[event_key] = result

                # Save individual file
                output_file = output_dir / f'{event_key}_lg.json'
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                print(f"WARNING: Could not compute Lambda_geo for {event_key}")

        if results:
            output_file = output_dir / 'all_historical_lg.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved all results: {output_file}")

    else:
        parser.print_help()
        return

    # Summary
    print("\n" + "=" * 70)
    print("LAMBDA_GEO FETCH SUMMARY")
    print("=" * 70)
    print(f"Events processed: {len(results)}")
    for key, data in results.items():
        if data.get('computed'):
            stats = data['statistics']
            print(f"  {key}: Peak ratio={stats['peak_ratio']:.1f}x, "
                  f"Pre-event={stats['n_pre_event']}d, Post-event={stats['n_post_event']}d")
        else:
            print(f"  {key}: FAILED - {data.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()
