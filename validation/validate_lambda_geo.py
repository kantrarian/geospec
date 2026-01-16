#!/usr/bin/env python3
"""
validate_lambda_geo.py - Validate calibrated Lambda_geo on unseen events

Tests the Lambda_geo calibration against 5 events NOT used in calibration:
1. Kaikoura, New Zealand 2016 M7.8
2. Kumamoto, Japan 2016 M7.0
3. El Mayor-Cucapah, Mexico/USA 2010 M7.2
4. Norcia, Italy 2016 M6.6
5. Noto Peninsula, Japan 2024 M7.5

Author: R.J. Mathews / Claude
Date: January 2026
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial import Delaunay

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Output directory
RESULTS_DIR = Path(__file__).parent / 'results'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'

# Calibration thresholds (from calibrate_lambda_geo.py)
CALIBRATED_THRESHOLDS = {
    'normal_max': 1.5,
    'watch_min': 1.5,
    'watch_max': 2.5,
    'elevated_min': 2.5,
    'elevated_max': 4.0,
    'critical_min': 4.0,
}

# Validation events (NOT in calibration set)
VALIDATION_EVENTS = {
    'kaikoura_2016': {
        'name': 'Kaikoura 2016 M7.8',
        'date': '2016-11-13T11:02:56',
        'magnitude': 7.8,
        'lat': -42.69,
        'lon': 173.02,
        'analysis_window_days': 30,
    },
    'kumamoto_2016': {
        'name': 'Kumamoto 2016 M7.0',
        'date': '2016-04-15T16:25:06',
        'magnitude': 7.0,
        'lat': 32.79,
        'lon': 130.75,
        'analysis_window_days': 30,
    },
    'el_mayor_2010': {
        'name': 'El Mayor-Cucapah 2010 M7.2',
        'date': '2010-04-04T22:40:42',
        'magnitude': 7.2,
        'lat': 32.26,
        'lon': -115.29,
        'analysis_window_days': 30,
    },
    'norcia_2016': {
        'name': 'Norcia Italy 2016 M6.6',
        'date': '2016-10-30T06:40:18',
        'magnitude': 6.6,
        'lat': 42.83,
        'lon': 13.11,
        'analysis_window_days': 30,
    },
    'noto_2024': {
        'name': 'Noto Peninsula 2024 M7.5',
        'date': '2024-01-01T07:10:09',
        'magnitude': 7.5,
        'lat': 37.50,
        'lon': 137.27,
        'analysis_window_days': 30,
    },
}


@dataclass
class StationData:
    """GPS station time series data."""
    code: str
    lat: float
    lon: float
    dates: List[str]
    east: List[float]  # meters
    north: List[float]  # meters
    up: List[float]  # meters


def parse_tenv3_file(filepath: Path) -> Optional[StationData]:
    """Parse NGL .tenv3 format GPS file.

    Format columns:
    0: site, 1: YYMMMDD, 2: yyyy.yyyy (decimal year)
    8: __east(m), 10: _north(m), 12: ____up(m)
    20: _latitude(deg), 21: _longitude(deg)
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return None

        dates = []
        east = []
        north = []
        up = []
        lat = None
        lon = None
        code = filepath.stem

        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) < 22:
                continue

            try:
                # Date in YYMMMDD format (column 1) - convert to YYYY-MM-DD
                date_str = parts[1]
                # Parse decimal year for date
                decimal_year = float(parts[2])
                year = int(decimal_year)
                day_of_year = int((decimal_year - year) * 365.25) + 1
                date = datetime(year, 1, 1) + timedelta(days=day_of_year-1)
                dates.append(date.strftime('%Y-%m-%d'))

                # Displacements in meters
                east.append(float(parts[8]))
                north.append(float(parts[10]))
                up.append(float(parts[12]))

                # Lat/lon
                if lat is None:
                    lat = float(parts[20])
                    lon = float(parts[21])

            except (ValueError, IndexError):
                continue

        if len(dates) < 10 or lat is None:
            return None

        return StationData(
            code=code,
            lat=lat,
            lon=lon,
            dates=dates,
            east=east,
            north=north,
            up=up
        )

    except Exception as e:
        return None


def compute_strain_tensor(stations: List[StationData], date: str) -> Optional[np.ndarray]:
    """Compute 2D strain tensor from GPS displacements using Delaunay triangulation."""

    # Get displacements for this date
    positions = []
    displacements = []

    for station in stations:
        if date in station.dates:
            idx = station.dates.index(date)
            positions.append([station.lon, station.lat])
            displacements.append([station.east[idx], station.north[idx]])

    if len(positions) < 3:
        return None

    positions = np.array(positions)
    displacements = np.array(displacements)

    try:
        # Delaunay triangulation
        tri = Delaunay(positions)

        strain_tensors = []
        for simplex in tri.simplices:
            p = positions[simplex]
            d = displacements[simplex]

            # Compute displacement gradient
            dx = p[1:] - p[0]
            dd = d[1:] - d[0]

            try:
                grad = np.linalg.lstsq(dx, dd, rcond=None)[0]
                # Strain tensor (symmetric part)
                strain = 0.5 * (grad + grad.T)
                strain_tensors.append(strain)
            except:
                continue

        if not strain_tensors:
            return None

        # Average strain tensor
        return np.mean(strain_tensors, axis=0)

    except Exception as e:
        return None


def compute_lambda_geo(strain: np.ndarray) -> float:
    """Compute Lambda_geo from strain tensor eigenvalues."""
    try:
        eigenvalues = np.linalg.eigvalsh(strain)
        # Lambda_geo = max eigenvalue magnitude
        return np.max(np.abs(eigenvalues))
    except:
        return 0.0


def classify_ratio(ratio: float) -> str:
    """Classify ratio using calibrated thresholds."""
    if ratio >= CALIBRATED_THRESHOLDS['critical_min']:
        return 'CRITICAL'
    elif ratio >= CALIBRATED_THRESHOLDS['elevated_min']:
        return 'ELEVATED'
    elif ratio >= CALIBRATED_THRESHOLDS['watch_min']:
        return 'WATCH'
    else:
        return 'NORMAL'


def validate_event(event_key: str, event: dict) -> dict:
    """Validate Lambda_geo against a single event."""

    print(f"\n{'='*60}")
    print(f"Validating: {event['name']}")
    print(f"{'='*60}")

    # Load GPS data
    gps_dir = DATA_DIR / event_key / 'gps'
    if not gps_dir.exists():
        print(f"  ERROR: GPS data directory not found: {gps_dir}")
        return {'computed': False, 'error': 'No GPS data'}

    # Parse station data
    stations = []
    tenv_files = list(gps_dir.glob('*.tenv3'))
    print(f"  Found {len(tenv_files)} .tenv3 files")

    for tenv_file in tenv_files:
        station = parse_tenv3_file(tenv_file)
        if station:
            stations.append(station)

    print(f"  Loaded {len(stations)} valid stations")

    if len(stations) < 4:
        print(f"  ERROR: Insufficient stations ({len(stations)} < 4)")
        return {'computed': False, 'error': 'Insufficient stations'}

    # Determine analysis period
    event_date = datetime.fromisoformat(event['date'].replace('Z', ''))
    analysis_end = event_date - timedelta(days=1)  # Day before event
    analysis_start = analysis_end - timedelta(days=event['analysis_window_days'])
    baseline_end = analysis_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=60)  # 60-day baseline

    print(f"  Baseline: {baseline_start.date()} to {baseline_end.date()}")
    print(f"  Analysis: {analysis_start.date()} to {analysis_end.date()}")

    # Compute Lambda_geo time series
    lambda_geo_baseline = []
    lambda_geo_analysis = []

    # Generate date range
    current = baseline_start
    while current <= analysis_end:
        date_str = current.strftime('%Y-%m-%d')
        strain = compute_strain_tensor(stations, date_str)

        if strain is not None:
            lg = compute_lambda_geo(strain)

            if current <= baseline_end:
                lambda_geo_baseline.append((date_str, lg))
            elif current >= analysis_start:
                lambda_geo_analysis.append((date_str, lg))

        current += timedelta(days=1)

    print(f"  Baseline data points: {len(lambda_geo_baseline)}")
    print(f"  Analysis data points: {len(lambda_geo_analysis)}")

    if len(lambda_geo_baseline) < 10 or len(lambda_geo_analysis) < 5:
        print(f"  ERROR: Insufficient data coverage")
        return {'computed': False, 'error': 'Insufficient data coverage'}

    # Compute baseline statistics
    baseline_values = [v for _, v in lambda_geo_baseline]
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)

    # Find peak in analysis period
    analysis_values = [v for _, v in lambda_geo_analysis]
    analysis_dates = [d for d, _ in lambda_geo_analysis]
    peak_idx = np.argmax(analysis_values)
    peak_value = analysis_values[peak_idx]
    peak_date = analysis_dates[peak_idx]

    # Compute ratio
    ratio = peak_value / baseline_mean if baseline_mean > 0 else 0

    # Compute z-score
    z_score = (peak_value - baseline_mean) / baseline_std if baseline_std > 0 else 0

    # Classify
    classification = classify_ratio(ratio)
    detected = classification in ['WATCH', 'ELEVATED', 'CRITICAL']

    # Compute lead time
    peak_dt = datetime.strptime(peak_date, '%Y-%m-%d')
    lead_days = (event_date - peak_dt).days

    print(f"\n  Results:")
    print(f"    Baseline mean: {baseline_mean:.2e}")
    print(f"    Baseline std:  {baseline_std:.2e}")
    print(f"    Peak value:    {peak_value:.2e}")
    print(f"    Ratio:         {ratio:.2f}x")
    print(f"    Z-score:       {z_score:.2f}")
    print(f"    Peak date:     {peak_date}")
    print(f"    Lead time:     {lead_days} days")
    print(f"    Classification: {classification}")
    print(f"    DETECTED:      {'YES' if detected else 'NO'}")

    return {
        'computed': True,
        'n_stations': len(stations),
        'n_triangles': len(stations) - 2,  # Approximate
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'lambda_geo_max': peak_value,
        'ratio_to_baseline': ratio,
        'z_score': z_score,
        'peak_date': peak_date,
        'lead_days': lead_days,
        'classification': classification,
        'detected': detected,
    }


def main():
    """Run validation against all events."""

    print("="*70)
    print("LAMBDA_GEO VALIDATION ON UNSEEN EVENTS")
    print("="*70)
    print("\nThese events were NOT used for calibration.")
    print("This tests true out-of-sample performance.\n")

    # Load calibration for reference
    calibration_file = RESULTS_DIR / 'lambda_geo_calibration.json'
    if calibration_file.exists():
        with open(calibration_file) as f:
            calibration = json.load(f)
        print(f"Using calibration from: {calibration['calibration_date'][:10]}")
        print(f"Calibration events: {', '.join(calibration['events_used'])}")

    print(f"\nCalibrated thresholds:")
    print(f"  WATCH: ratio >= {CALIBRATED_THRESHOLDS['watch_min']}x")
    print(f"  ELEVATED: ratio >= {CALIBRATED_THRESHOLDS['elevated_min']}x")
    print(f"  CRITICAL: ratio >= {CALIBRATED_THRESHOLDS['critical_min']}x")

    # Validate each event
    results = {
        'validation_date': datetime.now().isoformat(),
        'calibration_used': 'lambda_geo_calibration.json',
        'thresholds': CALIBRATED_THRESHOLDS,
        'events': {},
    }

    for event_key, event in VALIDATION_EVENTS.items():
        result = validate_event(event_key, event)
        results['events'][event_key] = {
            'event': event,
            'lambda_geo': result,
        }

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    detected = 0
    total = 0

    print(f"\n{'Event':<30} {'Mag':<6} {'Ratio':<8} {'Class':<12} {'Lead':<8} {'Result'}")
    print("-"*80)

    for event_key, data in results['events'].items():
        event = data['event']
        lg = data['lambda_geo']

        if lg['computed']:
            total += 1
            if lg['detected']:
                detected += 1

            print(f"{event['name']:<30} M{event['magnitude']:<5} "
                  f"{lg['ratio_to_baseline']:<8.2f}x {lg['classification']:<12} "
                  f"{lg['lead_days']:<8}d {'HIT' if lg['detected'] else 'MISS'}")
        else:
            print(f"{event['name']:<30} M{event['magnitude']:<5} "
                  f"{'N/A':<8} {'ERROR':<12} {'N/A':<8} {lg.get('error', 'Unknown')}")

    print("-"*80)

    # Detection rate
    if total > 0:
        hit_rate = 100 * detected / total
        print(f"\nVALIDATION HIT RATE: {detected}/{total} = {hit_rate:.0f}%")

    results['summary'] = {
        'total_events': len(VALIDATION_EVENTS),
        'events_computed': total,
        'events_detected': detected,
        'hit_rate': detected / total if total > 0 else 0,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'lambda_geo_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
