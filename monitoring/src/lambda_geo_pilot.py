#!/usr/bin/env python3
"""
lambda_geo_pilot.py
Adapter to connect RTCM pilot station positions to Lambda_geo computation.

This module reads NGL-format position files from the pilot stations
(COSO, GOLD, JPLM) and computes a simplified Lambda_geo metric.

Requirements for Lambda_geo computation:
- Minimum 3 non-collinear stations (we have COSO, GOLD, JPLM)
- Multiple days of position time series (need velocity)
- Strain rate computation via Delaunay triangulation

Status: EARLY INTEGRATION - Returns placeholder until sufficient data.

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pilot station configuration
PILOT_STATIONS = ['COSO00USA0', 'GOLD00USA0', 'JPLM00USA0']
PILOT_REGION = 'socal_saf_mojave'  # Region these stations cover

# Minimum requirements for Lambda_geo computation
MIN_DAYS_FOR_VELOCITY = 3  # Need at least 3 days of data
MIN_STATIONS = 3  # Need at least 3 stations for triangulation
MIN_EPOCHS_PER_DAY = 50  # Minimum epochs per station per day


@dataclass
class PilotDataStatus:
    """Status of pilot data availability."""
    available: bool
    days_accumulated: int
    stations_active: List[str]
    total_epochs: int
    ready_for_lambda_geo: bool
    message: str


def get_ngl_data_dir() -> Path:
    """Get the NGL format data directory."""
    return Path(__file__).parent.parent / 'data' / 'ngl_format'


def list_available_dates() -> List[str]:
    """List all dates with NGL position data."""
    data_dir = get_ngl_data_dir()
    if not data_dir.exists():
        return []

    dates = []
    for f in data_dir.glob('rtcm_positions_*.json'):
        # Skip QC files
        if '_qc' in f.stem:
            continue
        # Extract date from filename
        date_str = f.stem.replace('rtcm_positions_', '')
        dates.append(date_str)

    return sorted(dates)


def load_positions_for_date(date_str: str) -> Optional[Dict]:
    """Load NGL position data for a specific date."""
    data_dir = get_ngl_data_dir()
    file_path = data_dir / f'rtcm_positions_{date_str}.json'

    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def check_pilot_status() -> PilotDataStatus:
    """Check current status of pilot data availability."""
    dates = list_available_dates()

    if not dates:
        return PilotDataStatus(
            available=False,
            days_accumulated=0,
            stations_active=[],
            total_epochs=0,
            ready_for_lambda_geo=False,
            message="No pilot data files found"
        )

    # Check station coverage across all dates
    all_stations = set()
    total_epochs = 0

    for date_str in dates:
        data = load_positions_for_date(date_str)
        if data and 'stations' in data:
            for station, info in data['stations'].items():
                if station in PILOT_STATIONS:
                    all_stations.add(station)
                    total_epochs += info.get('epochs', 0)

    days = len(dates)
    active_stations = list(all_stations)

    # Check if ready for Lambda_geo
    ready = (
        days >= MIN_DAYS_FOR_VELOCITY and
        len(active_stations) >= MIN_STATIONS and
        total_epochs >= MIN_EPOCHS_PER_DAY * days
    )

    if ready:
        message = f"Ready: {days} days, {len(active_stations)} stations, {total_epochs} epochs"
    elif days < MIN_DAYS_FOR_VELOCITY:
        message = f"Need {MIN_DAYS_FOR_VELOCITY - days} more days of data (have {days})"
    elif len(active_stations) < MIN_STATIONS:
        message = f"Need {MIN_STATIONS - len(active_stations)} more stations (have {active_stations})"
    else:
        message = f"Insufficient epochs (have {total_epochs})"

    return PilotDataStatus(
        available=True,
        days_accumulated=days,
        stations_active=active_stations,
        total_epochs=total_epochs,
        ready_for_lambda_geo=ready,
        message=message
    )


def compute_simple_displacement_rate(
    dates: List[str],
    station: str
) -> Optional[Tuple[float, float, float]]:
    """
    Compute simple displacement rate (mm/day) for a station.

    Returns (e_rate, n_rate, u_rate) or None if insufficient data.
    """
    positions = []

    for date_str in sorted(dates):
        data = load_positions_for_date(date_str)
        if not data or 'stations' not in data:
            continue
        if station not in data['stations']:
            continue

        station_data = data['stations'][station]
        if 'ref_lla' in station_data:
            # Get reference position (first position)
            lla = station_data['ref_lla']
            positions.append({
                'date': date_str,
                'lat': lla[0],
                'lon': lla[1],
                'height': lla[2]
            })

    if len(positions) < 2:
        return None

    # Simple linear rate from first to last
    first = positions[0]
    last = positions[-1]

    # Convert to days
    d1 = datetime.fromisoformat(first['date'])
    d2 = datetime.fromisoformat(last['date'])
    days = (d2 - d1).days

    if days == 0:
        return None

    # Approximate displacement in mm (very rough - need proper geodetic calc)
    # 1 degree lat ~ 111 km, 1 degree lon ~ 111 km * cos(lat)
    lat_rad = np.radians(first['lat'])
    dlat = (last['lat'] - first['lat']) * 111e6  # mm
    dlon = (last['lon'] - first['lon']) * 111e6 * np.cos(lat_rad)  # mm
    dh = (last['height'] - first['height']) * 1000  # mm

    return (dlon / days, dlat / days, dh / days)


def compute_pilot_lambda_geo(target_date: str) -> Optional[float]:
    """
    Compute Lambda_geo proxy from pilot stations.

    This is a SIMPLIFIED computation for early integration.
    Full Lambda_geo requires proper strain tensor computation.

    Returns:
        Lambda_geo ratio (baseline multiplier) or None if not ready
    """
    status = check_pilot_status()

    if not status.ready_for_lambda_geo:
        logger.info(f"Lambda_geo pilot not ready: {status.message}")
        return None

    dates = list_available_dates()

    # Compute displacement rates for each station
    rates = {}
    for station in status.stations_active:
        rate = compute_simple_displacement_rate(dates, station)
        if rate:
            rates[station] = rate

    if len(rates) < MIN_STATIONS:
        logger.warning(f"Insufficient station rates: {len(rates)}")
        return None

    # Compute a simple strain proxy
    # This is NOT the full Lambda_geo, just a placeholder metric
    # Full implementation requires Delaunay triangulation and eigenframe analysis

    # For now: compute velocity gradient magnitude as proxy
    positions = {}
    for station in rates:
        data = load_positions_for_date(dates[-1])
        if data and station in data.get('stations', {}):
            lla = data['stations'][station].get('ref_lla', [0, 0, 0])
            positions[station] = (lla[1], lla[0])  # (lon, lat)

    if len(positions) < 3:
        return None

    # Compute velocity differences between stations
    station_list = list(rates.keys())
    vel_grads = []

    for i in range(len(station_list)):
        for j in range(i + 1, len(station_list)):
            s1, s2 = station_list[i], station_list[j]

            # Distance between stations (degrees -> km)
            dx = (positions[s2][0] - positions[s1][0]) * 111 * np.cos(np.radians(positions[s1][1]))
            dy = (positions[s2][1] - positions[s1][1]) * 111
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 10:  # Too close
                continue

            # Velocity difference
            dve = rates[s2][0] - rates[s1][0]  # mm/day
            dvn = rates[s2][1] - rates[s1][1]

            # Strain rate proxy (1/day)
            strain_rate = np.sqrt(dve**2 + dvn**2) / (dist * 1e6)  # 1/day
            vel_grads.append(strain_rate)

    if not vel_grads:
        return None

    # Convert to baseline ratio (very rough)
    # Normal strain rate ~ 1e-9 /day for stable regions
    # Elevated ~ 1e-8 /day
    mean_grad = np.mean(vel_grads)
    baseline = 1e-9  # Nominal background

    ratio = mean_grad / baseline if baseline > 0 else 1.0

    logger.info(f"Lambda_geo pilot proxy: ratio={ratio:.1f}x (mean_grad={mean_grad:.2e})")

    return ratio


def get_lambda_geo_for_ensemble(
    region: str,
    target_date: datetime
) -> Tuple[bool, float, str]:
    """
    Get Lambda_geo value for ensemble integration.

    Args:
        region: Region key (e.g., 'socal_saf_mojave')
        target_date: Date for assessment

    Returns:
        (available, ratio, notes) tuple
    """
    # Currently only supporting SoCal pilot region
    if region != PILOT_REGION:
        return (False, 0.0, f"No pilot coverage for {region}")

    status = check_pilot_status()

    if not status.available:
        return (False, 0.0, status.message)

    if not status.ready_for_lambda_geo:
        return (False, 0.0, f"Pilot: {status.message}")

    # Compute Lambda_geo proxy
    date_str = target_date.strftime('%Y-%m-%d')
    ratio = compute_pilot_lambda_geo(date_str)

    if ratio is None:
        return (False, 0.0, "Computation failed")

    notes = f"PILOT: {status.days_accumulated}d, {len(status.stations_active)} stations"

    return (True, ratio, notes)


if __name__ == '__main__':
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("Lambda_geo Pilot Status Check")
    print("=" * 50)

    status = check_pilot_status()
    print(f"Available: {status.available}")
    print(f"Days accumulated: {status.days_accumulated}")
    print(f"Stations active: {status.stations_active}")
    print(f"Total epochs: {status.total_epochs}")
    print(f"Ready for Lambda_geo: {status.ready_for_lambda_geo}")
    print(f"Message: {status.message}")

    print()
    print("Available dates:", list_available_dates())

    if status.ready_for_lambda_geo:
        print()
        print("Computing Lambda_geo proxy...")
        dates = list_available_dates()
        ratio = compute_pilot_lambda_geo(dates[-1] if dates else None)
        print(f"Result: {ratio}")
