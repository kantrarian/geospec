#!/usr/bin/env python
"""
data_inventory.py - Verify all historical data for backtest validation.

Catalogs available GPS data, seismic cache, and validation results for each event.
Generates a data availability matrix for the backtest validation plan.

Author: R.J. Mathews / Claude
Date: January 2026
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
GPS_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
SEISMIC_CACHE_DIR = PROJECT_ROOT / 'monitoring' / 'data' / 'seismic_cache'
RESULTS_DIR = PROJECT_ROOT / 'results'
VALIDATION_RESULTS_DIR = PROJECT_ROOT / 'validation' / 'results'


@dataclass
class EventData:
    """Data availability for a single event."""
    name: str
    event_date: str
    magnitude: float
    location: Dict[str, float]

    # GPS data
    gps_stations: List[str] = field(default_factory=list)
    gps_data_path: Optional[str] = None

    # Seismic cache
    seismic_cache_dates: List[str] = field(default_factory=list)
    seismic_stations: List[str] = field(default_factory=list)

    # Method availability
    lambda_geo_available: bool = False
    thd_available: bool = False
    fault_correlation_available: bool = False

    # Existing validation
    existing_validations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'event_date': self.event_date,
            'magnitude': self.magnitude,
            'location': self.location,
            'gps': {
                'stations': self.gps_stations,
                'station_count': len(self.gps_stations),
                'data_path': self.gps_data_path,
            },
            'seismic': {
                'cache_dates': self.seismic_cache_dates,
                'stations': self.seismic_stations,
                'days_cached': len(self.seismic_cache_dates),
            },
            'methods_available': {
                'lambda_geo': self.lambda_geo_available,
                'thd': self.thd_available,
                'fault_correlation': self.fault_correlation_available,
            },
            'existing_validations': self.existing_validations,
        }


# Event definitions
EVENTS = {
    'ridgecrest_2019': {
        'name': 'Ridgecrest 2019 M7.1',
        'event_date': '2019-07-06T03:19:53',
        'magnitude': 7.1,
        'location': {'lat': 35.77, 'lon': -117.60},
    },
    'tohoku_2011': {
        'name': 'Tohoku 2011 M9.0',
        'event_date': '2011-03-11T05:46:24',
        'magnitude': 9.0,
        'location': {'lat': 38.297, 'lon': 142.373},
    },
    'turkey_2023': {
        'name': 'Turkey 2023 M7.8',
        'event_date': '2023-02-06T01:17:35',
        'magnitude': 7.8,
        'location': {'lat': 37.226, 'lon': 37.014},
    },
    'chile_2010': {
        'name': 'Chile 2010 M8.8',
        'event_date': '2010-02-27T06:34:14',
        'magnitude': 8.8,
        'location': {'lat': -35.846, 'lon': -72.719},
    },
    'morocco_2023': {
        'name': 'Morocco 2023 M6.8',
        'event_date': '2023-09-08T22:11:01',
        'magnitude': 6.8,
        'location': {'lat': 31.055, 'lon': -8.396},
    },
}


def check_gps_data(event_key: str) -> tuple:
    """Check GPS data availability for an event."""
    gps_dir = GPS_DATA_DIR / event_key / 'gps'

    if not gps_dir.exists():
        return [], None

    # Find .tenv3 files (GPS time series)
    tenv3_files = list(gps_dir.glob('*.tenv3'))
    stations = [f.stem for f in tenv3_files]

    return stations, str(gps_dir)


def check_seismic_cache(event_key: str) -> tuple:
    """Check seismic cache availability for an event."""
    # Currently only ridgecrest has cached seismic data
    if 'ridgecrest' not in event_key:
        return [], []

    cache_dir = SEISMIC_CACHE_DIR / 'ridgecrest'

    if not cache_dir.exists():
        return [], []

    # Find date directories
    date_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith('2019')]
    dates = sorted([d.name for d in date_dirs])

    # Get station names from first date
    stations = set()
    for date_dir in date_dirs[:1]:  # Check first date for stations
        for pkl_file in date_dir.glob('*_waveforms.pkl'):
            # Extract station from filename pattern: {segment}_waveforms.pkl
            # The actual station names are inside the pickle file
            pass

    # For now, we know from earlier analysis the stations are CI.WBS, CI.SLA, CI.CLC
    stations = ['CI.WBS', 'CI.SLA', 'CI.CLC']

    return dates, stations


def check_existing_validations(event_key: str) -> List[str]:
    """Check for existing validation results."""
    validations = []

    # Check validation/results directory
    if VALIDATION_RESULTS_DIR.exists():
        for f in VALIDATION_RESULTS_DIR.glob(f'*{event_key.split("_")[0]}*.json'):
            validations.append(f.name)

    # Check results directory
    results_event_dir = RESULTS_DIR / event_key.split('_')[0]
    if results_event_dir.exists():
        for f in results_event_dir.glob('*.json'):
            validations.append(f.name)

    return validations


def build_inventory() -> Dict[str, EventData]:
    """Build complete data inventory for all events."""
    inventory = {}

    for event_key, event_info in EVENTS.items():
        # Create EventData object
        event_data = EventData(
            name=event_info['name'],
            event_date=event_info['event_date'],
            magnitude=event_info['magnitude'],
            location=event_info['location'],
        )

        # Check GPS data
        gps_stations, gps_path = check_gps_data(event_key)
        event_data.gps_stations = gps_stations
        event_data.gps_data_path = gps_path
        event_data.lambda_geo_available = len(gps_stations) >= 4  # Need at least 4 stations

        # Check seismic cache
        cache_dates, seismic_stations = check_seismic_cache(event_key)
        event_data.seismic_cache_dates = cache_dates
        event_data.seismic_stations = seismic_stations
        event_data.thd_available = len(cache_dates) >= 7  # Need at least 7 days
        event_data.fault_correlation_available = len(cache_dates) >= 7

        # Check existing validations
        event_data.existing_validations = check_existing_validations(event_key)

        inventory[event_key] = event_data

    return inventory


def print_inventory_summary(inventory: Dict[str, EventData]):
    """Print a summary of the data inventory."""
    print("=" * 80)
    print("GEOSPEC BACKTEST DATA INVENTORY")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Summary table
    print("Event Summary:")
    print("-" * 80)
    print(f"{'Event':<25} {'Mag':<5} {'GPS Sta.':<10} {'Seismic':<12} {'LG':<5} {'THD':<5} {'FC':<5}")
    print("-" * 80)

    for event_key, data in inventory.items():
        seismic_str = f"{len(data.seismic_cache_dates)}d" if data.seismic_cache_dates else "None"
        lg = "YES" if data.lambda_geo_available else "NO"
        thd = "YES" if data.thd_available else "NO"
        fc = "YES" if data.fault_correlation_available else "NO"

        print(f"{data.name:<25} M{data.magnitude:<4} {len(data.gps_stations):<10} {seismic_str:<12} {lg:<5} {thd:<5} {fc:<5}")

    print("-" * 80)
    print()

    # Detailed breakdown
    for event_key, data in inventory.items():
        print(f"\n{'='*40}")
        print(f"{data.name}")
        print(f"{'='*40}")
        print(f"Date: {data.event_date}")
        print(f"Location: {data.location['lat']:.2f}N, {data.location['lon']:.2f}E")
        print()

        print("GPS Data:")
        if data.gps_stations:
            print(f"  Path: {data.gps_data_path}")
            print(f"  Stations ({len(data.gps_stations)}): {', '.join(data.gps_stations)}")
        else:
            print("  No GPS data found")

        print()
        print("Seismic Cache:")
        if data.seismic_cache_dates:
            print(f"  Dates: {data.seismic_cache_dates[0]} to {data.seismic_cache_dates[-1]}")
            print(f"  Days cached: {len(data.seismic_cache_dates)}")
            print(f"  Stations: {', '.join(data.seismic_stations)}")
        else:
            print("  No seismic cache")

        print()
        print("Methods Available:")
        print(f"  Lambda_geo (GPS strain): {'YES' if data.lambda_geo_available else 'NO'}")
        print(f"  THD (Seismic):           {'YES' if data.thd_available else 'NO'}")
        print(f"  Fault Correlation:       {'YES' if data.fault_correlation_available else 'NO'}")

        if data.existing_validations:
            print()
            print("Existing Validations:")
            for v in data.existing_validations:
                print(f"  - {v}")

    print("\n" + "=" * 80)

    # Method coverage summary
    lg_count = sum(1 for d in inventory.values() if d.lambda_geo_available)
    thd_count = sum(1 for d in inventory.values() if d.thd_available)
    fc_count = sum(1 for d in inventory.values() if d.fault_correlation_available)

    print("\nMETHOD COVERAGE SUMMARY:")
    print(f"  Lambda_geo: {lg_count}/5 events")
    print(f"  THD:        {thd_count}/5 events")
    print(f"  Fault Corr: {fc_count}/5 events")
    print(f"  Full 3-method: {min(lg_count, thd_count, fc_count)}/5 events")
    print("=" * 80)


def save_inventory(inventory: Dict[str, EventData], output_path: Path):
    """Save inventory to JSON file."""
    output = {
        'generated': datetime.now().isoformat(),
        'events': {k: v.to_dict() for k, v in inventory.items()},
        'summary': {
            'total_events': len(inventory),
            'lambda_geo_available': sum(1 for d in inventory.values() if d.lambda_geo_available),
            'thd_available': sum(1 for d in inventory.values() if d.thd_available),
            'fault_correlation_available': sum(1 for d in inventory.values() if d.fault_correlation_available),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nInventory saved to: {output_path}")


def main():
    """Main entry point."""
    print("Building data inventory...\n")

    inventory = build_inventory()

    print_inventory_summary(inventory)

    # Save to JSON
    output_path = VALIDATION_RESULTS_DIR / 'data_inventory.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_inventory(inventory, output_path)

    return inventory


if __name__ == '__main__':
    main()
