#!/usr/bin/env python3
"""
download_validation_events.py - Download GPS data for validation events from NGL

Downloads real GPS data for events NOT used in calibration to validate
the Lambda_geo calibration.

Validation Events:
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
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Output directory
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'

# NGL URLs
NGL_CATALOG_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
NGL_TENV3_BASE = "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14"

# Validation events (NOT in calibration set)
VALIDATION_EVENTS = {
    'kaikoura_2016': {
        'name': 'Kaikoura 2016 M7.8',
        'date': '2016-11-13T11:02:56',
        'magnitude': 7.8,
        'lat': -42.69,
        'lon': 173.02,
        'search_radius_km': 300,
        'analysis_window_days': 30,
    },
    'kumamoto_2016': {
        'name': 'Kumamoto 2016 M7.0',
        'date': '2016-04-15T16:25:06',
        'magnitude': 7.0,
        'lat': 32.79,
        'lon': 130.75,
        'search_radius_km': 200,
        'analysis_window_days': 30,
    },
    'el_mayor_2010': {
        'name': 'El Mayor-Cucapah 2010 M7.2',
        'date': '2010-04-04T22:40:42',
        'magnitude': 7.2,
        'lat': 32.26,
        'lon': -115.29,
        'search_radius_km': 250,
        'analysis_window_days': 30,
    },
    'norcia_2016': {
        'name': 'Norcia Italy 2016 M6.6',
        'date': '2016-10-30T06:40:18',
        'magnitude': 6.6,
        'lat': 42.83,
        'lon': 13.11,
        'search_radius_km': 200,
        'analysis_window_days': 30,
    },
    'noto_2024': {
        'name': 'Noto Peninsula 2024 M7.5',
        'date': '2024-01-01T07:10:09',
        'magnitude': 7.5,
        'lat': 37.50,
        'lon': 137.27,
        'search_radius_km': 200,
        'analysis_window_days': 30,
    },
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    import math
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def load_ngl_catalog() -> List[Tuple[str, float, float]]:
    """Load NGL station catalog."""
    print("Loading NGL station catalog...")

    cache_file = DATA_DIR / 'ngl_catalog_cache.txt'

    # Use cache if recent
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mtime).days < 7:
            print("  Using cached catalog")
            with open(cache_file, 'r') as f:
                lines = f.readlines()
        else:
            lines = None
    else:
        lines = None

    if lines is None:
        print("  Downloading from NGL...")
        try:
            resp = requests.get(NGL_CATALOG_URL, timeout=60, verify=False)
            resp.raise_for_status()
            lines = resp.text.strip().split('\n')

            # Cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                f.write(resp.text)

        except Exception as e:
            print(f"  ERROR downloading catalog: {e}")
            return []

    stations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                code = parts[0]
                lat = float(parts[1])
                lon = float(parts[2])
                stations.append((code, lat, lon))
            except:
                continue

    print(f"  Loaded {len(stations)} stations")
    return stations


def find_stations_near_event(stations: List[Tuple[str, float, float]],
                            event: dict) -> List[Tuple[str, float, float, float]]:
    """Find stations within search radius of event."""
    nearby = []

    for code, lat, lon in stations:
        dist = haversine_distance(lat, lon, event['lat'], event['lon'])
        if dist <= event['search_radius_km']:
            nearby.append((code, lat, lon, dist))

    # Sort by distance
    nearby.sort(key=lambda x: x[3])
    return nearby


def download_station_data(station_code: str, output_dir: Path) -> bool:
    """Download tenv3 file for a station."""
    url = f"{NGL_TENV3_BASE}/{station_code}.tenv3"
    output_file = output_dir / f"{station_code}.tenv3"

    if output_file.exists():
        return True

    try:
        resp = requests.get(url, timeout=30, verify=False)
        if resp.status_code == 200:
            with open(output_file, 'w') as f:
                f.write(resp.text)
            return True
        else:
            return False
    except Exception as e:
        return False


def download_event_data(event_key: str, event: dict, stations: List[Tuple[str, float, float]]):
    """Download GPS data for an event."""
    print(f"\n{'='*60}")
    print(f"Downloading: {event['name']}")
    print(f"{'='*60}")

    # Find nearby stations
    nearby = find_stations_near_event(stations, event)
    print(f"Found {len(nearby)} stations within {event['search_radius_km']}km")

    if len(nearby) < 4:
        print(f"  WARNING: Only {len(nearby)} stations - may be insufficient")

    # Create output directory
    output_dir = DATA_DIR / event_key / 'gps'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download station data (limit to 50 closest)
    max_stations = 50
    downloaded = 0
    failed = 0

    for i, (code, lat, lon, dist) in enumerate(nearby[:max_stations]):
        success = download_station_data(code, output_dir)
        if success:
            downloaded += 1
        else:
            failed += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{min(len(nearby), max_stations)} "
                  f"(downloaded: {downloaded}, failed: {failed})")

        # Rate limiting
        time.sleep(0.1)

    print(f"\nDownloaded {downloaded} stations for {event['name']}")

    # Save station list
    station_list_file = output_dir.parent / 'stations.txt'
    with open(station_list_file, 'w') as f:
        f.write(f"# Stations for {event['name']}\n")
        f.write(f"# Event: {event['date']}, M{event['magnitude']}\n")
        f.write(f"# Location: {event['lat']}, {event['lon']}\n")
        f.write(f"# Search radius: {event['search_radius_km']}km\n\n")
        for code, lat, lon, dist in nearby[:max_stations]:
            f.write(f"{code}\t{lat:.4f}\t{lon:.4f}\t{dist:.1f}km\n")

    return downloaded


def main():
    """Download GPS data for all validation events."""
    print("="*70)
    print("DOWNLOADING GPS DATA FOR VALIDATION EVENTS")
    print("="*70)
    print("\nThese events are NOT in the calibration set.")
    print("They will be used to validate Lambda_geo calibration.\n")

    # Load station catalog
    stations = load_ngl_catalog()
    if not stations:
        print("ERROR: Could not load station catalog")
        return

    # Download each event
    results = {}
    for event_key, event in VALIDATION_EVENTS.items():
        n_downloaded = download_event_data(event_key, event, stations)
        results[event_key] = {
            'name': event['name'],
            'stations_downloaded': n_downloaded,
        }

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"\n{'Event':<35} {'Stations'}")
    print("-"*50)
    for event_key, r in results.items():
        print(f"{r['name']:<35} {r['stations_downloaded']}")
    print("-"*50)

    total = sum(r['stations_downloaded'] for r in results.values())
    print(f"\nTotal stations downloaded: {total}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
