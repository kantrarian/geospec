#!/usr/bin/env python3
"""
download_gps_data.py
Download NGL GPS .tenv3 files for historical earthquake events.

Data source: Nevada Geodetic Laboratory (NGL)
URL: http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/

Usage:
    python download_gps_data.py --event ridgecrest_2019
    python download_gps_data.py --all
    python download_gps_data.py --list

Author: R.J. Mathews
Date: January 2026
"""

import os
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data' / 'raw'

# NGL base URL for tenv3 files
NGL_BASE_URL = "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14"

# GPS stations for each event
# Selected based on proximity to epicenter and data availability
GPS_STATIONS = {
    'ridgecrest_2019': {
        'event_date': '2019-07-06',
        'location': (35.77, -117.60),
        'stations': [
            # PBO/NOTA stations near Ridgecrest
            'P595',  # China Lake area
            'P594',  # Searles Valley
            'P580',  # Mojave
            'P591',  # Ridgecrest
            'P583',  # Near Garlock
            'CCCC',  # China Lake
            'P582',  # Near event
            'P590',  # Near event
            'LINC',  # Lincoln
        ],
    },
    'tohoku_2011': {
        'event_date': '2011-03-11',
        'location': (38.30, 142.37),
        'stations': [
            # GEONET stations (Japan) - some available via NGL
            # Using IGS stations instead as GEONET requires separate access
            'MIZU',  # Mizusawa
            'USUD',  # Usuda
            'TSKB',  # Tsukuba
            'MTKA',  # Mitaka
            'AIRA',  # Aira
        ],
    },
    'turkey_2023': {
        'event_date': '2023-02-06',
        'location': (37.22, 37.02),
        'stations': [
            # IGS/EUREF stations in Turkey region
            'ANKA',  # Ankara
            'ISTA',  # Istanbul
            'IZMI',  # Izmir
            'TUBI',  # Gebze
            'ANKR',  # Ankara IGS
        ],
    },
    'chile_2010': {
        'event_date': '2010-02-27',
        'location': (-35.85, -72.72),
        'stations': [
            # IGS stations in Chile/South America
            'SANT',  # Santiago
            'CONZ',  # Concepcion
            'LPGS',  # La Plata (Argentina)
            'BRAZ',  # Brasilia
            'RIOG',  # Rio Grande
        ],
    },
    'morocco_2023': {
        'event_date': '2023-09-08',
        'location': (31.06, -8.39),
        'stations': [
            # IGS stations in Morocco/North Africa region
            'RABT',  # Rabat
            'MAS1',  # Maspalomas (Canary Islands)
            'LPAL',  # La Palma
            'CAGL',  # Cagliari
            'VILL',  # Villafranca
        ],
    },
    'kaikoura_2016': {
        'event_date': '2016-11-13',
        'location': (-42.69, 173.02),
        'stations': [
            # GeoNet NZ stations available via NGL
            'WGTN',  # Wellington
            'CHAT',  # Chatham Islands
            'AUCK',  # Auckland
            'DUNT',  # Dunedin
            'MQZG',  # MacQuarie
        ],
    },
    'anchorage_2018': {
        'event_date': '2018-11-30',
        'location': (61.35, -149.93),
        'stations': [
            # PBO/NOTA Alaska stations - use 4-char codes
            'AV01',  # Anchorage area
            'AV02',
            'AV06',
            'AV08',
            'AV09',
            'AV10',
            'AV13',
            'AV14',
            'AV15',
            'AV17',
            'AV26',
            'AV27',
            'AV29',
            'AV30',
            'PRIOR',  # Prior Island
            'AC17',
            'AC27',
            'AC37',
        ],
    },
    'kumamoto_2016': {
        'event_date': '2016-04-16',
        'location': (32.75, 130.76),
        'stations': [
            # Using same Japan IGS stations as Tohoku
            'AIRA',  # Closest to Kumamoto
            'MIZU',
            'USUD',
            'TSKB',
            'MTKA',
        ],
    },
    'hualien_2024': {
        'event_date': '2024-04-03',
        'location': (23.77, 121.67),
        'stations': [
            # Taiwan CWB/IGS stations
            'FLNM',  # Hualien area
            'HUAL',  # Hualien
            'YULI',  # Yuli
            'CHEN',  # Chengong
            'DULI',  # Dulan
            'S101',  # Taiwan
            'S104',
            'S105',
            'TASI',
            'PKGM',
            'TAIW',  # Taiwan IGS
            'TWTF',  # Taiwan time/freq
        ],
    },
}


def download_station(station_code: str, output_dir: Path, max_retries: int = 3) -> bool:
    """Download a single station's tenv3 file from NGL with retries."""
    import time

    url = f"{NGL_BASE_URL}/{station_code}.tenv3"
    output_file = output_dir / f"{station_code}.tenv3"

    if output_file.exists():
        size = output_file.stat().st_size
        if size > 1000:
            print(f"    {station_code}: Already exists ({size:,} bytes), skipping")
            return True

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"    {station_code}: Retry {attempt + 1}...", end=" ", flush=True)
                time.sleep(2)  # Wait before retry
            else:
                print(f"    {station_code}: Downloading...", end=" ", flush=True)

            # Use custom opener with longer timeout
            req = urllib.request.Request(url, headers={'User-Agent': 'GeoSpec/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

            # Check data size
            if len(data) < 1000:
                print(f"WARNING (only {len(data)} bytes)")
                continue

            # Write to file
            with open(output_file, 'wb') as f:
                f.write(data)

            print(f"OK ({len(data):,} bytes)")
            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"NOT FOUND (404)")
                return False
            else:
                print(f"HTTP ERROR {e.code}")
                if attempt < max_retries - 1:
                    continue
                return False

        except Exception as e:
            err_msg = str(e)[:50]
            if attempt < max_retries - 1:
                print(f"RETRY ({err_msg})")
                continue
            print(f"FAILED: {err_msg}")
            return False

    return False


def download_event(event_key: str) -> dict:
    """Download all GPS data for an event."""
    if event_key not in GPS_STATIONS:
        print(f"ERROR: Unknown event '{event_key}'")
        return {'success': 0, 'failed': 0}

    config = GPS_STATIONS[event_key]
    stations = config['stations']
    output_dir = DATA_DIR / event_key / 'gps'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Downloading GPS data: {event_key}")
    print(f"{'=' * 60}")
    print(f"Event date: {config['event_date']}")
    print(f"Location: {config['location']}")
    print(f"Stations: {len(stations)}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    success = 0
    failed = 0

    for station in stations:
        if download_station(station, output_dir):
            success += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Results: {success} downloaded, {failed} failed")

    return {'success': success, 'failed': failed}


def main():
    parser = argparse.ArgumentParser(
        description='Download NGL GPS .tenv3 files for historical events'
    )
    parser.add_argument('--event', type=str, help='Specific event to download')
    parser.add_argument('--all', action='store_true', help='Download all events')
    parser.add_argument('--list', action='store_true', help='List available events')

    args = parser.parse_args()

    if args.list:
        print("Available events for GPS data download:")
        print("-" * 60)
        for key, config in GPS_STATIONS.items():
            n_stations = len(config['stations'])
            print(f"  {key:20s} {config['event_date']} ({n_stations} stations)")
        return

    if args.event:
        result = download_event(args.event)
        if result['success'] < 4:
            print(f"\nWARNING: Only {result['success']} stations downloaded.")
            print("Lambda_geo computation requires at least 4 stations.")

    elif args.all:
        total_success = 0
        total_failed = 0

        for event_key in GPS_STATIONS:
            result = download_event(event_key)
            total_success += result['success']
            total_failed += result['failed']

        print(f"\n{'=' * 60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total stations downloaded: {total_success}")
        print(f"Total stations failed: {total_failed}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
