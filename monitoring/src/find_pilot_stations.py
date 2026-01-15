#!/usr/bin/env python3
"""
find_pilot_stations.py
Find IGS-IP NTRIP mountpoints near existing NGL stations for real-time pilot.

This script:
1. Fetches the IGS-IP sourcetable
2. Loads NGL station coordinates from live_data_fetcher
3. Finds nearby NTRIP mountpoints that can form good triangles
"""

import os
import sys
import math
import socket
import base64
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def fetch_sourcetable(caster: str, port: int, user: str, password: str) -> str:
    """Fetch NTRIP sourcetable."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)

    try:
        print(f"Connecting to {caster}:{port}...")
        sock.connect((caster, port))

        auth = base64.b64encode(f"{user}:{password}".encode()).decode()
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {caster}\r\n"
            f"Ntrip-Version: Ntrip/2.0\r\n"
            f"User-Agent: NTRIP GeoSpec/1.0\r\n"
            f"Authorization: Basic {auth}\r\n"
            f"\r\n"
        )
        sock.send(request.encode())

        response = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"ENDSOURCETABLE" in response:
                    break
            except socket.timeout:
                break

        return response.decode('utf-8', errors='replace')
    finally:
        sock.close()


def parse_sourcetable(text: str) -> List[Dict]:
    """Parse sourcetable into list of stream dicts."""
    streams = []
    for line in text.split('\n'):
        line = line.strip()
        if not line.startswith("STR;"):
            continue
        parts = line.split(";")
        if len(parts) < 11:
            continue
        try:
            lat = float(parts[9])
            lon = float(parts[10])
        except (ValueError, IndexError):
            continue

        streams.append({
            "mountpoint": parts[1],
            "identifier": parts[2] if len(parts) > 2 else "",
            "format": parts[3] if len(parts) > 3 else "",
            "format_details": parts[4] if len(parts) > 4 else "",
            "carrier": parts[5] if len(parts) > 5 else "",
            "nav": parts[6] if len(parts) > 6 else "",
            "network": parts[7] if len(parts) > 7 else "",
            "country": parts[8] if len(parts) > 8 else "",
            "lat": lat,
            "lon": lon,
        })
    return streams


def filter_bbox(streams: List[Dict], lat_min: float, lat_max: float,
                lon_min: float, lon_max: float) -> List[Dict]:
    """Filter streams by bounding box."""
    return [s for s in streams
            if lat_min <= s["lat"] <= lat_max and lon_min <= s["lon"] <= lon_max]


def filter_observation_streams(streams: List[Dict]) -> List[Dict]:
    """
    Filter to only observation streams (not correction streams).
    Correction streams like IGS02, IGS03 don't have real lat/lon.
    """
    return [s for s in streams
            if s["lat"] != 0.0 and s["lon"] != 0.0
            and "RTCM" in s["format"]]


def find_nearest(streams: List[Dict], targets: List[Dict], top_k: int = 20) -> List[Tuple[float, Dict, Dict]]:
    """Find mountpoints nearest to target stations."""
    results = []
    for s in streams:
        for t in targets:
            dist = haversine_km(s["lat"], s["lon"], t["lat"], t["lon"])
            results.append((dist, s, t))

    results.sort(key=lambda x: x[0])

    # Deduplicate by mountpoint (keep closest match)
    seen = set()
    unique = []
    for dist, stream, target in results:
        if stream["mountpoint"] not in seen:
            seen.add(stream["mountpoint"])
            unique.append((dist, stream, target))
            if len(unique) >= top_k:
                break

    return unique


# NGL station coordinates for SoCal/Ridgecrest region
# These are approximate - we'll verify from actual NGL data
NGL_STATIONS = {
    # Ridgecrest area
    'P595': {'lat': 35.7697, 'lon': -117.4308, 'region': 'ridgecrest'},
    'P594': {'lat': 35.6195, 'lon': -117.6547, 'region': 'ridgecrest'},
    'P580': {'lat': 35.9093, 'lon': -117.7608, 'region': 'ridgecrest'},
    'P591': {'lat': 35.4818, 'lon': -117.3459, 'region': 'ridgecrest'},
    'P593': {'lat': 35.7085, 'lon': -117.1968, 'region': 'ridgecrest'},
    'P592': {'lat': 35.5392, 'lon': -117.0933, 'region': 'ridgecrest'},

    # SoCal Mojave
    'CCCC': {'lat': 35.0253, 'lon': -117.3318, 'region': 'socal_mojave'},
    'P579': {'lat': 35.0639, 'lon': -118.1519, 'region': 'socal_mojave'},
    'P575': {'lat': 34.7892, 'lon': -118.5661, 'region': 'socal_mojave'},

    # NorCal Hayward (for comparison)
    'P224': {'lat': 37.8762, 'lon': -122.2465, 'region': 'norcal'},
    'P225': {'lat': 37.7508, 'lon': -122.1361, 'region': 'norcal'},
}


def main():
    print("=" * 70)
    print("  IGS-IP Station Finder for Lambda_geo Pilot")
    print("=" * 70)

    # Get credentials
    user = os.getenv('IGS_NTRIP_USER')
    password = os.getenv('IGS_NTRIP_PASSWORD')

    if not user or not password:
        print("ERROR: Set IGS_NTRIP_USER and IGS_NTRIP_PASSWORD in .env")
        sys.exit(1)

    # Fetch sourcetable from igs-ip.net (largest selection)
    print("\nFetching IGS-IP sourcetable...")
    sourcetable = fetch_sourcetable('igs-ip.net', 2101, user, password)

    # Save raw sourcetable
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    sourcetable_file = output_dir / 'igs-ip_sourcetable.txt'
    with open(sourcetable_file, 'w') as f:
        f.write(sourcetable)
    print(f"Saved sourcetable to: {sourcetable_file}")

    # Parse streams
    all_streams = parse_sourcetable(sourcetable)
    print(f"\nTotal mountpoints: {len(all_streams)}")

    # Filter to observation streams only
    obs_streams = filter_observation_streams(all_streams)
    print(f"Observation streams (with coordinates): {len(obs_streams)}")

    # Define regions to search
    regions = {
        'SoCal/Ridgecrest': {
            'bbox': (33.5, 36.5, -119.0, -116.0),
            'stations': ['P595', 'P594', 'P580', 'P591', 'P593', 'P592', 'CCCC', 'P579'],
        },
        'NorCal/Bay Area': {
            'bbox': (37.0, 38.5, -123.0, -121.5),
            'stations': ['P224', 'P225'],
        },
    }

    for region_name, config in regions.items():
        print(f"\n{'='*70}")
        print(f"  {region_name}")
        print('='*70)

        lat_min, lat_max, lon_min, lon_max = config['bbox']

        # Filter by bounding box
        regional = filter_bbox(obs_streams, lat_min, lat_max, lon_min, lon_max)
        print(f"\nMountpoints in bbox: {len(regional)}")

        if not regional:
            print("  No IGS-IP stations in this region!")
            print("  Consider: EarthScope NTRIP for US coverage")
            continue

        # Build target list from NGL stations
        targets = []
        for station_id in config['stations']:
            if station_id in NGL_STATIONS:
                info = NGL_STATIONS[station_id]
                targets.append({
                    'name': station_id,
                    'lat': info['lat'],
                    'lon': info['lon'],
                })

        if not targets:
            print("  No NGL target stations defined")
            continue

        # Find nearest mountpoints
        nearest = find_nearest(regional, targets, top_k=15)

        print(f"\nNearest NTRIP mountpoints to NGL stations:")
        print(f"{'Dist(km)':>8}  {'Mountpoint':<14} {'Format':<10} {'NavSys':<24} {'Near':<6}")
        print("-" * 70)

        for dist, stream, target in nearest:
            nav = stream['nav'][:22] if len(stream['nav']) > 22 else stream['nav']
            print(f"{dist:>8.1f}  {stream['mountpoint']:<14} {stream['format']:<10} {nav:<24} {target['name']:<6}")

        # Summary
        print(f"\n  Stations in region: {len(regional)}")
        if nearest:
            closest = nearest[0]
            print(f"  Closest to NGL network: {closest[1]['mountpoint']} ({closest[0]:.1f} km from {closest[2]['name']})")

    # Also check for US stations specifically
    print(f"\n{'='*70}")
    print("  All USA stations on IGS-IP")
    print('='*70)

    usa_streams = [s for s in obs_streams if s['country'] == 'USA']
    print(f"\nUSA mountpoints: {len(usa_streams)}")

    if usa_streams:
        print(f"\n{'Mountpoint':<14} {'Format':<10} {'NavSys':<28} {'Lat':>8} {'Lon':>10}")
        print("-" * 75)
        for s in sorted(usa_streams, key=lambda x: x['lat'], reverse=True)[:20]:
            nav = s['nav'][:26] if len(s['nav']) > 26 else s['nav']
            print(f"{s['mountpoint']:<14} {s['format']:<10} {nav:<28} {s['lat']:>8.3f} {s['lon']:>10.3f}")

    print("\n" + "=" * 70)
    print("  Recommendation")
    print("=" * 70)
    print("""
  IGS-IP has limited US coverage (mostly Alaska, Hawaii, Caribbean).

  For California/Cascadia pilot, you have two options:

  1. Use distant IGS-IP stations (proof of concept only)
     - Won't match NGL station geometry
     - Good for testing RTCM → position pipeline

  2. Apply for EarthScope NTRIP access (recommended for real pilot)
     - Email: rtgps@earthscope.org
     - Dense PBO network coverage (P595-style stations)
     - This is the right path for matching NGL geometry

  Next step: Email EarthScope for NTRIP access, then re-run this script
  with their sourcetable to find exact matches.
""")


if __name__ == '__main__':
    main()
