#!/usr/bin/env python3
"""
position_adapter.py
Convert RTKLIB .pos files to NGL-format DataFrames for Lambda_geo pipeline.

NGL format expected columns:
    - refepoch: decimal year
    - e, n, u: east/north/up in meters (relative to reference)
    - se, sn, su: standard errors
    - station: 4-char ID

RTKLIB .pos format (ECEF):
    %  GPST          x-ecef(m)      y-ecef(m)      z-ecef(m)   Q  ns   sdx(m)   sdy(m)   sdz(m)

Reference frame strategy:
    - Reference ECEF = median of first N epochs (default N=100)
    - Lat/Lon derived from reference ECEF
    - ENU computed relative to that reference
    This avoids hardcoded references that cause false "motion" signals.

Usage:
    python position_adapter.py --date 2026-01-11
    python position_adapter.py --station COSO00USA0 --date 2026-01-11
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_print(msg: str) -> None:
    """Print safely on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # semi-major axis (m)
WGS84_F = 1 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # first eccentricity squared


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF to geodetic latitude, longitude, height (WGS84).

    Uses iterative method for accuracy.
    Returns: (lat_deg, lon_deg, height_m)
    """
    lon = math.atan2(y, x)

    # Iterative latitude computation
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - WGS84_E2))

    for _ in range(10):  # Usually converges in 2-3 iterations
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(z, p * (1 - WGS84_E2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
    h = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), h


def ecef_to_enu(x: float, y: float, z: float,
                ref_x: float, ref_y: float, ref_z: float,
                lat_deg: float, lon_deg: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to ENU relative to reference.

    Args:
        x, y, z: ECEF position (meters)
        ref_x, ref_y, ref_z: Reference ECEF position (meters)
        lat_deg, lon_deg: Reference latitude/longitude (degrees)

    Returns:
        (east, north, up) in meters relative to reference
    """
    # Difference vector
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix ECEF -> ENU
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # ENU transformation
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def parse_datetime_to_decimal_year(date_str: str, time_str: str) -> float:
    """Parse RTKLIB datetime format to decimal year.

    Format: YYYY/MM/DD HH:MM:SS.SSS
    """
    dt_str = f"{date_str} {time_str}"
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        try:
            dt = datetime.strptime(dt_str.strip(), "%Y/%m/%d %H:%M:%S")
        except ValueError:
            # Handle other formats
            return 0.0

    year = dt.year
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    year_fraction = (dt - start_of_year).total_seconds() / (end_of_year - start_of_year).total_seconds()

    return year + year_fraction


def read_pos_file_raw(pos_file: Path) -> List[Dict]:
    """Read RTKLIB .pos file and return raw records.

    Handles both formats:
    - ECEF: YYYY/MM/DD HH:MM:SS.SSS x-ecef y-ecef z-ecef Q ns sdx sdy sdz
    - LLH:  WEEK SECS lat lon height Q ns sdn sde sdu ...

    Returns list of dicts with position data.
    """
    if not pos_file.exists():
        return []

    records = []
    is_llh_format = False

    with open(pos_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Check header for format hint
            if 'latitude' in line.lower() and 'longitude' in line.lower():
                is_llh_format = True
                continue

            # Skip comment lines
            if line.startswith('%') or line.startswith('#') or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                if is_llh_format or (parts[0].isdigit() and len(parts[0]) == 4 and int(parts[0]) > 2000):
                    # LLH format: WEEK SECS lat lon height Q ns sdn sde sdu ...
                    # Example: 2401  59339.997   35.982348846 -117.808918535  1460.6475   5   5   8.5244   9.6499  20.5605
                    gps_week = int(parts[0])
                    gps_secs = float(parts[1])
                    lat = float(parts[2])
                    lon = float(parts[3])
                    height = float(parts[4])
                    Q = int(parts[5]) if parts[5].isdigit() else 5
                    ns = int(parts[6]) if parts[6].isdigit() else 0
                    sdn = float(parts[7]) if len(parts) > 7 else 10.0
                    sde = float(parts[8]) if len(parts) > 8 else 10.0
                    sdu = float(parts[9]) if len(parts) > 9 else 20.0

                    # Skip obviously bad positions
                    if abs(lat) < 1 or abs(lon) < 1:
                        continue

                    records.append({
                        'gps_week': gps_week,
                        'gps_secs': gps_secs,
                        'lat': lat,
                        'lon': lon,
                        'height': height,
                        'Q': Q,
                        'ns': ns,
                        'sdn': sdn,
                        'sde': sde,
                        'sdu': sdu,
                        'format': 'llh',
                    })
                else:
                    # ECEF format: YYYY/MM/DD HH:MM:SS.SSS x y z Q ns sdx sdy sdz
                    date_str = parts[0]
                    time_str = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    Q = int(parts[5]) if parts[5].isdigit() else 5
                    ns = int(parts[6]) if parts[6].isdigit() else 0
                    sdx = float(parts[7])
                    sdy = float(parts[8])
                    sdz = float(parts[9]) if len(parts) > 9 else sdx

                    # Skip obviously bad positions (near origin)
                    if abs(x) < 1000 or abs(y) < 1000:
                        continue

                    records.append({
                        'date_str': date_str,
                        'time_str': time_str,
                        'x': x,
                        'y': y,
                        'z': z,
                        'Q': Q,
                        'ns': ns,
                        'sdx': sdx,
                        'sdy': sdy,
                        'sdz': sdz,
                        'format': 'ecef',
                    })

            except (ValueError, IndexError):
                continue

    return records


def compute_reference_from_data(records: List[Dict], n_epochs: int = 100) -> Optional[Dict]:
    """Compute reference position from median of first N epochs.

    Handles both LLH and ECEF format records.
    Returns dict with: lat, lon, height (and ref_x, ref_y, ref_z if ECEF)
    """
    if not records:
        return None

    # Filter to better quality epochs (Q <= 2 preferred, but take what we have)
    good_records = [r for r in records if r['Q'] <= 2 and r['ns'] >= 4]
    if len(good_records) < 10:
        good_records = [r for r in records if r['Q'] <= 3 and r['ns'] >= 3]
    if len(good_records) < 5:
        good_records = [r for r in records if r['Q'] <= 5 and r['ns'] >= 3]
    if len(good_records) < 3:
        good_records = records[:n_epochs]

    # Take first N epochs
    subset = good_records[:n_epochs]

    if len(subset) < 1:
        return None

    # Check format
    if subset[0].get('format') == 'llh':
        # LLH format - compute median lat/lon/height directly
        lat = float(np.median([r['lat'] for r in subset]))
        lon = float(np.median([r['lon'] for r in subset]))
        height = float(np.median([r['height'] for r in subset]))

        return {
            'lat': lat,
            'lon': lon,
            'height': height,
            'format': 'llh',
        }
    else:
        # ECEF format - compute median ECEF then convert to LLA
        ref_x = float(np.median([r['x'] for r in subset]))
        ref_y = float(np.median([r['y'] for r in subset]))
        ref_z = float(np.median([r['z'] for r in subset]))

        # Convert to LLA for ENU transformation
        lat, lon, height = ecef_to_lla(ref_x, ref_y, ref_z)

        return {
            'ref_x': ref_x,
            'ref_y': ref_y,
            'ref_z': ref_z,
            'lat': lat,
            'lon': lon,
            'height': height,
            'format': 'ecef',
        }


def gps_time_to_decimal_year(gps_week: int, gps_secs: float) -> float:
    """Convert GPS week and seconds to decimal year."""
    from datetime import timedelta
    # GPS epoch: January 6, 1980
    gps_epoch = datetime(1980, 1, 6)
    total_seconds = gps_week * 7 * 86400 + gps_secs
    dt = gps_epoch + timedelta(seconds=total_seconds)

    year = dt.year
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    year_fraction = (dt - start_of_year).total_seconds() / (end_of_year - start_of_year).total_seconds()

    return year + year_fraction


def lla_to_enu(lat: float, lon: float, height: float,
               ref_lat: float, ref_lon: float, ref_height: float) -> Tuple[float, float, float]:
    """Convert LLA position to ENU relative to reference.

    Uses simple approximation valid for small distances (<100km).
    Returns (east, north, up) in meters.
    """
    # Earth radius (approximate)
    R = 6378137.0

    # Convert to radians
    lat_rad = math.radians(lat)
    ref_lat_rad = math.radians(ref_lat)

    # Differences
    dlat = lat - ref_lat  # degrees
    dlon = lon - ref_lon  # degrees
    dh = height - ref_height  # meters

    # Convert to meters (approximate)
    # 1 degree latitude ~ 111km
    # 1 degree longitude ~ 111km * cos(lat)
    north = dlat * (math.pi / 180.0) * R
    east = dlon * (math.pi / 180.0) * R * math.cos(ref_lat_rad)
    up = dh

    return east, north, up


# Quality-conditional thresholds: (min_sats, max_h_sigma_m, max_v_sigma_m)
# Q: 1=fix, 2=float, 3=SBAS, 4=DGPS, 5=single, 6=PPP
QC_THRESHOLDS = {
    1: (6, 0.20, 0.40),    # fix (conservative)
    2: (6, 1.00, 2.00),    # float
    3: (6, 5.00, 10.00),   # SBAS
    4: (6, 5.00, 10.00),   # DGPS
    5: (6, 15.0, 30.0),    # single (broadcast)
    6: (6, 0.50, 1.00),    # PPP (conservative)
}


def convert_to_ngl_format(
    records: List[Dict],
    ref: Dict,
    station: str,
) -> pd.DataFrame:
    """Convert raw records to NGL-format ENU DataFrame + QC fields.

    Handles both LLH and ECEF format records.
    Adds quality flags for downstream filtering with Q-conditional thresholds.
    """
    ngl_records = []

    for r in records:
        q = int(r.get('Q', 99))
        ns = int(r.get('ns', 0))

        # Basic sanity - need at least 3 sats for a solution
        if ns < 3:
            continue

        if r.get('format') == 'llh':
            # LLH format - convert to ENU directly
            e, n, u = lla_to_enu(
                r['lat'], r['lon'], r['height'],
                ref['lat'], ref['lon'], ref['height']
            )
            refepoch = gps_time_to_decimal_year(r['gps_week'], r['gps_secs'])
            se = float(r.get('sde', np.nan))
            sn = float(r.get('sdn', np.nan))
            su = float(r.get('sdu', np.nan))
        else:
            # ECEF format - convert to ENU
            e, n, u = ecef_to_enu(
                r['x'], r['y'], r['z'],
                ref['ref_x'], ref['ref_y'], ref['ref_z'],
                ref['lat'], ref['lon']
            )
            refepoch = parse_datetime_to_decimal_year(r['date_str'], r['time_str'])
            se = (float(r.get('sdx', np.nan)) + float(r.get('sdy', np.nan))) / 2.0
            sn = se
            su = float(r.get('sdz', np.nan))

        if not refepoch:
            continue

        # Derived sigmas
        h_sigma = float(np.sqrt(se * se + sn * sn)) if np.isfinite(se) and np.isfinite(sn) else np.nan
        v_sigma = float(su) if np.isfinite(su) else np.nan

        # Q-conditional thresholds
        min_sats, max_h, max_v = QC_THRESHOLDS.get(q, (6, 15.0, 30.0))

        # Compute flags
        flag_low_sats = int(ns < min_sats)
        flag_high_sigma = int(
            (np.isfinite(h_sigma) and h_sigma > max_h) or
            (np.isfinite(v_sigma) and v_sigma > max_v)
        )
        flag_bad_q = int(q not in QC_THRESHOLDS)

        # Build human-readable reason string
        reasons = []
        if flag_low_sats:
            reasons.append(f'LOW_SATS({ns}<{min_sats})')
        if flag_high_sigma:
            reasons.append(f'HIGH_SIGMA(h={h_sigma:.1f}>{max_h}|v={v_sigma:.1f}>{max_v})')
        if flag_bad_q:
            reasons.append(f'BAD_Q({q})')
        qc_reason = ';'.join(reasons) if reasons else ''

        ngl_records.append({
            'refepoch': float(refepoch),
            'e': float(e),
            'n': float(n),
            'u': float(u),
            'se': float(se),
            'sn': float(sn),
            'su': float(su),
            'station': station[:4],  # NGL uses 4-char codes

            # QC passthrough / derived
            'q': q,
            'ns': ns,
            'h_sigma': h_sigma,
            'v_sigma': v_sigma,
            'flag_low_sats': flag_low_sats,
            'flag_high_sigma': flag_high_sigma,
            'flag_bad_q': flag_bad_q,
            'qc_reason': qc_reason,
        })

    if not ngl_records:
        return pd.DataFrame()

    return pd.DataFrame(ngl_records)


def process_station(pos_dir: Path, station: str, date_str: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Process all .pos files for a station on a given date.

    Returns: (DataFrame, reference_dict)
    """
    station_dir = pos_dir / station / date_str

    if not station_dir.exists():
        return None, None

    # Collect all raw records first
    all_records = []

    for pos_file in sorted(station_dir.glob("*.pos")):
        # Skip test files
        if pos_file.name.lower().startswith("test"):
            continue
        records = read_pos_file_raw(pos_file)
        all_records.extend(records)

    if not all_records:
        return None, None

    # Compute reference from data (median of first N epochs)
    ref = compute_reference_from_data(all_records, n_epochs=100)
    if ref is None:
        return None, None

    # Convert to NGL format
    df = convert_to_ngl_format(all_records, ref, station)

    if df.empty:
        return None, None

    df = df.sort_values('refepoch').reset_index(drop=True)

    # Round refepoch to avoid float jitter, then dedupe
    df['refepoch'] = df['refepoch'].round(10)
    df = df.drop_duplicates(subset=['station', 'refepoch'], keep='first').reset_index(drop=True)

    return df, ref


def main():
    parser = argparse.ArgumentParser(description="Convert RTKLIB .pos to NGL format")
    parser.add_argument('--date', required=True, help="Date to process (YYYY-MM-DD)")
    parser.add_argument('--station', help="Process single station only")
    parser.add_argument('--pos-dir', default='monitoring/data/positions',
                        help="Position files directory")
    parser.add_argument('--output', help="Output file path (default: auto)")
    parser.add_argument('--include-qc', action='store_true',
                        help="Include QC columns (q, ns, h_sigma, v_sigma, flags)")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent
    pos_dir = script_dir / args.pos_dir

    if not pos_dir.exists():
        safe_print(f"Position directory not found: {pos_dir}")
        safe_print("Run WSL processing first: process_rtcm.sh")
        return 1

    safe_print("=" * 60)
    safe_print("  Position Adapter: RTKLIB -> NGL Format")
    safe_print("=" * 60)
    safe_print(f"  Date: {args.date}")
    safe_print(f"  Input: {pos_dir}")
    safe_print("  Reference: computed from median of first 100 epochs")

    # Determine stations to process
    if args.station:
        stations = [args.station]
    else:
        # Find all station directories
        stations = []
        for d in pos_dir.iterdir():
            if d.is_dir() and (d / args.date).exists():
                stations.append(d.name)
        if not stations:
            stations = ['COSO00USA0', 'GOLD00USA0', 'JPLM00USA0']

    all_data = []
    all_refs = {}

    for station in stations:
        safe_print(f"\n  [{station}]")
        df, ref = process_station(pos_dir, station, args.date)

        if df is None or ref is None:
            safe_print("    No position data")
            continue

        safe_print(f"    Epochs: {len(df)}")
        if ref.get('format') == 'llh':
            safe_print(f"    Reference LLA: ({ref['lat']:.6f}, {ref['lon']:.6f}, {ref['height']:.1f}m)")
        else:
            safe_print(f"    Reference ECEF: ({ref['ref_x']:.3f}, {ref['ref_y']:.3f}, {ref['ref_z']:.3f})")
            safe_print(f"    Reference LLA: ({ref['lat']:.6f}, {ref['lon']:.6f}, {ref['height']:.1f}m)")
        safe_print(f"    Time span: {df['refepoch'].min():.6f} - {df['refepoch'].max():.6f}")
        safe_print(f"    E range: {df['e'].min()*1000:.1f} to {df['e'].max()*1000:.1f} mm")
        safe_print(f"    N range: {df['n'].min()*1000:.1f} to {df['n'].max()*1000:.1f} mm")
        safe_print(f"    U range: {df['u'].min()*1000:.1f} to {df['u'].max()*1000:.1f} mm")

        all_data.append(df)
        all_refs[station] = ref

    if not all_data:
        safe_print("\nNo position data to convert.")
        return 1

    # Combine all stations
    combined = pd.concat(all_data, ignore_index=True)

    # Output path - use _qc suffix for QC mode to preserve both outputs
    if args.output:
        output_path = Path(args.output)
    else:
        ngl_dir = script_dir / 'monitoring' / 'data' / 'ngl_format'
        ngl_dir.mkdir(parents=True, exist_ok=True)
        suffix = '_qc' if args.include_qc else ''
        output_path = ngl_dir / f'rtcm_positions_{args.date}{suffix}.csv'

    # Select columns - base NGL format or extended with QC
    ngl_columns = ['refepoch', 'e', 'n', 'u', 'se', 'sn', 'su', 'station']
    if args.include_qc:
        qc_columns = ['q', 'ns', 'h_sigma', 'v_sigma', 'flag_low_sats', 'flag_high_sigma', 'flag_bad_q', 'qc_reason']
        ngl_columns.extend([c for c in qc_columns if c in combined.columns])

    ngl_df = combined[ngl_columns].copy()

    # Save to CSV
    ngl_df.to_csv(output_path, index=False, float_format='%.8f')

    safe_print("\n" + "=" * 60)
    safe_print(f"  Output: {output_path}")
    safe_print(f"  Total epochs: {len(ngl_df)}")
    safe_print(f"  Stations: {ngl_df['station'].unique().tolist()}")
    safe_print("=" * 60)

    # Also save metadata with computed references
    meta_path = output_path.with_suffix('.json')
    metadata = {
        'date': args.date,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'source': 'RTKLIB via WSL',
        'reference_strategy': 'median_first_100_epochs',
        'include_qc': args.include_qc,
        'stations': {},
        'total_epochs': len(ngl_df),
        'time_range': {
            'start': float(ngl_df['refepoch'].min()),
            'end': float(ngl_df['refepoch'].max()),
        },
    }

    # Include QC thresholds in metadata for reproducibility
    if args.include_qc:
        metadata['qc_thresholds'] = {
            str(q): {'min_sats': t[0], 'max_h_sigma_m': t[1], 'max_v_sigma_m': t[2]}
            for q, t in QC_THRESHOLDS.items()
        }
        metadata['qc_threshold_desc'] = 'Q: 1=fix, 2=float, 3=SBAS, 4=DGPS, 5=single, 6=PPP'

    for station, ref in all_refs.items():
        station_meta = {
            'ref_lla': [ref['lat'], ref['lon'], ref['height']],
            'epochs': int(combined[combined['station'] == station[:4]].shape[0]),
        }
        if ref.get('format') == 'ecef':
            station_meta['ref_ecef'] = [ref['ref_x'], ref['ref_y'], ref['ref_z']]
        metadata['stations'][station] = station_meta

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    safe_print(f"  Metadata: {meta_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
