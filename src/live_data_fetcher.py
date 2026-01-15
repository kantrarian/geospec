"""
Live GPS Data Fetcher for Lambda_geo Operations

Fetches NGL Rapid solutions (IGS20 frame) with ~2 day latency.
This is the PRIMARY data source for operational Lambda_geo computation.

Data Source: Nevada Geodetic Laboratory (NGL)
URL: https://geodesy.unr.edu/gps_timeseries/IGS20/rapids/
Format: tenv3 (graticule distance coordinates)
Latency: ~24-48 hours

Author: R.J. Mathews
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NGL Rapid Solutions (IGS20 Frame) - organized by tectonic plate
NGL_RAPIDS_BASE = "https://geodesy.unr.edu/gps_timeseries/IGS20/rapids"

# Plate codes for major regions
PLATES = {
    'NA': 'North America',
    'PA': 'Pacific',
    'EU': 'Eurasia',
    'AF': 'Africa',
    'SA': 'South America',
    'AU': 'Australia',
    'AN': 'Antarctica',
    'IN': 'India',
    'AR': 'Arabia',
    'CA': 'Caribbean',
    'CO': 'Cocos',
    'NZ': 'Nazca',
    'OK': 'Okhotsk',
    'ON': 'Okinawa',
    'PS': 'Philippine Sea',
    'PM': 'Panama',
}

# Pre-defined station lists for monitored regions
# These are stations near major fault systems with good data continuity
REGION_STATIONS = {
    'ridgecrest': {
        'plate': 'NA',
        'stations': ['P595', 'P594', 'P580', 'P591', 'P593', 'P592',
                     'P579', 'P581', 'P582', 'P583', 'CCCC'],
        'bounds': (35.0, 36.5, -118.5, -117.0),  # lat_min, lat_max, lon_min, lon_max
    },
    'socal_saf_mojave': {
        'plate': 'NA',
        'stations': ['CCCC', 'P580', 'P579', 'P595', 'DSSC', 'OPCL',
                     'P575', 'P576', 'P577', 'P578', 'GVRS', 'P630'],
        'bounds': (34.0, 36.0, -118.5, -116.5),
    },
    'socal_coachella': {
        'plate': 'NA',
        'stations': ['P507', 'P508', 'P509', 'P510', 'P511', 'P512',
                     'P513', 'P514', 'P515', 'P516', 'WIDC', 'COTD'],
        'bounds': (33.0, 34.5, -117.0, -115.5),
    },
    'norcal_hayward': {
        'plate': 'NA',
        'stations': ['P224', 'P225', 'P226', 'P227', 'P228', 'P229',
                     'P230', 'P231', 'P232', 'MHCB', 'OHLN'],
        'bounds': (37.0, 38.5, -122.5, -121.5),
    },
    'cascadia': {
        'plate': 'NA',
        'stations': ['P403', 'P404', 'P405', 'P406', 'P407', 'P408',
                     'P409', 'P410', 'P411', 'P412', 'ALBH', 'PABH'],
        'bounds': (45.0, 49.0, -125.0, -122.0),
    },
    'tokyo_kanto': {
        'plate': 'OK',  # Okhotsk plate for Japan
        'stations': ['TSKB', '3009', '3011', '0602', 'USUD', 'GMSD'],  # GEONET & IGS
        'bounds': (35.0, 37.0, 139.0, 141.0),
    },
    'istanbul_marmara': {
        'plate': 'EU',
        'stations': ['ISTB', 'TUBI', 'ANKR', 'MERS', 'NICO', 'RAMO'],  # IGS stations
        'bounds': (40.0, 42.0, 27.0, 31.0),
    },
}


def parse_tenv3_line(line: str) -> Optional[Dict]:
    """
    Parse a single line of NGL tenv3 format.

    tenv3 columns:
    1.  station name
    2.  date (YYMMMDD format, e.g., 26JAN07)
    3.  decimal year
    4.  modified Julian day
    5.  GPS week
    6.  day of GPS week
    7.  reference longitude
    8.  eastings integer (m)
    9.  eastings fractional (m)
    10. northings integer (m)
    11. northings fractional (m)
    12. vertical integer (m)
    13. vertical fractional (m)
    14. antenna height (m)
    15. east sigma (m)
    16. north sigma (m)
    17. vertical sigma (m)
    18-20. correlation coefficients
    21. nominal latitude
    22. nominal longitude
    23. nominal height
    """
    parts = line.split()
    if len(parts) < 23:
        return None

    try:
        station = parts[0]
        date_str = parts[1]  # YYMMMDD format
        decimal_year = float(parts[2])
        mjd = int(parts[3])

        # Parse date (e.g., "26JAN07" -> 2026-01-07)
        try:
            dt = datetime.strptime(date_str, "%y%b%d")
        except ValueError:
            return None

        # Combine integer and fractional parts for coordinates
        east_int = int(parts[7])
        east_frac = float(parts[8])
        east_m = east_int + east_frac

        north_int = int(parts[9])
        north_frac = float(parts[10])
        north_m = north_int + north_frac

        up_int = int(parts[11])
        up_frac = float(parts[12])
        up_m = up_int + up_frac

        # Sigmas (uncertainties)
        sigma_e = float(parts[14])
        sigma_n = float(parts[15])
        sigma_u = float(parts[16])

        # Nominal position
        lat = float(parts[20])
        lon = float(parts[21])
        height = float(parts[22])

        return {
            'station': station,
            'datetime': dt,
            'decimal_year': decimal_year,
            'mjd': mjd,
            'east_m': east_m,
            'north_m': north_m,
            'up_m': up_m,
            'sigma_e': sigma_e,
            'sigma_n': sigma_n,
            'sigma_u': sigma_u,
            'lat': lat,
            'lon': lon,
            'height': height,
        }
    except (ValueError, IndexError) as e:
        return None


def fetch_station_data(station_id: str, plate: str = 'NA',
                       days_lookback: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch the last N days of Rapid solution data for a specific station.

    Args:
        station_id: 4-character station ID (e.g., 'P595')
        plate: Tectonic plate code (e.g., 'NA' for North America)
        days_lookback: Number of days of historical data to retrieve

    Returns:
        DataFrame with columns: [datetime, east_m, north_m, up_m, sigma_e, sigma_n, sigma_u, lat, lon]
    """
    url = f"{NGL_RAPIDS_BASE}/{plate}/{station_id}.{plate}.tenv3"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Could not fetch data for {station_id}: HTTP {response.status_code}")
            return None

        cutoff_date = datetime.now() - timedelta(days=days_lookback)
        data = []

        for line in response.text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue

            record = parse_tenv3_line(line)
            if record is None:
                continue

            if record['datetime'] < cutoff_date:
                continue

            data.append(record)

        if not data:
            logger.warning(f"No recent data found for {station_id}")
            return None

        df = pd.DataFrame(data)
        df = df.sort_values('datetime').reset_index(drop=True)

        return df

    except requests.RequestException as e:
        logger.error(f"Network error fetching {station_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing {station_id}: {e}")
        return None


def fetch_region_data(region: str, days_lookback: int = 90) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all stations in a predefined region.

    Args:
        region: Region name (e.g., 'ridgecrest', 'socal_saf_mojave')
        days_lookback: Number of days of historical data

    Returns:
        Dictionary mapping station_id -> DataFrame
    """
    if region not in REGION_STATIONS:
        raise ValueError(f"Unknown region: {region}. Valid: {list(REGION_STATIONS.keys())}")

    config = REGION_STATIONS[region]
    plate = config['plate']
    stations = config['stations']

    logger.info(f"Fetching {len(stations)} stations for {region} (plate: {plate})")

    network_data = {}
    for station in stations:
        logger.info(f"  Fetching {station}...")
        df = fetch_station_data(station, plate, days_lookback)
        if df is not None and not df.empty:
            network_data[station] = df
            latest = df.iloc[-1]['datetime'].strftime('%Y-%m-%d')
            logger.info(f"    -> {len(df)} records (latest: {latest})")
        else:
            logger.warning(f"    -> No data")

    return network_data


def get_network_positions(network_data: Dict[str, pd.DataFrame],
                          target_date: datetime) -> pd.DataFrame:
    """
    Extract station positions for a specific date.

    Args:
        network_data: Dictionary from fetch_region_data()
        target_date: Date to extract positions for

    Returns:
        DataFrame with columns: [station, lat, lon, east_m, north_m, up_m]
    """
    positions = []

    for station, df in network_data.items():
        # Find closest date
        df['date_diff'] = abs((df['datetime'] - target_date).dt.total_seconds())
        closest = df.loc[df['date_diff'].idxmin()]

        if closest['date_diff'] > 86400 * 2:  # Skip if more than 2 days off
            continue

        positions.append({
            'station': station,
            'lat': closest['lat'],
            'lon': closest['lon'],
            'east_m': closest['east_m'],
            'north_m': closest['north_m'],
            'up_m': closest['up_m'],
            'datetime': closest['datetime'],
        })

    return pd.DataFrame(positions)


def compute_position_velocities(network_data: Dict[str, pd.DataFrame],
                                window_days: int = 7) -> Dict[str, pd.DataFrame]:
    """
    Compute position velocities (dE/dt, dN/dt, dU/dt) for strain rate calculation.

    Uses a rolling window to compute velocities from position time series.

    Args:
        network_data: Dictionary from fetch_region_data()
        window_days: Window size for velocity computation

    Returns:
        Dictionary mapping station_id -> DataFrame with velocity columns added
    """
    velocity_data = {}

    for station, df in network_data.items():
        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)

        # Compute time differences in days
        df['dt_days'] = df['datetime'].diff().dt.total_seconds() / 86400

        # Compute position differences
        df['dE'] = df['east_m'].diff()
        df['dN'] = df['north_m'].diff()
        df['dU'] = df['up_m'].diff()

        # Compute velocities (m/day) using rolling mean for smoothing
        df['vE'] = (df['dE'] / df['dt_days']).rolling(window=window_days, min_periods=1).mean()
        df['vN'] = (df['dN'] / df['dt_days']).rolling(window=window_days, min_periods=1).mean()
        df['vU'] = (df['dU'] / df['dt_days']).rolling(window=window_days, min_periods=1).mean()

        # Also compute accelerations for [Ė, Ë] commutator
        df['dvE'] = df['vE'].diff() / df['dt_days']
        df['dvN'] = df['vN'].diff() / df['dt_days']
        df['dvU'] = df['vU'].diff() / df['dt_days']

        velocity_data[station] = df.dropna(subset=['vE', 'vN', 'vU'])

    return velocity_data


def check_data_latency(network_data: Dict[str, pd.DataFrame]) -> Tuple[datetime, int]:
    """
    Check the most recent data available across all stations.

    Returns:
        Tuple of (most_recent_date, number_of_stations_with_that_date)
    """
    latest_dates = []
    for station, df in network_data.items():
        if not df.empty:
            latest_dates.append(df['datetime'].max())

    if not latest_dates:
        return None, 0

    most_recent = max(latest_dates)
    count = sum(1 for d in latest_dates if d == most_recent)

    return most_recent, count


if __name__ == "__main__":
    print("=" * 60)
    print("GeoSpec Live Data Fetcher - NGL IGS20 Rapid Solutions")
    print("=" * 60)

    # Test with Ridgecrest region (site of 2019 M7.1)
    region = 'ridgecrest'
    print(f"\nFetching data for: {region}")
    print("-" * 40)

    network_data = fetch_region_data(region, days_lookback=30)

    if network_data:
        latest_date, count = check_data_latency(network_data)
        latency = (datetime.now() - latest_date).days if latest_date else None

        print(f"\n{'='*40}")
        print(f"SUMMARY")
        print(f"{'='*40}")
        print(f"Stations retrieved: {len(network_data)}")
        print(f"Most recent data:   {latest_date.strftime('%Y-%m-%d') if latest_date else 'N/A'}")
        print(f"Data latency:       {latency} days")
        print(f"\nReady for Lambda_geo strain computation.")
    else:
        print("ERROR: No data retrieved. Check network connection.")


# ============================================================================
# Real-Time Integration (NTRIP)
# ============================================================================

try:
    from ntrip_client import NtripClient
except ImportError:
    NtripClient = None

class RealTimeGNSSStreamer:
    """
    Real-time GNSS streamer using IGS-IP NTRIP.
    Expects NMEA solution streams (GPGGA) for direct position input.
    """
    
    def __init__(self, caster_url: str, mountpoints: Dict[str, str], user: str = None, password: str = None):
        """
        Args:
            caster_url: NTRIP Caster URL
            mountpoints: Dict mapping station_id -> mountpoint_name
            user: NTRIP username
            password: NTRIP password
        """
        self.caster_url = caster_url
        self.mountpoints = mountpoints
        self.user = user
        self.password = password
        self.clients = {}
        self.running = False
        
    def stream_positions(self):
        """
        Generator yielding real-time positions.
        Yields: (station_id, datetime, lat, lon, height)
        """
        if NtripClient is None:
            logger.error("NtripClient module not found. Real-time streaming unavailable.")
            return

        import pynmea2  # Assuming pynmea2 is available, or use simple parsing
        
        # Connect to all mountpoints (simplified: handling one for demo or threading needed)
        # For this implementation, we simply warn that threading is needed for multiple
        if len(self.mountpoints) > 1:
            logger.warning("Streaming multiple mountpoints requires threading. Using first one only.")
            
        station, mountpoint = list(self.mountpoints.items())[0]
        
        client = NtripClient(self.caster_url, mountpoint, self.user, self.password)
        leftover = client.connect()
        
        if not client.connected:
            return
            
        buffer = b""
        if leftover:
            buffer += leftover
            
        for chunk in client.read_stream():
            buffer += chunk
            
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                try:
                    line = line_bytes.decode('ascii', errors='ignore').strip()
                    if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                        # Parse NMEA
                        # $GPGGA,hhmmss.ss,llll.ll,a,yyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh
                        parts = line.split(',')
                        if len(parts) < 10:
                            continue
                            
                        # Extract basic data (simplified parsing)
                        time_str = parts[1]
                        lat_val = float(parts[2]) if parts[2] else 0.0
                        lat_dir = parts[3]
                        lon_val = float(parts[4]) if parts[4] else 0.0
                        lon_dir = parts[5]
                        height = float(parts[9]) if parts[9] else 0.0
                        
                        # Convert to decimal degrees
                        lat_deg = int(lat_val / 100)
                        lat_min = lat_val % 100
                        lat = lat_deg + lat_min/60.0
                        if lat_dir == 'S': lat = -lat
                        
                        lon_deg = int(lon_val / 100)
                        lon_min = lon_val % 100
                        lon = lon_deg + lon_min/60.0
                        if lon_dir == 'W': lon = -lon
                        
                        now = datetime.utcnow() # Use current date as NMEA often lacks date
                        
                        yield {
                            'station': station,
                            'datetime': now,
                            'lat': lat,
                            'lon': lon,
                            'height': height,
                            'source': 'NTRIP_REALTIME'
                        }
                        
                except Exception as e:
                    continue


def create_realtime_streamer_from_env(mountpoints: Dict[str, str] = None) -> Optional[RealTimeGNSSStreamer]:
    """
    Factory function to create RealTimeGNSSStreamer from environment variables.
    
    Environment variables (from .env):
        IGS_NTRIP_USER: NTRIP username
        IGS_NTRIP_PASSWORD: NTRIP password
        IGS_NTRIP_CASTER: Caster hostname (default: igs-ip.net)
        IGS_NTRIP_PORT: Caster port (default: 2101)
    
    Args:
        mountpoints: Dict mapping station_id -> mountpoint_name
                    If None, uses default California/Ridgecrest mountpoints
    
    Returns:
        RealTimeGNSSStreamer instance or None if credentials not configured
    """
    user = os.environ.get('IGS_NTRIP_USER')
    password = os.environ.get('IGS_NTRIP_PASSWORD')
    caster = os.environ.get('IGS_NTRIP_CASTER', 'igs-ip.net')
    port = os.environ.get('IGS_NTRIP_PORT', '2101')
    
    if not user or not password:
        logger.warning("NTRIP credentials not configured. Set IGS_NTRIP_USER and IGS_NTRIP_PASSWORD in .env")
        return None
    
    caster_url = f"http://{caster}:{port}"
    
    # Default mountpoints for California region if not specified
    if mountpoints is None:
        mountpoints = {
            'P595': 'P595',  # Near Ridgecrest
            'P580': 'P580',  # Near Ridgecrest  
        }
    
    logger.info(f"Creating RealTimeGNSSStreamer for {caster_url}")
    return RealTimeGNSSStreamer(caster_url, mountpoints, user, password)
