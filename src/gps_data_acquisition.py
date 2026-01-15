#!/usr/bin/env python3
"""
gps_data_acquisition.py
Download GPS time series from Nevada Geodetic Lab (NGL).

RETROSPECTIVE VALIDATION MODE
=============================
This fetches processed daily position solutions from NGL for historical
earthquake validation. NGL provides:

- Rapid Products: ~24-hour latency
- Final Products: ~2-week latency (higher accuracy)

For Ridgecrest 2019, we use Final Products (best accuracy for validation).

Data Format: tenv3 (daily troposphere-corrected positions)
Temporal Resolution: 1 sample/day (dt = 24 hours)
Precursor Window: 72 hours = 3 daily samples

URL: http://geodesy.unr.edu/gps_timeseries/tenv3/

Author: R.J. Mathews
Date: January 2026
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class GPSStation:
    """GPS station metadata."""
    code: str
    lat: float
    lon: float
    network: str
    distance_km: float = 0.0
    has_data: bool = False


@dataclass
class EarthquakeTarget:
    """Target earthquake for retrospective validation."""
    key: str
    name: str
    date: datetime
    lat: float
    lon: float
    magnitude: float
    search_radius_deg: float = 3.0
    data_window_days: int = 30


# ============================================================================
# RIDGECREST 2019 - KNOWN GOOD STATIONS
# ============================================================================
# These stations are confirmed to have data for the Ridgecrest 2019 event
# from the PBO (Plate Boundary Observatory) network processed by NGL.

RIDGECREST_STATIONS = {
    # Station: (lat, lon, distance_from_epicenter_km)
    "P595": (35.7572, -117.3906, 20),   # Very close to epicenter
    "P594": (35.8889, -117.4278, 22),   # Close
    "TONO": (35.0795, -117.8012, 80),   # Moderate distance
    "P580": (35.3842, -117.6892, 45),   # Moderate
    "P591": (35.4394, -117.0264, 60),   # Moderate
    "BILL": (36.0478, -117.9039, 50),   # Moderate
    "P627": (36.0533, -117.8469, 52),   # Moderate
    "P628": (36.2831, -117.7958, 65),   # Moderate
    "LINC": (34.8128, -117.1503, 115),  # Farther
    "BKAP": (35.0458, -119.0403, 145),  # Farther
    "P580": (35.3842, -117.6892, 45),   # Moderate
    "GNPS": (34.3169, -118.1542, 170),  # LA Basin
    "AZRY": (33.5408, -116.6317, 260),  # Southern CA
    "BEPK": (33.7442, -117.5458, 230),  # Orange County
    "CACT": (34.2583, -116.0331, 200),  # Joshua Tree
}

# Japan stations for Tohoku 2011 - BEST INSTRUMENTED EARTHQUAKE EVER
# Epicenter: 38.322N, 142.369E (offshore)
# GEONET has 1,200+ stations - using closest onshore stations
TOHOKU_STATIONS = {
    # Closest to epicenter (within 1.5 degrees - onshore)
    "X071": (38.398, 141.534, 75),     # Very close
    "S057": (38.495, 141.531, 80),     # Very close
    "J550": (38.301, 141.501, 80),     # Very close
    "S054": (38.267, 141.478, 85),     # Very close
    "S056": (38.586, 141.487, 85),     # Close
    "G205": (39.020, 141.753, 90),     # Close
    "S058": (38.364, 141.435, 90),     # Close
    "J036": (38.449, 141.441, 90),     # Close
    "J171": (39.024, 141.740, 90),     # Close
    "S066": (38.986, 141.670, 95),     # Close
    "J172": (38.903, 141.573, 95),     # Close
    "J175": (38.683, 141.449, 95),     # Close
    "S065": (39.158, 141.835, 95),     # Close
    "S055": (38.791, 141.494, 95),     # Close
    # Additional stations for better coverage
    "MIZU": (39.135, 141.133, 120),    # Mizusawa - important
    "J546": (39.143, 141.575, 110),    # 
    "J549": (38.425, 141.213, 115),    #
    "J914": (38.743, 141.318, 110),    #
    "J918": (38.510, 141.304, 105),    #
    "J170": (39.254, 141.798, 105),    #
}

# Chile 2010 M8.8 - Major subduction earthquake
# Epicenter: -35.846S, -72.719W
CHILE_STATIONS = {
    # Closest to epicenter
    "PELL": (-35.828, -72.606, 15),    # VERY close!
    "CBQC": (-36.147, -72.805, 35),    # Very close
    "CAUQ": (-35.968, -72.341, 45),    # Close
    "PLLN": (-35.491, -72.512, 45),    # Close
    "VITA": (-36.424, -72.865, 65),    # Close
    "CONS": (-35.331, -72.412, 60),    # Close
    "NIHU": (-36.395, -72.397, 70),    # Close
    "QLAP": (-36.084, -72.125, 70),    # Close
    "NRVL": (-35.544, -72.095, 75),    # Moderate
    "SOLD": (-36.700, -73.138, 100),   # Moderate
    "SJAV": (-35.595, -71.733, 105),   # Moderate
    "CONT": (-36.843, -73.025, 110),   # Concepcion
    "CONZ": (-36.844, -73.026, 110),   # Concepcion alt
    "HLPN": (-36.748, -73.190, 105),   # Moderate
    "CLL1": (-36.595, -72.080, 100),   # Moderate
}

# Morocco 2023 M6.8 - Atlas Mountains (different tectonic setting)
# Epicenter: 31.055N, -8.396W
# Note: Local Moroccan stations lack 2023 data, using regional stations
MOROCCO_STATIONS = {
    # North Africa stations with 2023 data
    "RABT": (33.998, -6.854, 340),     # Rabat, Morocco - HAS 2023 DATA!
    "TETN": (35.562, -5.365, 590),     # Tetouan, Morocco
    # Southern Spain/Gibraltar (closest with good data)
    "SFER": (36.464, -6.206, 630),     # San Fernando, Spain
    "CEUT": (35.892, -5.307, 580),     # Ceuta, Spain
    "ALME": (36.854, -2.459, 720),     # Almeria, Spain
    "MALA": (36.726, -4.391, 680),     # Malaga, Spain
    # Canary Islands (for triangulation coverage)
    "MAS1": (27.764, -15.633, 850),    # Maspalomas, Gran Canaria
    "LPAL": (28.764, -17.894, 970),    # La Palma
}

# Turkey stations - COMPREHENSIVE LIST near 2023 epicenter (37.226N, 37.014E)
# Updated from NGL station database search - 387 stations in Turkey region!
TURKEY_STATIONS = {
    # Closest to epicenter (within 1 degree)
    "ANTP": (37.065, 37.374, 35),      # Antep - CLOSEST!
    "KLIS": (36.709, 37.112, 60),      # Kilis - very close
    "ONIY": (37.102, 36.254, 70),      # Osmaniye
    "KAHR": (37.593, 36.862, 50),      # Kahramanmaras  
    "EKIZ": (38.059, 37.188, 95),      # Ekinozu
    "HAT1": (36.200, 36.156, 130),     # Hatay
    "HATA": (36.208, 36.153, 130),     # Hatay 2
    "ARST": (36.416, 35.885, 135),     # Arsuz
    # Close (1-2 degrees)
    "SURF": (37.192, 38.818, 165),     # Sanliurfa
    "SUF1": (37.168, 38.801, 165),     # Sanliurfa alt
    "HRR2": (37.175, 39.006, 180),     # Harran
    "HRRN": (37.175, 38.997, 180),     # Harran alt
    "AKLE": (36.710, 38.948, 180),     # Akale
    "ADIY": (37.746, 38.230, 130),     # Adiyaman
    "ADY1": (37.761, 38.261, 130),     # Adiyaman alt
    "MALY": (38.338, 38.217, 165),     # Malatya
    "MLY1": (38.342, 38.319, 170),     # Malatya alt
    "ELAZ": (38.645, 39.256, 230),     # Elazig
    "ADAN": (37.003, 35.344, 160),     # Adana
    "MERS": (36.566, 34.256, 260),     # Mersin
    # Additional coverage
    "DIYB": (37.954, 40.187, 290),     # Diyarbakir
    "SIVS": (39.744, 37.002, 280),     # Sivas
    "ERZI": (39.746, 39.506, 350),     # Erzincan
}


# ============================================================================
# NGL DATA DOWNLOADER
# ============================================================================

class NGLDataDownloader:
    """
    Download GPS time series from Nevada Geodetic Lab.
    
    NGL URL Structure (HTTPS required, SSL verification disabled):
    - Station list: https://geodesy.unr.edu/NGLStationPages/llh.out
    - IGS14 solutions: https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{STATION}.tenv3
    - Plate-fixed NA: https://geodesy.unr.edu/gps_timeseries/tenv3/plates/NA/{STATION}.NA.tenv3
    
    For Ridgecrest (Western US), use plates/NA (North American fixed frame).
    For Tohoku (Japan), use IGS14 (global frame).
    """
    
    BASE_URL = "https://geodesy.unr.edu"  # HTTPS required!
    
    # Reference frames - working URL patterns
    FRAMES = {
        'IGS14': f"{BASE_URL}/gps_timeseries/tenv3/IGS14",           # Global
        'NA': f"{BASE_URL}/gps_timeseries/tenv3/plates/NA",          # North America plate-fixed
        'plates': f"{BASE_URL}/gps_timeseries/tenv3/plates",         # Plate-fixed
    }
    
    def __init__(self, cache_dir: Path):
        """Initialize NGL downloader."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Disable SSL warnings (NGL has certificate issues)
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.session = requests.Session()
        self.session.verify = False  # Bypass SSL certificate verification
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) GeoSpec-LambdaGeo/1.0'
        })
    
    def get_known_stations(self, earthquake_key: str) -> Dict[str, GPSStation]:
        """Get known stations for a specific earthquake."""
        if earthquake_key == 'ridgecrest_2019':
            stations_dict = RIDGECREST_STATIONS
        elif earthquake_key == 'tohoku_2011':
            stations_dict = TOHOKU_STATIONS
        elif earthquake_key == 'turkey_2023':
            stations_dict = TURKEY_STATIONS
        elif earthquake_key == 'chile_2010':
            stations_dict = CHILE_STATIONS
        elif earthquake_key == 'morocco_2023':
            stations_dict = MOROCCO_STATIONS
        else:
            return {}
        
        return {
            code: GPSStation(
                code=code,
                lat=info[0],
                lon=info[1],
                network='PBO' if 'P' in code else 'NGL',
                distance_km=info[2]
            )
            for code, info in stations_dict.items()
        }
    
    def download_station_tenv3(self,
                                station_code: str,
                                output_dir: Path,
                                frame: str = 'NA') -> Optional[Path]:
        """
        Download tenv3 file for a single station.
        
        Args:
            station_code: 4-character station code
            output_dir: Directory to save data
            frame: Reference frame ('IGS14', 'NA')
            
        Returns:
            Path to downloaded file, or None if failed
        """
        output_file = output_dir / f"{station_code}.tenv3"
        
        # Check cache first
        if output_file.exists() and output_file.stat().st_size > 1000:
            return output_file
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Working URL patterns (HTTPS with correct paths)
        urls_to_try = []
        
        if frame == 'NA':
            # North America plate-fixed frame
            urls_to_try = [
                f"{self.BASE_URL}/gps_timeseries/tenv3/plates/NA/{station_code}.NA.tenv3",
                f"{self.BASE_URL}/gps_timeseries/tenv3/IGS14/{station_code}.tenv3",
            ]
        else:  # IGS14
            urls_to_try = [
                f"{self.BASE_URL}/gps_timeseries/tenv3/IGS14/{station_code}.tenv3",
                f"{self.BASE_URL}/gps_timeseries/tenv3/plates/NA/{station_code}.NA.tenv3",
            ]
        
        for url in urls_to_try:
            try:
                response = self.session.get(url, timeout=60)
                
                if response.status_code == 200:
                    content = response.content
                    # Verify it's actual tenv3 data (header starts with "site")
                    if len(content) > 1000 and (b'site' in content[:100] or b'SITE' in content[:100]):
                        with open(output_file, 'wb') as f:
                            f.write(content)
                        return output_file
                        
            except requests.RequestException as e:
                continue
        
        return None
    
    def parse_tenv3(self, 
                    filepath: Path,
                    start_date: datetime,
                    end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Parse NGL tenv3 format and filter to date range.
        
        tenv3 format (space-delimited, header line starts with 'site'):
        Col 0: site (4 char station code)
        Col 1: YYMMMDD (date string like "19JUN22")
        Col 2: yyyy.yyyy (decimal year)
        Col 3: MJD
        Col 4: GPS week
        Col 5: day of week
        Col 6: reflon
        Col 7: e0 reference
        Col 8: east position (m)
        Col 9: n0 reference  
        Col 10: north position (m)
        Col 11: u0 reference
        Col 12: up position (m)
        Col 13: antenna
        Col 14: sig_e (m)
        Col 15: sig_n (m)
        Col 16: sig_u (m)
        
        Returns:
            DataFrame with columns: datetime, n, e, u, sig_n, sig_e, sig_u (in mm)
        """
        try:
            data = []
            
            # Read file and handle different line endings
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use splitlines() to handle all line ending types
            for line in content.splitlines():
                # Skip header and empty lines
                if line.startswith('site') or line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) < 17:
                    continue
                
                try:
                    # Decimal year to datetime (column 2)
                    dec_year = float(parts[2])
                    year = int(dec_year)
                    day_of_year = (dec_year - year) * 365.25
                    dt = datetime(year, 1, 1) + timedelta(days=day_of_year)
                    
                    # Filter to date range
                    if dt < start_date or dt > end_date:
                        continue
                    
                    # Positions in meters -> mm (columns 8, 10, 12)
                    e_pos = float(parts[8]) * 1000   # East
                    n_pos = float(parts[10]) * 1000  # North
                    u_pos = float(parts[12]) * 1000  # Up
                    
                    # Sigmas (columns 14, 15, 16)
                    sig_e = float(parts[14]) * 1000
                    sig_n = float(parts[15]) * 1000
                    sig_u = float(parts[16]) * 1000
                    
                    data.append({
                        'datetime': dt,
                        'n': n_pos,
                        'e': e_pos,
                        'u': u_pos,
                        'sig_n': sig_n,
                        'sig_e': sig_e,
                        'sig_u': sig_u
                    })
                    
                except (ValueError, IndexError) as e:
                    continue
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def compute_daily_velocities(self, 
                                  df: pd.DataFrame,
                                  smoothing_days: int = 3) -> pd.DataFrame:
        """
        Compute velocities from daily positions using central differences.
        
        For daily data: dt = 24 hours = 1 day
        Velocity = (position[t+1] - position[t-1]) / 2 days
        
        Units: mm/day
        """
        df = df.copy()
        n = len(df)
        
        if n < 3:
            df['vn'] = 0.0
            df['ve'] = 0.0
            df['vu'] = 0.0
            return df
        
        # Central differences (dt = 2 days for central diff)
        vn = np.zeros(n)
        ve = np.zeros(n)
        vu = np.zeros(n)
        
        # Interior points: central difference
        vn[1:-1] = (df['n'].values[2:] - df['n'].values[:-2]) / 2.0
        ve[1:-1] = (df['e'].values[2:] - df['e'].values[:-2]) / 2.0
        vu[1:-1] = (df['u'].values[2:] - df['u'].values[:-2]) / 2.0
        
        # Boundaries: forward/backward difference
        vn[0] = df['n'].values[1] - df['n'].values[0]
        ve[0] = df['e'].values[1] - df['e'].values[0]
        vu[0] = df['u'].values[1] - df['u'].values[0]
        
        vn[-1] = df['n'].values[-1] - df['n'].values[-2]
        ve[-1] = df['e'].values[-1] - df['e'].values[-2]
        vu[-1] = df['u'].values[-1] - df['u'].values[-2]
        
        # Optional smoothing
        if smoothing_days > 1 and n >= smoothing_days:
            from scipy.ndimage import uniform_filter1d
            vn = uniform_filter1d(vn, size=smoothing_days)
            ve = uniform_filter1d(ve, size=smoothing_days)
            vu = uniform_filter1d(vu, size=smoothing_days)
        
        df['vn'] = vn
        df['ve'] = ve
        df['vu'] = vu
        
        return df
    
    def download_earthquake_data(self,
                                  earthquake_key: str,
                                  output_dir: Path,
                                  eq_config: dict) -> Dict[str, pd.DataFrame]:
        """
        Download all GPS data for an earthquake validation.
        
        RETROSPECTIVE MODE: Uses known station lists and date windows.
        """
        print(f"\n{'='*60}")
        print(f"DOWNLOADING GPS DATA: {eq_config['name']}")
        print(f"{'='*60}")
        print(f"Epicenter: {eq_config['lat']:.3f}N, {eq_config['lon']:.3f}E")
        print(f"Date: {eq_config['date']}")
        print(f"Window: {eq_config['data_window_days']} days before event")
        
        # Get known stations
        stations = self.get_known_stations(earthquake_key)
        print(f"Known stations: {len(stations)}")
        
        if not stations:
            print("WARNING: No known stations for this earthquake")
            return {}
        
        # Date range
        eq_date = datetime.fromisoformat(eq_config['date'].replace('Z', ''))
        start_date = eq_date - timedelta(days=eq_config['data_window_days'])
        end_date = eq_date + timedelta(days=1)
        
        # Determine reference frame
        if eq_config['lon'] > -130 and eq_config['lon'] < -60:
            frame = 'NA'  # North America plate-fixed
        else:
            frame = 'IGS14'  # Global
        
        print(f"Reference frame: {frame}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Download each station
        station_dir = output_dir / earthquake_key / "gps"
        station_dir.mkdir(parents=True, exist_ok=True)
        
        station_data = {}
        station_locations = {}
        
        print("\nDownloading station data...")
        for code, station in tqdm(stations.items(), desc="Stations"):
            filepath = self.download_station_tenv3(code, station_dir, frame)
            
            if filepath:
                df = self.parse_tenv3(filepath, start_date, end_date)
                
                if df is not None and len(df) >= 5:
                    # Compute velocities
                    df = self.compute_daily_velocities(df)
                    station_data[code] = df
                    station_locations[code] = (station.lat, station.lon)
                    
            time.sleep(0.2)  # Be nice to NGL servers
        
        print(f"\nSuccessfully downloaded: {len(station_data)} stations")
        
        # Save metadata
        metadata = {
            'earthquake_key': earthquake_key,
            'earthquake_info': eq_config,
            'stations_attempted': len(stations),
            'stations_successful': len(station_data),
            'station_codes': list(station_data.keys()),
            'station_locations': station_locations,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'reference_frame': frame,
            'temporal_resolution': 'daily',
            'dt_hours': 24
        }
        
        with open(station_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return station_data, station_locations


def download_ridgecrest_2019(output_dir: Path) -> Tuple[Dict, Dict]:
    """
    Convenience function to download Ridgecrest 2019 data.
    
    This is the primary validation target with excellent PBO coverage.
    """
    downloader = NGLDataDownloader(output_dir / "cache" / "ngl")
    
    eq_config = {
        "name": "2019 Ridgecrest M7.1",
        "date": "2019-07-06T03:19:53",
        "lat": 35.770,
        "lon": -117.599,
        "magnitude": 7.1,
        "data_window_days": 14
    }
    
    return downloader.download_earthquake_data(
        'ridgecrest_2019', 
        output_dir / "raw",
        eq_config
    )


def main():
    """Test GPS data acquisition for Ridgecrest 2019."""
    print("=" * 70)
    print("NGL GPS DATA ACQUISITION - RIDGECREST 2019")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Download Ridgecrest data
    station_data, station_locations = download_ridgecrest_2019(data_dir)
    
    if station_data:
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"Stations: {len(station_data)}")
        
        for code, df in list(station_data.items())[:3]:
            print(f"\n{code}: {len(df)} daily samples")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"  Velocity range (N): {df['vn'].min():.2f} to {df['vn'].max():.2f} mm/day")
    else:
        print("\nNo data downloaded - check network connection")


if __name__ == "__main__":
    main()
