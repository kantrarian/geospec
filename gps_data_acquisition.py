#!/usr/bin/env python3
"""
gps_data_acquisition.py
=======================
Download GPS time series from the "Big Three" geodetic networks:
1. Nevada Geodetic Laboratory (NGL) - Global coverage, 17,000+ stations
2. UNAVCO/EarthScope (GAGE/NOTA) - Americas, especially US West Coast
3. GEONET (GSI Japan) - Japan, densest network in the world

Author: R.J. Mathews
Date: January 2026
"""

import os
import re
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from io import StringIO
import time


@dataclass
class GPSStation:
    """GPS station metadata and time series."""
    code: str
    lat: float
    lon: float
    network: str
    times: np.ndarray           # datetime objects
    east_mm: np.ndarray         # East displacement (mm)
    north_mm: np.ndarray        # North displacement (mm)
    up_mm: np.ndarray           # Vertical displacement (mm)
    east_sigma: np.ndarray      # East uncertainty (mm)
    north_sigma: np.ndarray     # North uncertainty (mm)
    up_sigma: np.ndarray        # Vertical uncertainty (mm)


class NGLDataAcquisition:
    """
    Nevada Geodetic Laboratory Data Acquisition.
    
    NGL processes 17,000+ stations globally with consistent processing.
    Data format: .tenv3 files (ENV = East, North, Vertical)
    
    URL structure:
    http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{STATION}.tenv3
    """
    
    BASE_URL = "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14"
    STATION_LIST_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.station_catalog = None
        
    def load_station_catalog(self) -> pd.DataFrame:
        """Load the global station catalog from NGL."""
        cache_file = self.cache_dir / "ngl_station_catalog.csv"
        
        if cache_file.exists():
            # Check if cache is recent (< 7 days)
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(days=7):
                print("Loading cached NGL station catalog...")
                self.station_catalog = pd.read_csv(cache_file)
                return self.station_catalog
        
        print("Downloading NGL station catalog...")
        try:
            response = requests.get(self.STATION_LIST_URL, timeout=60)
            response.raise_for_status()
            
            # Parse the llh.out format (space-delimited: station lat lon height)
            lines = response.text.strip().split('\n')
            stations = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    stations.append({
                        'code': parts[0],
                        'lat': float(parts[1]),
                        'lon': float(parts[2]),
                        'height': float(parts[3])
                    })
            
            self.station_catalog = pd.DataFrame(stations)
            self.station_catalog.to_csv(cache_file, index=False)
            print(f"Loaded {len(self.station_catalog)} stations")
            
        except Exception as e:
            print(f"Error downloading catalog: {e}")
            # Create empty catalog
            self.station_catalog = pd.DataFrame(columns=['code', 'lat', 'lon', 'height'])
        
        return self.station_catalog
    
    def find_stations_in_region(self, 
                                 lat_min: float, lat_max: float,
                                 lon_min: float, lon_max: float,
                                 max_stations: int = 100) -> List[str]:
        """Find stations within a geographic bounding box."""
        if self.station_catalog is None:
            self.load_station_catalog()
        
        mask = (
            (self.station_catalog['lat'] >= lat_min) &
            (self.station_catalog['lat'] <= lat_max) &
            (self.station_catalog['lon'] >= lon_min) &
            (self.station_catalog['lon'] <= lon_max)
        )
        
        stations = self.station_catalog[mask]['code'].tolist()
        
        if len(stations) > max_stations:
            print(f"Found {len(stations)} stations, limiting to {max_stations}")
            # Prefer stations with longer names (often higher quality)
            stations = sorted(stations, key=len, reverse=True)[:max_stations]
        
        return stations
    
    def download_station_data(self, 
                               station_code: str,
                               start_date: datetime,
                               end_date: datetime) -> Optional[GPSStation]:
        """
        Download time series for a single station.
        
        NGL .tenv3 format columns:
        YYMMMDD YYYY.YYYY __MJD week d reflon    e0(m) n0(m) u0(m) 
        ant  e(m) n(m)  u(m) sig_e(m) sig_n(m) sig_u(m) corr_en corr_eu corr_nu
        """
        url = f"{self.BASE_URL}/{station_code}.tenv3"
        cache_file = self.cache_dir / f"{station_code}.tenv3"
        
        # Check cache
        if cache_file.exists():
            print(f"  Loading cached: {station_code}")
            with open(cache_file, 'r') as f:
                content = f.read()
        else:
            print(f"  Downloading: {station_code}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    print(f"    Not found (HTTP {response.status_code})")
                    return None
                content = response.text
                # Cache the file
                with open(cache_file, 'w') as f:
                    f.write(content)
                time.sleep(0.5)  # Be nice to the server
            except Exception as e:
                print(f"    Error: {e}")
                return None
        
        # Parse the data
        try:
            lines = content.strip().split('\n')
            
            times = []
            east = []
            north = []
            up = []
            sig_e = []
            sig_n = []
            sig_u = []
            lat = None
            lon = None
            
            for line in lines:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                    
                parts = line.split()
                if len(parts) < 16:
                    continue
                
                try:
                    # Parse decimal year
                    decimal_year = float(parts[1])
                    year = int(decimal_year)
                    day_of_year = (decimal_year - year) * 365.25
                    dt = datetime(year, 1, 1) + timedelta(days=day_of_year)
                    
                    # Check date range
                    if dt < start_date or dt > end_date:
                        continue
                    
                    # Get reference position (first valid line)
                    if lat is None:
                        # reflon is in column 5, we need to look up lat from catalog
                        pass
                    
                    times.append(dt)
                    east.append(float(parts[7]) * 1000)   # m to mm
                    north.append(float(parts[8]) * 1000)
                    up.append(float(parts[9]) * 1000)
                    sig_e.append(float(parts[10]) * 1000)
                    sig_n.append(float(parts[11]) * 1000)
                    sig_u.append(float(parts[12]) * 1000)
                    
                except (ValueError, IndexError):
                    continue
            
            if len(times) < 10:
                print(f"    Insufficient data points: {len(times)}")
                return None
            
            # Get lat/lon from catalog
            if self.station_catalog is not None:
                station_row = self.station_catalog[
                    self.station_catalog['code'] == station_code
                ]
                if len(station_row) > 0:
                    lat = station_row.iloc[0]['lat']
                    lon = station_row.iloc[0]['lon']
            
            if lat is None or lon is None:
                print(f"    No coordinates found")
                return None
            
            return GPSStation(
                code=station_code,
                lat=lat,
                lon=lon,
                network='NGL',
                times=np.array(times),
                east_mm=np.array(east),
                north_mm=np.array(north),
                up_mm=np.array(up),
                east_sigma=np.array(sig_e),
                north_sigma=np.array(sig_n),
                up_sigma=np.array(sig_u)
            )
            
        except Exception as e:
            print(f"    Parse error: {e}")
            return None
    
    def download_region(self,
                        lat_center: float, lon_center: float,
                        radius_deg: float,
                        start_date: datetime,
                        end_date: datetime,
                        max_stations: int = 50) -> List[GPSStation]:
        """Download all stations within radius of a point."""
        
        lat_min = lat_center - radius_deg
        lat_max = lat_center + radius_deg
        lon_min = lon_center - radius_deg
        lon_max = lon_center + radius_deg
        
        station_codes = self.find_stations_in_region(
            lat_min, lat_max, lon_min, lon_max, max_stations
        )
        
        print(f"Downloading {len(station_codes)} stations...")
        
        stations = []
        for code in station_codes:
            station = self.download_station_data(code, start_date, end_date)
            if station is not None:
                stations.append(station)
        
        print(f"Successfully downloaded {len(stations)} stations")
        return stations


class UNAVCODataAcquisition:
    """
    UNAVCO/EarthScope GAGE Data Acquisition.
    
    Manages NOTA (Network of the Americas) including former PBO stations.
    Best for US West Coast earthquakes (Ridgecrest, etc.)
    
    Note: UNAVCO data often requires authentication for bulk downloads.
    This implementation uses their public-facing endpoints where possible.
    """
    
    # UNAVCO moved to EarthScope - new endpoints
    BASE_URL = "https://www.unavco.org/data/gps-gnss/derived/derived.html"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_station_list(self, region: str = "california") -> List[Dict]:
        """
        Get station list for a region.
        
        For Ridgecrest, relevant stations include:
        P595, P093, CCCC, P627, P628, OASI, BILL, etc.
        """
        # Known stations near Ridgecrest (35.77N, -117.60W)
        ridgecrest_stations = [
            {'code': 'P595', 'lat': 35.9127, 'lon': -117.3848},
            {'code': 'P093', 'lat': 35.5623, 'lon': -117.6689},
            {'code': 'CCCC', 'lat': 35.5249, 'lon': -117.0628},
            {'code': 'P627', 'lat': 35.5769, 'lon': -117.0556},
            {'code': 'P628', 'lat': 35.8164, 'lon': -117.5950},
            {'code': 'OASI', 'lat': 35.7647, 'lon': -117.5753},
            {'code': 'BILL', 'lat': 36.0593, 'lon': -117.9052},
            {'code': 'GOLD', 'lat': 35.4251, 'lon': -116.8892},
        ]
        
        if region.lower() == "ridgecrest":
            return ridgecrest_stations
        
        return []
    
    def get_station_data_via_ngl(self, 
                                  station_code: str,
                                  start_date: datetime,
                                  end_date: datetime) -> Optional[GPSStation]:
        """
        Many UNAVCO/PBO stations are also processed by NGL.
        Use NGL as a backup data source.
        """
        ngl = NGLDataAcquisition(self.cache_dir)
        return ngl.download_station_data(station_code, start_date, end_date)


class GEONETDataAcquisition:
    """
    GEONET (GSI Japan) Data Acquisition.
    
    Japan's GPS network with ~1200 stations - densest in the world.
    Critical for Tohoku 2011 validation.
    
    Note: GEONET data often requires registration. 
    NGL mirrors many GEONET stations (codes like 0001, 0002, etc.)
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_tohoku_stations(self) -> List[Dict]:
        """
        Key GEONET stations for Tohoku region.
        
        The Tohoku earthquake epicenter: 38.297N, 142.373E
        """
        # Major GEONET stations in Tohoku region
        # NGL codes for these stations
        tohoku_stations = [
            {'code': '0175', 'lat': 38.26, 'lon': 140.85},
            {'code': '0550', 'lat': 39.02, 'lon': 141.11},
            {'code': '0181', 'lat': 38.45, 'lon': 141.30},
            {'code': '0183', 'lat': 38.07, 'lon': 140.62},
            {'code': '0556', 'lat': 38.90, 'lon': 141.57},
            {'code': '0171', 'lat': 37.76, 'lon': 140.47},
            {'code': '0173', 'lat': 37.95, 'lon': 140.10},
            {'code': '0180', 'lat': 38.30, 'lon': 141.00},
        ]
        return tohoku_stations
    
    def download_via_ngl(self,
                          start_date: datetime,
                          end_date: datetime,
                          max_stations: int = 50) -> List[GPSStation]:
        """
        Download GEONET data via NGL mirror.
        
        GEONET stations in NGL use 4-digit numeric codes.
        """
        ngl = NGLDataAcquisition(self.cache_dir)
        
        # Tohoku region bounds
        lat_min, lat_max = 36.0, 40.0
        lon_min, lon_max = 139.0, 143.0
        
        # Find all NGL stations in region (includes GEONET mirrors)
        ngl.load_station_catalog()
        
        # Filter for numeric codes (GEONET style)
        if ngl.station_catalog is not None:
            geonet_mask = ngl.station_catalog['code'].str.match(r'^\d{4}$')
            region_mask = (
                (ngl.station_catalog['lat'] >= lat_min) &
                (ngl.station_catalog['lat'] <= lat_max) &
                (ngl.station_catalog['lon'] >= lon_min) &
                (ngl.station_catalog['lon'] <= lon_max)
            )
            
            candidates = ngl.station_catalog[geonet_mask & region_mask]
            station_codes = candidates['code'].tolist()[:max_stations]
            
            print(f"Found {len(station_codes)} GEONET stations in NGL")
        else:
            # Fallback to known stations
            station_codes = [s['code'] for s in self.get_tohoku_stations()]
        
        stations = []
        for code in station_codes:
            station = ngl.download_station_data(code, start_date, end_date)
            if station is not None:
                station.network = 'GEONET'
                stations.append(station)
        
        return stations


# =============================================================================
# EARTHQUAKE-SPECIFIC DATA ACQUISITION
# =============================================================================

def download_tohoku_data(cache_dir: Path, 
                          days_before: int = 30) -> List[GPSStation]:
    """Download GPS data for Tohoku 2011 validation."""
    
    eq_time = datetime(2011, 3, 11, 5, 46, 24)
    start_date = eq_time - timedelta(days=days_before)
    end_date = eq_time + timedelta(days=1)
    
    print("="*60)
    print("TOHOKU 2011 DATA ACQUISITION")
    print(f"Earthquake: {eq_time}")
    print(f"Data window: {start_date} to {end_date}")
    print("="*60)
    
    geonet = GEONETDataAcquisition(cache_dir / "geonet")
    stations = geonet.download_via_ngl(start_date, end_date, max_stations=50)
    
    return stations


def download_ridgecrest_data(cache_dir: Path,
                              days_before: int = 14) -> List[GPSStation]:
    """Download GPS data for Ridgecrest 2019 validation."""
    
    # M7.1 mainshock
    eq_time = datetime(2019, 7, 6, 3, 19, 53)
    # M6.4 foreshock was July 4, 2019 - 34 hours before
    start_date = eq_time - timedelta(days=days_before)
    end_date = eq_time + timedelta(days=1)
    
    print("="*60)
    print("RIDGECREST 2019 DATA ACQUISITION")
    print(f"M7.1 Mainshock: {eq_time}")
    print(f"M6.4 Foreshock: 2019-07-04 (34 hours before)")
    print(f"Data window: {start_date} to {end_date}")
    print("="*60)
    
    ngl = NGLDataAcquisition(cache_dir / "ngl")
    
    # Ridgecrest epicenter: 35.77N, -117.60W
    stations = ngl.download_region(
        lat_center=35.77,
        lon_center=-117.60,
        radius_deg=2.0,
        start_date=start_date,
        end_date=end_date,
        max_stations=50
    )
    
    return stations


def download_turkey_data(cache_dir: Path,
                          days_before: int = 14) -> List[GPSStation]:
    """Download GPS data for Turkey 2023 validation."""
    
    eq_time = datetime(2023, 2, 6, 1, 17, 35)
    start_date = eq_time - timedelta(days=days_before)
    end_date = eq_time + timedelta(days=1)
    
    print("="*60)
    print("TURKEY 2023 DATA ACQUISITION")
    print(f"M7.8 Earthquake: {eq_time}")
    print("NOTE: No foreshocks detected - key test case!")
    print(f"Data window: {start_date} to {end_date}")
    print("="*60)
    
    ngl = NGLDataAcquisition(cache_dir / "ngl")
    
    # Turkey epicenter: 37.226N, 37.014E
    stations = ngl.download_region(
        lat_center=37.226,
        lon_center=37.014,
        radius_deg=3.0,  # Larger radius - sparser network
        start_date=start_date,
        end_date=end_date,
        max_stations=50
    )
    
    return stations


def save_stations_to_file(stations: List[GPSStation], 
                           output_file: Path,
                           earthquake_info: Dict):
    """Save downloaded station data to NPZ file."""
    
    # Find common time grid
    all_times = set()
    for s in stations:
        all_times.update(s.times)
    common_times = sorted(all_times)
    
    print(f"Common time range: {common_times[0]} to {common_times[-1]}")
    print(f"Total time points: {len(common_times)}")
    
    # Build arrays
    n_times = len(common_times)
    n_stations = len(stations)
    
    east = np.full((n_times, n_stations), np.nan)
    north = np.full((n_times, n_stations), np.nan)
    up = np.full((n_times, n_stations), np.nan)
    sig_e = np.full((n_times, n_stations), np.nan)
    sig_n = np.full((n_times, n_stations), np.nan)
    sig_u = np.full((n_times, n_stations), np.nan)
    
    time_to_idx = {t: i for i, t in enumerate(common_times)}
    
    lats = np.array([s.lat for s in stations])
    lons = np.array([s.lon for s in stations])
    codes = np.array([s.code for s in stations])
    
    for s_idx, station in enumerate(stations):
        for t_idx, t in enumerate(station.times):
            if t in time_to_idx:
                i = time_to_idx[t]
                east[i, s_idx] = station.east_mm[t_idx]
                north[i, s_idx] = station.north_mm[t_idx]
                up[i, s_idx] = station.up_mm[t_idx]
                sig_e[i, s_idx] = station.east_sigma[t_idx]
                sig_n[i, s_idx] = station.north_sigma[t_idx]
                sig_u[i, s_idx] = station.up_sigma[t_idx]
    
    # Save
    np.savez(
        output_file,
        times=np.array([t.isoformat() for t in common_times]),
        station_codes=codes,
        station_lats=lats,
        station_lons=lons,
        east_mm=east,
        north_mm=north,
        up_mm=up,
        sigma_east_mm=sig_e,
        sigma_north_mm=sig_n,
        sigma_up_mm=sig_u,
        earthquake_info=earthquake_info
    )
    
    print(f"Saved to: {output_file}")
    print(f"  Stations: {n_stations}")
    print(f"  Time points: {n_times}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Download GPS data for all target earthquakes."""
    
    base_dir = Path(__file__).parent / "data"
    
    # Tohoku 2011
    print("\n" + "="*70)
    stations = download_tohoku_data(base_dir / "cache", days_before=30)
    if stations:
        save_stations_to_file(
            stations,
            base_dir / "raw/tohoku_2011/gps_timeseries.npz",
            {
                'name': '2011 Tohoku M9.0',
                'date': '2011-03-11T05:46:24',
                'magnitude': 9.0,
                'lat': 38.297,
                'lon': 142.373
            }
        )
    
    # Ridgecrest 2019
    print("\n" + "="*70)
    stations = download_ridgecrest_data(base_dir / "cache", days_before=14)
    if stations:
        save_stations_to_file(
            stations,
            base_dir / "raw/ridgecrest_2019/gps_timeseries.npz",
            {
                'name': '2019 Ridgecrest M7.1',
                'date': '2019-07-06T03:19:53',
                'magnitude': 7.1,
                'lat': 35.770,
                'lon': -117.599
            }
        )
    
    # Turkey 2023
    print("\n" + "="*70)
    stations = download_turkey_data(base_dir / "cache", days_before=14)
    if stations:
        save_stations_to_file(
            stations,
            base_dir / "raw/turkey_2023/gps_timeseries.npz",
            {
                'name': '2023 Turkey-Syria M7.8',
                'date': '2023-02-06T01:17:35',
                'magnitude': 7.8,
                'lat': 37.226,
                'lon': 37.014
            }
        )
    
    print("\n" + "="*70)
    print("DATA ACQUISITION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
