#!/usr/bin/env python3
"""
live_data.py - Live GPS Data Acquisition for Prospective Monitoring

Downloads real GPS data from Nevada Geodetic Laboratory (NGL) and computes
Lambda_geo for each monitored region.

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
import requests
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.spatial import Delaunay
from scipy.signal import savgol_filter
from scipy.linalg import eigh
import json
import time

warnings.filterwarnings('ignore')

# Add parent path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.src.regions import FAULT_POLYGONS


@dataclass
class StationData:
    """GPS station time series data."""
    code: str
    lat: float
    lon: float
    times: List[datetime]
    east_mm: np.ndarray
    north_mm: np.ndarray
    up_mm: np.ndarray


@dataclass
class RegionLambdaGeo:
    """Lambda_geo results for a region."""
    region_id: str
    date: datetime
    lambda_geo_max: float
    lambda_geo_mean: float
    lambda_geo_grid: np.ndarray
    n_stations: int
    n_triangles: int
    station_codes: List[str]
    data_quality: str  # 'good', 'sparse', 'insufficient'


class NGLLiveAcquisition:
    """
    Download live GPS data from Nevada Geodetic Laboratory.

    NGL provides daily GPS solutions for 17,000+ stations globally.

    Reference frames:
    - IGS20/tenv: Current data (within 2 weeks of today) - PREFERRED
    - IGS14/tenv3: Historical data (ends ~Aug 2024)
    """

    # IGS20 has current data, IGS14 is historical
    BASE_URL_IGS20 = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv"
    BASE_URL_IGS14 = "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14"
    CATALOG_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"

    # Use IGS20 by default for current data
    USE_IGS20 = True

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.station_catalog = None
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.verify = False  # NGL has SSL issues sometimes
            self._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) GeoSpec/1.0',
                'Accept': 'text/plain,*/*',
            })
        return self._session

    def load_station_catalog(self) -> Dict[str, Tuple[float, float]]:
        """Load global station catalog from NGL."""
        cache_file = self.cache_dir / "ngl_catalog.json"

        # Use cache if recent (< 7 days)
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(days=7):
                with open(cache_file) as f:
                    self.station_catalog = json.load(f)
                return self.station_catalog

        print("  Downloading NGL station catalog...")
        try:
            response = self.session.get(self.CATALOG_URL, timeout=60)
            response.raise_for_status()

            catalog = {}
            for line in response.text.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    code = parts[0]
                    lat = float(parts[1])
                    lon = float(parts[2])
                    catalog[code] = (lat, lon)

            self.station_catalog = catalog
            with open(cache_file, 'w') as f:
                json.dump(catalog, f)

            print(f"  Loaded {len(catalog)} stations")

        except Exception as e:
            print(f"  Catalog download failed: {e}")
            self.station_catalog = {}

        return self.station_catalog

    def find_stations_in_polygon(self, polygon: List[Tuple[float, float]],
                                  buffer_deg: float = 0.5) -> List[str]:
        """Find all NGL stations within a polygon."""
        if self.station_catalog is None:
            self.load_station_catalog()

        # Get bounding box with buffer
        lats = [p[0] for p in polygon]
        lons = [p[1] for p in polygon]
        lat_min, lat_max = min(lats) - buffer_deg, max(lats) + buffer_deg
        lon_min, lon_max = min(lons) - buffer_deg, max(lons) + buffer_deg

        # NGL uses negative longitudes (-360 to 0), so convert if needed
        # Positive lons (0-180) become negative (-360 to -180)
        if lon_min > 0:
            lon_min = lon_min - 360
        if lon_max > 0:
            lon_max = lon_max - 360

        stations = []
        for code, (lat, lon) in self.station_catalog.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                # Simple bounding box check (polygon check is more expensive)
                stations.append(code)

        return stations

    def download_station(self, code: str, days_back: int = 120,
                         target_date: datetime = None) -> Optional[StationData]:
        """Download time series for a single station.

        Tries IGS20 (current data) first, falls back to IGS14 (historical).
        """
        # Try IGS20 first (current data, .tenv format)
        if self.USE_IGS20:
            cache_file_igs20 = self.cache_dir / f"{code}.igs20.tenv"

            use_cache = False
            if cache_file_igs20.exists():
                mtime = datetime.fromtimestamp(cache_file_igs20.stat().st_mtime)
                if datetime.now() - mtime < timedelta(days=1):
                    use_cache = True

            if use_cache:
                with open(cache_file_igs20, 'r') as f:
                    content = f.read()
                result = self._parse_tenv(code, content, days_back, target_date)
                if result:
                    return result
            else:
                url = f"{self.BASE_URL_IGS20}/{code}.tenv"
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        content = response.text
                        with open(cache_file_igs20, 'w') as f:
                            f.write(content)
                        time.sleep(0.2)
                        result = self._parse_tenv(code, content, days_back, target_date)
                        if result:
                            return result
                except Exception:
                    pass

        # Fall back to IGS14 (.tenv3 format) for historical data
        cache_file_igs14 = self.cache_dir / f"{code}.tenv3"

        use_cache = False
        if cache_file_igs14.exists():
            mtime = datetime.fromtimestamp(cache_file_igs14.stat().st_mtime)
            if datetime.now() - mtime < timedelta(days=1):
                use_cache = True

        if use_cache:
            with open(cache_file_igs14, 'r') as f:
                content = f.read()
        else:
            url = f"{self.BASE_URL_IGS14}/{code}.tenv3"
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    return None
                content = response.text
                with open(cache_file_igs14, 'w') as f:
                    f.write(content)
                time.sleep(0.2)
            except Exception:
                return None

        try:
            return self._parse_tenv3(code, content, days_back, target_date)
        except Exception:
            return None

    def _parse_tenv(self, code: str, content: str, days_back: int,
                    target_date: datetime = None) -> Optional[StationData]:
        """Parse NGL IGS20 TENV format (current data).

        Format: site date decyr MJD week day reflon east(m) north(m) up(m) ...
        Columns 7,8,9 are E,N,U positions in meters.
        """
        if target_date is None:
            target_date = datetime.now()
        cutoff_date = target_date - timedelta(days=days_back)

        times = []
        east = []
        north = []
        up = []
        lat, lon = None, None

        if code in self.station_catalog:
            lat, lon = self.station_catalog[code]

        for line in content.strip().split('\n'):
            if line.startswith('#') or line.startswith('site') or len(line.strip()) == 0:
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                decimal_year = float(parts[2])
                year = int(decimal_year)
                day_of_year = (decimal_year - year) * 365.25
                dt = datetime(year, 1, 1) + timedelta(days=day_of_year)

                if dt < cutoff_date:
                    continue
                
                if dt > target_date:
                    continue

                times.append(dt)
                # IGS20 tenv: columns 7,8,9 are E,N,U in meters
                east.append(float(parts[7]) * 1000)   # m to mm
                north.append(float(parts[8]) * 1000)
                up.append(float(parts[9]) * 1000)

            except (ValueError, IndexError):
                continue

        if len(times) < 30 or lat is None:
            return None

        return StationData(
            code=code,
            lat=lat,
            lon=lon,
            times=times,
            east_mm=np.array(east),
            north_mm=np.array(north),
            up_mm=np.array(up)
        )

    def _parse_tenv3(self, code: str, content: str, days_back: int,
                     target_date: datetime = None) -> Optional[StationData]:
        """Parse NGL TENV3 format."""
        if target_date is None:
            target_date = datetime.now()
        cutoff_date = target_date - timedelta(days=days_back)

        times = []
        east = []
        north = []
        up = []
        lat, lon = None, None

        if code in self.station_catalog:
            lat, lon = self.station_catalog[code]

        for line in content.strip().split('\n'):
            if line.startswith('#') or line.startswith('site') or len(line.strip()) == 0:
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            try:
                # Parse decimal year (column 2, format: yyyy.yyyy)
                decimal_year = float(parts[2])
                year = int(decimal_year)
                day_of_year = (decimal_year - year) * 365.25
                dt = datetime(year, 1, 1) + timedelta(days=day_of_year)

                if dt < cutoff_date:
                    continue
                
                if dt > target_date:
                    continue

                times.append(dt)
                # Column 8 is __east(m), column 10 is _north(m), column 12 is ____up(m)
                east.append(float(parts[8]) * 1000)   # m to mm
                north.append(float(parts[10]) * 1000)
                up.append(float(parts[12]) * 1000)

            except (ValueError, IndexError):
                continue

        if len(times) < 30 or lat is None:
            return None

        return StationData(
            code=code,
            lat=lat,
            lon=lon,
            times=times,
            east_mm=np.array(east),
            north_mm=np.array(north),
            up_mm=np.array(up)
        )


def filter_colocated_stations(stations: List[StationData],
                              min_separation_deg: float = 0.01) -> List[StationData]:
    """Remove co-located stations, keeping the first occurrence."""
    if len(stations) <= 1:
        return stations

    filtered = [stations[0]]
    for station in stations[1:]:
        is_colocated = False
        for existing in filtered:
            dist = np.sqrt((station.lat - existing.lat)**2 +
                          (station.lon - existing.lon)**2)
            if dist < min_separation_deg:
                is_colocated = True
                break
        if not is_colocated:
            filtered.append(station)

    return filtered


class StrainComputer:
    """Compute strain tensors from GPS velocities using Delaunay triangulation."""

    def __init__(self, smoothing_window: int = 7):
        self.smoothing_window = smoothing_window

    def compute_velocities(self, stations: List[StationData]) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Compute smoothed velocities from displacement time series.

        Returns:
            velocities: (n_times, n_stations, 2) array of [ve, vn] in mm/day
            positions: (n_stations, 2) array of [lat, lon]
            times: list of datetime objects
        """
        # Find common time grid
        all_times = set()
        for s in stations:
            all_times.update(s.times)
        common_times = sorted(all_times)

        if len(common_times) < 30:
            return None, None, None

        n_times = len(common_times)
        n_stations = len(stations)

        # Build displacement arrays
        east = np.full((n_times, n_stations), np.nan)
        north = np.full((n_times, n_stations), np.nan)

        time_to_idx = {t: i for i, t in enumerate(common_times)}

        for s_idx, station in enumerate(stations):
            for t_idx, t in enumerate(station.times):
                if t in time_to_idx:
                    i = time_to_idx[t]
                    east[i, s_idx] = station.east_mm[t_idx]
                    north[i, s_idx] = station.north_mm[t_idx]

        # Smooth and compute velocities
        velocities = np.zeros((n_times, n_stations, 2))

        for s in range(n_stations):
            # Interpolate NaN gaps
            e = self._interpolate_gaps(east[:, s])
            n = self._interpolate_gaps(north[:, s])

            # Smooth with Savitzky-Golay filter
            if self.smoothing_window > 0 and len(e) > self.smoothing_window:
                window = min(self.smoothing_window, len(e) // 2 * 2 - 1)
                if window >= 3:
                    e = savgol_filter(e, window, 2)
                    n = savgol_filter(n, window, 2)

            # Central difference for velocity (mm/day)
            velocities[1:-1, s, 0] = (e[2:] - e[:-2]) / 2  # ve
            velocities[1:-1, s, 1] = (n[2:] - n[:-2]) / 2  # vn
            # Forward difference at start, backward difference at end
            velocities[0, s, 0] = e[1] - e[0]
            velocities[0, s, 1] = n[1] - n[0]
            velocities[-1, s, 0] = e[-1] - e[-2]
            velocities[-1, s, 1] = n[-1] - n[-2]

        positions = np.array([[s.lat, s.lon] for s in stations])

        return velocities, positions, common_times

    def _interpolate_gaps(self, arr: np.ndarray) -> np.ndarray:
        """Linear interpolation of NaN gaps."""
        arr = arr.copy()
        nans = np.isnan(arr)
        if np.all(nans):
            return np.zeros_like(arr)

        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        return arr

    def compute_strain_field(self, velocities: np.ndarray,
                             positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute strain tensor field using Delaunay triangulation.

        Returns:
            strain_tensors: (n_times, n_triangles, 3, 3) array
            triangle_centers: (n_triangles, 2) array of [lat, lon]
        """
        n_times, n_stations, _ = velocities.shape

        if n_stations < 3:
            return None, None

        # Build Delaunay triangulation
        try:
            tri = Delaunay(positions)
        except Exception:
            return None, None

        n_triangles = len(tri.simplices)
        strain_tensors = np.zeros((n_times, n_triangles, 3, 3))
        triangle_centers = np.zeros((n_triangles, 2))

        # Compute strain for each triangle
        for tri_idx, simplex in enumerate(tri.simplices):
            i, j, k = simplex

            # Triangle vertices
            p1 = positions[i]
            p2 = positions[j]
            p3 = positions[k]

            # Triangle center
            triangle_centers[tri_idx] = (p1 + p2 + p3) / 3

            # Convert to local Cartesian (km)
            center_lat = triangle_centers[tri_idx, 0]
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))

            # Local coordinates (km from center)
            x1, y1 = (p1[1] - triangle_centers[tri_idx, 1]) * km_per_deg_lon, (p1[0] - triangle_centers[tri_idx, 0]) * km_per_deg_lat
            x2, y2 = (p2[1] - triangle_centers[tri_idx, 1]) * km_per_deg_lon, (p2[0] - triangle_centers[tri_idx, 0]) * km_per_deg_lat
            x3, y3 = (p3[1] - triangle_centers[tri_idx, 1]) * km_per_deg_lon, (p3[0] - triangle_centers[tri_idx, 0]) * km_per_deg_lat

            # Design matrix for velocity gradient
            # v = v0 + L * x where L is the velocity gradient tensor
            A = np.array([
                [1, 0, x1, y1, 0, 0],
                [0, 1, 0, 0, x1, y1],
                [1, 0, x2, y2, 0, 0],
                [0, 1, 0, 0, x2, y2],
                [1, 0, x3, y3, 0, 0],
                [0, 1, 0, 0, x3, y3],
            ])

            for t in range(n_times):
                # Velocities at vertices (mm/day -> mm/day/km for gradient)
                v = np.array([
                    velocities[t, i, 0], velocities[t, i, 1],
                    velocities[t, j, 0], velocities[t, j, 1],
                    velocities[t, k, 0], velocities[t, k, 1],
                ])

                # Solve for velocity gradient
                try:
                    params, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
                    # params = [v0_e, v0_n, dve/dx, dve/dy, dvn/dx, dvn/dy]

                    L = np.array([
                        [params[2], params[3]],
                        [params[4], params[5]]
                    ])

                    # Strain rate tensor (symmetric part)
                    E_2d = 0.5 * (L + L.T)

                    # Extend to 3D (plane strain assumption)
                    strain_tensors[t, tri_idx, 0, 0] = E_2d[0, 0]
                    strain_tensors[t, tri_idx, 0, 1] = E_2d[0, 1]
                    strain_tensors[t, tri_idx, 1, 0] = E_2d[1, 0]
                    strain_tensors[t, tri_idx, 1, 1] = E_2d[1, 1]
                    # E_33 = -(E_11 + E_22) for incompressibility
                    strain_tensors[t, tri_idx, 2, 2] = -(E_2d[0, 0] + E_2d[1, 1])

                except Exception:
                    pass

        return strain_tensors, triangle_centers


class LambdaGeoComputer:
    """Compute Lambda_geo = ||[E, E_dot]||_F."""

    def compute(self, strain_tensors: np.ndarray) -> np.ndarray:
        """
        Compute Lambda_geo for each triangle at each time.

        Args:
            strain_tensors: (n_times, n_triangles, 3, 3)

        Returns:
            lambda_geo: (n_times, n_triangles)
        """
        n_times, n_triangles = strain_tensors.shape[:2]

        # Compute E_dot via central differences
        E_dot = np.zeros_like(strain_tensors)
        E_dot[1:-1] = (strain_tensors[2:] - strain_tensors[:-2]) / 2
        E_dot[0] = strain_tensors[1] - strain_tensors[0]
        E_dot[-1] = strain_tensors[-1] - strain_tensors[-2]

        # Compute commutator [E, E_dot] = E @ E_dot - E_dot @ E
        commutator = np.einsum('...ij,...jk->...ik', strain_tensors, E_dot) - \
                     np.einsum('...ij,...jk->...ik', E_dot, strain_tensors)

        # Frobenius norm
        lambda_geo = np.sqrt(np.einsum('...ij,...ij->...', commutator, commutator))

        return lambda_geo


def acquire_region_data(region_id: str,
                        ngl: NGLLiveAcquisition,
                        days_back: int = 120,
                        target_date: datetime = None) -> Optional[RegionLambdaGeo]:
    """
    Acquire live GPS data and compute Lambda_geo for a region.
    """
    if target_date is None:
        target_date = datetime.now()

    config = FAULT_POLYGONS.get(region_id)
    if config is None:
        return None

    polygon = config['polygon']
    print(f"  Finding stations in {config['name']}...")

    # Find stations in region
    station_codes = ngl.find_stations_in_polygon(polygon)
    print(f"  Found {len(station_codes)} candidate stations")

    if len(station_codes) == 0:
        return RegionLambdaGeo(
            region_id=region_id,
            date=target_date,
            lambda_geo_max=0.0,
            lambda_geo_mean=0.0,
            lambda_geo_grid=np.array([]),
            n_stations=0,
            n_triangles=0,
            station_codes=[],
            data_quality='insufficient'
        )

    # Download station data
    print(f"  Downloading GPS data...")
    stations = []
    for code in station_codes[:50]:  # Limit to 50 stations
        station = ngl.download_station(code, days_back, target_date)
        if station is not None:
            stations.append(station)

    # Filter out co-located stations (causes degenerate triangles)
    n_before = len(stations)
    stations = filter_colocated_stations(stations, min_separation_deg=0.01)
    if len(stations) < n_before:
        print(f"  Removed {n_before - len(stations)} co-located stations")

    print(f"  Successfully downloaded {len(stations)} stations")

    if len(stations) < 3:
        return RegionLambdaGeo(
            region_id=region_id,
            date=target_date,
            lambda_geo_max=0.0,
            lambda_geo_mean=0.0,
            lambda_geo_grid=np.array([]),
            n_stations=len(stations),
            n_triangles=0,
            station_codes=[s.code for s in stations],
            data_quality='insufficient'
        )

    # Compute velocities
    print(f"  Computing velocities...")
    strain_computer = StrainComputer(smoothing_window=7)
    velocities, positions, times = strain_computer.compute_velocities(stations)

    if velocities is None:
        return RegionLambdaGeo(
            region_id=region_id,
            date=target_date,
            lambda_geo_max=0.0,
            lambda_geo_mean=0.0,
            lambda_geo_grid=np.array([]),
            n_stations=len(stations),
            n_triangles=0,
            station_codes=[s.code for s in stations],
            data_quality='insufficient'
        )

    # Compute strain field
    print(f"  Computing strain field...")
    strain_tensors, triangle_centers = strain_computer.compute_strain_field(velocities, positions)

    if strain_tensors is None:
        return RegionLambdaGeo(
            region_id=region_id,
            date=target_date,
            lambda_geo_max=0.0,
            lambda_geo_mean=0.0,
            lambda_geo_grid=np.array([]),
            n_stations=len(stations),
            n_triangles=0,
            station_codes=[s.code for s in stations],
            data_quality='insufficient'
        )

    # Compute Lambda_geo
    print(f"  Computing Lambda_geo...")
    lambda_computer = LambdaGeoComputer()
    lambda_geo = lambda_computer.compute(strain_tensors)

    # Get most recent values
    current_lambda = lambda_geo[-1] if len(lambda_geo) > 0 else np.array([0.0])

    n_triangles = strain_tensors.shape[1]
    data_quality = 'good' if len(stations) >= 10 else 'sparse'

    return RegionLambdaGeo(
        region_id=region_id,
        date=target_date,
        lambda_geo_max=float(np.nanmax(current_lambda)) if len(current_lambda) > 0 else 0.0,
        lambda_geo_mean=float(np.nanmean(current_lambda)) if len(current_lambda) > 0 else 0.0,
        lambda_geo_grid=lambda_geo,
        n_stations=len(stations),
        n_triangles=n_triangles,
        station_codes=[s.code for s in stations],
        data_quality=data_quality
    )


def acquire_all_regions(cache_dir: Path, days_back: int = 120) -> Dict[str, RegionLambdaGeo]:
    """
    Acquire live GPS data for all configured regions.
    """
    ngl = NGLLiveAcquisition(cache_dir)
    ngl.load_station_catalog()

    results = {}
    for region_id in FAULT_POLYGONS.keys():
        print(f"\n[{region_id}]")
        result = acquire_region_data(region_id, ngl, days_back)
        if result:
            results[region_id] = result
            print(f"  Lambda_geo max: {result.lambda_geo_max:.6f}")
            print(f"  Stations: {result.n_stations}, Triangles: {result.n_triangles}")
            print(f"  Quality: {result.data_quality}")

    return results


if __name__ == "__main__":
    cache_dir = Path(__file__).parent.parent / "data" / "gps_cache"
    results = acquire_all_regions(cache_dir)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for region_id, result in results.items():
        print(f"{region_id}: max={result.lambda_geo_max:.6f}, stations={result.n_stations}, quality={result.data_quality}")
