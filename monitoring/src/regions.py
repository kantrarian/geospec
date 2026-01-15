"""
regions.py - Region + Station Eligibility Validator

Given a fault polygon + station metadata, determines if a region
is "monitorable" before computing any alerts.

Eligibility criteria:
- Station count >= 10 (preferred >= 20)
- Station history >= 3 years
- Delaunay triangles >= 6
- Triangle quality (min angle) > 15°
- Max station spacing < 50 km
- Data completeness > 80%
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from scipy.spatial import Delaunay
import math


@dataclass
class Station:
    """GPS station metadata."""
    id: str
    lat: float
    lon: float
    start_date: datetime
    end_date: Optional[datetime] = None
    completeness: float = 1.0  # Fraction of days with data
    
    def history_years(self, as_of: datetime = None) -> float:
        """Compute years of operational history."""
        as_of = as_of or datetime.now()
        end = self.end_date or as_of
        return (end - self.start_date).days / 365.25


@dataclass
class Triangle:
    """Delaunay triangle with quality metrics."""
    vertices: Tuple[int, int, int]  # Station indices
    min_angle_deg: float
    area_km2: float
    
    @property
    def is_quality(self) -> bool:
        """Check if triangle meets quality threshold."""
        return self.min_angle_deg > 15.0


@dataclass
class RegionEligibility:
    """Eligibility assessment for a monitoring region."""
    region_id: str
    polygon_name: str
    
    # Station metrics
    n_stations: int
    n_stations_with_history: int
    min_history_years: float
    mean_completeness: float
    
    # Triangle metrics
    n_triangles: int
    n_quality_triangles: int
    min_triangle_angle: float
    mean_triangle_angle: float
    
    # Spacing metrics
    max_station_spacing_km: float
    mean_station_spacing_km: float
    
    # Overall assessment
    eligible: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'region_id': self.region_id,
            'polygon_name': self.polygon_name,
            'stations': {
                'count': self.n_stations,
                'with_history': self.n_stations_with_history,
                'min_history_years': self.min_history_years,
                'mean_completeness': self.mean_completeness,
            },
            'triangles': {
                'count': self.n_triangles,
                'quality_count': self.n_quality_triangles,
                'min_angle_deg': self.min_triangle_angle,
                'mean_angle_deg': self.mean_triangle_angle,
            },
            'spacing': {
                'max_km': self.max_station_spacing_km,
                'mean_km': self.mean_station_spacing_km,
            },
            'eligible': self.eligible,
            'reasons': self.reasons,
            'warnings': self.warnings,
        }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance between two points in kilometers."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting."""
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > lon) != (polygon[j][1] > lon) and
            lat < (polygon[j][0] - polygon[i][0]) * (lon - polygon[i][1]) / 
                  (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i
    
    return inside


def compute_triangle_angles(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    """Compute angles of a triangle in degrees."""
    # Side lengths
    a = np.linalg.norm(p2 - p3)  # Opposite to p1
    b = np.linalg.norm(p1 - p3)  # Opposite to p2
    c = np.linalg.norm(p1 - p2)  # Opposite to p3
    
    # Angles using law of cosines
    if a > 0 and b > 0 and c > 0:
        angle1 = math.degrees(math.acos(np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1)))
        angle2 = math.degrees(math.acos(np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1)))
        angle3 = 180 - angle1 - angle2
        return angle1, angle2, angle3
    
    return 0.0, 0.0, 0.0


def compute_triangle_area_km2(lat1: float, lon1: float, 
                               lat2: float, lon2: float,
                               lat3: float, lon3: float) -> float:
    """Compute approximate area of a triangle on Earth's surface."""
    # Use cross product approximation for small triangles
    # Convert to local Cartesian (km)
    center_lat = (lat1 + lat2 + lat3) / 3
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))
    
    x1, y1 = lon1 * km_per_deg_lon, lat1 * km_per_deg_lat
    x2, y2 = lon2 * km_per_deg_lon, lat2 * km_per_deg_lat
    x3, y3 = lon3 * km_per_deg_lon, lat3 * km_per_deg_lat
    
    # Shoelace formula
    area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2)
    return area


class RegionValidator:
    """
    Validates whether a region is suitable for Λ_geo monitoring.
    """
    
    # Eligibility thresholds
    MIN_STATIONS = 10
    PREFERRED_STATIONS = 20
    MIN_HISTORY_YEARS = 3.0
    MIN_TRIANGLES = 6
    MIN_TRIANGLE_ANGLE = 15.0
    MAX_STATION_SPACING_KM = 50.0
    MIN_COMPLETENESS = 0.80
    
    def __init__(self, region_id: str, polygon: List[Tuple[float, float]], 
                 buffer_km: float = 20.0):
        """
        Args:
            region_id: Unique identifier for the region
            polygon: List of (lat, lon) tuples defining the monitoring polygon
            buffer_km: Buffer around polygon for station selection (km)
        """
        self.region_id = region_id
        self.polygon = polygon
        self.buffer_km = buffer_km
        self.stations: List[Station] = []
        self.triangles: List[Triangle] = []
    
    def add_stations(self, stations: List[Station]):
        """Add stations, filtering to those within/near polygon."""
        for station in stations:
            # Check if station is in polygon or within buffer
            if self._is_near_polygon(station.lat, station.lon):
                self.stations.append(station)
    
    def _is_near_polygon(self, lat: float, lon: float) -> bool:
        """Check if point is in polygon or within buffer distance."""
        if point_in_polygon(lat, lon, self.polygon):
            return True
        
        # Check distance to polygon edges
        for i in range(len(self.polygon)):
            p1 = self.polygon[i]
            p2 = self.polygon[(i + 1) % len(self.polygon)]
            dist = haversine_km(lat, lon, p1[0], p1[1])
            if dist < self.buffer_km:
                return True
        
        return False
    
    def build_triangulation(self):
        """Build Delaunay triangulation and compute quality metrics."""
        if len(self.stations) < 3:
            return
        
        # Get station positions
        points = np.array([[s.lat, s.lon] for s in self.stations])
        
        # Build Delaunay triangulation
        try:
            tri = Delaunay(points)
        except Exception as e:
            print(f"Triangulation failed: {e}")
            return
        
        # Compute triangle metrics
        self.triangles = []
        for simplex in tri.simplices:
            i, j, k = simplex
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            # Compute angles
            angles = compute_triangle_angles(p1, p2, p3)
            min_angle = min(angles)
            
            # Compute area
            area = compute_triangle_area_km2(
                p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
            )
            
            self.triangles.append(Triangle(
                vertices=(i, j, k),
                min_angle_deg=min_angle,
                area_km2=area
            ))
    
    def compute_station_spacing(self) -> Tuple[float, float]:
        """Compute max and mean station spacing."""
        if len(self.stations) < 2:
            return float('inf'), float('inf')
        
        # Compute all pairwise distances
        distances = []
        for i, s1 in enumerate(self.stations):
            for s2 in self.stations[i+1:]:
                d = haversine_km(s1.lat, s1.lon, s2.lat, s2.lon)
                distances.append(d)
        
        # For max spacing, we want the minimum distance to nearest neighbor
        # for each station (network spacing characteristic)
        min_neighbor_dists = []
        for i, s1 in enumerate(self.stations):
            min_dist = float('inf')
            for j, s2 in enumerate(self.stations):
                if i != j:
                    d = haversine_km(s1.lat, s1.lon, s2.lat, s2.lon)
                    min_dist = min(min_dist, d)
            min_neighbor_dists.append(min_dist)
        
        return max(min_neighbor_dists), np.mean(min_neighbor_dists)
    
    def validate(self) -> RegionEligibility:
        """Run full eligibility validation."""
        
        # Build triangulation if not done
        if not self.triangles:
            self.build_triangulation()
        
        reasons = []
        warnings = []
        eligible = True
        
        # === Station Metrics ===
        n_stations = len(self.stations)
        stations_with_history = [s for s in self.stations 
                                  if s.history_years() >= self.MIN_HISTORY_YEARS]
        n_stations_with_history = len(stations_with_history)
        
        min_history = min((s.history_years() for s in self.stations), default=0)
        mean_completeness = np.mean([s.completeness for s in self.stations]) if self.stations else 0
        
        if n_stations < self.MIN_STATIONS:
            eligible = False
            reasons.append(f"Insufficient stations: {n_stations} < {self.MIN_STATIONS}")
        elif n_stations < self.PREFERRED_STATIONS:
            warnings.append(f"Suboptimal station count: {n_stations} < {self.PREFERRED_STATIONS}")
        
        if n_stations_with_history < self.MIN_STATIONS:
            eligible = False
            reasons.append(f"Insufficient stations with {self.MIN_HISTORY_YEARS}+ year history: {n_stations_with_history}")
        
        if mean_completeness < self.MIN_COMPLETENESS:
            eligible = False
            reasons.append(f"Low data completeness: {mean_completeness:.0%} < {self.MIN_COMPLETENESS:.0%}")
        
        # === Triangle Metrics ===
        n_triangles = len(self.triangles)
        quality_triangles = [t for t in self.triangles if t.is_quality]
        n_quality_triangles = len(quality_triangles)
        
        min_angle = min((t.min_angle_deg for t in self.triangles), default=0)
        mean_angle = np.mean([t.min_angle_deg for t in self.triangles]) if self.triangles else 0
        
        if n_quality_triangles < self.MIN_TRIANGLES:
            eligible = False
            reasons.append(f"Insufficient quality triangles: {n_quality_triangles} < {self.MIN_TRIANGLES}")
        
        if min_angle < 10:
            warnings.append(f"Very skinny triangles present (min angle {min_angle:.1f}°)")
        
        # === Spacing Metrics ===
        max_spacing, mean_spacing = self.compute_station_spacing()
        
        if max_spacing > self.MAX_STATION_SPACING_KM:
            eligible = False
            reasons.append(f"Station spacing too large: {max_spacing:.1f}km > {self.MAX_STATION_SPACING_KM}km")
        
        # Build result
        return RegionEligibility(
            region_id=self.region_id,
            polygon_name=f"{len(self.polygon)}-vertex polygon",
            n_stations=n_stations,
            n_stations_with_history=n_stations_with_history,
            min_history_years=min_history,
            mean_completeness=mean_completeness,
            n_triangles=n_triangles,
            n_quality_triangles=n_quality_triangles,
            min_triangle_angle=min_angle,
            mean_triangle_angle=mean_angle,
            max_station_spacing_km=max_spacing,
            mean_station_spacing_km=mean_spacing,
            eligible=eligible,
            reasons=reasons,
            warnings=warnings,
        )


# === Predefined Fault Polygons ===

FAULT_POLYGONS = {
    "ridgecrest": {
        "name": "Ridgecrest/Eastern California Shear Zone",
        "polygon": [
            (35.0, -118.5),  # SW corner
            (36.5, -118.5),  # NW corner
            (36.5, -116.5),  # NE corner
            (35.0, -116.5),  # SE corner
        ],
        "expected_m7_rate": 0.01,
    },
    "socal_saf_mojave": {
        "name": "Southern California - San Andreas (Mojave)",
        "polygon": [
            (34.0, -118.5),  # SW corner
            (35.5, -118.5),  # NW corner
            (35.5, -116.5),  # NE corner
            (34.0, -116.5),  # SE corner
        ],
        "expected_m7_rate": 0.01,  # Per year (rough)
    },
    "socal_saf_coachella": {
        "name": "Southern California - San Andreas (Coachella)",
        "polygon": [
            (33.0, -116.5),
            (34.0, -116.5),
            (34.0, -115.0),
            (33.0, -115.0),
        ],
        "expected_m7_rate": 0.008,
    },
    "norcal_hayward": {
        "name": "Northern California - Hayward Fault",
        "polygon": [
            (37.3, -122.3),
            (38.0, -122.3),
            (38.0, -121.8),
            (37.3, -121.8),
        ],
        "expected_m7_rate": 0.02,
    },
    "tokyo_kanto": {
        "name": "Tokyo/Kanto Region",
        "polygon": [
            (34.5, 138.5),
            (36.5, 138.5),
            (36.5, 141.0),
            (34.5, 141.0),
        ],
        "expected_m7_rate": 0.05,
    },
    "istanbul_marmara": {
        "name": "Istanbul - Marmara Segment",
        # Expanded polygon to capture more regional stations
        "polygon": [
            (39.5, 26.0),   # SW - includes Aegean coast
            (41.5, 26.0),   # NW
            (41.5, 32.0),   # NE - includes Black Sea coast
            (39.5, 32.0),   # SE - includes Ankara area
        ],
        "expected_m7_rate": 0.02,
    },
    "cascadia": {
        "name": "Cascadia Subduction Zone",
        "polygon": [
            (42.0, -125.0),
            (48.0, -125.0),
            (48.0, -122.0),
            (42.0, -122.0),
        ],
        "expected_m7_rate": 0.01,
    },
    "turkey_kahramanmaras": {
        "name": "Turkey - East Anatolian Fault (Kahramanmaras)",
        # 2023 M7.8 earthquake region
        "polygon": [
            (36.5, 36.0),   # SW corner
            (38.5, 36.0),   # NW corner
            (38.5, 38.5),   # NE corner
            (36.5, 38.5),   # SE corner
        ],
        "expected_m7_rate": 0.01,
    },
    "campi_flegrei": {
        "name": "Campi Flegrei Caldera (Italy)",
        # Active volcanic caldera near Naples
        "polygon": [
            (40.75, 13.95),  # SW corner
            (40.90, 13.95),  # NW corner
            (40.90, 14.25),  # NE corner
            (40.75, 14.25),  # SE corner
        ],
        "expected_m7_rate": 0.001,  # Volcanic, different risk model
    },
}


def validate_all_regions(stations: List[Station], 
                          output_path: Optional[Path] = None) -> Dict[str, RegionEligibility]:
    """
    Validate all predefined fault polygons.
    
    Returns dict of region_id -> eligibility result.
    """
    results = {}
    
    for region_id, config in FAULT_POLYGONS.items():
        validator = RegionValidator(
            region_id=region_id,
            polygon=config['polygon']
        )
        validator.add_stations(stations)
        eligibility = validator.validate()
        results[region_id] = eligibility
        
        # Print summary
        status = "✓ ELIGIBLE" if eligibility.eligible else "✗ INELIGIBLE"
        print(f"\n{region_id}: {status}")
        print(f"  Stations: {eligibility.n_stations} ({eligibility.n_stations_with_history} with history)")
        print(f"  Triangles: {eligibility.n_quality_triangles}/{eligibility.n_triangles} quality")
        print(f"  Spacing: max {eligibility.max_station_spacing_km:.1f}km, mean {eligibility.mean_station_spacing_km:.1f}km")
        
        for reason in eligibility.reasons:
            print(f"  ✗ {reason}")
        for warning in eligibility.warnings:
            print(f"  ⚠ {warning}")
    
    # Save results
    if output_path:
        output_data = {rid: e.to_dict() for rid, e in results.items()}
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    return results
