#!/usr/bin/env python3
"""
gps_to_strain.py
Convert scattered GPS velocities to strain tensor field.

THE CRITICAL MISSING LINK - This module converts discrete GPS station
velocity measurements into a continuous strain-rate tensor field using
Delaunay Triangulation.

Mathematical Foundation:
-----------------------
1. Triangulate GPS stations using Delaunay algorithm
2. Within each triangle, assume linear velocity field:
   v(x,y) = [a*x + b*y + c, d*x + e*y + f]
3. The velocity gradient is constant per triangle:
   grad(v) = [[a, b], [d, e]]
4. The strain-rate tensor is the symmetric part:
   E = 0.5 * (grad(v) + grad(v)^T)
5. Rasterize triangles onto regular grid

This approach is EXACT for the linear velocity assumption within
each triangle - no fitting or approximation errors!

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json


@dataclass
class StrainTensorField:
    """Container for computed strain tensor field."""
    # Grid coordinates
    lats: np.ndarray          # 1D latitude grid
    lons: np.ndarray          # 1D longitude grid
    times: np.ndarray         # Time points (datetime or float)
    
    # Strain tensors: shape (n_times, n_lat, n_lon, 3, 3)
    # But for 2D horizontal strain, we use (n_times, n_lat, n_lon, 2, 2)
    # Extended to 3D by adding zeros for vertical
    strain_tensors: np.ndarray
    
    # Velocity field (for diagnostics)
    velocity_n: np.ndarray    # North velocity (n_times, n_lat, n_lon)
    velocity_e: np.ndarray    # East velocity
    
    # Quality metrics
    triangle_count: np.ndarray  # Number of triangles at each grid point
    
    # Metadata
    station_codes: List[str]
    station_lats: np.ndarray
    station_lons: np.ndarray
    earthquake_info: dict


class GPSToStrainConverter:
    """
    Convert GPS velocity field to strain tensor field.
    
    Uses Delaunay Triangulation:
    
    GPS Stations          Delaunay Triangles         Strain Grid
        *                     /\ /\ /\               +---------+
      * * *      -->       /\ /\ /\ /\     -->       |E(lat,lon)|
        * *                  /\ /\ /\                +---------+
    
    Within each triangle:
    - Velocity is linear: v = A*x + b (A is 2x2, b is 2x1)
    - Strain rate E = 0.5*(A + A^T) is CONSTANT
    
    This is the EXACT solution for piecewise-linear velocity fields.
    """
    
    def __init__(self,
                 grid_resolution: float = 0.1,  # degrees
                 temporal_smoothing_window: int = 7,  # days
                 min_triangle_quality: float = 0.1,
                 spatial_smoothing: bool = True):
        """
        Initialize the converter.
        
        Args:
            grid_resolution: Output grid spacing in degrees
            temporal_smoothing_window: Savitzky-Golay window for velocities
            min_triangle_quality: Minimum triangle quality (0-1, filters skinny triangles)
            spatial_smoothing: Apply spatial smoothing to final field
        """
        self.grid_resolution = grid_resolution
        self.temporal_window = temporal_smoothing_window
        self.min_quality = min_triangle_quality
        self.spatial_smoothing = spatial_smoothing
    
    # =========================================================================
    # DELAUNAY TRIANGULATION
    # =========================================================================
    
    def triangulate_stations(self,
                            station_lats: np.ndarray,
                            station_lons: np.ndarray) -> Tuple[Delaunay, np.ndarray]:
        """
        Create Delaunay triangulation of GPS stations.
        
        Args:
            station_lats: Station latitudes
            station_lons: Station longitudes
            
        Returns:
            tri: Delaunay triangulation object
            quality: Quality metric for each triangle (0-1)
        """
        # Stack coordinates for triangulation
        points = np.column_stack([station_lons, station_lats])
        
        # Create triangulation
        tri = Delaunay(points)
        
        # Compute triangle quality (ratio of inscribed to circumscribed circle)
        # Higher = more equilateral = better
        quality = self._compute_triangle_quality(points, tri.simplices)
        
        n_triangles = len(tri.simplices)
        n_good = np.sum(quality >= self.min_quality)
        print(f"  Triangulation: {n_triangles} triangles, {n_good} good quality")
        
        return tri, quality
    
    def _compute_triangle_quality(self,
                                   points: np.ndarray,
                                   simplices: np.ndarray) -> np.ndarray:
        """
        Compute quality metric for each triangle.
        
        Quality = 2 * r_inscribed / r_circumscribed
        Perfect equilateral triangle has quality = 1
        Degenerate (collinear) has quality = 0
        """
        quality = np.zeros(len(simplices))
        
        for i, simplex in enumerate(simplices):
            p0, p1, p2 = points[simplex]
            
            # Edge lengths
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)
            
            # Semi-perimeter
            s = (a + b + c) / 2
            
            # Area using Heron's formula
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0:
                quality[i] = 0
                continue
            
            area = np.sqrt(area_sq)
            
            # Inscribed circle radius
            r_in = area / s
            
            # Circumscribed circle radius
            r_circ = (a * b * c) / (4 * area)
            
            # Quality ratio
            quality[i] = 2 * r_in / r_circ if r_circ > 0 else 0
        
        return quality
    
    # =========================================================================
    # VELOCITY GRADIENT COMPUTATION
    # =========================================================================
    
    def compute_velocity_gradient(self,
                                   tri: Delaunay,
                                   velocities_n: np.ndarray,
                                   velocities_e: np.ndarray) -> np.ndarray:
        """
        Compute velocity gradient for each triangle.
        
        Within a triangle with vertices (x0,y0), (x1,y1), (x2,y2)
        and velocities (v0,v1,v2), the linear velocity field is:
        
        v(x,y) = v0 + (x-x0)*dvdx + (y-y0)*dvdy
        
        Solve: [v1-v0, v2-v0] = [[x1-x0, y1-y0], [x2-x0, y2-y0]] @ [dvdx, dvdy]
        
        Args:
            tri: Delaunay triangulation
            velocities_n: North velocity at each station
            velocities_e: East velocity at each station
            
        Returns:
            grad_v: Velocity gradient for each triangle, shape (n_triangles, 2, 2)
                    grad_v[i, j, k] = dv_j/dx_k where j=0,1 (N,E) and k=0,1 (lon, lat)
        """
        n_triangles = len(tri.simplices)
        grad_v = np.zeros((n_triangles, 2, 2))
        
        points = tri.points
        
        for i, simplex in enumerate(tri.simplices):
            i0, i1, i2 = simplex
            
            # Coordinates (lon, lat)
            x0, y0 = points[i0]
            x1, y1 = points[i1]
            x2, y2 = points[i2]
            
            # Velocities (N, E)
            vn0, vn1, vn2 = velocities_n[i0], velocities_n[i1], velocities_n[i2]
            ve0, ve1, ve2 = velocities_e[i0], velocities_e[i1], velocities_e[i2]
            
            # Coordinate differences
            dx1, dy1 = x1 - x0, y1 - y0
            dx2, dy2 = x2 - x0, y2 - y0
            
            # Matrix for solving: M @ [dvdx, dvdy]^T = [dv]
            M = np.array([[dx1, dy1],
                          [dx2, dy2]])
            
            # Velocity differences
            dvn = np.array([vn1 - vn0, vn2 - vn0])
            dve = np.array([ve1 - ve0, ve2 - ve0])
            
            # Solve for gradients
            det = dx1 * dy2 - dx2 * dy1
            if np.abs(det) > 1e-10:
                # [dvn/dlon, dvn/dlat]
                grad_vn = np.linalg.solve(M, dvn)
                # [dve/dlon, dve/dlat]
                grad_ve = np.linalg.solve(M, dve)
                
                # Store: grad_v[i] = [[dvn/dlon, dvn/dlat], [dve/dlon, dve/dlat]]
                grad_v[i, 0, :] = grad_vn
                grad_v[i, 1, :] = grad_ve
        
        return grad_v
    
    def gradient_to_strain(self, grad_v: np.ndarray) -> np.ndarray:
        """
        Convert velocity gradient to strain-rate tensor.
        
        E = 0.5 * (grad_v + grad_v^T)
        
        For 2D:
        E = [[E_lon_lon, E_lon_lat],
             [E_lat_lon, E_lat_lat]]
           = [[dvn/dlon, 0.5*(dvn/dlat + dve/dlon)],
              [0.5*(dvn/dlat + dve/dlon), dve/dlat]]
        
        Args:
            grad_v: Velocity gradients, shape (n_triangles, 2, 2)
            
        Returns:
            strain: Strain-rate tensors, shape (n_triangles, 2, 2)
        """
        # Symmetric part: E = 0.5 * (L + L^T)
        strain = 0.5 * (grad_v + np.swapaxes(grad_v, -1, -2))
        return strain
    
    # =========================================================================
    # GRID RASTERIZATION
    # =========================================================================
    
    def create_grid(self,
                    station_lats: np.ndarray,
                    station_lons: np.ndarray,
                    padding: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create regular lat/lon grid covering the station network.
        
        Args:
            station_lats: Station latitudes
            station_lons: Station longitudes
            padding: Padding around stations in degrees
            
        Returns:
            lats: 1D latitude array
            lons: 1D longitude array
        """
        lat_min = np.min(station_lats) - padding
        lat_max = np.max(station_lats) + padding
        lon_min = np.min(station_lons) - padding
        lon_max = np.max(station_lons) + padding
        
        lats = np.arange(lat_min, lat_max, self.grid_resolution)
        lons = np.arange(lon_min, lon_max, self.grid_resolution)
        
        return lats, lons
    
    def rasterize_triangles(self,
                            tri: Delaunay,
                            strain_triangles: np.ndarray,
                            quality: np.ndarray,
                            lats: np.ndarray,
                            lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rasterize triangle strain values onto regular grid.
        
        For each grid point, find which triangle contains it
        and assign that triangle's strain value.
        
        Args:
            tri: Delaunay triangulation
            strain_triangles: Strain tensor per triangle, shape (n_tri, 2, 2)
            quality: Triangle quality scores
            lats: Output latitude grid
            lons: Output longitude grid
            
        Returns:
            strain_grid: Strain field, shape (n_lat, n_lon, 2, 2)
            triangle_count: Number of valid triangles at each point
        """
        n_lat, n_lon = len(lats), len(lons)
        strain_grid = np.zeros((n_lat, n_lon, 2, 2))
        triangle_count = np.zeros((n_lat, n_lon))
        
        # Create meshgrid of query points
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        query_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        
        # Find which triangle contains each grid point
        triangle_ids = tri.find_simplex(query_points)
        triangle_ids = triangle_ids.reshape(n_lat, n_lon)
        
        # Assign strain values
        for i in range(n_lat):
            for j in range(n_lon):
                tid = triangle_ids[i, j]
                if tid >= 0 and quality[tid] >= self.min_quality:
                    strain_grid[i, j] = strain_triangles[tid]
                    triangle_count[i, j] = 1
        
        # Optional spatial smoothing to fill gaps
        if self.spatial_smoothing:
            strain_grid = self._smooth_strain_field(strain_grid, triangle_count)
        
        return strain_grid, triangle_count
    
    def _smooth_strain_field(self,
                              strain_grid: np.ndarray,
                              valid_mask: np.ndarray,
                              sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to strain field."""
        from scipy.ndimage import gaussian_filter
        
        # Smooth each component
        for i in range(2):
            for j in range(2):
                component = strain_grid[:, :, i, j].copy()
                # Only smooth where we have data
                component_smoothed = gaussian_filter(component, sigma)
                valid_smoothed = gaussian_filter(valid_mask.astype(float), sigma)
                
                # Normalize by valid count
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = component_smoothed / valid_smoothed
                    normalized[~np.isfinite(normalized)] = 0
                
                strain_grid[:, :, i, j] = normalized
        
        return strain_grid
    
    # =========================================================================
    # EXTEND TO 3D
    # =========================================================================
    
    def extend_to_3d(self, strain_2d: np.ndarray) -> np.ndarray:
        """
        Extend 2D horizontal strain to 3D strain tensor.
        
        Assumes no vertical strain (plane stress approximation).
        
        Args:
            strain_2d: Shape (..., 2, 2) horizontal strain
            
        Returns:
            strain_3d: Shape (..., 3, 3) with vertical components zero
        """
        shape_3d = strain_2d.shape[:-2] + (3, 3)
        strain_3d = np.zeros(shape_3d)
        
        # Copy 2D components to upper-left 2x2 block
        # Mapping: 0->N->0, 1->E->1, vertical->2
        strain_3d[..., 0, 0] = strain_2d[..., 0, 0]  # E_NN
        strain_3d[..., 0, 1] = strain_2d[..., 0, 1]  # E_NE
        strain_3d[..., 1, 0] = strain_2d[..., 1, 0]  # E_EN
        strain_3d[..., 1, 1] = strain_2d[..., 1, 1]  # E_EE
        
        # For incompressibility, could set E_33 = -(E_11 + E_22)
        # But for now, leave as zero (plane stress)
        
        return strain_3d
    
    # =========================================================================
    # MAIN CONVERSION PIPELINE
    # =========================================================================
    
    def convert(self,
                station_data: Dict[str, pd.DataFrame],
                station_locations: Dict[str, Tuple[float, float]],
                earthquake_info: dict) -> StrainTensorField:
        """
        Convert GPS station data to strain tensor field.
        
        Main entry point for the conversion pipeline.
        
        Args:
            station_data: Dict of station_code -> DataFrame with columns:
                          datetime, n, e, u, vn, ve, vu
            station_locations: Dict of station_code -> (lat, lon)
            earthquake_info: Dictionary with earthquake metadata
            
        Returns:
            StrainTensorField with computed strain tensors
        """
        print(f"\nConverting GPS velocities to strain tensor field...")
        
        # Extract station coordinates
        codes = list(station_data.keys())
        n_stations = len(codes)
        print(f"  Stations: {n_stations}")
        
        station_lats = np.array([station_locations[c][0] for c in codes])
        station_lons = np.array([station_locations[c][1] for c in codes])
        
        # Get common time grid
        # Find overlapping time range
        all_times = []
        for code, df in station_data.items():
            all_times.extend(df['datetime'].tolist())
        
        unique_times = sorted(set(all_times))
        print(f"  Time points: {len(unique_times)}")
        
        # Triangulate stations
        tri, quality = self.triangulate_stations(station_lats, station_lons)
        
        # Create output grid
        lats, lons = self.create_grid(station_lats, station_lons)
        n_lat, n_lon = len(lats), len(lons)
        print(f"  Output grid: {n_lat} x {n_lon}")
        
        # Initialize output arrays
        n_times = len(unique_times)
        strain_3d = np.zeros((n_times, n_lat, n_lon, 3, 3))
        velocity_n = np.zeros((n_times, n_lat, n_lon))
        velocity_e = np.zeros((n_times, n_lat, n_lon))
        triangle_count = np.zeros((n_times, n_lat, n_lon))
        
        # Process each time step
        print("  Computing strain field...")
        for t_idx, time_val in enumerate(unique_times):
            # Get velocities at this time (interpolate if needed)
            vn = np.zeros(n_stations)
            ve = np.zeros(n_stations)
            valid = np.zeros(n_stations, dtype=bool)
            
            for s_idx, code in enumerate(codes):
                df = station_data[code]
                
                # Find closest time
                time_diffs = np.abs((df['datetime'] - time_val).dt.total_seconds())
                if len(time_diffs) == 0:
                    continue
                    
                closest_idx = time_diffs.argmin()
                if time_diffs.iloc[closest_idx] < 86400 * 2:  # Within 2 days
                    if 'vn' in df.columns and 've' in df.columns:
                        vn[s_idx] = df['vn'].iloc[closest_idx]
                        ve[s_idx] = df['ve'].iloc[closest_idx]
                        valid[s_idx] = True
            
            if np.sum(valid) < 3:
                continue
            
            # Compute velocity gradient for each triangle
            grad_v = self.compute_velocity_gradient(tri, vn, ve)
            
            # Convert to strain
            strain_2d = self.gradient_to_strain(grad_v)
            
            # Rasterize onto grid
            strain_grid, tri_count = self.rasterize_triangles(
                tri, strain_2d, quality, lats, lons
            )
            
            # Extend to 3D
            strain_3d[t_idx] = self.extend_to_3d(strain_grid)
            triangle_count[t_idx] = tri_count
            
            # Also interpolate velocities for diagnostics
            valid_lats = station_lats[valid]
            valid_lons = station_lons[valid]
            if len(valid_lats) >= 3:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                
                try:
                    velocity_n[t_idx] = griddata(
                        (valid_lons, valid_lats), vn[valid],
                        (lon_grid, lat_grid), method='linear', fill_value=0
                    )
                    velocity_e[t_idx] = griddata(
                        (valid_lons, valid_lats), ve[valid],
                        (lon_grid, lat_grid), method='linear', fill_value=0
                    )
                except Exception:
                    pass
        
        print(f"  Strain field computed: {strain_3d.shape}")
        
        # Create result object
        result = StrainTensorField(
            lats=lats,
            lons=lons,
            times=np.array(unique_times),
            strain_tensors=strain_3d,
            velocity_n=velocity_n,
            velocity_e=velocity_e,
            triangle_count=triangle_count,
            station_codes=codes,
            station_lats=station_lats,
            station_lons=station_lons,
            earthquake_info=earthquake_info
        )
        
        return result
    
    def save_strain_field(self, 
                          field: StrainTensorField, 
                          output_file: Path):
        """Save strain field to NPZ file."""
        np.savez(
            output_file,
            lats=field.lats,
            lons=field.lons,
            times=np.array([t.isoformat() if hasattr(t, 'isoformat') else str(t) 
                           for t in field.times]),
            strain_tensors=field.strain_tensors,
            velocity_n=field.velocity_n,
            velocity_e=field.velocity_e,
            triangle_count=field.triangle_count,
            station_codes=field.station_codes,
            station_lats=field.station_lats,
            station_lons=field.station_lons,
            earthquake_info=json.dumps(field.earthquake_info)
        )
        print(f"  Saved strain field to {output_file}")
    
    def load_strain_field(self, input_file: Path) -> StrainTensorField:
        """Load strain field from NPZ file."""
        data = np.load(input_file, allow_pickle=True)
        
        return StrainTensorField(
            lats=data['lats'],
            lons=data['lons'],
            times=data['times'],
            strain_tensors=data['strain_tensors'],
            velocity_n=data['velocity_n'],
            velocity_e=data['velocity_e'],
            triangle_count=data['triangle_count'],
            station_codes=list(data['station_codes']),
            station_lats=data['station_lats'],
            station_lons=data['station_lons'],
            earthquake_info=json.loads(str(data['earthquake_info']))
        )


def convert_strain_field_to_stations(field: StrainTensorField) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert grid-based strain field to station-based format for lambda_geo.
    
    This reshapes (n_times, n_lat, n_lon, 3, 3) -> (n_times, n_points, 3, 3)
    where n_points = n_lat * n_lon.
    
    Returns:
        strain_tensors: Shape (n_times, n_points, 3, 3)
        station_lats: Shape (n_points,)
        station_lons: Shape (n_points,)
        times: Time array
    """
    n_times = len(field.times)
    n_lat, n_lon = len(field.lats), len(field.lons)
    n_points = n_lat * n_lon
    
    # Reshape strain tensors
    strain_tensors = field.strain_tensors.reshape(n_times, n_points, 3, 3)
    
    # Create coordinate arrays
    lon_grid, lat_grid = np.meshgrid(field.lons, field.lats)
    station_lats = lat_grid.ravel()
    station_lons = lon_grid.ravel()
    
    return strain_tensors, station_lats, station_lons, field.times


def main():
    """Test GPS to strain conversion."""
    print("GPS to Strain Conversion Module")
    print("================================")
    print("\nThis module converts GPS velocity data to strain tensor fields")
    print("using Delaunay triangulation.")
    print("\nUsage:")
    print("  from gps_to_strain import GPSToStrainConverter")
    print("  converter = GPSToStrainConverter(grid_resolution=0.1)")
    print("  field = converter.convert(station_data, station_locations, eq_info)")


if __name__ == "__main__":
    main()
