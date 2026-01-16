#!/usr/bin/env python3
"""
gps_to_strain.py
================
Convert GPS velocity field to strain-rate tensor field using
Delaunay Triangulation Strain Calculation.

This is THE CRITICAL MODULE that bridges raw GPS data to Lambda_geo.

Mathematical Foundation:
------------------------
1. Triangulate: Connect nearby GPS stations to form triangles
2. Linear Basis: Assume velocity v(x) varies linearly inside each triangle
3. Gradient: Compute nabla(v) - constant per triangle
4. Strain Rate: E = 0.5 * (nabla(v) + nabla(v)^T)
5. Rasterize: Map triangle values to a regular grid

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import warnings


@dataclass
class StrainField:
    """Container for computed strain-rate tensor field."""
    times: np.ndarray                    # Time points
    grid_lats: np.ndarray                # Grid latitude coordinates
    grid_lons: np.ndarray                # Grid longitude coordinates
    strain_tensors: np.ndarray           # Shape: (n_times, n_grid, 3, 3)
    triangulation: Optional[Delaunay]    # Delaunay triangulation used
    station_lats: np.ndarray             # Original station latitudes
    station_lons: np.ndarray             # Original station longitudes
    quality_mask: np.ndarray             # Grid points with good data


class GPSToStrainConverter:
    """
    Convert GPS displacement/velocity time series to strain-rate tensor field.
    
    The key insight from Navier-Stokes:
    - The velocity gradient tensor nabla(u) captures local deformation
    - For tectonics, GPS velocities give us v(x,t) at sparse points
    - Delaunay triangulation lets us compute nabla(v) robustly
    - The symmetric part is the strain-rate tensor E
    
    Method:
    -------
    1. For each triangle formed by 3 GPS stations:
       - Fit linear velocity field: v(x) = a*x + b*y + c
       - The gradient is constant: nabla(v) = [[dv_e/dx, dv_e/dy], [dv_n/dx, dv_n/dy]]
       
    2. Strain rate tensor components:
       - E_xx = dv_e/dx           (East-East strain rate)
       - E_yy = dv_n/dy           (North-North strain rate)  
       - E_xy = 0.5*(dv_e/dy + dv_n/dx)  (Shear strain rate)
       
    3. Convert to 3x3 tensor (assuming plane strain in horizontal):
       - E_zz computed from trace-free condition if desired
       - E_xz = E_yz = 0 (horizontal measurements only)
    """
    
    def __init__(self,
                 grid_resolution: int = 50,
                 min_triangle_quality: float = 0.1,
                 temporal_filter: str = 'savgol',
                 temporal_window: int = 7,
                 spatial_smoothing: float = 0.0):
        """
        Initialize the converter.
        
        Args:
            grid_resolution: Number of grid points in each direction
            min_triangle_quality: Minimum aspect ratio for triangles (0-1)
            temporal_filter: 'savgol', 'kalman', 'median', or 'none'
            temporal_window: Window size for temporal filtering (days)
            spatial_smoothing: Gaussian smoothing sigma (grid cells)
        """
        self.grid_resolution = grid_resolution
        self.min_triangle_quality = min_triangle_quality
        self.temporal_filter = temporal_filter
        self.temporal_window = temporal_window
        self.spatial_smoothing = spatial_smoothing
    
    # =========================================================================
    # TEMPORAL FILTERING (Critical for real GPS data!)
    # =========================================================================
    
    def filter_timeseries(self, 
                          data: np.ndarray,
                          sigma: np.ndarray = None) -> np.ndarray:
        """
        Apply temporal filter to reduce GPS noise.
        
        GPS daily solutions have ~2-5mm horizontal noise.
        Raw differentiation amplifies this noise catastrophically.
        Filtering is ESSENTIAL for real data.
        
        Args:
            data: Time series, shape (n_times, n_stations) or (n_times,)
            sigma: Uncertainties (used for weighted filtering)
            
        Returns:
            Filtered data, same shape as input
        """
        if self.temporal_filter == 'none':
            return data
        
        # Ensure 2D
        was_1d = data.ndim == 1
        if was_1d:
            data = data[:, np.newaxis]
        
        n_times, n_stations = data.shape
        filtered = np.zeros_like(data)
        
        for s in range(n_stations):
            ts = data[:, s]
            valid = ~np.isnan(ts)
            
            if np.sum(valid) < self.temporal_window:
                filtered[:, s] = ts
                continue
            
            if self.temporal_filter == 'savgol':
                # Savitzky-Golay filter - preserves edges and trends
                window = min(self.temporal_window, np.sum(valid))
                if window % 2 == 0:
                    window -= 1
                window = max(window, 3)
                
                # Interpolate NaN gaps
                ts_interp = np.interp(
                    np.arange(n_times),
                    np.where(valid)[0],
                    ts[valid]
                )
                
                filtered[:, s] = savgol_filter(ts_interp, window, 2)
                
            elif self.temporal_filter == 'median':
                # Median filter - robust to outliers
                from scipy.ndimage import median_filter as mf
                ts_interp = np.interp(
                    np.arange(n_times),
                    np.where(valid)[0],
                    ts[valid]
                )
                filtered[:, s] = mf(ts_interp, size=self.temporal_window)
                
            elif self.temporal_filter == 'gaussian':
                # Gaussian smoothing
                ts_interp = np.interp(
                    np.arange(n_times),
                    np.where(valid)[0],
                    ts[valid]
                )
                filtered[:, s] = gaussian_filter(ts_interp, self.temporal_window / 2)
                
            else:
                filtered[:, s] = ts
        
        if was_1d:
            filtered = filtered[:, 0]
        
        return filtered
    
    def compute_velocities(self,
                            displacements: np.ndarray,
                            dt_days: float = 1.0) -> np.ndarray:
        """
        Compute velocities from filtered displacements.
        
        Uses Savitzky-Golay derivative for stability.
        
        Args:
            displacements: Filtered displacement time series (mm)
            dt_days: Time step in days
            
        Returns:
            Velocities in mm/day
        """
        n_times = displacements.shape[0]
        
        # Ensure 2D
        was_1d = displacements.ndim == 1
        if was_1d:
            displacements = displacements[:, np.newaxis]
        
        n_stations = displacements.shape[1]
        velocities = np.zeros_like(displacements)
        
        for s in range(n_stations):
            ts = displacements[:, s]
            valid = ~np.isnan(ts)
            
            if np.sum(valid) < 5:
                velocities[:, s] = np.nan
                continue
            
            # Use Savitzky-Golay for derivative (more stable than finite diff)
            window = min(7, np.sum(valid))
            if window % 2 == 0:
                window -= 1
            window = max(window, 3)
            
            ts_interp = np.interp(
                np.arange(n_times),
                np.where(valid)[0],
                ts[valid]
            )
            
            try:
                velocities[:, s] = savgol_filter(
                    ts_interp, window, 2, deriv=1, delta=dt_days
                )
            except:
                # Fallback to central difference
                velocities[1:-1, s] = (ts_interp[2:] - ts_interp[:-2]) / (2 * dt_days)
                velocities[0, s] = velocities[1, s]
                velocities[-1, s] = velocities[-2, s]
        
        if was_1d:
            velocities = velocities[:, 0]
        
        return velocities
    
    # =========================================================================
    # DELAUNAY TRIANGULATION STRAIN CALCULATION
    # =========================================================================
    
    def create_triangulation(self,
                              lats: np.ndarray,
                              lons: np.ndarray) -> Tuple[Delaunay, np.ndarray]:
        """
        Create Delaunay triangulation of station network.
        
        Args:
            lats: Station latitudes
            lons: Station longitudes
            
        Returns:
            tri: Delaunay triangulation object
            quality: Quality metric for each triangle (0-1)
        """
        # Convert to local Cartesian (km)
        lat0, lon0 = np.mean(lats), np.mean(lons)
        x_km = (lons - lon0) * 111.0 * np.cos(np.radians(lat0))
        y_km = (lats - lat0) * 111.0
        
        points = np.column_stack([x_km, y_km])
        
        # Create triangulation
        tri = Delaunay(points)
        
        # Compute quality metric for each triangle
        # Quality = 4 * sqrt(3) * area / (a^2 + b^2 + c^2)
        # where a, b, c are side lengths
        # Quality = 1 for equilateral, -> 0 for degenerate
        
        n_triangles = len(tri.simplices)
        quality = np.zeros(n_triangles)
        
        for i, simplex in enumerate(tri.simplices):
            p0, p1, p2 = points[simplex]
            
            # Side lengths
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p0 - p2)
            
            # Area via cross product
            area = 0.5 * abs(np.cross(p1 - p0, p2 - p0))
            
            # Quality metric
            denom = a**2 + b**2 + c**2
            if denom > 0:
                quality[i] = 4 * np.sqrt(3) * area / denom
            else:
                quality[i] = 0
        
        return tri, quality
    
    def compute_strain_per_triangle(self,
                                     v_east: np.ndarray,
                                     v_north: np.ndarray,
                                     lats: np.ndarray,
                                     lons: np.ndarray,
                                     tri: Delaunay) -> np.ndarray:
        """
        Compute strain-rate tensor for each triangle.
        
        For a triangle with vertices (x0,y0), (x1,y1), (x2,y2) and
        velocities (ve0,vn0), (ve1,vn1), (ve2,vn2):
        
        Linear velocity field: v(x,y) = a*x + b*y + c
        
        Solve: [x0 y0 1] [a]   [ve0]
               [x1 y1 1] [b] = [ve1]
               [x2 y2 1] [c]   [ve2]
               
        Then:
        dve/dx = a, dve/dy = b
        dvn/dx = d, dvn/dy = e (similar system for north)
        
        Strain rate tensor:
        E_xx = dve/dx
        E_yy = dvn/dy
        E_xy = 0.5 * (dve/dy + dvn/dx)
        
        Args:
            v_east: East velocities at stations (mm/day)
            v_north: North velocities at stations (mm/day)
            lats: Station latitudes
            lons: Station longitudes
            tri: Delaunay triangulation
            
        Returns:
            strain: Shape (n_triangles, 3, 3) strain-rate tensors
        """
        # Convert to local Cartesian (km)
        lat0, lon0 = np.mean(lats), np.mean(lons)
        x_km = (lons - lon0) * 111.0 * np.cos(np.radians(lat0))
        y_km = (lats - lat0) * 111.0
        
        n_triangles = len(tri.simplices)
        strain = np.zeros((n_triangles, 3, 3))
        
        for i, simplex in enumerate(tri.simplices):
            i0, i1, i2 = simplex
            
            # Coordinates (km)
            x = np.array([x_km[i0], x_km[i1], x_km[i2]])
            y = np.array([y_km[i0], y_km[i1], y_km[i2]])
            
            # Velocities (mm/day)
            ve = np.array([v_east[i0], v_east[i1], v_east[i2]])
            vn = np.array([v_north[i0], v_north[i1], v_north[i2]])
            
            # Check for NaN
            if np.any(np.isnan(ve)) or np.any(np.isnan(vn)):
                strain[i] = np.nan
                continue
            
            # Design matrix
            A = np.column_stack([x, y, np.ones(3)])
            
            # Solve for velocity gradient coefficients
            try:
                # East velocity: ve = a*x + b*y + c
                coef_e = np.linalg.solve(A, ve)
                a, b, c = coef_e
                
                # North velocity: vn = d*x + e*y + f
                coef_n = np.linalg.solve(A, vn)
                d, e, f = coef_n
                
            except np.linalg.LinAlgError:
                strain[i] = np.nan
                continue
            
            # Velocity gradient tensor (1/day, since velocity is mm/day and x is km)
            # Need to convert: (mm/day) / km = 1e-6 / day = ~1e-11 / s (nanostrain/s)
            scale = 1e-6  # mm/km = microstrain
            
            dve_dx = a * scale
            dve_dy = b * scale
            dvn_dx = d * scale
            dvn_dy = e * scale
            
            # Strain rate tensor (symmetric part of velocity gradient)
            # 2D horizontal strain, embedded in 3D
            E_xx = dve_dx
            E_yy = dvn_dy
            E_xy = 0.5 * (dve_dy + dvn_dx)
            
            # Assume incompressible (trace = 0) to get E_zz
            # Or set E_zz = 0 for plane strain
            E_zz = -(E_xx + E_yy)  # Incompressible
            
            strain[i] = np.array([
                [E_xx, E_xy, 0],
                [E_xy, E_yy, 0],
                [0,    0,    E_zz]
            ])
        
        return strain
    
    def rasterize_strain_to_grid(self,
                                  triangle_strain: np.ndarray,
                                  tri: Delaunay,
                                  triangle_quality: np.ndarray,
                                  lats: np.ndarray,
                                  lons: np.ndarray,
                                  grid_lats: np.ndarray,
                                  grid_lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate triangle strain values to a regular grid.
        
        Uses barycentric interpolation within triangles,
        weighted by triangle quality.
        
        Args:
            triangle_strain: Strain per triangle, shape (n_tri, 3, 3)
            tri: Delaunay triangulation
            triangle_quality: Quality metric per triangle
            lats, lons: Station coordinates
            grid_lats, grid_lons: 1D arrays defining grid
            
        Returns:
            grid_strain: Shape (n_grid_y, n_grid_x, 3, 3)
            quality_mask: Boolean mask of valid grid points
        """
        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)
        
        # Convert to local Cartesian
        lat0, lon0 = np.mean(lats), np.mean(lons)
        x_grid = (lon_mesh - lon0) * 111.0 * np.cos(np.radians(lat0))
        y_grid = (lat_mesh - lat0) * 111.0
        
        x_stations = (lons - lon0) * 111.0 * np.cos(np.radians(lat0))
        y_stations = (lats - lat0) * 111.0
        
        # Find which triangle each grid point belongs to
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        triangle_indices = tri.find_simplex(grid_points)
        
        # Initialize output
        n_y, n_x = lon_mesh.shape
        grid_strain = np.full((n_y, n_x, 3, 3), np.nan)
        quality_mask = np.zeros((n_y, n_x), dtype=bool)
        
        # Assign strain values
        for idx, (gx, gy) in enumerate(grid_points):
            tri_idx = triangle_indices[idx]
            
            if tri_idx < 0:  # Outside convex hull
                continue
            
            if triangle_quality[tri_idx] < self.min_triangle_quality:
                continue
            
            if np.any(np.isnan(triangle_strain[tri_idx])):
                continue
            
            # Get grid indices
            iy = idx // n_x
            ix = idx % n_x
            
            grid_strain[iy, ix] = triangle_strain[tri_idx]
            quality_mask[iy, ix] = True
        
        # Optional spatial smoothing
        if self.spatial_smoothing > 0:
            for i in range(3):
                for j in range(3):
                    component = grid_strain[:, :, i, j].copy()
                    valid = ~np.isnan(component)
                    if np.any(valid):
                        # Fill NaN with nearest neighbor before smoothing
                        from scipy.ndimage import distance_transform_edt
                        indices = distance_transform_edt(
                            np.isnan(component),
                            return_distances=False,
                            return_indices=True
                        )
                        component = component[tuple(indices)]
                        component = gaussian_filter(component, self.spatial_smoothing)
                        grid_strain[:, :, i, j] = np.where(quality_mask, component, np.nan)
        
        return grid_strain, quality_mask
    
    # =========================================================================
    # MAIN CONVERSION PIPELINE
    # =========================================================================
    
    def convert(self,
                times: np.ndarray,
                station_lats: np.ndarray,
                station_lons: np.ndarray,
                east_mm: np.ndarray,
                north_mm: np.ndarray,
                up_mm: np.ndarray = None) -> StrainField:
        """
        Convert GPS time series to strain-rate tensor field.
        
        This is the main entry point.
        
        Args:
            times: Time points (datetime strings or array)
            station_lats: Station latitudes, shape (n_stations,)
            station_lons: Station longitudes, shape (n_stations,)
            east_mm: East displacements, shape (n_times, n_stations)
            north_mm: North displacements, shape (n_times, n_stations)
            up_mm: Vertical displacements (optional)
            
        Returns:
            StrainField with computed strain tensors
        """
        n_times, n_stations = east_mm.shape
        print(f"Converting GPS to strain field...")
        print(f"  Stations: {n_stations}")
        print(f"  Time points: {n_times}")
        
        # Step 1: Temporal filtering
        print("  Applying temporal filter...")
        east_filtered = self.filter_timeseries(east_mm)
        north_filtered = self.filter_timeseries(north_mm)
        
        # Step 2: Compute velocities
        print("  Computing velocities...")
        v_east = self.compute_velocities(east_filtered)  # mm/day
        v_north = self.compute_velocities(north_filtered)
        
        # Step 3: Create triangulation (once)
        print("  Creating Delaunay triangulation...")
        tri, triangle_quality = self.create_triangulation(station_lats, station_lons)
        n_triangles = len(tri.simplices)
        print(f"    Created {n_triangles} triangles")
        good_triangles = np.sum(triangle_quality >= self.min_triangle_quality)
        print(f"    Good quality triangles: {good_triangles}")
        
        # Step 4: Define output grid
        lat_min, lat_max = np.min(station_lats), np.max(station_lats)
        lon_min, lon_max = np.min(station_lons), np.max(station_lons)
        
        # Add small buffer
        lat_buffer = 0.1 * (lat_max - lat_min)
        lon_buffer = 0.1 * (lon_max - lon_min)
        
        grid_lats = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, 
                                self.grid_resolution)
        grid_lons = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer,
                                self.grid_resolution)
        
        n_grid = self.grid_resolution ** 2
        
        # Step 5: Compute strain at each time step
        print("  Computing strain field for each time step...")
        strain_tensors = np.full((n_times, n_grid, 3, 3), np.nan)
        quality_mask_all = np.zeros((n_times, n_grid), dtype=bool)
        
        for t in range(n_times):
            if t % 50 == 0:
                print(f"    Time step {t}/{n_times}")
            
            # Strain per triangle at this time
            tri_strain = self.compute_strain_per_triangle(
                v_east[t], v_north[t],
                station_lats, station_lons, tri
            )
            
            # Rasterize to grid
            grid_strain, quality_mask = self.rasterize_strain_to_grid(
                tri_strain, tri, triangle_quality,
                station_lats, station_lons,
                grid_lats, grid_lons
            )
            
            # Flatten and store
            strain_tensors[t] = grid_strain.reshape(-1, 3, 3)
            quality_mask_all[t] = quality_mask.ravel()
        
        # Final quality mask: valid at > 50% of times
        final_quality = np.mean(quality_mask_all, axis=0) > 0.5
        
        print(f"  Grid points with good coverage: {np.sum(final_quality)}/{n_grid}")
        
        return StrainField(
            times=times,
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            strain_tensors=strain_tensors,
            triangulation=tri,
            station_lats=station_lats,
            station_lons=station_lons,
            quality_mask=final_quality
        )


# =============================================================================
# INTEGRATION WITH LAMBDA_GEO PIPELINE
# =============================================================================

def convert_gps_to_lambda_geo_format(
    gps_file: Path,
    output_file: Path,
    grid_resolution: int = 50,
    temporal_window: int = 7
) -> Path:
    """
    Convert GPS time series file to Lambda_geo input format.
    
    This bridges the data acquisition module to the Lambda_geo module.
    
    Args:
        gps_file: Path to GPS time series NPZ file
        output_file: Path to output strain tensor NPZ file
        grid_resolution: Grid resolution for strain field
        temporal_window: Days for temporal smoothing
        
    Returns:
        Path to output file
    """
    print("="*60)
    print("GPS TO LAMBDA_GEO FORMAT CONVERSION")
    print("="*60)
    
    # Load GPS data
    print(f"Loading: {gps_file}")
    data = np.load(gps_file, allow_pickle=True)
    
    times = data['times']
    lats = data['station_lats']
    lons = data['station_lons']
    east = data['east_mm']
    north = data['north_mm']
    eq_info = data['earthquake_info'].item() if 'earthquake_info' in data else {}
    
    print(f"  Stations: {len(lats)}")
    print(f"  Time points: {len(times)}")
    
    # Convert
    converter = GPSToStrainConverter(
        grid_resolution=grid_resolution,
        min_triangle_quality=0.1,
        temporal_filter='savgol',
        temporal_window=temporal_window,
        spatial_smoothing=1.0
    )
    
    strain_field = converter.convert(times, lats, lons, east, north)
    
    # Save in Lambda_geo format
    print(f"Saving to: {output_file}")
    
    np.savez(
        output_file,
        strain_tensors=strain_field.strain_tensors,
        times=strain_field.times,
        station_lats=strain_field.grid_lats,  # Grid coordinates
        station_lons=strain_field.grid_lons,
        earthquake_info=eq_info,
        quality_mask=strain_field.quality_mask,
        original_station_lats=strain_field.station_lats,
        original_station_lons=strain_field.station_lons
    )
    
    print("Conversion complete!")
    return output_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Demonstrate GPS to strain conversion."""
    
    # Test with synthetic data
    print("="*60)
    print("GPS TO STRAIN CONVERTER - DEMONSTRATION")
    print("="*60)
    
    # Generate synthetic GPS data
    np.random.seed(42)
    n_stations = 30
    n_times = 100
    
    # Station locations (grid-ish pattern)
    lats = np.random.uniform(35, 36, n_stations)
    lons = np.random.uniform(-118, -117, n_stations)
    
    # Synthetic displacements with linear trend + noise
    times = np.arange(n_times)
    east = np.zeros((n_times, n_stations))
    north = np.zeros((n_times, n_stations))
    
    for s in range(n_stations):
        # Linear velocity
        v_e = np.random.uniform(-5, 5)  # mm/day
        v_n = np.random.uniform(-5, 5)
        
        east[:, s] = v_e * times + np.random.normal(0, 2, n_times)  # 2mm noise
        north[:, s] = v_n * times + np.random.normal(0, 2, n_times)
    
    # Convert
    converter = GPSToStrainConverter(
        grid_resolution=20,
        temporal_filter='savgol',
        temporal_window=7
    )
    
    strain_field = converter.convert(times, lats, lons, east, north)
    
    print(f"\nOutput strain field:")
    print(f"  Shape: {strain_field.strain_tensors.shape}")
    print(f"  Grid: {len(strain_field.grid_lats)} x {len(strain_field.grid_lons)}")
    print(f"  Valid points: {np.sum(strain_field.quality_mask)}")
    
    # Check values
    valid_strain = strain_field.strain_tensors[:, strain_field.quality_mask, :]
    print(f"\nStrain statistics (valid points):")
    print(f"  E_xx range: [{np.nanmin(valid_strain[:,:,0,0]):.2e}, {np.nanmax(valid_strain[:,:,0,0]):.2e}]")
    print(f"  E_yy range: [{np.nanmin(valid_strain[:,:,1,1]):.2e}, {np.nanmax(valid_strain[:,:,1,1]):.2e}]")
    print(f"  E_xy range: [{np.nanmin(valid_strain[:,:,0,1]):.2e}, {np.nanmax(valid_strain[:,:,0,1]):.2e}]")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
