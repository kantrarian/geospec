#!/usr/bin/env python3
"""
spatial_analysis.py
Spatial visualization of Λ_geo field and earthquake localization.

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import json
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from lambda_geo import LambdaGeoAnalyzer, load_strain_data


def create_spatial_evolution_figure(result, eq_info, output_path):
    """Create figure showing spatial evolution of Λ_geo toward earthquake."""
    
    n_times = len(result.times)
    dt = result.computation_params['dt_hours']
    eq_time_idx = eq_info['data_window_days'] * 24
    
    # Select time snapshots
    times_before = [72, 48, 24, 12, 6, 1]  # hours before earthquake
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Spatial Evolution of Λ_geo: {eq_info['name']}", fontsize=14, fontweight='bold')
    
    # Normalize colors across all panels
    vmax = np.percentile(result.lambda_geo, 99)
    
    scatter = None
    for idx, hours_before in enumerate(times_before):
        ax = axes.flat[idx]
        t_idx = int(eq_time_idx - hours_before / dt)
        t_idx = max(0, min(t_idx, n_times - 1))
        
        # Scatter plot of stations colored by Λ_geo
        scatter = ax.scatter(
            result.station_lons,
            result.station_lats,
            c=result.lambda_geo[t_idx],
            s=100,
            cmap='hot',
            vmin=0,
            vmax=vmax,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Mark epicenter
        ax.scatter(eq_info['lon'], eq_info['lat'], 
                   marker='*', s=300, c='cyan', edgecolors='black', 
                   linewidths=2, zorder=10, label='Epicenter')
        
        ax.set_title(f"t = -{hours_before}h")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Colorbar
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', 
                            fraction=0.05, pad=0.1)
        cbar.set_label('Λ_geo (strain tensor commutator norm)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Spatial evolution figure saved: {output_path}")


def create_epicenter_focusing_figure(result, eq_info, output_path):
    """Show how Λ_geo signal focuses toward epicenter."""
    
    eq_lat, eq_lon = eq_info['lat'], eq_info['lon']
    n_times = len(result.times)
    dt = result.computation_params['dt_hours']
    eq_time_idx = int(eq_info['data_window_days'] * 24)
    
    # Compute distance of each station from epicenter
    distances = np.sqrt(
        (result.station_lats - eq_lat)**2 + 
        (result.station_lons - eq_lon)**2
    ) * 111  # km
    
    # Bin stations by distance
    dist_bins = [0, 50, 100, 150, 200, 300]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hours_before = (np.arange(n_times) - eq_time_idx) * dt
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dist_bins)-1))
    
    for i in range(len(dist_bins) - 1):
        d_min, d_max = dist_bins[i], dist_bins[i+1]
        mask = (distances >= d_min) & (distances < d_max)
        
        if np.any(mask):
            lambda_mean = np.mean(result.lambda_geo[:, mask], axis=1)
            ax.plot(hours_before, lambda_mean, 
                    color=colors[i], linewidth=2,
                    label=f'{d_min}-{d_max} km from epicenter')
    
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Earthquake')
    ax.axvspan(-72, -24, alpha=0.2, color='orange', label='Precursor window')
    
    ax.set_xlabel('Hours before earthquake')
    ax.set_ylabel('Mean Λ_geo')
    ax.set_title(f'Λ_geo by Distance from Epicenter: {eq_info["name"]}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Epicenter focusing figure saved: {output_path}")


def create_principal_direction_figure(result, eq_info, output_path):
    """Visualize the principal strain direction field evolution."""
    
    n_times = len(result.times)
    dt = result.computation_params['dt_hours']
    eq_time_idx = eq_info['data_window_days'] * 24
    
    # Select time snapshots
    times_before = [72, 24, 6]  # hours before earthquake
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Principal Strain Direction Evolution: {eq_info['name']}", 
                 fontsize=14, fontweight='bold')
    
    for idx, hours_before in enumerate(times_before):
        ax = axes[idx]
        t_idx = int(eq_time_idx - hours_before / dt)
        t_idx = max(0, min(t_idx, n_times - 1))
        
        # Get principal directions at this time
        e1 = result.principal_direction[t_idx]
        
        # Plot as quiver
        ax.quiver(
            result.station_lons,
            result.station_lats,
            e1[:, 0],  # x component
            e1[:, 1],  # y component
            result.lambda_geo[t_idx],
            cmap='hot',
            scale=5,
            width=0.005
        )
        
        # Mark epicenter
        ax.scatter(eq_info['lon'], eq_info['lat'], 
                   marker='*', s=300, c='cyan', edgecolors='black', 
                   linewidths=2, zorder=10, label='Epicenter')
        
        ax.set_title(f"t = -{hours_before}h")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Principal direction figure saved: {output_path}")


def main():
    """Generate spatial analysis figures for all earthquakes."""
    
    # Paths - use relative to project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    results_dir = project_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    earthquakes = ['tohoku_2011', 'ridgecrest_2019', 'turkey_2023']
    
    analyzer = LambdaGeoAnalyzer(dt_hours=1.0, smoothing_window=2)
    
    for eq_key in earthquakes:
        data_file = data_dir / eq_key / f"{eq_key}_synthetic_strain.npz"
        
        if not data_file.exists():
            print(f"Skipping {eq_key}: data not found")
            continue
        
        print(f"\nProcessing {eq_key}...")
        
        # Load and analyze
        strain, times, lats, lons, eq_info = load_strain_data(data_file)
        result = analyzer.analyze(strain, times, lats, lons, eq_info)
        
        # Spatial evolution
        create_spatial_evolution_figure(
            result, eq_info,
            results_dir / f"{eq_key}_spatial_evolution.png"
        )
        
        # Epicenter focusing
        create_epicenter_focusing_figure(
            result, eq_info,
            results_dir / f"{eq_key}_epicenter_focusing.png"
        )
        
        # Principal direction
        create_principal_direction_figure(
            result, eq_info,
            results_dir / f"{eq_key}_principal_direction.png"
        )
    
    print("\nSpatial analysis complete!")


if __name__ == "__main__":
    main()
