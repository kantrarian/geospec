#!/usr/bin/env python3
"""
data_acquisition.py
Download geodetic strain data for target earthquakes.

Author: R.J. Mathews
Date: January 2026
"""

import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Configuration - use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Target earthquakes for validation
EARTHQUAKES = {
    "tohoku_2011": {
        "name": "2011 Tohoku M9.0",
        "date": "2011-03-11T05:46:24",
        "magnitude": 9.0,
        "lat": 38.297,
        "lon": 142.373,
        "depth_km": 29.0,
        "data_window_days": 30,  # Days before event to analyze
        "data_sources": ["geonet_japan", "unavco"]
    },
    "ridgecrest_2019": {
        "name": "2019 Ridgecrest M7.1", 
        "date": "2019-07-06T03:19:53",
        "magnitude": 7.1,
        "lat": 35.770,
        "lon": -117.599,
        "depth_km": 8.0,
        "data_window_days": 14,
        "data_sources": ["pbo", "unavco"]
    },
    "turkey_2023": {
        "name": "2023 Turkey-Syria M7.8",
        "date": "2023-02-06T01:17:35",
        "magnitude": 7.8,
        "lat": 37.226,
        "lon": 37.014,
        "depth_km": 10.0,
        "data_window_days": 14,
        "data_sources": ["aria_insar", "unavco"]
    },
    "chile_2010": {
        "name": "2010 Chile M8.8",
        "date": "2010-02-27T06:34:14",
        "magnitude": 8.8,
        "lat": -35.846,
        "lon": -72.719,
        "depth_km": 35.0,
        "data_window_days": 21,
        "data_sources": ["ngl", "unavco"]
    },
    "morocco_2023": {
        "name": "2023 Morocco M6.8",
        "date": "2023-09-08T22:11:01",
        "magnitude": 6.8,
        "lat": 31.055,
        "lon": -8.396,
        "depth_km": 18.0,
        "data_window_days": 14,
        "data_sources": ["ngl"]
    }
}


def download_unavco_strain_grids(earthquake_key, output_dir):
    """
    Download UNAVCO strain rate grids.
    
    UNAVCO provides processed strain rate products from GPS networks.
    """
    eq = EARTHQUAKES[earthquake_key]
    print(f"Downloading UNAVCO data for {eq['name']}")
    
    # UNAVCO data access requires authentication
    # For sprint, document the manual download process
    
    instructions = f"""
    UNAVCO Data Download Instructions for {eq['name']}:
    
    1. Go to: https://www.unavco.org/data/gps-gnss/gps-gnss.html
    2. Select "Strain Rate" product
    3. Set region: {eq['lat']-5} to {eq['lat']+5} lat, {eq['lon']-5} to {eq['lon']+5} lon
    4. Set time range: {eq['date'][:10]} minus {eq['data_window_days']} days
    5. Download NetCDF or CSV format
    6. Save to: {output_dir}
    
    Alternative: Use ARIA InSAR products from JPL
    https://aria.jpl.nasa.gov/
    """
    
    readme_file = output_dir / f"{earthquake_key}_download_instructions.txt"
    with open(readme_file, 'w') as f:
        f.write(instructions)
    
    print(f"Instructions saved to {readme_file}")
    return readme_file


def generate_realistic_synthetic_strain_data(earthquake_key, output_dir):
    """
    Generate realistic synthetic strain tensor data for testing.
    
    This allows us to validate the pipeline before real data is acquired.
    The synthetic data embeds known precursor signals for ground truth.
    """
    eq = EARTHQUAKES[earthquake_key]
    print(f"Generating synthetic strain data for {eq['name']}")
    
    # Parameters
    n_hours = eq['data_window_days'] * 24
    n_stations = 50  # GPS stations in network
    dt_hours = 1.0
    
    # Station locations (random within region)
    np.random.seed(hash(earthquake_key) % 2**32)
    station_lats = eq['lat'] + np.random.uniform(-3, 3, n_stations)
    station_lons = eq['lon'] + np.random.uniform(-3, 3, n_stations)
    
    # Time axis
    eq_time = datetime.fromisoformat(eq['date'].replace('Z', '+00:00'))
    start_time = eq_time - timedelta(days=eq['data_window_days'])
    times = [start_time + timedelta(hours=h) for h in range(n_hours)]
    
    # Generate strain tensor time series for each station
    # Shape: (n_hours, n_stations, 3, 3)
    strain_tensors = np.zeros((n_hours, n_stations, 3, 3))
    
    # Background strain rate (nanostrain/year typical values)
    background_rate = 1e-9  # ~30 nanostrain/year
    
    for s in range(n_stations):
        # Distance from epicenter
        dist_km = np.sqrt((station_lats[s] - eq['lat'])**2 + 
                          (station_lons[s] - eq['lon'])**2) * 111  # deg to km
        
        # Strain accumulation direction (radial from epicenter)
        theta = np.arctan2(station_lats[s] - eq['lat'], 
                          station_lons[s] - eq['lon'])
        
        for t in range(n_hours):
            hours_before_eq = n_hours - t
            
            # Base strain tensor (extension/compression pattern)
            e1 = np.array([np.cos(theta), np.sin(theta), 0])
            e3 = np.array([0, 0, 1])
            e2 = np.cross(e3, e1)
            
            # Eigenvalues: extension along e1, compression along e2
            lam1 = background_rate * (1 + 0.1 * np.sin(2*np.pi*t/168))  # Weekly cycle
            lam3 = -background_rate * 0.5
            lam2 = -(lam1 + lam3)
            
            # PRECURSOR SIGNAL: 72-24 hours before earthquake
            if 24 < hours_before_eq < 72:
                # Proximity to epicenter affects signal strength
                proximity_factor = np.exp(-dist_km / 200)  # 200 km decay length
                
                # Progress through precursor window
                precursor_progress = (72 - hours_before_eq) / 48
                
                # Eigenframe rotation (the Λ_geo signal!)
                rotation_angle = 0.3 * precursor_progress**2 * proximity_factor
                
                # Rotate e1 toward e2
                cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
                e1_rot = cos_r * e1 + sin_r * e2
                e2_rot = -sin_r * e1 + cos_r * e2
                e1, e2 = e1_rot, e2_rot
                
                # Strain rate increase
                amplification = 1 + 3 * precursor_progress**2 * proximity_factor
                lam1 *= amplification
                lam3 *= amplification
                lam2 = -(lam1 + lam3)
            
            # IMMEDIATE PRECURSOR: Final 24 hours
            if hours_before_eq <= 24:
                proximity_factor = np.exp(-dist_km / 100)
                
                # Rapid eigenframe rotation
                rotation_angle = 0.5 * proximity_factor * (24 - hours_before_eq) / 24
                cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
                e1_rot = cos_r * e1 + sin_r * e2
                e2_rot = -sin_r * e1 + cos_r * e2
                e1, e2 = e1_rot, e2_rot
                
                # Strong strain rate increase
                amplification = 1 + 5 * proximity_factor
                lam1 *= amplification
            
            # Add realistic noise
            noise_scale = background_rate * 0.2
            lam1 += np.random.normal(0, noise_scale)
            lam2 += np.random.normal(0, noise_scale)
            lam3 = -(lam1 + lam2)  # Maintain trace = 0
            
            # Construct strain tensor
            V = np.column_stack([e1, e2, e3])
            D = np.diag([lam1, lam2, lam3])
            strain_tensors[t, s] = V @ D @ V.T
    
    # Save data
    output_file = output_dir / f"{earthquake_key}_synthetic_strain.npz"
    np.savez(output_file,
             strain_tensors=strain_tensors,
             times=np.array([t.isoformat() for t in times]),
             station_lats=station_lats,
             station_lons=station_lons,
             earthquake_info=json.dumps(eq))
    
    print(f"Synthetic data saved to {output_file}")
    print(f"  Shape: {strain_tensors.shape}")
    print(f"  Time range: {times[0]} to {times[-1]}")
    print(f"  Stations: {n_stations}")
    
    return output_file


def main():
    """Main data acquisition routine."""
    print("=" * 70)
    print("GEOSPEC DATA ACQUISITION")
    print("=" * 70)
    
    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for eq_key, eq_info in EARTHQUAKES.items():
        print(f"\n{'='*50}")
        print(f"Processing: {eq_info['name']}")
        print(f"{'='*50}")
        
        eq_dir = RAW_DIR / eq_key
        eq_dir.mkdir(exist_ok=True)
        
        # Generate synthetic data for pipeline testing
        generate_realistic_synthetic_strain_data(eq_key, eq_dir)
        
        # Document real data download process
        download_unavco_strain_grids(eq_key, eq_dir)
    
    print("\n" + "=" * 70)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review download instructions for real data")
    print("2. Proceed with synthetic data for pipeline validation")
    print("3. Replace with real data when available")


if __name__ == "__main__":
    main()
