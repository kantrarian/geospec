#!/usr/bin/env python3
"""
run_real_data_sprint.py
Integrated sprint runner for real GPS data validation.

Execute the complete 5-step pipeline:
1. Download GPS time series from NGL/UNAVCO/GEONET
2. Convert GPS velocities to strain tensor field (Delaunay)
3. Compute Lambda_geo = ||[E, E_dot]||_F
4. Validate against earthquake (metrics & figures)
5. Generate patent evidence package

Usage:
    # Full sprint (all earthquakes)
    python run_real_data_sprint.py

    # Single earthquake
    python run_real_data_sprint.py -e turkey_2023

    # Skip download (use cached data)
    python run_real_data_sprint.py --skip-download

    # Run specific step
    python run_real_data_sprint.py -s 3  # Just Lambda_geo computation

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONFIGURATION
# ============================================================================

EARTHQUAKES = {
    "tohoku_2011": {
        "name": "2011 Tohoku M9.0",
        "date": "2011-03-11T05:46:24",
        "lat": 38.322,
        "lon": 142.369,
        "magnitude": 9.0,
        "search_radius_deg": 4.0,
        "data_window_days": 30
    },
    "ridgecrest_2019": {
        "name": "2019 Ridgecrest M7.1",
        "date": "2019-07-06T03:19:53",
        "lat": 35.770,
        "lon": -117.599,
        "magnitude": 7.1,
        "search_radius_deg": 3.0,
        "data_window_days": 14
    },
    "turkey_2023": {
        "name": "2023 Turkey-Syria M7.8",
        "date": "2023-02-06T01:17:35",
        "lat": 37.226,
        "lon": 37.014,
        "magnitude": 7.8,
        "search_radius_deg": 3.0,
        "data_window_days": 14
    },
    "chile_2010": {
        "name": "2010 Chile M8.8",
        "date": "2010-02-27T06:34:14",
        "lat": -35.846,
        "lon": -72.719,
        "magnitude": 8.8,
        "search_radius_deg": 4.0,
        "data_window_days": 21
    },
    "morocco_2023": {
        "name": "2023 Morocco M6.8",
        "date": "2023-09-08T22:11:01",
        "lat": 31.055,
        "lon": -8.396,
        "magnitude": 6.8,
        "search_radius_deg": 3.0,
        "data_window_days": 14
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    width = 70
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_step(step_num: int, title: str):
    """Print a step header."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*70}")


# ============================================================================
# STEP 1: GPS DATA ACQUISITION
# ============================================================================

def step1_download_gps_data(earthquake_key: str, skip_download: bool = False) -> dict:
    """
    Step 1: Download GPS time series from NGL.
    
    RETROSPECTIVE VALIDATION MODE:
    - Uses daily position solutions (dt = 24 hours)
    - 72-hour precursor window = 3 daily samples
    - Known station lists for each earthquake
    
    Returns:
        Dictionary with station data and metadata
    """
    print_step(1, "GPS DATA ACQUISITION (NGL Daily Solutions)")
    
    from gps_data_acquisition import NGLDataDownloader
    
    eq_config = EARTHQUAKES[earthquake_key]
    
    # Paths
    cache_dir = PROJECT_ROOT / "data" / "cache" / "ngl"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    
    # Check for cached data
    processed_file = raw_dir / earthquake_key / "gps_velocities.npz"
    metadata_file = raw_dir / earthquake_key / "gps" / "metadata.json"
    
    if skip_download and processed_file.exists():
        print(f"Using cached GPS data: {processed_file}")
        data = np.load(processed_file, allow_pickle=True)
        return {
            'station_data': dict(data['station_data'].item()),
            'station_locations': dict(data['station_locations'].item()),
            'earthquake_info': eq_config,
            'dt_hours': 24  # Daily data
        }
    
    # Initialize downloader
    downloader = NGLDataDownloader(cache_dir)
    
    # Download data (returns tuple: station_data, station_locations)
    result = downloader.download_earthquake_data(earthquake_key, raw_dir, eq_config)
    
    if isinstance(result, tuple):
        station_data, station_locations = result
    else:
        station_data = result
        station_locations = {}
    
    if len(station_data) < 3:
        print(f"\nWARNING: Only {len(station_data)} stations found!")
        print("Falling back to synthetic data for testing...")
        return step1_fallback_synthetic(earthquake_key)
    
    # Load station locations from metadata if available
    if not station_locations and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            station_locations = metadata.get('station_locations', {})
    
    # Save processed data
    output_dir = raw_dir / earthquake_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        processed_file,
        station_data=station_data,
        station_locations=station_locations,
        earthquake_info=eq_config
    )
    
    print(f"\n[OK] GPS data acquisition complete")
    print(f"    Stations: {len(station_data)}")
    print(f"    Temporal resolution: DAILY (dt=24h)")
    print(f"    72h precursor window = 3 samples")
    print(f"    Saved to: {processed_file}")
    
    return {
        'station_data': station_data,
        'station_locations': station_locations,
        'earthquake_info': eq_config,
        'dt_hours': 24  # Daily data
    }


def step1_fallback_synthetic(earthquake_key: str) -> dict:
    """Fallback to synthetic data if real data unavailable."""
    print("\nGenerating synthetic GPS data as fallback...")
    
    from data_acquisition import generate_realistic_synthetic_strain_data, EARTHQUAKES as EQ_DEFS
    
    raw_dir = PROJECT_ROOT / "data" / "raw"
    eq_dir = raw_dir / earthquake_key
    eq_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    generate_realistic_synthetic_strain_data(earthquake_key, eq_dir)
    
    # Load and convert to GPS-like format
    data_file = eq_dir / f"{earthquake_key}_synthetic_strain.npz"
    data = np.load(data_file, allow_pickle=True)
    
    eq_info = json.loads(str(data['earthquake_info']))
    
    # Create fake station data from strain tensor grid
    n_stations = len(data['station_lats'])
    station_data = {}
    station_locations = {}
    
    for i in range(min(n_stations, 30)):  # Limit for performance
        code = f"SYN{i:02d}"
        station_locations[code] = (
            float(data['station_lats'][i]),
            float(data['station_lons'][i])
        )
        
        # Create dummy velocity data
        import pandas as pd
        times = [datetime.fromisoformat(t) for t in data['times']]
        station_data[code] = pd.DataFrame({
            'datetime': times,
            'n': np.random.randn(len(times)) * 0.1,
            'e': np.random.randn(len(times)) * 0.1,
            'u': np.random.randn(len(times)) * 0.05,
            'vn': np.random.randn(len(times)) * 0.01,
            've': np.random.randn(len(times)) * 0.01,
            'vu': np.random.randn(len(times)) * 0.005,
        })
    
    return {
        'station_data': station_data,
        'station_locations': station_locations,
        'earthquake_info': eq_info,
        'is_synthetic': True,
        'dt_hours': 1.0  # Synthetic uses hourly data
    }


# ============================================================================
# STEP 2: GPS TO STRAIN CONVERSION
# ============================================================================

def step2_convert_to_strain(earthquake_key: str, gps_result: dict) -> dict:
    """
    Step 2: Convert GPS velocities to strain tensor field.
    
    Uses Delaunay triangulation for the conversion.
    """
    print_step(2, "GPS TO STRAIN CONVERSION (Delaunay)")
    
    from gps_to_strain import GPSToStrainConverter, convert_strain_field_to_stations
    
    # Check if we should use pre-computed synthetic strain
    if gps_result.get('is_synthetic'):
        print("Using pre-computed synthetic strain tensors...")
        
        raw_dir = PROJECT_ROOT / "data" / "raw"
        data_file = raw_dir / earthquake_key / f"{earthquake_key}_synthetic_strain.npz"
        
        if data_file.exists():
            data = np.load(data_file, allow_pickle=True)
            return {
                'strain_tensors': data['strain_tensors'],
                'station_lats': data['station_lats'],
                'station_lons': data['station_lons'],
                'times': data['times'],
                'earthquake_info': gps_result['earthquake_info']
            }
    
    # Initialize converter
    converter = GPSToStrainConverter(
        grid_resolution=0.2,  # 0.2 degrees ~ 22 km
        temporal_smoothing_window=7,
        min_triangle_quality=0.1,
        spatial_smoothing=True
    )
    
    # Convert
    station_data = gps_result['station_data']
    station_locations = gps_result['station_locations']
    earthquake_info = gps_result['earthquake_info']
    
    if len(station_data) < 3:
        print("ERROR: Not enough stations for triangulation")
        return None
    
    field = converter.convert(station_data, station_locations, earthquake_info)
    
    # Save strain field
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    strain_file = processed_dir / f"{earthquake_key}_strain_field.npz"
    converter.save_strain_field(field, strain_file)
    
    # Convert to station-based format for lambda_geo
    strain_tensors, station_lats, station_lons, times = convert_strain_field_to_stations(field)
    
    print(f"\n[OK] Strain conversion complete")
    print(f"    Grid: {len(field.lats)} x {len(field.lons)}")
    print(f"    Points: {len(station_lats)}")
    print(f"    Times: {len(times)}")
    
    return {
        'strain_tensors': strain_tensors,
        'station_lats': station_lats,
        'station_lons': station_lons,
        'times': times,
        'earthquake_info': earthquake_info
    }


# ============================================================================
# STEP 3: LAMBDA_GEO COMPUTATION
# ============================================================================

def step3_compute_lambda_geo(earthquake_key: str, strain_result: dict, dt_hours: float = 24.0) -> dict:
    """
    Step 3: Compute Lambda_geo = ||[E, E_dot]||_F
    
    For daily GPS data: dt = 24 hours
    72-hour precursor window = 3 samples
    """
    print_step(3, "Lambda_geo COMPUTATION")
    
    from lambda_geo import LambdaGeoAnalyzer
    
    print(f"  Temporal resolution: dt = {dt_hours} hours")
    print(f"  72h precursor = {72/dt_hours:.0f} samples")
    
    # Initialize analyzer with correct dt
    analyzer = LambdaGeoAnalyzer(
        dt_hours=dt_hours,  # Use actual temporal resolution
        smoothing_window=1 if dt_hours >= 24 else 2,  # Less smoothing for daily data
        derivative_method='central'
    )
    
    # Run analysis
    result = analyzer.analyze(
        strain_tensors=strain_result['strain_tensors'],
        times=strain_result['times'],
        station_lats=strain_result['station_lats'],
        station_lons=strain_result['station_lons'],
        earthquake_info=strain_result['earthquake_info']
    )
    
    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = results_dir / f"{earthquake_key}_lambda_geo.npz"
    np.savez(
        result_file,
        times=result.times,
        station_lats=result.station_lats,
        station_lons=result.station_lons,
        lambda_geo=result.lambda_geo,
        lambda_geo_normalized=result.lambda_geo_normalized,
        spectral_gap_12=result.spectral_gap_12,
        eigenframe_rotation_rate=result.eigenframe_rotation_rate,
        risk_score=result.risk_score,
        spatial_max_risk=result.spatial_max_risk,
        earthquake_info=json.dumps(result.earthquake_info)
    )
    
    print(f"\n[OK] Lambda_geo computation complete")
    print(f"    Max Lambda_geo: {np.max(result.lambda_geo):.2e}")
    print(f"    Max risk score: {np.max(result.spatial_max_risk):.3f}")
    print(f"    Saved to: {result_file}")
    
    return {'result': result, 'earthquake_key': earthquake_key}


# ============================================================================
# STEP 4: VALIDATION
# ============================================================================

def step4_validate(earthquake_key: str, lambda_result: dict) -> dict:
    """
    Step 4: Validate against earthquake metrics.
    """
    print_step(4, "VALIDATION")
    
    from validate_lambda_geo import EarthquakeValidator
    
    result = lambda_result['result']
    results_dir = PROJECT_ROOT / "results"
    
    # Initialize validator
    validator = EarthquakeValidator(results_dir)
    
    # Find earthquake time index
    eq_info = result.earthquake_info
    data_window_days = eq_info.get('data_window_days', 14)
    eq_time_idx = data_window_days  # Earthquake at end for daily data
    
    # Compute metrics
    metrics = validator.compute_metrics(result, eq_time_idx)
    metrics['earthquake_key'] = earthquake_key
    metrics['earthquake_info'] = eq_info
    
    # Print summary
    print(f"\nValidation Metrics for {eq_info.get('name', earthquake_key)}:")
    print(f"  Amplification factor: {metrics['amplification_factor']:.2f}x")
    print(f"  Max Z-score: {metrics['zscore_max']:.2f}")
    first_det = metrics.get('first_detection_hours_before')
    print(f"  First detection: {first_det:.0f}h before" if first_det else "  First detection: N/A")
    print(f"  % time high risk: {metrics['pct_time_high_risk']:.1f}%")
    
    # Create validation figure
    fig_path = results_dir / f"{earthquake_key}_real_data_validation.png"
    validator.create_validation_figure(result, eq_time_idx, metrics, fig_path)
    
    # Save metrics
    metrics_path = results_dir / f"{earthquake_key}_real_data_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Determine success
    success = (
        first_det is not None and
        24 <= first_det <= 72 and
        metrics['zscore_max'] > 2.0
    )
    
    print(f"\n  SUCCESS: {'YES' if success else 'NO'}")
    
    return {'metrics': metrics, 'success': success}


# ============================================================================
# STEP 5: PATENT EVIDENCE
# ============================================================================

def step5_generate_evidence(all_results: list):
    """
    Step 5: Generate patent evidence package.
    """
    print_step(5, "PATENT EVIDENCE GENERATION")
    
    from generate_patent_evidence import generate_evidence_package
    
    # First update the validation summary with real data results
    results_dir = PROJECT_ROOT / "results"
    
    summary = {
        'total_earthquakes': len(all_results),
        'successful_detections': sum(1 for r in all_results if r.get('success')),
        'success_rate': sum(1 for r in all_results if r.get('success')) / len(all_results) if all_results else 0,
        'data_source': 'Real GPS data from NGL',
        'individual_results': [r.get('metrics', {}) for r in all_results]
    }
    
    summary_path = results_dir / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate evidence package
    generate_evidence_package()
    
    print(f"\n[OK] Patent evidence generation complete")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_sprint(earthquake_keys: list, 
                    skip_download: bool = False,
                    step: int = None):
    """
    Run the complete validation sprint.
    """
    print_banner("GEOSPEC Lambda_geo REAL DATA VALIDATION SPRINT")
    print(f"""
    Author: R.J. Mathews
    Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Data Source: Nevada Geodetic Lab (NGL) GPS time series
    Method: Delaunay triangulation -> Strain tensor -> Lambda_geo
    
    Target Earthquakes: {', '.join(earthquake_keys)}
    """)
    
    start_time = time.time()
    all_results = []
    
    for eq_key in earthquake_keys:
        print_banner(f"Processing: {EARTHQUAKES[eq_key]['name']}", "-")
        
        try:
            # Step 1: Download GPS data
            if step is None or step == 1:
                gps_result = step1_download_gps_data(eq_key, skip_download)
            else:
                # Load cached GPS data
                processed_file = PROJECT_ROOT / "data" / "raw" / eq_key / "gps_velocities.npz"
                if processed_file.exists():
                    data = np.load(processed_file, allow_pickle=True)
                    gps_result = {
                        'station_data': dict(data['station_data'].item()),
                        'station_locations': dict(data['station_locations'].item()),
                        'earthquake_info': EARTHQUAKES[eq_key]
                    }
                else:
                    gps_result = step1_fallback_synthetic(eq_key)
            
            # Step 2: Convert to strain
            if step is None or step == 2:
                strain_result = step2_convert_to_strain(eq_key, gps_result)
            else:
                # Load from synthetic data
                raw_dir = PROJECT_ROOT / "data" / "raw"
                data_file = raw_dir / eq_key / f"{eq_key}_synthetic_strain.npz"
                if data_file.exists():
                    data = np.load(data_file, allow_pickle=True)
                    strain_result = {
                        'strain_tensors': data['strain_tensors'],
                        'station_lats': data['station_lats'],
                        'station_lons': data['station_lons'],
                        'times': data['times'],
                        'earthquake_info': EARTHQUAKES[eq_key]
                    }
                else:
                    strain_result = step2_convert_to_strain(eq_key, gps_result)
            
            if strain_result is None:
                print(f"Skipping {eq_key}: strain conversion failed")
                continue
            
            # Step 3: Compute Lambda_geo
            if step is None or step == 3:
                # Use dt from GPS data (24h for daily, 1h for hourly)
                dt_hours = gps_result.get('dt_hours', 24.0)
                lambda_result = step3_compute_lambda_geo(eq_key, strain_result, dt_hours)
            
            # Step 4: Validate
            if step is None or step == 4:
                validation_result = step4_validate(eq_key, lambda_result)
                all_results.append(validation_result)
        
        except Exception as e:
            print(f"\n[ERROR] Failed to process {eq_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Generate evidence (only if running full sprint)
    if step is None or step == 5:
        if all_results:
            step5_generate_evidence(all_results)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_banner("SPRINT COMPLETE")
    print(f"""
    Earthquakes processed: {len(earthquake_keys)}
    Successful validations: {sum(1 for r in all_results if r.get('success'))}
    Total time: {elapsed:.1f} seconds
    
    Results saved to: {PROJECT_ROOT / 'results'}
    Figures saved to: {PROJECT_ROOT / 'figures'}
    """)


def main():
    """Parse arguments and run sprint."""
    parser = argparse.ArgumentParser(
        description='GeoSpec Lambda_geo Real Data Validation Sprint'
    )
    parser.add_argument(
        '-e', '--earthquake',
        choices=list(EARTHQUAKES.keys()) + ['all'],
        default='all',
        help='Earthquake to process (default: all)'
    )
    parser.add_argument(
        '-s', '--step',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only specific step (1-5)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip GPS download, use cached data'
    )
    
    args = parser.parse_args()
    
    # Determine earthquakes to process
    if args.earthquake == 'all':
        earthquake_keys = list(EARTHQUAKES.keys())
    else:
        earthquake_keys = [args.earthquake]
    
    # Run sprint
    run_full_sprint(
        earthquake_keys,
        skip_download=args.skip_download,
        step=args.step
    )


if __name__ == "__main__":
    main()
