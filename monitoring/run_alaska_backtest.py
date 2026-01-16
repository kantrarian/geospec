
import sys
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Local imports
from src.live_data import NGLLiveAcquisition, acquire_region_data
from src.ensemble import GeoSpecEnsemble, lambda_geo_to_risk
from src.regions import FAULT_POLYGONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_alaska_backtest():
    print("="*80)
    print("ALASKA (ANCHORAGE 2018) BACKTEST")
    print("="*80)
    
    region = 'anchorage'
    start_date = datetime(2018, 11, 20)
    end_date = datetime(2018, 12, 10)
    
    # Event: Nov 30, 2018 M7.1
    
    # Cache directory for this backtest
    cache_dir = Path(__file__).parent / "data" / "backtest" / "anchorage_2018" / "gps_full"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize NGL
    ngl = NGLLiveAcquisition(cache_dir)
    
    # Initialize Ensemble
    ensemble = GeoSpecEnsemble(region)
    
    # Storage for results
    results = []
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\nProcessing {date_str}...")
        
        # 1. Compute Lambda_geo (120 days window)
        lg_result = acquire_region_data(region, ngl, days_back=120, target_date=current_date)
        
        lg_val = 0.0
        lg_ratio = 0.0
        n_stations = 0
        
        if lg_result and lg_result.n_stations >= 3:
            lg_val = lg_result.lambda_geo_max
            n_stations = lg_result.n_stations
            
            # Baseline estimation
            baseline = 0.5
            lg_ratio = lg_val / baseline if baseline > 0 else 0
            
            # Update ensemble
            ensemble.set_lambda_geo(current_date, lg_ratio)
            
            print(f"  Lambda_geo: {lg_val:.6f} (Ratio ~{lg_ratio:.1f}x vs {baseline})")
            print(f"  Stations: {n_stations}")
        else:
            print("  Lambda_geo: Insufficient data")

        # 2. Compute Ensemble Risk
        risk_result = ensemble.compute_lambda_geo_risk(current_date)
        
        results.append({
            'date': date_str,
            'lambda_geo': lg_val,
            'ratio': lg_ratio,
            'risk': risk_result.risk_score,
            'stations': n_stations
        })
        
        current_date += timedelta(days=1)
        
    # Save results
    output_file = Path(__file__).parent / "data" / "backtest" / "anchorage_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_file}")
    
    print("\nSummary:")
    for r in results:
        print(f"{r['date']}: L_geo={r['lambda_geo']:.6f} Risk={r['risk']:.2f} Stn={r['stations']}")

if __name__ == "__main__":
    run_alaska_backtest()
