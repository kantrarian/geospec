#!/usr/bin/env python3
"""
run_daily.py - Daily Monitoring Pipeline

Runs the Λ_geo monitoring pipeline for all configured regions.

Usage:
    python -m src.run_daily --date auto
    python -m src.run_daily --date 2024-01-15
    python -m src.run_daily --region socal_saf_mojave
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.src.baseline import RollingBaseline
from monitoring.src.coherence import SpatialCoherence
from monitoring.src.alerts import AlertStateMachine, AlertStorage, DailyState, AlertTier
from monitoring.src.regions import FAULT_POLYGONS
from monitoring.src.live_data import NGLLiveAcquisition, acquire_region_data


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def load_lambda_geo_data(region_id: str, end_date: datetime, 
                          days: int = 120) -> Optional[Dict]:
    """
    Load Λ_geo time series for a region.
    
    In production, this would:
    1. Download GPS data from NGL
    2. Convert to strain via Delaunay
    3. Compute Λ_geo
    
    For now, returns synthetic data or cached results.
    """
    # Check for cached results
    results_dir = Path(__file__).parent.parent.parent / "results"
    
    # Map region to earthquake data if available
    region_to_eq = {
        'socal_saf_mojave': 'ridgecrest_2019',
        'socal_saf_coachella': 'ridgecrest_2019',
    }
    
    eq_key = region_to_eq.get(region_id)
    if eq_key:
        cache_file = results_dir / f"{eq_key}_lambda_geo.npz"
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            return {
                'lambda_geo': data['lambda_geo'],
                'times': data['times'],
                'source': 'cache',
            }
    
    # Generate synthetic baseline data for testing
    np.random.seed(hash(region_id) % 2**32)
    
    n_days = days
    dates = [end_date - timedelta(days=n_days-1-i) for i in range(n_days)]
    
    # Baseline level with seasonal variation
    decimal_days = np.arange(n_days)
    seasonal = 0.02 * np.sin(2 * np.pi * decimal_days / 365)
    baseline = 0.1 + seasonal + 0.01 * np.random.randn(n_days)
    
    # Create grid (10x10)
    lambda_geo = np.zeros((n_days, 10, 10))
    for t in range(n_days):
        lambda_geo[t] = baseline[t] + 0.005 * np.random.randn(10, 10)
    
    return {
        'lambda_geo': lambda_geo,
        'times': dates,
        'source': 'synthetic',
    }


def run_monitoring_for_region(
    region_id: str,
    target_date: datetime,
    state_machines: Dict[str, AlertStateMachine],
    storage: AlertStorage,
    version_hash: str
) -> Optional[DailyState]:
    """
    Run monitoring pipeline for a single region.
    """
    print(f"\n{'='*50}")
    print(f"Region: {region_id}")
    print(f"Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"{'='*50}")
    
    # Load data
    data = load_lambda_geo_data(region_id, target_date, days=120)
    if data is None:
        print(f"  [ERROR] Could not load data for {region_id}")
        return None
    
    print(f"  Data source: {data['source']}")
    
    lambda_geo = data['lambda_geo']
    times = data['times']
    
    # Get state machine
    if region_id not in state_machines:
        state_machines[region_id] = AlertStateMachine(region_id)
    sm = state_machines[region_id]
    
    # Compute spatial max at each time (handle different array shapes)
    if lambda_geo.ndim == 3:
        lambda_max_series = np.nanmax(lambda_geo, axis=(1, 2))
    elif lambda_geo.ndim == 2:
        lambda_max_series = np.nanmax(lambda_geo, axis=1)
    else:
        lambda_max_series = lambda_geo
    
    # Compute rolling baseline
    baseline_computer = RollingBaseline(
        lookback_days=90,
        exclude_recent_days=14,
        seasonal_detrend=True
    )
    
    baseline_result = baseline_computer.compute(
        lambda_max_series, times, current_date=target_date
    )
    
    if baseline_result is None:
        print(f"  [ERROR] Could not compute baseline")
        return None
    
    print(f"  Baseline: median={baseline_result.median:.4f}, robust_std={baseline_result.robust_std:.4f}")
    print(f"  Baseline window: {baseline_result.window_start.strftime('%Y-%m-%d')} to {baseline_result.window_end.strftime('%Y-%m-%d')}")
    
    # Get current value (last day)
    current_lambda_max = lambda_max_series[-1]
    if lambda_geo.ndim == 3:
        current_grid = lambda_geo[-1]
    else:
        # Create a fake grid for 2D data
        current_grid = lambda_geo[-1].reshape(-1, 1) if lambda_geo.ndim == 2 else np.array([[current_lambda_max]])
    
    # Compute ratio and Z-score
    ratio = current_lambda_max / baseline_result.median if baseline_result.median > 0 else 0
    zscore = baseline_result.zscore(current_lambda_max)
    
    print(f"  Current: lambda_max={current_lambda_max:.4f}, ratio={ratio:.2f}×, Z={zscore:.2f}")
    
    # Check spatial coherence
    coherence_checker = SpatialCoherence(min_cluster_size=3, min_fraction=0.10)
    threshold = baseline_result.threshold(5.0)  # 5× threshold
    coherence = coherence_checker.check(current_grid, threshold)
    
    print(f"  Coherence: {coherence.reason}")
    
    # Update state machine
    new_tier = sm.update(
        date=target_date,
        lambda_max=current_lambda_max,
        baseline_median=baseline_result.median,
        zscore=zscore,
        is_coherent=coherence.is_coherent,
        cluster_size=coherence.max_cluster_size,
        fraction_elevated=coherence.fraction_elevated,
        baseline_n_days=baseline_result.n_days,
        baseline_start=baseline_result.window_start,
        baseline_end=baseline_result.window_end,
        version_hash=version_hash,
    )
    
    # Get the state that was just created
    state = sm.get_latest_state()
    
    # Print tier
    tier_desc = sm.get_tier_description()
    tier_symbol = {0: '[OK]', 1: '[WATCH]', 2: '[ELEVATED]', 3: '[HIGH]'}
    print(f"\n  {tier_symbol.get(new_tier.value, '[?]')} TIER {new_tier.value}: {tier_desc}")
    
    # Check for transitions
    if len(sm.transitions) > 0:
        latest_transition = sm.transitions[-1]
        if latest_transition.timestamp == target_date:
            print(f"  [!] TRANSITION: {latest_transition.reason}")
            storage.save_transition(region_id, latest_transition)
    
    # Save state
    storage.save_state(state)
    
    return state


def run_all_regions(target_date: datetime, regions: Optional[List[str]] = None):
    """
    Run monitoring for all configured regions.
    """
    print("=" * 60)
    print("Lambda_geo DAILY MONITORING RUN")
    print("=" * 60)
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    version_hash = get_git_hash()
    print(f"Version: {version_hash}")
    
    # Initialize storage
    data_dir = Path(__file__).parent.parent / "data"
    storage = AlertStorage(data_dir)
    
    # State machines (would be loaded from persistent state in production)
    state_machines: Dict[str, AlertStateMachine] = {}
    
    # Select regions
    if regions:
        selected_regions = [r for r in regions if r in FAULT_POLYGONS]
    else:
        # Run ALL configured regions
        selected_regions = list(FAULT_POLYGONS.keys())
    
    print(f"Regions: {', '.join(selected_regions)}")
    
    # Run each region
    results = {}
    for region_id in selected_regions:
        state = run_monitoring_for_region(
            region_id, target_date, state_machines, storage, version_hash
        )
        if state:
            results[region_id] = state
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Region':<25} {'Tier':>5} {'Ratio':>8} {'Z-score':>8}")
    print("-" * 60)
    
    for region_id, state in results.items():
        tier_sym = {0: '[OK]', 1: '[!]', 2: '[!!]', 3: '[!!!]'}
        print(f"{region_id:<25} {tier_sym.get(state.tier.value, '[?]'):>6} {state.tier.value:>3} {state.ratio:>7.2f}x {state.zscore:>8.2f}")
    
    print("-" * 60)
    print(f"\nResults saved to: {data_dir}")
    
    # Return summary for GitHub Actions
    return {
        'date': target_date.strftime('%Y-%m-%d'),
        'regions': {
            rid: {
                'tier': s.tier.value,
                'ratio': s.ratio,
                'zscore': s.zscore,
            }
            for rid, s in results.items()
        },
        'any_elevated': any(s.tier >= AlertTier.ELEVATED for s in results.values()),
    }


def main():
    parser = argparse.ArgumentParser(description='Run daily Λ_geo monitoring')
    parser.add_argument('--date', default='auto',
                        help='Target date (YYYY-MM-DD or "auto" for today)')
    parser.add_argument('--region', nargs='*',
                        help='Specific regions to monitor (default: all)')
    
    args = parser.parse_args()
    
    # Parse date
    if args.date == 'auto':
        target_date = datetime.now()
    else:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    # Run monitoring
    result = run_all_regions(target_date, args.region)
    
    # Write result for CI
    result_file = Path(__file__).parent.parent / "data" / "latest_run.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Exit with code based on alert level
    if result.get('any_elevated'):
        print("\n[!] ELEVATED ALERT DETECTED - Review required")
        sys.exit(1)  # Non-zero exit to trigger notification
    else:
        print("\n[OK] All regions normal")
        sys.exit(0)


if __name__ == "__main__":
    main()
