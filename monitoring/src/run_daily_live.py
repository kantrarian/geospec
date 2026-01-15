#!/usr/bin/env python3
"""
run_daily_live.py - Daily Monitoring Pipeline with Live GPS Data

Runs the Lambda_geo monitoring pipeline for all configured regions
using real GPS data from Nevada Geodetic Laboratory.

Usage:
    python -m monitoring.src.run_daily_live --date auto
    python -m monitoring.src.run_daily_live --region socal_saf_mojave
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')

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


class DataAcquisitionError(Exception):
    """Raised when live GPS data cannot be acquired."""
    pass


def load_lambda_geo_data_live(region_id: str, end_date: datetime,
                               days: int = 120) -> Dict:
    """
    Load Lambda_geo time series using LIVE GPS data from NGL.

    NO FALLBACK TO SYNTHETIC DATA - if live data fails, we need to fix it.
    Raises DataAcquisitionError on failure.
    """
    cache_dir = Path(__file__).parent.parent / "data" / "gps_cache"

    # Initialize NGL acquisition
    ngl = NGLLiveAcquisition(cache_dir)

    # Load station catalog - REQUIRED
    catalog = ngl.load_station_catalog()
    if not catalog or len(catalog) == 0:
        raise DataAcquisitionError(
            f"Failed to load NGL station catalog. Check network connectivity to geodesy.unr.edu"
        )

    # Acquire live data for this region
    result = acquire_region_data(region_id, ngl, days, end_date)

    if result is None:
        raise DataAcquisitionError(
            f"Failed to acquire GPS data for region {region_id}"
        )

    if result.data_quality == 'insufficient':
        raise DataAcquisitionError(
            f"Insufficient GPS stations for region {region_id}: "
            f"found {result.n_stations} stations, need >= 3"
        )

    if result.n_triangles == 0:
        raise DataAcquisitionError(
            f"Could not create triangulation for region {region_id}: "
            f"{result.n_stations} stations but 0 valid triangles"
        )

    # Convert to expected format
    lambda_geo = result.lambda_geo_grid

    if lambda_geo is None or (hasattr(lambda_geo, 'size') and lambda_geo.size == 0):
        raise DataAcquisitionError(
            f"Lambda_geo computation failed for region {region_id}"
        )

    # Build time series
    n_times = lambda_geo.shape[0] if len(lambda_geo.shape) > 0 and lambda_geo.shape[0] > 0 else days
    dates = [end_date - timedelta(days=n_times-1-i) for i in range(n_times)]

    return {
        'lambda_geo': lambda_geo,
        'times': dates,
        'source': f'live_ngl ({result.n_stations} stations, {result.n_triangles} triangles)',
        'n_stations': result.n_stations,
        'n_triangles': result.n_triangles,
        'data_quality': result.data_quality,
    }


# NO SYNTHETIC FALLBACK - if live data fails, the monitoring run should fail and alert us


def run_monitoring_for_region(
    region_id: str,
    target_date: datetime,
    state_machines: Dict[str, AlertStateMachine],
    storage: AlertStorage,
    version_hash: str
) -> Optional[DailyState]:
    """
    Run monitoring pipeline for a single region with live data.
    """
    print(f"\n{'='*50}")
    print(f"Region: {region_id}")
    print(f"Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"{'='*50}")

    # Load LIVE data
    data = load_lambda_geo_data_live(region_id, target_date, days=120)
    if data is None:
        print(f"  [ERROR] Could not load data for {region_id}")
        return None

    print(f"  Data source: {data['source']}")
    if 'n_stations' in data:
        print(f"  Stations: {data['n_stations']}, Triangles: {data.get('n_triangles', 'N/A')}")

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
        print(f"  [WARN] Could not compute baseline, using defaults")
        baseline_median = np.nanmedian(lambda_max_series) if len(lambda_max_series) > 0 else 0.1
        baseline_robust_std = np.nanstd(lambda_max_series) if len(lambda_max_series) > 0 else 0.01
    else:
        baseline_median = baseline_result.median
        baseline_robust_std = baseline_result.robust_std
        print(f"  Baseline: median={baseline_median:.6f}, robust_std={baseline_robust_std:.6f}")
        print(f"  Baseline window: {baseline_result.window_start.strftime('%Y-%m-%d')} to {baseline_result.window_end.strftime('%Y-%m-%d')}")

    # Get current value (last day)
    current_lambda_max = lambda_max_series[-1] if len(lambda_max_series) > 0 else 0
    if lambda_geo.ndim == 3:
        current_grid = lambda_geo[-1]
    elif lambda_geo.ndim == 2:
        current_grid = lambda_geo[-1].reshape(-1, 1) if lambda_geo.ndim == 2 else np.array([[current_lambda_max]])
    else:
        current_grid = np.array([[current_lambda_max]])

    # Compute ratio and Z-score
    ratio = current_lambda_max / baseline_median if baseline_median > 0 else 0
    zscore = (current_lambda_max - baseline_median) / baseline_robust_std if baseline_robust_std > 0 else 0

    print(f"  Current: lambda_max={current_lambda_max:.6f}, ratio={ratio:.2f}x, Z={zscore:.2f}")

    # Check spatial coherence
    coherence_checker = SpatialCoherence(min_cluster_size=3, min_fraction=0.10)
    threshold = baseline_median * 5.0 if baseline_median > 0 else 0.5  # 5x threshold
    coherence = coherence_checker.check(current_grid, threshold)

    print(f"  Coherence: {coherence.reason}")

    # Update state machine
    new_tier = sm.update(
        date=target_date,
        lambda_max=current_lambda_max,
        baseline_median=baseline_median,
        zscore=zscore,
        is_coherent=coherence.is_coherent,
        cluster_size=coherence.max_cluster_size,
        fraction_elevated=coherence.fraction_elevated,
        baseline_n_days=baseline_result.n_days if baseline_result else 0,
        baseline_start=baseline_result.window_start if baseline_result else target_date,
        baseline_end=baseline_result.window_end if baseline_result else target_date,
        version_hash=version_hash,
    )

    # Get the state that was just created
    state = sm.get_latest_state()

    # Print tier
    tier_desc = sm.get_tier_description()
    tier_symbol = {0: '[OK]', 1: '[WATCH]', 2: '[ELEVATED]', 3: '[HIGH]'}
    print(f"\n  {tier_symbol.get(new_tier.value, '[?]')} TIER {new_tier.value}: {tier_desc}")

    # Alert on elevated activity
    if new_tier >= AlertTier.WATCH:
        print(f"  *** ELEVATED ACTIVITY DETECTED: {ratio:.1f}x baseline ***")

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
    Run monitoring for all configured regions with live GPS data.
    """
    print("=" * 60)
    print("LAMBDA_GEO DAILY MONITORING - LIVE GPS DATA")
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

    # Run each region - track successes and failures
    results = {}
    failed_regions = {}

    for region_id in selected_regions:
        try:
            state = run_monitoring_for_region(
                region_id, target_date, state_machines, storage, version_hash
            )
            if state:
                results[region_id] = state
            else:
                failed_regions[region_id] = "Unknown error - no state returned"
        except DataAcquisitionError as e:
            print(f"\n  [FAILED] {region_id}: {e}")
            failed_regions[region_id] = str(e)
        except Exception as e:
            print(f"\n  [FAILED] {region_id}: Unexpected error: {e}")
            failed_regions[region_id] = f"Unexpected: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Region':<25} {'Tier':>5} {'Ratio':>8} {'Z-score':>8}")
    print("-" * 60)

    elevated_regions = []
    for region_id, state in results.items():
        tier_sym = {0: '[OK]', 1: '[!]', 2: '[!!]', 3: '[!!!]'}
        print(f"{region_id:<25} {tier_sym.get(state.tier.value, '[?]'):>6} {state.tier.value:>3} {state.ratio:>7.2f}x {state.zscore:>8.2f}")
        if state.tier >= AlertTier.WATCH:
            elevated_regions.append(region_id)

    # Print failed regions
    for region_id, error in failed_regions.items():
        print(f"{region_id:<25} {'[FAIL]':>6}     {'DATA ACQUISITION FAILED':>20}")

    print("-" * 60)

    # Report failures prominently
    if failed_regions:
        print(f"\n*** DATA ACQUISITION FAILED FOR: {', '.join(failed_regions.keys())} ***")
        for region_id, error in failed_regions.items():
            print(f"    {region_id}: {error}")

    if elevated_regions:
        print(f"\n*** ELEVATED REGIONS: {', '.join(elevated_regions)} ***")

    print(f"\nResults saved to: {data_dir}")
    print(f"Successful: {len(results)}/{len(selected_regions)}")

    if failed_regions:
        print(f"FAILED: {len(failed_regions)}/{len(selected_regions)} - REQUIRES ATTENTION")

    # Return summary for CI
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
        'failed_regions': failed_regions,
        'any_elevated': any(s.tier >= AlertTier.ELEVATED for s in results.values()),
        'any_failed': len(failed_regions) > 0,
        'elevated_regions': elevated_regions,
    }


def detect_latest_data_date(cache_dir: Path) -> datetime:
    """
    Detect the most recent date with available GPS data.
    Checks IGS20 (current) and IGS14 (historical) to find latest data.
    """
    import requests

    test_stations = ['P595', 'ALBH']  # California, Cascadia

    latest_date = None
    session = requests.Session()
    session.verify = False
    session.headers.update({'User-Agent': 'Mozilla/5.0 GeoSpec/1.0'})

    # Check IGS20 first (current data)
    for code in test_stations:
        try:
            url = f"https://geodesy.unr.edu/gps_timeseries/IGS20/tenv/{code}.tenv"
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                lines = [l for l in response.text.strip().split('\n')
                        if not l.startswith('#') and l.strip()]
                if lines:
                    last = lines[-1].split()
                    decimal_year = float(last[2])
                    year = int(decimal_year)
                    day_of_year = (decimal_year - year) * 365.25
                    dt = datetime(year, 1, 1) + timedelta(days=day_of_year)
                    if latest_date is None or dt > latest_date:
                        latest_date = dt
        except Exception:
            continue

    # If IGS20 not available, check IGS14
    if latest_date is None:
        for code in test_stations:
            try:
                url = f"https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{code}.tenv3"
                response = session.get(url, timeout=30)
                if response.status_code == 200:
                    lines = [l for l in response.text.strip().split('\n')
                            if not l.startswith('site') and l.strip()]
                    if lines:
                        last = lines[-1].split()
                        decimal_year = float(last[2])
                        year = int(decimal_year)
                        day_of_year = (decimal_year - year) * 365.25
                        dt = datetime(year, 1, 1) + timedelta(days=day_of_year)
                        if latest_date is None or dt > latest_date:
                            latest_date = dt
            except Exception:
                continue

    return latest_date if latest_date else datetime.now()


def main():
    parser = argparse.ArgumentParser(description='Run daily Lambda_geo monitoring with live GPS data')
    parser.add_argument('--date', default='auto',
                        help='Target date (YYYY-MM-DD or "auto" for latest available)')
    parser.add_argument('--region', nargs='*',
                        help='Specific regions to monitor (default: all)')

    args = parser.parse_args()

    # Parse date
    if args.date == 'auto':
        # Detect most recent available data
        cache_dir = Path(__file__).parent.parent / "data" / "gps_cache"
        target_date = detect_latest_data_date(cache_dir)
        print(f"[Auto-detected latest data: {target_date.strftime('%Y-%m-%d')}]")
    else:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')

    # Run monitoring
    result = run_all_regions(target_date, args.region)

    # Write result for CI
    result_file = Path(__file__).parent.parent / "data" / "latest_run.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Exit with code based on alert level and failures
    # Exit code 0: All normal
    # Exit code 1: Elevated alert detected (needs review)
    # Exit code 2: Data acquisition failed (needs fix)
    if result.get('any_failed'):
        print("\n[!!] DATA ACQUISITION FAILED - Immediate attention required")
        sys.exit(2)  # Exit code 2 for data failures
    elif result.get('any_elevated'):
        print("\n[!] ELEVATED ALERT DETECTED - Review required")
        sys.exit(1)  # Exit code 1 for elevated alerts
    else:
        print("\n[OK] All regions normal")
        sys.exit(0)


if __name__ == "__main__":
    main()
