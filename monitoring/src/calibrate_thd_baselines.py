#!/usr/bin/env python3
"""
calibrate_thd_baselines.py - Compute THD baselines using the actual pipeline

This script computes THD baselines for each station by:
1. Fetching data over a calibration window (default 60 days)
2. Computing daily THD using the same SeismicTHDAnalyzer settings
3. Calculating robust statistics (median + MAD -> sigma)
4. Outputting results for station_baselines.py

Usage:
    python calibrate_thd_baselines.py                    # All stations
    python calibrate_thd_baselines.py --station IU.TUC   # Single station
    python calibrate_thd_baselines.py --days 90          # 90-day window

Author: R.J. Mathews / Claude
Date: January 2026
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import THD components
try:
    from seismic_thd import SeismicTHDAnalyzer
    from seismic_data import SeismicDataFetcher
except ImportError as e:
    logger.error(f"Failed to import THD modules: {e}")
    raise

# Stations to calibrate (network.station -> region hint)
CALIBRATION_STATIONS = {
    'IU.TUC': 'SoCal/Ridgecrest (Tucson, AZ)',
    'BK.BKS': 'NorCal Hayward (Berkeley)',
    'IU.COR': 'Cascadia (Corvallis, OR)',
    'IU.MAJO': 'Tokyo Kanto (Matsushiro, Japan)',
    'IU.ANTO': 'Turkey (Ankara)',
    'IV.CAFE': 'Campi Flegrei (Italy)',
}


def compute_daily_thd(
    fetcher: SeismicDataFetcher,
    analyzer: SeismicTHDAnalyzer,
    network: str,
    station: str,
    date: datetime,
) -> Optional[float]:
    """Compute THD for a single day using the actual pipeline."""
    try:
        # Fetch 25 hours of data (same as production)
        start = date.replace(hour=0, minute=0, second=0)
        end = start + timedelta(hours=25)

        stream = fetcher.fetch_continuous_data_for_thd(
            network=network,
            station=station,
            start_time=start,
            end_time=end,
        )

        if stream is None or len(stream) == 0:
            return None

        # Analyze using same settings as production
        result = analyzer.analyze_window(stream)

        if result and result.thd_value > 0:
            return result.thd_value

        return None

    except Exception as e:
        logger.debug(f"Failed to compute THD for {network}.{station} on {date.date()}: {e}")
        return None


def calibrate_station(
    network: str,
    station: str,
    days_back: int = 60,
    end_date: datetime = None,
) -> Dict:
    """
    Calibrate baseline for a single station.

    Returns dict with:
        - station: network.station
        - mean_thd: robust mean (median)
        - std_thd: robust std (MAD * 1.4826)
        - n_samples: number of valid daily samples
        - calibration_period: date range
        - daily_values: list of (date, thd) for inspection
    """
    if end_date is None:
        end_date = datetime.now() - timedelta(days=2)  # Account for data latency

    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Calibrating {network}.{station} from {start_date.date()} to {end_date.date()}")

    # Initialize components
    cache_dir = Path(__file__).parent.parent / 'data' / 'seismic_cache'
    fetcher = SeismicDataFetcher(cache_dir=cache_dir)
    analyzer = SeismicTHDAnalyzer()

    # Collect daily THD values
    daily_values = []
    current = start_date

    while current <= end_date:
        thd = compute_daily_thd(fetcher, analyzer, network, station, current)
        if thd is not None:
            daily_values.append((current.strftime('%Y-%m-%d'), thd))
            logger.debug(f"  {current.date()}: THD={thd:.4f}")
        current += timedelta(days=1)

    if len(daily_values) < 10:
        logger.warning(f"Insufficient samples for {network}.{station}: {len(daily_values)}")
        return {
            'station': f'{network}.{station}',
            'mean_thd': None,
            'std_thd': None,
            'n_samples': len(daily_values),
            'calibration_period': f'{start_date.date()} to {end_date.date()}',
            'daily_values': daily_values,
            'error': 'Insufficient samples',
        }

    # Extract THD values
    thd_values = np.array([v[1] for v in daily_values])

    # Robust statistics (median + MAD)
    median_thd = np.median(thd_values)
    mad = np.median(np.abs(thd_values - median_thd))
    robust_std = mad * 1.4826  # MAD to sigma conversion

    # Also compute regular mean/std for comparison
    mean_thd = np.mean(thd_values)
    std_thd = np.std(thd_values)

    logger.info(f"  {network}.{station}: median={median_thd:.4f}, MAD_std={robust_std:.4f}, "
                f"mean={mean_thd:.4f}, std={std_thd:.4f}, n={len(daily_values)}")

    return {
        'station': f'{network}.{station}',
        'mean_thd': round(median_thd, 4),  # Use median as "mean" for robustness
        'std_thd': round(robust_std, 4),   # Use MAD-based std
        'mean_thd_classic': round(mean_thd, 4),
        'std_thd_classic': round(std_thd, 4),
        'n_samples': len(daily_values),
        'calibration_period': f'{start_date.date()} to {end_date.date()}',
        'daily_values': daily_values,
    }


def generate_baseline_code(results: List[Dict]) -> str:
    """Generate Python code for station_baselines.py."""
    lines = [
        "# Auto-generated baseline entries",
        "# Run: python calibrate_thd_baselines.py",
        f"# Generated: {datetime.now().isoformat()}",
        "",
    ]

    for r in results:
        if r.get('mean_thd') is None:
            lines.append(f"# {r['station']}: SKIPPED - {r.get('error', 'unknown error')}")
            continue

        station = r['station']
        lines.append(f'    "{station}": StationBaseline(')
        lines.append(f'        station="{station}",')
        lines.append(f'        mean_thd={r["mean_thd"]},')
        lines.append(f'        std_thd={r["std_thd"]},')
        lines.append(f'        n_samples={r["n_samples"]},')
        lines.append(f'        calibration_period="{r["calibration_period"]}",')
        lines.append(f'        notes="Auto-calibrated. Classic mean={r.get(\"mean_thd_classic\", \"N/A\")}, std={r.get(\"std_thd_classic\", \"N/A\")}",')
        lines.append(f'    ),')
        lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Calibrate THD baselines')
    parser.add_argument('--station', type=str, help='Single station (e.g., IU.TUC)')
    parser.add_argument('--days', type=int, default=60, help='Calibration window in days')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()

    # Determine stations to calibrate
    if args.station:
        stations = [args.station]
    else:
        stations = list(CALIBRATION_STATIONS.keys())

    results = []

    for station_key in stations:
        parts = station_key.split('.')
        if len(parts) != 2:
            logger.error(f"Invalid station format: {station_key}")
            continue

        network, station = parts
        result = calibrate_station(network, station, days_back=args.days)
        results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("CALIBRATION RESULTS")
    print("="*80)
    print(f"{'Station':<15} {'Mean THD':<12} {'Std THD':<12} {'N':<6} {'Status'}")
    print("-"*80)

    for r in results:
        if r.get('mean_thd') is not None:
            print(f"{r['station']:<15} {r['mean_thd']:<12.4f} {r['std_thd']:<12.4f} {r['n_samples']:<6} OK")
        else:
            print(f"{r['station']:<15} {'N/A':<12} {'N/A':<12} {r['n_samples']:<6} {r.get('error', 'FAILED')}")

    # Generate code
    print("\n" + "="*80)
    print("CODE FOR station_baselines.py")
    print("="*80)
    print(generate_baseline_code(results))

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
