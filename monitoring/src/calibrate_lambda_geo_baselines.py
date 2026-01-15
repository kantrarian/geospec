#!/usr/bin/env python3
"""
calibrate_lambda_geo_baselines.py - Compute region-specific Lambda_geo baselines

This script computes Lambda_geo baselines from REAL NGL GPS data for each
monitored region. Baselines are computed from a 90-day quiet period to ensure
consistency with the live computation pipeline.

CRITICAL: This uses the SAME computation as run_ensemble_daily.py to ensure
unit consistency. The baseline and live values will be in identical units.

Per CLAUDE.md data integrity rules:
- All baselines computed from REAL GPS data (NGL IGS20/IGS14)
- No simulated or literature-derived values
- If insufficient data, marked as "no_data" not fabricated

Author: R.J. Mathews / Claude
Date: January 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from live_data import NGLLiveAcquisition, acquire_region_data, RegionLambdaGeo
from regions import FAULT_POLYGONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output paths
BASELINE_DIR = Path(__file__).parent.parent / 'data' / 'baselines'
OUTPUT_FILE = BASELINE_DIR / 'lambda_geo_baselines.json'

# Calibration parameters
BASELINE_DAYS = 90  # Days of data to use for baseline
MIN_VALID_DAYS = 30  # Minimum valid data points required


def compute_region_baseline(
    region_id: str,
    ngl: NGLLiveAcquisition,
    end_date: datetime,
    days_back: int = BASELINE_DAYS
) -> Dict:
    """
    Compute Lambda_geo baseline for a single region from real GPS data.

    Args:
        region_id: Region identifier
        ngl: NGL data acquisition instance
        end_date: End date for baseline period
        days_back: Number of days to include in baseline

    Returns:
        Dict with baseline statistics or error info
    """
    logger.info(f"Computing baseline for {region_id}...")

    # Check if region has polygon definition
    if region_id not in FAULT_POLYGONS:
        return {
            'available': False,
            'reason': f'No polygon definition for {region_id}',
            'data_source': 'no_data'
        }

    # Acquire GPS data and compute Lambda_geo time series
    try:
        result = acquire_region_data(
            region_id=region_id,
            ngl=ngl,
            days_back=days_back,
            target_date=end_date
        )

        if result is None or result.n_stations < 3:
            return {
                'available': False,
                'reason': f'Insufficient stations: {result.n_stations if result else 0} < 3',
                'n_stations': result.n_stations if result else 0,
                'data_source': 'no_data'
            }

        # Get Lambda_geo grid (time series for all triangles)
        lambda_geo_grid = result.lambda_geo_grid  # Shape: (n_times, n_triangles)

        if lambda_geo_grid is None or len(lambda_geo_grid) == 0:
            return {
                'available': False,
                'reason': 'No valid Lambda_geo values computed',
                'n_stations': result.n_stations,
                'data_source': 'no_data'
            }

        # Compute daily max Lambda_geo (across all triangles)
        daily_max = np.nanmax(lambda_geo_grid, axis=1)  # Shape: (n_times,)

        # Filter out NaN and zero values
        valid_values = daily_max[~np.isnan(daily_max) & (daily_max > 0)]

        if len(valid_values) < MIN_VALID_DAYS:
            return {
                'available': False,
                'reason': f'Insufficient valid days: {len(valid_values)} < {MIN_VALID_DAYS}',
                'n_stations': result.n_stations,
                'n_valid_days': len(valid_values),
                'data_source': 'no_data'
            }

        # Compute baseline statistics
        mean_lambda_geo = float(np.mean(valid_values))
        std_lambda_geo = float(np.std(valid_values))
        median_lambda_geo = float(np.median(valid_values))
        max_lambda_geo = float(np.max(valid_values))
        min_lambda_geo = float(np.min(valid_values))

        # Determine quality
        if len(valid_values) >= 60:
            quality = 'good'
        elif len(valid_values) >= MIN_VALID_DAYS:
            quality = 'acceptable'
        else:
            quality = 'poor'

        # Compute calibration period string
        start_date = end_date - timedelta(days=days_back)
        calibration_period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        return {
            'available': True,
            'mean_lambda_geo': mean_lambda_geo,
            'std_lambda_geo': std_lambda_geo,
            'median_lambda_geo': median_lambda_geo,
            'max_lambda_geo': max_lambda_geo,
            'min_lambda_geo': min_lambda_geo,
            'n_samples': len(valid_values),
            'n_stations': result.n_stations,
            'n_triangles': result.n_triangles,
            'calibration_period': calibration_period,
            'quality': quality,
            'data_source': 'real_gps',
            'station_codes': result.station_codes[:10],  # First 10 stations
            'notes': f'Computed from {result.n_stations} NGL GPS stations'
        }

    except Exception as e:
        logger.error(f"Error computing baseline for {region_id}: {e}")
        return {
            'available': False,
            'reason': str(e),
            'data_source': 'error'
        }


def calibrate_all_regions(
    end_date: Optional[datetime] = None,
    days_back: int = BASELINE_DAYS,
    regions: Optional[List[str]] = None
) -> Dict:
    """
    Compute Lambda_geo baselines for all monitored regions.

    Args:
        end_date: End date for baseline period (default: today)
        days_back: Number of days to include in baseline
        regions: List of region IDs (default: all regions with polygons)

    Returns:
        Dict with all baselines and metadata
    """
    if end_date is None:
        end_date = datetime.now()

    if regions is None:
        regions = list(FAULT_POLYGONS.keys())

    logger.info(f"Calibrating Lambda_geo baselines for {len(regions)} regions")
    logger.info(f"Baseline period: {days_back} days ending {end_date.strftime('%Y-%m-%d')}")

    # Initialize NGL acquisition
    cache_dir = Path(__file__).parent.parent / 'data' / 'gps_cache'
    ngl = NGLLiveAcquisition(cache_dir)

    # Load station catalog
    logger.info("Loading NGL station catalog...")
    ngl.load_station_catalog()

    # Compute baselines for each region
    baselines = {}

    for region_id in regions:
        baseline = compute_region_baseline(
            region_id=region_id,
            ngl=ngl,
            end_date=end_date,
            days_back=days_back
        )
        baselines[region_id] = baseline

        if baseline['available']:
            logger.info(f"  {region_id}: mean={baseline['mean_lambda_geo']:.2e}, "
                       f"std={baseline['std_lambda_geo']:.2e}, "
                       f"n={baseline['n_samples']} ({baseline['quality']})")
        else:
            logger.warning(f"  {region_id}: UNAVAILABLE - {baseline.get('reason', 'unknown')}")

    # Create output structure
    output = {
        'calibration_timestamp': datetime.now().isoformat(),
        'baseline_days': days_back,
        'baseline_end_date': end_date.strftime('%Y-%m-%d'),
        'min_valid_days': MIN_VALID_DAYS,
        'data_integrity': 'All baselines computed from real NGL GPS data. No simulated values.',
        'regions': baselines,
        'summary': {
            'total_regions': len(regions),
            'regions_with_baseline': sum(1 for b in baselines.values() if b['available']),
            'regions_without_baseline': sum(1 for b in baselines.values() if not b['available']),
        }
    }

    return output


def save_baselines(baselines: Dict, output_path: Path = OUTPUT_FILE):
    """Save baselines to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2)

    logger.info(f"Saved baselines to {output_path}")


def main():
    """Main entry point for baseline calibration."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute Lambda_geo baselines from real NGL GPS data'
    )
    parser.add_argument(
        '--days', type=int, default=BASELINE_DAYS,
        help=f'Number of days for baseline (default: {BASELINE_DAYS})'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='End date for baseline period (default: today, format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--regions', type=str, nargs='+', default=None,
        help='Specific regions to calibrate (default: all)'
    )
    parser.add_argument(
        '--output', type=str, default=str(OUTPUT_FILE),
        help=f'Output file path (default: {OUTPUT_FILE})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be done without saving'
    )

    args = parser.parse_args()

    # Parse end date
    end_date = None
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Run calibration
    baselines = calibrate_all_regions(
        end_date=end_date,
        days_back=args.days,
        regions=args.regions
    )

    # Print summary
    print("\n" + "=" * 60)
    print("LAMBDA_GEO BASELINE CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Calibration timestamp: {baselines['calibration_timestamp']}")
    print(f"Baseline period: {baselines['baseline_days']} days ending {baselines['baseline_end_date']}")
    print(f"Data integrity: {baselines['data_integrity']}")
    print()

    print("Region Baselines:")
    print("-" * 60)
    for region_id, baseline in baselines['regions'].items():
        if baseline['available']:
            print(f"  {region_id}:")
            print(f"    mean: {baseline['mean_lambda_geo']:.4e}")
            print(f"    std:  {baseline['std_lambda_geo']:.4e}")
            print(f"    n:    {baseline['n_samples']} samples")
            print(f"    quality: {baseline['quality']}")
        else:
            print(f"  {region_id}: NO DATA - {baseline.get('reason', 'unknown')}")

    print()
    print(f"Summary: {baselines['summary']['regions_with_baseline']}/{baselines['summary']['total_regions']} regions calibrated")

    # Save if not dry run
    if not args.dry_run:
        save_baselines(baselines, Path(args.output))
        print(f"\nBaselines saved to: {args.output}")
    else:
        print("\n[DRY RUN - not saving]")

    return baselines


if __name__ == '__main__':
    main()
