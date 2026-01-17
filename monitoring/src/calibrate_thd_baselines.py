#!/usr/bin/env python3
"""
calibrate_thd_baselines.py - Compute THD baselines using the actual pipeline

This script computes THD baselines for each station by:
1. Fetching data over a calibration window (default 90 days, exclude recent 7)
2. Computing daily THD using the same SeismicTHDAnalyzer settings
3. Calculating robust statistics (median + MAD -> sigma)
4. Running QA checks (coverage, drift, stability)
5. Outputting results for station_baselines.py

Usage:
    python calibrate_thd_baselines.py                    # All stations
    python calibrate_thd_baselines.py --station IU.TUC   # Single station
    python calibrate_thd_baselines.py --days 90          # 90-day window
    python calibrate_thd_baselines.py --output baselines_20260112.json

Author: R.J. Mathews / Claude
Date: January 2026
Version: 2.0.0 - Added QA checks, drift detection, rolling window
"""

import argparse
import json
import logging
import sys
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
    from seismic_thd import SeismicTHDAnalyzer, fetch_continuous_data_for_thd
    from baseline_qa import compute_baseline_qa, generate_qa_report, BaselineQA
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    raise


# =============================================================================
# STATION CONFIGURATION
# =============================================================================
# Stations to calibrate with metadata

CALIBRATION_STATIONS = {
    'IU.TUC': {
        'region': 'SoCal/Ridgecrest',
        'location': 'Tucson, AZ',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Serves ridgecrest, socal_saf_mojave, socal_saf_coachella',
    },
    'BK.BKS': {
        'region': 'NorCal Hayward',
        'location': 'Berkeley, CA',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Direct Hayward fault coverage',
    },
    'IU.COR': {
        'region': 'Cascadia',
        'location': 'Corvallis, OR',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Cascadia subduction zone',
    },
    'IU.MAJO': {
        'region': 'Tokyo Kanto',
        'location': 'Matsushiro, Japan',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Placeholder until Hi-net integration',
    },
    'IU.ANTO': {
        'region': 'Turkey',
        'location': 'Ankara',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Serves istanbul_marmara, turkey_kahramanmaras',
    },
    'IV.CAFE': {
        'region': 'Campi Flegrei',
        'location': 'Italy',
        'sample_rate': 100,
        'channel': 'HHZ',
        'notes': 'Direct volcanic monitoring',
    },
    'IU.TATO': {
        'region': 'Hualien/Taiwan',
        'location': 'Taipei, Taiwan',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Serves hualien region',
    },
    'IU.COLA': {
        'region': 'Anchorage/Alaska',
        'location': 'College, AK',
        'sample_rate': 40,
        'channel': 'BHZ',
        'notes': 'Fallback for anchorage (primary AK.SSL often unavailable)',
    },
}


# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def compute_daily_thd(
    network: str,
    station: str,
    date: datetime,
    channel: str = 'BHZ',
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute THD for a single day using the production pipeline.

    Args:
        network: Network code (e.g., 'IU')
        station: Station code (e.g., 'TUC')
        date: Date to compute THD for
        channel: Channel code (e.g., 'BHZ')

    Returns:
        Tuple of (thd_value, sample_rate) or (None, None) on failure
    """
    try:
        # Fetch 25 hours of data (same as production)
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=25)

        # Use the same function as production
        data, sample_rate = fetch_continuous_data_for_thd(
            station_network=network,
            station_code=station,
            start=start,
            end=end,
            channel=channel,
        )

        if data is None or len(data) == 0:
            return None, None

        # Initialize analyzer with production settings
        analyzer = SeismicTHDAnalyzer(
            n_harmonics=5,
            freq_tolerance=0.1,
            window_hours=24,
        )

        # Compute THD
        thd, p1, harmonics, f1 = analyzer.compute_thd(data, sample_rate)

        if thd > 0 and p1 > 0:
            return thd, sample_rate

        return None, None

    except Exception as e:
        logger.debug(f"Failed to compute THD for {network}.{station} on {date.date()}: {e}")
        return None, None


def calibrate_station(
    network: str,
    station: str,
    days_back: int = 90,
    exclude_recent_days: int = 7,
    end_date: Optional[datetime] = None,
    channel: str = 'BHZ',
    expected_sample_rate: int = 40,
) -> Dict:
    """
    Calibrate baseline for a single station using rolling window.

    Args:
        network: Network code
        station: Station code
        days_back: Total days to look back
        exclude_recent_days: Exclude recent N days (data latency buffer)
        end_date: End date (default: now - exclude_recent_days)
        channel: Channel code
        expected_sample_rate: Expected native sample rate

    Returns:
        Dictionary with baseline statistics and QA metrics
    """
    # Calculate date range
    if end_date is None:
        end_date = datetime.now() - timedelta(days=exclude_recent_days)

    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Calibrating {network}.{station}")
    logger.info(f"  Window: {start_date.date()} to {end_date.date()} ({days_back} days)")
    logger.info(f"  Excluded recent: {exclude_recent_days} days")

    # Collect daily THD values
    daily_values = []
    sample_rates_seen = []
    current = start_date
    n_attempted = 0

    while current <= end_date:
        n_attempted += 1
        thd, sr = compute_daily_thd(network, station, current, channel)

        if thd is not None:
            daily_values.append((current.strftime('%Y-%m-%d'), thd))
            if sr is not None:
                sample_rates_seen.append(sr)
            logger.debug(f"  {current.date()}: THD={thd:.4f}")
        else:
            logger.debug(f"  {current.date()}: NO DATA")

        current += timedelta(days=1)

    n_valid = len(daily_values)
    logger.info(f"  Retrieved: {n_valid}/{n_attempted} days ({100*n_valid/n_attempted:.1f}%)")

    # Run QA checks
    actual_sample_rate = int(np.median(sample_rates_seen)) if sample_rates_seen else expected_sample_rate
    qa = compute_baseline_qa(
        station=f"{network}.{station}",
        daily_values=daily_values,
        n_days_requested=n_attempted,
        sample_rate_hz=actual_sample_rate,
    )

    # Handle insufficient data
    if n_valid < 10:
        logger.warning(f"  FAILED: Insufficient samples ({n_valid})")
        return {
            'station': f'{network}.{station}',
            'mean_thd': None,
            'std_thd': None,
            'n_samples': n_valid,
            'calibration_period': f'{start_date.date()} to {end_date.date()}',
            'daily_values': daily_values,
            'qa': qa.to_dict(),
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

    # Percentile statistics for threshold calibration
    p50 = np.percentile(thd_values, 50)
    p90 = np.percentile(thd_values, 90)
    p95 = np.percentile(thd_values, 95)
    p99 = np.percentile(thd_values, 99)

    logger.info(f"  Results:")
    logger.info(f"    Median THD: {median_thd:.4f}")
    logger.info(f"    MAD-std:    {robust_std:.4f}")
    logger.info(f"    P90/P95/P99: {p90:.4f}/{p95:.4f}/{p99:.4f}")
    logger.info(f"    QA Grade:   {qa.quality_grade}")
    if qa.issues:
        for issue in qa.issues:
            logger.warning(f"    QA Issue: {issue}")

    return {
        'station': f'{network}.{station}',
        'mean_thd': round(median_thd, 6),        # Use median as "mean" for robustness
        'std_thd': round(robust_std, 6),         # Use MAD-based std
        'mean_thd_classic': round(mean_thd, 6),
        'std_thd_classic': round(std_thd, 6),
        'percentiles': {
            'p50': round(p50, 6),
            'p90': round(p90, 6),
            'p95': round(p95, 6),
            'p99': round(p99, 6),
        },
        'n_samples': n_valid,
        'n_attempted': n_attempted,
        'coverage_pct': round(100 * n_valid / n_attempted, 1),
        'sample_rate_hz': actual_sample_rate,
        'calibration_period': f'{start_date.date()} to {end_date.date()}',
        'exclude_recent_days': exclude_recent_days,
        'daily_values': daily_values,
        'qa': qa.to_dict(),
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_baseline_code(results: List[Dict]) -> str:
    """Generate Python code for station_baselines.py."""
    lines = [
        "# Auto-generated baseline entries",
        "# Run: python calibrate_thd_baselines.py --days 90",
        f"# Generated: {datetime.now().isoformat()}",
        "#",
        "# These baselines use:",
        "#   - Robust statistics (median + MAD*1.4826)",
        "#   - 90-day rolling window",
        "#   - 7-day recent exclusion (data latency)",
        "",
    ]

    for r in results:
        if r.get('mean_thd') is None:
            lines.append(f"# {r['station']}: SKIPPED - {r.get('error', 'unknown error')}")
            continue

        station = r['station']
        qa_grade = r.get('qa', {}).get('quality_grade', 'unknown')

        lines.append(f'    "{station}": StationBaseline(')
        lines.append(f'        station="{station}",')
        lines.append(f'        mean_thd={r["mean_thd"]},')
        lines.append(f'        std_thd={r["std_thd"]},')
        lines.append(f'        n_samples={r["n_samples"]},')
        lines.append(f'        calibration_period="{r["calibration_period"]}",')
        lines.append(f'        notes="Auto-calibrated. QA={qa_grade}. '
                    f'Classic mean={r.get("mean_thd_classic", "N/A"):.4f}, '
                    f'std={r.get("std_thd_classic", "N/A"):.4f}. '
                    f'P95={r.get("percentiles", {}).get("p95", "N/A"):.4f}",')
        lines.append(f'    ),')
        lines.append('')

    return '\n'.join(lines)


def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_json_output(results: List[Dict], qa_results: List[BaselineQA]) -> Dict:
    """Generate JSON output with full metadata."""
    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'version': '2.0.0',
            'pipeline_lockfile': 'monitoring/config/pipeline_lockfile.json',
        },
        'summary': {
            'n_stations': len(results),
            'n_successful': sum(1 for r in results if r.get('mean_thd') is not None),
            'n_good_qa': sum(1 for qa in qa_results if qa.quality_grade == 'good'),
            'n_acceptable_qa': sum(1 for qa in qa_results if qa.quality_grade == 'acceptable'),
            'n_poor_qa': sum(1 for qa in qa_results if qa.quality_grade == 'poor'),
            'n_fail_qa': sum(1 for qa in qa_results if qa.quality_grade == 'fail'),
        },
        'baselines': results,
    }
    # Convert numpy types to native Python types
    return _convert_numpy_types(output)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calibrate THD baselines with QA checks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Calibrate all stations with default 90-day window
    python calibrate_thd_baselines.py

    # Calibrate single station
    python calibrate_thd_baselines.py --station IU.TUC

    # Use custom window
    python calibrate_thd_baselines.py --days 60 --exclude-recent 3

    # Save output to JSON
    python calibrate_thd_baselines.py --output thd_baselines_20260112.json
        """
    )
    parser.add_argument('--station', type=str, help='Single station (e.g., IU.TUC)')
    parser.add_argument('--days', type=int, default=90, help='Calibration window in days (default: 90)')
    parser.add_argument('--exclude-recent', type=int, default=7,
                       help='Exclude recent N days for data latency (default: 7)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine stations to calibrate
    if args.station:
        if args.station not in CALIBRATION_STATIONS:
            logger.error(f"Unknown station: {args.station}")
            logger.info(f"Available stations: {list(CALIBRATION_STATIONS.keys())}")
            sys.exit(1)
        stations = [args.station]
    else:
        stations = list(CALIBRATION_STATIONS.keys())

    logger.info(f"Calibrating {len(stations)} station(s) with {args.days}-day window")

    results = []
    qa_results = []

    for station_key in stations:
        parts = station_key.split('.')
        if len(parts) != 2:
            logger.error(f"Invalid station format: {station_key}")
            continue

        network, station = parts
        config = CALIBRATION_STATIONS[station_key]

        result = calibrate_station(
            network=network,
            station=station,
            days_back=args.days,
            exclude_recent_days=args.exclude_recent,
            channel=config.get('channel', 'BHZ'),
            expected_sample_rate=config.get('sample_rate', 40),
        )
        results.append(result)

        # Collect QA results
        if result.get('qa'):
            qa_obj = BaselineQA(
                station=result['station'],
                n_days_requested=result.get('n_attempted', args.days),
                n_days_valid=result.get('n_samples', 0),
                coverage_pct=result.get('coverage_pct', 0),
                cv_ratio=result['qa'].get('cv_ratio', 0),
                cv_flag=result['qa'].get('cv_flag', 'unknown'),
                drift_detected=result['qa'].get('drift_detected', False),
                drift_sigma=result['qa'].get('drift_sigma', 0),
                mad_inflation_detected=result['qa'].get('mad_inflation_detected', False),
                mad_inflation_pct=result['qa'].get('mad_inflation_pct', 0),
                sample_rate_hz=result.get('sample_rate_hz', 40),
                quality_grade=result['qa'].get('quality_grade', 'unknown'),
                issues=result['qa'].get('issues', []),
            )
            qa_results.append(qa_obj)

    # Print summary
    print("\n" + "="*80)
    print("CALIBRATION RESULTS")
    print("="*80)
    print(f"{'Station':<12} {'Mean THD':>10} {'Std THD':>10} {'N':>5} {'Coverage':>10} {'QA Grade':<12}")
    print("-"*80)

    for r in results:
        if r.get('mean_thd') is not None:
            qa_grade = r.get('qa', {}).get('quality_grade', '?')
            print(f"{r['station']:<12} {r['mean_thd']:>10.4f} {r['std_thd']:>10.4f} "
                  f"{r['n_samples']:>5} {r.get('coverage_pct', 0):>9.1f}% {qa_grade:<12}")
        else:
            print(f"{r['station']:<12} {'N/A':>10} {'N/A':>10} "
                  f"{r.get('n_samples', 0):>5} {'N/A':>10} {'FAILED':<12}")

    # Print QA report
    if qa_results:
        print("\n")
        print(generate_qa_report(qa_results))

    # Generate Python code
    print("\n" + "="*80)
    print("CODE FOR station_baselines.py")
    print("="*80)
    print(generate_baseline_code(results))

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_data = generate_json_output(results, qa_results)

        # Remove daily_values for compact output (large!)
        for baseline in output_data['baselines']:
            if 'daily_values' in baseline:
                # Keep just last 10 for inspection
                baseline['daily_values_sample'] = baseline['daily_values'][-10:]
                del baseline['daily_values']

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Exit with error if any failures
    n_failed = sum(1 for r in results if r.get('mean_thd') is None)
    n_poor = sum(1 for qa in qa_results if qa.quality_grade in ('poor', 'fail'))

    if n_failed > 0:
        logger.warning(f"{n_failed} station(s) failed calibration")
    if n_poor > 0:
        logger.warning(f"{n_poor} station(s) have poor/fail QA grade")

    return 0 if n_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
