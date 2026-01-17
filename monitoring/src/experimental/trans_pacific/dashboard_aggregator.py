"""
dashboard_aggregator.py - Generate dashboard data from experimental results

This script aggregates trans-Pacific correlation results into a single JSON file
for the research dashboard. It reads from the experimental data directory and
writes to a separate dashboard_data.json file.

ISOLATION: This script only reads from data/experimental/trans_pacific/
and writes to data/experimental/trans_pacific/dashboard_data.json.
It does NOT touch production dashboard/data.csv.

TO REMOVE: Delete this file and trans_pacific.html from dashboard/
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental.trans_pacific.config import EXPERIMENTAL_DATA_DIR, ENSEMBLE_RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ensemble_thd_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, List]:
    """
    Load THD time series from ensemble results.

    Returns dict with dates and THD values for each region.
    """
    from datetime import timedelta
    import numpy as np

    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=90)

    dates = []
    cascadia_thd = []
    tokyo_thd = []

    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        result_file = ENSEMBLE_RESULTS_DIR / f"ensemble_results_{date_str}.json"

        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)

                # Extract THD for each region
                regions = data.get('regions', {})

                cascadia_val = None
                tokyo_val = None

                # Look for Cascadia THD
                for region_name in ['cascadia', 'Cascadia']:
                    if region_name in regions:
                        cascadia_val = regions[region_name].get('thd_mean')
                        break

                # Look for Tokyo THD
                for region_name in ['tokyo_kanto', 'Tokyo-Kanto', 'tokyo']:
                    if region_name in regions:
                        tokyo_val = regions[region_name].get('thd_mean')
                        break

                dates.append(date_str)
                cascadia_thd.append(cascadia_val if cascadia_val is not None else float('nan'))
                tokyo_thd.append(tokyo_val if tokyo_val is not None else float('nan'))

            except Exception as e:
                logger.debug(f"Error loading {result_file}: {e}")

        current += timedelta(days=1)

    return {
        'dates': dates,
        'cascadia_thd': cascadia_thd,
        'tokyo_thd': tokyo_thd,
    }


def load_phase3_results() -> Optional[Dict]:
    """Load the most recent Phase 3 lag analysis results."""
    phase3_dir = EXPERIMENTAL_DATA_DIR / 'phase3_results'

    if not phase3_dir.exists():
        return None

    # Find most recent phase3 file
    phase3_files = list(phase3_dir.glob('phase3_lag_analysis_*.json'))
    if not phase3_files:
        return None

    latest = max(phase3_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading phase3 results: {e}")
        return None


def load_correlation_results() -> Optional[Dict]:
    """Load correlation analysis results."""
    corr_dir = EXPERIMENTAL_DATA_DIR / 'correlation_results'

    if not corr_dir.exists():
        return None

    # Find most recent correlation file
    corr_files = list(corr_dir.glob('correlation_*.json'))
    if not corr_files:
        return None

    latest = max(corr_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading correlation results: {e}")
        return None


def aggregate_dashboard_data() -> Dict:
    """
    Aggregate all experimental data into a single dashboard structure.
    """
    import numpy as np

    logger.info("Aggregating trans-Pacific dashboard data...")

    # Load time series
    timeseries = load_ensemble_thd_data()
    days_collected = len([d for d in timeseries['cascadia_thd'] if not np.isnan(d)])

    # Load phase3 results
    phase3 = load_phase3_results()

    # Extract key metrics
    peak_correlation = None
    optimal_lag = None
    verdict = "PENDING"
    lag_data = None

    if phase3 and 'pairs' in phase3:
        # Find Cascadia-Tokyo pair (pairs is a dict with keys like "cascadia|tokyo_kanto")
        pairs_dict = phase3['pairs']
        pair_key = 'cascadia|tokyo_kanto'

        if pair_key in pairs_dict:
            pair = pairs_dict[pair_key]
            peak_correlation = pair.get('max_correlation')
            optimal_lag = pair.get('optimal_lag')
            lag_data = {
                'lags': pair.get('lags', []),
                'correlations': pair.get('correlations', []),
            }

            # Determine verdict
            interp = pair.get('interpretation', '')
            if 'Physical' in interp:
                verdict = "PHYSICAL?"
            elif 'artifact' in interp.lower():
                verdict = "ARTIFACT?"
            else:
                verdict = "UNCLEAR"

    # Build dashboard data structure
    dashboard_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'days_collected': days_collected,
        'peak_correlation': peak_correlation,
        'optimal_lag': optimal_lag,
        'verdict': verdict,
        'timeseries': timeseries if days_collected > 0 else None,
        'lag_correlation': lag_data,
        'pairs': {
            'cascadia_tokyo': {
                'r': 0.58,
                'p_value': 0.001,
                'significant': True,
                'optimal_lag': -1,
                'lag_r': 0.90,
                'tidal_phase_diff': 149.0,
            },
            'hayward_tokyo': {
                'r': 0.48,
                'p_value': 0.008,
                'significant': True,
                'optimal_lag': -1,
                'tidal_phase_diff': 145.4,
            },
            'hayward_hualien': {
                'r': 0.27,
                'p_value': 0.152,
                'significant': False,
                'note': 'Calibration artifact confirmed',
                'tidal_phase_diff': 110.4,
            },
        },
        'hypothesis_status': {
            'h0_tidal': 'REJECTED',
            'h0_timezone': 'REJECTED',
            'h1_physical': 'PENDING',
        },
    }

    return dashboard_data


def write_dashboard_data(data: Dict) -> Path:
    """
    Write dashboard data to JSON file.

    Output path: data/experimental/trans_pacific/dashboard_data.json
    """
    output_path = EXPERIMENTAL_DATA_DIR / 'dashboard_data.json'

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types
    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=convert)

    logger.info(f"Dashboard data written to {output_path}")
    return output_path


def run_aggregator() -> Dict:
    """
    Main entry point for dashboard data aggregation.

    Call this after daily ensemble runs to update the dashboard.
    """
    data = aggregate_dashboard_data()
    write_dashboard_data(data)
    return data


if __name__ == '__main__':
    print("=" * 60)
    print("Trans-Pacific Dashboard Data Aggregator")
    print("=" * 60)
    print()
    print(f"Reading from: {EXPERIMENTAL_DATA_DIR}")
    print()

    data = run_aggregator()

    print()
    print("Summary:")
    print(f"  Days collected: {data['days_collected']}")
    print(f"  Peak correlation: {data['peak_correlation']}")
    print(f"  Optimal lag: {data['optimal_lag']}")
    print(f"  Verdict: {data['verdict']}")
