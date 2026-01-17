"""
backtest_trans_pacific.py - Phase 3: Lag Analysis and Causality Testing

This script performs detailed lag-correlation analysis to determine:
1. Whether correlations are synchronous (common driver) or lagged (causal propagation)
2. Which region leads in each correlated pair
3. Optimal lag time for maximum correlation

Focus Pairs (from Phase 2 findings):
- Cascadia-Tokyo: r=0.58, p<0.001 (PRIMARY - both subduction zones)
- Hayward-Tokyo: r=0.48, p=0.008 (SECONDARY)
- Hayward-Hualien: r=0.27, p=0.15 (CONTROL - not significant)

Reference: docs/TRANS_PACIFIC_CORRELATION_PAPER_SKELETON.md Section 9.2
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np

# Ensure experimental module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental.trans_pacific.config import get_config, EXPERIMENTAL_DATA_DIR
from experimental.trans_pacific.thd_correlation import THDCorrelationAnalyzer, THDTimeSeries
from experimental.trans_pacific.tidal_correction import TidalPhaseCorrector

# Optional: matplotlib for figures
try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_full_lag_correlation(
    series_a: THDTimeSeries,
    series_b: THDTimeSeries,
    max_lag: int = 15,
    min_samples: int = 10,
) -> Dict:
    """
    Compute correlation at all lags from -max_lag to +max_lag.

    Positive lag: series_a leads series_b (A happens first)
    Negative lag: series_b leads series_a (B happens first)

    Args:
        series_a: First region's THD series
        series_b: Second region's THD series
        max_lag: Maximum lag in days
        min_samples: Minimum valid samples for correlation

    Returns:
        Dict with lag analysis results
    """
    # Align by date
    dates_a = set(series_a.dates)
    dates_b = set(series_b.dates)
    common_dates = sorted(dates_a & dates_b)

    if len(common_dates) < min_samples:
        return {
            'error': f'Insufficient common dates: {len(common_dates)}',
            'lags': [],
            'correlations': [],
        }

    date_to_a = dict(zip(series_a.dates, series_a.values))
    date_to_b = dict(zip(series_b.dates, series_b.values))

    values_a = np.array([date_to_a[d] for d in common_dates])
    values_b = np.array([date_to_b[d] for d in common_dates])

    lags = []
    correlations = []
    p_values = []
    n_samples_list = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # B leads A: compare A[|lag|:] with B[:-|lag|]
            a_slice = values_a[-lag:]
            b_slice = values_b[:lag]
        elif lag > 0:
            # A leads B: compare A[:-lag] with B[lag:]
            a_slice = values_a[:-lag]
            b_slice = values_b[lag:]
        else:
            a_slice = values_a
            b_slice = values_b

        # Filter NaN
        valid = ~(np.isnan(a_slice) | np.isnan(b_slice))
        n_valid = valid.sum()

        if n_valid >= min_samples:
            r = np.corrcoef(a_slice[valid], b_slice[valid])[0, 1]

            # Compute p-value
            if abs(r) < 0.9999 and n_valid > 2:
                t_stat = r * np.sqrt((n_valid - 2) / (1 - r**2))
                try:
                    from scipy import stats
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_valid-2))
                except ImportError:
                    p_val = np.nan
            else:
                p_val = 0.0 if abs(r) > 0.9999 else np.nan

            lags.append(lag)
            correlations.append(r)
            p_values.append(p_val)
            n_samples_list.append(n_valid)

    if not correlations:
        return {
            'error': 'No valid correlations computed',
            'lags': [],
            'correlations': [],
        }

    # Find optimal lag (maximum absolute correlation)
    abs_corrs = [abs(c) for c in correlations]
    max_idx = np.argmax(abs_corrs)
    optimal_lag = lags[max_idx]
    max_corr = correlations[max_idx]
    max_p = p_values[max_idx]

    # Zero-lag correlation
    zero_idx = lags.index(0) if 0 in lags else None
    zero_corr = correlations[zero_idx] if zero_idx is not None else np.nan
    zero_p = p_values[zero_idx] if zero_idx is not None else np.nan

    # Interpretation
    if optimal_lag == 0:
        interpretation = "Synchronous - common driver or instantaneous coupling"
    elif optimal_lag > 0:
        interpretation = f"Region A leads by {optimal_lag} days - A may drive B"
    else:
        interpretation = f"Region B leads by {-optimal_lag} days - B may drive A"

    return {
        'region_a': series_a.region,
        'region_b': series_b.region,
        'lags': lags,
        'correlations': correlations,
        'p_values': p_values,
        'n_samples': n_samples_list,
        'optimal_lag': optimal_lag,
        'max_correlation': max_corr,
        'max_p_value': max_p,
        'zero_lag_correlation': zero_corr,
        'zero_lag_p_value': zero_p,
        'interpretation': interpretation,
        'n_common_dates': len(common_dates),
    }


def generate_lag_correlation_plot(
    lag_results: Dict,
    output_path: Path,
    title: Optional[str] = None,
) -> bool:
    """
    Generate lag-correlation plot for paper.

    Args:
        lag_results: Output from compute_full_lag_correlation
        output_path: Path to save figure
        title: Optional custom title

    Returns:
        True if successful
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot generation")
        return False

    lags = lag_results['lags']
    correlations = lag_results['correlations']
    p_values = lag_results.get('p_values', [])
    optimal_lag = lag_results['optimal_lag']
    max_corr = lag_results['max_correlation']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot correlation vs lag
    ax.plot(lags, correlations, 'b-', linewidth=2, marker='o', markersize=4)

    # Highlight significant correlations (p < 0.05)
    if p_values:
        sig_lags = [l for l, p in zip(lags, p_values) if p < 0.05]
        sig_corrs = [c for c, p in zip(correlations, p_values) if p < 0.05]
        ax.scatter(sig_lags, sig_corrs, color='green', s=60, zorder=5, label='p < 0.05')

    # Mark optimal lag
    ax.axvline(x=optimal_lag, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal lag = {optimal_lag} days')
    ax.scatter([optimal_lag], [max_corr], color='red', s=100, zorder=6, marker='*')

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # Labels
    region_a = lag_results.get('region_a', 'Region A')
    region_b = lag_results.get('region_b', 'Region B')

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Lag-Correlation: {region_a} vs {region_b}', fontsize=14)

    ax.set_xlabel('Lag (days)\n[Positive = A leads B, Negative = B leads A]', fontsize=12)
    ax.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax.set_xlim(-max(abs(l) for l in lags) - 1, max(abs(l) for l in lags) + 1)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    interp = lag_results.get('interpretation', '')
    ax.text(0.02, 0.98, interp, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved lag-correlation plot to {output_path}")
    return True


def run_phase3_backtest(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run Phase 3 backtest with full lag analysis.

    Focus pairs:
    - Cascadia-Tokyo (primary)
    - Hayward-Tokyo (secondary)
    - Hayward-Hualien (control)

    Args:
        start_date: Start of analysis period
        end_date: End of analysis period
        output_dir: Directory for output files and figures

    Returns:
        Dict with all lag analysis results
    """
    # Defaults
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=31)
    if output_dir is None:
        output_dir = EXPERIMENTAL_DATA_DIR / 'phase3_results'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Phase 3: Lag Analysis Backtest ===")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Output: {output_dir}")

    # Initialize analyzer
    analyzer = THDCorrelationAnalyzer()

    # Define pairs - focus on North Pacific, Hualien as control
    pairs = [
        ('cascadia', 'tokyo_kanto', 'PRIMARY'),
        ('norcal_hayward', 'tokyo_kanto', 'SECONDARY'),
        ('norcal_hayward', 'hualien', 'CONTROL'),
    ]

    results = {
        'timestamp': datetime.now().isoformat(),
        'date_range': [str(start_date.date()), str(end_date.date())],
        'pairs': {},
    }

    for region_a, region_b, group in pairs:
        logger.info(f"\nAnalyzing {region_a} <-> {region_b} [{group}]")

        # Load time series
        series_a = analyzer.load_thd_timeseries(region_a, start_date, end_date)
        series_b = analyzer.load_thd_timeseries(region_b, start_date, end_date)

        # Compute full lag correlation
        lag_result = compute_full_lag_correlation(series_a, series_b, max_lag=15)
        lag_result['group'] = group

        pair_key = f"{region_a}|{region_b}"
        results['pairs'][pair_key] = lag_result

        # Log summary
        if 'error' not in lag_result:
            logger.info(f"  Zero-lag r: {lag_result['zero_lag_correlation']:.4f} (p={lag_result['zero_lag_p_value']:.4f})")
            logger.info(f"  Optimal lag: {lag_result['optimal_lag']} days")
            logger.info(f"  Max r: {lag_result['max_correlation']:.4f} (p={lag_result['max_p_value']:.4f})")
            logger.info(f"  Interpretation: {lag_result['interpretation']}")

            # Generate plot
            if MATPLOTLIB_AVAILABLE:
                fig_path = output_dir / 'figures' / f"lag_correlation_{region_a}_{region_b}.png"
                title = f"Lag-Correlation: {region_a.replace('_', ' ').title()} vs {region_b.replace('_', ' ').title()} [{group}]"
                generate_lag_correlation_plot(lag_result, fig_path, title)
        else:
            logger.warning(f"  Error: {lag_result['error']}")

    # Summary
    logger.info("\n=== Phase 3 Summary ===")
    for pair_key, res in results['pairs'].items():
        if 'error' not in res:
            sig = "SIGNIFICANT" if res.get('zero_lag_p_value', 1) < 0.05 else "not significant"
            logger.info(f"{pair_key} [{res['group']}]: r={res['zero_lag_correlation']:.3f} ({sig}), optimal_lag={res['optimal_lag']}")

    # Save results
    results_path = output_dir / f"phase3_lag_analysis_{end_date.strftime('%Y%m%d')}.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = json.loads(
        json.dumps(results, default=convert_numpy)
    )

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Trans-Pacific Phase 3: Lag Analysis and Causality Testing")
    print("=" * 60)
    print()

    # Run backtest
    results = run_phase3_backtest()

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)

    # Print key findings
    print("\nKEY FINDINGS:")
    for pair_key, res in results['pairs'].items():
        if 'error' not in res:
            print(f"\n{pair_key} [{res['group']}]:")
            print(f"  Zero-lag correlation: r = {res['zero_lag_correlation']:.4f}")
            print(f"  Optimal lag: {res['optimal_lag']} days (r = {res['max_correlation']:.4f})")
            print(f"  Interpretation: {res['interpretation']}")
