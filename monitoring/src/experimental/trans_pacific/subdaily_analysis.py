"""
subdaily_analysis.py - Phase 4: Sub-daily (6-hour) THD Analysis

This module implements 6-hour window THD analysis to distinguish between:
- 17-hour lag (timezone/diurnal noise effect)
- 24-hour lag (physical propagation)

Key Test:
- If lag peaks at -3 bins (18h): Timezone artifact
- If lag peaks at -4 bins (24h): Physical signal (~87 m/s propagation)

The 17-hour timezone difference (Tokyo UTC+9, Cascadia UTC-8) would manifest
as a ~3-bin lag in 6-hour resolution if the correlation is driven by
diurnal noise patterns (e.g., quieter nighttime data).

Reference: docs/TRANS_PACIFIC_CORRELATION_PAPER_SKELETON.md Section 9.5
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np

# Ensure module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental.trans_pacific.config import EXPERIMENTAL_DATA_DIR, ENSEMBLE_RESULTS_DIR

# Try importing ObsPy for seismic data
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False

# Try matplotlib for plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Station configurations for Cascadia and Tokyo
STATIONS = {
    'cascadia': {'network': 'IU', 'station': 'COR', 'channel': 'BHZ'},
    'tokyo_kanto': {'network': 'IU', 'station': 'MAJO', 'channel': 'BHZ'},
}

# THD calculation parameters
M2_FREQ = 1.0 / (12.42 * 3600)  # M2 tidal frequency in Hz
N_HARMONICS = 5
WINDOW_HOURS = 6  # Sub-daily window


def fetch_seismic_data(
    network: str,
    station: str,
    channel: str,
    start: datetime,
    end: datetime,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Fetch seismic data for a time window.

    Returns:
        Tuple of (data array, sample rate) or (None, 0) if failed
    """
    if not OBSPY_AVAILABLE:
        logger.error("ObsPy not available")
        return None, 0.0

    clients_to_try = ['IRIS', 'GEOFON']

    for client_name in clients_to_try:
        try:
            client = Client(client_name, timeout=60)
            st = client.get_waveforms(
                network=network,
                station=station,
                location='*',
                channel=channel,
                starttime=UTCDateTime(start),
                endtime=UTCDateTime(end)
            )

            if len(st) > 0:
                st.merge(method=1, fill_value='interpolate')
                st.detrend('demean')
                st.detrend('linear')

                data = st[0].data
                sample_rate = st[0].stats.sampling_rate

                return data, sample_rate

        except Exception as e:
            logger.debug(f"{client_name} failed: {e}")
            continue

    return None, 0.0


def compute_thd(data: np.ndarray, sample_rate: float) -> float:
    """
    Compute Total Harmonic Distortion for a data segment.

    THD = sqrt(sum(P(n*f0)) / P(f0)) for n = 2 to N

    Args:
        data: Seismic data array
        sample_rate: Sample rate in Hz

    Returns:
        THD value (0-1 range typically)
    """
    if len(data) < sample_rate * 3600:  # Need at least 1 hour
        return np.nan

    # Resample to 1 Hz for efficiency
    if sample_rate > 1:
        factor = int(sample_rate)
        data = data[::factor]
        sample_rate = 1.0

    # Compute FFT
    n = len(data)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power = np.abs(fft) ** 2

    # Find fundamental (M2) and harmonics
    fundamental_idx = np.argmin(np.abs(freqs - M2_FREQ))

    # Sum power at harmonics
    harmonic_power = 0.0
    for h in range(2, N_HARMONICS + 1):
        h_freq = h * M2_FREQ
        h_idx = np.argmin(np.abs(freqs - h_freq))
        if h_idx < len(power):
            harmonic_power += power[h_idx]

    fundamental_power = power[fundamental_idx]

    if fundamental_power > 0:
        thd = np.sqrt(harmonic_power / fundamental_power)
    else:
        thd = np.nan

    return thd


def compute_subdaily_thd_series(
    region: str,
    start_date: datetime,
    end_date: datetime,
    window_hours: int = 6,
) -> Tuple[List[datetime], List[float]]:
    """
    Compute THD at sub-daily (6-hour) resolution.

    Args:
        region: Region key ('cascadia' or 'tokyo_kanto')
        start_date: Start of analysis period
        end_date: End of analysis period
        window_hours: Window size in hours (default 6)

    Returns:
        Tuple of (timestamps, thd_values)
    """
    if region not in STATIONS:
        logger.error(f"Unknown region: {region}")
        return [], []

    config = STATIONS[region]
    timestamps = []
    thd_values = []

    current = start_date
    while current < end_date:
        window_end = current + timedelta(hours=window_hours)

        logger.info(f"Fetching {region} data for {current} to {window_end}")

        data, sample_rate = fetch_seismic_data(
            network=config['network'],
            station=config['station'],
            channel=config['channel'],
            start=current,
            end=window_end,
        )

        if data is not None and len(data) > 0:
            thd = compute_thd(data, sample_rate)
            timestamps.append(current)
            thd_values.append(thd)
            logger.info(f"  THD = {thd:.4f}")
        else:
            timestamps.append(current)
            thd_values.append(np.nan)
            logger.warning(f"  No data available")

        current = window_end

    return timestamps, thd_values


def compute_subdaily_lag_correlation(
    timestamps_a: List[datetime],
    values_a: List[float],
    timestamps_b: List[datetime],
    values_b: List[float],
    max_lag_bins: int = 8,
) -> Dict:
    """
    Compute lag correlation at sub-daily resolution.

    Args:
        timestamps_a: Timestamps for region A
        values_a: THD values for region A
        timestamps_b: Timestamps for region B
        values_b: THD values for region B
        max_lag_bins: Maximum lag in 6-hour bins (8 bins = 48 hours)

    Returns:
        Dict with lag analysis results
    """
    # Align by timestamp
    ts_a = set(timestamps_a)
    ts_b = set(timestamps_b)
    common_ts = sorted(ts_a & ts_b)

    if len(common_ts) < 10:
        return {'error': 'Insufficient common timestamps'}

    val_map_a = dict(zip(timestamps_a, values_a))
    val_map_b = dict(zip(timestamps_b, values_b))

    a = np.array([val_map_a[t] for t in common_ts])
    b = np.array([val_map_b[t] for t in common_ts])

    results = {
        'lags_bins': [],
        'lags_hours': [],
        'correlations': [],
        'n_samples': [],
    }

    for lag in range(-max_lag_bins, max_lag_bins + 1):
        if lag < 0:
            a_slice = a[-lag:]
            b_slice = b[:lag]
        elif lag > 0:
            a_slice = a[:-lag]
            b_slice = b[lag:]
        else:
            a_slice = a
            b_slice = b

        valid = ~(np.isnan(a_slice) | np.isnan(b_slice))
        n_valid = valid.sum()

        if n_valid >= 5:
            r = np.corrcoef(a_slice[valid], b_slice[valid])[0, 1]
        else:
            r = np.nan

        results['lags_bins'].append(lag)
        results['lags_hours'].append(lag * 6)  # 6-hour bins
        results['correlations'].append(r)
        results['n_samples'].append(n_valid)

    # Find optimal lag
    valid_corrs = [(i, c) for i, c in enumerate(results['correlations']) if not np.isnan(c)]
    if valid_corrs:
        best_idx = max(valid_corrs, key=lambda x: abs(x[1]))[0]
        results['optimal_lag_bins'] = results['lags_bins'][best_idx]
        results['optimal_lag_hours'] = results['lags_hours'][best_idx]
        results['max_correlation'] = results['correlations'][best_idx]

        # Interpretation
        opt_hours = results['optimal_lag_hours']
        if abs(opt_hours - 17) <= 3:
            results['interpretation'] = f"Lag ~{opt_hours}h suggests TIMEZONE/DIURNAL effect"
            results['verdict'] = 'TIMEZONE_ARTIFACT'
        elif abs(opt_hours - 24) <= 3:
            results['interpretation'] = f"Lag ~{opt_hours}h suggests PHYSICAL propagation (~87 m/s)"
            results['verdict'] = 'PHYSICAL_SIGNAL'
        else:
            results['interpretation'] = f"Lag {opt_hours}h - interpretation unclear"
            results['verdict'] = 'UNCLEAR'

    return results


def generate_subdaily_lag_plot(
    results: Dict,
    output_path: Path,
    title: str = "Sub-daily (6-hour) Lag Correlation",
) -> bool:
    """Generate lag-correlation plot at 6-hour resolution."""
    if not MATPLOTLIB_AVAILABLE:
        return False

    lags = results['lags_hours']
    corrs = results['correlations']

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(lags, corrs, 'b-', linewidth=2, marker='o', markersize=6)

    # Mark key reference points
    ax.axvline(x=-17, color='orange', linestyle='--', alpha=0.7, label='17h (Timezone diff)')
    ax.axvline(x=-24, color='green', linestyle='--', alpha=0.7, label='24h (Physical)')
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Mark optimal lag
    opt_h = results.get('optimal_lag_hours', 0)
    max_r = results.get('max_correlation', 0)
    ax.scatter([opt_h], [max_r], color='red', s=150, zorder=5, marker='*',
               label=f'Optimal: {opt_h}h (r={max_r:.3f})')

    ax.set_xlabel('Lag (hours)\n[Negative = Tokyo leads Cascadia]', fontsize=12)
    ax.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    interp = results.get('interpretation', '')
    verdict = results.get('verdict', '')
    color = 'red' if verdict == 'TIMEZONE_ARTIFACT' else 'green' if verdict == 'PHYSICAL_SIGNAL' else 'gray'
    ax.text(0.02, 0.98, f"VERDICT: {verdict}\n{interp}", transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sub-daily lag plot to {output_path}")
    return True


def run_phase4_subdaily_test(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run Phase 4 sub-daily analysis.

    This is the critical test to distinguish:
    - 17h lag → Timezone/diurnal artifact
    - 24h lag → Physical propagation (~87 m/s)

    Args:
        start_date: Start of analysis (default: 7 days ago)
        end_date: End of analysis (default: now)
        output_dir: Output directory

    Returns:
        Dict with analysis results
    """
    if not OBSPY_AVAILABLE:
        logger.error("ObsPy required for sub-daily analysis")
        return {'error': 'ObsPy not available'}

    # Defaults
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    if output_dir is None:
        output_dir = EXPERIMENTAL_DATA_DIR / 'phase4_results'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 4: Sub-daily (6-hour) Lag Analysis")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date} UTC")
    logger.info(f"Window: 6 hours")

    # Compute sub-daily THD for both regions
    logger.info("Fetching Cascadia data...")
    ts_cascadia, thd_cascadia = compute_subdaily_thd_series(
        'cascadia', start_date, end_date, window_hours=6
    )

    logger.info("Fetching Tokyo data...")
    ts_tokyo, thd_tokyo = compute_subdaily_thd_series(
        'tokyo_kanto', start_date, end_date, window_hours=6
    )

    # Compute lag correlation
    logger.info("Computing lag correlation...")
    lag_results = compute_subdaily_lag_correlation(
        ts_cascadia, thd_cascadia,
        ts_tokyo, thd_tokyo,
        max_lag_bins=8  # ±48 hours
    )

    # Generate plot
    if MATPLOTLIB_AVAILABLE:
        fig_path = output_dir / 'figures' / 'subdaily_lag_correlation.png'
        generate_subdaily_lag_plot(lag_results, fig_path)

    # Save results
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'date_range': [start_date.isoformat(), end_date.isoformat()],
        'window_hours': 6,
        'cascadia_samples': len([t for t in thd_cascadia if not np.isnan(t)]),
        'tokyo_samples': len([t for t in thd_tokyo if not np.isnan(t)]),
        'lag_analysis': lag_results,
    }

    results_path = output_dir / f"phase4_subdaily_{end_date.strftime('%Y%m%d')}.json"

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    print()
    print("=" * 60)
    print("PHASE 4 RESULTS")
    print("=" * 60)

    if 'error' not in lag_results:
        print(f"Optimal lag: {lag_results.get('optimal_lag_hours', 'N/A')} hours")
        print(f"Max correlation: {lag_results.get('max_correlation', 'N/A'):.4f}")
        print(f"Interpretation: {lag_results.get('interpretation', 'N/A')}")
        print()
        print(f"VERDICT: {lag_results.get('verdict', 'N/A')}")
    else:
        print(f"Error: {lag_results.get('error')}")

    return results


# =============================================================================
# ALTERNATIVE APPROACH: RMS Envelope (Works at 6-hour resolution)
# =============================================================================

def compute_rms(data: np.ndarray) -> float:
    """
    Compute Root Mean Square of a data segment.

    RMS = sqrt(mean(x^2))

    Unlike THD, RMS can be computed in any window size.

    Args:
        data: Seismic data array

    Returns:
        RMS value in same units as input
    """
    if len(data) == 0:
        return np.nan
    return np.sqrt(np.mean(data ** 2))


def compute_subdaily_rms_series(
    region: str,
    start_date: datetime,
    end_date: datetime,
    window_hours: int = 6,
) -> Tuple[List[datetime], List[float]]:
    """
    Compute RMS at sub-daily (6-hour) resolution.

    This is an alternative to THD that works at any window size.

    Args:
        region: Region key ('cascadia' or 'tokyo_kanto')
        start_date: Start of analysis period
        end_date: End of analysis period
        window_hours: Window size in hours (default 6)

    Returns:
        Tuple of (timestamps, rms_values)
    """
    if region not in STATIONS:
        logger.error(f"Unknown region: {region}")
        return [], []

    config = STATIONS[region]
    timestamps = []
    rms_values = []

    current = start_date
    while current < end_date:
        window_end = current + timedelta(hours=window_hours)

        logger.info(f"Fetching {region} data for {current} to {window_end}")

        data, sample_rate = fetch_seismic_data(
            network=config['network'],
            station=config['station'],
            channel=config['channel'],
            start=current,
            end=window_end,
        )

        if data is not None and len(data) > 0:
            # Detrend and compute RMS
            data = data - np.mean(data)
            rms = compute_rms(data)
            timestamps.append(current)
            rms_values.append(rms)
            logger.info(f"  RMS = {rms:.2f}")
        else:
            timestamps.append(current)
            rms_values.append(np.nan)
            logger.warning(f"  No data available")

        current = window_end

    return timestamps, rms_values


def run_phase4_rms_test(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run Phase 4 sub-daily analysis using RMS (alternative to THD).

    RMS can be computed in any window size, unlike THD which requires
    at least one M2 tidal cycle (12.42 hours).

    This test distinguishes:
    - ~17h lag (3 bins) -> Timezone/diurnal artifact
    - ~24h lag (4 bins) -> Physical propagation (~87 m/s)

    Args:
        start_date: Start of analysis (default: 7 days ago)
        end_date: End of analysis (default: now)
        output_dir: Output directory

    Returns:
        Dict with analysis results
    """
    if not OBSPY_AVAILABLE:
        logger.error("ObsPy required for sub-daily analysis")
        return {'error': 'ObsPy not available'}

    # Defaults
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    if output_dir is None:
        output_dir = EXPERIMENTAL_DATA_DIR / 'phase4_results'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 4 ALTERNATIVE: Sub-daily (6-hour) RMS Lag Analysis")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date} UTC")
    logger.info(f"Window: 6 hours")
    logger.info(f"Metric: RMS (root mean square amplitude)")

    # Compute sub-daily RMS for both regions
    logger.info("Fetching Cascadia data...")
    ts_cascadia, rms_cascadia = compute_subdaily_rms_series(
        'cascadia', start_date, end_date, window_hours=6
    )

    logger.info("Fetching Tokyo data...")
    ts_tokyo, rms_tokyo = compute_subdaily_rms_series(
        'tokyo_kanto', start_date, end_date, window_hours=6
    )

    # Compute lag correlation (reuse existing function)
    logger.info("Computing lag correlation...")
    lag_results = compute_subdaily_lag_correlation(
        ts_cascadia, rms_cascadia,
        ts_tokyo, rms_tokyo,
        max_lag_bins=8  # +/- 48 hours
    )

    # Generate plot
    if MATPLOTLIB_AVAILABLE:
        fig_path = output_dir / 'figures' / 'subdaily_rms_lag_correlation.png'
        generate_subdaily_lag_plot(
            lag_results,
            fig_path,
            title="Sub-daily (6-hour) RMS Lag Correlation"
        )

    # Save results
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'date_range': [start_date.isoformat(), end_date.isoformat()],
        'window_hours': 6,
        'metric': 'RMS',
        'cascadia_samples': len([r for r in rms_cascadia if not np.isnan(r)]),
        'tokyo_samples': len([r for r in rms_tokyo if not np.isnan(r)]),
        'lag_analysis': lag_results,
    }

    results_path = output_dir / f"phase4_rms_{end_date.strftime('%Y%m%d')}.json"

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    print()
    print("=" * 60)
    print("PHASE 4 RMS RESULTS")
    print("=" * 60)

    if 'error' not in lag_results:
        opt_hours = lag_results.get('optimal_lag_hours', 'N/A')
        max_r = lag_results.get('max_correlation', 0)
        print(f"Optimal lag: {opt_hours} hours")
        print(f"Max correlation: {max_r:.4f}")
        print(f"Interpretation: {lag_results.get('interpretation', 'N/A')}")
        print()
        print(f"VERDICT: {lag_results.get('verdict', 'N/A')}")
        print()
        print("Key Reference Points:")
        print(f"  - 17h (3 bins): Timezone artifact threshold")
        print(f"  - 24h (4 bins): Physical propagation threshold")
    else:
        print(f"Error: {lag_results.get('error')}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Trans-Pacific Phase 4: Sub-daily Lag Analysis")
    print("=" * 60)
    print()
    print("This test distinguishes:")
    print("  - 17h lag -> Timezone/diurnal artifact (FAIL)")
    print("  - 24h lag -> Physical propagation at ~87 m/s (PASS)")
    print()
    print("NOTE: THD method requires M2 period (12.42h) > window size.")
    print("      Using RMS alternative for 6-hour windows.")
    print()

    # Use RMS method instead of THD
    results = run_phase4_rms_test()
