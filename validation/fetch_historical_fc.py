#!/usr/bin/env python3
"""
fetch_historical_fc.py
Compute Fault Correlation (L2/L1) metrics for historical earthquake backtests.

This script fetches continuous seismic waveform data from IRIS/FDSN and computes
cross-correlation metrics across fault segments to derive L2/L1 ratios.

Methodology:
1. Download waveforms for stations on either side of fault segments
2. Apply bandpass filtering (0.01-1.0 Hz) and Hilbert transform for envelopes
3. Compute cross-correlation matrix between station pairs
4. Apply SVD to correlation matrix
5. Calculate L2/L1 = second eigenvalue / first eigenvalue

Low L2/L1 (<0.2) indicates decorrelation, suggesting stress transfer and
potential precursory activity.

Usage:
    python fetch_historical_fc.py --event tohoku_2011
    python fetch_historical_fc.py --all
    python fetch_historical_fc.py --list

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
from scipy import signal
from scipy.linalg import svd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fault segment configurations for historical events
# Each segment needs at least 2 stations on opposite sides of the fault
FAULT_SEGMENTS = {
    'tohoku_2011': {
        'name': 'Japan Trench',
        'event_date': datetime(2011, 3, 11, 5, 46, 24),
        'magnitude': 9.0,
        'lead_days': 14,
        'segments': [
            {
                'name': 'tohoku_north',
                'stations': [
                    # Stations from F-net/Hi-net straddling the fault
                    {'network': 'IU', 'code': 'MAJO', 'lat': 36.54, 'lon': 138.20},  # inland
                    {'network': 'II', 'code': 'ERM', 'lat': 42.02, 'lon': 143.16},   # coastal
                ]
            },
            {
                'name': 'tohoku_south',
                'stations': [
                    {'network': 'IU', 'code': 'MAJO', 'lat': 36.54, 'lon': 138.20},
                    {'network': 'PS', 'code': 'TSK', 'lat': 36.21, 'lon': 140.11},   # Tsukuba
                ]
            }
        ],
        'notes': 'Limited to IU/II stations. Hi-net would provide better coverage but requires account.'
    },
    'turkey_2023': {
        'name': 'East Anatolian Fault',
        'event_date': datetime(2023, 2, 6, 1, 17, 35),
        'magnitude': 7.8,
        'lead_days': 14,
        'segments': [
            {
                'name': 'eaf_north',
                'stations': [
                    {'network': 'IU', 'code': 'ANTO', 'lat': 39.87, 'lon': 32.79},  # Ankara
                    {'network': 'GE', 'code': 'ISP', 'lat': 37.82, 'lon': 30.51},   # Isparta
                ]
            },
            {
                'name': 'eaf_south',
                'stations': [
                    {'network': 'GE', 'code': 'ISP', 'lat': 37.82, 'lon': 30.51},
                    {'network': 'GE', 'code': 'CSS', 'lat': 34.96, 'lon': 33.33},   # Cyprus
                ]
            }
        ],
        'notes': 'Using GEOFON (GE) network. KOERI (KO) would be better but may be restricted.'
    },
    'chile_2010': {
        'name': 'Nazca-South American Subduction',
        'event_date': datetime(2010, 2, 27, 6, 34, 14),
        'magnitude': 8.8,
        'lead_days': 14,
        'segments': [
            {
                'name': 'maule_north',
                'stations': [
                    {'network': 'IU', 'code': 'LVC', 'lat': -22.62, 'lon': -68.91},  # Limon Verde
                    {'network': 'C', 'code': 'GO01', 'lat': -33.45, 'lon': -70.66},  # Santiago area
                ]
            }
        ],
        'notes': 'Limited pre-2010 station coverage. CSN network expanded after this earthquake.'
    },
    'ridgecrest_2019': {
        'name': 'Eastern California Shear Zone',
        'event_date': datetime(2019, 7, 6, 3, 19, 53),
        'magnitude': 7.1,
        'lead_days': 14,
        'segments': [
            {
                'name': 'ridgecrest_main',
                'stations': [
                    {'network': 'CI', 'code': 'WBS', 'lat': 35.98, 'lon': -117.82},
                    {'network': 'CI', 'code': 'CCC', 'lat': 35.52, 'lon': -117.36},
                    {'network': 'CI', 'code': 'SLA', 'lat': 35.89, 'lon': -117.28},
                ]
            },
            {
                'name': 'garlock_junction',
                'stations': [
                    {'network': 'CI', 'code': 'CLC', 'lat': 35.82, 'lon': -117.60},
                    {'network': 'CI', 'code': 'JRC2', 'lat': 35.98, 'lon': -117.81},
                ]
            }
        ],
        'notes': 'Best coverage - dense CI network in region. Already have cached FC data.'
    }
}


def fetch_waveforms(
    network: str,
    station: str,
    start: datetime,
    end: datetime,
    channel: str = 'BHZ'
) -> Tuple[Optional[np.ndarray], float]:
    """
    Fetch continuous waveform data from FDSN services.

    Returns:
        Tuple of (data_array, sample_rate) or (None, 0) on failure
    """
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    # Select data center based on network
    clients_to_try = []
    if network == 'CI':
        clients_to_try = ['SCEDC', 'IRIS']
    elif network in ('NC', 'BK'):
        clients_to_try = ['NCEDC', 'IRIS']
    elif network == 'GE':
        clients_to_try = ['GEOFON', 'GFZ', 'IRIS']
    elif network in ('IU', 'II', 'C', 'PS'):
        clients_to_try = ['IRIS']
    else:
        clients_to_try = ['IRIS', 'GEOFON']

    for client_name in clients_to_try:
        try:
            client = Client(client_name, timeout=120)

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

                logger.info(f"Retrieved {len(data)} samples from {client_name} "
                           f"for {network}.{station}")
                return data, sample_rate

        except Exception as e:
            logger.debug(f"{client_name} failed for {network}.{station}: {e}")
            continue

    logger.warning(f"Could not retrieve data for {network}.{station}")
    return None, 0.0


def compute_envelope(data: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Compute envelope of seismic signal using Hilbert transform.

    Steps:
    1. Bandpass filter 0.01-1.0 Hz
    2. Apply Hilbert transform
    3. Take absolute value for envelope
    """
    # Design bandpass filter
    nyquist = sample_rate / 2
    low = 0.01 / nyquist
    high = min(1.0 / nyquist, 0.95)  # Ensure high < 1

    if low >= high:
        logger.warning("Invalid filter bounds, skipping filter")
        filtered = data
    else:
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
        except Exception as e:
            logger.warning(f"Filter failed: {e}, using raw data")
            filtered = data

    # Hilbert transform for envelope
    analytic_signal = signal.hilbert(filtered)
    envelope = np.abs(analytic_signal)

    return envelope


def compute_correlation_matrix(
    envelopes: List[np.ndarray],
    window_samples: int,
    step_samples: int
) -> np.ndarray:
    """
    Compute correlation matrix between station envelopes.

    For each time window, compute pairwise correlations between all stations.
    Returns the average correlation matrix.
    """
    n_stations = len(envelopes)
    min_len = min(len(env) for env in envelopes)

    # Truncate all to same length
    envelopes = [env[:min_len] for env in envelopes]

    n_windows = (min_len - window_samples) // step_samples + 1
    if n_windows < 1:
        logger.warning("Not enough data for correlation windows")
        return np.eye(n_stations)

    correlation_matrices = []

    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples

        # Extract windows
        windows = [env[start_idx:end_idx] for env in envelopes]

        # Compute correlation matrix for this window
        R = np.zeros((n_stations, n_stations))
        for j in range(n_stations):
            for k in range(n_stations):
                if j == k:
                    R[j, k] = 1.0
                else:
                    # Pearson correlation
                    cov = np.cov(windows[j], windows[k])
                    if cov[0, 0] > 0 and cov[1, 1] > 0:
                        R[j, k] = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                    else:
                        R[j, k] = 0.0

        correlation_matrices.append(R)

    # Average correlation matrix
    avg_R = np.mean(correlation_matrices, axis=0)
    return avg_R


def compute_l2_l1_ratio(correlation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute L2/L1 ratio from correlation matrix using SVD.

    Returns:
        Tuple of (l2_l1_ratio, lambda1, lambda2)
    """
    try:
        # SVD decomposition
        U, S, Vh = svd(correlation_matrix)

        # Eigenvalues are singular values squared (for symmetric matrices)
        eigenvalues = S ** 2

        if len(eigenvalues) >= 2:
            lambda1 = eigenvalues[0]
            lambda2 = eigenvalues[1]

            if lambda1 > 0:
                l2_l1 = lambda2 / lambda1
            else:
                l2_l1 = 0.0

            return l2_l1, lambda1, lambda2
        else:
            return 1.0, eigenvalues[0] if len(eigenvalues) > 0 else 0.0, 0.0

    except Exception as e:
        logger.error(f"SVD failed: {e}")
        return 1.0, 0.0, 0.0


def compute_fc_timeseries(
    event_key: str,
    window_hours: int = 24,
    step_hours: int = 6,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Compute FC (L2/L1) time series for a historical event.
    """
    if event_key not in FAULT_SEGMENTS:
        print(f"ERROR: Unknown event '{event_key}'")
        return None

    config = FAULT_SEGMENTS[event_key]
    event_date = config['event_date']
    lead_days = config['lead_days']

    if verbose:
        print("=" * 70)
        print(f"COMPUTING FC DATA: {config['name']}")
        print("=" * 70)
        print(f"Event: M{config['magnitude']} on {event_date.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"Segments: {len(config['segments'])}")
        print("-" * 70)

    # Time window
    start_time = event_date - timedelta(days=lead_days)
    end_time = event_date + timedelta(hours=12)

    all_timeseries = []

    # Process each segment
    for segment in config['segments']:
        segment_name = segment['name']
        stations = segment['stations']

        if verbose:
            print(f"\nProcessing segment: {segment_name}")
            station_list = [f"{s['network']}.{s['code']}" for s in stations]
            print(f"  Stations: {station_list}")

        # Fetch waveforms for all stations in segment
        envelopes = []
        sample_rates = []
        station_names = []

        for station_info in stations:
            network = station_info['network']
            code = station_info['code']

            data, sr = fetch_waveforms(network, code, start_time, end_time)

            if data is not None and len(data) > 0:
                # Compute envelope
                env = compute_envelope(data, sr)
                envelopes.append(env)
                sample_rates.append(sr)
                station_names.append(f"{network}.{code}")
                if verbose:
                    print(f"    {network}.{code}: {len(data)} samples OK")
            else:
                if verbose:
                    print(f"    {network}.{code}: FAILED")

        if len(envelopes) < 2:
            if verbose:
                print(f"  SKIP: Need at least 2 stations, got {len(envelopes)}")
            continue

        # Use minimum sample rate
        min_sr = min(sample_rates)

        # Resample all to same rate if needed
        resampled_envelopes = []
        for env, sr in zip(envelopes, sample_rates):
            if sr != min_sr:
                factor = int(sr / min_sr)
                env = signal.decimate(env, factor, zero_phase=True)
            resampled_envelopes.append(env)

        # Compute correlation in sliding windows
        window_samples = int(window_hours * 3600 * min_sr)
        step_samples = int(step_hours * 3600 * min_sr)

        min_len = min(len(env) for env in resampled_envelopes)
        n_windows = (min_len - window_samples) // step_samples + 1

        if verbose:
            print(f"  Computing L2/L1 for {n_windows} windows...")

        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples

            # Extract windows
            windows = [env[start_idx:end_idx] for env in resampled_envelopes]

            # Compute correlation matrix
            R = np.zeros((len(windows), len(windows)))
            for j in range(len(windows)):
                for k in range(len(windows)):
                    if j == k:
                        R[j, k] = 1.0
                    else:
                        corr = np.corrcoef(windows[j], windows[k])[0, 1]
                        R[j, k] = corr if not np.isnan(corr) else 0.0

            # Compute L2/L1
            l2_l1, lambda1, lambda2 = compute_l2_l1_ratio(R)

            # Calculate timestamp
            window_start_sec = start_idx / min_sr
            window_time = start_time + timedelta(seconds=window_start_sec)
            days_before = (event_date - window_time).total_seconds() / 86400

            all_timeseries.append({
                'date': window_time.strftime('%Y-%m-%d'),
                'datetime': window_time.isoformat(),
                'days_before_event': round(days_before, 2),
                'segment': segment_name,
                'l2_l1': round(l2_l1, 4),
                'lambda1': round(lambda1, 4),
                'lambda2': round(lambda2, 4),
                'stations': station_names,
                'tier': 'CRITICAL' if l2_l1 < 0.08 else ('ELEVATED' if l2_l1 < 0.15 else 'NORMAL')
            })

    if not all_timeseries:
        if verbose:
            print("\nERROR: No FC data computed (insufficient station coverage)")
        return None

    # Sort by time
    all_timeseries.sort(key=lambda x: x['datetime'])

    # Calculate statistics
    l2_l1_values = [ts['l2_l1'] for ts in all_timeseries]
    baseline_values = [ts['l2_l1'] for ts in all_timeseries
                       if ts['days_before_event'] > 7]

    if baseline_values:
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
    else:
        baseline_mean = np.mean(l2_l1_values)
        baseline_std = np.std(l2_l1_values)

    min_l2_l1 = min(l2_l1_values)
    min_ts = min(all_timeseries, key=lambda x: x['l2_l1'])

    output = {
        'event_key': event_key,
        'event': {
            'name': config['name'],
            'date': event_date.isoformat(),
            'magnitude': config['magnitude'],
        },
        'data_source': {
            'segments': [s['name'] for s in config['segments']],
            'fetch_date': datetime.now().isoformat(),
            'window_hours': window_hours,
            'step_hours': step_hours,
        },
        'statistics': {
            'baseline_mean': round(baseline_mean, 4),
            'baseline_std': round(baseline_std, 4),
            'min_l2_l1': round(min_l2_l1, 4),
            'min_date': min_ts['datetime'],
            'n_elevated': sum(1 for ts in all_timeseries if ts['tier'] == 'ELEVATED'),
            'n_critical': sum(1 for ts in all_timeseries if ts['tier'] == 'CRITICAL'),
        },
        'timeseries': all_timeseries,
        'notes': config['notes']
    }

    if verbose:
        print("\n" + "-" * 70)
        print("FC SUMMARY")
        print("-" * 70)
        print(f"Total windows: {len(all_timeseries)}")
        print(f"Baseline L2/L1: {baseline_mean:.4f} +/- {baseline_std:.4f}")
        print(f"Minimum L2/L1: {min_l2_l1:.4f} on {min_ts['date']}")
        print(f"Elevated readings: {output['statistics']['n_elevated']}")
        print(f"Critical readings: {output['statistics']['n_critical']}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Compute FC (L2/L1) metrics for historical earthquakes'
    )
    parser.add_argument('--event', type=str, help='Specific event to process')
    parser.add_argument('--all', action='store_true', help='Process all events')
    parser.add_argument('--list', action='store_true', help='List available events')
    parser.add_argument('--window', type=int, default=24, help='Window hours (default: 24)')
    parser.add_argument('--step', type=int, default=6, help='Step hours (default: 6)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    if args.list:
        print("Available events for FC computation:")
        print("-" * 60)
        for key, config in FAULT_SEGMENTS.items():
            n_segments = len(config['segments'])
            print(f"  {key:20s} M{config['magnitude']} {config['name']} ({n_segments} segments)")
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'results' / 'fc_historical'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.event:
        result = compute_fc_timeseries(args.event, window_hours=args.window, step_hours=args.step)
        if result:
            results[args.event] = result
            output_file = output_dir / f'{args.event}_fc.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {output_file}")

    elif args.all:
        for event_key in FAULT_SEGMENTS:
            print(f"\n{'#' * 70}")
            print(f"# Processing: {event_key}")
            print(f"{'#' * 70}\n")

            result = compute_fc_timeseries(event_key, window_hours=args.window, step_hours=args.step)
            if result:
                results[event_key] = result
            else:
                print(f"WARNING: Could not compute FC for {event_key}")

        if results:
            output_file = output_dir / 'all_historical_fc.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved all results: {output_file}")

    else:
        parser.print_help()
        return

    # Summary
    print("\n" + "=" * 70)
    print("FC FETCH SUMMARY")
    print("=" * 70)
    print(f"Events processed: {len(results)}")
    for key, data in results.items():
        min_l2l1 = data['statistics']['min_l2_l1']
        n_critical = data['statistics']['n_critical']
        print(f"  {key}: Min L2/L1={min_l2l1:.4f}, Critical readings={n_critical}")


if __name__ == '__main__':
    main()
