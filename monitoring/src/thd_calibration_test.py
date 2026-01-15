#!/usr/bin/env python3
"""
THD Sample Rate Calibration Test

Verifies whether THD differences between 20Hz and 40Hz stations are due to
sample rate artifacts. Tests by resampling both to a standard rate.

Author: R.J. Mathews
Date: January 2026
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from datetime import datetime, timedelta
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# M2 tidal frequency
M2_FREQ = 1.0 / (12.42 * 3600)  # ~0.0000224 Hz


@dataclass
class THDCalibrationResult:
    """Results from calibration test."""
    station: str
    network: str
    client: str
    native_sample_rate: float
    resampled_rate: float
    native_thd: float
    resampled_thd: float
    n_samples_native: int
    n_samples_resampled: int
    fundamental_power_native: float
    fundamental_power_resampled: float


def compute_thd(data: np.ndarray, sample_rate: float, n_harmonics: int = 5) -> Dict:
    """
    Compute THD with detailed output.

    Returns dict with:
        thd: Total Harmonic Distortion value
        fundamental_power: Power at M2 frequency
        harmonic_powers: List of powers at 2*M2, 3*M2, etc.
        dominant_freq: Frequency with maximum power in tidal band
    """
    n = len(data)

    # Detrend
    data = signal.detrend(data, type='linear')

    # FFT
    freqs = fftfreq(n, 1/sample_rate)
    spectrum = np.abs(fft(data)) ** 2

    # Only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    spectrum = spectrum[pos_mask]

    # Frequency resolution
    df = sample_rate / n

    def get_power_at_freq(target_f: float, bandwidth_factor: float = 0.2) -> float:
        """Get integrated power in band around target frequency."""
        # Use a fixed bandwidth relative to target frequency
        half_bw = target_f * bandwidth_factor
        low = target_f - half_bw
        high = target_f + half_bw
        mask = (freqs >= low) & (freqs <= high)
        if np.sum(mask) > 0:
            return np.sum(spectrum[mask])  # Integrated power, not max
        return 0

    # Fundamental (M2) power
    f_power = get_power_at_freq(M2_FREQ)

    # Harmonic powers
    harmonic_powers = []
    for h in range(2, n_harmonics + 2):
        h_power = get_power_at_freq(h * M2_FREQ)
        harmonic_powers.append(h_power)

    harmonic_sum = sum(harmonic_powers)

    # THD calculation
    if f_power > 0:
        thd = np.sqrt(harmonic_sum / f_power)
    else:
        thd = 0.0

    return {
        'thd': thd,
        'fundamental_power': f_power,
        'harmonic_powers': harmonic_powers,
        'sample_rate': sample_rate,
        'n_samples': n,
        'freq_resolution': df
    }


def resample_to_rate(data: np.ndarray, from_rate: float, to_rate: float) -> np.ndarray:
    """Resample data to target sample rate using scipy.signal.resample."""
    if abs(from_rate - to_rate) < 0.1:
        return data

    # Calculate new number of samples
    duration = len(data) / from_rate
    new_n = int(duration * to_rate)

    # Resample
    resampled = signal.resample(data, new_n)

    return resampled


def test_station(
    client_name: str,
    network: str,
    station: str,
    target_rate: float = 40.0,
    hours: int = 24
) -> Optional[THDCalibrationResult]:
    """
    Test a station's THD at native and resampled rates.
    """
    print(f"\n--- Testing {network}.{station} via {client_name} ---")

    try:
        client = Client(client_name, timeout=120)

        end_time = UTCDateTime() - 3600  # 1 hour ago
        start_time = end_time - 3600 * hours

        st = client.get_waveforms(
            network=network,
            station=station,
            location='*',
            channel='BHZ',
            starttime=start_time,
            endtime=end_time
        )

        if len(st) == 0:
            print("  No data available")
            return None

        # Merge and preprocess
        st.merge(method=1, fill_value='interpolate')
        trace = st[0]
        trace.detrend('demean')
        trace.detrend('linear')

        data = trace.data.astype(np.float64)
        native_rate = trace.stats.sampling_rate

        print(f"  Native: {len(data)} samples at {native_rate} Hz")

        # Compute THD at native rate
        native_result = compute_thd(data, native_rate)
        print(f"  Native THD: {native_result['thd']:.4f}")
        print(f"  Native fundamental power: {native_result['fundamental_power']:.2e}")

        # Resample to target rate
        if abs(native_rate - target_rate) > 0.1:
            resampled_data = resample_to_rate(data, native_rate, target_rate)
            print(f"  Resampled: {len(resampled_data)} samples at {target_rate} Hz")

            # Compute THD at resampled rate
            resampled_result = compute_thd(resampled_data, target_rate)
            print(f"  Resampled THD: {resampled_result['thd']:.4f}")
            print(f"  Resampled fundamental power: {resampled_result['fundamental_power']:.2e}")
        else:
            print(f"  Already at target rate ({target_rate} Hz), no resampling needed")
            resampled_data = data
            resampled_result = native_result

        # Ratio
        if native_result['thd'] > 0:
            ratio = resampled_result['thd'] / native_result['thd']
            print(f"  THD ratio (resampled/native): {ratio:.3f}")

        return THDCalibrationResult(
            station=station,
            network=network,
            client=client_name,
            native_sample_rate=native_rate,
            resampled_rate=target_rate,
            native_thd=native_result['thd'],
            resampled_thd=resampled_result['thd'],
            n_samples_native=len(data),
            n_samples_resampled=len(resampled_data),
            fundamental_power_native=native_result['fundamental_power'],
            fundamental_power_resampled=resampled_result['fundamental_power']
        )

    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_calibration_test():
    """Run full calibration test comparing 20Hz and 40Hz stations."""
    print("=" * 70)
    print("THD Sample Rate Calibration Test")
    print("=" * 70)
    print("\nGoal: Verify if THD differences are due to sample rate artifacts")
    print("Method: Compare native THD vs THD after resampling to 40Hz")

    TARGET_RATE = 40.0  # Standard rate for comparison

    # Stations to test
    stations = [
        ('GEOFON', 'GE', 'ARPR', 'Arapgir, Turkey (20Hz)'),
        ('IRIS', 'IU', 'ANTO', 'Ankara, Turkey (40Hz)'),
        ('GEOFON', 'GE', 'CSS', 'Cyprus (20Hz)'),
        ('IRIS', 'IU', 'TUC', 'Tucson, AZ (40Hz)'),
    ]

    results = []
    for client_name, network, station, description in stations:
        result = test_station(client_name, network, station, TARGET_RATE)
        if result:
            results.append((description, result))

    # Summary table
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Station':<25} {'Native Hz':>10} {'Native THD':>12} {'@40Hz THD':>12} {'Ratio':>8}")
    print("-" * 70)

    for desc, r in results:
        ratio = r.resampled_thd / r.native_thd if r.native_thd > 0 else 0
        print(f"{r.network}.{r.station:<21} {r.native_sample_rate:>10.0f} "
              f"{r.native_thd:>12.4f} {r.resampled_thd:>12.4f} {ratio:>8.3f}")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    # Compare 20Hz stations before/after resampling
    hz20_stations = [(d, r) for d, r in results if r.native_sample_rate < 30]
    hz40_stations = [(d, r) for d, r in results if r.native_sample_rate >= 30]

    if hz20_stations and hz40_stations:
        avg_20hz_native = np.mean([r.native_thd for _, r in hz20_stations])
        avg_20hz_resampled = np.mean([r.resampled_thd for _, r in hz20_stations])
        avg_40hz = np.mean([r.native_thd for _, r in hz40_stations])

        print(f"\nAverage THD (20Hz stations, native): {avg_20hz_native:.4f}")
        print(f"Average THD (20Hz stations, resampled to 40Hz): {avg_20hz_resampled:.4f}")
        print(f"Average THD (40Hz stations): {avg_40hz:.4f}")

        convergence = abs(avg_20hz_resampled - avg_40hz) / avg_40hz if avg_40hz > 0 else float('inf')
        print(f"\nConvergence after resampling: {(1-convergence)*100:.1f}%")

        if convergence < 0.5:
            print("\n** RESAMPLING RESOLVES THE DISCREPANCY **")
            print("   Recommendation: Implement standard 40Hz resampling in THD pipeline")
        else:
            print("\n** RESAMPLING DOES NOT FULLY RESOLVE DISCREPANCY **")
            print("   The difference may be due to:")
            print("   - Station-specific noise characteristics")
            print("   - Local geology differences")
            print("   - Instrument response differences")
            print("   Recommendation: Use station-specific baselines")

    # Specific comparison for Turkey
    print("\n" + "-" * 70)
    print("TURKEY REGION COMPARISON")
    print("-" * 70)

    arpr_result = next((r for d, r in results if r.station == 'ARPR'), None)
    anto_result = next((r for d, r in results if r.station == 'ANTO'), None)

    if arpr_result and anto_result:
        print(f"\nGE.ARPR (Arapgir):")
        print(f"  Native (20Hz): THD = {arpr_result.native_thd:.4f}")
        print(f"  Resampled (40Hz): THD = {arpr_result.resampled_thd:.4f}")

        print(f"\nIU.ANTO (Ankara):")
        print(f"  Native (40Hz): THD = {anto_result.native_thd:.4f}")

        native_ratio = arpr_result.native_thd / anto_result.native_thd if anto_result.native_thd > 0 else 0
        resampled_ratio = arpr_result.resampled_thd / anto_result.native_thd if anto_result.native_thd > 0 else 0

        print(f"\nRatio (ARPR/ANTO):")
        print(f"  Native: {native_ratio:.1f}x")
        print(f"  After resampling: {resampled_ratio:.1f}x")

        if resampled_ratio < 3:
            print("\n  Resampling brings stations into reasonable agreement.")
            print("  GE.ARPR could be used with 40Hz resampling preprocessing.")
        else:
            print("\n  Stations still differ significantly after resampling.")
            print("  Recommend using IU.ANTO as primary for Turkey.")

    return results


if __name__ == "__main__":
    run_calibration_test()
