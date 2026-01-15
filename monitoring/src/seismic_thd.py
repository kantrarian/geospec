"""
seismic_thd.py
Seismic Total Harmonic Distortion (THD) Analysis for Earthquake Precursors.

Physical Basis:
- Rocks respond LINEARLY to tidal forcing under normal stress conditions
- Near failure, rocks enter NONLINEAR regime as microcracks develop
- Nonlinearity creates HARMONICS (2f₀, 3f₀, ...) in the seismic response
- THD = sqrt(sum of harmonic power / fundamental power)

Earth Tide Frequencies:
- M2: Principal lunar semidiurnal (12.42 hours) = 0.0000224 Hz
- S2: Principal solar semidiurnal (12.00 hours) = 0.0000231 Hz
- K1: Lunar diurnal (23.93 hours) = 0.0000116 Hz
- O1: Principal lunar diurnal (25.82 hours) = 0.0000108 Hz

Expected Pattern:
- Normal (linear elastic): THD < 0.05
- Transitional: 0.05 < THD < 0.15
- Pre-failure (nonlinear): THD > 0.15

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Earth Tide Frequencies (in Hz)
TIDAL_FREQUENCIES = {
    'M2': 1.0 / (12.42 * 3600),   # Principal lunar semidiurnal: 0.0000224 Hz
    'S2': 1.0 / (12.00 * 3600),   # Principal solar semidiurnal: 0.0000231 Hz
    'K1': 1.0 / (23.93 * 3600),   # Lunar diurnal: 0.0000116 Hz
    'O1': 1.0 / (25.82 * 3600),   # Principal lunar diurnal: 0.0000108 Hz
}

# Primary tidal frequency for THD calculation (M2 is strongest)
PRIMARY_TIDAL_FREQ = TIDAL_FREQUENCIES['M2']

# THD Thresholds
THD_THRESHOLDS = {
    'normal': 0.05,      # Linear elastic behavior
    'elevated': 0.10,    # Transitional
    'critical': 0.15,    # Nonlinear pre-failure
}


@dataclass
class THDResult:
    """Container for THD analysis results."""
    station: str
    date: datetime
    thd_value: float
    fundamental_power: float
    harmonic_powers: List[float]  # Powers at 2f, 3f, 4f, ...
    dominant_frequency: float
    is_elevated: bool
    is_critical: bool
    snr: float  # Signal-to-noise ratio
    sample_rate_hz: float = 0.0  # Native sample rate
    baseline_mean: float = 0.0  # Station baseline THD (if available)
    baseline_std: float = 0.0  # Station baseline std dev
    z_score: float = 0.0  # Anomaly score (std devs above baseline)
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'station': self.station,
            'date': self.date.isoformat(),
            'thd_value': float(self.thd_value),
            'fundamental_power': float(self.fundamental_power),
            'harmonic_powers': [float(p) for p in self.harmonic_powers],
            'dominant_frequency': float(self.dominant_frequency),
            'is_elevated': bool(self.is_elevated),
            'is_critical': bool(self.is_critical),
            'snr': float(self.snr),
            'sample_rate_hz': float(self.sample_rate_hz),
            'baseline_mean': float(self.baseline_mean),
            'baseline_std': float(self.baseline_std),
            'z_score': float(self.z_score),
            'notes': self.notes,
        }


class SeismicTHDAnalyzer:
    """
    Analyzes Total Harmonic Distortion in seismic signals relative to tidal forcing.

    High THD indicates nonlinear rock behavior, which increases before failure.

    Attributes:
        fundamental_freq: Primary tidal frequency for analysis (default M2)
        n_harmonics: Number of harmonics to include (default 5: 2f, 3f, 4f, 5f, 6f)
        freq_tolerance: Frequency matching tolerance (fraction of fundamental)
        window_hours: Analysis window size in hours
    """

    def __init__(
        self,
        fundamental_freq: float = PRIMARY_TIDAL_FREQ,
        n_harmonics: int = 5,
        freq_tolerance: float = 0.1,
        window_hours: int = 24,
    ):
        """
        Initialize the THD Analyzer.

        Args:
            fundamental_freq: Fundamental tidal frequency in Hz
            n_harmonics: Number of harmonics to analyze (2f, 3f, ...)
            freq_tolerance: Relative tolerance for frequency matching
            window_hours: Window size for analysis in hours
        """
        self.fundamental_freq = fundamental_freq
        self.n_harmonics = n_harmonics
        self.freq_tolerance = freq_tolerance
        self.window_hours = window_hours

        logger.info(f"SeismicTHDAnalyzer initialized: f0={fundamental_freq:.2e} Hz, "
                   f"harmonics={n_harmonics}, window={window_hours}h")

    def compute_spectrum(
        self,
        data: np.ndarray,
        sample_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of seismic data.

        Args:
            data: Time series data
            sample_rate: Sampling rate in Hz

        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        n = len(data)

        # Remove mean and trend
        data = data - np.mean(data)
        data = signal.detrend(data, type='linear')

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(n)
        data_windowed = data * window

        # Compute FFT
        fft_result = fft(data_windowed)
        freqs = fftfreq(n, d=1.0/sample_rate)

        # Power spectrum (positive frequencies only)
        positive_mask = freqs > 0
        freqs = freqs[positive_mask]
        power = np.abs(fft_result[positive_mask]) ** 2

        # Normalize by window power
        power = power / (np.sum(window ** 2))

        return freqs, power

    def find_spectral_peak(
        self,
        freqs: np.ndarray,
        power: np.ndarray,
        target_freq: float,
        tolerance: Optional[float] = None
    ) -> Tuple[float, float, int]:
        """
        Find spectral peak near target frequency.

        Args:
            freqs: Frequency array
            power: Power spectrum
            target_freq: Target frequency to search near
            tolerance: Frequency tolerance (uses self.freq_tolerance if None)

        Returns:
            Tuple of (peak_frequency, peak_power, peak_index)
        """
        tolerance = tolerance or self.freq_tolerance

        # Find frequency range to search
        freq_min = target_freq * (1 - tolerance)
        freq_max = target_freq * (1 + tolerance)

        # Mask for search region
        mask = (freqs >= freq_min) & (freqs <= freq_max)

        if not np.any(mask):
            return target_freq, 0.0, -1

        # Find peak in region
        search_power = power.copy()
        search_power[~mask] = 0
        peak_idx = np.argmax(search_power)

        return freqs[peak_idx], power[peak_idx], peak_idx

    def get_band_integrated_power(
        self,
        freqs: np.ndarray,
        power: np.ndarray,
        target_freq: float,
        min_bins: int = 3
    ) -> Tuple[float, float]:
        """
        Get band-integrated power around target frequency.

        More robust than peak-finding for coarse frequency resolution.
        For 24h windows at 1Hz: df ≈ 1.16e-5 Hz, which is comparable to
        the tidal frequency itself (~2.24e-5 Hz). Band integration
        ensures we capture the power even if the peak falls between bins.

        Args:
            freqs: Frequency array
            power: Power spectrum
            target_freq: Target frequency
            min_bins: Minimum number of FFT bins for band (default 3)

        Returns:
            Tuple of (integrated_power, band_center_freq)
        """
        if len(freqs) < 2:
            return 0.0, target_freq

        # Frequency resolution
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1e-6

        # Band width: use either tolerance-based or minimum bins, whichever is larger
        tolerance_width = target_freq * self.freq_tolerance * 2
        min_bin_width = df * min_bins

        half_band = max(tolerance_width, min_bin_width) / 2

        # Find band boundaries
        freq_min = target_freq - half_band
        freq_max = target_freq + half_band

        # Mask for integration region
        mask = (freqs >= freq_min) & (freqs <= freq_max)

        if not np.any(mask):
            return 0.0, target_freq

        # Integrate power in band
        integrated_power = np.sum(power[mask])

        # Find band center (power-weighted average frequency)
        masked_freqs = freqs[mask]
        masked_power = power[mask]
        if np.sum(masked_power) > 0:
            band_center = np.sum(masked_freqs * masked_power) / np.sum(masked_power)
        else:
            band_center = target_freq

        return integrated_power, band_center

    def compute_thd(
        self,
        data: np.ndarray,
        sample_rate: float,
        use_band_integration: bool = True
    ) -> Tuple[float, float, List[float], float]:
        """
        Compute Total Harmonic Distortion.

        THD = sqrt(sum(P_n for n=2..N) / P_1)

        where P_n is power at nth harmonic.

        Uses band-integrated power by default for robustness against
        coarse frequency resolution (24h windows at 1Hz have df ≈ 1.16e-5 Hz,
        which is comparable to the tidal frequency itself ~2.24e-5 Hz).

        Args:
            data: Time series data (at least 24 hours recommended)
            sample_rate: Sampling rate in Hz
            use_band_integration: If True, integrate power over bands (more robust).
                                  If False, use peak-finding (legacy behavior).

        Returns:
            Tuple of (thd, fundamental_power, harmonic_powers, dominant_freq)
        """
        if len(data) < sample_rate * 3600 * 12:  # Need at least 12 hours
            logger.warning("Data too short for reliable THD calculation")
            return 0.0, 0.0, [], 0.0

        # Compute spectrum
        freqs, power = self.compute_spectrum(data, sample_rate)

        if use_band_integration:
            # Band-integrated approach (robust for coarse frequency resolution)
            p1, f1 = self.get_band_integrated_power(freqs, power, self.fundamental_freq)

            if p1 <= 0:
                return 0.0, 0.0, [], f1

            # Find harmonic band powers
            harmonic_powers = []
            for n in range(2, self.n_harmonics + 2):  # 2f, 3f, ..., (n+1)f
                target_f = self.fundamental_freq * n
                pn, _ = self.get_band_integrated_power(freqs, power, target_f)
                harmonic_powers.append(pn)
        else:
            # Peak-finding approach (legacy)
            f1, p1, idx1 = self.find_spectral_peak(freqs, power, self.fundamental_freq)

            if p1 <= 0:
                return 0.0, 0.0, [], f1

            # Find harmonic peaks
            harmonic_powers = []
            for n in range(2, self.n_harmonics + 2):  # 2f, 3f, ..., (n+1)f
                target_f = self.fundamental_freq * n
                _, pn, _ = self.find_spectral_peak(freqs, power, target_f)
                harmonic_powers.append(pn)

        # Calculate THD
        # THD = sqrt(sum of harmonic powers / fundamental power)
        sum_harmonic_power = sum(harmonic_powers)

        if p1 > 0:
            thd = np.sqrt(sum_harmonic_power / p1)
        else:
            thd = 0.0

        return thd, p1, harmonic_powers, f1

    def compute_thd_with_noise(
        self,
        data: np.ndarray,
        sample_rate: float,
        use_band_integration: bool = True
    ) -> Tuple[float, float, List[float], float, float]:
        """
        Compute THD with noise estimation for SNR calculation.

        Args:
            data: Time series data
            sample_rate: Sampling rate in Hz
            use_band_integration: If True, use band-integrated power (more robust)

        Returns:
            Tuple of (thd, fundamental_power, harmonic_powers, dominant_freq, snr)
        """
        freqs, power = self.compute_spectrum(data, sample_rate)

        # Get fundamental power (using band integration for consistency)
        if use_band_integration:
            p1, f1 = self.get_band_integrated_power(freqs, power, self.fundamental_freq)
        else:
            f1, p1, _ = self.find_spectral_peak(freqs, power, self.fundamental_freq)

        # Estimate noise floor (median of spectrum away from tidal bands)
        # Exclude regions around tidal frequencies and their harmonics
        noise_mask = np.ones(len(freqs), dtype=bool)
        for n in range(1, self.n_harmonics + 2):
            f_center = self.fundamental_freq * n
            f_min = f_center * 0.7  # Wider exclusion for band integration
            f_max = f_center * 1.3
            noise_mask &= ~((freqs >= f_min) & (freqs <= f_max))

        if np.any(noise_mask):
            noise_floor = np.median(power[noise_mask])
        else:
            noise_floor = np.median(power)

        # SNR (for band integration, compare band power to comparable noise bandwidth)
        if use_band_integration and len(freqs) > 1:
            df = freqs[1] - freqs[0]
            # Estimate how many bins are in our integration band
            band_width = max(self.fundamental_freq * self.freq_tolerance * 2, df * 3)
            n_band_bins = int(band_width / df) + 1
            # Scale noise floor to comparable bandwidth
            noise_power_in_band = noise_floor * n_band_bins
            snr = p1 / noise_power_in_band if noise_power_in_band > 0 else 0.0
        else:
            snr = p1 / noise_floor if noise_floor > 0 else 0.0

        # Compute THD
        thd, p1, harmonic_powers, f1 = self.compute_thd(data, sample_rate, use_band_integration)

        return thd, p1, harmonic_powers, f1, snr

    def analyze_window(
        self,
        data: np.ndarray,
        sample_rate: float,
        station: str,
        window_time: datetime
    ) -> THDResult:
        """
        Analyze a single time window for THD.

        Args:
            data: Time series data for window
            sample_rate: Sampling rate in Hz
            station: Station identifier
            window_time: Center time of window

        Returns:
            THDResult object
        """
        thd, p1, harmonics, f1, snr = self.compute_thd_with_noise(data, sample_rate)

        is_elevated = thd >= THD_THRESHOLDS['elevated']
        is_critical = thd >= THD_THRESHOLDS['critical']

        result = THDResult(
            station=station,
            date=window_time,
            thd_value=thd,
            fundamental_power=p1,
            harmonic_powers=harmonics,
            dominant_frequency=f1,
            is_elevated=is_elevated,
            is_critical=is_critical,
            snr=snr,
        )

        return result

    def compute_thd_timeseries(
        self,
        data: np.ndarray,
        sample_rate: float,
        station: str,
        start_time: datetime,
        window_hours: Optional[int] = None,
        step_hours: int = 6
    ) -> List[THDResult]:
        """
        Compute THD time series over a longer period.

        Args:
            data: Full time series data
            sample_rate: Sampling rate in Hz
            station: Station identifier
            start_time: Start time of data
            window_hours: Window size (uses instance default if None)
            step_hours: Step size between windows

        Returns:
            List of THDResult objects
        """
        window_hours = window_hours or self.window_hours
        samples_per_window = int(window_hours * 3600 * sample_rate)
        samples_per_step = int(step_hours * 3600 * sample_rate)

        results = []
        n_samples = len(data)

        position = 0
        while position + samples_per_window <= n_samples:
            # Extract window
            window_data = data[position:position + samples_per_window]

            # Calculate window center time
            center_samples = position + samples_per_window // 2
            center_seconds = center_samples / sample_rate
            window_time = start_time + timedelta(seconds=center_seconds)

            # Analyze window
            result = self.analyze_window(window_data, sample_rate, station, window_time)
            results.append(result)

            position += samples_per_step

        logger.info(f"Computed {len(results)} THD values for {station}")

        return results


def fetch_continuous_data_for_thd(
    station_network: str,
    station_code: str,
    start: datetime,
    end: datetime,
    channel: str = 'BHZ'
) -> Tuple[Optional[np.ndarray], float]:
    """
    Fetch continuous waveform data for THD analysis.

    Args:
        station_network: Network code (e.g., 'CI')
        station_code: Station code (e.g., 'WBS')
        start: Start datetime
        end: End datetime
        channel: Channel code (default 'BHZ')

    Returns:
        Tuple of (data_array, sample_rate) or (None, 0) on failure
    """
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    # Select appropriate data centers based on network
    clients_to_try = []
    if station_network == 'CI':
        clients_to_try = ['SCEDC', 'IRIS']
    elif station_network in ('BK', 'NC'):
        clients_to_try = ['NCEDC', 'IRIS']
    elif station_network == 'GE':
        # GEOFON network - use GEOFON data center (better availability than IRIS)
        clients_to_try = ['GEOFON', 'GFZ', 'IRIS']
    elif station_network in ('IU', 'II', 'CN', 'UW'):
        clients_to_try = ['IRIS']
    else:
        clients_to_try = ['IRIS', 'GEOFON', 'SCEDC', 'NCEDC']  # Try all

    for client_name in clients_to_try:
        try:
            client = Client(client_name, timeout=120)

            st = client.get_waveforms(
                network=station_network,
                station=station_code,
                location='*',
                channel=channel,
                starttime=UTCDateTime(start),
                endtime=UTCDateTime(end)
            )

            if len(st) > 0:
                # Merge traces
                st.merge(method=1, fill_value='interpolate')

                # Detrend
                st.detrend('demean')
                st.detrend('linear')

                data = st[0].data
                sample_rate = st[0].stats.sampling_rate

                logger.info(f"Retrieved {len(data)} samples from {client_name} "
                           f"({len(data)/sample_rate/3600:.1f} hours)")

                return data, sample_rate

        except Exception as e:
            logger.debug(f"{client_name} failed for {station_network}.{station_code}: {e}")
            continue

    logger.error(f"Could not retrieve data for {station_network}.{station_code}")
    return None, 0.0


def run_thd_test():
    """Quick test of THD calculation with synthetic data."""
    print("=" * 60)
    print("THD Analyzer Test (Synthetic Data)")
    print("=" * 60)

    analyzer = SeismicTHDAnalyzer(
        fundamental_freq=PRIMARY_TIDAL_FREQ,
        n_harmonics=5,
        window_hours=24
    )

    # Create synthetic data with known THD
    sample_rate = 1.0  # 1 Hz (decimated)
    duration_hours = 48
    n_samples = int(duration_hours * 3600 * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Linear case: pure tidal signal
    f0 = PRIMARY_TIDAL_FREQ
    signal_linear = np.sin(2 * np.pi * f0 * t)

    # Nonlinear case: add harmonics
    signal_nonlinear = np.sin(2 * np.pi * f0 * t)
    signal_nonlinear += 0.3 * np.sin(2 * np.pi * 2*f0 * t)  # 2nd harmonic
    signal_nonlinear += 0.2 * np.sin(2 * np.pi * 3*f0 * t)  # 3rd harmonic
    signal_nonlinear += 0.1 * np.sin(2 * np.pi * 4*f0 * t)  # 4th harmonic

    # Add noise
    noise = np.random.normal(0, 0.1, n_samples)
    signal_linear += noise
    signal_nonlinear += noise

    # Test linear signal
    thd_linear, p1_linear, _, _ = analyzer.compute_thd(signal_linear, sample_rate)
    print(f"\nLinear signal (pure tidal):")
    print(f"  THD = {thd_linear:.4f} (expected < 0.05)")

    # Test nonlinear signal
    thd_nonlinear, p1_nonlinear, harmonics, _ = analyzer.compute_thd(signal_nonlinear, sample_rate)
    print(f"\nNonlinear signal (with harmonics):")
    print(f"  THD = {thd_nonlinear:.4f} (expected ~ 0.37)")
    print(f"  Harmonic powers: {[f'{p:.4f}' for p in harmonics[:3]]}")

    # Expected THD for nonlinear: sqrt(0.3^2 + 0.2^2 + 0.1^2) = sqrt(0.14) ≈ 0.374
    expected_thd = np.sqrt(0.3**2 + 0.2**2 + 0.1**2)
    print(f"  Expected THD: {expected_thd:.4f}")

    print("\n" + "=" * 60)
    if thd_linear < 0.1 and thd_nonlinear > 0.2:
        print("TEST PASSED: THD correctly distinguishes linear vs nonlinear")
    else:
        print("TEST NEEDS REVIEW: Check THD calculation")
    print("=" * 60)


if __name__ == "__main__":
    run_thd_test()
