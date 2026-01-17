"""
thd_correlation.py - Cross-Region THD Correlation Analysis

This module computes Total Harmonic Distortion (THD) correlations between
geographically separated monitoring regions to test hypotheses about
trans-Pacific tectonic stress coupling.

Key Functions:
- Load THD time series from ensemble results
- Compute Pearson correlation between region pairs
- Apply lag analysis to detect delayed coupling
- Integrate with tidal phase correction for H0 testing

Reference: docs/TRANS_PACIFIC_CORRELATION_PAPER_SKELETON.md
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np

from .config import get_config, ENSEMBLE_RESULTS_DIR
from .tidal_correction import TidalPhaseCorrector, TidalCorrectionResult

logger = logging.getLogger(__name__)


@dataclass
class THDTimeSeries:
    """THD time series for a single region."""
    region: str
    dates: List[datetime]
    values: List[float]
    station: Optional[str] = None

    @property
    def n_samples(self) -> int:
        return len(self.values)

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of non-NaN values."""
        return ~np.isnan(self.values)

    @property
    def mean(self) -> float:
        """Mean of valid values."""
        arr = np.array(self.values)
        valid = arr[~np.isnan(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan

    @property
    def std(self) -> float:
        """Standard deviation of valid values."""
        arr = np.array(self.values)
        valid = arr[~np.isnan(arr)]
        return float(np.std(valid)) if len(valid) > 0 else np.nan

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values)


@dataclass
class CorrelationResult:
    """Result of cross-region THD correlation analysis."""
    region_a: str
    region_b: str

    # Sample information
    n_samples: int
    n_valid_pairs: int
    date_range: Tuple[str, str]

    # Correlation metrics
    pearson_r: float
    pearson_p: float  # p-value for correlation significance

    # Lag analysis (if performed)
    optimal_lag_days: int = 0
    lag_correlation: float = 0.0
    lag_range_tested: Tuple[int, int] = (-15, 15)

    # Tidal correction results (if applied)
    tidal_correction: Optional[TidalCorrectionResult] = None
    corrected_correlation: Optional[float] = None

    # Interpretation
    is_significant: bool = False  # p < 0.05
    correlation_strength: str = ""  # "weak", "moderate", "strong"
    notes: str = ""

    def __post_init__(self):
        """Set derived fields after initialization."""
        # Determine correlation strength
        abs_r = abs(self.pearson_r) if not np.isnan(self.pearson_r) else 0
        if abs_r < 0.3:
            self.correlation_strength = "weak"
        elif abs_r < 0.6:
            self.correlation_strength = "moderate"
        else:
            self.correlation_strength = "strong"

        # Check significance
        self.is_significant = self.pearson_p < 0.05 if not np.isnan(self.pearson_p) else False

    def to_dict(self) -> Dict:
        return {
            'region_a': self.region_a,
            'region_b': self.region_b,
            'n_samples': self.n_samples,
            'n_valid_pairs': self.n_valid_pairs,
            'date_range': self.date_range,
            'pearson_r': round(self.pearson_r, 4) if not np.isnan(self.pearson_r) else None,
            'pearson_p': round(self.pearson_p, 6) if not np.isnan(self.pearson_p) else None,
            'optimal_lag_days': self.optimal_lag_days,
            'lag_correlation': round(self.lag_correlation, 4) if not np.isnan(self.lag_correlation) else None,
            'is_significant': self.is_significant,
            'correlation_strength': self.correlation_strength,
            'tidal_correction': self.tidal_correction.to_dict() if self.tidal_correction else None,
            'corrected_correlation': round(self.corrected_correlation, 4) if self.corrected_correlation else None,
            'notes': self.notes,
        }


class THDCorrelationAnalyzer:
    """
    Analyzes cross-region THD correlations for trans-Pacific hypothesis testing.

    Workflow:
    1. Load THD time series from ensemble results
    2. Compute raw correlation between region pairs
    3. Apply lag analysis to detect delayed coupling
    4. Apply tidal phase correction to test H0 (tidal artifact)
    5. Report results with statistical significance
    """

    def __init__(
        self,
        ensemble_dir: Optional[Path] = None,
        min_samples: int = 20,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize the THD correlation analyzer.

        Args:
            ensemble_dir: Directory containing ensemble JSON files
            min_samples: Minimum samples required for valid correlation
            significance_threshold: p-value threshold for significance
        """
        self.ensemble_dir = ensemble_dir or ENSEMBLE_RESULTS_DIR
        self.min_samples = min_samples
        self.significance_threshold = significance_threshold
        self.tidal_corrector = TidalPhaseCorrector()

        logger.info(f"THDCorrelationAnalyzer initialized with ensemble_dir={self.ensemble_dir}")

    def load_thd_timeseries(
        self,
        region: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> THDTimeSeries:
        """
        Load THD time series for a region from ensemble results.

        Args:
            region: Region name (e.g., 'norcal_hayward', 'hualien')
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            THDTimeSeries with dates and values
        """
        dates = []
        values = []
        station = None

        # Find all ensemble files
        ensemble_files = sorted(self.ensemble_dir.glob("ensemble_*.json"))

        for filepath in ensemble_files:
            # Skip latest symlink/copy
            if 'latest' in filepath.name:
                continue

            # Parse date from filename
            try:
                date_str = filepath.stem.replace('ensemble_', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue

            # Apply date filter
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            # Load JSON and extract THD for region
            try:
                with open(filepath) as f:
                    data = json.load(f)

                regions_data = data.get('regions', {})
                region_data = regions_data.get(region, {})

                if not region_data:
                    dates.append(file_date)
                    values.append(np.nan)
                    continue

                thd_component = region_data.get('components', {}).get('seismic_thd', {})

                if thd_component.get('available', False):
                    thd_value = thd_component.get('raw_value', np.nan)
                    dates.append(file_date)
                    values.append(float(thd_value))

                    # Extract station name if available
                    if station is None and 'notes' in thd_component:
                        notes = thd_component['notes']
                        if 'sta=' in notes:
                            station = notes.split('sta=')[1].split(',')[0]
                else:
                    dates.append(file_date)
                    values.append(np.nan)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading {filepath}: {e}")
                dates.append(file_date)
                values.append(np.nan)

        # Sort by date
        if dates:
            sorted_pairs = sorted(zip(dates, values))
            dates, values = zip(*sorted_pairs)
            dates = list(dates)
            values = list(values)

        logger.info(f"Loaded {len(dates)} samples for {region} (valid: {sum(1 for v in values if not np.isnan(v))})")

        return THDTimeSeries(
            region=region,
            dates=dates,
            values=values,
            station=station,
        )

    def compute_correlation(
        self,
        series_a: THDTimeSeries,
        series_b: THDTimeSeries,
        apply_tidal_correction: bool = True,
    ) -> CorrelationResult:
        """
        Compute correlation between two THD time series.

        Args:
            series_a: First region's THD series
            series_b: Second region's THD series
            apply_tidal_correction: Whether to apply M2 tidal phase correction

        Returns:
            CorrelationResult with statistical analysis
        """
        # Align time series by date
        dates_a = set(series_a.dates)
        dates_b = set(series_b.dates)
        common_dates = sorted(dates_a & dates_b)

        if len(common_dates) < self.min_samples:
            return CorrelationResult(
                region_a=series_a.region,
                region_b=series_b.region,
                n_samples=len(common_dates),
                n_valid_pairs=0,
                date_range=(str(min(common_dates)) if common_dates else '',
                           str(max(common_dates)) if common_dates else ''),
                pearson_r=np.nan,
                pearson_p=np.nan,
                notes=f"Insufficient common dates: {len(common_dates)} < {self.min_samples}",
            )

        # Build aligned arrays
        date_to_val_a = dict(zip(series_a.dates, series_a.values))
        date_to_val_b = dict(zip(series_b.dates, series_b.values))

        values_a = np.array([date_to_val_a[d] for d in common_dates])
        values_b = np.array([date_to_val_b[d] for d in common_dates])

        # Find valid pairs (both non-NaN)
        valid_mask = ~(np.isnan(values_a) | np.isnan(values_b))
        n_valid = valid_mask.sum()

        if n_valid < self.min_samples:
            return CorrelationResult(
                region_a=series_a.region,
                region_b=series_b.region,
                n_samples=len(common_dates),
                n_valid_pairs=n_valid,
                date_range=(str(common_dates[0]), str(common_dates[-1])),
                pearson_r=np.nan,
                pearson_p=np.nan,
                notes=f"Insufficient valid pairs: {n_valid} < {self.min_samples}",
            )

        # Compute Pearson correlation
        valid_a = values_a[valid_mask]
        valid_b = values_b[valid_mask]

        pearson_r = np.corrcoef(valid_a, valid_b)[0, 1]

        # Compute p-value using t-distribution approximation
        # t = r * sqrt((n-2) / (1-r^2))
        n = len(valid_a)
        if abs(pearson_r) < 0.9999:
            t_stat = pearson_r * np.sqrt((n - 2) / (1 - pearson_r**2))
            # Two-tailed p-value approximation
            # Using normal approximation for large n
            from scipy import stats
            try:
                pearson_p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            except ImportError:
                # Fallback: rough approximation
                pearson_p = 2 * np.exp(-0.5 * t_stat**2) / np.sqrt(2 * np.pi)
        else:
            pearson_p = 0.0  # Perfect correlation

        result = CorrelationResult(
            region_a=series_a.region,
            region_b=series_b.region,
            n_samples=len(common_dates),
            n_valid_pairs=n_valid,
            date_range=(str(common_dates[0].date()), str(common_dates[-1].date())),
            pearson_r=pearson_r,
            pearson_p=pearson_p,
        )

        # Apply tidal correction if requested
        if apply_tidal_correction:
            tidal_result = self.tidal_corrector.analyze_tidal_aliasing(
                series_a.region,
                series_b.region,
                values_a,
                values_b,
            )
            result.tidal_correction = tidal_result
            result.corrected_correlation = tidal_result.corrected_correlation

            if tidal_result.h0_supported:
                result.notes = f"H0 supported: Tidal aliasing likely explains correlation"
            elif tidal_result.is_opposing_phase:
                result.notes = f"H0 rejected: Correlation persists after tidal correction"
            else:
                result.notes = f"Non-opposing phases: Tidal aliasing unlikely"

        return result

    def compute_lag_correlation(
        self,
        series_a: THDTimeSeries,
        series_b: THDTimeSeries,
        lag_range: Tuple[int, int] = (-15, 15),
    ) -> Tuple[int, float, List[Tuple[int, float]]]:
        """
        Compute correlation at different lags to detect delayed coupling.

        Args:
            series_a: First region's THD series (reference)
            series_b: Second region's THD series (lagged)
            lag_range: Range of lags in days (min, max)

        Returns:
            Tuple of (optimal_lag, max_correlation, all_lag_correlations)
        """
        # Align time series
        dates_a = set(series_a.dates)
        dates_b = set(series_b.dates)
        common_dates = sorted(dates_a & dates_b)

        if len(common_dates) < self.min_samples:
            return (0, np.nan, [])

        date_to_val_a = dict(zip(series_a.dates, series_a.values))
        date_to_val_b = dict(zip(series_b.dates, series_b.values))

        values_a = np.array([date_to_val_a[d] for d in common_dates])
        values_b = np.array([date_to_val_b[d] for d in common_dates])

        lag_correlations = []
        best_lag = 0
        best_corr = np.nan

        for lag in range(lag_range[0], lag_range[1] + 1):
            if lag < 0:
                # B leads A: shift A forward
                a_slice = values_a[-lag:]
                b_slice = values_b[:lag]
            elif lag > 0:
                # A leads B: shift B forward
                a_slice = values_a[:-lag]
                b_slice = values_b[lag:]
            else:
                a_slice = values_a
                b_slice = values_b

            # Compute correlation for this lag
            valid = ~(np.isnan(a_slice) | np.isnan(b_slice))
            if valid.sum() >= self.min_samples // 2:
                corr = np.corrcoef(a_slice[valid], b_slice[valid])[0, 1]
                lag_correlations.append((lag, corr))

                if np.isnan(best_corr) or abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        return (best_lag, best_corr, lag_correlations)

    def analyze_region_pair(
        self,
        region_a: str,
        region_b: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        apply_tidal_correction: bool = True,
        compute_lags: bool = True,
    ) -> CorrelationResult:
        """
        Full analysis of a region pair.

        Args:
            region_a: First region name
            region_b: Second region name
            start_date: Start of analysis period
            end_date: End of analysis period
            apply_tidal_correction: Whether to apply M2 phase correction
            compute_lags: Whether to compute lag correlations

        Returns:
            CorrelationResult with full analysis
        """
        logger.info(f"Analyzing THD correlation: {region_a} ↔ {region_b}")

        # Load time series
        series_a = self.load_thd_timeseries(region_a, start_date, end_date)
        series_b = self.load_thd_timeseries(region_b, start_date, end_date)

        # Compute correlation
        result = self.compute_correlation(series_a, series_b, apply_tidal_correction)

        # Compute lag analysis if requested
        if compute_lags and result.n_valid_pairs >= self.min_samples:
            lag, lag_corr, all_lags = self.compute_lag_correlation(series_a, series_b)
            result.optimal_lag_days = lag
            result.lag_correlation = lag_corr

        logger.info(f"  r={result.pearson_r:.3f}, p={result.pearson_p:.4f}, "
                   f"significant={result.is_significant}, n={result.n_valid_pairs}")

        return result

    def analyze_all_pairs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, CorrelationResult]:
        """
        Analyze all configured region pairs.

        Returns:
            Dict mapping "region_a|region_b" to CorrelationResult
        """
        config = get_config()
        results = {}

        for region_a, region_b in config.region_pairs:
            key = f"{region_a}|{region_b}"
            results[key] = self.analyze_region_pair(
                region_a, region_b,
                start_date, end_date,
                apply_tidal_correction=config.tidal_phase_correction,
            )

        return results


def run_hayward_hualien_analysis(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> CorrelationResult:
    """
    Convenience function to analyze the primary Hayward-Hualien pair.

    This is the pair that showed r ≈ -0.72 inverse correlation,
    potentially due to ~180° M2 tidal phase difference.
    """
    analyzer = THDCorrelationAnalyzer()
    return analyzer.analyze_region_pair(
        'norcal_hayward',
        'hualien',
        start_date,
        end_date,
        apply_tidal_correction=True,
        compute_lags=True,
    )


# Quick test when run directly
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=== Trans-Pacific THD Correlation Analysis ===\n")

    analyzer = THDCorrelationAnalyzer()

    # Analyze Hayward-Hualien pair (primary observation)
    result = analyzer.analyze_region_pair('norcal_hayward', 'hualien')

    print(f"Hayward ↔ Hualien:")
    print(f"  Samples: {result.n_valid_pairs} valid of {result.n_samples}")
    print(f"  Date range: {result.date_range[0]} to {result.date_range[1]}")
    print(f"  Pearson r: {result.pearson_r:.4f} ({result.correlation_strength})")
    print(f"  p-value: {result.pearson_p:.6f}")
    print(f"  Significant: {result.is_significant}")
    print(f"  Optimal lag: {result.optimal_lag_days} days (r={result.lag_correlation:.4f})")

    if result.tidal_correction:
        tc = result.tidal_correction
        print(f"\n  Tidal Correction:")
        print(f"    Phase difference: {tc.phase_difference_degrees:.1f}°")
        print(f"    Opposing phases: {tc.is_opposing_phase}")
        print(f"    Original r: {tc.original_correlation:.4f}" if tc.original_correlation else "")
        print(f"    Corrected r: {tc.corrected_correlation:.4f}" if tc.corrected_correlation else "")
        print(f"    H0 supported: {tc.h0_supported}")

    print(f"\n  Notes: {result.notes}")
