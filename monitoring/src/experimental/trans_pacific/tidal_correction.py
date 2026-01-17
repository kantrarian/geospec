"""
tidal_correction.py - M2 Tidal Phase Correction for Trans-Pacific Correlation

This module implements tidal phase correction to test H0: whether the observed
inverse correlation between trans-Pacific regions is due to tidal aliasing
rather than genuine tectonic coupling.

The M2 semidiurnal tide has a ~12.42 hour period and propagates westward
following the moon. Regions ~180° apart in longitude will experience
opposing tidal phases, creating an artificial inverse correlation in
THD (Total Harmonic Distortion) measurements.

Key Insight:
- Taiwan (Hualien, ~121°E) and California (Hayward, ~122°W)
- Longitude difference: 121 - (-122) = 243° ≈ 16.2 hours
- M2 period: 12.42 hours → 243° = 243/360 * 12.42 = ~8.4 hours phase shift
- This is close to half a period (~6.2 hours), creating ~180° phase opposition

Implementation Strategy:
Uses longitudinal phase approximation to avoid heavy astronomical dependencies
(pytides, astropy, skyfield). For testing the ~180° phase hypothesis, this
geometric approximation is sufficient.

Formula: Phase (hours) ≈ Longitude (degrees) / 15°/hour
(The Earth rotates 15° per hour, so the tide advances ~15°/hour westward)

Reference: docs/TRANS_PACIFIC_CORRELATION_PAPER_SKELETON.md Section 3.3
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# M2 tidal constants
M2_PERIOD_HOURS = 12.42  # Principal lunar semidiurnal tide period
EARTH_ROTATION_DEG_PER_HOUR = 15.0  # 360° / 24 hours


@dataclass
class RegionTidalInfo:
    """Tidal phase information for a monitoring region."""
    region: str
    longitude: float  # Degrees, positive = East, negative = West
    latitude: float   # Degrees
    primary_station: str

    @property
    def m2_phase_hours(self) -> float:
        """
        Approximate M2 tidal phase using longitudinal approximation.

        Returns hours relative to UTC=0, Greenwich.
        Positive = tide arrives later (west of Greenwich)
        Negative = tide arrives earlier (east of Greenwich)
        """
        # Tide propagates westward following the moon
        # At longitude L, phase = L / 15 hours
        return self.longitude / EARTH_ROTATION_DEG_PER_HOUR

    @property
    def m2_phase_degrees(self) -> float:
        """M2 phase in degrees (0-360)."""
        phase_hours = self.m2_phase_hours
        phase_deg = (phase_hours / M2_PERIOD_HOURS) * 360.0
        return phase_deg % 360.0


# Pre-defined region tidal info
# Coordinates from USGS/IRIS station metadata
REGION_TIDAL_INFO: Dict[str, RegionTidalInfo] = {
    # California - West Coast USA
    'norcal_hayward': RegionTidalInfo(
        region='norcal_hayward',
        longitude=-122.0,  # ~122°W
        latitude=37.8,
        primary_station='BK.BKS'
    ),
    'socal_san_andreas': RegionTidalInfo(
        region='socal_san_andreas',
        longitude=-117.5,  # ~117.5°W
        latitude=34.0,
        primary_station='CI.PAS'
    ),

    # Taiwan
    'hualien': RegionTidalInfo(
        region='hualien',
        longitude=121.5,  # ~121.5°E
        latitude=24.0,
        primary_station='IU.TATO'
    ),

    # Japan
    'tokyo_kanto': RegionTidalInfo(
        region='tokyo_kanto',
        longitude=139.7,  # ~139.7°E (Tokyo)
        latitude=35.7,
        primary_station='N.KI2H'  # Hi-net or IU.MAJO
    ),

    # Alaska
    'anchorage': RegionTidalInfo(
        region='anchorage',
        longitude=-149.9,  # ~149.9°W
        latitude=61.2,
        primary_station='IU.COLA'
    ),

    # Cascadia
    'cascadia': RegionTidalInfo(
        region='cascadia',
        longitude=-123.5,  # ~123.5°W (Oregon coast)
        latitude=44.5,
        primary_station='IU.COR'
    ),

    # Turkey
    'istanbul_marmara': RegionTidalInfo(
        region='istanbul_marmara',
        longitude=29.0,  # ~29°E
        latitude=41.0,
        primary_station='IU.ANTO'
    ),
}


@dataclass
class TidalCorrectionResult:
    """Result of tidal phase correction analysis."""
    region_a: str
    region_b: str

    # Phase information
    phase_a_hours: float
    phase_b_hours: float
    phase_difference_hours: float
    phase_difference_degrees: float

    # Is this pair likely affected by tidal aliasing?
    # True if phase difference is close to half-period (~6.2 hours or ~180°)
    is_opposing_phase: bool

    # Original and corrected correlation
    original_correlation: Optional[float] = None
    corrected_correlation: Optional[float] = None

    # Interpretation
    h0_supported: bool = False  # True if tidal correction explains correlation
    notes: str = ''

    def to_dict(self) -> Dict:
        return {
            'region_a': self.region_a,
            'region_b': self.region_b,
            'phase_a_hours': round(self.phase_a_hours, 2),
            'phase_b_hours': round(self.phase_b_hours, 2),
            'phase_difference_hours': round(self.phase_difference_hours, 2),
            'phase_difference_degrees': round(self.phase_difference_degrees, 1),
            'is_opposing_phase': self.is_opposing_phase,
            'original_correlation': round(self.original_correlation, 3) if self.original_correlation else None,
            'corrected_correlation': round(self.corrected_correlation, 3) if self.corrected_correlation else None,
            'h0_supported': self.h0_supported,
            'notes': self.notes,
        }


class TidalPhaseCorrector:
    """
    Corrects for M2 tidal phase differences between monitoring regions.

    The correction works by:
    1. Computing the M2 phase difference between two regions
    2. If phases are ~180° apart, THD signals are expected to be inversely correlated
    3. We can "align" the signals by shifting one by the phase difference
    4. If correlation disappears after alignment, H0 (tidal artifact) is supported
    """

    def __init__(
        self,
        opposing_threshold_degrees: float = 150.0,  # Within 30° of 180° = opposing
    ):
        """
        Initialize the tidal phase corrector.

        Args:
            opposing_threshold_degrees: Phase difference (from 180°) to consider "opposing"
        """
        self.opposing_threshold = opposing_threshold_degrees
        logger.info(f"TidalPhaseCorrector initialized with opposing_threshold={opposing_threshold_degrees}°")

    def get_region_info(self, region: str) -> Optional[RegionTidalInfo]:
        """Get tidal info for a region."""
        return REGION_TIDAL_INFO.get(region)

    def compute_phase_difference(
        self,
        region_a: str,
        region_b: str,
    ) -> TidalCorrectionResult:
        """
        Compute M2 tidal phase difference between two regions.

        Args:
            region_a: First region name
            region_b: Second region name

        Returns:
            TidalCorrectionResult with phase analysis
        """
        info_a = self.get_region_info(region_a)
        info_b = self.get_region_info(region_b)

        if not info_a:
            logger.warning(f"No tidal info for region: {region_a}")
            return TidalCorrectionResult(
                region_a=region_a,
                region_b=region_b,
                phase_a_hours=0,
                phase_b_hours=0,
                phase_difference_hours=0,
                phase_difference_degrees=0,
                is_opposing_phase=False,
                notes=f"Missing tidal info for {region_a}"
            )

        if not info_b:
            logger.warning(f"No tidal info for region: {region_b}")
            return TidalCorrectionResult(
                region_a=region_a,
                region_b=region_b,
                phase_a_hours=info_a.m2_phase_hours,
                phase_b_hours=0,
                phase_difference_hours=0,
                phase_difference_degrees=0,
                is_opposing_phase=False,
                notes=f"Missing tidal info for {region_b}"
            )

        phase_a = info_a.m2_phase_hours
        phase_b = info_b.m2_phase_hours

        # Compute phase difference in hours
        phase_diff_hours = abs(phase_a - phase_b)

        # Wrap to M2 period (0 to 12.42 hours)
        phase_diff_hours = phase_diff_hours % M2_PERIOD_HOURS

        # Convert to degrees (0 to 360)
        phase_diff_degrees = (phase_diff_hours / M2_PERIOD_HOURS) * 360.0

        # Check if phases are opposing (close to 180°)
        deviation_from_180 = abs(phase_diff_degrees - 180.0)
        is_opposing = deviation_from_180 <= (180.0 - self.opposing_threshold)

        # Build notes
        notes_parts = [
            f"{region_a} ({info_a.longitude}°) phase: {phase_a:.1f}h",
            f"{region_b} ({info_b.longitude}°) phase: {phase_b:.1f}h",
            f"Difference: {phase_diff_degrees:.1f}° ({phase_diff_hours:.1f}h)",
        ]
        if is_opposing:
            notes_parts.append("WARNING: OPPOSING PHASES - High tidal aliasing risk")
        else:
            notes_parts.append("OK: Non-opposing phases - Low tidal aliasing risk")

        return TidalCorrectionResult(
            region_a=region_a,
            region_b=region_b,
            phase_a_hours=phase_a,
            phase_b_hours=phase_b,
            phase_difference_hours=phase_diff_hours,
            phase_difference_degrees=phase_diff_degrees,
            is_opposing_phase=is_opposing,
            notes=' | '.join(notes_parts)
        )

    def compute_phase_shifted_correlation(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
        phase_diff_hours: float,
        sampling_interval_hours: float = 24.0,  # Daily data
    ) -> Tuple[float, float, int]:
        """
        Compute correlation after phase-shifting one signal.

        For daily THD data, the phase shift is applied as a lag in days.

        Args:
            values_a: THD values for region A (daily)
            values_b: THD values for region B (daily)
            phase_diff_hours: Phase difference in hours
            sampling_interval_hours: Sampling interval (default 24h for daily data)

        Returns:
            Tuple of (original_correlation, shifted_correlation, optimal_lag_days)
        """
        if len(values_a) != len(values_b):
            raise ValueError(f"Arrays must have same length: {len(values_a)} vs {len(values_b)}")

        if len(values_a) < 5:
            raise ValueError(f"Need at least 5 data points, got {len(values_a)}")

        # Original correlation (no lag)
        valid_mask = ~(np.isnan(values_a) | np.isnan(values_b))
        if valid_mask.sum() < 5:
            return (np.nan, np.nan, 0)

        original_corr = np.corrcoef(values_a[valid_mask], values_b[valid_mask])[0, 1]

        # Compute lag in samples (days for daily data)
        # Phase difference in hours → lag in days
        lag_days = int(round(phase_diff_hours / sampling_interval_hours))

        # Limit lag to reasonable range
        max_lag = len(values_a) // 3
        lag_days = max(-max_lag, min(max_lag, lag_days))

        # Apply lag to find best correlation
        # We try the computed lag and a few neighbors to find the optimal alignment
        best_corr = original_corr
        best_lag = 0

        for test_lag in range(max(-max_lag, lag_days - 2), min(max_lag, lag_days + 3)):
            if test_lag == 0:
                continue

            if test_lag > 0:
                a_shifted = values_a[test_lag:]
                b_shifted = values_b[:-test_lag]
            else:
                a_shifted = values_a[:test_lag]
                b_shifted = values_b[-test_lag:]

            valid = ~(np.isnan(a_shifted) | np.isnan(b_shifted))
            if valid.sum() >= 5:
                corr = np.corrcoef(a_shifted[valid], b_shifted[valid])[0, 1]
                # For inverse correlations, we want to find if shifting makes it less negative
                if abs(corr) < abs(best_corr):
                    best_corr = corr
                    best_lag = test_lag

        return (original_corr, best_corr, best_lag)

    def analyze_tidal_aliasing(
        self,
        region_a: str,
        region_b: str,
        values_a: np.ndarray,
        values_b: np.ndarray,
        significance_threshold: float = 0.3,  # |r| < 0.3 = weak correlation
    ) -> TidalCorrectionResult:
        """
        Full tidal aliasing analysis for a region pair.

        Tests H0: The observed correlation is due to tidal phase opposition.

        Args:
            region_a: First region name
            region_b: Second region name
            values_a: THD values for region A
            values_b: THD values for region B
            significance_threshold: Correlation threshold below which H0 is supported

        Returns:
            TidalCorrectionResult with full analysis
        """
        # Get phase difference
        result = self.compute_phase_difference(region_a, region_b)

        try:
            # Compute correlations
            orig_corr, shifted_corr, lag = self.compute_phase_shifted_correlation(
                values_a, values_b, result.phase_difference_hours
            )

            result.original_correlation = orig_corr
            result.corrected_correlation = shifted_corr

            # Determine if H0 is supported
            # H0: Tidal aliasing explains the correlation
            # Supported if:
            # 1. Phases are opposing (or close to it)
            # 2. Original correlation was significant (|r| > threshold)
            # 3. Shifted correlation is weak (|r| < threshold)

            orig_significant = abs(orig_corr) > significance_threshold
            shifted_weak = abs(shifted_corr) < significance_threshold

            if result.is_opposing_phase and orig_significant and shifted_weak:
                result.h0_supported = True
                result.notes += f" | H0 SUPPORTED: Correlation {orig_corr:.3f} → {shifted_corr:.3f} after {lag}-day shift"
            elif result.is_opposing_phase and orig_significant and not shifted_weak:
                result.h0_supported = False
                result.notes += f" | H0 REJECTED: Correlation persists {orig_corr:.3f} → {shifted_corr:.3f} despite phase correction"
            elif not orig_significant:
                result.h0_supported = True
                result.notes += f" | H0 TRIVIALLY SUPPORTED: Original correlation {orig_corr:.3f} already weak"
            else:
                result.notes += f" | Non-opposing phases, tidal aliasing unlikely"

        except Exception as e:
            result.notes += f" | Error in correlation analysis: {str(e)}"
            logger.warning(f"Tidal aliasing analysis failed for {region_a}-{region_b}: {e}")

        return result


def compute_hayward_hualien_phase_difference() -> TidalCorrectionResult:
    """
    Convenience function to compute phase difference for the primary region pair.

    This is the pair that showed r ≈ -0.72 inverse correlation.
    """
    corrector = TidalPhaseCorrector()
    return corrector.compute_phase_difference('norcal_hayward', 'hualien')


# Quick verification when module is run directly
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=== Trans-Pacific Tidal Phase Analysis ===\n")

    corrector = TidalPhaseCorrector()

    # Analyze key pairs
    pairs = [
        ('norcal_hayward', 'hualien'),
        ('norcal_hayward', 'tokyo_kanto'),
        ('cascadia', 'tokyo_kanto'),
        ('norcal_hayward', 'anchorage'),
    ]

    for region_a, region_b in pairs:
        result = corrector.compute_phase_difference(region_a, region_b)
        print(f"{region_a} ↔ {region_b}:")
        print(f"  Phase difference: {result.phase_difference_degrees:.1f}°")
        print(f"  Opposing phases: {result.is_opposing_phase}")
        print(f"  Notes: {result.notes}")
        print()
