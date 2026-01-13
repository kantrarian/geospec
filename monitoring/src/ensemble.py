"""
ensemble.py
GeoSpec Ensemble Risk Assessment - Three-Method Integration

Combines three independent earthquake precursor detection methods:
1. Lambda_geo (GPS): Surface strain eigenframe rotation
2. Fault Correlation: Seismic segment decoupling
3. Seismic THD: Rock nonlinearity (harmonic distortion)

Physical Basis:
- Each method detects different aspects of pre-earthquake stress evolution
- THD: First to elevate (~14 days) - early warning
- Lambda_geo: Days before - confirms building stress
- Fault Correlation: Hours before - imminent rupture

Risk Combination:
- Convert each method to 0-1 risk score
- Weights: lambda_geo=0.4, fault_corr=0.3, thd=0.3
- Confidence boost when methods agree
- Flag potential false positives when methods disagree

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

# Local imports
from fault_correlation import FaultCorrelationMonitor, CorrelationResult
from fault_segments import get_segments_for_region, FAULT_SEGMENTS
from seismic_thd import SeismicTHDAnalyzer, THDResult, fetch_continuous_data_for_thd


# =============================================================================
# CANONICAL REGION MAPPING
# =============================================================================
# Maps runner/dashboard region keys to fault_segments canonical keys.
# This prevents silent failures when region naming conventions differ.
# Runner keys are user-facing (e.g., "tokyo_kanto" for dashboard readability).
# FC keys must match FAULT_SEGMENTS dict in fault_segments.py.

REGION_KEY_MAP = {
    # California
    'ridgecrest': 'ridgecrest',
    'socal_saf_mojave': 'socal_saf_mojave',
    'socal_saf_coachella': 'socal_coachella',  # Runner uses socal_saf_coachella, FC uses socal_coachella
    'norcal_hayward': 'norcal_hayward',
    'cascadia': 'cascadia',
    # International
    'tokyo_kanto': 'japan_tohoku',  # Runner uses tokyo_kanto, FC uses japan_tohoku
    'istanbul_marmara': 'istanbul_marmara',
    'turkey_kahramanmaras': 'turkey_kahramanmaras',
    # Italy - Volcanic pilot
    'campi_flegrei': 'campi_flegrei',
    # Future regions (add here as they're defined in fault_segments.py)
    'chile_maule': 'chile_maule',
    'japan_tohoku': 'japan_tohoku',
    'socal_coachella': 'socal_coachella',
}


def get_fc_region_key(runner_key: str) -> str:
    """
    Translate runner/dashboard region key to fault_segments canonical key.

    Returns the canonical key if mapping exists, otherwise returns input unchanged
    with a warning logged.
    """
    if runner_key in REGION_KEY_MAP:
        fc_key = REGION_KEY_MAP[runner_key]
        if fc_key != runner_key:
            logger.debug(f"Region key mapped: {runner_key} -> {fc_key}")
        return fc_key
    else:
        logger.warning(f"Unknown region key '{runner_key}' - not in REGION_KEY_MAP. "
                      f"Valid keys: {list(REGION_KEY_MAP.keys())}")
        return runner_key


def validate_region_keys():
    """
    Validate that all mapped FC keys exist in FAULT_SEGMENTS.
    Call at startup to catch config errors early.
    """
    errors = []
    for runner_key, fc_key in REGION_KEY_MAP.items():
        if fc_key not in FAULT_SEGMENTS:
            errors.append(f"  {runner_key} -> {fc_key} (not in FAULT_SEGMENTS)")

    if errors:
        logger.error("Region key validation failed! Invalid FC mappings:\n" + "\n".join(errors))
        logger.error(f"Valid FAULT_SEGMENTS keys: {list(FAULT_SEGMENTS.keys())}")
        return False

    logger.info(f"Region key validation passed: {len(REGION_KEY_MAP)} mappings OK")
    return True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# RISK TIERS
# =============================================================================

RISK_TIERS = {
    0: {'name': 'NORMAL', 'min_risk': 0.0, 'max_risk': 0.25, 'color': 'green'},
    1: {'name': 'WATCH', 'min_risk': 0.25, 'max_risk': 0.50, 'color': 'yellow'},
    2: {'name': 'ELEVATED', 'min_risk': 0.50, 'max_risk': 0.75, 'color': 'orange'},
    3: {'name': 'CRITICAL', 'min_risk': 0.75, 'max_risk': 1.0, 'color': 'red'},
    -1: {'name': 'DEGRADED', 'min_risk': -1, 'max_risk': -1, 'color': 'gray'},  # Insufficient data
}

# Minimum requirements for operational status (non-DEGRADED)
MIN_METHODS_FOR_OPERATIONAL = 1  # At least 1 method must be available
MIN_COVERAGE_PCT_FOR_OPERATIONAL = 0  # Coverage check (0 = disabled for now)

# Method weights
WEIGHTS = {
    'lambda_geo': 0.4,
    'fault_correlation': 0.3,
    'seismic_thd': 0.3,
}


# =============================================================================
# RISK CONVERSION FUNCTIONS
# =============================================================================

def lambda_geo_to_risk(ratio: float) -> float:
    """
    Convert Lambda_geo baseline ratio to 0-1 risk score.

    Design Note: Historical detections show ratios 485× to 7999×.
    Using 1000× saturation keeps 50× and 5000× distinguishable.
    Raw ratio is always preserved for analyst review.

    Mapping (logarithmic with 1000× saturation):
    - ratio < 3: NORMAL (risk ~ 0.00-0.16)
    - ratio 3-10: WATCH (risk ~ 0.16-0.33)
    - ratio 10-100: ELEVATED (risk ~ 0.33-0.67)
    - ratio 100-1000: CRITICAL (risk ~ 0.67-1.00)
    - ratio > 1000: Capped at 1.0 (raw value shown separately)
    """
    if ratio <= 1:
        return 0.0

    # Logarithmic mapping: log10(ratio) / log10(max_ratio)
    # At ratio=3:    log10(3)/log10(1000) ≈ 0.16
    # At ratio=10:   log10(10)/log10(1000) ≈ 0.33
    # At ratio=100:  log10(100)/log10(1000) ≈ 0.67
    # At ratio=1000: log10(1000)/log10(1000) = 1.0

    max_log_ratio = np.log10(1000)  # Saturation point at 1000×
    risk = np.log10(max(1, ratio)) / max_log_ratio
    return min(1.0, risk)


def fault_correlation_to_risk(l2_l1_ratio: float, participation_ratio: float) -> float:
    """
    Convert fault correlation metrics to 0-1 risk score.

    Low L2/L1 = decorrelated = higher risk
    Low participation ratio = concentrated stress = higher risk

    Normal: L2/L1 > 0.3, PR > 2.0 -> low risk
    Critical: L2/L1 < 0.1, PR < 1.5 -> high risk
    """
    # L2/L1 contribution (lower = more risk)
    # 0.3 -> risk ~0.25, 0.1 -> risk ~0.75, 0.05 -> risk ~0.9
    if l2_l1_ratio >= 0.3:
        l2_l1_risk = 0.25 * (1 - (l2_l1_ratio - 0.3) / 0.7)  # 0-0.25 for normal
    elif l2_l1_ratio >= 0.1:
        l2_l1_risk = 0.25 + 0.5 * (0.3 - l2_l1_ratio) / 0.2  # 0.25-0.75
    else:
        l2_l1_risk = 0.75 + 0.25 * (0.1 - l2_l1_ratio) / 0.1  # 0.75-1.0

    # Participation ratio contribution (lower = more risk)
    # PR=3 -> low risk, PR=1 -> high risk
    if participation_ratio >= 2.5:
        pr_risk = 0.1
    elif participation_ratio >= 1.5:
        pr_risk = 0.1 + 0.4 * (2.5 - participation_ratio)
    else:
        pr_risk = 0.5 + 0.5 * (1.5 - participation_ratio) / 0.5

    # Combine: weight L2/L1 more heavily (it's the primary diagnostic)
    risk = 0.7 * l2_l1_risk + 0.3 * pr_risk
    return min(1.0, max(0.0, risk))


def thd_to_risk(thd_value: float) -> float:
    """
    Convert THD value to 0-1 risk score using absolute thresholds.

    DEPRECATED: Use thd_to_risk_with_baseline() when station baseline is available.

    Normal: THD < 0.05 -> risk ~0.1
    Elevated: 0.05 < THD < 0.15 -> risk ~0.3-0.6
    Critical: THD > 0.15 -> risk ~0.7-1.0

    Very high THD (>0.5) indicates extreme nonlinearity.
    """
    if thd_value < 0.05:
        return 0.1 * (thd_value / 0.05)
    elif thd_value < 0.15:
        return 0.1 + 0.5 * (thd_value - 0.05) / 0.10
    elif thd_value < 0.5:
        return 0.6 + 0.3 * (thd_value - 0.15) / 0.35
    else:
        return min(1.0, 0.9 + 0.1 * (thd_value - 0.5) / 0.5)


def thd_to_risk_with_baseline(
    thd_value: float,
    baseline_mean: float,
    baseline_std: float
) -> tuple:
    """
    Convert THD value to risk using station-specific baseline.

    Uses z-score (standard deviations above baseline) for anomaly detection.
    This is more robust than absolute thresholds since each station has
    different characteristics.

    Returns:
        Tuple of (risk_score, z_score)

    Z-score interpretation:
        z < 0: Below baseline (normal)
        z = 0-1: Normal variation
        z = 1-2: Elevated (WATCH)
        z = 2-3: Significant anomaly (ELEVATED)
        z > 3: Critical anomaly (CRITICAL)
    """
    if baseline_std <= 0:
        # No baseline available, fall back to absolute thresholds
        return thd_to_risk(thd_value), 0.0

    # Compute z-score
    z_score = (thd_value - baseline_mean) / baseline_std

    # Convert to risk
    if z_score < 0:
        risk = max(0.0, 0.1 + z_score * 0.05)
    elif z_score < 1:
        risk = 0.1 + z_score * 0.15
    elif z_score < 2:
        risk = 0.25 + (z_score - 1) * 0.25
    elif z_score < 3:
        risk = 0.50 + (z_score - 2) * 0.25
    else:
        risk = min(1.0, 0.75 + (z_score - 3) * 0.08)

    return risk, z_score


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class MethodResult:
    """Result from a single method."""
    name: str
    available: bool
    raw_value: float  # Primary metric value
    raw_secondary: Optional[float] = None  # Secondary metric if applicable
    risk_score: float = 0.0
    is_elevated: bool = False
    is_critical: bool = False
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'available': self.available,
            'raw_value': float(self.raw_value) if self.raw_value is not None else None,
            'raw_secondary': float(self.raw_secondary) if self.raw_secondary is not None else None,
            'risk_score': float(self.risk_score),
            'is_elevated': bool(self.is_elevated),
            'is_critical': bool(self.is_critical),
            'notes': self.notes,
        }


@dataclass
class EnsembleResult:
    """Combined result from all methods."""
    region: str
    date: datetime
    combined_risk: float
    tier: int
    tier_name: str
    components: Dict[str, MethodResult]
    confidence: float
    agreement: str  # 'all_normal', 'all_elevated', 'mixed', 'disagreement'
    methods_available: int
    notes: str = ""
    # Coverage tracking
    segments_defined: int = 0  # Total fault segments defined for region
    segments_working: int = 0  # Segments with sufficient data
    segment_names: List[str] = field(default_factory=list)  # Names of working segments
    effective_weights: Dict[str, float] = field(default_factory=dict)  # Weights after renorm

    def to_dict(self) -> Dict:
        return {
            'region': self.region,
            'date': self.date.isoformat(),
            'combined_risk': float(self.combined_risk),
            'tier': self.tier,
            'tier_name': self.tier_name,
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'confidence': float(self.confidence),
            'agreement': self.agreement,
            'methods_available': self.methods_available,
            'notes': self.notes,
            # Coverage tracking
            'coverage': {
                'segments_defined': self.segments_defined,
                'segments_working': self.segments_working,
                'segment_names': self.segment_names,
                'coverage_pct': (self.segments_working / self.segments_defined * 100) if self.segments_defined > 0 else 0.0,
            },
            'effective_weights': self.effective_weights,
        }


# =============================================================================
# ENSEMBLE CLASS
# =============================================================================

class GeoSpecEnsemble:
    """
    Combine GPS (Lambda_geo) + Seismic (Fault Correlation + THD)
    into unified risk assessment.

    Attributes:
        region: Region name for analysis
        fault_corr_monitor: FaultCorrelationMonitor instance
        thd_analyzer: SeismicTHDAnalyzer instance
        weights: Method weights for combination
    """

    def __init__(
        self,
        region: str,
        weights: Optional[Dict[str, float]] = None,
        thd_window_hours: int = 24,
        corr_window_hours: int = 24,
    ):
        """
        Initialize the ensemble.

        Args:
            region: Region name (e.g., 'ridgecrest')
            weights: Custom weights dict, or use defaults
            thd_window_hours: Window size for THD analysis
            corr_window_hours: Window size for correlation analysis
        """
        self.region = region
        self.weights = weights or WEIGHTS.copy()

        # Initialize seismic analyzers
        self.fault_corr_monitor = FaultCorrelationMonitor(
            window_hours=corr_window_hours,
            decorrelation_threshold=0.3
        )
        self.thd_analyzer = SeismicTHDAnalyzer(
            window_hours=thd_window_hours
        )

        # Cache for Lambda_geo results (loaded from external source)
        self._lambda_geo_cache: Dict[str, float] = {}

        logger.info(f"GeoSpecEnsemble initialized for {region}")
        logger.info(f"  Weights: {self.weights}")

    def set_lambda_geo(self, date: datetime, ratio: float):
        """
        Set Lambda_geo value for a specific date.

        In production, this would come from the GPS pipeline.
        For validation, we inject historical values.
        """
        key = date.strftime('%Y-%m-%d')
        self._lambda_geo_cache[key] = ratio

    def get_lambda_geo(self, date: datetime) -> Optional[float]:
        """Get Lambda_geo ratio for date."""
        key = date.strftime('%Y-%m-%d')
        return self._lambda_geo_cache.get(key)

    def compute_lambda_geo_risk(self, date: datetime) -> MethodResult:
        """Compute Lambda_geo risk component."""
        ratio = self.get_lambda_geo(date)

        if ratio is None:
            return MethodResult(
                name='lambda_geo',
                available=False,
                raw_value=0.0,
                risk_score=0.0,
                notes='No Lambda_geo data available'
            )

        risk = lambda_geo_to_risk(ratio)

        return MethodResult(
            name='lambda_geo',
            available=True,
            raw_value=ratio,
            risk_score=risk,
            is_elevated=risk >= 0.5,
            is_critical=risk >= 0.75,
            notes=f'ratio={ratio:.1f}x'
        )

    def compute_fault_correlation_risk(self, date: datetime) -> Tuple[MethodResult, int, int, List[str]]:
        """
        Compute fault correlation risk component with coverage tracking.

        Returns:
            Tuple of (MethodResult, segments_defined, segments_working, segment_names)
        """
        # Translate runner region key to fault_segments canonical key
        fc_region = get_fc_region_key(self.region)

        # Get total segments defined for this region
        try:
            all_segments = get_segments_for_region(fc_region)
            segments_defined = len(all_segments)
        except ValueError:
            segments_defined = 0

        try:
            result = self.fault_corr_monitor.analyze_region(fc_region, date)
            segments_working = len(result.segment_names)
            segment_names = result.segment_names

            if not result.segment_names:
                return (
                    MethodResult(
                        name='fault_correlation',
                        available=False,
                        raw_value=0.0,
                        notes='Insufficient segment data'
                    ),
                    segments_defined,
                    0,
                    []
                )

            l2_l1 = result.eigenvalue_ratios[0] if len(result.eigenvalue_ratios) > 0 else 1.0
            pr = result.participation_ratio

            risk = fault_correlation_to_risk(l2_l1, pr)

            coverage_note = f'{segments_working}/{segments_defined} segments'

            return (
                MethodResult(
                    name='fault_correlation',
                    available=True,
                    raw_value=l2_l1,
                    raw_secondary=pr,
                    risk_score=risk,
                    is_elevated=risk >= 0.5,
                    is_critical=risk >= 0.75,
                    notes=f'L2/L1={l2_l1:.4f}, PR={pr:.2f}, {coverage_note}'
                ),
                segments_defined,
                segments_working,
                segment_names
            )

        except Exception as e:
            logger.warning(f"Fault correlation failed: {e}")
            return (
                MethodResult(
                    name='fault_correlation',
                    available=False,
                    raw_value=0.0,
                    notes=f'Error: {str(e)}'
                ),
                segments_defined,
                0,
                []
            )

    def compute_thd_risk(
        self,
        date: datetime,
        station_network: str = 'CI',
        station_code: str = 'CCC'
    ) -> MethodResult:
        """Compute THD risk component using station-specific baselines."""
        try:
            # Import station baselines
            try:
                from station_baselines import get_baseline, STATION_BASELINES
            except ImportError:
                get_baseline = None

            # Need 24+ hours of data ending at target date
            end_time = date
            start_time = date - timedelta(hours=self.thd_analyzer.window_hours + 1)

            data, sample_rate = fetch_continuous_data_for_thd(
                station_network=station_network,
                station_code=station_code,
                start=start_time,
                end=end_time
            )

            if data is None or len(data) < sample_rate * 3600 * 12:
                return MethodResult(
                    name='seismic_thd',
                    available=False,
                    raw_value=0.0,
                    notes=f'Insufficient data from {station_network}.{station_code}'
                )

            # Store native sample rate for logging
            native_sample_rate = sample_rate

            # Resample to target rate (1 Hz) for consistent THD computation
            # Using resample_poly for predictable filtering regardless of input rate
            TARGET_THD_RATE = 1.0  # Hz - sufficient for tidal frequencies (~1e-5 Hz)
            if sample_rate > TARGET_THD_RATE * 1.5:
                from scipy.signal import resample_poly
                from math import gcd
                # Compute rational resampling factors
                # E.g., 40Hz -> 1Hz: up=1, down=40
                # E.g., 20Hz -> 1Hz: up=1, down=20
                up = int(TARGET_THD_RATE * 100)  # Scale to avoid float issues
                down = int(sample_rate * 100)
                common = gcd(up, down)
                up //= common
                down //= common
                logger.debug(f"Resampling {sample_rate:.1f}Hz -> {TARGET_THD_RATE}Hz (up={up}, down={down})")
                data = resample_poly(data, up, down)
                sample_rate = TARGET_THD_RATE

            # Compute THD
            thd_result = self.thd_analyzer.analyze_window(
                data=data,
                sample_rate=sample_rate,
                station=f'{station_network}.{station_code}',
                window_time=date
            )

            # Get station baseline if available
            baseline = get_baseline(station_code, station_network) if get_baseline else None

            if baseline:
                # Use baseline-aware risk calculation
                risk, z_score = thd_to_risk_with_baseline(
                    thd_result.thd_value,
                    baseline.mean_thd,
                    baseline.std_thd
                )
                # Include full baseline context for diagnostics
                notes = (f'sta={station_network}.{station_code}, '
                        f'THD={thd_result.thd_value:.4f}, z={z_score:.2f}, '
                        f'baseline_mean={baseline.mean_thd:.4f}, baseline_std={baseline.std_thd:.4f}, '
                        f'n={baseline.n_samples}, rate={native_sample_rate:.0f}Hz')
            else:
                # Fall back to absolute thresholds
                risk = thd_to_risk(thd_result.thd_value)
                z_score = 0.0
                notes = f'sta={station_network}.{station_code}, THD={thd_result.thd_value:.4f}, rate={native_sample_rate:.0f}Hz (no baseline)'

            return MethodResult(
                name='seismic_thd',
                available=True,
                raw_value=thd_result.thd_value,
                raw_secondary=thd_result.snr,
                risk_score=risk,
                is_elevated=risk >= 0.5,
                is_critical=risk >= 0.75,
                notes=notes
            )

        except Exception as e:
            logger.warning(f"THD analysis failed: {e}")
            return MethodResult(
                name='seismic_thd',
                available=False,
                raw_value=0.0,
                notes=f'Error: {str(e)}'
            )

    def compute_confidence(self, components: Dict[str, MethodResult]) -> Tuple[float, str]:
        """
        Compute confidence score based on method agreement.

        Returns:
            Tuple of (confidence, agreement_type)
        """
        available = [c for c in components.values() if c.available]
        n_available = len(available)

        if n_available == 0:
            return 0.0, 'no_data'

        if n_available == 1:
            return 0.5, 'single_method'

        # Check agreement
        elevated_count = sum(1 for c in available if c.is_elevated)
        critical_count = sum(1 for c in available if c.is_critical)

        if critical_count == n_available:
            return 0.95, 'all_critical'
        elif elevated_count == n_available:
            return 0.85, 'all_elevated'
        elif elevated_count == 0:
            return 0.80, 'all_normal'
        elif elevated_count >= n_available - 1:
            return 0.75, 'mostly_elevated'
        else:
            return 0.60, 'mixed'

    def get_tier(self, risk: float) -> Tuple[int, str]:
        """Get tier number and name from risk score."""
        for tier, info in RISK_TIERS.items():
            if info['min_risk'] <= risk < info['max_risk']:
                return tier, info['name']
        return 3, 'CRITICAL'  # Default to critical for risk >= 1.0

    def compute_risk(
        self,
        target_date: datetime,
        thd_station: str = 'CCC',
        thd_network: str = 'CI'
    ) -> EnsembleResult:
        """
        Compute combined risk assessment.

        Args:
            target_date: Date/time to assess
            thd_station: Station code for THD analysis
            thd_network: Network code for THD station (e.g., 'IU', 'BK', 'CI')

        Returns:
            EnsembleResult with combined risk and components
        """
        logger.info(f"Computing ensemble risk for {self.region} at {target_date}")

        # Compute each method
        components = {}

        # Lambda_geo
        components['lambda_geo'] = self.compute_lambda_geo_risk(target_date)

        # Fault Correlation (with coverage tracking)
        fc_result, segments_defined, segments_working, segment_names = \
            self.compute_fault_correlation_risk(target_date)
        components['fault_correlation'] = fc_result

        # THD
        components['seismic_thd'] = self.compute_thd_risk(
            target_date,
            station_network=thd_network,
            station_code=thd_station
        )

        # Compute weighted risk and track effective weights
        total_weight = 0.0
        weighted_risk = 0.0
        effective_weights = {}

        for name, result in components.items():
            if result.available:
                weight = self.weights.get(name, 0.0)
                weighted_risk += weight * result.risk_score
                total_weight += weight
                effective_weights[name] = weight

        # Renormalize weights for output
        if total_weight > 0:
            combined_risk = weighted_risk / total_weight
            for name in effective_weights:
                effective_weights[name] = effective_weights[name] / total_weight
        else:
            combined_risk = 0.0

        # Normalize to account for missing methods
        # Boost risk slightly if multiple methods agree on elevated
        confidence, agreement = self.compute_confidence(components)

        if agreement in ['all_critical', 'all_elevated']:
            combined_risk = min(1.0, combined_risk * 1.1)

        # Get tier
        tier, tier_name = self.get_tier(combined_risk)

        methods_available = sum(1 for c in components.values() if c.available)

        # DEGRADED STATE: No methods available = cannot assess
        # This prevents "NORMAL" being displayed when we actually have no data
        tier_downgraded = False
        original_tier = tier
        notes = ""

        if methods_available < MIN_METHODS_FOR_OPERATIONAL:
            tier = -1
            tier_name = 'DEGRADED'
            notes = f"Insufficient data: {methods_available} methods available, need >={MIN_METHODS_FOR_OPERATIONAL}"
            logger.warning(f"Region {self.region} in DEGRADED state: no methods available")

        # TIER GATING: Require >=2 methods for Tier >=2 (ELEVATED/CRITICAL)
        # A single method should not drive high-tier alerts alone
        elif methods_available < 2 and tier >= 2:
            tier = 1  # Cap at WATCH
            tier_name = 'WATCH'
            tier_downgraded = True
            logger.warning(f"Tier downgraded from {original_tier} to 1 (WATCH): "
                          f"only {methods_available} method(s) available, need >=2 for ELEVATED/CRITICAL")
            notes = f"Tier capped at WATCH (was {RISK_TIERS[original_tier]['name']}): need >=2 methods for ELEVATED+"

        result = EnsembleResult(
            region=self.region,
            date=target_date,
            combined_risk=combined_risk,
            tier=tier,
            tier_name=tier_name,
            components=components,
            confidence=confidence,
            agreement=agreement,
            methods_available=methods_available,
            notes=notes,
            # Coverage tracking
            segments_defined=segments_defined,
            segments_working=segments_working,
            segment_names=segment_names,
            effective_weights=effective_weights,
        )

        logger.info(f"  Combined risk: {combined_risk:.3f} ({tier_name})")
        if tier_downgraded:
            logger.info(f"  ** TIER CAPPED: {notes}")
        logger.info(f"  Confidence: {confidence:.2f} ({agreement})")
        logger.info(f"  Coverage: {segments_working}/{segments_defined} segments")
        logger.info(f"  Effective weights: {effective_weights}")

        return result

    def compute_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        step_hours: int = 12,
        lambda_geo_data: Optional[Dict[str, float]] = None,
        thd_station: str = 'CCC'
    ) -> List[EnsembleResult]:
        """
        Compute ensemble risk time series.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            step_hours: Step between assessments
            lambda_geo_data: Dict mapping date strings to ratios
            thd_station: Station for THD analysis

        Returns:
            List of EnsembleResult objects
        """
        # Load Lambda_geo data if provided
        if lambda_geo_data:
            for date_str, ratio in lambda_geo_data.items():
                date = datetime.fromisoformat(date_str)
                self.set_lambda_geo(date, ratio)

        results = []
        current = start_date

        while current <= end_date:
            result = self.compute_risk(current, thd_station)
            results.append(result)
            current += timedelta(hours=step_hours)

        return results


def run_ensemble_test():
    """Quick test of ensemble with synthetic data."""
    print("=" * 60)
    print("GeoSpec Ensemble Test")
    print("=" * 60)

    ensemble = GeoSpecEnsemble(region='ridgecrest')

    # Set some test Lambda_geo values
    ensemble.set_lambda_geo(datetime(2019, 7, 4, 12, 0), 5489.0)  # Ridgecrest value

    # Test risk conversion functions
    print("\nRisk Conversion Tests:")
    print("-" * 40)

    # Lambda_geo (1000× saturation)
    for ratio in [1.0, 3.0, 10.0, 100.0, 500.0, 1000.0, 5489.0]:
        risk = lambda_geo_to_risk(ratio)
        print(f"  Lambda_geo ratio={ratio:>7.1f}x -> risk={risk:.3f}")

    print()

    # Fault Correlation
    for l2_l1 in [0.5, 0.3, 0.1, 0.05, 0.02]:
        risk = fault_correlation_to_risk(l2_l1, 1.5)
        print(f"  Fault Corr L2/L1={l2_l1:.2f} -> risk={risk:.3f}")

    print()

    # THD
    for thd in [0.02, 0.05, 0.10, 0.15, 0.30, 1.82]:
        risk = thd_to_risk(thd)
        print(f"  THD={thd:.2f} -> risk={risk:.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ensemble_test()
