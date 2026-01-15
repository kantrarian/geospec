"""
fault_correlation.py
Fault Correlation Dynamics Analysis for Earthquake Precursor Detection.

Physical Basis:
- Normal state: Fault segments are correlated (stress distributed)
- Pre-earthquake: Segments decouple as stress concentrates
- Observable: Eigenvalue ratios DROP before rupture

This module computes:
1. Correlation matrix between fault segment activities
2. Eigenvalue spectrum (especially λ2/λ1 ratio)
3. Participation ratio (stress distribution measure)
4. Decorrelation detection (precursor signal)

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

from fault_segments import FaultSegment, get_segments_for_region
from seismic_data import SeismicDataFetcher, compute_segment_activity_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for fault correlation analysis results."""
    region: str
    date: datetime
    segment_names: List[str]
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray  # λ₁, λ₂, ... (descending)
    eigenvalue_ratios: np.ndarray  # λᵢ/λ₁ for i>1
    participation_ratio: float  # (Σλᵢ)² / Σλᵢ²
    dominant_eigenvector: np.ndarray
    is_decorrelated: bool
    decorrelation_threshold: float
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'region': self.region,
            'date': self.date.isoformat(),
            'segment_names': self.segment_names,
            'eigenvalues': self.eigenvalues.tolist(),
            'eigenvalue_ratios': self.eigenvalue_ratios.tolist(),
            'participation_ratio': float(self.participation_ratio),
            'lambda2_lambda1': float(self.eigenvalue_ratios[0]) if len(self.eigenvalue_ratios) > 0 else 0.0,
            'is_decorrelated': bool(self.is_decorrelated),
            'notes': self.notes,
        }


class FaultCorrelationMonitor:
    """
    Monitors fault segment correlation dynamics for earthquake precursors.

    The key diagnostic is the eigenvalue spectrum of the correlation matrix:
    - High λ2/λ1: Segments coupled, stress distributed (normal)
    - Low λ2/λ1: Segments decoupled, stress concentrated (pre-earthquake)

    Attributes:
        data_fetcher: SeismicDataFetcher for waveform acquisition
        window_hours: Window size for correlation computation (default 24h)
        step_hours: Step size for time evolution (default 6h)
        decorrelation_threshold: λ2/λ1 threshold for alert (default 0.3)
    """

    def __init__(
        self,
        data_fetcher: Optional[SeismicDataFetcher] = None,
        window_hours: int = 24,
        step_hours: int = 6,
        decorrelation_threshold: float = 0.3,
    ):
        """
        Initialize the FaultCorrelationMonitor.

        Args:
            data_fetcher: SeismicDataFetcher instance (created if None)
            window_hours: Window size for correlation in hours
            step_hours: Step size for evolution in hours
            decorrelation_threshold: λ2/λ1 ratio below which to flag decorrelation
        """
        self.data_fetcher = data_fetcher or SeismicDataFetcher()
        self.window_hours = window_hours
        self.step_hours = step_hours
        self.decorrelation_threshold = decorrelation_threshold

        logger.info(f"FaultCorrelationMonitor initialized: "
                   f"window={window_hours}h, step={step_hours}h, "
                   f"threshold={decorrelation_threshold}")

    def compute_segment_activity(
        self,
        segment: FaultSegment,
        start: datetime,
        end: datetime
    ) -> Optional[np.ndarray]:
        """
        Compute aggregate seismic activity for a fault segment.

        Args:
            segment: FaultSegment object
            start: Start datetime
            end: End datetime

        Returns:
            Activity time series or None if insufficient data
        """
        envelopes = self.data_fetcher.get_segment_envelopes(segment, start, end)

        if len(envelopes) < 2:
            logger.warning(f"Insufficient stations for {segment.name}: {len(envelopes)}")
            return None

        activity = compute_segment_activity_index(envelopes)

        if len(activity) == 0:
            return None

        return activity

    def compute_correlation_matrix(
        self,
        region: str,
        start: datetime,
        end: datetime
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Compute correlation matrix between fault segments.

        Args:
            region: Region name
            start: Start datetime
            end: End datetime

        Returns:
            Tuple of (correlation_matrix, segment_names) or (None, [])
        """
        segments = get_segments_for_region(region)

        # Get activity for each segment
        activities = {}
        for segment in segments:
            activity = self.compute_segment_activity(segment, start, end)
            if activity is not None:
                activities[segment.name] = activity

        if len(activities) < 2:
            logger.warning(f"Insufficient segments with data for {region}: {len(activities)}")
            return None, []

        # Align to minimum length
        min_len = min(len(a) for a in activities.values())
        segment_names = list(activities.keys())
        n_segments = len(segment_names)

        # Build activity matrix (rows=segments, cols=time)
        A = np.zeros((n_segments, min_len))
        for i, name in enumerate(segment_names):
            A[i, :] = activities[name][:min_len]

        # Normalize rows (zero mean, unit variance)
        A = A - A.mean(axis=1, keepdims=True)
        std = A.std(axis=1, keepdims=True)
        std[std < 1e-10] = 1.0  # Avoid division by zero
        A = A / std

        # Compute correlation matrix
        C = np.corrcoef(A)

        # Handle NaN (can happen with constant signals)
        C = np.nan_to_num(C, nan=0.0)

        return C, segment_names

    def analyze_eigenvalue_spectrum(
        self,
        C: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Analyze eigenvalue spectrum of correlation matrix.

        Args:
            C: Correlation matrix

        Returns:
            Tuple of (eigenvalues, ratios, participation_ratio, dominant_eigenvector)
        """
        # Compute eigenvalues (symmetric matrix -> use eigh for stability)
        eigenvalues, eigenvectors = eigh(C)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Ensure positive (numerical noise can make small negative)
        eigenvalues = np.maximum(eigenvalues, 0)

        # Eigenvalue ratios (λᵢ/λ₁)
        if eigenvalues[0] > 1e-10:
            ratios = eigenvalues[1:] / eigenvalues[0]
        else:
            ratios = np.zeros(len(eigenvalues) - 1)

        # Participation ratio: (Σλᵢ)² / Σλᵢ²
        # Measures how distributed the stress is
        # PR = n means uniform distribution, PR = 1 means concentrated
        sum_lambda = eigenvalues.sum()
        sum_lambda_sq = (eigenvalues ** 2).sum()
        if sum_lambda_sq > 1e-10:
            participation_ratio = (sum_lambda ** 2) / sum_lambda_sq
        else:
            participation_ratio = 1.0

        # Dominant eigenvector
        dominant_eigenvector = eigenvectors[:, 0]

        return eigenvalues, ratios, participation_ratio, dominant_eigenvector

    def detect_decorrelation(
        self,
        baseline_ratios: np.ndarray,
        current_ratios: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Detect decorrelation by comparing current to baseline.

        Args:
            baseline_ratios: Baseline eigenvalue ratios
            current_ratios: Current eigenvalue ratios
            threshold: Decorrelation threshold (uses instance default if None)

        Returns:
            Tuple of (is_decorrelated, drop_factor)
        """
        threshold = threshold or self.decorrelation_threshold

        if len(baseline_ratios) == 0 or len(current_ratios) == 0:
            return False, 1.0

        # Compare λ2/λ1 ratios
        baseline_ratio = baseline_ratios[0]  # λ2/λ1
        current_ratio = current_ratios[0]

        if baseline_ratio < 1e-10:
            return False, 1.0

        drop_factor = current_ratio / baseline_ratio

        # Decorrelated if ratio dropped significantly
        is_decorrelated = current_ratio < threshold

        return is_decorrelated, drop_factor

    def analyze_region(
        self,
        region: str,
        target_date: datetime,
        baseline_days: int = 30,
        gap_days: int = 7
    ) -> CorrelationResult:
        """
        Perform full correlation analysis for a region.

        Args:
            region: Region name
            target_date: Date to analyze
            baseline_days: Days to use for baseline
            gap_days: Gap between baseline and current (avoid contamination)

        Returns:
            CorrelationResult with full analysis
        """
        logger.info(f"Analyzing {region} for {target_date.date()}")

        # Current window
        current_start = target_date - timedelta(hours=self.window_hours)
        current_end = target_date

        # Baseline window
        baseline_end = target_date - timedelta(days=gap_days)
        baseline_start = baseline_end - timedelta(days=baseline_days)

        # Compute current correlation
        C_current, segment_names = self.compute_correlation_matrix(
            region, current_start, current_end
        )

        if C_current is None:
            return CorrelationResult(
                region=region,
                date=target_date,
                segment_names=[],
                correlation_matrix=np.array([]),
                eigenvalues=np.array([]),
                eigenvalue_ratios=np.array([]),
                participation_ratio=0.0,
                dominant_eigenvector=np.array([]),
                is_decorrelated=False,
                decorrelation_threshold=self.decorrelation_threshold,
                notes="Insufficient data for current window"
            )

        # Analyze eigenvalues
        eigenvalues, ratios, participation_ratio, dominant_eigenvector = \
            self.analyze_eigenvalue_spectrum(C_current)

        # Check for decorrelation
        # For real-time, we compare against absolute threshold
        # For validation, we'd compute baseline and compare
        is_decorrelated = ratios[0] < self.decorrelation_threshold if len(ratios) > 0 else False

        result = CorrelationResult(
            region=region,
            date=target_date,
            segment_names=segment_names,
            correlation_matrix=C_current,
            eigenvalues=eigenvalues,
            eigenvalue_ratios=ratios,
            participation_ratio=participation_ratio,
            dominant_eigenvector=dominant_eigenvector,
            is_decorrelated=is_decorrelated,
            decorrelation_threshold=self.decorrelation_threshold,
            notes=""
        )

        logger.info(f"  Segments: {len(segment_names)}")
        logger.info(f"  L2/L1: {ratios[0]:.4f}" if len(ratios) > 0 else "  L2/L1: N/A")
        logger.info(f"  Participation ratio: {participation_ratio:.2f}")
        logger.info(f"  Decorrelated: {is_decorrelated}")

        return result

    def compute_evolution(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CorrelationResult]:
        """
        Compute time evolution of correlation metrics.

        Args:
            region: Region name
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of CorrelationResult for each time step
        """
        results = []
        current = start_date + timedelta(hours=self.window_hours)

        while current <= end_date:
            result = self.analyze_region(region, current)
            results.append(result)
            current += timedelta(hours=self.step_hours)

        return results


def run_ridgecrest_validation():
    """
    Validate fault correlation on 2019 Ridgecrest sequence.

    Expected: λ2/λ1 should DROP before the M6.4 foreshock and M7.1 mainshock.
    """
    print("=" * 60)
    print("Ridgecrest Fault Correlation Validation")
    print("=" * 60)

    monitor = FaultCorrelationMonitor(
        window_hours=24,
        step_hours=12,
        decorrelation_threshold=0.3
    )

    # Key dates:
    # M6.4 foreshock: 2019-07-04 17:33:49 UTC
    # M7.1 mainshock: 2019-07-06 03:19:53 UTC

    # Analyze period before foreshock
    print("\n--- Pre-Foreshock Period ---")
    pre_foreshock = datetime(2019, 7, 4, 12, 0)  # 5 hours before M6.4
    result = monitor.analyze_region('ridgecrest', pre_foreshock)

    if result.segment_names:
        print(f"Segments: {result.segment_names}")
        print(f"L2/L1 ratio: {result.eigenvalue_ratios[0]:.4f}" if len(result.eigenvalue_ratios) > 0 else "N/A")
        print(f"Participation ratio: {result.participation_ratio:.2f}")
        print(f"Decorrelated: {result.is_decorrelated}")
    else:
        print("Could not retrieve sufficient data for Ridgecrest")
        print("NOTE: CI network data requires SCEDC access which may be rate-limited")

    return result


if __name__ == "__main__":
    result = run_ridgecrest_validation()
