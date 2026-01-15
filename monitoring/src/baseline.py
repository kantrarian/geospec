"""
baseline.py - Rolling Baseline + Seasonal Detrending

Real-time baseline computation using:
- 90-day lookback window
- 14-day exclusion gap (avoid signal contamination)
- Robust statistics (median, MAD)
- Optional seasonal detrending (annual + semiannual)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime, timedelta


@dataclass
class BaselineResult:
    """Result of baseline computation."""
    median: float
    mad: float  # Median Absolute Deviation
    robust_std: float  # MAD * 1.4826
    mean: float
    std: float
    n_days: int
    window_start: datetime
    window_end: datetime
    seasonal_removed: bool
    seasonal_amplitude: Optional[float] = None
    
    def threshold(self, factor: float) -> float:
        """Compute threshold as factor × median."""
        return factor * self.median
    
    def zscore(self, value: float) -> float:
        """Compute Z-score using robust std."""
        if self.robust_std > 0:
            return (value - self.median) / self.robust_std
        return 0.0
    
    def to_dict(self) -> dict:
        return {
            'median': self.median,
            'mad': self.mad,
            'robust_std': self.robust_std,
            'mean': self.mean,
            'std': self.std,
            'n_days': self.n_days,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'seasonal_removed': self.seasonal_removed,
            'seasonal_amplitude': self.seasonal_amplitude,
        }


class RollingBaseline:
    """
    Computes rolling baseline for real-time monitoring.
    
    Timeline:
    [-------- lookback_days --------][-- exclude --][current]
    ^                               ^               ^
    baseline_start            baseline_end      today
    """
    
    def __init__(self,
                 lookback_days: int = 90,
                 exclude_recent_days: int = 14,
                 seasonal_detrend: bool = True,
                 min_data_fraction: float = 0.5):
        """
        Args:
            lookback_days: Days of history for baseline (default 90)
            exclude_recent_days: Gap between baseline and current (default 14)
            seasonal_detrend: Whether to fit/remove annual+semiannual
            min_data_fraction: Minimum fraction of non-NaN data required
        """
        self.lookback_days = lookback_days
        self.exclude_recent_days = exclude_recent_days
        self.seasonal_detrend = seasonal_detrend
        self.min_data_fraction = min_data_fraction
    
    def compute(self, 
                values: np.ndarray, 
                dates: List[datetime],
                current_date: Optional[datetime] = None) -> Optional[BaselineResult]:
        """
        Compute baseline from historical data.
        
        Args:
            values: Array of Lambda_geo values (one per day)
            dates: Corresponding dates
            current_date: Reference date (default: last date in series)
            
        Returns:
            BaselineResult or None if insufficient data
        """
        if len(values) != len(dates):
            raise ValueError("values and dates must have same length")
        
        if current_date is None:
            current_date = dates[-1]
        
        # Convert to arrays
        values = np.asarray(values, dtype=float)
        dates = list(dates)
        
        # Find baseline window indices
        baseline_end_date = current_date - timedelta(days=self.exclude_recent_days)
        baseline_start_date = baseline_end_date - timedelta(days=self.lookback_days)
        
        # Select data in window
        mask = np.array([baseline_start_date <= d <= baseline_end_date for d in dates])
        baseline_values = values[mask]
        baseline_dates = [d for d, m in zip(dates, mask) if m]
        
        # Check sufficient data
        valid_mask = ~np.isnan(baseline_values)
        if np.sum(valid_mask) / len(baseline_values) < self.min_data_fraction:
            return None
        
        # Remove seasonal component if requested
        seasonal_amplitude = None
        if self.seasonal_detrend and len(baseline_values) >= 60:
            baseline_values, seasonal_amplitude = self._remove_seasonality(
                baseline_values, baseline_dates
            )
        
        # Compute robust statistics
        valid_values = baseline_values[valid_mask]
        
        median = np.nanmedian(valid_values)
        mad = np.nanmedian(np.abs(valid_values - median))
        robust_std = 1.4826 * mad  # Convert MAD to std estimate
        
        mean = np.nanmean(valid_values)
        std = np.nanstd(valid_values)
        
        return BaselineResult(
            median=float(median),
            mad=float(mad),
            robust_std=float(robust_std),
            mean=float(mean),
            std=float(std),
            n_days=int(np.sum(valid_mask)),
            window_start=baseline_start_date,
            window_end=baseline_end_date,
            seasonal_removed=self.seasonal_detrend and len(baseline_values) >= 60,
            seasonal_amplitude=seasonal_amplitude,
        )
    
    def _remove_seasonality(self, 
                            values: np.ndarray, 
                            dates: List[datetime]) -> Tuple[np.ndarray, float]:
        """
        Fit and remove annual + semiannual harmonic components.
        
        Model: y = a + b*sin(2πt) + c*cos(2πt) + d*sin(4πt) + e*cos(4πt)
        where t is decimal year.
        """
        # Convert dates to decimal year
        decimal_years = np.array([
            d.year + (d.timetuple().tm_yday - 1) / 365.25 
            for d in dates
        ])
        
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 10:
            return values, 0.0
        
        # Build design matrix
        t = decimal_years[valid_mask]
        y = values[valid_mask]
        
        X = np.column_stack([
            np.ones(len(t)),                    # Constant
            np.sin(2 * np.pi * t),              # Annual sine
            np.cos(2 * np.pi * t),              # Annual cosine
            np.sin(4 * np.pi * t),              # Semiannual sine
            np.cos(4 * np.pi * t),              # Semiannual cosine
        ])
        
        # Fit via least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            return values, 0.0
        
        # Compute seasonal model for all points (including NaN positions)
        X_full = np.column_stack([
            np.ones(len(decimal_years)),
            np.sin(2 * np.pi * decimal_years),
            np.cos(2 * np.pi * decimal_years),
            np.sin(4 * np.pi * decimal_years),
            np.cos(4 * np.pi * decimal_years),
        ])
        
        seasonal_model = X_full @ coeffs
        
        # Compute amplitude of seasonal variation
        # Annual amplitude = sqrt(b^2 + c^2)
        annual_amp = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
        semiannual_amp = np.sqrt(coeffs[3]**2 + coeffs[4]**2)
        seasonal_amplitude = float(annual_amp + semiannual_amp)
        
        # Remove seasonal, preserve median level
        detrended = values - seasonal_model + np.nanmedian(values)
        
        return detrended, seasonal_amplitude


class CommonModeRemoval:
    """
    Remove common-mode error from station displacements.
    
    GPS networks often share regional atmospheric/processing artifacts.
    Subtracting the network mean (or PC1) reduces fake spikes.
    """
    
    def __init__(self, method: str = 'mean'):
        """
        Args:
            method: 'mean' (network average) or 'pca' (first principal component)
        """
        self.method = method
    
    def remove(self, displacements: np.ndarray) -> np.ndarray:
        """
        Remove common-mode from displacement matrix.
        
        Args:
            displacements: Array of shape (n_times, n_stations, n_components)
            
        Returns:
            Cleaned displacements with same shape
        """
        n_times, n_stations, n_comp = displacements.shape
        cleaned = np.zeros_like(displacements)
        
        for c in range(n_comp):
            data = displacements[:, :, c]  # (n_times, n_stations)
            
            if self.method == 'mean':
                # Simple network mean
                common_mode = np.nanmean(data, axis=1, keepdims=True)
            elif self.method == 'pca':
                # First principal component
                common_mode = self._pca_common_mode(data)
            else:
                common_mode = 0
            
            cleaned[:, :, c] = data - common_mode
        
        return cleaned
    
    def _pca_common_mode(self, data: np.ndarray) -> np.ndarray:
        """Extract first PC as common mode."""
        # Handle NaN by filling with column mean
        data_filled = data.copy()
        col_means = np.nanmean(data, axis=0)
        for j in range(data.shape[1]):
            mask = np.isnan(data_filled[:, j])
            data_filled[mask, j] = col_means[j]
        
        # Center
        data_centered = data_filled - np.mean(data_filled, axis=0)
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
            # First PC contribution
            pc1 = U[:, 0:1] @ np.diag(S[0:1]) @ Vt[0:1, :]
            # Common mode = mean of PC1 contribution across stations
            common_mode = np.mean(pc1, axis=1, keepdims=True)
        except Exception:
            common_mode = np.nanmean(data, axis=1, keepdims=True)
        
        return common_mode


# === Unit Tests ===

def test_baseline_with_seasonality():
    """Test that seasonal detrending works correctly."""
    np.random.seed(42)
    
    # Generate 365 days of data with seasonal pattern
    n_days = 365
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Seasonal signal: annual + semiannual
    decimal_years = np.array([d.timetuple().tm_yday / 365.25 for d in dates])
    seasonal = 0.1 * np.sin(2 * np.pi * decimal_years) + 0.05 * np.cos(4 * np.pi * decimal_years)
    
    # Add noise
    noise = 0.02 * np.random.randn(n_days)
    
    # Total signal
    values = 1.0 + seasonal + noise
    
    # Compute baseline
    baseline = RollingBaseline(lookback_days=90, exclude_recent_days=14, seasonal_detrend=True)
    result = baseline.compute(values, dates, current_date=dates[-1])
    
    print(f"Seasonal test:")
    print(f"  Raw std: {np.std(values):.4f}")
    print(f"  Baseline robust_std: {result.robust_std:.4f}")
    print(f"  Seasonal amplitude: {result.seasonal_amplitude:.4f}")
    print(f"  Expected seasonal: ~0.15")
    
    # Check that detrending reduced variability
    assert result.robust_std < np.std(values) * 0.5, "Detrending should reduce variability"
    assert result.seasonal_amplitude > 0.1, "Should detect seasonal signal"
    
    print("  ✓ Test passed")


def test_baseline_step_detection():
    """Test that step artifacts produce high values."""
    np.random.seed(42)
    
    n_days = 120
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Baseline period: stable
    values = np.ones(n_days) * 0.1 + 0.01 * np.random.randn(n_days)
    
    # Add step artifact on day 100
    values[100:] += 0.5
    
    # Compute baseline (excluding last 14 days where step occurred)
    baseline = RollingBaseline(lookback_days=90, exclude_recent_days=14)
    result = baseline.compute(values, dates, current_date=dates[-1])
    
    # Current value should be very high relative to baseline
    current_value = values[-1]
    zscore = result.zscore(current_value)
    
    print(f"Step detection test:")
    print(f"  Baseline median: {result.median:.4f}")
    print(f"  Current value: {current_value:.4f}")
    print(f"  Z-score: {zscore:.1f}")
    
    assert zscore > 10, "Step should produce high Z-score"
    print("  ✓ Test passed")


if __name__ == "__main__":
    print("Running baseline tests...\n")
    test_baseline_with_seasonality()
    print()
    test_baseline_step_detection()
    print("\nAll tests passed!")
