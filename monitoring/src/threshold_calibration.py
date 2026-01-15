"""
threshold_calibration.py - Calibrate Tier Thresholds to Meet FAR Budget

Implements threshold tuning to meet the global false alarm rate budget.

Target: ≤1 false Tier-2 alert per year globally
Method: Bonferroni correction per region/day

With 9 regions × 365 days = 3,285 tests/year
Per-test α = 0.0003 for 1/year global FAR

This module:
1. Fits distributions to baseline THD data
2. Computes percentile thresholds
3. Generates sensitivity analysis
4. Outputs calibrated threshold config

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# FAR Budget
GLOBAL_FAR_TARGET = 1.0  # 1 false alert per year globally
N_REGIONS = 9            # Number of monitored regions
N_DAYS_PER_YEAR = 365

# Per-test significance level (Bonferroni correction)
# α_per_test = FAR_target / (N_regions × N_days)
# = 1.0 / (9 × 365) = 1.0 / 3285 ≈ 0.0003
ALPHA_PER_TEST = GLOBAL_FAR_TARGET / (N_REGIONS * N_DAYS_PER_YEAR)

# Percentile for threshold (1 - α)
THRESHOLD_PERCENTILE = 100 * (1 - ALPHA_PER_TEST)  # 99.97th percentile


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DistributionFit:
    """Result of fitting a distribution to THD data."""
    station: str
    distribution: str  # 'normal', 'lognormal', 'gamma'
    params: Dict[str, float]
    ks_statistic: float  # Kolmogorov-Smirnov test statistic
    p_value: float       # KS test p-value
    aic: float           # Akaike Information Criterion
    threshold_99_97: float  # 99.97th percentile threshold

    def to_dict(self) -> Dict:
        return {
            'station': self.station,
            'distribution': self.distribution,
            'params': self.params,
            'ks_statistic': float(self.ks_statistic),
            'p_value': float(self.p_value),
            'aic': float(self.aic),
            'threshold_99_97': float(self.threshold_99_97),
        }


@dataclass
class ThresholdConfig:
    """Calibrated threshold configuration."""
    station: str
    z_score_threshold: float      # Z-score threshold for Tier-2
    thd_threshold: float          # Absolute THD threshold
    baseline_mean: float
    baseline_std: float
    distribution_fit: str         # Best-fit distribution name
    calibration_date: str
    far_target: float             # Target FAR
    alpha_per_test: float         # Per-test significance level

    def to_dict(self) -> Dict:
        return {
            'station': self.station,
            'z_score_threshold': float(self.z_score_threshold),
            'thd_threshold': float(self.thd_threshold),
            'baseline_mean': float(self.baseline_mean),
            'baseline_std': float(self.baseline_std),
            'distribution_fit': self.distribution_fit,
            'calibration_date': self.calibration_date,
            'far_target': float(self.far_target),
            'alpha_per_test': float(self.alpha_per_test),
        }


# =============================================================================
# DISTRIBUTION FITTING
# =============================================================================

def fit_distributions(
    thd_values: np.ndarray,
    station: str,
) -> List[DistributionFit]:
    """
    Fit multiple distributions to THD data and return sorted by fit quality.

    Tests: Normal, Lognormal, Gamma distributions.
    """
    if len(thd_values) < 30:
        logger.warning(f"Insufficient data for {station}: {len(thd_values)} points")
        return []

    fits = []

    # Normal distribution
    try:
        mu, sigma = stats.norm.fit(thd_values)
        ks_stat, p_val = stats.kstest(thd_values, 'norm', args=(mu, sigma))
        threshold = stats.norm.ppf(1 - ALPHA_PER_TEST, loc=mu, scale=sigma)
        n_params = 2
        log_likelihood = np.sum(stats.norm.logpdf(thd_values, loc=mu, scale=sigma))
        aic = 2 * n_params - 2 * log_likelihood

        fits.append(DistributionFit(
            station=station,
            distribution='normal',
            params={'mu': mu, 'sigma': sigma},
            ks_statistic=ks_stat,
            p_value=p_val,
            aic=aic,
            threshold_99_97=threshold,
        ))
    except Exception as e:
        logger.debug(f"Normal fit failed for {station}: {e}")

    # Lognormal distribution (requires positive values)
    positive_values = thd_values[thd_values > 0]
    if len(positive_values) > 30:
        try:
            shape, loc, scale = stats.lognorm.fit(positive_values, floc=0)
            ks_stat, p_val = stats.kstest(positive_values, 'lognorm', args=(shape, loc, scale))
            threshold = stats.lognorm.ppf(1 - ALPHA_PER_TEST, shape, loc=loc, scale=scale)
            n_params = 3
            log_likelihood = np.sum(stats.lognorm.logpdf(positive_values, shape, loc=loc, scale=scale))
            aic = 2 * n_params - 2 * log_likelihood

            fits.append(DistributionFit(
                station=station,
                distribution='lognormal',
                params={'shape': shape, 'loc': loc, 'scale': scale},
                ks_statistic=ks_stat,
                p_value=p_val,
                aic=aic,
                threshold_99_97=threshold,
            ))
        except Exception as e:
            logger.debug(f"Lognormal fit failed for {station}: {e}")

    # Gamma distribution (requires positive values)
    if len(positive_values) > 30:
        try:
            a, loc, scale = stats.gamma.fit(positive_values, floc=0)
            ks_stat, p_val = stats.kstest(positive_values, 'gamma', args=(a, loc, scale))
            threshold = stats.gamma.ppf(1 - ALPHA_PER_TEST, a, loc=loc, scale=scale)
            n_params = 3
            log_likelihood = np.sum(stats.gamma.logpdf(positive_values, a, loc=loc, scale=scale))
            aic = 2 * n_params - 2 * log_likelihood

            fits.append(DistributionFit(
                station=station,
                distribution='gamma',
                params={'a': a, 'loc': loc, 'scale': scale},
                ks_statistic=ks_stat,
                p_value=p_val,
                aic=aic,
                threshold_99_97=threshold,
            ))
        except Exception as e:
            logger.debug(f"Gamma fit failed for {station}: {e}")

    # Sort by AIC (lower is better)
    fits.sort(key=lambda f: f.aic)

    return fits


def compute_z_threshold_for_percentile(
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    """
    Compute z-score threshold for given percentile assuming normal distribution.

    For 99.97th percentile: z ≈ 3.43
    """
    return stats.norm.ppf(percentile / 100)


# =============================================================================
# THRESHOLD CALIBRATION
# =============================================================================

def calibrate_station_threshold(
    station: str,
    thd_values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
) -> ThresholdConfig:
    """
    Calibrate threshold for a single station.

    Args:
        station: Station identifier
        thd_values: Array of daily THD values
        baseline_mean: Baseline mean (median) THD
        baseline_std: Baseline std (MAD-based) THD

    Returns:
        ThresholdConfig with calibrated thresholds
    """
    # Fit distributions
    fits = fit_distributions(thd_values, station)

    if not fits:
        # Fall back to empirical percentile
        threshold = np.percentile(thd_values, THRESHOLD_PERCENTILE)
        z_threshold = (threshold - baseline_mean) / baseline_std if baseline_std > 0 else 3.5
        dist_name = 'empirical'
    else:
        # Use best fit
        best_fit = fits[0]
        threshold = best_fit.threshold_99_97
        z_threshold = (threshold - baseline_mean) / baseline_std if baseline_std > 0 else 3.5
        dist_name = best_fit.distribution

        logger.info(f"{station}: Best fit = {dist_name} (AIC={best_fit.aic:.1f}, "
                   f"p={best_fit.p_value:.3f}), threshold={threshold:.4f}")

    return ThresholdConfig(
        station=station,
        z_score_threshold=z_threshold,
        thd_threshold=threshold,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        distribution_fit=dist_name,
        calibration_date=datetime.now().strftime('%Y-%m-%d'),
        far_target=GLOBAL_FAR_TARGET,
        alpha_per_test=ALPHA_PER_TEST,
    )


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(
    thd_values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    z_range: Tuple[float, float] = (2.0, 5.0),
    n_points: int = 20,
) -> List[Dict]:
    """
    Analyze sensitivity of FAR to threshold choice.

    Args:
        thd_values: Array of daily THD values
        baseline_mean: Baseline mean THD
        baseline_std: Baseline std THD
        z_range: Range of z-scores to test
        n_points: Number of points to evaluate

    Returns:
        List of dicts with z_threshold, far_estimate, detection_fraction
    """
    results = []

    z_thresholds = np.linspace(z_range[0], z_range[1], n_points)

    for z in z_thresholds:
        thd_threshold = baseline_mean + z * baseline_std

        # Estimate false alarm rate (fraction of days exceeding threshold)
        n_exceeds = np.sum(thd_values > thd_threshold)
        far_per_day = n_exceeds / len(thd_values)
        far_per_year = far_per_day * N_DAYS_PER_YEAR * N_REGIONS

        results.append({
            'z_threshold': float(z),
            'thd_threshold': float(thd_threshold),
            'fraction_exceeding': float(n_exceeds / len(thd_values)),
            'far_per_year_global': float(far_per_year),
        })

    return results


# =============================================================================
# OUTPUT
# =============================================================================

def save_thresholds_json(
    thresholds: List[ThresholdConfig],
    sensitivity: Dict[str, List[Dict]],
    output_path: Path,
):
    """Save calibrated thresholds to JSON."""
    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'far_target': GLOBAL_FAR_TARGET,
            'n_regions': N_REGIONS,
            'alpha_per_test': ALPHA_PER_TEST,
            'threshold_percentile': THRESHOLD_PERCENTILE,
        },
        'thresholds': {t.station: t.to_dict() for t in thresholds},
        'sensitivity': sensitivity,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved thresholds to {output_path}")


def print_threshold_summary(thresholds: List[ThresholdConfig]):
    """Print threshold summary."""
    print("\n" + "=" * 80)
    print("CALIBRATED THRESHOLDS")
    print("=" * 80)
    print(f"Target: ≤{GLOBAL_FAR_TARGET:.0f} false Tier-2 alerts per year globally")
    print(f"Per-test α: {ALPHA_PER_TEST:.6f} ({THRESHOLD_PERCENTILE:.2f}th percentile)")
    print("-" * 80)
    print(f"{'Station':<12} {'Z-Threshold':>12} {'THD-Threshold':>14} {'Distribution':<12}")
    print("-" * 80)

    for t in thresholds:
        print(f"{t.station:<12} {t.z_score_threshold:>12.2f} {t.thd_threshold:>14.4f} {t.distribution_fit:<12}")

    print("=" * 80)

    # Compute theoretical z-threshold
    theoretical_z = compute_z_threshold_for_percentile()
    print(f"\nTheoretical z-threshold (normal): {theoretical_z:.2f}")
    print("Note: Individual stations may deviate if their distributions are non-normal.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run threshold calibration using baseline data."""
    import argparse

    parser = argparse.ArgumentParser(description='Calibrate THD thresholds for FAR budget')
    parser.add_argument('--baselines', type=str, help='Path to baselines JSON file')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load baseline data
    if args.baselines:
        baselines_path = Path(args.baselines)
    else:
        baselines_path = Path(__file__).parent.parent / 'data' / 'baselines' / 'thd_baselines_latest.json'

    if not baselines_path.exists():
        logger.warning(f"Baselines file not found: {baselines_path}")
        logger.info("Using synthetic data for demonstration...")

        # Generate synthetic baseline data for demo
        stations = ['IU.TUC', 'BK.BKS', 'IU.COR', 'IU.MAJO', 'IU.ANTO', 'IV.CAFE']
        thresholds = []
        sensitivity_data = {}

        for station in stations:
            # Generate synthetic THD values (lognormal-ish)
            np.random.seed(hash(station) % 2**32)
            base_mean = 0.1 + np.random.random() * 0.1
            base_std = base_mean * 0.3
            thd_values = np.random.lognormal(
                mean=np.log(base_mean),
                sigma=0.3,
                size=90
            )

            # Calibrate
            config = calibrate_station_threshold(
                station=station,
                thd_values=thd_values,
                baseline_mean=np.median(thd_values),
                baseline_std=np.median(np.abs(thd_values - np.median(thd_values))) * 1.4826,
            )
            thresholds.append(config)

            # Sensitivity analysis
            sensitivity_data[station] = sensitivity_analysis(
                thd_values,
                config.baseline_mean,
                config.baseline_std,
            )

        print_threshold_summary(thresholds)

        # Save output
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(__file__).parent.parent / 'config' / 'thresholds_demo.json'

        save_thresholds_json(thresholds, sensitivity_data, output_path)

        return 0

    # Load actual baseline data
    with open(baselines_path) as f:
        data = json.load(f)

    thresholds = []
    sensitivity_data = {}

    for baseline in data.get('baselines', []):
        station = baseline.get('station')
        if not station:
            continue

        # Get daily values
        daily_values = baseline.get('daily_values', [])
        if not daily_values:
            logger.warning(f"No daily values for {station}")
            continue

        thd_values = np.array([v[1] for v in daily_values])

        # Calibrate
        config = calibrate_station_threshold(
            station=station,
            thd_values=thd_values,
            baseline_mean=baseline.get('mean_thd', np.median(thd_values)),
            baseline_std=baseline.get('std_thd', 0.05),
        )
        thresholds.append(config)

        # Sensitivity analysis
        sensitivity_data[station] = sensitivity_analysis(
            thd_values,
            config.baseline_mean,
            config.baseline_std,
        )

    print_threshold_summary(thresholds)

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        output_path = Path(__file__).parent.parent / 'config' / f'thresholds_{date_str}.json'

    save_thresholds_json(thresholds, sensitivity_data, output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
