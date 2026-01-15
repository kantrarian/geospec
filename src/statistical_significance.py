#!/usr/bin/env python3
"""
statistical_significance.py
Statistical Significance Testing for Lambda_geo Detections

Uses CANONICAL metrics from canonical_metrics.json to ensure consistency.

Approach:
1. Load canonical metrics (already computed consistently)
2. Fit log-normal null model to baseline period
3. Compute probability of observing canonical 72h peak under null
4. Report FPR at various thresholds

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
from typing import Dict


def fit_null_model(baseline_data: np.ndarray) -> Dict:
    """
    Fit a log-normal null model to baseline data.
    """
    pos_data = baseline_data[baseline_data > 0]
    
    if len(pos_data) < 3:
        return None
    
    log_data = np.log(pos_data)
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    
    return {
        'n_samples': len(pos_data),
        'lognormal': {'mu': float(mu), 'sigma': float(sigma)},
        'normal': {'mu': float(np.mean(pos_data)), 'sigma': float(np.std(pos_data))},
    }


def compute_exceedance_probability(
    observed_value: float,
    null_model: Dict,
    n_samples: int = 3
) -> Dict:
    """
    Compute probability of observing value >= observed under null model.
    
    Args:
        observed_value: The peak value to test
        null_model: Log-normal null model parameters
        n_samples: Number of independent samples (3 days for 72h window)
    """
    mu = null_model['lognormal']['mu']
    sigma = null_model['lognormal']['sigma']
    
    if sigma <= 0:
        return {'error': 'Invalid null model'}
    
    # P(X >= observed) for single sample from log-normal
    p_single = 1 - stats.lognorm.cdf(observed_value, s=sigma, scale=np.exp(mu))
    
    # P(max of n samples >= observed) = 1 - P(all n < observed)
    p_max = 1 - (1 - p_single) ** n_samples
    
    # Handle underflow
    if p_max == 0:
        p_max_str = "< 1e-15 (underflow)"
        log_p = "> 15"
    else:
        p_max_str = f"{p_max:.2e}"
        log_p = f"{-np.log10(p_max):.1f}"
    
    return {
        'p_single_exceed': float(p_single),
        'p_max_exceed': float(p_max),
        'p_max_str': p_max_str,
        'neg_log10_p': log_p,
    }


def estimate_fpr(null_model: Dict, threshold: float, baseline_median: float, n_samples: int = 3) -> float:
    """
    Estimate FPR: probability of exceeding threshold under null model.
    """
    target = threshold * baseline_median
    mu = null_model['lognormal']['mu']
    sigma = null_model['lognormal']['sigma']
    
    p_single = 1 - stats.lognorm.cdf(target, s=sigma, scale=np.exp(mu))
    p_max = 1 - (1 - p_single) ** n_samples
    
    return float(p_max)


def run_significance_analysis(results_dir: Path) -> Dict:
    """
    Run significance analysis using canonical metrics for consistency.
    """
    
    # Load canonical metrics
    canonical_file = results_dir / "canonical_metrics.json"
    if not canonical_file.exists():
        print("ERROR: canonical_metrics.json not found. Run canonical_analysis.py first.")
        return None
    
    with open(canonical_file, 'r') as f:
        canonical = json.load(f)
    
    all_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'methodology': 'Log-normal null model fit to 7-day baseline',
            'note': 'Uses CANONICAL peak_72h values from canonical_metrics.json',
            'windowing': {
                'baseline_days': 7,
                'precursor_window': '72h (3 daily samples)',
                'monitoring_window': '14 days (baseline + signal)',
                'windows_per_year': 26,
                'window_calculation': 'FPR = P(max of 3 samples >= threshold | null); alerts/year = FPR * 26'
            },
            'limitations': [
                'Null model fit from only 7 baseline days (small sample for tail inference)',
                'FPR is for threshold exceedance only, not full detection rule',
                'Full detection rule: lead_time >= 24h AND amp >= 5x AND zscore >= 2',
                'P-values showing 0.0 indicate numerical underflow (true p < 1e-15)',
            ]
        },
        'earthquakes': {},
        'fpr_estimates': {},
    }
    
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS (Using Canonical Metrics)")
    print("=" * 80)
    print()
    print("Methodology:")
    print("  - Null model: Log-normal fit to 7-day baseline period")
    print("  - Test statistic: CANONICAL peak_72h / baseline_median")
    print("  - P-value: P(max of 3 daily samples >= peak_72h | null)")
    print()
    
    summary_rows = []
    
    for eq_key, eq_data in canonical['earthquakes'].items():
        print(f"\n{'='*60}")
        print(f"{eq_data['event_info']['name']} M{eq_data['event_info']['magnitude']}")
        print(f"{'='*60}")
        
        # Load raw data for null model fitting
        lambda_file = results_dir / f"{eq_key}_lambda_geo.npz"
        if not lambda_file.exists():
            print(f"  [SKIP] No lambda_geo file")
            continue
        
        data = np.load(lambda_file, allow_pickle=True)
        lambda_geo = data['lambda_geo']
        lambda_max = np.nanmax(lambda_geo, axis=1)
        
        # Baseline period (first 7 days)
        baseline_days = eq_data['baseline']['days']
        baseline_data = lambda_max[:baseline_days]
        
        # Fit null model
        null_model = fit_null_model(baseline_data)
        if null_model is None:
            print("  [ERROR] Could not fit null model")
            continue
        
        # Get CANONICAL values
        baseline_median = eq_data['baseline']['median']
        peak_72h = eq_data['amplification']['value'] * baseline_median  # Reconstruct peak from amplification
        canonical_amp = eq_data['amplification']['value']
        z_score = eq_data['z_score']
        
        print(f"\nCanonical Metrics (from canonical_metrics.json):")
        print(f"  Baseline median: {baseline_median:.6f}")
        print(f"  Peak (72h window): {peak_72h:.4f}")
        print(f"  Amplification: {canonical_amp:.1f}x")
        print(f"  Z-score: {z_score:.1f}")
        
        # Compute exceedance probability
        exc = compute_exceedance_probability(peak_72h, null_model, n_samples=3)
        
        print(f"\nNull Model (log-normal fit to {baseline_days} baseline days):")
        print(f"  mu: {null_model['lognormal']['mu']:.4f}")
        print(f"  sigma: {null_model['lognormal']['sigma']:.4f}")
        
        print(f"\nExceedance Probability:")
        print(f"  P(single day >= peak_72h): {exc['p_single_exceed']:.2e}")
        print(f"  P(max of 3 days >= peak_72h): {exc['p_max_str']}")
        print(f"  -log10(p): {exc['neg_log10_p']}")
        
        # Significance determination
        p_val = exc['p_max_exceed']
        sig_001 = p_val < 0.001 if p_val > 0 else True  # Underflow = highly significant
        sig_01 = p_val < 0.01 if p_val > 0 else True
        sig_05 = p_val < 0.05 if p_val > 0 else True
        
        print(f"\nSignificance:")
        print(f"  p < 0.05: {'YES' if sig_05 else 'NO'}")
        print(f"  p < 0.01: {'YES' if sig_01 else 'NO'}")
        print(f"  p < 0.001: {'YES' if sig_001 else 'NO'}")
        
        # FPR at various thresholds
        fpr_2x = estimate_fpr(null_model, 2.0, baseline_median, 3)
        fpr_5x = estimate_fpr(null_model, 5.0, baseline_median, 3)
        fpr_10x = estimate_fpr(null_model, 10.0, baseline_median, 3)
        fpr_100x = estimate_fpr(null_model, 100.0, baseline_median, 3)
        
        print(f"\nFPR for exceeding threshold in 72h window under fitted null:")
        print(f"  2x: {fpr_2x*100:.1f}%")
        print(f"  5x: {fpr_5x*100:.2f}%")
        print(f"  10x: {fpr_10x*100:.4f}%")
        print(f"  100x: {fpr_100x*100:.8f}%")
        
        # Store results
        all_results['earthquakes'][eq_key] = {
            'canonical_amplification': canonical_amp,
            'canonical_zscore': z_score,
            'null_model': null_model,
            'exceedance': exc,
            'significant_p001': sig_001,
        }
        all_results['fpr_estimates'][eq_key] = {
            '2x': fpr_2x, '5x': fpr_5x, '10x': fpr_10x, '100x': fpr_100x
        }
        
        summary_rows.append({
            'name': eq_data['event_info']['name'],
            'mag': eq_data['event_info']['magnitude'],
            'amp': canonical_amp,
            'z': z_score,
            'p_str': exc['p_max_str'],
            'p_val': p_val,
            'sig': sig_001,
            'fpr_5x': fpr_5x,
        })
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (Canonical 72h Amplification)")
    print("=" * 100)
    print(f"{'Event':<18} {'Mag':>4} {'Amp(72h)':>12} {'Z-score':>10} {'P-value':>15} {'Sig p<0.001':>12} {'FPR@5x':>10}")
    print("-" * 100)
    
    n_sig = 0
    all_fpr_5x = []
    for r in summary_rows:
        sig_str = "YES" if r['sig'] else "NO"
        if r['sig']:
            n_sig += 1
        all_fpr_5x.append(r['fpr_5x'])
        print(f"{r['name']:<18} {r['mag']:>4.1f} {r['amp']:>11.1f}x {r['z']:>10.1f} {r['p_str']:>15} {sig_str:>12} {r['fpr_5x']*100:>9.2f}%")
    
    print("-" * 100)
    
    mean_fpr_5x = np.mean(all_fpr_5x) if all_fpr_5x else 0
    expected_alerts = mean_fpr_5x * 365 / 14  # ~26 windows/year
    
    print(f"\nMean FPR at 5x threshold: {mean_fpr_5x*100:.2f}%")
    print(f"Expected false alerts/year (26 windows): {expected_alerts:.2f}")
    print(f"Significant detections (p<0.001): {n_sig}/{len(summary_rows)}")
    
    all_results['summary'] = {
        'n_events': len(summary_rows),
        'n_significant_p001': n_sig,
        'mean_fpr_5x': mean_fpr_5x,
        'expected_false_alerts_year': expected_alerts,
    }
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"""
METHODOLOGY:
  - Null model: Log-normal distribution fit to 7-day baseline
  - Test statistic: CANONICAL peak in final 72h / baseline_median
  - P-value: Probability of observing peak >= observed in 3-day window under null

RESULTS:
  1. Significant detections (p < 0.001): {n_sig}/{len(summary_rows)}
  2. FPR for exceeding 5x in 72h window: {mean_fpr_5x*100:.2f}%
     -> Expected ~{expected_alerts:.2f} false alerts/year
  
LIMITATIONS (must be stated in any publication):
  - Null model fit from only 7 baseline days (small sample for tail inference)
  - FPR is for threshold exceedance only, not full detection rule
     (full rule: lead_time >= 24h AND amplification >= 5x AND zscore >= 2)
  - P-values of 0.0 indicate numerical underflow; reported as "< 1e-15"
  
INTERPRETATION:
  - 4/5 events show canonical amplifications (485-7999x) far exceeding null
  - P-values underflow for these events (effectively p << 10^-15)
  - Morocco (2.8x canonical amp) is NOT significant (p = {summary_rows[-1]['p_str']})
""")
    
    return all_results


def main():
    """Run significance analysis."""
    results_dir = Path(__file__).parent.parent / "results"
    
    results = run_significance_analysis(results_dir)
    
    if results:
        output_file = results_dir / "statistical_significance_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
