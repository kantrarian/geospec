#!/usr/bin/env python3
"""
validate_lambda_geo.py
Validate Λ_geo diagnostic against historical earthquakes.

Success Criteria:
- Detect anomalous Λ_geo 24-72 hours before earthquake
- False positive rate < 30%
- Spatial localization near epicenter

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from lambda_geo import LambdaGeoAnalyzer, load_strain_data, LambdaGeoResult


class EarthquakeValidator:
    """Validate Λ_geo against known earthquakes."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_metrics(self, 
                        result: LambdaGeoResult,
                        eq_time_idx: int,
                        precursor_window_hours: Tuple[int, int] = (24, 72)
                       ) -> Dict:
        """
        Compute validation metrics.
        
        Args:
            result: Λ_geo analysis result
            eq_time_idx: Time index of earthquake
            precursor_window_hours: (min, max) hours before earthquake to check
            
        Returns:
            Dictionary of validation metrics
        """
        min_hours, max_hours = precursor_window_hours
        dt = result.computation_params['dt_hours']
        
        # Define time windows
        precursor_start = eq_time_idx - int(max_hours / dt)
        precursor_end = eq_time_idx - int(min_hours / dt)
        background_end = eq_time_idx - int(max_hours / dt) - int(48 / dt)  # 48h buffer
        
        # Ensure valid indices
        precursor_start = max(0, precursor_start)
        precursor_end = min(len(result.times) - 1, precursor_end)
        background_end = max(0, background_end)
        
        # Background statistics (before precursor window)
        if background_end > 10:
            bg_lambda = result.lambda_geo[:background_end]
            bg_risk = result.spatial_max_risk[:background_end]
            bg_mean = np.mean(bg_lambda)
            bg_std = np.std(bg_lambda)
            bg_risk_mean = np.mean(bg_risk)
        else:
            bg_mean = np.mean(result.lambda_geo)
            bg_std = np.std(result.lambda_geo)
            bg_risk_mean = np.mean(result.spatial_max_risk)
        
        # Precursor window statistics
        pre_lambda = result.lambda_geo[precursor_start:precursor_end]
        pre_risk = result.spatial_max_risk[precursor_start:precursor_end]
        
        # Key metrics
        metrics = {
            # Signal strength
            'max_lambda_geo_precursor': float(np.max(pre_lambda)),
            'mean_lambda_geo_precursor': float(np.mean(pre_lambda)),
            'max_lambda_geo_background': float(np.max(result.lambda_geo[:background_end])) if background_end > 0 else 0,
            'mean_lambda_geo_background': float(bg_mean),
            'std_lambda_geo_background': float(bg_std),
            
            # Amplification
            'amplification_factor': float(np.max(pre_lambda) / bg_mean) if bg_mean > 0 else 0,
            'zscore_max': float((np.max(pre_lambda) - bg_mean) / bg_std) if bg_std > 0 else 0,
            
            # Risk assessment
            'max_risk_precursor': float(np.max(pre_risk)),
            'mean_risk_precursor': float(np.mean(pre_risk)),
            'mean_risk_background': float(bg_risk_mean),
            'pct_time_high_risk': float(np.mean(pre_risk > 0.7) * 100),
            
            # Temporal detection
            'first_detection_hours_before': None,
            'peak_detection_hours_before': None,
            
            # Spatial localization
            'epicenter_proximity_at_peak': None,
        }
        
        # Find first detection (risk > 0.7)
        high_risk_mask = result.spatial_max_risk > 0.7
        if np.any(high_risk_mask[:eq_time_idx]):
            first_detection_idx = np.where(high_risk_mask[:eq_time_idx])[0]
            if len(first_detection_idx) > 0:
                # Find first detection in precursor window
                valid_detections = first_detection_idx[first_detection_idx >= precursor_start]
                if len(valid_detections) > 0:
                    first_idx = valid_detections[0]
                    metrics['first_detection_hours_before'] = float((eq_time_idx - first_idx) * dt)
        
        # Peak detection time
        if precursor_end > precursor_start:
            peak_idx = precursor_start + np.argmax(pre_risk)
            metrics['peak_detection_hours_before'] = float((eq_time_idx - peak_idx) * dt)
        
        # Spatial localization at peak
        if 'lat' in result.earthquake_info and 'lon' in result.earthquake_info:
            eq_lat = result.earthquake_info['lat']
            eq_lon = result.earthquake_info['lon']
            
            # Find station with max risk at peak time
            if precursor_end > precursor_start:
                peak_time_idx = precursor_start + np.argmax(np.max(pre_lambda, axis=1))
                peak_station_idx = np.argmax(result.lambda_geo[peak_time_idx])
                
                station_lat = result.station_lats[peak_station_idx]
                station_lon = result.station_lons[peak_station_idx]
                
                # Distance in km (approximate)
                dist_km = np.sqrt((station_lat - eq_lat)**2 + (station_lon - eq_lon)**2) * 111
                metrics['epicenter_proximity_at_peak'] = float(dist_km)
        
        return metrics
    
    def create_validation_figure(self,
                                  result: LambdaGeoResult,
                                  eq_time_idx: int,
                                  metrics: Dict,
                                  output_path: Path):
        """Create comprehensive validation figure."""
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(f"Λ_geo Validation: {result.earthquake_info.get('name', 'Unknown')}", 
                     fontsize=14, fontweight='bold')
        
        dt = result.computation_params['dt_hours']
        n_times = len(result.times)
        
        # Convert to hours before earthquake
        hours_before = (np.arange(n_times) - eq_time_idx) * dt
        
        # Precursor window shading
        def add_precursor_shading(ax):
            ax.axvspan(-72, -24, alpha=0.2, color='orange', label='Precursor window')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Earthquake')
        
        # Panel 1: Λ_geo time series
        ax = axes[0]
        # Plot spatial mean and max
        lambda_mean = np.mean(result.lambda_geo, axis=1)
        lambda_max = np.max(result.lambda_geo, axis=1)
        ax.fill_between(hours_before, lambda_mean, lambda_max, alpha=0.3, color='blue')
        ax.plot(hours_before, lambda_max, 'b-', linewidth=1.5, label='Max Λ_geo')
        ax.plot(hours_before, lambda_mean, 'b--', linewidth=1, label='Mean Λ_geo')
        add_precursor_shading(ax)
        ax.set_ylabel('Λ_geo')
        ax.set_title('Strain Tensor Commutator Diagnostic')
        ax.legend(loc='upper left')
        ax.set_xlim(hours_before[0], hours_before[-1])
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Spectral gap
        ax = axes[1]
        gap_mean = np.mean(np.abs(result.spectral_gap_12), axis=1)
        gap_max = np.max(np.abs(result.spectral_gap_12), axis=1)
        ax.fill_between(hours_before, gap_mean, gap_max, alpha=0.3, color='green')
        ax.plot(hours_before, gap_max, 'g-', linewidth=1.5, label='Max |δ|')
        ax.plot(hours_before, gap_mean, 'g--', linewidth=1, label='Mean |δ|')
        add_precursor_shading(ax)
        ax.set_ylabel('Spectral Gap δ')
        ax.set_title('Spectral Gap (λ₁ - λ₂)')
        ax.legend(loc='upper left')
        ax.set_xlim(hours_before[0], hours_before[-1])
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Eigenframe rotation rate
        ax = axes[2]
        rot_mean = np.mean(result.eigenframe_rotation_rate, axis=1)
        rot_max = np.max(result.eigenframe_rotation_rate, axis=1)
        ax.semilogy(hours_before, rot_max, 'purple', linewidth=1.5, label='Max rotation rate')
        ax.semilogy(hours_before, rot_mean, 'purple', linewidth=1, linestyle='--', label='Mean rotation rate')
        add_precursor_shading(ax)
        ax.set_ylabel('Λ_geo / δ')
        ax.set_title('Eigenframe Rotation Rate Bound')
        ax.legend(loc='upper left')
        ax.set_xlim(hours_before[0], hours_before[-1])
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Risk score
        ax = axes[3]
        ax.fill_between(hours_before, 0, result.spatial_max_risk, alpha=0.3, color='red')
        ax.plot(hours_before, result.spatial_max_risk, 'r-', linewidth=2, label='Max Risk')
        ax.axhline(0.7, color='darkred', linestyle=':', label='High risk threshold')
        add_precursor_shading(ax)
        ax.set_xlabel('Hours before earthquake')
        ax.set_ylabel('Risk Score')
        ax.set_title('Ensemble Risk Score')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        ax.set_xlim(hours_before[0], hours_before[-1])
        ax.grid(True, alpha=0.3)
        
        # Add metrics text box
        first_det = metrics.get('first_detection_hours_before')
        peak_det = metrics.get('peak_detection_hours_before')
        metrics_text = (
            f"Amplification: {metrics['amplification_factor']:.1f}x\n"
            f"Max Z-score: {metrics['zscore_max']:.1f}\n"
            f"First detection: {first_det:.0f}h before\n" if first_det else f"First detection: N/A\n"
            f"Peak detection: {peak_det:.0f}h before\n" if peak_det else f"Peak detection: N/A\n"
            f"% time high risk: {metrics['pct_time_high_risk']:.0f}%"
        )
        fig.text(0.98, 0.98, metrics_text, transform=fig.transFigure,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 family='monospace')
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure saved: {output_path}")
    
    def validate_earthquake(self,
                            data_file: Path,
                            earthquake_key: str) -> Dict:
        """
        Validate Λ_geo for a single earthquake.
        
        Returns validation metrics and creates figures.
        """
        print(f"\n{'='*60}")
        print(f"Validating: {earthquake_key}")
        print(f"{'='*60}")
        
        # Load data
        strain_tensors, times, station_lats, station_lons, eq_info = load_strain_data(data_file)
        
        # Initialize analyzer
        analyzer = LambdaGeoAnalyzer(
            dt_hours=1.0,
            smoothing_window=2,
            derivative_method='central'
        )
        
        # Run analysis
        result = analyzer.analyze(
            strain_tensors, times, station_lats, station_lons, eq_info
        )
        
        # Find earthquake time index
        eq_time = eq_info['data_window_days'] * 24  # Earthquake at end of window
        eq_time_idx = int(eq_time)
        
        # Compute metrics
        metrics = self.compute_metrics(result, eq_time_idx)
        metrics['earthquake_key'] = earthquake_key
        metrics['earthquake_info'] = eq_info
        
        # Print summary
        print(f"\nValidation Metrics:")
        print(f"  Amplification factor: {metrics['amplification_factor']:.1f}x")
        print(f"  Max Z-score in precursor window: {metrics['zscore_max']:.1f}")
        print(f"  First detection: {metrics['first_detection_hours_before']} hours before")
        print(f"  Peak detection: {metrics['peak_detection_hours_before']} hours before")
        print(f"  % time at high risk: {metrics['pct_time_high_risk']:.1f}%")
        
        # Create figure
        fig_path = self.results_dir / f"{earthquake_key}_validation.png"
        self.create_validation_figure(result, eq_time_idx, metrics, fig_path)
        
        # Save metrics
        metrics_path = self.results_dir / f"{earthquake_key}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Metrics saved: {metrics_path}")
        
        return metrics
    
    def generate_summary_report(self, all_metrics: List[Dict]):
        """Generate summary report across all earthquakes."""
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY REPORT")
        print("="*70)
        
        # Summary table
        print("\n{:<20} {:>10} {:>12} {:>15} {:>12}".format(
            "Earthquake", "Amp.", "Z-score", "Detection (h)", "Success"
        ))
        print("-" * 70)
        
        successes = 0
        for m in all_metrics:
            eq_name = m['earthquake_key'][:18]
            amp = m['amplification_factor']
            zscore = m['zscore_max']
            detection = m['first_detection_hours_before']
            
            # Success criteria
            success = (detection is not None and 
                       24 <= detection <= 72 and
                       zscore > 2.0)
            
            if success:
                successes += 1
                status = "YES"
            else:
                status = "NO"
            
            print("{:<20} {:>10.1f} {:>12.1f} {:>15} {:>12}".format(
                eq_name, amp, zscore, 
                f"{detection:.0f}" if detection else "N/A",
                status
            ))
        
        print("-" * 70)
        print(f"\nOverall Success Rate: {successes}/{len(all_metrics)} ({100*successes/len(all_metrics):.0f}%)")
        
        # Save summary
        summary = {
            'total_earthquakes': len(all_metrics),
            'successful_detections': successes,
            'success_rate': successes / len(all_metrics) if all_metrics else 0,
            'individual_results': all_metrics
        }
        
        summary_path = self.results_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary saved: {summary_path}")
        
        return summary


def main():
    """Run validation pipeline."""
    
    # Paths - use relative to project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    results_dir = project_root / "results"
    
    # Initialize validator
    validator = EarthquakeValidator(results_dir)
    
    # Target earthquakes
    earthquakes = ['tohoku_2011', 'ridgecrest_2019', 'turkey_2023']
    
    all_metrics = []
    
    for eq_key in earthquakes:
        data_file = data_dir / eq_key / f"{eq_key}_synthetic_strain.npz"
        
        if data_file.exists():
            metrics = validator.validate_earthquake(data_file, eq_key)
            all_metrics.append(metrics)
        else:
            print(f"\nSkipping {eq_key}: data file not found")
            print(f"  Expected: {data_file}")
    
    # Generate summary
    if all_metrics:
        validator.generate_summary_report(all_metrics)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
