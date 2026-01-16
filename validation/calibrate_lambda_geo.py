#!/usr/bin/env python3
"""
calibrate_lambda_geo.py - Calibrate Lambda_geo using REAL data only

Analyzes the real-data backtest results and derives appropriate thresholds
for the Lambda_geo risk conversion function.

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_FILE = Path(__file__).parent / 'results' / 'backtest_real_data_only.json'
CALIBRATION_OUTPUT = Path(__file__).parent / 'results' / 'lambda_geo_calibration.json'


def analyze_real_data():
    """Analyze real data backtest results for calibration."""

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    print("=" * 70)
    print("LAMBDA_GEO CALIBRATION FROM REAL DATA")
    print("=" * 70)
    print()

    # Extract key metrics
    events = []
    for event_key, event_data in data['events'].items():
        lg = event_data['lambda_geo']
        ev = event_data['event']

        if not lg['computed']:
            continue

        # Compute z-score
        if lg['baseline_std'] > 0:
            z_score = (lg['lambda_geo_max'] - lg['baseline_mean']) / lg['baseline_std']
        else:
            z_score = 0

        events.append({
            'name': ev['name'],
            'magnitude': ev['magnitude'],
            'ratio': lg['ratio_to_baseline'],
            'z_score': z_score,
            'peak_date': lg['peak_date'],
            'event_date': ev['date'][:10],
            'baseline_mean': lg['baseline_mean'],
            'baseline_std': lg['baseline_std'],
            'lambda_geo_max': lg['lambda_geo_max'],
            'n_stations': lg['n_stations'],
        })

    # Sort by magnitude (descending)
    events.sort(key=lambda x: -x['magnitude'])

    print("REAL DATA ANALYSIS:")
    print("-" * 70)
    print(f"{'Event':<25} {'Mag':<6} {'Ratio':<8} {'Z-score':<10} {'Lead Days':<10}")
    print("-" * 70)

    for ev in events:
        # Calculate lead time
        from datetime import datetime
        peak = datetime.strptime(ev['peak_date'], '%Y-%m-%d')
        event = datetime.strptime(ev['event_date'], '%Y-%m-%d')
        lead_days = (event - peak).days

        print(f"{ev['name']:<25} {ev['magnitude']:<6.1f} {ev['ratio']:<8.1f}x {ev['z_score']:<10.1f} {lead_days:<10}")

    print("-" * 70)

    # Statistical analysis
    ratios = [e['ratio'] for e in events]
    z_scores = [e['z_score'] for e in events]

    print(f"\nRatio Statistics:")
    print(f"  Min: {min(ratios):.2f}x")
    print(f"  Max: {max(ratios):.2f}x")
    print(f"  Mean: {np.mean(ratios):.2f}x")
    print(f"  Median: {np.median(ratios):.2f}x")

    print(f"\nZ-score Statistics:")
    print(f"  Min: {min(z_scores):.2f}")
    print(f"  Max: {max(z_scores):.2f}")
    print(f"  Mean: {np.mean(z_scores):.2f}")
    print(f"  Median: {np.median(z_scores):.2f}")

    # Derive calibration thresholds
    # Using ratio-based thresholds calibrated to real data
    print("\n" + "=" * 70)
    print("CALIBRATION THRESHOLDS (Based on Real Data)")
    print("=" * 70)

    # The highest ratio (Tohoku M9.0) is 5.6x
    # The lowest useful signal is around 2.0x
    # We need to set thresholds that would detect these events

    calibration = {
        'calibration_date': datetime.now().isoformat(),
        'data_source': 'Real GPS data from 5 historical earthquakes',
        'events_used': [e['name'] for e in events],

        # Ratio-based thresholds (primary)
        'ratio_thresholds': {
            'normal_max': 1.5,      # Below this = definitely normal
            'watch_min': 1.5,       # 1.5x-2.5x = WATCH
            'watch_max': 2.5,
            'elevated_min': 2.5,    # 2.5x-4.0x = ELEVATED
            'elevated_max': 4.0,
            'critical_min': 4.0,    # 4.0x+ = CRITICAL
        },

        # Z-score thresholds (secondary, for baseline-normalized detection)
        'z_score_thresholds': {
            'watch': 1.5,           # 1.5σ above baseline
            'elevated': 2.5,        # 2.5σ above baseline
            'critical': 4.0,        # 4.0σ above baseline
        },

        # What each event would classify as with new calibration
        'reclassification': {},

        # Statistics from calibration data
        'statistics': {
            'n_events': len(events),
            'ratio_range': [min(ratios), max(ratios)],
            'z_score_range': [min(z_scores), max(z_scores)],
            'magnitude_range': [min(e['magnitude'] for e in events),
                               max(e['magnitude'] for e in events)],
        }
    }

    # Reclassify events with new thresholds
    print(f"\n{'Event':<25} {'Ratio':<8} {'Old Class':<12} {'New Class':<12}")
    print("-" * 60)

    for ev in events:
        ratio = ev['ratio']

        # Old classification (10x threshold)
        old_class = 'HIT' if ratio >= 10.0 else 'MISS'

        # New classification
        if ratio >= calibration['ratio_thresholds']['critical_min']:
            new_class = 'CRITICAL'
        elif ratio >= calibration['ratio_thresholds']['elevated_min']:
            new_class = 'ELEVATED'
        elif ratio >= calibration['ratio_thresholds']['watch_min']:
            new_class = 'WATCH'
        else:
            new_class = 'NORMAL'

        calibration['reclassification'][ev['name']] = {
            'ratio': ratio,
            'z_score': ev['z_score'],
            'old_classification': old_class,
            'new_classification': new_class,
        }

        print(f"{ev['name']:<25} {ratio:<8.1f}x {old_class:<12} {new_class:<12}")

    print("-" * 60)

    # Count detections with new calibration
    new_detections = sum(1 for e in events
                        if calibration['reclassification'][e['name']]['new_classification']
                        in ['WATCH', 'ELEVATED', 'CRITICAL'])

    print(f"\nWith NEW calibration:")
    print(f"  Detection rate: {new_detections}/{len(events)} = {100*new_detections/len(events):.0f}%")
    print(f"  (Detection = WATCH or higher)")

    # Save calibration
    with open(CALIBRATION_OUTPUT, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: {CALIBRATION_OUTPUT}")

    return calibration


def generate_updated_risk_function():
    """Generate the updated risk conversion function code."""

    print("\n" + "=" * 70)
    print("UPDATED RISK CONVERSION FUNCTION")
    print("=" * 70)

    code = '''
def lambda_geo_to_risk(ratio: float) -> float:
    """
    Convert Lambda_geo baseline ratio to 0-1 risk score.

    CALIBRATED FROM REAL DATA (January 2026):
    - Based on 5 historical earthquakes (M6.8-M9.0)
    - Ratios observed: 2.0x to 5.6x
    - Thresholds set to detect all calibration events

    Mapping:
    - ratio < 1.5: NORMAL (risk ~ 0.00-0.20)
    - ratio 1.5-2.5: WATCH (risk ~ 0.20-0.45)
    - ratio 2.5-4.0: ELEVATED (risk ~ 0.45-0.70)
    - ratio > 4.0: CRITICAL (risk ~ 0.70-1.00)
    """
    if ratio <= 1.0:
        return 0.0
    elif ratio < 1.5:
        # Transition from 0 to 0.20
        return 0.20 * (ratio - 1.0) / 0.5
    elif ratio < 2.5:
        # WATCH: 0.20 to 0.45
        return 0.20 + 0.25 * (ratio - 1.5) / 1.0
    elif ratio < 4.0:
        # ELEVATED: 0.45 to 0.70
        return 0.45 + 0.25 * (ratio - 2.5) / 1.5
    else:
        # CRITICAL: 0.70 to 1.0 (saturates at 8x)
        return min(1.0, 0.70 + 0.30 * (ratio - 4.0) / 4.0)
'''

    print(code)
    print("=" * 70)

    return code


def main():
    """Run calibration analysis."""
    calibration = analyze_real_data()
    code = generate_updated_risk_function()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Update monitoring/src/ensemble.py with the new lambda_geo_to_risk() function
2. Find NEW events (not used in calibration) for validation:
   - Need M6.5+ events with GPS coverage
   - Should be from different regions than calibration set
3. Run validation against new events to measure true performance
    """)


if __name__ == '__main__':
    main()
