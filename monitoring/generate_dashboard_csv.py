#!/usr/bin/env python
"""
generate_dashboard_csv.py - Generate data.csv for dashboard from ensemble results.

Reads all ensemble JSON files and generates a CSV with the format expected
by the dashboard's history chart.

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
from datetime import datetime
from pathlib import Path

# Paths
ENSEMBLE_DIR = Path(__file__).parent / 'data' / 'ensemble_results'
DASHBOARD_DIR = Path(__file__).parent / 'dashboard'
OUTPUT_CSV = DASHBOARD_DIR / 'data.csv'

# Calibration cutoff - only include data from this date onwards
# THD baselines were properly calibrated on 2026-01-11
CALIBRATION_CUTOFF = '2026-01-11'


def load_ensemble_files():
    """Load all ensemble JSON files."""
    results = []

    for json_file in sorted(ENSEMBLE_DIR.glob('ensemble_2*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")

    return results


def generate_csv(ensemble_results):
    """Generate data.csv from ensemble results.

    Only includes data from CALIBRATION_CUTOFF onwards to exclude
    pre-calibration data with inflated THD values.
    """
    rows = []

    # Header
    rows.append('date,region,tier,risk,confidence,methods,agreement')

    for result in ensemble_results:
        date = result.get('date', 'unknown')

        # Skip pre-calibration data
        if date < CALIBRATION_CUTOFF:
            continue

        regions = result.get('regions', {})

        for region_name, region_data in regions.items():
            tier = region_data.get('tier', 0)
            risk = region_data.get('combined_risk', 0)
            confidence = region_data.get('confidence', 0)
            methods = region_data.get('methods_available', 0)
            agreement = region_data.get('agreement', 'unknown')

            rows.append(f'{date},{region_name},{tier},{risk:.4f},{confidence:.2f},{methods},{agreement}')

    return '\n'.join(rows)


def main():
    print("=" * 60)
    print("GENERATING DASHBOARD DATA.CSV")
    print("=" * 60)

    print(f"\nLoading ensemble results from: {ENSEMBLE_DIR}")
    results = load_ensemble_files()
    print(f"  Found {len(results)} ensemble files")

    if not results:
        print("  ERROR: No ensemble files found!")
        return

    print("\nGenerating CSV...")
    csv_content = generate_csv(results)

    # Count rows
    n_rows = len(csv_content.split('\n')) - 1  # Minus header
    print(f"  Generated {n_rows} data rows")

    print(f"\nWriting to: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w') as f:
        f.write(csv_content)

    print("  Done!")

    # Print sample
    print("\nSample output (first 10 lines):")
    for line in csv_content.split('\n')[:10]:
        print(f"  {line}")

    print("=" * 60)


if __name__ == '__main__':
    main()
