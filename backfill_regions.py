#!/usr/bin/env python3
"""
backfill_regions.py
Backfill ensemble results for all regions across historical dates.
"""

import subprocess
import sys
from datetime import datetime, timedelta

# 9 original regions that need backfill
REGIONS = [
    'ridgecrest',
    'socal_saf_mojave',
    'socal_saf_coachella',
    'norcal_hayward',
    'cascadia',
    'tokyo_kanto',
    'istanbul_marmara',
    'turkey_kahramanmaras',
    'campi_flegrei'
]

# Dates to backfill (Dec 17 2025 to Jan 10 2026)
start_date = datetime(2025, 12, 17)
end_date = datetime(2026, 1, 10)

def main():
    current = start_date
    dates = []
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    print(f"Backfilling {len(REGIONS)} regions x {len(dates)} dates = {len(REGIONS)*len(dates)} runs")

    total = len(REGIONS) * len(dates)
    count = 0

    for region in REGIONS:
        for date in dates:
            count += 1
            print(f"[{count}/{total}] {region} @ {date}...")
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'monitoring.src.run_ensemble_daily',
                     '--date', date, '--region', region, '--quiet'],
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                if result.returncode != 0:
                    print(f"  Warning: {result.stderr[:100] if result.stderr else 'unknown error'}")
            except subprocess.TimeoutExpired:
                print(f"  Timeout - skipping")
            except Exception as e:
                print(f"  Error: {e}")
        print(f"Completed {region}")

    print("\nBackfill complete!")

if __name__ == '__main__':
    main()
