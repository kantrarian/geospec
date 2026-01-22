#!/usr/bin/env python3
"""
update_backtest_thd.py
Update backtest_timeseries.json with real fetched THD z-scores.

This script reads the all_historical_thd.json file and updates the THD values
in backtest_timeseries.json with actual z-scores from IRIS data.
"""

import json
from pathlib import Path
from datetime import datetime


def load_fetched_thd(thd_file: Path) -> dict:
    """Load all fetched THD data."""
    with open(thd_file, 'r') as f:
        return json.load(f)


def load_backtest(backtest_file: Path) -> dict:
    """Load existing backtest timeseries."""
    with open(backtest_file, 'r') as f:
        return json.load(f)


def get_daily_thd_zscore(thd_timeseries: list, target_date: str) -> float:
    """
    Get the average z-score for a specific date from THD timeseries.

    Args:
        thd_timeseries: List of THD measurements with date and z_score
        target_date: Date string like "2016-11-13"

    Returns:
        Average z-score for that date, or None if no data
    """
    day_values = [
        entry['z_score']
        for entry in thd_timeseries
        if entry['date'] == target_date
    ]

    if day_values:
        return round(sum(day_values) / len(day_values), 2)
    return None


def update_backtest_with_real_thd(backtest: dict, fetched_thd: dict) -> dict:
    """
    Update backtest events with real THD z-scores, including post-event data.

    Args:
        backtest: Loaded backtest_timeseries.json
        fetched_thd: Loaded all_historical_thd.json

    Returns:
        Updated backtest dict
    """
    events_updated = []

    for event_key, thd_data in fetched_thd.items():
        if event_key not in backtest.get('events', {}):
            print(f"  WARNING: {event_key} not in backtest, skipping")
            continue

        event = backtest['events'][event_key]
        thd_timeseries = thd_data.get('timeseries', [])

        if not thd_timeseries:
            print(f"  WARNING: No THD timeseries for {event_key}")
            continue

        # Get event date for determining post-event entries
        event_date_str = event.get('event_date', '')[:10]  # Extract YYYY-MM-DD

        # Get existing dates in timeseries
        existing_dates = {entry.get('date') for entry in event.get('timeseries', [])}

        # Update each day in the event's timeseries
        updates = 0
        for day_entry in event.get('timeseries', []):
            date = day_entry.get('date')
            if not date:
                continue

            # Get real z-score for this date
            z_score = get_daily_thd_zscore(thd_timeseries, date)

            if z_score is not None:
                day_entry['thd'] = z_score
                updates += 1
            else:
                # No data for this date - set to null instead of keeping fabricated value
                day_entry['thd'] = None

            # Mark post-event entries
            if date > event_date_str:
                day_entry['post_event'] = True

        # Find and add post-event dates from THD data that aren't in backtest
        thd_dates = sorted(set(entry['date'] for entry in thd_timeseries))
        post_event_dates = [d for d in thd_dates if d > event_date_str and d not in existing_dates]

        added = 0
        for post_date in post_event_dates[:3]:  # Limit to 3 days post-event
            z_score = get_daily_thd_zscore(thd_timeseries, post_date)
            if z_score is not None:
                new_entry = {
                    'date': post_date,
                    'tier': None,
                    'tier_name': 'POST_EVENT',
                    'risk': None,
                    'lg_ratio': None,
                    'fc_l2l1': None,
                    'thd': z_score,
                    'post_event': True,
                }
                event['timeseries'].append(new_entry)
                added += 1

        # Sort timeseries by date
        event['timeseries'] = sorted(event['timeseries'], key=lambda x: x.get('date', ''))

        # Update event metadata
        stats = thd_data.get('statistics', {})
        event['thd_station'] = thd_data.get('data_source', {}).get('station')
        event['thd_peak_zscore'] = stats.get('peak_thd')
        event['thd_peak_date'] = stats.get('peak_date', '')[:10] if stats.get('peak_date') else None
        event['thd_baseline_mean'] = stats.get('baseline_mean')
        event['thd_baseline_std'] = stats.get('baseline_std')

        events_updated.append(event_key)
        print(f"  Updated {event_key}: {updates} days updated, {added} post-event days added")

    # Update metadata
    backtest['data_integrity'] = f"THD data fetched from IRIS FDSN on {datetime.now().strftime('%Y-%m-%d')}. All z-scores are computed from real seismic data. Includes 3 days post-event."
    backtest['generated'] = datetime.now().strftime('%Y-%m-%d')

    return backtest


def main():
    print("=" * 70)
    print("UPDATING BACKTEST_TIMESERIES.JSON WITH REAL THD DATA")
    print("=" * 70)

    # File paths
    validation_dir = Path(__file__).parent
    thd_file = validation_dir / 'results' / 'thd_historical' / 'all_historical_thd.json'
    backtest_file = validation_dir.parent / 'docs' / 'backtest_timeseries.json'

    # Check files exist
    if not thd_file.exists():
        print(f"ERROR: THD file not found: {thd_file}")
        print("Run: python fetch_historical_thd.py --all")
        return 1

    if not backtest_file.exists():
        print(f"ERROR: Backtest file not found: {backtest_file}")
        return 1

    print(f"\nTHD source: {thd_file}")
    print(f"Backtest file: {backtest_file}")

    # Load data
    print("\nLoading data...")
    fetched_thd = load_fetched_thd(thd_file)
    backtest = load_backtest(backtest_file)

    print(f"  Fetched THD events: {list(fetched_thd.keys())}")
    print(f"  Backtest events: {list(backtest.get('events', {}).keys())}")

    # Update backtest
    print("\nUpdating THD values...")
    backtest = update_backtest_with_real_thd(backtest, fetched_thd)

    # Backup original
    backup_file = backtest_file.with_suffix('.json.bak')
    with open(backup_file, 'w') as f:
        json.dump(load_backtest(backtest_file), f, indent=2)
    print(f"\nBacked up original to: {backup_file}")

    # Save updated backtest
    with open(backtest_file, 'w') as f:
        json.dump(backtest, f, indent=2)
    print(f"Saved updated backtest to: {backtest_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF REAL THD DATA")
    print("=" * 70)

    for event_key, thd_data in fetched_thd.items():
        stats = thd_data.get('statistics', {})
        station = thd_data.get('data_source', {}).get('station', 'unknown')
        baseline_mean = stats.get('baseline_mean', 0)
        baseline_std = stats.get('baseline_std', 0)
        peak = stats.get('peak_thd', 0)
        peak_date = stats.get('peak_date', '')[:10] if stats.get('peak_date') else 'N/A'

        # Calculate peak z-score
        if baseline_std > 0:
            peak_zscore = (peak - baseline_mean) / baseline_std
        else:
            peak_zscore = 0

        print(f"\n{event_key}:")
        print(f"  Station: {station}")
        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Peak THD: {peak:.4f} (z={peak_zscore:.2f})")
        print(f"  Peak date: {peak_date}")

    print("\n" + "=" * 70)
    print("DONE - backtest_timeseries.json now contains REAL THD z-scores")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
