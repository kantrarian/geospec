#!/usr/bin/env python3
"""
update_backtest_all_methods.py
Unified script to update backtest_timeseries.json with real data from all methods:
- THD (Total Harmonic Distortion) from IRIS FDSN
- Lambda_geo (GPS strain) from NGL .tenv3 files
- FC (Fault Correlation L2/L1) from IRIS/SCEDC waveforms

This script reads results from each method's output directory and merges them
into the master backtest_timeseries.json file, including post-event data.

Usage:
    python update_backtest_all_methods.py
    python update_backtest_all_methods.py --thd-only
    python update_backtest_all_methods.py --lg-only
    python update_backtest_all_methods.py --fc-only

Author: R.J. Mathews
Date: January 2026
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
BACKTEST_FILE = SCRIPT_DIR.parent / 'docs' / 'backtest_timeseries.json'

# Result directories
THD_RESULTS = RESULTS_DIR / 'thd_historical' / 'all_historical_thd.json'
LG_RESULTS = RESULTS_DIR / 'lg_historical' / 'all_historical_lg.json'
FC_RESULTS = RESULTS_DIR / 'fc_historical' / 'all_historical_fc.json'


def load_json(filepath: Path) -> Optional[Dict]:
    """Load JSON file if it exists."""
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def load_backtest() -> Dict:
    """Load existing backtest_timeseries.json."""
    if not BACKTEST_FILE.exists():
        print(f"ERROR: Backtest file not found: {BACKTEST_FILE}")
        return {}
    with open(BACKTEST_FILE, 'r') as f:
        return json.load(f)


def get_daily_value(timeseries: list, target_date: str, field: str) -> Optional[float]:
    """Get value for a specific date from a timeseries."""
    for entry in timeseries:
        if entry.get('date') == target_date:
            return entry.get(field)
    return None


def update_thd(backtest: Dict, thd_data: Dict) -> Dict:
    """Update backtest with THD z-scores."""
    if not thd_data:
        print("  No THD data to update")
        return backtest

    print("\nUpdating THD values...")
    for event_key, data in thd_data.items():
        if event_key not in backtest.get('events', {}):
            print(f"  WARNING: {event_key} not in backtest, skipping")
            continue

        event = backtest['events'][event_key]
        thd_ts = data.get('timeseries', [])
        event_date_str = event.get('event_date', '')[:10]
        existing_dates = {e.get('date') for e in event.get('timeseries', [])}

        # Update existing entries
        updates = 0
        for entry in event.get('timeseries', []):
            date = entry.get('date')
            if not date:
                continue

            # Get average z-score for this date
            day_values = [e['z_score'] for e in thd_ts if e.get('date') == date]
            if day_values:
                entry['thd'] = round(sum(day_values) / len(day_values), 2)
                updates += 1
            else:
                entry['thd'] = None

            # Mark post-event
            if date > event_date_str:
                entry['post_event'] = True

        # Add post-event entries not in backtest
        thd_dates = sorted(set(e['date'] for e in thd_ts))
        post_dates = [d for d in thd_dates if d > event_date_str and d not in existing_dates]

        added = 0
        for post_date in post_dates[:3]:
            day_values = [e['z_score'] for e in thd_ts if e.get('date') == post_date]
            if day_values:
                new_entry = {
                    'date': post_date,
                    'tier': None,
                    'tier_name': 'POST_EVENT',
                    'risk': None,
                    'lg_ratio': None,
                    'fc_l2l1': None,
                    'thd': round(sum(day_values) / len(day_values), 2),
                    'post_event': True,
                }
                event['timeseries'].append(new_entry)
                added += 1

        # Sort by date
        event['timeseries'] = sorted(event['timeseries'], key=lambda x: x.get('date', ''))

        # Update metadata
        stats = data.get('statistics', {})
        event['thd_station'] = data.get('data_source', {}).get('station')
        event['thd_peak_zscore'] = stats.get('peak_thd')

        print(f"  {event_key}: {updates} days updated, {added} post-event added")

    return backtest


def update_lambda_geo(backtest: Dict, lg_data: Dict) -> Dict:
    """Update backtest with Lambda_geo ratios."""
    if not lg_data:
        print("  No Lambda_geo data to update")
        return backtest

    print("\nUpdating Lambda_geo values...")
    for event_key, data in lg_data.items():
        if event_key not in backtest.get('events', {}):
            print(f"  WARNING: {event_key} not in backtest, skipping")
            continue

        if not data.get('computed'):
            print(f"  WARNING: {event_key} Lambda_geo not computed, skipping")
            continue

        event = backtest['events'][event_key]
        lg_ts = data.get('timeseries', [])
        event_date_str = event.get('event_date', '')[:10]
        existing_dates = {e.get('date') for e in event.get('timeseries', [])}

        # Build lookup dict for lg timeseries
        lg_lookup = {e['date']: e for e in lg_ts}

        # Update existing entries
        updates = 0
        for entry in event.get('timeseries', []):
            date = entry.get('date')
            if not date or date not in lg_lookup:
                continue

            lg_entry = lg_lookup[date]
            entry['lg_ratio'] = lg_entry.get('ratio')
            updates += 1

            # Mark post-event
            if lg_entry.get('post_event'):
                entry['post_event'] = True

        # Add post-event entries not in backtest
        post_entries = [e for e in lg_ts if e.get('post_event') and e['date'] not in existing_dates]

        added = 0
        for lg_entry in post_entries[:3]:
            # Check if entry already exists (might have been added by THD)
            existing = next((e for e in event['timeseries'] if e.get('date') == lg_entry['date']), None)
            if existing:
                existing['lg_ratio'] = lg_entry.get('ratio')
            else:
                new_entry = {
                    'date': lg_entry['date'],
                    'tier': None,
                    'tier_name': 'POST_EVENT',
                    'risk': None,
                    'lg_ratio': lg_entry.get('ratio'),
                    'fc_l2l1': None,
                    'thd': None,
                    'post_event': True,
                }
                event['timeseries'].append(new_entry)
                added += 1

        # Sort by date
        event['timeseries'] = sorted(event['timeseries'], key=lambda x: x.get('date', ''))

        # Update metadata
        stats = data.get('statistics', {})
        event['lg_n_stations'] = data.get('data_source', {}).get('n_stations')
        event['lg_peak_ratio'] = stats.get('peak_ratio')
        event['lg_peak_date'] = stats.get('peak_date')

        print(f"  {event_key}: {updates} days updated, {added} post-event added")

    return backtest


def update_fc(backtest: Dict, fc_data: Dict) -> Dict:
    """Update backtest with FC L2/L1 values."""
    if not fc_data:
        print("  No FC data to update")
        return backtest

    print("\nUpdating FC values...")
    for event_key, data in fc_data.items():
        if event_key not in backtest.get('events', {}):
            print(f"  WARNING: {event_key} not in backtest, skipping")
            continue

        event = backtest['events'][event_key]
        fc_ts = data.get('timeseries', [])
        event_date_str = event.get('event_date', '')[:10]
        existing_dates = {e.get('date') for e in event.get('timeseries', [])}

        # Build lookup dict for FC timeseries (average L2/L1 per day)
        fc_by_date = {}
        for entry in fc_ts:
            date = entry.get('date')
            if date not in fc_by_date:
                fc_by_date[date] = {'values': [], 'post_event': entry.get('post_event', False)}
            fc_by_date[date]['values'].append(entry.get('l2_l1', 0))
            if entry.get('post_event'):
                fc_by_date[date]['post_event'] = True

        # Average L2/L1 per day
        fc_lookup = {}
        for date, info in fc_by_date.items():
            if info['values']:
                fc_lookup[date] = {
                    'l2_l1': round(sum(info['values']) / len(info['values']), 4),
                    'post_event': info['post_event']
                }

        # Update existing entries
        updates = 0
        for entry in event.get('timeseries', []):
            date = entry.get('date')
            if not date or date not in fc_lookup:
                continue

            fc_entry = fc_lookup[date]
            entry['fc_l2l1'] = fc_entry['l2_l1']
            updates += 1

            # Mark post-event
            if fc_entry.get('post_event'):
                entry['post_event'] = True

        # Add post-event entries not in backtest
        post_dates = [d for d, info in fc_lookup.items()
                      if info.get('post_event') and d not in existing_dates]

        added = 0
        for post_date in sorted(post_dates)[:3]:
            fc_entry = fc_lookup[post_date]
            # Check if entry already exists
            existing = next((e for e in event['timeseries'] if e.get('date') == post_date), None)
            if existing:
                existing['fc_l2l1'] = fc_entry['l2_l1']
            else:
                new_entry = {
                    'date': post_date,
                    'tier': None,
                    'tier_name': 'POST_EVENT',
                    'risk': None,
                    'lg_ratio': None,
                    'fc_l2l1': fc_entry['l2_l1'],
                    'thd': None,
                    'post_event': True,
                }
                event['timeseries'].append(new_entry)
                added += 1

        # Sort by date
        event['timeseries'] = sorted(event['timeseries'], key=lambda x: x.get('date', ''))

        # Update metadata
        stats = data.get('statistics', {})
        event['fc_min_l2l1'] = stats.get('min_l2_l1')
        event['fc_min_date'] = stats.get('min_date', '')[:10] if stats.get('min_date') else None

        print(f"  {event_key}: {updates} days updated, {added} post-event added")

    return backtest


def main():
    parser = argparse.ArgumentParser(
        description='Update backtest_timeseries.json with data from all methods'
    )
    parser.add_argument('--thd-only', action='store_true', help='Update THD only')
    parser.add_argument('--lg-only', action='store_true', help='Update Lambda_geo only')
    parser.add_argument('--fc-only', action='store_true', help='Update FC only')
    args = parser.parse_args()

    print("=" * 70)
    print("UPDATING BACKTEST_TIMESERIES.JSON WITH ALL METHODS")
    print("=" * 70)

    # Determine what to update
    update_all = not (args.thd_only or args.lg_only or args.fc_only)
    update_thd_flag = update_all or args.thd_only
    update_lg_flag = update_all or args.lg_only
    update_fc_flag = update_all or args.fc_only

    # Load backtest
    print("\nLoading backtest file...")
    backtest = load_backtest()
    if not backtest:
        return 1

    print(f"  Events in backtest: {list(backtest.get('events', {}).keys())}")

    # Load results
    print("\nLoading method results...")
    thd_data = load_json(THD_RESULTS) if update_thd_flag else None
    lg_data = load_json(LG_RESULTS) if update_lg_flag else None
    fc_data = load_json(FC_RESULTS) if update_fc_flag else None

    # Update each method
    if update_thd_flag and thd_data:
        backtest = update_thd(backtest, thd_data)

    if update_lg_flag and lg_data:
        backtest = update_lambda_geo(backtest, lg_data)

    if update_fc_flag and fc_data:
        backtest = update_fc(backtest, fc_data)

    # Update metadata
    methods_updated = []
    if update_thd_flag and thd_data:
        methods_updated.append('THD')
    if update_lg_flag and lg_data:
        methods_updated.append('Lambda_geo')
    if update_fc_flag and fc_data:
        methods_updated.append('FC')

    backtest['data_integrity'] = (
        f"Real data from: {', '.join(methods_updated)}. "
        f"Updated on {datetime.now().strftime('%Y-%m-%d')}. "
        "Includes 3 days post-event data."
    )
    backtest['generated'] = datetime.now().strftime('%Y-%m-%d')
    backtest['methods_updated'] = methods_updated

    # Backup original
    backup_file = BACKTEST_FILE.with_suffix('.json.bak')
    with open(backup_file, 'w') as f:
        json.dump(load_backtest(), f, indent=2)
    print(f"\nBacked up original to: {backup_file}")

    # Save updated backtest
    with open(BACKTEST_FILE, 'w') as f:
        json.dump(backtest, f, indent=2)
    print(f"Saved updated backtest to: {BACKTEST_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("UPDATE SUMMARY")
    print("=" * 70)

    for event_key, event in backtest.get('events', {}).items():
        ts = event.get('timeseries', [])
        pre_event = [e for e in ts if not e.get('post_event')]
        post_event = [e for e in ts if e.get('post_event')]

        # Count available data
        lg_count = sum(1 for e in ts if e.get('lg_ratio') is not None)
        fc_count = sum(1 for e in ts if e.get('fc_l2l1') is not None)
        thd_count = sum(1 for e in ts if e.get('thd') is not None)

        print(f"\n{event_key}:")
        print(f"  Pre-event days: {len(pre_event)}")
        print(f"  Post-event days: {len(post_event)}")
        print(f"  LG data: {lg_count}, FC data: {fc_count}, THD data: {thd_count}")

    print("\n" + "=" * 70)
    print("DONE - backtest_timeseries.json updated with all available real data")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
