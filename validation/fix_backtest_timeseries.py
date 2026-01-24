#!/usr/bin/env python3
"""
fix_backtest_timeseries.py
Fix the backtest_timeseries.json to correctly:
1. Mark event day as POST_EVENT (fix the > vs >= bug)
2. Recalculate pre-event peak statistics excluding event day
3. Add missing peak dates to timeseries from raw lg_historical data

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

SCRIPT_DIR = Path(__file__).parent
BACKTEST_FILE = SCRIPT_DIR.parent / 'docs' / 'backtest_timeseries.json'
LG_RESULTS = SCRIPT_DIR / 'results' / 'lg_historical' / 'all_historical_lg.json'


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def fix_backtest():
    print("=" * 70)
    print("FIXING BACKTEST_TIMESERIES.JSON")
    print("=" * 70)

    # Load files
    backtest = load_json(BACKTEST_FILE)
    lg_data = load_json(LG_RESULTS) if LG_RESULTS.exists() else {}

    # Backup
    backup_file = BACKTEST_FILE.with_suffix('.json.bak.pre_fix')
    with open(backup_file, 'w') as f:
        json.dump(backtest, f, indent=2)
    print(f"\nBacked up to: {backup_file}")

    for event_key, event in backtest.get('events', {}).items():
        print(f"\n--- {event_key} ---")

        event_date_str = event.get('event_date', '')[:10]
        print(f"  Event date: {event_date_str}")

        # Get raw lg data for this event
        lg_event = lg_data.get(event_key, {})
        lg_ts = lg_event.get('timeseries', [])
        lg_lookup = {e['date']: e for e in lg_ts}

        # Fix 1: Mark event day as POST_EVENT
        for entry in event.get('timeseries', []):
            date = entry.get('date', '')
            if date >= event_date_str:  # FIX: >= not >
                if not entry.get('post_event'):
                    print(f"  Marking {date} as POST_EVENT")
                entry['post_event'] = True
                entry['tier'] = None
                entry['tier_name'] = 'POST_EVENT'
                entry['risk'] = None

        # Fix 2: Calculate TRUE pre-event peak (excluding event day)
        pre_event_entries = [e for e in event.get('timeseries', [])
                           if not e.get('post_event') and e.get('lg_ratio') is not None]

        if pre_event_entries:
            pre_event_lg = [(e['date'], e['lg_ratio']) for e in pre_event_entries]
            if pre_event_lg:
                true_peak = max(pre_event_lg, key=lambda x: x[1])
                print(f"  TRUE pre-event LG peak: {true_peak[1]:.2f}x on {true_peak[0]}")

                # Update metadata with true pre-event peak
                event['lg_peak_ratio'] = true_peak[1]
                event['lg_peak_date'] = true_peak[0]

        # Fix 3: Add missing dates from raw lg data
        existing_dates = {e['date'] for e in event.get('timeseries', [])}

        # Add any pre-event dates from lg_data that are missing
        for lg_entry in lg_ts:
            date = lg_entry.get('date')
            if date and date < event_date_str and date not in existing_dates:
                ratio = lg_entry.get('ratio')
                if ratio is not None:
                    print(f"  Adding missing date {date} with lg_ratio={ratio}")
                    new_entry = {
                        'date': date,
                        'tier': 0 if ratio < 1.5 else (1 if ratio < 2.5 else 2),
                        'tier_name': 'NORMAL' if ratio < 1.5 else ('WATCH' if ratio < 2.5 else 'ELEVATED'),
                        'risk': min(0.35, ratio * 0.1) if ratio < 1.5 else min(0.75, 0.35 + (ratio - 1.5) * 0.1),
                        'lg_ratio': ratio,
                        'fc_l2l1': None,
                        'thd': None,
                    }
                    event['timeseries'].append(new_entry)

        # Sort timeseries by date
        event['timeseries'] = sorted(event['timeseries'], key=lambda x: x.get('date', ''))

        # Print summary
        pre = [e for e in event['timeseries'] if not e.get('post_event')]
        post = [e for e in event['timeseries'] if e.get('post_event')]
        print(f"  Final: {len(pre)} pre-event, {len(post)} post-event entries")

    # Update metadata
    backtest['generated'] = datetime.now().strftime('%Y-%m-%d')
    backtest['data_integrity'] = (
        "Fixed on " + datetime.now().strftime('%Y-%m-%d') + ": "
        "Event day now correctly marked as POST_EVENT. "
        "Pre-event peaks recalculated excluding co-seismic data."
    )

    # Save
    with open(BACKTEST_FILE, 'w') as f:
        json.dump(backtest, f, indent=2)
    print(f"\n\nSaved fixed backtest to: {BACKTEST_FILE}")

    # Also sync to monitoring dashboard
    dashboard_file = SCRIPT_DIR.parent / 'monitoring' / 'dashboard' / 'backtest_timeseries.json'
    if dashboard_file.parent.exists():
        with open(dashboard_file, 'w') as f:
            json.dump(backtest, f, indent=2)
        print(f"Synced to: {dashboard_file}")

    print("\n" + "=" * 70)
    print("FIX COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    fix_backtest()
