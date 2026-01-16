#!/usr/bin/env python3
"""
view_status.py - GeoSpec Monitoring Dashboard

View current status, recent alerts, and logs.
Updated to support three-method ensemble system.

Usage:
    python -m monitoring.view_status
    python -m monitoring.view_status --logs
    python -m monitoring.view_status --history 7
    python -m monitoring.view_status --ensemble
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3


def load_latest_run():
    """Load the most recent monitoring run results."""
    data_dir = Path(__file__).parent / "data"
    latest_file = data_dir / "latest_run.json"

    if not latest_file.exists():
        return None

    with open(latest_file) as f:
        return json.load(f)


def load_latest_ensemble():
    """Load the most recent ensemble assessment results."""
    data_dir = Path(__file__).parent / "data" / "ensemble_results"

    if not data_dir.exists():
        return None

    # Find most recent ensemble file
    ensemble_files = sorted(data_dir.glob("ensemble_*.json"), reverse=True)
    if not ensemble_files:
        return None

    with open(ensemble_files[0]) as f:
        return json.load(f)


def load_daily_states(days: int = 7):
    """Load recent daily states from CSV."""
    data_dir = Path(__file__).parent / "data"
    csv_file = data_dir / "daily_states.csv"

    if not csv_file.exists():
        return []

    states = []
    cutoff = datetime.now() - timedelta(days=days)

    with open(csv_file) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= len(header):
                row = dict(zip(header, parts))
                try:
                    date = datetime.strptime(row.get('date', ''), '%Y-%m-%d')
                    if date >= cutoff:
                        states.append(row)
                except ValueError:
                    continue

    return states


def load_transitions(days: int = 30):
    """Load recent tier transitions."""
    data_dir = Path(__file__).parent / "data"
    trans_file = data_dir / "transitions.json"

    if not trans_file.exists():
        return []

    with open(trans_file) as f:
        all_trans = json.load(f)

    cutoff = datetime.now() - timedelta(days=days)
    recent = []

    for t in all_trans:
        try:
            date = datetime.strptime(t.get('timestamp', ''), '%Y-%m-%d')
            if date >= cutoff:
                recent.append(t)
        except ValueError:
            continue

    return recent


def view_logs(n_lines: int = 50):
    """View recent log entries."""
    log_dir = Path(__file__).parent / "logs"

    if not log_dir.exists():
        print("No logs directory found.")
        return

    # Find most recent log file
    log_files = sorted(log_dir.glob("monitoring_*.log"), reverse=True)

    if not log_files:
        print("No log files found.")
        return

    print(f"\n=== Recent Logs ({log_files[0].name}) ===\n")

    with open(log_files[0]) as f:
        lines = f.readlines()
        for line in lines[-n_lines:]:
            print(line.rstrip())


def print_ensemble_dashboard():
    """Print ensemble monitoring dashboard with all three methods."""
    print("=" * 80)
    print("  GEOSPEC THREE-METHOD ENSEMBLE DASHBOARD")
    print("=" * 80)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    ensemble = load_latest_ensemble()
    if not ensemble:
        print("  No ensemble data available. Run the ensemble pipeline first:")
        print("    python -m monitoring.src.run_ensemble_daily")
        print()
        print("=" * 80)
        return

    print(f"  Assessment Date: {ensemble.get('date', 'Unknown')}")
    print(f"  Generated: {ensemble.get('timestamp', 'Unknown')}")
    print()

    # Summary
    summary = ensemble.get('summary', {})
    tier_counts = summary.get('tier_counts', {})
    print("  SUMMARY")
    print("  " + "-" * 76)
    print(f"  Regions Assessed: {summary.get('total_regions', 0)}")
    print(f"  Tier Distribution: NORMAL={tier_counts.get('0', tier_counts.get(0, 0))} "
          f"WATCH={tier_counts.get('1', tier_counts.get(1, 0))} "
          f"ELEVATED={tier_counts.get('2', tier_counts.get(2, 0))} "
          f"CRITICAL={tier_counts.get('3', tier_counts.get(3, 0))}")

    max_region = summary.get('max_risk_region')
    max_risk = summary.get('max_risk', 0)
    if max_region:
        print(f"  Highest Risk: {max_region} ({max_risk:.3f})")
    print()

    # Region details
    print("  REGION STATUS")
    print("  " + "-" * 76)
    print(f"  {'Region':<22} {'Risk':>7} {'Tier':<10} {'Conf':>6} {'Methods':>7} {'Agreement':<15}")
    print("  " + "-" * 76)

    regions = ensemble.get('regions', {})
    for region_id, data in sorted(regions.items(), key=lambda x: -x[1].get('combined_risk', 0)):
        risk = data.get('combined_risk', 0)
        tier_name = data.get('tier_name', 'UNKNOWN')
        confidence = data.get('confidence', 0)
        methods = data.get('methods_available', 0)
        agreement = data.get('agreement', 'unknown')

        # Tier indicator
        tier = data.get('tier', 0)
        tier_sym = {0: '', 1: '[!]', 2: '[!!]', 3: '[!!!]'}
        indicator = tier_sym.get(tier, '')

        print(f"  {region_id:<22} {risk:>7.3f} {tier_name:<10} {confidence:>6.2f} "
              f"{methods:>7} {agreement:<15} {indicator}")

    print("  " + "-" * 76)
    print()

    # Method breakdown for elevated/critical regions
    elevated_regions = [r for r, d in regions.items() if d.get('tier', 0) >= 2]
    if elevated_regions:
        print("  *** ELEVATED/CRITICAL REGIONS - METHOD BREAKDOWN ***")
        print()
        for region_id in elevated_regions:
            data = regions[region_id]
            print(f"  {region_id}: {data.get('tier_name', 'UNKNOWN')} (risk={data.get('combined_risk', 0):.3f})")
            print(f"    Agreement: {data.get('agreement', 'unknown')} | Confidence: {data.get('confidence', 0):.2f}")

            components = data.get('components', {})
            if 'lambda_geo' in components:
                lg = components['lambda_geo']
                print(f"    Lambda_geo: ratio={lg.get('ratio', 'N/A')}, risk={lg.get('risk_score', 'N/A')}")
            if 'fault_correlation' in components:
                fc = components['fault_correlation']
                print(f"    Fault Corr: participation={fc.get('participation_ratio', 'N/A')}, risk={fc.get('risk_score', 'N/A')}")
            if 'seismic_thd' in components:
                thd = components['seismic_thd']
                print(f"    Seismic THD: thd={thd.get('thd', 'N/A')}, risk={thd.get('risk_score', 'N/A')}")
            print()

    print("=" * 80)
    print()

    # Legend
    print("  LEGEND")
    print("  " + "-" * 40)
    print("  Risk Tiers: NORMAL (0-0.25), WATCH (0.25-0.50),")
    print("              ELEVATED (0.50-0.75), CRITICAL (0.75-1.00)")
    print("  Agreement: all_critical (95%), all_elevated (85%),")
    print("             mostly_elevated (75%), mixed (60%), single_method (50%)")
    print()


def print_dashboard():
    """Print current monitoring status dashboard (Lambda_geo only)."""
    print("=" * 70)
    print("  GEOSPEC LAMBDA_GEO MONITORING DASHBOARD")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Latest run
    latest = load_latest_run()
    if latest:
        print(f"  Last Run: {latest.get('date', 'Unknown')}")
        print()

        # Region status
        print("  REGION STATUS")
        print("  " + "-" * 66)
        print(f"  {'Region':<25} {'Tier':>6} {'Ratio':>10} {'Z-score':>10} {'Status':>10}")
        print("  " + "-" * 66)

        regions = latest.get('regions', {})
        failed = latest.get('failed_regions', {})

        for region_id, data in regions.items():
            tier = data.get('tier', 0)
            ratio = data.get('ratio', 0)
            zscore = data.get('zscore', 0)

            tier_sym = {0: '[OK]', 1: '[!]', 2: '[!!]', 3: '[!!!]'}
            status = "NORMAL" if tier == 0 else "ELEVATED" if tier < 3 else "HIGH"

            print(f"  {region_id:<25} {tier_sym.get(tier, '?'):>6} {ratio:>9.2f}x {zscore:>10.2f} {status:>10}")

        for region_id in failed:
            print(f"  {region_id:<25} {'[FAIL]':>6} {'---':>10} {'---':>10} {'FAILED':>10}")

        print("  " + "-" * 66)

        # Alerts
        elevated = latest.get('elevated_regions', [])
        if elevated:
            print()
            print("  *** ELEVATED ALERTS ***")
            for r in elevated:
                print(f"    - {r}")

        if failed:
            print()
            print("  *** DATA ACQUISITION FAILURES ***")
            for r, err in failed.items():
                print(f"    - {r}: {err[:50]}...")
    else:
        print("  No monitoring data available. Run the pipeline first:")
        print("    python -m monitoring.src.run_daily_live --date auto")

    print()
    print("=" * 70)

    # Recent transitions
    transitions = load_transitions(30)
    if transitions:
        print("\n  RECENT TIER TRANSITIONS (Last 30 days)")
        print("  " + "-" * 66)
        for t in transitions[-5:]:
            print(f"  {t.get('timestamp', '?')} | {t.get('region_id', '?')} | "
                  f"Tier {t.get('from_tier', '?')} -> {t.get('to_tier', '?')} | "
                  f"{t.get('reason', '')[:30]}")
        print()


def main():
    parser = argparse.ArgumentParser(description='GeoSpec Monitoring Dashboard')
    parser.add_argument('--logs', action='store_true', help='View recent logs')
    parser.add_argument('--history', type=int, default=0, help='Show N days of history')
    parser.add_argument('--lines', type=int, default=50, help='Number of log lines to show')
    parser.add_argument('--ensemble', action='store_true',
                        help='Show three-method ensemble dashboard')

    args = parser.parse_args()

    if args.logs:
        view_logs(args.lines)
    elif args.ensemble:
        print_ensemble_dashboard()
    elif args.history > 0:
        states = load_daily_states(args.history)
        print(f"\n=== Last {args.history} days of monitoring ===\n")
        for state in states:
            print(f"{state.get('date', '?')} | {state.get('region_id', '?')} | "
                  f"Tier {state.get('tier', '?')} | {state.get('ratio', '?')}x baseline")
    else:
        print_dashboard()


if __name__ == "__main__":
    main()
