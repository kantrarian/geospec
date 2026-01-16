#!/usr/bin/env python
"""
run_full_backtest.py - Master Orchestration Script for GeoSpec Backtest Validation

Runs all backtest validation scripts and aggregates results into a unified report.

Events Validated:
1. Ridgecrest 2019 M7.1 - Full 3-method validation (Lambda_geo, THD, Fault Correlation)
2. Tohoku 2011 M9.0 - GPS-only (Lambda_geo)
3. Turkey 2023 M7.8 - GPS-only (Lambda_geo)
4. Chile 2010 M8.8 - GPS-only (Lambda_geo)
5. Morocco 2023 M6.8 - GPS-only (Lambda_geo) - Expected failure case

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_DIR = Path(__file__).parent
RESULTS_DIR = VALIDATION_DIR / 'results'

# Backtest scripts to run
BACKTEST_SCRIPTS = [
    {
        'name': 'Ridgecrest 2019 (Full 3-Method)',
        'script': 'backtest_ridgecrest_full.py',
        'output': 'ridgecrest_2019_full_backtest.json',
        'type': 'full_3_method',
    },
    {
        'name': 'Tohoku 2011 (GPS Only)',
        'script': 'backtest_tohoku_gps.py',
        'output': 'tohoku_2011_gps_backtest.json',
        'type': 'gps_only',
    },
    {
        'name': 'Turkey 2023 (GPS Only)',
        'script': 'backtest_turkey_gps.py',
        'output': 'turkey_2023_gps_backtest.json',
        'type': 'gps_only',
    },
    {
        'name': 'Chile 2010 (GPS Only)',
        'script': 'backtest_chile_gps.py',
        'output': 'chile_2010_gps_backtest.json',
        'type': 'gps_only',
    },
    {
        'name': 'Morocco 2023 (GPS Only - Expected Failure)',
        'script': 'backtest_morocco_gps.py',
        'output': 'morocco_2023_gps_backtest.json',
        'type': 'gps_only',
    },
]


def run_backtest_script(script_name: str) -> bool:
    """Run a single backtest script and return success status."""
    script_path = VALIDATION_DIR / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False
        return True

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Script timed out")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def load_result(output_file: str) -> Optional[Dict]:
    """Load a backtest result JSON file."""
    path = RESULTS_DIR / output_file

    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate all backtest results into a unified summary."""

    # Count outcomes
    hits = sum(1 for r in results if r.get('classification') == 'HIT')
    marginals = sum(1 for r in results if r.get('classification') == 'MARGINAL')
    misses = sum(1 for r in results if r.get('classification') == 'MISS')

    # Calculate lead times
    lead_times = []
    for r in results:
        if r.get('classification') == 'HIT':
            lead_hours = r.get('scoring', {}).get('lead_time_hours', 0)
            if lead_hours > 0:
                lead_times.append(lead_hours)

    mean_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0

    # Calculate detection rates by method
    lambda_geo_detected = sum(1 for r in results
                               if r.get('methods', {}).get('lambda_geo', {}).get('detected'))
    lambda_geo_available = sum(1 for r in results
                                if r.get('methods', {}).get('lambda_geo', {}).get('available'))

    thd_detected = sum(1 for r in results
                       if r.get('methods', {}).get('thd', {}).get('detected'))
    thd_available = sum(1 for r in results
                        if r.get('methods', {}).get('thd', {}).get('available'))

    fc_detected = sum(1 for r in results
                      if r.get('methods', {}).get('fault_correlation', {}).get('detected'))
    fc_available = sum(1 for r in results
                       if r.get('methods', {}).get('fault_correlation', {}).get('available'))

    # Event details
    events = []
    for r in results:
        event = r.get('event', {})
        methods = r.get('methods', {})

        events.append({
            'name': event.get('name', 'Unknown'),
            'magnitude': event.get('mainshock_magnitude') or event.get('magnitude'),
            'event_date': event.get('event_date') or event.get('mainshock_date'),
            'classification': r.get('classification'),
            'lead_time_hours': r.get('scoring', {}).get('lead_time_hours', 0),
            'methods_used': {
                'lambda_geo': methods.get('lambda_geo', {}).get('available', False),
                'thd': methods.get('thd', {}).get('available', False),
                'fault_correlation': methods.get('fault_correlation', {}).get('available', False),
            },
        })

    return {
        'total_events': len(results),
        'hits': hits,
        'marginals': marginals,
        'misses': misses,
        'hit_rate': hits / len(results) if results else 0,
        'mean_lead_time_hours': mean_lead_time,
        'mean_lead_time_days': mean_lead_time / 24,
        'method_detection': {
            'lambda_geo': {
                'detected': lambda_geo_detected,
                'available': lambda_geo_available,
                'rate': lambda_geo_detected / lambda_geo_available if lambda_geo_available else 0,
            },
            'thd': {
                'detected': thd_detected,
                'available': thd_available,
                'rate': thd_detected / thd_available if thd_available else 0,
            },
            'fault_correlation': {
                'detected': fc_detected,
                'available': fc_available,
                'rate': fc_detected / fc_available if fc_available else 0,
            },
        },
        'events': events,
    }


def print_summary_table(summary: Dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("MULTI-EVENT BACKTEST SUMMARY")
    print("=" * 80)

    print(f"\n{'Event':<30} {'Mag':>5} {'Class':>10} {'Lead (h)':>10} {'LG':>5} {'THD':>5} {'FC':>5}")
    print("-" * 80)

    for event in summary['events']:
        methods = event['methods_used']
        lg = "YES" if methods['lambda_geo'] else "-"
        thd = "YES" if methods['thd'] else "-"
        fc = "YES" if methods['fault_correlation'] else "-"

        lead = f"{event['lead_time_hours']:.1f}" if event['lead_time_hours'] > 0 else "-"

        print(f"{event['name']:<30} {event['magnitude']:>5.1f} {event['classification']:>10} "
              f"{lead:>10} {lg:>5} {thd:>5} {fc:>5}")

    print("-" * 80)

    # Totals row
    print(f"\n{'TOTALS':<30}")
    print(f"  Events validated: {summary['total_events']}")
    print(f"  HITs: {summary['hits']}")
    print(f"  MARGINALs: {summary['marginals']}")
    print(f"  MISSes: {summary['misses']}")

    print(f"\n{'METRICS':<30}")
    print(f"  Hit Rate: {summary['hit_rate']:.0%}")
    print(f"  Mean Lead Time: {summary['mean_lead_time_hours']:.1f} hours ({summary['mean_lead_time_days']:.1f} days)")

    print(f"\n{'METHOD DETECTION RATES':<30}")
    for method, stats in summary['method_detection'].items():
        if stats['available'] > 0:
            print(f"  {method}: {stats['detected']}/{stats['available']} ({stats['rate']:.0%})")


def run_full_backtest():
    """Run all backtest scripts and generate unified report."""
    print("=" * 80)
    print("GEOSPEC FULL BACKTEST VALIDATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check data inventory first
    inventory_path = RESULTS_DIR / 'data_inventory.json'
    if not inventory_path.exists():
        print("Running data inventory check...")
        run_backtest_script('data_inventory.py')

    # Run all backtest scripts
    results = []

    for spec in BACKTEST_SCRIPTS:
        print(f"\n{'='*40}")
        print(f"Running: {spec['name']}")
        print(f"{'='*40}")

        success = run_backtest_script(spec['script'])

        if success:
            result = load_result(spec['output'])
            if result:
                results.append(result)
                print(f"  Classification: {result.get('classification', 'UNKNOWN')}")
            else:
                print(f"  ERROR: Could not load result file")
        else:
            print(f"  ERROR: Script execution failed")

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)

    summary = aggregate_results(results)

    # Generate unified output
    output = {
        'backtest_run': 'full_multi_event',
        'generated': datetime.now().isoformat(),
        'summary': summary,
        'individual_results': [
            {
                'event_id': r.get('event', {}).get('event_id'),
                'classification': r.get('classification'),
                'lead_time_hours': r.get('scoring', {}).get('lead_time_hours', 0),
            }
            for r in results
        ],
        'acceptance_criteria': {
            'target_hit_rate': 0.6,
            'achieved_hit_rate': summary['hit_rate'],
            'target_met': summary['hit_rate'] >= 0.6,
            'target_lead_time_hours': 24,
            'achieved_mean_lead_time': summary['mean_lead_time_hours'],
            'lead_time_target_met': summary['mean_lead_time_hours'] >= 24,
        },
    }

    # Print summary
    print_summary_table(summary)

    # Print acceptance criteria
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA")
    print("=" * 80)

    for criterion, result in output['acceptance_criteria'].items():
        if 'target' in criterion and 'met' not in criterion:
            continue
        if '_met' in criterion:
            status = "PASS" if result else "FAIL"
            print(f"  {criterion}: {status}")

    # Save unified output
    output_path = RESULTS_DIR / 'full_backtest_summary.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nUnified results saved to: {output_path}")
    print("=" * 80)

    return output


if __name__ == '__main__':
    run_full_backtest()
