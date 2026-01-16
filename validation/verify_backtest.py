#!/usr/bin/env python3
"""
verify_backtest.py - Verification Gates for Backtest Results

Checks backtest results against acceptance criteria.
Exits 0 if all criteria met, 1 otherwise.

Usage:
    python verify_backtest.py --results backtest_metrics_*.json
    python verify_backtest.py --results backtest_metrics.json --criteria acceptance_criteria.yaml

Author: R.J. Mathews / Claude
Date: January 2026
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# =============================================================================
# DEFAULT ACCEPTANCE CRITERIA
# =============================================================================

DEFAULT_CRITERIA = {
    # Performance metrics
    'max_tier2_false_alarms_per_year': 1.0,  # FAR budget from monitoring spec
    'min_hit_rate': 0.60,                     # 60% of M6+ events detected
    'min_precision': 0.30,                    # 30% of WATCH+ alerts are true
    'max_time_in_warning': 0.15,              # <15% of days at WATCH or higher

    # Data quality metrics
    'min_baseline_coverage': 0.80,            # 80% of region-days have valid baseline
    'no_missing_baselines': True,             # All configured stations must have baselines

    # Event scoring
    'min_events_scored': 1,                   # At least 1 event for meaningful stats
}


# =============================================================================
# VERIFICATION
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    actual_value: float
    threshold: float
    comparison: str  # '>=', '<=', '=='
    message: str


def load_criteria(criteria_path: Optional[Path]) -> Dict:
    """Load acceptance criteria from YAML or use defaults."""
    if criteria_path and criteria_path.exists():
        with open(criteria_path) as f:
            return yaml.safe_load(f)
    return DEFAULT_CRITERIA


def check_metric(
    name: str,
    actual: float,
    threshold: float,
    comparison: str,
) -> CheckResult:
    """Check a single metric against threshold."""
    if comparison == '>=':
        passed = actual >= threshold
        msg = f"{name}: {actual:.4f} >= {threshold:.4f}" if passed else f"{name}: {actual:.4f} < {threshold:.4f} FAILED"
    elif comparison == '<=':
        passed = actual <= threshold
        msg = f"{name}: {actual:.4f} <= {threshold:.4f}" if passed else f"{name}: {actual:.4f} > {threshold:.4f} FAILED"
    elif comparison == '==':
        passed = actual == threshold
        msg = f"{name}: {actual} == {threshold}" if passed else f"{name}: {actual} != {threshold} FAILED"
    else:
        passed = False
        msg = f"Unknown comparison: {comparison}"

    return CheckResult(
        name=name,
        passed=passed,
        actual_value=actual,
        threshold=threshold,
        comparison=comparison,
        message=msg,
    )


def verify_backtest_results(
    results_path: Path,
    criteria: Dict,
) -> Tuple[bool, List[CheckResult]]:
    """
    Verify backtest results against acceptance criteria.

    Args:
        results_path: Path to backtest_metrics.json
        criteria: Acceptance criteria dict

    Returns:
        Tuple of (all_passed, list of CheckResult)
    """
    with open(results_path) as f:
        data = json.load(f)

    metrics = data.get('metrics', {})
    checks = []

    # FAR check
    # Convert false_alarm_rate to per-year
    far = metrics.get('false_alarm_rate', 0)
    n_regions = len(data.get('metadata', {}).get('regions', []))
    # far is per region-day, convert to per year
    far_per_year = far * 365 * max(1, n_regions)

    checks.append(check_metric(
        'false_alarms_per_year',
        far_per_year,
        criteria.get('max_tier2_false_alarms_per_year', 1.0),
        '<='
    ))

    # Hit rate
    checks.append(check_metric(
        'hit_rate',
        metrics.get('hit_rate', 0),
        criteria.get('min_hit_rate', 0.60),
        '>='
    ))

    # Precision
    checks.append(check_metric(
        'precision',
        metrics.get('precision', 0),
        criteria.get('min_precision', 0.30),
        '>='
    ))

    # Time in warning
    checks.append(check_metric(
        'time_in_warning',
        metrics.get('time_in_warning_pct', 0),
        criteria.get('max_time_in_warning', 0.15),
        '<='
    ))

    # Minimum events (for statistical significance)
    total_events = metrics.get('total_events', 0) - metrics.get('aftershocks_excluded', 0)
    checks.append(check_metric(
        'events_scored',
        total_events,
        criteria.get('min_events_scored', 1),
        '>='
    ))

    all_passed = all(c.passed for c in checks)

    return all_passed, checks


def print_verification_report(
    checks: List[CheckResult],
    results_path: Path,
):
    """Print verification report."""
    print("\n" + "=" * 70)
    print("BACKTEST VERIFICATION REPORT")
    print("=" * 70)
    print(f"Results file: {results_path}")
    print("-" * 70)
    print(f"{'Check':<30} {'Status':<10} {'Value':<15} {'Threshold':<15}")
    print("-" * 70)

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{check.name:<30} {status:<10} {check.actual_value:<15.4f} "
              f"{check.comparison}{check.threshold:<14.4f}")

    print("-" * 70)

    n_passed = sum(1 for c in checks if c.passed)
    n_total = len(checks)
    all_passed = n_passed == n_total

    if all_passed:
        print(f"RESULT: ALL CHECKS PASSED ({n_passed}/{n_total})")
    else:
        print(f"RESULT: FAILED ({n_passed}/{n_total} passed)")
        print("\nFailed checks:")
        for check in checks:
            if not check.passed:
                print(f"  - {check.message}")

    print("=" * 70)

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Verify backtest results against acceptance criteria',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_backtest.py --results backtest_metrics_20190101_20191231.json
    python verify_backtest.py --results results.json --criteria acceptance_criteria.yaml
        """
    )

    parser.add_argument('--results', type=str, required=True,
                       help='Path to backtest_metrics.json')
    parser.add_argument('--criteria', type=str,
                       help='Path to acceptance_criteria.yaml (optional)')
    parser.add_argument('--json-output', type=str,
                       help='Output verification results as JSON')

    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return 1

    criteria_path = Path(args.criteria) if args.criteria else None
    criteria = load_criteria(criteria_path)

    all_passed, checks = verify_backtest_results(results_path, criteria)
    print_verification_report(checks, results_path)

    # Save JSON output if requested
    if args.json_output:
        output = {
            'results_file': str(results_path),
            'criteria_file': str(criteria_path) if criteria_path else 'defaults',
            'all_passed': all_passed,
            'checks': [
                {
                    'name': c.name,
                    'passed': c.passed,
                    'actual': c.actual_value,
                    'threshold': c.threshold,
                    'comparison': c.comparison,
                }
                for c in checks
            ]
        }
        with open(args.json_output, 'w') as f:
            json.dump(output, f, indent=2)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
