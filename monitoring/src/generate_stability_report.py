#!/usr/bin/env python3
"""
generate_stability_report.py
Generates 7 or 30 day stability reports for GeoSpec monitoring.

Shows per-region:
- Days at each tier
- Coverage consistency (e.g., '3/4 segments 90% of the time')
- Methods available trend
- Biggest single-day risk change
- Persistence summary

Usage:
    python generate_stability_report.py                # 7-day report
    python generate_stability_report.py --days 30     # 30-day report
    python generate_stability_report.py --region ridgecrest  # Single region

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results_for_period(
    output_dir: Path,
    end_date: datetime,
    days: int
) -> Dict[str, List[Dict]]:
    """
    Load all ensemble results for a time period.

    Returns:
        Dict mapping date string to result data
    """
    results = {}

    for day_offset in range(days):
        date = end_date - timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        result_file = output_dir / f'ensemble_{date_str}.json'

        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    results[date_str] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")

    return results


def compute_region_stability(
    region: str,
    daily_results: Dict[str, Dict],
    days: int
) -> Dict:
    """
    Compute stability metrics for a single region.

    Returns:
        Dict with stability metrics
    """
    # Collect data points
    tiers = []
    risks = []
    coverage_pcts = []
    segments_defined = []
    segments_working = []
    methods_available = []
    dates = []

    for date_str, data in sorted(daily_results.items()):
        if 'regions' not in data or region not in data['regions']:
            continue

        region_data = data['regions'][region]
        dates.append(date_str)
        tiers.append(region_data.get('tier', 0))
        risks.append(region_data.get('combined_risk', 0.0))
        methods_available.append(region_data.get('methods_available', 0))

        # Coverage from new coverage field
        coverage = region_data.get('coverage', {})
        if coverage:
            segments_defined.append(coverage.get('segments_defined', 0))
            segments_working.append(coverage.get('segments_working', 0))
            coverage_pcts.append(coverage.get('coverage_pct', 0.0))
        else:
            # Legacy format - try to extract from notes
            segments_defined.append(0)
            segments_working.append(0)
            coverage_pcts.append(0.0)

    if not dates:
        return {
            'region': region,
            'days_with_data': 0,
            'tier_distribution': {},
            'avg_risk': 0.0,
            'max_risk': 0.0,
            'min_risk': 0.0,
            'risk_volatility': 0.0,
            'biggest_daily_change': 0.0,
            'biggest_change_date': None,
            'avg_coverage_pct': 0.0,
            'coverage_consistency': 'N/A',
            'avg_methods': 0.0,
            'methods_trend': 'stable',
            'notes': 'No data available',
        }

    # Tier distribution
    tier_counts = Counter(tiers)
    tier_distribution = {
        'NORMAL': tier_counts.get(0, 0),
        'WATCH': tier_counts.get(1, 0),
        'ELEVATED': tier_counts.get(2, 0),
        'CRITICAL': tier_counts.get(3, 0),
    }

    # Risk statistics
    avg_risk = sum(risks) / len(risks) if risks else 0.0
    max_risk = max(risks) if risks else 0.0
    min_risk = min(risks) if risks else 0.0

    # Risk volatility (standard deviation)
    if len(risks) > 1:
        mean = avg_risk
        variance = sum((r - mean) ** 2 for r in risks) / len(risks)
        risk_volatility = variance ** 0.5
    else:
        risk_volatility = 0.0

    # Biggest daily change
    biggest_change = 0.0
    biggest_change_date = None
    for i in range(1, len(risks)):
        change = abs(risks[i] - risks[i-1])
        if change > biggest_change:
            biggest_change = change
            biggest_change_date = dates[i]

    # Coverage consistency
    if coverage_pcts and any(c > 0 for c in coverage_pcts):
        avg_coverage = sum(coverage_pcts) / len(coverage_pcts)
        # Check if segments are consistent
        if segments_defined and segments_working:
            most_common_defined = Counter(segments_defined).most_common(1)[0][0]
            most_common_working = Counter(segments_working).most_common(1)[0][0]
            consistency_pct = sum(1 for d, w in zip(segments_defined, segments_working)
                                  if d == most_common_defined and w == most_common_working) / len(segments_defined) * 100
            coverage_consistency = f"{most_common_working}/{most_common_defined} segments {consistency_pct:.0f}% of the time"
        else:
            coverage_consistency = f"Avg {avg_coverage:.1f}%"
    else:
        avg_coverage = 0.0
        coverage_consistency = "No coverage data"

    # Methods trend
    avg_methods = sum(methods_available) / len(methods_available) if methods_available else 0.0
    if len(methods_available) >= 3:
        # Check if methods are trending up, down, or stable
        first_third = sum(methods_available[:len(methods_available)//3]) / (len(methods_available)//3)
        last_third = sum(methods_available[-len(methods_available)//3:]) / (len(methods_available)//3)
        if last_third > first_third + 0.2:
            methods_trend = "improving"
        elif last_third < first_third - 0.2:
            methods_trend = "degrading"
        else:
            methods_trend = "stable"
    else:
        methods_trend = "stable"

    # Persistence summary
    consecutive_watch_days = 0
    max_consecutive_watch = 0
    for tier in tiers:
        if tier >= 1:
            consecutive_watch_days += 1
            max_consecutive_watch = max(max_consecutive_watch, consecutive_watch_days)
        else:
            consecutive_watch_days = 0

    return {
        'region': region,
        'days_with_data': len(dates),
        'tier_distribution': tier_distribution,
        'avg_risk': avg_risk,
        'max_risk': max_risk,
        'min_risk': min_risk,
        'risk_volatility': risk_volatility,
        'biggest_daily_change': biggest_change,
        'biggest_change_date': biggest_change_date,
        'avg_coverage_pct': avg_coverage if coverage_pcts else 0.0,
        'coverage_consistency': coverage_consistency,
        'avg_methods': avg_methods,
        'methods_trend': methods_trend,
        'max_consecutive_watch_days': max_consecutive_watch,
        'notes': '',
    }


def generate_report(
    output_dir: Path,
    end_date: datetime,
    days: int,
    regions: Optional[List[str]] = None,
) -> str:
    """
    Generate stability report as formatted text.

    Args:
        output_dir: Directory containing ensemble results
        end_date: End date of report period
        days: Number of days to include
        regions: Optional list of regions to include

    Returns:
        Formatted report text
    """
    # Load all results
    daily_results = load_results_for_period(output_dir, end_date, days)

    if not daily_results:
        return f"No results found for {days}-day period ending {end_date.date()}"

    # Get all regions
    all_regions = set()
    for data in daily_results.values():
        if 'regions' in data:
            all_regions.update(data['regions'].keys())

    if regions:
        all_regions = all_regions.intersection(set(regions))

    if not all_regions:
        return "No regions found in results"

    # Compute stability for each region
    region_stability = {}
    for region in sorted(all_regions):
        region_stability[region] = compute_region_stability(region, daily_results, days)

    # Generate report
    lines = []
    lines.append("=" * 90)
    lines.append(f"GEOSPEC {days}-DAY STABILITY REPORT")
    lines.append(f"Period: {(end_date - timedelta(days=days-1)).date()} to {end_date.date()}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 90)
    lines.append("")

    # Summary table
    lines.append("REGION SUMMARY")
    lines.append("-" * 90)
    header = f"{'Region':<22} {'Days':>5} {'Avg Risk':>9} {'Max':>7} {'Vol':>6} {'Trend':>10} {'Coverage':<20}"
    lines.append(header)
    lines.append("-" * 90)

    for region, stats in sorted(region_stability.items(), key=lambda x: -x[1]['avg_risk']):
        lines.append(
            f"{region:<22} "
            f"{stats['days_with_data']:>5} "
            f"{stats['avg_risk']:>9.3f} "
            f"{stats['max_risk']:>7.3f} "
            f"{stats['risk_volatility']:>6.3f} "
            f"{stats['methods_trend']:>10} "
            f"{stats['coverage_consistency'][:20]:<20}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Tier distribution
    lines.append("TIER DISTRIBUTION (days at each tier)")
    lines.append("-" * 90)
    header = f"{'Region':<22} {'NORMAL':>8} {'WATCH':>8} {'ELEVATED':>10} {'CRITICAL':>10} {'Max Consec Watch':>17}"
    lines.append(header)
    lines.append("-" * 90)

    for region, stats in sorted(region_stability.items()):
        td = stats['tier_distribution']
        lines.append(
            f"{region:<22} "
            f"{td['NORMAL']:>8} "
            f"{td['WATCH']:>8} "
            f"{td['ELEVATED']:>10} "
            f"{td['CRITICAL']:>10} "
            f"{stats.get('max_consecutive_watch_days', 0):>17}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Biggest changes
    lines.append("BIGGEST SINGLE-DAY CHANGES")
    lines.append("-" * 90)

    changes = [(region, stats['biggest_daily_change'], stats['biggest_change_date'])
               for region, stats in region_stability.items()
               if stats['biggest_daily_change'] > 0.05]

    if changes:
        for region, change, date in sorted(changes, key=lambda x: -x[1])[:10]:
            lines.append(f"  {region}: {change:.3f} change on {date}")
    else:
        lines.append("  No significant daily changes (>0.05) detected")

    lines.append("")

    # Alerts summary
    lines.append("ALERTS SUMMARY")
    lines.append("-" * 90)

    watch_regions = [r for r, s in region_stability.items()
                     if s['tier_distribution']['WATCH'] > 0 or
                        s['tier_distribution']['ELEVATED'] > 0 or
                        s['tier_distribution']['CRITICAL'] > 0]

    if watch_regions:
        lines.append(f"Regions with WATCH or higher: {len(watch_regions)}")
        for region in watch_regions:
            stats = region_stability[region]
            td = stats['tier_distribution']
            total_elevated = td['WATCH'] + td['ELEVATED'] + td['CRITICAL']
            pct = total_elevated / stats['days_with_data'] * 100 if stats['days_with_data'] > 0 else 0
            lines.append(f"  - {region}: {total_elevated} days ({pct:.0f}%) at WATCH+, "
                        f"max {stats.get('max_consecutive_watch_days', 0)} consecutive days")
    else:
        lines.append("No regions with WATCH or higher during this period")

    lines.append("")
    lines.append("=" * 90)
    lines.append("END OF REPORT")
    lines.append("=" * 90)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate GeoSpec Stability Report'
    )
    parser.add_argument(
        '--days', type=int, default=7,
        help='Number of days to include (default: 7)'
    )
    parser.add_argument(
        '--region', type=str, default=None,
        help='Single region to report on (default: all)'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='End date YYYY-MM-DD (default: yesterday)'
    )
    parser.add_argument(
        '--input-dir', type=str, default=None,
        help='Input directory with ensemble results (default: monitoring/data/ensemble_results)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: print to console)'
    )

    args = parser.parse_args()

    # Parse end date
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now() - timedelta(days=1)

    # Input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(__file__).parent.parent / 'data' / 'ensemble_results'

    # Regions
    regions = [args.region] if args.region else None

    # Generate report
    report = generate_report(
        output_dir=input_dir,
        end_date=end_date,
        days=args.days,
        regions=regions,
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        logger.info(f"Report saved to: {output_path}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
