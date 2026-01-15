"""
backtest.py - Daily Replay Backtest Harness for GeoSpec Monitoring

Runs the monitoring ensemble over historical data to validate performance.
Produces metrics for FAR, hit rate, precision, and time-in-warning.

Usage:
    python backtest.py --start 2019-01-01 --end 2019-12-31 --regions ridgecrest,norcal_hayward

Output:
    - backtest_results.csv: Daily tier states
    - backtest_metrics.json: Aggregate statistics
    - backtest_events.csv: Event scoring details

Author: R.J. Mathews / Claude
Date: January 2026
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

# Local imports
from event_scorer import (
    EventScorer, EarthquakeEvent, ScoringResult, FalseAlarmResult,
    BacktestMetrics, load_events_from_usgs, flag_aftershocks, DEFAULT_CONFIG
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_date: datetime
    end_date: datetime
    regions: List[str]
    config_path: Optional[Path] = None

    # Scoring parameters (loaded from config or defaults)
    min_magnitude: float = 6.0
    lead_window_days: int = 7
    aftershock_exclusion_days: int = 30
    false_alarm_forward_days: int = 14
    hit_min_tier: int = 2

    # Station mapping (region -> station config)
    station_mapping: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path, start: datetime, end: datetime) -> 'BacktestConfig':
        """Load config from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Get regions from stations section
        stations = data.get('stations', {})
        regions = list(stations.keys())

        # Get evaluation parameters
        eval_config = data.get('evaluation', {})
        accept = data.get('acceptance_criteria', {})

        return cls(
            start_date=start,
            end_date=end,
            regions=regions,
            config_path=config_path,
            min_magnitude=eval_config.get('min_magnitude', 6.0),
            lead_window_days=eval_config.get('lead_window_days', 7),
            aftershock_exclusion_days=eval_config.get('aftershock_exclusion_days', 30),
            false_alarm_forward_days=eval_config.get('false_alarm_forward_days', 14),
            hit_min_tier=2,  # ELEVATED
            station_mapping=stations,
        )


# =============================================================================
# DAY RESULT
# =============================================================================

@dataclass
class DayResult:
    """Result of running monitoring for a single day."""
    date: datetime
    region_results: Dict[str, Dict]  # region -> {tier, combined_risk, components...}
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'region_results': self.region_results,
            'success': self.success,
            'error': self.error,
        }


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

class BacktestRunner:
    """
    Runs the monitoring ensemble over historical data.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest runner.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.daily_results: Dict[str, Dict] = {}  # date_str -> region -> result
        self.day_results: List[DayResult] = []

        logger.info(f"BacktestRunner initialized")
        logger.info(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
        logger.info(f"  Regions: {config.regions}")

    def run_day(self, target_date: datetime) -> DayResult:
        """
        Run monitoring ensemble for a single day.

        This is a simplified version that loads pre-computed results
        or generates mock results for testing.

        In production, this would call the full ensemble pipeline.
        """
        region_results = {}

        for region in self.config.regions:
            try:
                # Try to load from historical results if available
                result = self._load_historical_result(region, target_date)

                if result is None:
                    # Generate mock result for testing
                    result = self._generate_mock_result(region, target_date)

                region_results[region] = result

            except Exception as e:
                logger.warning(f"Failed to process {region} for {target_date.date()}: {e}")
                region_results[region] = {
                    'tier': -1,
                    'tier_name': 'DEGRADED',
                    'combined_risk': 0.0,
                    'error': str(e),
                }

        return DayResult(
            date=target_date,
            region_results=region_results,
            success=True,
        )

    def _load_historical_result(
        self,
        region: str,
        date: datetime
    ) -> Optional[Dict]:
        """
        Load pre-computed historical result if available.

        Looks for JSON files in monitoring/data/ensemble_results/
        """
        results_dir = Path(__file__).parent.parent / 'data' / 'ensemble_results'
        date_str = date.strftime('%Y-%m-%d')
        result_file = results_dir / f'ensemble_{date_str}.json'

        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                return data.get('regions', {}).get(region)
            except Exception:
                pass

        return None

    def _generate_mock_result(
        self,
        region: str,
        date: datetime
    ) -> Dict:
        """
        Generate mock result for testing when historical data unavailable.

        Uses simple random walk for testing purposes only.
        """
        import random
        random.seed(hash((region, date.strftime('%Y-%m-%d'))))

        # Generate correlated random risk
        base_risk = 0.15 + random.gauss(0, 0.05)
        base_risk = max(0, min(1, base_risk))

        # Determine tier
        if base_risk < 0.25:
            tier, tier_name = 0, 'NORMAL'
        elif base_risk < 0.50:
            tier, tier_name = 1, 'WATCH'
        elif base_risk < 0.75:
            tier, tier_name = 2, 'ELEVATED'
        else:
            tier, tier_name = 3, 'CRITICAL'

        return {
            'tier': tier,
            'tier_name': tier_name,
            'combined_risk': base_risk,
            'components': {
                'lambda_geo': {'available': False},
                'fault_correlation': {'available': False},
                'seismic_thd': {'available': False},
            },
            'note': 'Mock result for backtest testing',
        }

    def run_backtest(self) -> Tuple[List[DayResult], Dict[str, Dict]]:
        """
        Run full backtest over date range.

        Returns:
            Tuple of (day_results, daily_results_dict)
        """
        current = self.config.start_date
        n_days = (self.config.end_date - self.config.start_date).days + 1

        logger.info(f"Running backtest for {n_days} days...")

        day_count = 0
        while current <= self.config.end_date:
            day_result = self.run_day(current)
            self.day_results.append(day_result)

            # Store in lookup dict
            date_str = current.strftime('%Y-%m-%d')
            self.daily_results[date_str] = day_result.region_results

            day_count += 1
            if day_count % 30 == 0:
                logger.info(f"  Processed {day_count}/{n_days} days...")

            current += timedelta(days=1)

        logger.info(f"Backtest complete: {len(self.day_results)} days processed")

        return self.day_results, self.daily_results

    def score_against_events(
        self,
        events: List[EarthquakeEvent],
    ) -> Tuple[List[ScoringResult], List[FalseAlarmResult], BacktestMetrics]:
        """
        Score backtest results against historical events.

        Args:
            events: List of earthquake events to score against

        Returns:
            Tuple of (event_results, false_alarms, metrics)
        """
        # Create scorer with config
        scorer_config = {
            'min_magnitude': self.config.min_magnitude,
            'lead_window_days': self.config.lead_window_days,
            'aftershock_exclusion_days': self.config.aftershock_exclusion_days,
            'false_alarm_forward_days': self.config.false_alarm_forward_days,
            'hit_min_tier': self.config.hit_min_tier,
        }
        scorer = EventScorer(scorer_config)

        # Score each event
        event_results = []
        for event in events:
            result = scorer.score_event(event, self.daily_results)
            event_results.append(result)

        # Find false alarms
        false_alarms = scorer.find_false_alarms(self.daily_results, events)

        # Compute metrics
        metrics = scorer.compute_metrics(
            event_results,
            false_alarms,
            self.daily_results,
            self.config.regions,
        )

        return event_results, false_alarms, metrics


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def save_daily_results_csv(
    day_results: List[DayResult],
    output_path: Path,
):
    """Save daily results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Get all regions from first result
        regions = list(day_results[0].region_results.keys()) if day_results else []

        # Header
        header = ['date']
        for region in regions:
            header.extend([f'{region}_tier', f'{region}_risk'])
        writer.writerow(header)

        # Data
        for day in day_results:
            row = [day.date.strftime('%Y-%m-%d')]
            for region in regions:
                result = day.region_results.get(region, {})
                row.append(result.get('tier', -1))
                row.append(f"{result.get('combined_risk', 0.0):.4f}")
            writer.writerow(row)

    logger.info(f"Saved daily results to {output_path}")


def save_event_results_csv(
    event_results: List[ScoringResult],
    output_path: Path,
):
    """Save event scoring results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'event_id', 'date', 'magnitude', 'region',
            'classification', 'max_tier', 'max_risk',
            'lead_days', 'is_aftershock', 'notes'
        ])

        # Data
        for result in event_results:
            event = result.event
            writer.writerow([
                event.event_id,
                event.time.strftime('%Y-%m-%d'),
                f"{event.magnitude:.1f}",
                event.region or 'unknown',
                result.classification,
                result.max_tier_in_window,
                f"{result.max_risk_in_window:.4f}",
                result.warning_lead_days or '',
                event.is_aftershock,
                result.notes,
            ])

    logger.info(f"Saved event results to {output_path}")


def save_metrics_json(
    metrics: BacktestMetrics,
    event_results: List[ScoringResult],
    false_alarms: List[FalseAlarmResult],
    config: BacktestConfig,
    output_path: Path,
):
    """Save metrics and details to JSON."""
    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'start_date': config.start_date.strftime('%Y-%m-%d'),
            'end_date': config.end_date.strftime('%Y-%m-%d'),
            'regions': config.regions,
            'config_file': str(config.config_path) if config.config_path else None,
        },
        'scoring_config': {
            'min_magnitude': config.min_magnitude,
            'lead_window_days': config.lead_window_days,
            'aftershock_exclusion_days': config.aftershock_exclusion_days,
            'false_alarm_forward_days': config.false_alarm_forward_days,
            'hit_min_tier': config.hit_min_tier,
        },
        'metrics': metrics.to_dict(),
        'event_results_summary': {
            'hits': [r.to_dict() for r in event_results if r.classification == 'hit'],
            'misses': [r.to_dict() for r in event_results if r.classification == 'miss'],
        },
        'false_alarms': [fa.to_dict() for fa in false_alarms[:20]],  # Limit output
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved metrics to {output_path}")


def print_summary(metrics: BacktestMetrics, config: BacktestConfig):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Regions: {', '.join(config.regions)}")
    print(f"Total region-days: {metrics.total_region_days}")
    print("-" * 70)
    print(f"{'Metric':<30} {'Value':<15} {'Target':<15} {'Status'}")
    print("-" * 70)

    # Check against targets
    targets = {
        'hit_rate': (0.60, '>='),
        'precision': (0.30, '>='),
        'false_alarm_rate': (1.0/365/len(config.regions), '<='),  # ~1/year/region
        'time_in_warning_pct': (0.15, '<='),
    }

    print(f"{'Total Events':<30} {metrics.total_events:<15}")
    print(f"{'Hits':<30} {metrics.hits:<15}")
    print(f"{'Misses':<30} {metrics.misses:<15}")
    print(f"{'Aftershocks Excluded':<30} {metrics.aftershocks_excluded:<15}")
    print(f"{'False Alarms':<30} {metrics.false_alarms:<15}")
    print("-" * 70)

    for metric, (target, op) in targets.items():
        value = getattr(metrics, metric)

        if op == '>=':
            status = 'PASS' if value >= target else 'FAIL'
        else:
            status = 'PASS' if value <= target else 'FAIL'

        if metric == 'false_alarm_rate':
            value_str = f"{value:.6f}"
            target_str = f"<={target:.6f}"
        else:
            value_str = f"{value:.4f}"
            target_str = f"{op}{target:.2f}"

        print(f"{metric:<30} {value_str:<15} {target_str:<15} {status}")

    print(f"{'Mean Lead Time (days)':<30} {metrics.mean_lead_days:<15.1f}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run GeoSpec monitoring backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run backtest for 2019
    python backtest.py --start 2019-01-01 --end 2019-12-31

    # Run with specific regions
    python backtest.py --start 2019-01-01 --end 2019-12-31 --regions ridgecrest,norcal_hayward

    # Use config file
    python backtest.py --start 2019-01-01 --end 2019-12-31 --config ../config/backtest_config.yaml

    # Specify output directory
    python backtest.py --start 2019-01-01 --end 2019-12-31 --output-dir ../data/backtest_results
        """
    )

    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--regions', type=str, help='Comma-separated list of regions')
    parser.add_argument('--config', type=str, help='Path to backtest_config.yaml')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    # Load config
    if args.config:
        config_path = Path(args.config)
        config = BacktestConfig.from_yaml(config_path, start_date, end_date)

        # Override regions if specified
        if args.regions:
            config.regions = [r.strip() for r in args.regions.split(',')]
    else:
        # Default config
        regions = ['ridgecrest', 'norcal_hayward', 'cascadia']
        if args.regions:
            regions = [r.strip() for r in args.regions.split(',')]

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            regions=regions,
        )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'data' / 'backtest_results'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run backtest
    runner = BacktestRunner(config)
    day_results, daily_results = runner.run_backtest()

    # Load events
    logger.info("Loading earthquake events from USGS...")
    events = load_events_from_usgs(
        start_date=start_date - timedelta(days=30),  # Include events before start
        end_date=end_date + timedelta(days=30),      # Include events after end
        min_magnitude=config.min_magnitude - 0.5,    # Get slightly smaller for aftershock check
        regions=config.regions,
    )

    # Flag aftershocks
    events = flag_aftershocks(events, config.aftershock_exclusion_days)

    # Filter to target magnitude
    events = [e for e in events if e.magnitude >= config.min_magnitude]

    # Score against events
    event_results, false_alarms, metrics = runner.score_against_events(events)

    # Save outputs
    date_suffix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    save_daily_results_csv(
        day_results,
        output_dir / f'backtest_daily_{date_suffix}.csv'
    )

    save_event_results_csv(
        event_results,
        output_dir / f'backtest_events_{date_suffix}.csv'
    )

    save_metrics_json(
        metrics, event_results, false_alarms, config,
        output_dir / f'backtest_metrics_{date_suffix}.json'
    )

    # Print summary
    print_summary(metrics, config)

    # Return exit code based on acceptance criteria
    passes_acceptance = (
        metrics.hit_rate >= 0.60 and
        metrics.precision >= 0.30 and
        metrics.time_in_warning_pct <= 0.15
    )

    return 0 if passes_acceptance else 1


if __name__ == '__main__':
    sys.exit(main())
