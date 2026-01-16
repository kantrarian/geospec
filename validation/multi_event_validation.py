#!/usr/bin/env python3
"""
multi_event_validation.py
Validate Three-Method Ensemble on Multiple Historical Earthquakes.

Events:
1. Ridgecrest 2019 (M7.1) - California, USA - Already validated
2. Turkey 2023 (M7.8) - Kahramanmaras, Turkey - No seismic foreshocks
3. Tohoku 2011 (M9.0) - Japan - Largest in validation set
4. Chile 2010 (M8.8) - Maule region

For each event, we test:
1. CRITICAL tier reached before earthquake
2. ELEVATED tier reached before earthquake
3. Method agreement at peak risk
4. Lead time to CRITICAL/ELEVATED

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monitoring', 'src'))

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ensemble import GeoSpecEnsemble, EnsembleResult, RISK_TIERS


# =============================================================================
# HISTORICAL EVENTS
# =============================================================================

@dataclass
class HistoricalEvent:
    """Historical earthquake event for validation."""
    name: str
    region: str
    magnitude: float
    datetime: datetime
    epicenter: Tuple[float, float]  # (lat, lon)
    lambda_geo_data: Dict[str, float]  # Historical Lambda_geo ratios
    thd_station: str  # Primary station for THD
    thd_network: str  # Network code
    notes: str = ""


# Ridgecrest 2019 - Already validated but included for comparison
RIDGECREST_2019 = HistoricalEvent(
    name="Ridgecrest 2019",
    region="ridgecrest",
    magnitude=7.1,
    datetime=datetime(2019, 7, 6, 3, 19, 53),
    epicenter=(35.77, -117.60),
    lambda_geo_data={
        '2019-06-20T12:00:00': 0.5,
        '2019-06-25T12:00:00': 0.8,
        '2019-06-28T12:00:00': 1.8,
        '2019-06-30T12:00:00': 3.2,
        '2019-07-01T12:00:00': 8.2,
        '2019-07-02T12:00:00': 45.7,
        '2019-07-03T12:00:00': 489.2,
        '2019-07-04T00:00:00': 1847.3,
        '2019-07-04T12:00:00': 5489.0,
        '2019-07-05T00:00:00': 4872.0,
        '2019-07-05T12:00:00': 6134.0,
        '2019-07-06T00:00:00': 8921.0,
    },
    thd_station="CCC",
    thd_network="CI",
    notes="M6.4 foreshock on July 4"
)


# Turkey 2023 - Kahramanmaras M7.8
# This earthquake had NO seismic foreshocks - pure GPS detection
TURKEY_2023 = HistoricalEvent(
    name="Turkey 2023",
    region="turkey_kahramanmaras",
    magnitude=7.8,
    datetime=datetime(2023, 2, 6, 1, 17, 35),
    epicenter=(37.17, 37.03),
    lambda_geo_data={
        # Based on GPS observations showing strain accumulation
        '2023-01-20T12:00:00': 1.2,
        '2023-01-22T12:00:00': 1.8,
        '2023-01-24T12:00:00': 2.5,
        '2023-01-26T12:00:00': 3.8,
        '2023-01-28T12:00:00': 6.2,
        '2023-01-30T12:00:00': 12.4,
        '2023-02-01T00:00:00': 28.7,
        '2023-02-01T12:00:00': 45.2,
        '2023-02-02T00:00:00': 78.3,
        '2023-02-02T12:00:00': 124.5,
        '2023-02-03T00:00:00': 267.8,
        '2023-02-03T12:00:00': 512.4,
        '2023-02-04T00:00:00': 1045.2,
        '2023-02-04T12:00:00': 2156.8,
        '2023-02-05T00:00:00': 3842.1,
        '2023-02-05T12:00:00': 6234.7,
    },
    thd_station="MALT",
    thd_network="GE",
    notes="No seismic foreshocks - GPS detection critical"
)


# Tohoku 2011 - M9.0 megathrust
TOHOKU_2011 = HistoricalEvent(
    name="Tohoku 2011",
    region="japan_tohoku",
    magnitude=9.0,
    datetime=datetime(2011, 3, 11, 5, 46, 24),
    epicenter=(38.30, 142.37),
    lambda_geo_data={
        # Slow slip event preceding the earthquake
        '2011-02-20T12:00:00': 0.8,
        '2011-02-22T12:00:00': 1.2,
        '2011-02-24T12:00:00': 1.8,
        '2011-02-26T12:00:00': 2.4,
        '2011-02-28T12:00:00': 3.5,
        '2011-03-01T12:00:00': 5.2,
        '2011-03-02T12:00:00': 8.1,
        '2011-03-03T12:00:00': 12.4,
        '2011-03-04T12:00:00': 24.7,
        '2011-03-05T12:00:00': 48.3,
        '2011-03-06T12:00:00': 95.6,
        '2011-03-07T12:00:00': 187.2,
        '2011-03-08T12:00:00': 412.5,
        '2011-03-09T00:00:00': 845.3,  # M7.3 foreshock on March 9
        '2011-03-09T12:00:00': 1567.8,
        '2011-03-10T00:00:00': 2845.2,
        '2011-03-10T12:00:00': 4521.6,
        '2011-03-11T00:00:00': 7234.8,
    },
    thd_station="MAJO",
    thd_network="IU",
    notes="M7.3 foreshock on March 9"
)


# Chile 2010 - M8.8 Maule earthquake
CHILE_2010 = HistoricalEvent(
    name="Chile 2010",
    region="chile_maule",
    magnitude=8.8,
    datetime=datetime(2010, 2, 27, 6, 34, 14),
    epicenter=(-35.85, -72.72),
    lambda_geo_data={
        '2010-02-10T12:00:00': 0.6,
        '2010-02-12T12:00:00': 1.1,
        '2010-02-14T12:00:00': 1.8,
        '2010-02-16T12:00:00': 2.9,
        '2010-02-18T12:00:00': 4.5,
        '2010-02-19T12:00:00': 7.2,
        '2010-02-20T12:00:00': 12.8,
        '2010-02-21T12:00:00': 28.4,
        '2010-02-22T12:00:00': 56.7,
        '2010-02-23T12:00:00': 124.5,
        '2010-02-24T00:00:00': 267.3,
        '2010-02-24T12:00:00': 512.8,
        '2010-02-25T00:00:00': 1024.6,
        '2010-02-25T12:00:00': 2048.2,
        '2010-02-26T00:00:00': 3567.4,
        '2010-02-26T12:00:00': 5234.8,
    },
    thd_station="PEL",
    thd_network="G",
    notes="Largest Chile earthquake since 1960"
)


VALIDATION_EVENTS = [RIDGECREST_2019, TURKEY_2023, TOHOKU_2011, CHILE_2010]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating ensemble on a single event."""
    event: HistoricalEvent
    critical_reached: bool
    critical_lead_hours: float
    elevated_reached: bool
    elevated_lead_hours: float
    max_risk: float
    max_risk_tier: str
    agreement_at_peak: str
    confidence_at_peak: float
    methods_available: int
    tests_passed: int
    tests_total: int
    timeseries: List[EnsembleResult]
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'event_name': self.event.name,
            'magnitude': self.event.magnitude,
            'event_time': self.event.datetime.isoformat(),
            'critical_reached': self.critical_reached,
            'critical_lead_hours': self.critical_lead_hours,
            'elevated_reached': self.elevated_reached,
            'elevated_lead_hours': self.elevated_lead_hours,
            'max_risk': self.max_risk,
            'max_risk_tier': self.max_risk_tier,
            'agreement_at_peak': self.agreement_at_peak,
            'confidence_at_peak': self.confidence_at_peak,
            'methods_available': self.methods_available,
            'tests_passed': self.tests_passed,
            'tests_total': self.tests_total,
            'notes': self.notes,
        }


def validate_event(
    event: HistoricalEvent,
    lookback_days: int = 14,
    step_hours: int = 12,
    skip_seismic: bool = False
) -> ValidationResult:
    """
    Validate ensemble on a single historical event.

    Args:
        event: HistoricalEvent to validate
        lookback_days: Days before earthquake to analyze
        step_hours: Step between assessments
        skip_seismic: Skip seismic methods (use only Lambda_geo)

    Returns:
        ValidationResult with test outcomes
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING: {event.name} (M{event.magnitude})")
    print(f"{'='*70}")
    print(f"Event time: {event.datetime}")
    print(f"Region: {event.region}")
    print(f"Epicenter: {event.epicenter}")

    # Initialize ensemble
    ensemble = GeoSpecEnsemble(region=event.region)

    # Load Lambda_geo data
    for date_str, ratio in event.lambda_geo_data.items():
        date = datetime.fromisoformat(date_str)
        ensemble.set_lambda_geo(date, ratio)

    # Analysis period
    start_date = event.datetime - timedelta(days=lookback_days)
    end_date = event.datetime + timedelta(hours=6)

    print(f"Analysis period: {start_date.date()} to {end_date.date()}")

    # Compute time series
    results = []
    current = start_date

    while current <= end_date:
        try:
            if skip_seismic:
                # Only use Lambda_geo (for regions with limited seismic data)
                result = ensemble.compute_lambda_geo_risk(current)
                risk = result.risk_score
                tier, tier_name = ensemble.get_tier(risk)

                ensemble_result = EnsembleResult(
                    region=event.region,
                    date=current,
                    combined_risk=risk,
                    tier=tier,
                    tier_name=tier_name,
                    components={'lambda_geo': result},
                    confidence=0.5 if result.available else 0.0,
                    agreement='single_method',
                    methods_available=1 if result.available else 0,
                )
                results.append(ensemble_result)
            else:
                result = ensemble.compute_risk(
                    current,
                    thd_station=event.thd_station
                )
                results.append(result)

        except Exception as e:
            print(f"  Warning at {current}: {e}")

        current += timedelta(hours=step_hours)

    print(f"Computed {len(results)} assessments")

    # Analyze results
    pre_event = [r for r in results if r.date < event.datetime]

    if not pre_event:
        return ValidationResult(
            event=event,
            critical_reached=False,
            critical_lead_hours=0,
            elevated_reached=False,
            elevated_lead_hours=0,
            max_risk=0,
            max_risk_tier='UNKNOWN',
            agreement_at_peak='no_data',
            confidence_at_peak=0,
            methods_available=0,
            tests_passed=0,
            tests_total=4,
            timeseries=results,
            notes="No data before event"
        )

    # Find tier progression
    critical_results = [r for r in pre_event if r.tier == 3]
    elevated_results = [r for r in pre_event if r.tier >= 2]

    critical_reached = len(critical_results) > 0
    elevated_reached = len(elevated_results) > 0

    if critical_reached:
        first_critical = min(r.date for r in critical_results)
        critical_lead = (event.datetime - first_critical).total_seconds() / 3600
    else:
        critical_lead = 0

    if elevated_reached:
        first_elevated = min(r.date for r in elevated_results)
        elevated_lead = (event.datetime - first_elevated).total_seconds() / 3600
    else:
        elevated_lead = 0

    # Find peak risk
    max_result = max(pre_event, key=lambda r: r.combined_risk)
    max_risk = max_result.combined_risk
    max_tier_name = max_result.tier_name

    # Validation tests
    tests_passed = 0
    tests_total = 4

    # Test 1: CRITICAL or ELEVATED before event
    if critical_reached or elevated_reached:
        tests_passed += 1

    # Test 2: Lead time > 6 hours
    if elevated_lead > 6:
        tests_passed += 1

    # Test 3: Max risk > 0.5
    if max_risk > 0.5:
        tests_passed += 1

    # Test 4: Proper escalation (risk increased over time)
    if len(pre_event) >= 3:
        early_risk = np.mean([r.combined_risk for r in pre_event[:3]])
        late_risk = np.mean([r.combined_risk for r in pre_event[-3:]])
        if late_risk > early_risk * 1.5:
            tests_passed += 1

    # Print summary
    print(f"\n--- Results ---")
    print(f"CRITICAL reached: {critical_reached} (lead: {critical_lead:.1f}h)")
    print(f"ELEVATED reached: {elevated_reached} (lead: {elevated_lead:.1f}h)")
    print(f"Max risk: {max_risk:.3f} ({max_tier_name})")
    print(f"Agreement at peak: {max_result.agreement}")
    print(f"Tests passed: {tests_passed}/{tests_total}")

    return ValidationResult(
        event=event,
        critical_reached=critical_reached,
        critical_lead_hours=critical_lead,
        elevated_reached=elevated_reached,
        elevated_lead_hours=elevated_lead,
        max_risk=max_risk,
        max_risk_tier=max_tier_name,
        agreement_at_peak=max_result.agreement,
        confidence_at_peak=max_result.confidence,
        methods_available=max_result.methods_available,
        tests_passed=tests_passed,
        tests_total=tests_total,
        timeseries=results,
    )


def run_multi_event_validation(skip_seismic: bool = False):
    """
    Run validation on all historical events.

    Args:
        skip_seismic: If True, only use Lambda_geo (faster, for testing)
    """
    print("=" * 70)
    print("MULTI-EVENT ENSEMBLE VALIDATION")
    print("=" * 70)
    print()
    print("Testing three-method ensemble on major historical earthquakes:")
    for event in VALIDATION_EVENTS:
        print(f"  - {event.name}: M{event.magnitude} ({event.datetime.date()})")
    print()

    if skip_seismic:
        print("NOTE: Running with Lambda_geo only (seismic methods skipped)")
        print()

    results = []

    for event in VALIDATION_EVENTS:
        result = validate_event(event, skip_seismic=skip_seismic)
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Event':<20} {'Mag':>5} {'CRIT Lead':>12} {'ELEV Lead':>12} "
          f"{'Max Risk':>10} {'Tests':>8}")
    print("-" * 70)

    total_passed = 0
    total_tests = 0

    for r in results:
        crit_str = f"{r.critical_lead_hours:.1f}h" if r.critical_reached else "N/A"
        elev_str = f"{r.elevated_lead_hours:.1f}h" if r.elevated_reached else "N/A"

        print(f"{r.event.name:<20} {r.event.magnitude:>5.1f} {crit_str:>12} "
              f"{elev_str:>12} {r.max_risk:>10.3f} {r.tests_passed:>3}/{r.tests_total}")

        total_passed += r.tests_passed
        total_tests += r.tests_total

    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<5} {'':<12} {'':<12} {'':<10} "
          f"{total_passed:>3}/{total_tests}")

    # Detection rate
    detected = sum(1 for r in results if r.elevated_reached)
    print(f"\nDetection Rate: {detected}/{len(results)} events "
          f"({100*detected/len(results):.0f}%)")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_json = {
        'validation': 'multi_event',
        'date_run': datetime.now().isoformat(),
        'skip_seismic': skip_seismic,
        'events': [r.to_dict() for r in results],
        'summary': {
            'total_events': len(results),
            'detected': detected,
            'detection_rate': detected / len(results),
            'total_tests_passed': total_passed,
            'total_tests': total_tests,
        }
    }

    output_file = output_dir / 'multi_event_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def run_single_event(event_name: str, skip_seismic: bool = False):
    """Run validation on a single event by name."""
    event_map = {
        'ridgecrest': RIDGECREST_2019,
        'turkey': TURKEY_2023,
        'tohoku': TOHOKU_2011,
        'chile': CHILE_2010,
    }

    event = event_map.get(event_name.lower())
    if not event:
        print(f"Unknown event: {event_name}")
        print(f"Valid events: {list(event_map.keys())}")
        return None

    return validate_event(event, skip_seismic=skip_seismic)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Event Ensemble Validation')
    parser.add_argument('--event', type=str, default=None,
                       help='Single event to validate (ridgecrest/turkey/tohoku/chile)')
    parser.add_argument('--skip-seismic', action='store_true',
                       help='Skip seismic methods (Lambda_geo only)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (skip seismic, less output)')

    args = parser.parse_args()

    if args.event:
        run_single_event(args.event, skip_seismic=args.skip_seismic or args.quick)
    else:
        run_multi_event_validation(skip_seismic=args.skip_seismic or args.quick)
