#!/usr/bin/env python3
"""
validate_predictions.py - Real-time Prediction Validation

Automatically correlates past predictions with actual earthquake events to build
a track record of validated hits, misses, and false alarms.

Runs as part of daily monitoring or standalone to:
1. Look back at predictions from 7-14 days ago (configurable)
2. Fetch actual M5.5+ earthquakes from USGS for those regions
3. Classify as hit/miss/false alarm
4. Append validated events to monitoring/data/validated_events.json

This builds an ongoing track record without manual intervention.

Usage:
    python validate_predictions.py                    # Default 7-14 day lookback
    python validate_predictions.py --lookback-start 7 --lookback-end 14
    python validate_predictions.py --rebuild          # Rebuild from all historical data

Author: R.J. Mathews / Claude
Date: January 2026
"""

import argparse
import csv
import json
import logging
import urllib.request
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

VALIDATION_CONFIG = {
    'min_magnitude': 4.5,           # Minimum magnitude to count as significant event
    'hit_min_tier': 2,              # Tier >= ELEVATED counts as prediction (2=ELEVATED, 3=CRITICAL)
                                    # WATCH (Tier 1) is awareness only, not scored as prediction
    'lead_window_days': 7,          # Event within N days AFTER prediction = hit
    'lookback_start_days': 7,       # Start looking back N days ago
    'lookback_end_days': 14,        # End looking back N days ago
    'region_buffer_km': 150,        # Event within N km of region center counts
}

# Region definitions with center coordinates and bounding boxes
REGION_DEFINITIONS = {
    'ridgecrest': {
        'name': 'Ridgecrest',
        'center': (35.77, -117.60),
        'bounds': {'lat': (35.0, 36.5), 'lon': (-118.5, -117.0)},
    },
    'socal_saf_mojave': {
        'name': 'SoCal SAF Mojave',
        'center': (34.5, -117.5),
        'bounds': {'lat': (34.0, 35.5), 'lon': (-118.5, -116.5)},
    },
    'socal_saf_coachella': {
        'name': 'SoCal SAF Coachella',
        'center': (33.5, -116.0),
        'bounds': {'lat': (33.0, 34.0), 'lon': (-116.5, -115.0)},
    },
    'norcal_hayward': {
        'name': 'NorCal Hayward',
        'center': (37.6, -122.1),
        'bounds': {'lat': (37.3, 38.0), 'lon': (-122.5, -121.5)},
    },
    'cascadia': {
        'name': 'Cascadia',
        'center': (45.0, -123.5),
        'bounds': {'lat': (42.0, 49.0), 'lon': (-126.0, -122.0)},
    },
    'tokyo_kanto': {
        'name': 'Tokyo Kanto',
        'center': (35.7, 139.7),
        'bounds': {'lat': (34.5, 37.0), 'lon': (138.5, 141.5)},
    },
    'istanbul_marmara': {
        'name': 'Istanbul Marmara',
        'center': (40.8, 29.0),
        'bounds': {'lat': (40.4, 41.2), 'lon': (28.0, 30.5)},
    },
    'turkey_kahramanmaras': {
        'name': 'Turkey Kahramanmaras',
        'center': (37.5, 37.0),
        'bounds': {'lat': (36.5, 38.5), 'lon': (35.5, 38.5)},
    },
    'campi_flegrei': {
        'name': 'Campi Flegrei',
        'center': (40.82, 14.14),
        'bounds': {'lat': (40.7, 40.95), 'lon': (13.9, 14.4)},
    },
    'kaikoura': {
        'name': 'Kaikoura',
        'center': (-42.4, 173.7),
        'bounds': {'lat': (-43.0, -41.5), 'lon': (172.5, 174.5)},
    },
    'anchorage': {
        'name': 'Anchorage',
        'center': (61.2, -150.0),
        'bounds': {'lat': (59.5, 62.5), 'lon': (-152.0, -148.0)},
    },
    'kumamoto': {
        'name': 'Kumamoto',
        'center': (32.8, 130.7),
        'bounds': {'lat': (32.0, 33.5), 'lon': (130.0, 131.5)},
    },
    'hualien': {
        'name': 'Hualien',
        'center': (23.9, 121.6),
        'bounds': {'lat': (23.0, 24.5), 'lon': (121.0, 122.0)},
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Prediction:
    """A single prediction record."""
    date: str
    region: str
    tier: int
    risk: float
    confidence: float
    methods: int


@dataclass
class EarthquakeEvent:
    """An actual earthquake event from USGS."""
    event_id: str
    time: datetime
    latitude: float
    longitude: float
    magnitude: float
    depth_km: float
    place: str
    region: Optional[str] = None


@dataclass
class ValidatedEvent:
    """A validated prediction-event pair."""
    prediction_date: str
    prediction_region: str
    prediction_tier: int
    prediction_risk: float
    event_id: str
    event_time: str
    event_magnitude: float
    event_location: str
    lead_days: float           # Days between prediction and event
    classification: str        # 'hit', 'miss', 'false_alarm'
    validated_date: str        # When this validation was performed
    notes: str = ""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions_from_csv(
    csv_path: Path,
    start_date: datetime,
    end_date: datetime,
    min_tier: int = None,
) -> List[Prediction]:
    """
    Load predictions from daily_states.csv for a date range.

    Args:
        csv_path: Path to daily_states.csv
        start_date: Start of date range
        end_date: End of date range
        min_tier: Minimum tier to include (default: from VALIDATION_CONFIG)

    Returns:
        List of Prediction objects
    """
    # Use config default if not specified
    if min_tier is None:
        min_tier = VALIDATION_CONFIG['hit_min_tier']

    predictions = []

    if not csv_path.exists():
        logger.warning(f"Predictions CSV not found: {csv_path}")
        return predictions

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pred_date = datetime.strptime(row['date'], '%Y-%m-%d')
                if start_date <= pred_date <= end_date:
                    tier = int(row['tier'])
                    if tier >= min_tier:
                        predictions.append(Prediction(
                            date=row['date'],
                            region=row['region'],
                            tier=tier,
                            risk=float(row['risk']),
                            confidence=float(row['confidence']),
                            methods=int(row['methods']),
                        ))
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping row: {e}")
                continue

    logger.info(f"Loaded {len(predictions)} predictions with tier >= {min_tier} "
                f"from {start_date.date()} to {end_date.date()}")
    return predictions


def fetch_usgs_events(
    start_date: datetime,
    end_date: datetime,
    min_magnitude: float = 5.5,
) -> List[EarthquakeEvent]:
    """
    Fetch earthquake events from USGS ComCat API.

    Args:
        start_date: Start of date range
        end_date: End of date range
        min_magnitude: Minimum magnitude

    Returns:
        List of EarthquakeEvent objects
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'starttime': start_date.strftime('%Y-%m-%d'),
        'endtime': end_date.strftime('%Y-%m-%d'),
        'minmagnitude': str(min_magnitude),
        'orderby': 'time-asc',
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    logger.info(f"Fetching USGS events: M{min_magnitude}+ from {start_date.date()} to {end_date.date()}")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        logger.error(f"Failed to fetch USGS events: {e}")
        return []

    events = []
    for feature in data.get('features', []):
        props = feature['properties']
        coords = feature['geometry']['coordinates']

        event = EarthquakeEvent(
            event_id=feature['id'],
            time=datetime.utcfromtimestamp(props['time'] / 1000),
            latitude=coords[1],
            longitude=coords[0],
            magnitude=props['mag'],
            depth_km=coords[2] if len(coords) > 2 else 0,
            place=props.get('place', 'Unknown'),
        )

        # Assign region if within bounds
        event.region = assign_region_to_event(event)
        events.append(event)

    logger.info(f"Fetched {len(events)} events from USGS")
    return events


def assign_region_to_event(event: EarthquakeEvent) -> Optional[str]:
    """Assign a region to an event based on coordinates."""
    for region_id, region_def in REGION_DEFINITIONS.items():
        bounds = region_def['bounds']
        if (bounds['lat'][0] <= event.latitude <= bounds['lat'][1] and
            bounds['lon'][0] <= event.longitude <= bounds['lon'][1]):
            return region_id
    return None


# =============================================================================
# VALIDATION LOGIC
# =============================================================================

def validate_predictions(
    predictions: List[Prediction],
    events: List[EarthquakeEvent],
    lead_window_days: int = 7,
) -> Tuple[List[ValidatedEvent], Dict]:
    """
    Validate predictions against actual events.

    A prediction is a HIT if:
    - An event M5.5+ occurred in the same region
    - Within lead_window_days AFTER the prediction date

    A prediction is a FALSE ALARM if:
    - Tier >= WATCH but no M5.5+ event within lead_window_days

    Args:
        predictions: List of predictions to validate
        events: List of actual earthquake events
        lead_window_days: Days after prediction to look for events

    Returns:
        Tuple of (validated_events list, summary stats dict)
    """
    validated = []
    now = datetime.now()

    # Group events by region
    events_by_region: Dict[str, List[EarthquakeEvent]] = {}
    for event in events:
        if event.region:
            events_by_region.setdefault(event.region, []).append(event)

    # Track which predictions we've processed
    processed_predictions = set()

    # Stats
    stats = {
        'total_predictions': len(predictions),
        'hits': 0,
        'false_alarms': 0,
        'pending': 0,  # Too recent to validate
        'events_matched': 0,
    }

    for pred in predictions:
        pred_key = (pred.date, pred.region)
        if pred_key in processed_predictions:
            continue
        processed_predictions.add(pred_key)

        pred_date = datetime.strptime(pred.date, '%Y-%m-%d')
        window_end = pred_date + timedelta(days=lead_window_days)

        # Check if we're still in the validation window (can't validate yet)
        if window_end > now:
            stats['pending'] += 1
            continue

        # Look for events in this region within the lead window
        region_events = events_by_region.get(pred.region, [])
        matching_events = [
            e for e in region_events
            if pred_date <= e.time <= window_end
        ]

        if matching_events:
            # HIT - at least one event matched
            for event in matching_events:
                lead_days = (event.time - pred_date).total_seconds() / 86400

                validated.append(ValidatedEvent(
                    prediction_date=pred.date,
                    prediction_region=pred.region,
                    prediction_tier=pred.tier,
                    prediction_risk=pred.risk,
                    event_id=event.event_id,
                    event_time=event.time.isoformat(),
                    event_magnitude=event.magnitude,
                    event_location=event.place,
                    lead_days=round(lead_days, 2),
                    classification='hit',
                    validated_date=now.strftime('%Y-%m-%d'),
                    notes=f"Tier {pred.tier} prediction {lead_days:.1f} days before M{event.magnitude:.1f}",
                ))
                stats['events_matched'] += 1

            stats['hits'] += 1
        else:
            # FALSE ALARM - elevated prediction but no event
            validated.append(ValidatedEvent(
                prediction_date=pred.date,
                prediction_region=pred.region,
                prediction_tier=pred.tier,
                prediction_risk=pred.risk,
                event_id='',
                event_time='',
                event_magnitude=0.0,
                event_location='',
                lead_days=0.0,
                classification='false_alarm',
                validated_date=now.strftime('%Y-%m-%d'),
                notes=f"Tier {pred.tier} with no M{VALIDATION_CONFIG['min_magnitude']}+ event within {lead_window_days} days",
            ))
            stats['false_alarms'] += 1

    # Calculate rates
    validated_total = stats['hits'] + stats['false_alarms']
    if validated_total > 0:
        stats['hit_rate'] = round(stats['hits'] / validated_total, 4)
        stats['precision'] = round(stats['hits'] / validated_total, 4)
    else:
        stats['hit_rate'] = 0.0
        stats['precision'] = 0.0

    logger.info(f"Validation complete: {stats['hits']} hits, {stats['false_alarms']} false alarms, "
                f"{stats['pending']} pending")

    return validated, stats


# =============================================================================
# PERSISTENCE
# =============================================================================

def load_validated_events(json_path: Path) -> List[Dict]:
    """Load existing validated events from JSON file."""
    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('events', [])
    except Exception as e:
        logger.error(f"Failed to load validated events: {e}")
        return []


def save_validated_events(
    json_path: Path,
    events: List[ValidatedEvent],
    stats: Dict,
    existing_events: Optional[List[Dict]] = None,
) -> None:
    """
    Save validated events to JSON file, merging with existing data.

    Args:
        json_path: Path to output JSON file
        events: New validated events
        stats: Validation statistics
        existing_events: Existing events to merge with
    """
    # Convert new events to dicts
    new_event_dicts = [asdict(e) for e in events]

    # Merge with existing, avoiding duplicates
    if existing_events:
        existing_keys = {
            (e['prediction_date'], e['prediction_region'], e.get('event_id', ''))
            for e in existing_events
        }

        merged = list(existing_events)
        for event in new_event_dicts:
            key = (event['prediction_date'], event['prediction_region'], event.get('event_id', ''))
            if key not in existing_keys:
                merged.append(event)
                existing_keys.add(key)
    else:
        merged = new_event_dicts

    # Sort by prediction date descending
    merged.sort(key=lambda x: x['prediction_date'], reverse=True)

    # Calculate aggregate stats
    all_hits = [e for e in merged if e['classification'] == 'hit']
    all_false_alarms = [e for e in merged if e['classification'] == 'false_alarm']

    total = len(all_hits) + len(all_false_alarms)
    aggregate_stats = {
        'total_validated': total,
        'total_hits': len(all_hits),
        'total_false_alarms': len(all_false_alarms),
        'hit_rate': round(len(all_hits) / total, 4) if total > 0 else 0.0,
        'unique_events_predicted': len(set(e['event_id'] for e in all_hits if e['event_id'])),
        'last_updated': datetime.now().isoformat(),
    }

    # Build output structure
    output = {
        'metadata': {
            'description': 'GeoSpec Validated Predictions - Track Record',
            'validation_config': VALIDATION_CONFIG,
            'last_validation_run': datetime.now().isoformat(),
            'last_run_stats': stats,
        },
        'aggregate_stats': aggregate_stats,
        'events': merged,
    }

    # Ensure directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(merged)} validated events to {json_path}")
    logger.info(f"Aggregate: {aggregate_stats['total_hits']} hits, "
                f"{aggregate_stats['total_false_alarms']} false alarms, "
                f"hit rate: {aggregate_stats['hit_rate']:.1%}")


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

def run_validation(
    lookback_start_days: int = 7,
    lookback_end_days: int = 14,
    min_tier: int = None,
    min_magnitude: float = None,
    output_path: Optional[Path] = None,
) -> Tuple[List[ValidatedEvent], Dict]:
    """
    Run prediction validation for a lookback window.

    Args:
        lookback_start_days: Start of lookback window (days ago)
        lookback_end_days: End of lookback window (days ago)
        min_tier: Minimum tier to validate
        min_magnitude: Minimum earthquake magnitude
        output_path: Path to save results (default: monitoring/data/validated_events.json)

    Returns:
        Tuple of (validated events, stats)
    """
    # Use config defaults if not specified
    if min_tier is None:
        min_tier = VALIDATION_CONFIG['hit_min_tier']
    if min_magnitude is None:
        min_magnitude = VALIDATION_CONFIG['min_magnitude']

    # Paths
    monitoring_dir = Path(__file__).parent.parent
    csv_path = monitoring_dir / 'data' / 'ensemble_results' / 'daily_states.csv'

    if output_path is None:
        output_path = monitoring_dir / 'data' / 'validated_events.json'

    # Date range for predictions (lookback window)
    now = datetime.now()
    pred_start = now - timedelta(days=lookback_end_days)
    pred_end = now - timedelta(days=lookback_start_days)

    # Date range for events (predictions + lead window)
    event_end = now

    logger.info(f"Validating predictions from {pred_start.date()} to {pred_end.date()}")

    # Load predictions
    predictions = load_predictions_from_csv(csv_path, pred_start, pred_end, min_tier)

    if not predictions:
        logger.warning("No predictions found in lookback window")
        return [], {'total_predictions': 0}

    # Fetch events
    events = fetch_usgs_events(pred_start, event_end, min_magnitude)

    # Validate
    validated, stats = validate_predictions(
        predictions, events,
        lead_window_days=VALIDATION_CONFIG['lead_window_days']
    )

    # Load existing and merge
    existing = load_validated_events(output_path)

    # Save
    save_validated_events(output_path, validated, stats, existing)

    return validated, stats


def rebuild_all_validations(
    min_tier: int = None,
    min_magnitude: float = None,
    output_path: Optional[Path] = None,
) -> Tuple[List[ValidatedEvent], Dict]:
    """
    Rebuild validation from all historical data.

    This processes the entire daily_states.csv to build a complete track record.
    """
    # Use config defaults if not specified
    if min_tier is None:
        min_tier = VALIDATION_CONFIG['hit_min_tier']
    if min_magnitude is None:
        min_magnitude = VALIDATION_CONFIG['min_magnitude']

    monitoring_dir = Path(__file__).parent.parent
    csv_path = monitoring_dir / 'data' / 'ensemble_results' / 'daily_states.csv'

    if output_path is None:
        output_path = monitoring_dir / 'data' / 'validated_events.json'

    # Find date range in CSV
    min_date = datetime.max
    max_date = datetime.min

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = datetime.strptime(row['date'], '%Y-%m-%d')
                min_date = min(min_date, d)
                max_date = max(max_date, d)
            except:
                continue

    logger.info(f"Rebuilding validations from {min_date.date()} to {max_date.date()}")

    # Can only validate predictions old enough to have passed the lead window
    now = datetime.now()
    lead_window = VALIDATION_CONFIG['lead_window_days']
    validatable_end = now - timedelta(days=lead_window)

    if max_date > validatable_end:
        logger.info(f"Note: Predictions after {validatable_end.date()} are too recent to validate")

    # Load all validatable predictions
    predictions = load_predictions_from_csv(csv_path, min_date, validatable_end, min_tier)

    # Fetch events for entire range plus lead window
    events = fetch_usgs_events(min_date, now, min_magnitude)

    # Validate
    validated, stats = validate_predictions(
        predictions, events,
        lead_window_days=lead_window
    )

    # Save (replace existing)
    save_validated_events(output_path, validated, stats, None)

    return validated, stats


def main():
    parser = argparse.ArgumentParser(
        description='Validate GeoSpec predictions against actual earthquake events'
    )
    parser.add_argument(
        '--lookback-start', type=int, default=7,
        help='Start of lookback window in days (default: 7)'
    )
    parser.add_argument(
        '--lookback-end', type=int, default=14,
        help='End of lookback window in days (default: 14)'
    )
    parser.add_argument(
        '--min-tier', type=int, default=VALIDATION_CONFIG['hit_min_tier'],
        help=f"Minimum tier to validate (default: {VALIDATION_CONFIG['hit_min_tier']}=ELEVATED)"
    )
    parser.add_argument(
        '--min-magnitude', type=float, default=VALIDATION_CONFIG['min_magnitude'],
        help=f"Minimum earthquake magnitude (default: {VALIDATION_CONFIG['min_magnitude']})"
    )
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Rebuild from all historical data'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress console output'
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    print("\n" + "=" * 70)
    print("GEOSPEC PREDICTION VALIDATION")
    print("=" * 70 + "\n")

    if args.rebuild:
        print("Rebuilding from all historical data...\n")
        validated, stats = rebuild_all_validations(
            min_tier=args.min_tier,
            min_magnitude=args.min_magnitude,
        )
    else:
        print(f"Validating predictions from {args.lookback_end} to {args.lookback_start} days ago...\n")
        validated, stats = run_validation(
            lookback_start_days=args.lookback_start,
            lookback_end_days=args.lookback_end,
            min_tier=args.min_tier,
            min_magnitude=args.min_magnitude,
        )

    # Print summary
    print("\n" + "-" * 70)
    print("VALIDATION SUMMARY")
    print("-" * 70)
    print(f"  Predictions checked: {stats.get('total_predictions', 0)}")
    print(f"  Hits (correct):      {stats.get('hits', 0)}")
    print(f"  False alarms:        {stats.get('false_alarms', 0)}")
    print(f"  Pending (too recent):{stats.get('pending', 0)}")
    print(f"  Hit rate:            {stats.get('hit_rate', 0):.1%}")
    print("-" * 70)

    # Print recent hits
    hits = [v for v in validated if v.classification == 'hit']
    if hits:
        print("\nRECENT VALIDATED HITS:")
        for hit in hits[:5]:
            print(f"  {hit.prediction_date} {hit.prediction_region}: "
                  f"Tier {hit.prediction_tier} -> M{hit.event_magnitude:.1f} "
                  f"({hit.lead_days:.1f} days later)")

    print()
    return 0


if __name__ == "__main__":
    exit(main())
