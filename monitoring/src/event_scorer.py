"""
event_scorer.py - Event Scoring for Backtest Validation

Scores monitoring system performance against historical earthquake events.
Implements hit/miss/false-alarm classification per MONITORING_SPECIFICATION_v2.

Scoring Rules:
- Hit: Tier >= WATCH within lead window before M6+ event
- Miss: No warning in lead window before M6+ event
- False Alarm: Tier >= ELEVATED with no M6+ in forward window
- Aftershock Exclusion: Skip events within 30 days of larger event in same region

Author: R.J. Mathews / Claude
Date: January 2026
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default scoring parameters (can be overridden)
DEFAULT_CONFIG = {
    'min_magnitude': 6.0,           # Minimum magnitude to consider
    'lead_window_days': 7,          # Days before event to look for warning
    'aftershock_exclusion_days': 30,  # Skip events within N days of larger event
    'false_alarm_forward_days': 14,  # No M6+ within N days = false alarm
    'hit_min_tier': 2,              # Tier >= ELEVATED counts as hit (0=NORMAL, 1=WATCH, 2=ELEVATED, 3=CRITICAL)
    'event_region_buffer_km': 100,  # Event within N km of region boundary counts
}

# Region bounding boxes (approximate, for event filtering)
REGION_BOUNDS = {
    'ridgecrest': {'lat': (35.0, 36.5), 'lon': (-118.5, -117.0)},
    'socal_saf_mojave': {'lat': (34.0, 35.5), 'lon': (-118.5, -116.5)},
    'socal_saf_coachella': {'lat': (33.0, 34.0), 'lon': (-116.5, -115.0)},
    'norcal_hayward': {'lat': (37.3, 38.0), 'lon': (-122.3, -121.8)},
    'cascadia': {'lat': (42.0, 48.0), 'lon': (-125.0, -122.0)},
    'tokyo_kanto': {'lat': (34.5, 36.5), 'lon': (138.5, 141.0)},
    'istanbul_marmara': {'lat': (40.4, 41.2), 'lon': (28.5, 30.0)},
    'turkey_kahramanmaras': {'lat': (36.5, 38.5), 'lon': (36.0, 38.0)},
    'campi_flegrei': {'lat': (40.7, 40.9), 'lon': (14.0, 14.3)},
    'kaikoura': {'lat': (-43.0, -41.5), 'lon': (172.5, 174.5)},
    'anchorage': {'lat': (59.5, 62.5), 'lon': (-152.0, -148.0)},
    'kumamoto': {'lat': (32.0, 33.5), 'lon': (130.0, 131.5)},
    'hualien': {'lat': (23.0, 24.5), 'lon': (121.0, 122.0)},
    'mexico_guerrero': {'lat': (15.5, 18.5), 'lon': (-101.0, -97.0)},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EarthquakeEvent:
    """A single earthquake event."""
    event_id: str
    time: datetime
    latitude: float
    longitude: float
    magnitude: float
    depth_km: float
    region: Optional[str] = None  # Assigned region if within bounds
    is_aftershock: bool = False   # Flagged as aftershock of larger event

    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'time': self.time.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'magnitude': self.magnitude,
            'depth_km': self.depth_km,
            'region': self.region,
            'is_aftershock': self.is_aftershock,
        }


@dataclass
class ScoringResult:
    """Result of scoring a single event."""
    event: EarthquakeEvent
    classification: str  # 'hit', 'miss', 'aftershock_excluded'
    max_tier_in_window: int
    max_risk_in_window: float
    warning_lead_days: Optional[float] = None  # Days before event when warning issued
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'event': self.event.to_dict(),
            'classification': self.classification,
            'max_tier_in_window': self.max_tier_in_window,
            'max_risk_in_window': float(self.max_risk_in_window),
            'warning_lead_days': self.warning_lead_days,
            'notes': self.notes,
        }


@dataclass
class FalseAlarmResult:
    """A detected false alarm."""
    region: str
    date: datetime
    tier: int
    risk: float
    duration_days: int  # How many consecutive days at elevated tier
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'region': self.region,
            'date': self.date.isoformat(),
            'tier': self.tier,
            'risk': float(self.risk),
            'duration_days': self.duration_days,
            'notes': self.notes,
        }


@dataclass
class BacktestMetrics:
    """Aggregate metrics from backtest scoring."""
    total_events: int
    hits: int
    misses: int
    aftershocks_excluded: int
    false_alarms: int
    hit_rate: float           # hits / (hits + misses)
    precision: float          # hits / (hits + false_alarms)
    false_alarm_rate: float   # false_alarms / total_region_days
    time_in_warning_pct: float  # Days at WATCH+ / total days
    mean_lead_days: float     # Average warning lead time for hits
    total_region_days: int    # Total region-days in backtest

    def to_dict(self) -> Dict:
        return {
            'total_events': self.total_events,
            'hits': self.hits,
            'misses': self.misses,
            'aftershocks_excluded': self.aftershocks_excluded,
            'false_alarms': self.false_alarms,
            'hit_rate': round(self.hit_rate, 4),
            'precision': round(self.precision, 4),
            'false_alarm_rate': round(self.false_alarm_rate, 6),
            'time_in_warning_pct': round(self.time_in_warning_pct, 4),
            'mean_lead_days': round(self.mean_lead_days, 2),
            'total_region_days': self.total_region_days,
        }


# =============================================================================
# EVENT LOADING
# =============================================================================

def load_events_from_usgs(
    start_date: datetime,
    end_date: datetime,
    min_magnitude: float = 5.5,
    regions: Optional[List[str]] = None,
) -> List[EarthquakeEvent]:
    """
    Load earthquake events from USGS ComCat.

    Args:
        start_date: Start of period
        end_date: End of period
        min_magnitude: Minimum magnitude to fetch
        regions: If specified, only return events in these regions

    Returns:
        List of EarthquakeEvent objects
    """
    import urllib.request
    import urllib.parse

    # Build USGS API URL
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'starttime': start_date.strftime('%Y-%m-%d'),
        'endtime': end_date.strftime('%Y-%m-%d'),
        'minmagnitude': str(min_magnitude),
        'orderby': 'time-asc',
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    logger.info(f"Fetching events from USGS: {url}")

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
            time=datetime.fromtimestamp(props['time'] / 1000),
            latitude=coords[1],
            longitude=coords[0],
            magnitude=props['mag'],
            depth_km=coords[2],
        )

        # Assign region if within bounds
        event.region = assign_region(event.latitude, event.longitude)

        # Filter by region if specified
        if regions is None or event.region in regions:
            events.append(event)

    logger.info(f"Loaded {len(events)} events from USGS")
    return events


def assign_region(lat: float, lon: float) -> Optional[str]:
    """Assign a region based on lat/lon coordinates."""
    for region, bounds in REGION_BOUNDS.items():
        if (bounds['lat'][0] <= lat <= bounds['lat'][1] and
            bounds['lon'][0] <= lon <= bounds['lon'][1]):
            return region
    return None


def flag_aftershocks(
    events: List[EarthquakeEvent],
    exclusion_days: int = 30,
) -> List[EarthquakeEvent]:
    """
    Flag events that are aftershocks of larger events.

    An event is flagged as aftershock if:
    - It's within exclusion_days of a larger event
    - It's in the same region as the larger event

    Args:
        events: List of events sorted by time
        exclusion_days: Window for aftershock exclusion

    Returns:
        Events with is_aftershock flag set
    """
    # Sort by magnitude descending (process largest first)
    by_mag = sorted(events, key=lambda e: -e.magnitude)

    for i, event in enumerate(by_mag):
        if event.is_aftershock:
            continue

        # Check if any earlier larger event makes this an aftershock
        for larger in by_mag[:i]:
            if larger.region != event.region:
                continue
            if larger.is_aftershock:
                continue

            time_diff = abs((event.time - larger.time).days)
            if time_diff <= exclusion_days and larger.magnitude > event.magnitude:
                event.is_aftershock = True
                logger.debug(f"Flagged {event.event_id} M{event.magnitude} as aftershock of "
                           f"{larger.event_id} M{larger.magnitude}")
                break

    n_aftershocks = sum(1 for e in events if e.is_aftershock)
    logger.info(f"Flagged {n_aftershocks} aftershocks out of {len(events)} events")

    return events


# =============================================================================
# EVENT SCORING
# =============================================================================

class EventScorer:
    """
    Scores monitoring results against historical events.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the scorer.

        Args:
            config: Scoring configuration (uses defaults if not provided)
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        logger.info(f"EventScorer initialized with config: {self.config}")

    def score_event(
        self,
        event: EarthquakeEvent,
        daily_results: Dict[str, Dict],  # date_str -> region -> tier/risk
    ) -> ScoringResult:
        """
        Score a single event against daily monitoring results.

        Args:
            event: The earthquake event to score
            daily_results: Dict mapping date strings to region->tier/risk dicts

        Returns:
            ScoringResult with classification
        """
        if event.is_aftershock:
            return ScoringResult(
                event=event,
                classification='aftershock_excluded',
                max_tier_in_window=0,
                max_risk_in_window=0.0,
                notes='Excluded as aftershock of larger event',
            )

        if event.region is None:
            return ScoringResult(
                event=event,
                classification='miss',
                max_tier_in_window=0,
                max_risk_in_window=0.0,
                notes='Event not in monitored region',
            )

        # Look back in lead window for warnings
        lead_days = self.config['lead_window_days']
        hit_min_tier = self.config['hit_min_tier']

        max_tier = 0
        max_risk = 0.0
        warning_date = None

        for days_before in range(lead_days + 1):
            check_date = event.time.date() - timedelta(days=days_before)
            date_str = check_date.strftime('%Y-%m-%d')

            if date_str not in daily_results:
                continue

            region_result = daily_results[date_str].get(event.region)
            if region_result is None:
                continue

            tier = region_result.get('tier', 0)
            risk = region_result.get('combined_risk', 0.0)

            if tier > max_tier:
                max_tier = tier
                warning_date = check_date
            max_risk = max(max_risk, risk)

        # Classify
        if max_tier >= hit_min_tier:
            lead = (event.time.date() - warning_date).days if warning_date else None
            return ScoringResult(
                event=event,
                classification='hit',
                max_tier_in_window=max_tier,
                max_risk_in_window=max_risk,
                warning_lead_days=lead,
                notes=f'Warning issued {lead} days before event',
            )
        else:
            return ScoringResult(
                event=event,
                classification='miss',
                max_tier_in_window=max_tier,
                max_risk_in_window=max_risk,
                notes=f'Max tier {max_tier} below threshold {hit_min_tier}',
            )

    def find_false_alarms(
        self,
        daily_results: Dict[str, Dict],
        events: List[EarthquakeEvent],
    ) -> List[FalseAlarmResult]:
        """
        Find false alarms (elevated tiers with no event in forward window).

        Args:
            daily_results: Dict mapping date strings to region->tier/risk dicts
            events: List of events (for checking forward window)

        Returns:
            List of FalseAlarmResult objects
        """
        hit_min_tier = self.config['hit_min_tier']
        forward_days = self.config['false_alarm_forward_days']
        min_mag = self.config['min_magnitude']

        # Build event lookup by region
        events_by_region: Dict[str, List[EarthquakeEvent]] = {}
        for event in events:
            if event.region and not event.is_aftershock:
                events_by_region.setdefault(event.region, []).append(event)

        false_alarms = []
        checked_periods = set()  # (region, date) already counted

        for date_str, regions in sorted(daily_results.items()):
            check_date = datetime.strptime(date_str, '%Y-%m-%d').date()

            for region, result in regions.items():
                tier = result.get('tier', 0)
                risk = result.get('combined_risk', 0.0)

                # Skip if not elevated
                if tier < hit_min_tier:
                    continue

                # Skip if already counted in a previous FA period
                if (region, date_str) in checked_periods:
                    continue

                # Check forward window for events
                forward_start = check_date
                forward_end = check_date + timedelta(days=forward_days)

                has_event = False
                for event in events_by_region.get(region, []):
                    event_date = event.time.date()
                    if forward_start <= event_date <= forward_end:
                        if event.magnitude >= min_mag:
                            has_event = True
                            break

                if not has_event:
                    # Count consecutive days at elevated tier
                    duration = 1
                    for d in range(1, 30):  # Check up to 30 days
                        next_date = (check_date + timedelta(days=d)).strftime('%Y-%m-%d')
                        if next_date in daily_results:
                            next_result = daily_results[next_date].get(region, {})
                            if next_result.get('tier', 0) >= hit_min_tier:
                                duration += 1
                                checked_periods.add((region, next_date))
                            else:
                                break
                        else:
                            break

                    false_alarms.append(FalseAlarmResult(
                        region=region,
                        date=datetime.strptime(date_str, '%Y-%m-%d'),
                        tier=tier,
                        risk=risk,
                        duration_days=duration,
                        notes=f'No M{min_mag}+ within {forward_days} days',
                    ))

                checked_periods.add((region, date_str))

        logger.info(f"Found {len(false_alarms)} false alarm periods")
        return false_alarms

    def compute_metrics(
        self,
        event_results: List[ScoringResult],
        false_alarms: List[FalseAlarmResult],
        daily_results: Dict[str, Dict],
        regions: List[str],
    ) -> BacktestMetrics:
        """
        Compute aggregate metrics from scoring results.

        Args:
            event_results: List of event scoring results
            false_alarms: List of false alarm results
            daily_results: Dict of daily monitoring results
            regions: List of monitored regions

        Returns:
            BacktestMetrics object
        """
        hits = sum(1 for r in event_results if r.classification == 'hit')
        misses = sum(1 for r in event_results if r.classification == 'miss')
        aftershocks = sum(1 for r in event_results if r.classification == 'aftershock_excluded')

        # Hit rate
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

        # Precision
        n_fa = len(false_alarms)
        precision = hits / (hits + n_fa) if (hits + n_fa) > 0 else 0.0

        # False alarm rate (per region-day)
        total_region_days = len(daily_results) * len(regions)
        far = n_fa / total_region_days if total_region_days > 0 else 0.0

        # Time in warning
        hit_min_tier = self.config['hit_min_tier']
        days_at_warning = 0
        for date_str, region_results in daily_results.items():
            for region in regions:
                result = region_results.get(region, {})
                if result.get('tier', 0) >= hit_min_tier:
                    days_at_warning += 1

        time_in_warning = days_at_warning / total_region_days if total_region_days > 0 else 0.0

        # Mean lead time for hits
        lead_times = [r.warning_lead_days for r in event_results
                     if r.classification == 'hit' and r.warning_lead_days is not None]
        mean_lead = np.mean(lead_times) if lead_times else 0.0

        return BacktestMetrics(
            total_events=len(event_results),
            hits=hits,
            misses=misses,
            aftershocks_excluded=aftershocks,
            false_alarms=n_fa,
            hit_rate=hit_rate,
            precision=precision,
            false_alarm_rate=far,
            time_in_warning_pct=time_in_warning,
            mean_lead_days=mean_lead,
            total_region_days=total_region_days,
        )


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Event Scorer Test")
    print("=" * 60)

    # Test event loading
    events = load_events_from_usgs(
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2019, 12, 31),
        min_magnitude=5.5,
    )

    print(f"\nLoaded {len(events)} events")
    for event in events[:5]:
        print(f"  {event.time.date()} M{event.magnitude:.1f} {event.region or 'unassigned'}")

    # Flag aftershocks
    events = flag_aftershocks(events)
    n_aftershocks = sum(1 for e in events if e.is_aftershock)
    print(f"\nAftershocks flagged: {n_aftershocks}")

    # Test scoring (with mock daily results)
    scorer = EventScorer()

    # Mock daily results
    daily_results = {
        '2019-07-04': {
            'ridgecrest': {'tier': 2, 'combined_risk': 0.55},
        },
        '2019-07-05': {
            'ridgecrest': {'tier': 3, 'combined_risk': 0.82},
        },
    }

    # Score events
    for event in events:
        if event.region == 'ridgecrest':
            result = scorer.score_event(event, daily_results)
            print(f"\n{event.time.date()} M{event.magnitude:.1f}: {result.classification}")
            print(f"  Max tier: {result.max_tier_in_window}")
            print(f"  Notes: {result.notes}")
