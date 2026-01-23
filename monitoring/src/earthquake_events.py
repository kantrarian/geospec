"""
earthquake_events.py
Fetch recent earthquake events from USGS for correlation with precursor signals.

Uses USGS FDSN Event Web Service:
https://earthquake.usgs.gov/fdsnws/event/1/

Purpose:
- Show recent M4+ events in monitored regions
- Allow visual correlation between Lambda_geo signals and actual earthquakes
- Support validation of precursor detection

Author: R.J. Mathews
Date: January 2026
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# REGION BOUNDING BOXES
# =============================================================================
# Format: (minlat, maxlat, minlon, maxlon)
# These should encompass the monitored fault segments plus some buffer

REGION_BOUNDS = {
    # California
    'ridgecrest': (35.0, 36.5, -118.5, -116.5),
    'socal_saf_mojave': (34.0, 36.5, -118.5, -115.5),
    'socal_saf_coachella': (33.0, 34.5, -117.0, -115.0),
    'socal_coachella': (33.0, 34.5, -117.0, -115.0),  # Alias
    'norcal_hayward': (37.0, 38.5, -123.0, -121.5),
    'cascadia': (42.0, 49.0, -130.0, -122.0),  # Extended westward for offshore events

    # Alaska
    'anchorage': (59.0, 63.0, -152.0, -147.0),

    # Japan
    'tokyo_kanto': (34.5, 37.0, 138.5, 141.5),
    'japan_tohoku': (36.0, 42.0, 139.0, 145.0),
    'kumamoto': (31.5, 34.0, 129.5, 132.0),  # Kyushu region

    # Taiwan
    'hualien': (23.0, 25.5, 120.5, 122.5),

    # New Zealand
    'kaikoura': (-43.5, -41.5, 172.0, 175.0),

    # Turkey
    'istanbul_marmara': (40.0, 41.5, 27.5, 30.5),
    'turkey_kahramanmaras': (36.5, 38.5, 36.0, 38.5),

    # Italy
    'campi_flegrei': (40.5, 41.0, 13.8, 14.5),

    # South America
    'chile_maule': (-37.0, -34.0, -73.5, -70.5),
}

# Minimum magnitude to fetch (lower = more events but more noise)
DEFAULT_MIN_MAGNITUDE = 4.0

# Days of history to fetch
DEFAULT_LOOKBACK_DAYS = 90


@dataclass
class EarthquakeEvent:
    """Single earthquake event from USGS."""
    event_id: str
    time: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    mag_type: str
    place: str
    url: str

    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'time': self.time.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'depth_km': self.depth_km,
            'magnitude': self.magnitude,
            'mag_type': self.mag_type,
            'place': self.place,
            'url': self.url,
        }

    def __str__(self) -> str:
        return f"M{self.magnitude:.1f} {self.place} ({self.time.strftime('%Y-%m-%d %H:%M')})"


@dataclass
class RegionEvents:
    """Collection of earthquake events for a region."""
    region: str
    bounds: tuple
    events: List[EarthquakeEvent] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    largest_event: Optional[EarthquakeEvent] = None
    most_recent_event: Optional[EarthquakeEvent] = None
    event_count: int = 0

    def to_dict(self) -> Dict:
        # Sort events by date (newest first), then by magnitude (largest first)
        sorted_events = sorted(
            self.events,
            key=lambda e: (e.time, e.magnitude),
            reverse=True
        )

        # Filter M6.5+ events for chart markers (30-day window)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        m65_plus_events = [
            {
                'event_id': e.event_id,
                'time': e.time.strftime('%Y-%m-%d'),  # Date only for chart matching
                'magnitude': e.magnitude,
                'place': e.place,
            }
            for e in self.events
            if e.magnitude >= 6.5 and e.time >= thirty_days_ago
        ]

        return {
            'region': self.region,
            'bounds': {
                'minlat': self.bounds[0],
                'maxlat': self.bounds[1],
                'minlon': self.bounds[2],
                'maxlon': self.bounds[3],
            },
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'event_count': self.event_count,
            'largest_event': self.largest_event.to_dict() if self.largest_event else None,
            'most_recent_event': self.most_recent_event.to_dict() if self.most_recent_event else None,
            'events': [e.to_dict() for e in sorted_events[:5]],  # Top 5 most recent
            'm65_plus_events': m65_plus_events,  # M6.5+ events for chart markers
        }


def fetch_usgs_events(
    minlat: float,
    maxlat: float,
    minlon: float,
    maxlon: float,
    start_time: datetime,
    end_time: datetime,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    timeout: int = 30,
) -> List[EarthquakeEvent]:
    """
    Fetch earthquake events from USGS FDSN web service.

    Args:
        minlat, maxlat, minlon, maxlon: Bounding box
        start_time: Start of search window
        end_time: End of search window
        min_magnitude: Minimum magnitude to fetch
        timeout: Request timeout in seconds

    Returns:
        List of EarthquakeEvent objects, sorted by time (newest first)
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        'format': 'geojson',
        'minlatitude': minlat,
        'maxlatitude': maxlat,
        'minlongitude': minlon,
        'maxlongitude': maxlon,
        'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'minmagnitude': min_magnitude,
        'orderby': 'time',  # Newest first
    }

    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query_string}"

    logger.debug(f"Fetching USGS events: {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'GeoSpec/1.0 (earthquake research)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        logger.error(f"USGS fetch failed: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"USGS response parse failed: {e}")
        return []

    events = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        coords = geom.get('coordinates', [0, 0, 0])

        # Parse time (USGS returns milliseconds since epoch)
        time_ms = props.get('time', 0)
        event_time = datetime.utcfromtimestamp(time_ms / 1000)

        event = EarthquakeEvent(
            event_id=feature.get('id', ''),
            time=event_time,
            latitude=coords[1],
            longitude=coords[0],
            depth_km=coords[2],
            magnitude=props.get('mag', 0.0),
            mag_type=props.get('magType', ''),
            place=props.get('place', 'Unknown'),
            url=props.get('url', ''),
        )
        events.append(event)

    logger.info(f"Fetched {len(events)} events (M>={min_magnitude})")
    return events


def fetch_region_events(
    region: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
) -> Optional[RegionEvents]:
    """
    Fetch earthquake events for a monitored region.

    Args:
        region: Region key (must be in REGION_BOUNDS)
        lookback_days: Days of history to fetch
        min_magnitude: Minimum magnitude

    Returns:
        RegionEvents object or None if region not found
    """
    if region not in REGION_BOUNDS:
        logger.warning(f"Unknown region: {region}. Valid: {list(REGION_BOUNDS.keys())}")
        return None

    bounds = REGION_BOUNDS[region]
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)

    events = fetch_usgs_events(
        minlat=bounds[0],
        maxlat=bounds[1],
        minlon=bounds[2],
        maxlon=bounds[3],
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_magnitude,
    )

    result = RegionEvents(
        region=region,
        bounds=bounds,
        events=events,
        last_updated=datetime.utcnow(),
        event_count=len(events),
    )

    if events:
        # Find largest and most recent
        result.most_recent_event = events[0]  # Already sorted by time
        result.largest_event = max(events, key=lambda e: e.magnitude)

    return result


def fetch_all_regions(
    regions: Optional[List[str]] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
) -> Dict[str, RegionEvents]:
    """
    Fetch earthquake events for multiple regions.

    Args:
        regions: List of region keys, or None for all defined regions
        lookback_days: Days of history to fetch
        min_magnitude: Minimum magnitude

    Returns:
        Dict mapping region names to RegionEvents
    """
    if regions is None:
        regions = list(REGION_BOUNDS.keys())

    results = {}
    for region in regions:
        logger.info(f"Fetching events for {region}...")
        result = fetch_region_events(region, lookback_days, min_magnitude)
        if result:
            results[region] = result
            if result.largest_event:
                logger.info(f"  Largest: {result.largest_event}")
            else:
                logger.info(f"  No M>={min_magnitude} events in last {lookback_days} days")

    return results


def save_events_json(
    events_by_region: Dict[str, RegionEvents],
    output_path: Path,
) -> None:
    """Save events to JSON file for dashboard."""
    output = {
        'generated': datetime.utcnow().isoformat(),
        'lookback_days': DEFAULT_LOOKBACK_DAYS,
        'min_magnitude': DEFAULT_MIN_MAGNITUDE,
        'regions': {k: v.to_dict() for k, v in events_by_region.items()},
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved events to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("USGS Earthquake Event Fetcher")
    print("=" * 60)

    # Test regions (or all if no args)
    regions = sys.argv[1:] if len(sys.argv) > 1 else None

    results = fetch_all_regions(
        regions=regions,
        lookback_days=90,
        min_magnitude=4.0,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for region, data in results.items():
        print(f"\n{region.upper()}:")
        print(f"  Events (M>=4.0): {data.event_count}")
        if data.largest_event:
            e = data.largest_event
            # Handle Unicode characters safely
            place = e.place.encode('ascii', 'replace').decode('ascii')
            print(f"  Largest: M{e.magnitude:.1f} on {e.time.strftime('%Y-%m-%d')} - {place}")
        if data.most_recent_event:
            e = data.most_recent_event
            place = e.place.encode('ascii', 'replace').decode('ascii')
            print(f"  Most recent: M{e.magnitude:.1f} on {e.time.strftime('%Y-%m-%d')} - {place}")

    # Save to file
    output_path = Path(__file__).parent.parent / "dashboard" / "events.json"
    save_events_json(results, output_path)
    print(f"\nSaved to: {output_path}")
