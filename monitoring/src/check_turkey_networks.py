#!/usr/bin/env python3
"""
check_turkey_networks.py
Turkey Network Availability Scanner for Fault Correlation (Method 2)

Goal: Determine if we can activate Method 2 for Turkey using alternative networks
since KO (Kandilli Observatory) has restricted access.

Target Networks:
- TU: Turkish National Seismic Network (via ORFEUS/EIDA)
- GE: GEOFON (German Research Centre for Geosciences)
- HL: National Observatory of Athens (may have Aegean coverage)

Data Centers to Query:
- ORFEUS (ODC): https://www.orfeus-eu.org/fdsnws/
- GEOFON (GFZ): https://geofon.gfz-potsdam.de/fdsnws/
- KOERI: https://eida.koeri.boun.edu.tr/fdsnws/ (Kandilli - may work for metadata)

Author: R.J. Mathews / Claude
Date: January 2026
"""

import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    print("ERROR: ObsPy not installed. Run: pip install obspy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# REGION BOUNDING BOXES
# =============================================================================

TURKEY_REGIONS = {
    'istanbul_marmara': {
        'name': 'Istanbul / Marmara',
        'minlat': 40.0,
        'maxlat': 41.5,
        'minlon': 27.0,
        'maxlon': 30.5,
        'target_fault': 'North Anatolian Fault (Marmara segment)',
    },
    'turkey_kahramanmaras': {
        'name': 'Kahramanmaras / East Anatolian',
        'minlat': 36.5,
        'maxlat': 38.5,
        'minlon': 36.0,
        'maxlon': 38.5,
        'target_fault': 'East Anatolian Fault Zone',
    },
}

# Networks to scan (in priority order)
TARGET_NETWORKS = ['TU', 'GE', 'KO', 'HL', 'HT']

# Data centers to try
DATA_CENTERS = [
    ('ORFEUS', 'ODC'),           # ORFEUS Data Center - aggregates European data
    ('GEOFON', 'GFZ'),           # GEOFON - strong coverage in Turkey
    ('KOERI', 'KOERI'),          # Kandilli - may work for inventory
    ('IRIS', 'IRIS'),            # IRIS - global fallback
    ('RESIF', 'RESIF'),          # French - may have Mediterranean
]

# Minimum requirements for Fault Correlation
MIN_STATIONS_PER_REGION = 3  # Need 3+ for triangulation
MIN_AVAILABILITY_PCT = 80.0  # 80% data availability required
TEST_WINDOW_DAYS = 7  # Test last 7 days of data


@dataclass
class StationInfo:
    """Station metadata and availability."""
    network: str
    station: str
    location: str
    channel: str
    latitude: float
    longitude: float
    start_date: datetime
    end_date: Optional[datetime]
    data_center: str
    # Availability metrics (filled after waveform test)
    tested: bool = False
    available: bool = False
    availability_pct: float = 0.0
    latency_hours: float = 0.0
    notes: str = ""


@dataclass
class RegionReport:
    """Availability report for a region."""
    region_key: str
    region_name: str
    stations_found: int
    stations_tested: int
    stations_available: int
    networks_represented: List[str]
    best_availability_pct: float
    recommended_stations: List[StationInfo]
    can_enable_fc: bool
    notes: str


def get_client(data_center: str) -> Optional[Client]:
    """Get FDSN client for a data center."""
    try:
        client = Client(data_center)
        logger.debug(f"Connected to {data_center}")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to {data_center}: {e}")
        return None


def query_stations_in_region(
    region_key: str,
    networks: List[str] = None,
    data_centers: List[Tuple[str, str]] = None,
) -> List[StationInfo]:
    """
    Query all available stations within a region's bounding box.

    Args:
        region_key: Key from TURKEY_REGIONS
        networks: List of network codes to query (default: TARGET_NETWORKS)
        data_centers: List of (name, code) tuples (default: DATA_CENTERS)

    Returns:
        List of StationInfo objects
    """
    if networks is None:
        networks = TARGET_NETWORKS
    if data_centers is None:
        data_centers = DATA_CENTERS

    region = TURKEY_REGIONS.get(region_key)
    if not region:
        logger.error(f"Unknown region: {region_key}")
        return []

    logger.info(f"Scanning {region['name']} ({region_key})")
    logger.info(f"  Bounding box: {region['minlat']:.1f}-{region['maxlat']:.1f}N, "
                f"{region['minlon']:.1f}-{region['maxlon']:.1f}E")
    logger.info(f"  Target networks: {networks}")

    all_stations = []
    seen = set()  # Track (network, station) to avoid duplicates

    for dc_name, dc_code in data_centers:
        client = get_client(dc_code)
        if not client:
            continue

        for network in networks:
            try:
                inventory = client.get_stations(
                    network=network,
                    minlatitude=region['minlat'],
                    maxlatitude=region['maxlat'],
                    minlongitude=region['minlon'],
                    maxlongitude=region['maxlon'],
                    channel="BH?,HH?",  # Broadband channels for FC
                    level="channel",
                )

                for net in inventory:
                    for sta in net:
                        key = (net.code, sta.code)
                        if key in seen:
                            continue
                        seen.add(key)

                        # Get first BH/HH channel
                        for chan in sta:
                            if chan.code.startswith(('BH', 'HH')):
                                station_info = StationInfo(
                                    network=net.code,
                                    station=sta.code,
                                    location=chan.location_code or "",
                                    channel=chan.code,
                                    latitude=sta.latitude,
                                    longitude=sta.longitude,
                                    start_date=chan.start_date.datetime if chan.start_date else None,
                                    end_date=chan.end_date.datetime if chan.end_date else None,
                                    data_center=dc_name,
                                )
                                all_stations.append(station_info)
                                logger.debug(f"  Found: {net.code}.{sta.code}.{chan.location_code}.{chan.code} "
                                           f"({sta.latitude:.2f}, {sta.longitude:.2f}) via {dc_name}")
                                break  # Only need one channel per station

            except FDSNNoDataException:
                logger.debug(f"  No {network} stations in {region_key} via {dc_name}")
            except FDSNException as e:
                logger.warning(f"  FDSN error querying {network} via {dc_name}: {e}")
            except Exception as e:
                logger.warning(f"  Error querying {network} via {dc_name}: {e}")

    logger.info(f"  Found {len(all_stations)} unique stations")
    return all_stations


def test_station_availability(
    station: StationInfo,
    test_days: int = TEST_WINDOW_DAYS,
) -> StationInfo:
    """
    Test actual data availability for a station.

    Attempts to fetch waveform data for the test window and measures:
    - Whether data is available at all
    - Percentage of requested time with data
    - Data latency (how recent is the latest data)
    """
    station.tested = True

    # Try the data center that found this station first
    dc_priority = [station.data_center] + [dc[1] for dc in DATA_CENTERS if dc[0] != station.data_center]

    end_time = UTCDateTime.now()
    start_time = end_time - (test_days * 86400)

    for dc_code in dc_priority:
        client = get_client(dc_code)
        if not client:
            continue

        try:
            # Try to get waveforms
            st = client.get_waveforms(
                network=station.network,
                station=station.station,
                location=station.location or "*",
                channel=station.channel,
                starttime=start_time,
                endtime=end_time,
            )

            if len(st) == 0:
                continue

            # Calculate availability
            total_seconds = test_days * 86400
            data_seconds = sum(tr.stats.npts / tr.stats.sampling_rate for tr in st)
            availability = min(100.0, (data_seconds / total_seconds) * 100)

            # Calculate latency (time since last sample)
            latest_end = max(tr.stats.endtime for tr in st)
            latency_hours = (end_time - latest_end) / 3600

            station.available = True
            station.availability_pct = availability
            station.latency_hours = latency_hours
            station.notes = f"via {dc_code}"

            logger.info(f"  {station.network}.{station.station}: "
                       f"{availability:.1f}% available, {latency_hours:.1f}h latency")
            return station

        except FDSNNoDataException:
            continue
        except Exception as e:
            logger.debug(f"  Error testing {station.network}.{station.station} via {dc_code}: {e}")
            continue

    station.available = False
    station.notes = "No data available from any data center"
    logger.info(f"  {station.network}.{station.station}: NO DATA")
    return station


def generate_region_report(
    region_key: str,
    stations: List[StationInfo],
    test_availability: bool = True,
) -> RegionReport:
    """
    Generate availability report for a region.

    Args:
        region_key: Region key
        stations: List of stations found in region
        test_availability: Whether to test actual waveform availability

    Returns:
        RegionReport with recommendations
    """
    region = TURKEY_REGIONS[region_key]

    # Test availability if requested
    if test_availability:
        logger.info(f"Testing data availability for {len(stations)} stations...")
        for i, station in enumerate(stations):
            logger.info(f"  [{i+1}/{len(stations)}] Testing {station.network}.{station.station}...")
            test_station_availability(station)

    # Analyze results
    tested_stations = [s for s in stations if s.tested]
    available_stations = [s for s in stations if s.available]

    # Find stations meeting minimum requirements
    good_stations = [s for s in available_stations
                     if s.availability_pct >= MIN_AVAILABILITY_PCT]

    # Get unique networks represented
    networks = list(set(s.network for s in available_stations))

    # Best availability
    best_pct = max((s.availability_pct for s in available_stations), default=0.0)

    # Can we enable FC?
    can_enable = len(good_stations) >= MIN_STATIONS_PER_REGION

    # Build notes
    if can_enable:
        notes = f"READY: {len(good_stations)} stations meet requirements for Fault Correlation"
    elif len(available_stations) >= MIN_STATIONS_PER_REGION:
        notes = f"MARGINAL: {len(available_stations)} stations available but only {len(good_stations)} meet {MIN_AVAILABILITY_PCT}% threshold"
    elif len(available_stations) > 0:
        notes = f"INSUFFICIENT: Only {len(available_stations)} stations available, need {MIN_STATIONS_PER_REGION}"
    else:
        notes = "NO DATA: No stations returned waveform data"

    return RegionReport(
        region_key=region_key,
        region_name=region['name'],
        stations_found=len(stations),
        stations_tested=len(tested_stations),
        stations_available=len(available_stations),
        networks_represented=networks,
        best_availability_pct=best_pct,
        recommended_stations=good_stations[:5],  # Top 5
        can_enable_fc=can_enable,
        notes=notes,
    )


def print_report(report: RegionReport):
    """Print formatted report."""
    print()
    print("=" * 70)
    print(f"REGION: {report.region_name} ({report.region_key})")
    print("=" * 70)
    print(f"Stations found:     {report.stations_found}")
    print(f"Stations tested:    {report.stations_tested}")
    print(f"Stations available: {report.stations_available}")
    print(f"Networks:           {', '.join(report.networks_represented) or 'None'}")
    print(f"Best availability:  {report.best_availability_pct:.1f}%")
    print(f"FC Ready:           {'YES' if report.can_enable_fc else 'NO'}")
    print()
    print(f"Assessment: {report.notes}")

    if report.recommended_stations:
        print()
        print("Recommended Stations for Fault Correlation:")
        print("-" * 70)
        print(f"{'Network':<8} {'Station':<8} {'Channel':<8} {'Avail%':>8} {'Latency':>10} {'Location'}")
        print("-" * 70)
        for s in report.recommended_stations:
            print(f"{s.network:<8} {s.station:<8} {s.channel:<8} {s.availability_pct:>7.1f}% "
                  f"{s.latency_hours:>8.1f}h  ({s.latitude:.2f}, {s.longitude:.2f})")
    print()


def main():
    """Run Turkey network reconnaissance."""
    print()
    print("=" * 70)
    print("TURKEY NETWORK AVAILABILITY SCANNER")
    print("Goal: Enable Fault Correlation (Method 2) for Turkey Regions")
    print("=" * 70)
    print()
    print(f"Target Networks: {', '.join(TARGET_NETWORKS)}")
    print(f"Data Centers: {', '.join(dc[0] for dc in DATA_CENTERS)}")
    print(f"Test Window: Last {TEST_WINDOW_DAYS} days")
    print(f"Min Stations Required: {MIN_STATIONS_PER_REGION}")
    print(f"Min Availability: {MIN_AVAILABILITY_PCT}%")
    print()

    reports = []

    for region_key in TURKEY_REGIONS:
        print(f"\n{'='*70}")
        print(f"SCANNING: {region_key}")
        print(f"{'='*70}")

        # Query stations
        stations = query_stations_in_region(region_key)

        if not stations:
            print(f"  No stations found in {region_key}")
            reports.append(RegionReport(
                region_key=region_key,
                region_name=TURKEY_REGIONS[region_key]['name'],
                stations_found=0,
                stations_tested=0,
                stations_available=0,
                networks_represented=[],
                best_availability_pct=0.0,
                recommended_stations=[],
                can_enable_fc=False,
                notes="NO STATIONS: No stations found in bounding box",
            ))
            continue

        # Generate report with availability testing
        report = generate_region_report(region_key, stations, test_availability=True)
        reports.append(report)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    for report in reports:
        print_report(report)

    # Final recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    ready_regions = [r for r in reports if r.can_enable_fc]
    if ready_regions:
        print(f"\nREADY TO ENABLE FC: {len(ready_regions)} region(s)")
        for r in ready_regions:
            print(f"  - {r.region_name}: {r.stations_available} stations via {', '.join(r.networks_represented)}")
        print("\nNext step: Add these stations to fault_segments.py")
    else:
        print("\nNO REGIONS READY for Fault Correlation")
        print("Options:")
        print("  1. Try additional networks (check EMSC, ORFEUS routing)")
        print("  2. Accept lower availability threshold")
        print("  3. Continue with THD-only coverage for Turkey")

    print()
    return 0 if ready_regions else 1


if __name__ == "__main__":
    sys.exit(main())
