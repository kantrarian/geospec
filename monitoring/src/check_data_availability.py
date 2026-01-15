#!/usr/bin/env python3
"""
check_data_availability.py - Diagnose seismic data availability

Checks hourly data availability going back 7 days to identify
where data stops being available for each station/network.
"""

import sys
from datetime import datetime, timedelta
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

# Stations to check (network, station, description)
STATIONS = [
    # California - SCEDC
    ('CI', 'CCC', 'Ridgecrest - China Lake'),
    ('CI', 'WBS', 'Mojave - Willow Beach'),
    ('CI', 'WMC', 'Coachella - White Water'),
    ('CI', 'GSC', 'Goldstone'),
    ('CI', 'PAS', 'Pasadena'),
    # NorCal - NCEDC
    ('BK', 'BKS', 'Berkeley'),
    ('NC', 'WENL', 'Hayward'),
    # Cascadia - IRIS
    ('CN', 'PHC', 'Port Hardy'),
    ('CN', 'WALA', 'Wala'),
    ('UW', 'LON', 'Longmire'),
    # Global reference
    ('IU', 'ANMO', 'Albuquerque (reference)'),
]

CLIENTS = {
    'CI': 'SCEDC',
    'NC': 'NCEDC',
    'BK': 'NCEDC',
    'CN': 'IRIS',
    'UW': 'IRIS',
    'IU': 'IRIS',
}


def check_station_availability(network, station, days_back=7, hour_step=6):
    """Check data availability for a station going back N days."""
    client_name = CLIENTS.get(network, 'IRIS')

    try:
        client = Client(client_name)
    except Exception as e:
        return None, f"Client error: {e}"

    now = datetime.utcnow()
    results = []

    for hours_ago in range(0, days_back * 24, hour_step):
        end_time = now - timedelta(hours=hours_ago)
        start_time = end_time - timedelta(hours=1)

        try:
            st = client.get_waveforms(
                network=network,
                station=station,
                location='*',
                channel='BHZ',
                starttime=UTCDateTime(start_time),
                endtime=UTCDateTime(end_time)
            )
            has_data = len(st) > 0 and len(st[0].data) > 0
            results.append((hours_ago, has_data, len(st[0].data) if has_data else 0))
        except FDSNNoDataException:
            results.append((hours_ago, False, 0))
        except Exception as e:
            results.append((hours_ago, None, str(e)[:30]))

    return results, None


def main():
    print("=" * 80)
    print("  SEISMIC DATA AVAILABILITY CHECK")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 80)
    print()
    print("Checking hourly data availability for past 7 days (6-hour steps)...")
    print()

    for network, station, desc in STATIONS:
        print(f"\n{network}.{station} ({desc})")
        print("-" * 60)

        results, error = check_station_availability(network, station)

        if error:
            print(f"  ERROR: {error}")
            continue

        if not results:
            print("  No results")
            continue

        # Find last available data
        last_available = None
        first_gap = None

        for hours_ago, has_data, samples in results:
            if has_data:
                if last_available is None:
                    last_available = hours_ago
            elif has_data is False and first_gap is None and last_available is not None:
                first_gap = hours_ago

        # Print timeline
        timeline = ""
        for hours_ago, has_data, samples in results:
            if has_data is True:
                timeline += "+"
            elif has_data is False:
                timeline += "-"
            else:
                timeline += "?"

        print(f"  Timeline (0h=now -> 168h=7d ago): {timeline}")
        print(f"  Legend: + = data available, - = no data, ? = error")

        if last_available is not None:
            print(f"  Most recent data: {last_available} hours ago")
        else:
            print(f"  Most recent data: NONE in past 7 days")

        # Count availability
        available = sum(1 for _, has_data, _ in results if has_data is True)
        total = len(results)
        print(f"  Availability: {available}/{total} time slots ({100*available/total:.0f}%)")

    print()
    print("=" * 80)
    print("  RECOMMENDATIONS")
    print("=" * 80)
    print("""
Based on results:
- If station shows recent gaps, increase latency offset
- If station has no data, find alternative station
- Check IRIS/SCEDC/NCEDC status pages for outages
""")


if __name__ == "__main__":
    main()
