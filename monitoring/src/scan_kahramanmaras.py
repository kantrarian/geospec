"""
Scan for available seismic stations in the Kahramanmaras/East Anatolian Fault region.
Tests waveform availability across multiple data centers.

Region bounds (2023 M7.8 earthquake zone):
- Lat: 36.5 - 38.5
- Lon: 35.5 - 39.5
"""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from datetime import datetime, timedelta
import sys

# Kahramanmaras region bounds (expanded for better coverage)
MIN_LAT, MAX_LAT = 36.0, 39.0
MIN_LON, MAX_LON = 35.0, 40.0

# Data centers to scan
DATA_CENTERS = [
    ('KOERI', ['KO', 'TU']),      # Primary for Turkey
    ('IRIS', ['IU', 'II', 'GE']),  # Global networks
    ('ODC', ['HL', 'HT', 'TU']),   # ORFEUS - European
    ('GFZ', ['GE']),               # GEOFON
]

# Test window - recent 7 days
END_TIME = UTCDateTime.now() - 86400  # Yesterday
START_TIME = END_TIME - 7 * 86400      # 7 days ago

def scan_data_center(dc_name, networks):
    """Scan a data center for stations in the region."""
    print(f"\n{'='*60}")
    print(f"Scanning {dc_name} for networks: {networks}")
    print(f"{'='*60}")

    try:
        client = Client(dc_name, timeout=60)
    except Exception as e:
        print(f"  ERROR: Could not connect to {dc_name}: {e}")
        return []

    stations_found = []

    for network in networks:
        try:
            inventory = client.get_stations(
                network=network,
                minlatitude=MIN_LAT,
                maxlatitude=MAX_LAT,
                minlongitude=MIN_LON,
                maxlongitude=MAX_LON,
                level='station'
            )
        except Exception as e:
            print(f"  {network}: No stations found or error: {e}")
            continue

        for net in inventory:
            for sta in net:
                sta_info = {
                    'network': net.code,
                    'station': sta.code,
                    'lat': sta.latitude,
                    'lon': sta.longitude,
                    'name': sta.site.name if sta.site else '',
                    'data_center': dc_name,
                    'has_waveforms': False
                }

                # Test waveform availability
                for channel in ['BHZ', 'HHZ', 'HNZ', 'EHZ']:
                    try:
                        st = client.get_waveforms(
                            network=net.code,
                            station=sta.code,
                            location='*',
                            channel=channel,
                            starttime=START_TIME,
                            endtime=START_TIME + 3600  # 1 hour test
                        )
                        if len(st) > 0:
                            sta_info['has_waveforms'] = True
                            sta_info['channel'] = channel
                            sta_info['samples'] = st[0].stats.npts
                            break
                    except:
                        continue

                stations_found.append(sta_info)

                status = "[OK] WAVEFORMS" if sta_info['has_waveforms'] else "[--] no data"
                chan = sta_info.get('channel', '---')
                print(f"  {net.code}.{sta.code:<6} ({sta.latitude:.2f}, {sta.longitude:.2f}) "
                      f"{chan} {status} - {sta.site.name if sta.site else ''}")

    return stations_found


def main():
    print("=" * 70)
    print("Kahramanmaras Region Station Scanner")
    print("=" * 70)
    print(f"Bounds: Lat {MIN_LAT}-{MAX_LAT}, Lon {MIN_LON}-{MAX_LON}")
    print(f"Test window: {START_TIME.date} to {END_TIME.date}")

    all_stations = []

    for dc_name, networks in DATA_CENTERS:
        stations = scan_data_center(dc_name, networks)
        all_stations.extend(stations)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Stations with Waveform Data")
    print("=" * 70)

    working = [s for s in all_stations if s['has_waveforms']]

    # Deduplicate by network.station
    seen = set()
    unique_working = []
    for s in working:
        key = f"{s['network']}.{s['station']}"
        if key not in seen:
            seen.add(key)
            unique_working.append(s)

    if unique_working:
        # Sort by latitude (north to south)
        unique_working.sort(key=lambda x: -x['lat'])

        print(f"\nFound {len(unique_working)} stations with verified waveform access:\n")
        for s in unique_working:
            print(f"  SeismicStation(\"{s['network']}\", \"{s['station']}\", "
                  f"{s['lat']:.3f}, {s['lon']:.3f}, \"{s['name']}\"),")

        # Group by approximate segment
        print("\n" + "-" * 50)
        print("Suggested Segment Assignments:")
        print("-" * 50)

        north = [s for s in unique_working if s['lat'] >= 38.0]
        central = [s for s in unique_working if 37.0 <= s['lat'] < 38.0]
        south = [s for s in unique_working if s['lat'] < 37.0]

        print(f"\nNorth (>= 38.0°N): {len(north)} stations")
        for s in north:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")

        print(f"\nCentral (37-38°N): {len(central)} stations")
        for s in central:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")

        print(f"\nSouth (< 37°N): {len(south)} stations")
        for s in south:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")
    else:
        print("\nNo stations with verified waveform data found!")
        print("This may indicate:")
        print("  - Data access restrictions for the region")
        print("  - Stations offline due to 2023 earthquake damage")
        print("  - Need to try different time windows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
