"""
Scan for available seismic stations in California regions.
Tests waveform availability across SCEDC, NCEDC, and IRIS.

Regions:
- SoCal Mojave: 34.0-36.0N, 118.5-115.5W
- SoCal Coachella: 33.0-34.5N, 117.0-115.0W
- NorCal Hayward: 37.0-38.5N, 123.0-121.5W
"""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from datetime import datetime, timedelta
import sys

# Region definitions
REGIONS = {
    'socal_mojave': {
        'name': 'SoCal SAF Mojave',
        'bounds': (34.0, 36.0, -118.5, -115.5),  # minlat, maxlat, minlon, maxlon
        'data_centers': [('SCEDC', ['CI']), ('IRIS', ['IU', 'II'])],
    },
    'socal_coachella': {
        'name': 'SoCal SAF Coachella',
        'bounds': (33.0, 34.5, -117.0, -115.0),
        'data_centers': [('SCEDC', ['CI']), ('IRIS', ['IU', 'II'])],
    },
    'norcal_hayward': {
        'name': 'NorCal Hayward',
        'bounds': (37.0, 38.5, -123.0, -121.5),
        'data_centers': [('NCEDC', ['NC', 'BK']), ('IRIS', ['IU', 'II'])],
    },
}

# Test window - 7 days ago (to avoid same-day latency issues)
END_TIME = UTCDateTime.now() - 3 * 86400  # 3 days ago
START_TIME = END_TIME - 7 * 86400          # 10 days ago

def scan_region(region_key, region_config):
    """Scan a region for available stations."""
    name = region_config['name']
    minlat, maxlat, minlon, maxlon = region_config['bounds']

    print(f"\n{'='*70}")
    print(f"Scanning: {name}")
    print(f"{'='*70}")
    print(f"Bounds: {minlat}-{maxlat}N, {abs(maxlon)}-{abs(minlon)}W")
    print(f"Test window: {START_TIME.date} to {END_TIME.date}")

    all_stations = []

    for dc_name, networks in region_config['data_centers']:
        print(f"\n--- {dc_name} ---")

        try:
            client = Client(dc_name, timeout=60)
        except Exception as e:
            print(f"  ERROR: Could not connect to {dc_name}: {e}")
            continue

        for network in networks:
            try:
                inventory = client.get_stations(
                    network=network,
                    minlatitude=minlat,
                    maxlatitude=maxlat,
                    minlongitude=minlon,
                    maxlongitude=maxlon,
                    level='station'
                )
            except Exception as e:
                print(f"  {network}: No stations or error: {e}")
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
                        'has_waveforms': False,
                        'channel': None,
                        'sample_rate': None,
                    }

                    # Test waveform availability with multiple channels
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
                                sta_info['sample_rate'] = st[0].stats.sampling_rate
                                break
                        except:
                            continue

                    all_stations.append(sta_info)

                    status = "[OK]" if sta_info['has_waveforms'] else "[--]"
                    chan = sta_info.get('channel') or '---'
                    rate = f"{sta_info['sample_rate']:.0f}Hz" if sta_info['sample_rate'] else ''
                    print(f"  {net.code}.{sta.code:<6} ({sta.latitude:.2f}, {sta.longitude:.2f}) "
                          f"{chan} {rate:>5} {status} {sta.site.name if sta.site else ''}")

    return all_stations


def print_summary(region_key, stations):
    """Print summary with code snippets for fault_segments.py."""
    working = [s for s in stations if s['has_waveforms']]

    # Deduplicate
    seen = set()
    unique = []
    for s in working:
        key = f"{s['network']}.{s['station']}"
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(unique)} stations with verified waveforms")
    print(f"{'='*70}")

    if not unique:
        print("No stations with verified waveform data found!")
        return

    # Sort by latitude (north to south)
    unique.sort(key=lambda x: -x['lat'])

    print("\nCopy-paste for fault_segments.py:\n")
    for s in unique:
        print(f'    SeismicStation("{s["network"]}", "{s["station"]}", '
              f'{s["lat"]:.3f}, {s["lon"]:.3f}, "{s["name"]}"),')

    # Group by latitude bands for segment assignment
    if region_key == 'socal_mojave':
        north = [s for s in unique if s['lat'] >= 35.0]
        central = [s for s in unique if 34.5 <= s['lat'] < 35.0]
        south = [s for s in unique if s['lat'] < 34.5]

        print(f"\n--- Suggested Segment Assignments ---")
        print(f"mojave_north (>=35.0N): {len(north)} stations")
        for s in north:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")
        print(f"mojave_central (34.5-35.0N): {len(central)} stations")
        for s in central:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")
        print(f"mojave_south (<34.5N): {len(south)} stations")
        for s in south:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")

    elif region_key == 'norcal_hayward':
        north = [s for s in unique if s['lat'] >= 38.0]
        hayward = [s for s in unique if 37.5 <= s['lat'] < 38.0]
        south = [s for s in unique if s['lat'] < 37.5]

        print(f"\n--- Suggested Segment Assignments ---")
        print(f"rodgers_creek (>=38.0N): {len(north)} stations")
        for s in north:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")
        print(f"hayward (37.5-38.0N): {len(hayward)} stations")
        for s in hayward:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")
        print(f"calaveras (<37.5N): {len(south)} stations")
        for s in south:
            print(f"    {s['network']}.{s['station']} @ {s['lat']:.2f}N")


def main():
    print("="*70)
    print("California Seismic Station Scanner")
    print("="*70)
    print(f"Testing waveform availability from {START_TIME.date} to {END_TIME.date}")
    print("(Using 3-10 day old data to avoid same-day latency issues)")

    # Scan all regions
    for region_key, region_config in REGIONS.items():
        stations = scan_region(region_key, region_config)
        print_summary(region_key, stations)

    return 0


if __name__ == "__main__":
    sys.exit(main())
