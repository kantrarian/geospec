#!/usr/bin/env python3
"""
Quick scan of Campi Flegrei IV network availability.
"""
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Campi Flegrei bounding box (tighter than user spec to focus on caldera)
MINLAT, MAXLAT = 40.80, 40.88
MINLON, MAXLON = 14.05, 14.20

print("Scanning Campi Flegrei (IV Network)")
print("=" * 50)
print(f"Bounding box: {MINLAT}-{MAXLAT}N, {MINLON}-{MAXLON}E")
print()

for dc in ['INGV', 'ODC', 'IRIS']:
    try:
        print(f"Trying {dc}...")
        client = Client(dc)
        inv = client.get_stations(
            network='IV',
            minlatitude=MINLAT, maxlatitude=MAXLAT,
            minlongitude=MINLON, maxlongitude=MAXLON,
            channel='HH?,BH?',
            level='station'
        )

        print(f"Data center: {dc}")
        stations = []
        for net in inv:
            for sta in net:
                stations.append(sta)
                name = sta.site.name if sta.site else 'Unknown'
                print(f"  IV.{sta.code}: ({sta.latitude:.4f}, {sta.longitude:.4f}) {name}")

        print(f"\nTotal: {len(stations)} stations")
        print()
        print("Testing waveform availability (last hour)...")

        end = UTCDateTime.now()
        start = end - 3600

        available = 0
        online_stations = []

        for sta in stations[:10]:
            try:
                st = client.get_waveforms('IV', sta.code, '*', 'HHZ,BHZ', start, end)
                if len(st) > 0:
                    print(f"  IV.{sta.code}: ONLINE ({len(st)} traces)")
                    available += 1
                    online_stations.append(sta)
                else:
                    print(f"  IV.{sta.code}: No data")
            except Exception as e:
                print(f"  IV.{sta.code}: Error - {str(e)[:30]}")

        print()
        print(f"Result: {available}/{min(10, len(stations))} tested stations ONLINE")

        if available >= 3:
            print("FC READY: YES")
            print()
            print("Recommended stations for fault_segments.py:")
            for sta in online_stations[:5]:
                print(f'    SeismicStation("IV", "{sta.code}", {sta.latitude:.3f}, {sta.longitude:.3f}, "{sta.site.name if sta.site else sta.code}"),')
        else:
            print("FC READY: NO (insufficient stations)")

        break

    except Exception as e:
        print(f"{dc}: {e}")
        continue
