#!/usr/bin/env python
"""Manually extract SAC from Hi-net WIN32 data."""
import os
import subprocess
import struct

print('Manual Hi-net SAC Extraction')
print('=' * 70)

os.chdir(os.path.join(os.path.dirname(__file__), 'hinet_extracted'))

# Parse channel table to get channel info
print('\n1. Parsing channel table...')
ch_file = '01_01_20260109.euc.ch'
channels = {}

with open(ch_file, 'r', encoding='euc-jp', errors='replace') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        if len(parts) >= 15:
            ch_num = parts[0]
            station = parts[3]
            component = parts[4]
            lat = float(parts[13])
            lon = float(parts[14])
            channels[ch_num] = {
                'station': station,
                'component': component,
                'lat': lat,
                'lon': lon
            }

print(f'   Found {len(channels)} channels')

# Find Kanto region stations (roughly 34.5-37N, 138.5-141.5E)
kanto_channels = {}
for ch_num, info in channels.items():
    if 34.5 <= info['lat'] <= 37.0 and 138.5 <= info['lon'] <= 141.5:
        kanto_channels[ch_num] = info

print(f'   Kanto region: {len(kanto_channels)} channels')

# Show some stations
print('\n   Sample Kanto stations:')
seen_stations = set()
for ch_num, info in sorted(kanto_channels.items()):
    if info['station'] not in seen_stations and len(seen_stations) < 10:
        print(f'     {info["station"]} ({info["lat"]:.2f}N, {info["lon"]:.2f}E) - ch {ch_num}')
        seen_stations.add(info['station'])

# Create win.prm file manually (required by win2sac_32)
print('\n2. Creating win.prm file...')
with open('win.prm', 'w') as f:
    for ch_num, info in channels.items():
        # Format: channel_num station component lat lon
        f.write(f'{ch_num} {info["station"]} {info["component"]} {info["lat"]} {info["lon"]}\n')
print('   Created win.prm')

# Try extracting one channel using win2sac_32
print('\n3. Extracting sample channel...')

# Pick a Kanto station
target_station = None
target_ch = None
for ch_num, info in kanto_channels.items():
    if info['component'] == 'U':  # Vertical component
        target_station = info['station']
        target_ch = ch_num
        break

if target_ch:
    print(f'   Target: {target_station} (ch {target_ch})')

    cnt_file = '2026010909000101VM.cnt'
    sac_file = f'{target_station}.U.sac'

    # Run win2sac_32 via WSL
    cmd = f'cd /mnt/c/GeoSpec/geospec_sprint/hinet_extracted && /mnt/c/GeoSpec/win32tools/win2sac.src/win2sac_32 {cnt_file} {target_ch} {sac_file}'

    result = subprocess.run(
        ['powershell.exe', '-Command', f"wsl bash -c '{cmd}'"],
        capture_output=True,
        text=True,
        timeout=60
    )

    print(f'   Return code: {result.returncode}')
    if result.stdout:
        print(f'   Output: {result.stdout[:200]}')
    if result.stderr:
        lines = result.stderr.strip().split('\n')
        for line in lines[:5]:
            print(f'   {line}')

    # Check if file was created
    if os.path.exists(sac_file):
        size = os.path.getsize(sac_file)
        print(f'\n   SUCCESS! Created {sac_file} ({size} bytes)')

        # Try reading with ObsPy
        try:
            from obspy import read
            st = read(sac_file)
            tr = st[0]
            print(f'   ObsPy read: {tr.stats.npts} samples @ {tr.stats.sampling_rate} Hz')
            print(f'   Duration: {tr.stats.endtime - tr.stats.starttime:.1f} seconds')
        except Exception as e:
            print(f'   ObsPy error: {e}')
    else:
        # Check for any SAC files
        sac_files = [f for f in os.listdir('.') if f.endswith('.sac') or f.endswith('.SAC')]
        print(f'   SAC files found: {sac_files[:10]}')

print('\n' + '=' * 70)
