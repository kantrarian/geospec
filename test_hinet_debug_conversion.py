#!/usr/bin/env python
"""Debug Hi-net SAC conversion."""
import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'monitoring', 'src'))

# Set credentials
os.environ['HINET_USER'] = 'devilldog'
os.environ['HINET_PASSWORD'] = 'Geospec20261'

from datetime import datetime, timedelta
from hinet_data import HinetClient, KANTO_BOUNDS

print('Debug Hi-net SAC Conversion')
print('=' * 70)

# Create a persistent output directory
output_dir = Path('C:/GeoSpec/geospec_sprint/hinet_test_output')
output_dir.mkdir(parents=True, exist_ok=True)

try:
    fetcher = HinetClient()
    print('Initialized HinetClient')

    # Test connection (this will authenticate)
    print('\nAuthenticating...')
    if not fetcher.test_connection():
        print('Authentication failed')
        sys.exit(1)
    print('Authentication successful!')

    # Submit request
    start_time = datetime.utcnow() - timedelta(days=3)
    start_time = start_time.replace(hour=9, minute=0, second=0, microsecond=0)
    print(f'\nSubmitting request for: {start_time}')

    if not fetcher._submit_request('0101', start_time, 3):
        print('Request submission failed')
        sys.exit(1)
    print('Request submitted!')

    # Poll for available
    print('\nPolling for available download...')
    request_id = fetcher._poll_for_available()
    if not request_id:
        print('No download available')
        sys.exit(1)
    print(f'Found download ID: {request_id}')

    # Download
    zip_path = output_dir / f'hinet_{request_id}.zip'
    print(f'\nDownloading to: {zip_path}')
    if not fetcher._download_by_id(request_id, zip_path):
        print('Download failed')
        sys.exit(1)
    print(f'Downloaded: {zip_path.stat().st_size / 1e6:.2f} MB')

    # Extract
    extract_dir = output_dir / 'win32'
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nExtracting to: {extract_dir}')

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
        files = zf.namelist()
    print(f'Extracted files: {files}')

    # Find channel table
    ch_file = None
    cnt_files = []
    for f in files:
        if f.endswith('.euc.ch'):
            ch_file = f
        elif f.endswith('.cnt'):
            cnt_files.append(f)
    print(f'\nChannel file: {ch_file}')
    print(f'Data files: {cnt_files}')

    # Parse channel table
    ch_path = extract_dir / ch_file
    print(f'\nParsing channel table: {ch_path}')

    channels = {}
    with open(ch_path, 'r', encoding='euc-jp', errors='replace') as f:
        for line_num, line in enumerate(f):
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

                # Filter to Kanto region + vertical component
                if (KANTO_BOUNDS['min_lat'] <= lat <= KANTO_BOUNDS['max_lat'] and
                    KANTO_BOUNDS['min_lon'] <= lon <= KANTO_BOUNDS['max_lon'] and
                    component == 'U'):
                    channels[ch_num] = {
                        'station': station,
                        'component': component,
                        'lat': lat,
                        'lon': lon
                    }

    print(f'Found {len(channels)} Kanto vertical channels')
    if channels:
        print('Sample channels:')
        for ch_num, info in list(channels.items())[:5]:
            print(f'  {ch_num}: {info["station"]} ({info["lat"]:.2f}N, {info["lon"]:.2f}E)')

    # Create win.prm file
    prm_path = extract_dir / 'win.prm'
    print(f'\nCreating win.prm: {prm_path}')
    with open(prm_path, 'w') as f:
        f.write(f".\n{ch_file}\n.\n.")
    print('win.prm content:')
    print(open(prm_path).read())

    # Try converting one channel
    if channels and cnt_files:
        ch_num, info = list(channels.items())[0]
        cnt_file = cnt_files[0]
        station = info['station']

        sac_dir = output_dir / 'sac'
        sac_dir.mkdir(parents=True, exist_ok=True)
        sac_name = f'{station}.U.SAC'
        sac_path = sac_dir / sac_name

        print(f'\nConverting channel {ch_num} ({station})...')

        # Convert paths for WSL
        wsl_work_dir = str(extract_dir).replace('C:', '/mnt/c').replace('\\', '/')
        wsl_sac_dir = str(sac_dir).replace('C:', '/mnt/c').replace('\\', '/')

        # Build command - win2sac_32 creates file in current directory
        cmd = f"cd {wsl_work_dir} && /mnt/c/GeoSpec/win32tools/win2sac.src/win2sac_32 {cnt_file} {ch_num} {sac_name}"

        print(f'Command: {cmd}')

        result = subprocess.run(
            ['wsl', 'bash', '-c', cmd],
            capture_output=True,
            text=True,
            timeout=60
        )

        print(f'Return code: {result.returncode}')
        if result.stdout:
            print(f'stdout: {result.stdout[:500]}')
        if result.stderr:
            print(f'stderr: {result.stderr[:500]}')

        # Check for output in work dir
        expected_in_workdir = extract_dir / sac_name
        print(f'\nChecking for SAC file in work dir: {expected_in_workdir}')
        if expected_in_workdir.exists():
            print(f'Found! Size: {expected_in_workdir.stat().st_size} bytes')

            # Copy to sac_dir
            import shutil
            shutil.copy(expected_in_workdir, sac_path)
            print(f'Copied to: {sac_path}')

            # Read with ObsPy
            try:
                from obspy import read
                st = read(str(sac_path))
                tr = st[0]
                print(f'\nObsPy read SUCCESS:')
                print(f'  Station: {tr.stats.station}')
                print(f'  Samples: {tr.stats.npts}')
                print(f'  Rate: {tr.stats.sampling_rate} Hz')
            except Exception as e:
                print(f'ObsPy error: {e}')
        else:
            print('SAC file not found in work dir')
            # List what's in the directory
            print(f'\nFiles in {extract_dir}:')
            for f in os.listdir(extract_dir):
                print(f'  {f}')

except Exception as e:
    print(f'\nError: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 70)
