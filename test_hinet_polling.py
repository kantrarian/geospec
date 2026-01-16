#!/usr/bin/env python
"""Test the updated hinet_data.py with status page polling."""
import os
import sys

# Add monitoring/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'monitoring', 'src'))

# Set credentials
os.environ['HINET_USER'] = 'devilldog'
os.environ['HINET_PASSWORD'] = 'Geospec20261'

from datetime import datetime, timedelta
from hinet_data import HinetClient

print('Testing Hi-net Data Fetcher with Status Page Polling')
print('=' * 70)

try:
    fetcher = HinetClient()
    print(f'Initialized HinetClient')

    # Test connection (this will authenticate)
    print('\nTesting connection...')
    if fetcher.test_connection():
        print('Authentication successful!')

        print('\nFetching 3 minutes of Hi-net data from Kanto region...')
        # Use a time 3 days ago (data should be available)
        start_time = datetime.utcnow() - timedelta(days=3)
        start_time = start_time.replace(hour=9, minute=0, second=0, microsecond=0)

        print(f'Start time: {start_time}')

        stream = fetcher.fetch_continuous(
            network='0101',  # Kanto
            start_time=start_time,
            duration_minutes=3
        )

        if stream and len(stream) > 0:
            print(f'\nSUCCESS! Retrieved {len(stream)} traces')
            for i, tr in enumerate(stream[:5]):
                print(f'  {i+1}. {tr.id}: {tr.stats.npts} samples @ {tr.stats.sampling_rate} Hz')
            if len(stream) > 5:
                print(f'  ... and {len(stream)-5} more traces')
        else:
            print('\nNo data returned')
    else:
        print('Authentication failed')

except Exception as e:
    print(f'\nError: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 70)
