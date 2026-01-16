#!/usr/bin/env python3
"""
seismic_waveform_fetcher.py
Robust seismic waveform acquisition with multi-network fallback.

Handles fetching of waveform data for Fault Correlation and Seismic THD methods,
automatically trying primary regional networks (SCEDC, NCEDC) and falling back
to global aggregators (IRIS) if specific stations are unavailable.
"""

import time
import logging
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta

try:
    from obspy import UTCDateTime, Stream
    from obspy.clients.fdsn import Client
except ImportError:
    raise ImportError("ObsPy is required for seismic_waveform_fetcher.py")

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
TIMEOUT = 30  # seconds

# Network Priority Map
# Defines which FDSN client to try first for a given network code
NETWORK_PRIORITY = {
    'CI': ['SCEDC', 'IRIS'],      # Southern California
    'NC': ['NCEDC', 'IRIS'],      # Northern California
    'BK': ['NCEDC', 'IRIS'],      # Berkeley
    'IU': ['IRIS', 'SCEDC'],      # Global Seismographic Network
    'US': ['IRIS'],               # US National Network
    'II': ['IRIS'],               # Global Seismographic Network (IDA)
    'JP': ['IRIS'],               # Japan (F-net often accessible via IRIS)
    'TU': ['IRIS', 'KOERI'],      # Turkey (KOERI check needed)
    'GE': ['IRIS', 'GFZ'],        # GEOFON
    'NZ': ['GEONET', 'IRIS'],     # New Zealand (GeoNet)
    'AK': ['IRIS'],               # Alaska (AEC)
    'TW': ['IRIS'],               # Taiwan (BATS)
}

DEFAULT_CLIENTS = ['IRIS', 'GEONET']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeismicWaveformFetcher:
    """
    Robust fetcher for seismic waveforms.
    """
    
    def __init__(self):
        self.clients = {}  # Cache initialized clients
        
    def _get_client(self, service_name: str) -> Optional[Client]:
        """Get or initialize an FDSN client."""
        if service_name in self.clients:
            return self.clients[service_name]
            
        try:
            client = Client(service_name, timeout=TIMEOUT)
            self.clients[service_name] = client
            return client
        except Exception as e:
            logger.warning(f"Could not initialize client '{service_name}': {e}")
            return None

    def get_waveforms(self, network: str, station: str, location: str, channel: str,
                      start_time: Union[datetime, UTCDateTime],
                      end_time: Union[datetime, UTCDateTime]) -> Optional[Stream]:
        """
        Fetch waveforms with fallback logic.
        
        Args:
            network: Network code (e.g., 'CI')
            station: Station code (e.g., 'WBS')
            location: Location code (e.g., '--' or '00')
            channel: Channel code (e.g., 'BHZ')
            start_time: Start time
            end_time: End time
            
        Returns:
            ObsPy Stream object or None if failed
        """
        # Determine client priority list
        services = NETWORK_PRIORITY.get(network, DEFAULT_CLIENTS)
        
        for service in services:
            client = self._get_client(service)
            if not client:
                continue
                
            stream = self._fetch_with_retry(client, network, station, location, channel, start_time, end_time)
            
            if stream and len(stream) > 0:
                logger.info(f"Successfully fetched {network}.{station} from {service}")
                return stream
            else:
                logger.debug(f"Failed to fetch {network}.{station} from {service}, trying next...")
                
        logger.error(f"Failed to fetch waveforms for {network}.{station} from all sources.")
        return None

    def _fetch_with_retry(self, client: Client, net, sta, loc, cha, t1, t2) -> Optional[Stream]:
        """Helper to fetch from a specific client with retries."""
        attempts = 0
        while attempts < MAX_RETRIES:
            try:
                st = client.get_waveforms(net, sta, loc, cha, UTCDateTime(t1), UTCDateTime(t2))
                return st
            except Exception as e:
                attempts += 1
                if "No data available" in str(e):
                    # Don't retry if data is simply missing from this datacenter
                    return None
                
                logger.warning(f"Fetch attempt {attempts}/{MAX_RETRIES} failed: {e}")
                time.sleep(RETRY_DELAY)
                
        return None

    def verify_station_availability(self, network: str, station: str) -> bool:
        """
        Quickly check if a station is available in any configured datacenter.
        Useful for health checks.
        """
        t_end = UTCDateTime.now() - 3600 # 1 hour ago
        t_start = t_end - 600 # 10 minutes duration
        
        st = self.get_waveforms(network, station, "*", "BHZ", t_start, t_end)
        return st is not None and len(st) > 0

if __name__ == "__main__":
    # Smoke test
    print("Testing SeismicWaveformFetcher...")
    fetcher = SeismicWaveformFetcher()
    
    # Test 1: SCEDC Station (CI.WBS)
    print("\nTest 1: Fetching CI.WBS (SCEDC)...")
    t = UTCDateTime.now() - 3600
    st = fetcher.get_waveforms("CI", "WBS", "*", "BHZ", t, t+10)
    if st:
        print(st)
    
    # Test 2: Global Station (IU.ANMO) - Should fallback to IRIS if SCEDC tried first (unlikely for IU but good test)
    print("\nTest 2: Fetching IU.ANMO...")
    st = fetcher.get_waveforms("IU", "ANMO", "00", "BHZ", t, t+10)
    if st:
        print(st)

    # Test 3: Fallback check (Simulated by asking for NC station from SCEDC priority?)
    # NC is set to [NCEDC, IRIS].
    print("\nTest 3: Fetching NC.JCC (NorCal)...")
    st = fetcher.get_waveforms("NC", "JCC", "*", "BHZ", t, t+10)
    if st:
        print(st)
