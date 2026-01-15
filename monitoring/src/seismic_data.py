"""
seismic_data.py
Seismic waveform data acquisition for fault correlation analysis.

Fetches continuous seismic data from FDSN web services (IRIS, SCEDC, NCEDC)
and processes for correlation analysis.

Processing Pipeline:
1. Fetch waveforms for segment stations
2. Merge, detrend, bandpass filter (0.01-1.0 Hz)
3. Decimate to 1 Hz
4. Compute envelope via Hilbert transform

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.signal.filter import bandpass
from scipy.signal import hilbert, decimate
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import logging
import json

from fault_segments import FaultSegment, SeismicStation, get_segments_for_region

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FDSN Client Configuration
FDSN_CLIENTS = {
    'IRIS': {
        'url': 'IRIS',
        'networks': ['IU', 'II', 'US', 'UW', 'CN', 'PB', 'GE'],  # Added GE (GEOFON)
        'timeout': 120,
    },
    'SCEDC': {
        'url': 'SCEDC',
        'networks': ['CI', 'AZ'],
        'timeout': 120,
        'notes': 'Rate limits apply - use caching'
    },
    'NCEDC': {
        'url': 'NCEDC',
        'networks': ['NC', 'BK'],
        'timeout': 120,
    },
    # Turkey - Kandilli Observatory
    'KOERI': {
        'url': 'KOERI',
        'networks': ['KO', 'TU'],
        'timeout': 120,
        'notes': 'Turkish networks - waveform access verified Jan 2026'
    },
    # Italy - INGV (Campi Flegrei, Vesuvius, etc.)
    'INGV': {
        'url': 'INGV',
        'networks': ['IV'],
        'timeout': 120,
        'notes': 'Italian volcanic monitoring - open data'
    },
    # European fallback (ORFEUS federation)
    'ODC': {
        'url': 'ODC',
        'networks': ['HL', 'HT'],  # Greek networks
        'timeout': 120,
        'notes': 'ORFEUS Data Center - European federation'
    },
}

# Default processing parameters
DEFAULT_FILTER = {
    'freqmin': 0.01,  # 0.01 Hz = 100 second period
    'freqmax': 1.0,   # 1.0 Hz = 1 second period
    'corners': 4,
    'zerophase': True,
}

DEFAULT_DECIMATE_FACTOR = 40  # 40 Hz -> 1 Hz


class SeismicDataFetcher:
    """
    Fetches and processes seismic waveform data for fault correlation analysis.

    Attributes:
        cache_dir: Directory for caching downloaded waveforms
        cache_ttl_days: Cache time-to-live in days
        clients: Dictionary of FDSN client connections
    """

    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_days: int = 7):
        """
        Initialize the SeismicDataFetcher.

        Args:
            cache_dir: Directory for caching data. Defaults to monitoring/data/seismic_cache
            cache_ttl_days: Number of days to keep cached data
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / 'data' / 'seismic_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_days = cache_ttl_days
        self.clients: Dict[str, Client] = {}

        logger.info(f"SeismicDataFetcher initialized. Cache: {self.cache_dir}")

    def _get_client(self, network: str) -> Optional[Client]:
        """Get or create FDSN client for a network."""
        # Find which service hosts this network
        for service_name, config in FDSN_CLIENTS.items():
            if network in config['networks']:
                if service_name not in self.clients:
                    try:
                        self.clients[service_name] = Client(
                            config['url'],
                            timeout=config['timeout']
                        )
                        logger.info(f"Connected to {service_name}")
                    except Exception as e:
                        logger.error(f"Could not connect to {service_name}: {e}")
                        return None
                return self.clients[service_name]

        # Fallback to IRIS
        if 'IRIS' not in self.clients:
            try:
                self.clients['IRIS'] = Client('IRIS', timeout=120)
            except Exception as e:
                logger.error(f"Could not connect to IRIS: {e}")
                return None
        return self.clients['IRIS']

    def _cache_path(self, region: str, date: datetime, data_type: str) -> Path:
        """Generate cache file path."""
        date_str = date.strftime('%Y%m%d')
        return self.cache_dir / region / date_str / f'{data_type}.pkl'

    def _load_cache(self, path: Path) -> Optional[object]:
        """Load data from cache if valid."""
        if not path.exists():
            return None

        # Check TTL
        file_age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if file_age.days > self.cache_ttl_days:
            logger.debug(f"Cache expired: {path}")
            return None

        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def _save_cache(self, path: Path, data: object):
        """Save data to cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached: {path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def fetch_station_waveforms(
        self,
        station: SeismicStation,
        start: datetime,
        end: datetime,
        channels: List[str] = ['BHZ', 'HHZ', 'HNZ']
    ) -> Optional[Stream]:
        """
        Fetch waveforms for a single station.

        Args:
            station: SeismicStation object
            start: Start datetime
            end: End datetime
            channels: List of channel codes to try (in order of preference)

        Returns:
            ObsPy Stream object or None if fetch fails
        """
        client = self._get_client(station.network)
        if client is None:
            return None

        start_utc = UTCDateTime(start)
        end_utc = UTCDateTime(end)

        # Try each channel in order
        for channel in channels:
            try:
                st = client.get_waveforms(
                    network=station.network,
                    station=station.code,
                    location='*',
                    channel=channel,
                    starttime=start_utc,
                    endtime=end_utc
                )
                if len(st) > 0:
                    logger.debug(f"Retrieved {station.nslc}.{channel}: {len(st)} traces")
                    return st
            except Exception as e:
                logger.debug(f"No data for {station.nslc}.{channel}: {e}")
                continue

        logger.warning(f"No data available for {station.nslc}")
        return None

    def fetch_segment_waveforms(
        self,
        segment: FaultSegment,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> Dict[str, Stream]:
        """
        Fetch waveforms for all stations in a fault segment.

        Args:
            segment: FaultSegment object
            start: Start datetime
            end: End datetime
            use_cache: Whether to use/update cache

        Returns:
            Dictionary mapping station NSLC -> Stream
        """
        # Check cache
        cache_key = f"{segment.name}_waveforms"
        cache_path = self._cache_path(segment.region, start, cache_key)

        if use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                logger.info(f"Loaded {segment.name} waveforms from cache")
                return cached

        # Fetch from service
        logger.info(f"Fetching waveforms for {segment.name} ({len(segment.stations)} stations)")
        waveforms = {}

        for station in segment.stations:
            st = self.fetch_station_waveforms(station, start, end)
            if st is not None:
                waveforms[station.nslc] = st

        logger.info(f"Retrieved data for {len(waveforms)}/{len(segment.stations)} stations")

        # Cache results
        if use_cache and waveforms:
            self._save_cache(cache_path, waveforms)

        return waveforms

    def process_waveforms(
        self,
        stream: Stream,
        filter_params: Optional[Dict] = None,
        decimate_factor: int = DEFAULT_DECIMATE_FACTOR
    ) -> Optional[np.ndarray]:
        """
        Process a waveform stream for correlation analysis.

        Steps:
        1. Merge traces
        2. Detrend
        3. Bandpass filter
        4. Decimate

        Args:
            stream: ObsPy Stream object
            filter_params: Dictionary with freqmin, freqmax, corners, zerophase
            decimate_factor: Factor to decimate by (default 40: 40Hz -> 1Hz)

        Returns:
            Processed data array or None if processing fails
        """
        if stream is None or len(stream) == 0:
            return None

        filter_params = filter_params or DEFAULT_FILTER

        try:
            # Make a copy to avoid modifying original
            st = stream.copy()

            # Merge traces (handles gaps)
            st.merge(method=1, fill_value='interpolate')

            # Detrend
            st.detrend('demean')
            st.detrend('linear')

            # Get sample rate for filtering
            if len(st) == 0:
                return None

            sample_rate = st[0].stats.sampling_rate

            # Bandpass filter
            st.filter(
                'bandpass',
                freqmin=filter_params['freqmin'],
                freqmax=filter_params['freqmax'],
                corners=filter_params['corners'],
                zerophase=filter_params['zerophase']
            )

            # Decimate if needed
            if sample_rate > 1.5:  # Only if more than ~1 Hz
                factor = int(sample_rate)  # Decimate to ~1 Hz
                if factor > 1:
                    st.decimate(factor=min(factor, 10))  # Max 10x per step
                    if st[0].stats.sampling_rate > 1.5:
                        st.decimate(factor=int(st[0].stats.sampling_rate))

            return st[0].data

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None

    def compute_envelope(self, data: np.ndarray) -> np.ndarray:
        """
        Compute signal envelope using Hilbert transform.

        The envelope represents the instantaneous amplitude, useful for
        correlation analysis where we care about activity level rather
        than phase.

        Args:
            data: 1D array of seismic data

        Returns:
            Envelope (absolute value of analytic signal)
        """
        analytic = hilbert(data)
        envelope = np.abs(analytic)
        return envelope

    def get_segment_envelopes(
        self,
        segment: FaultSegment,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get processed envelopes for all stations in a segment.

        Args:
            segment: FaultSegment object
            start: Start datetime
            end: End datetime
            use_cache: Whether to use caching

        Returns:
            Dictionary mapping station NSLC -> envelope array
        """
        # Check envelope cache
        cache_key = f"{segment.name}_envelopes"
        cache_path = self._cache_path(segment.region, start, cache_key)

        if use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                logger.info(f"Loaded {segment.name} envelopes from cache")
                return cached

        # Fetch and process waveforms
        waveforms = self.fetch_segment_waveforms(segment, start, end, use_cache)

        envelopes = {}
        for nslc, stream in waveforms.items():
            processed = self.process_waveforms(stream)
            if processed is not None:
                envelope = self.compute_envelope(processed)
                envelopes[nslc] = envelope

        logger.info(f"Computed envelopes for {len(envelopes)} stations")

        # Cache
        if use_cache and envelopes:
            self._save_cache(cache_path, envelopes)

        return envelopes

    def get_region_activity(
        self,
        region: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get envelopes for all segments in a region.

        Args:
            region: Region name (e.g., 'ridgecrest')
            start: Start datetime
            end: End datetime

        Returns:
            Nested dictionary: segment_name -> station_nslc -> envelope
        """
        segments = get_segments_for_region(region)
        activity = {}

        for segment in segments:
            envelopes = self.get_segment_envelopes(segment, start, end)
            if envelopes:
                activity[segment.name] = envelopes

        return activity


def compute_segment_activity_index(envelopes: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute aggregate activity index for a segment from station envelopes.

    Uses the median envelope across stations to be robust to individual
    station noise.

    Args:
        envelopes: Dictionary of station NSLC -> envelope array

    Returns:
        Aggregate activity time series
    """
    if not envelopes:
        return np.array([])

    # Find minimum length (in case of gaps)
    min_len = min(len(env) for env in envelopes.values())

    # Stack envelopes
    stacked = np.array([env[:min_len] for env in envelopes.values()])

    # Median across stations
    activity = np.median(stacked, axis=0)

    return activity


if __name__ == "__main__":
    # Test the seismic data fetcher
    print("=" * 60)
    print("Seismic Data Fetcher Test")
    print("=" * 60)

    fetcher = SeismicDataFetcher()

    # Test with IU.ANMO (Global Network - definitely available on IRIS)
    from fault_segments import SeismicStation

    test_station = SeismicStation("IU", "ANMO", 34.946, -106.457, "Albuquerque")

    start = datetime(2019, 7, 4, 0, 0)
    end = datetime(2019, 7, 4, 1, 0)

    print(f"\nFetching {test_station.nslc} from {start} to {end}")

    st = fetcher.fetch_station_waveforms(
        test_station, start, end,
        channels=['BHZ']  # Use BHZ for IU network
    )

    if st:
        print(f"SUCCESS: Retrieved {len(st)} trace(s)")
        print(f"  Samples: {st[0].stats.npts}")
        print(f"  Sample rate: {st[0].stats.sampling_rate} Hz")

        # Process
        processed = fetcher.process_waveforms(st)
        if processed is not None:
            print(f"  Processed length: {len(processed)}")

            envelope = fetcher.compute_envelope(processed)
            print(f"  Envelope mean: {envelope.mean():.4f}")
            print(f"  Envelope max: {envelope.max():.4f}")

            print("\nSUCCESS: Seismic data pipeline working!")
    else:
        print("FAILED: Could not retrieve waveforms")
