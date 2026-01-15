"""
hinet_data.py - Hi-net Data Acquisition for Japan Coverage

Provides access to NIED Hi-net seismic data for the Tokyo/Kanto region.
Uses status page polling method to work around HinetPy timing issues.

Prerequisites:
    - NIED Hi-net account (approved)
    - HinetPy library installed: pip install HinetPy
    - win32tools installed via WSL (for format conversion)

Usage:
    from hinet_data import HinetClient
    client = HinetClient(username='devilldog', password='YOUR_PASSWORD')
    stream = client.fetch_continuous('0101', datetime(2026, 1, 10), 5)

Author: R.J. Mathews / Claude
Date: January 2026
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Hi-net Network Codes
HINET_NETWORKS = {
    '0101': 'Hi-net',           # NIED Hi-net
    '0103': 'F-net',            # NIED F-net (broadband)
    '0201': 'Kanto-Tokai',      # JMA Kanto-Tokai
}

# Kanto region bounding box
KANTO_BOUNDS = {
    'min_lat': 34.5,
    'max_lat': 37.0,
    'min_lon': 138.5,
    'max_lon': 141.5,
}

# Hi-net request constraints
MAX_MINUTES_PER_REQUEST = 5      # Hi-net limit: 5 minutes per request
STATUS_POLL_INTERVAL = 5         # Seconds between status checks
STATUS_POLL_TIMEOUT = 120        # Max seconds to wait for data preparation
DOWNLOAD_TIMEOUT = 120           # Seconds for download

# WIN32 tools path (WSL)
WIN2SAC_PATH = '/mnt/c/GeoSpec/win32tools/win2sac.src/win2sac_32'


# =============================================================================
# HI-NET CLIENT
# =============================================================================

class HinetClient:
    """
    Client for fetching data from NIED Hi-net.

    Uses status page polling for reliable data download, with automatic
    WIN32 to SAC/ObsPy conversion.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the Hi-net client.

        Args:
            username: Hi-net username (or from HINET_USER env var)
            password: Hi-net password (or from HINET_PASSWORD env var)
            cache_dir: Directory for caching downloaded data
        """
        self.username = username or os.environ.get('HINET_USER')
        self.password = password or os.environ.get('HINET_PASSWORD')

        if not self.username or not self.password:
            logger.warning("Hi-net credentials not configured. "
                          "Set HINET_USER and HINET_PASSWORD environment variables.")

        self.cache_dir = cache_dir or Path(__file__).parent.parent / 'data' / 'hinet_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._session = None
        self._authenticated = False

        logger.info(f"HinetClient initialized. Cache: {self.cache_dir}")

    def _get_client(self):
        """Get or create authenticated HinetPy client."""
        if self._client is not None and self._authenticated:
            return self._client

        try:
            from HinetPy import Client
            self._client = Client(self.username, self.password)
            self._session = self._client.session
            self._authenticated = True
            logger.info("Hi-net authentication successful")
            return self._client
        except ImportError:
            logger.error("HinetPy not installed. Run: pip install HinetPy")
            raise
        except Exception as e:
            logger.error(f"Hi-net authentication failed: {e}")
            raise

    def test_connection(self) -> bool:
        """Test Hi-net connection and authentication."""
        try:
            client = self._get_client()
            logger.info("Hi-net connection test successful")
            return True
        except Exception as e:
            logger.error(f"Hi-net connection test failed: {e}")
            return False

    def _submit_request(
        self,
        network: str,
        start_time: datetime,
        duration_minutes: int,
    ) -> bool:
        """
        Submit a data request to NIED.

        Args:
            network: Network code (e.g., '0101' for Hi-net)
            start_time: Start time for data
            duration_minutes: Duration in minutes

        Returns:
            True if request submitted successfully
        """
        client = self._get_client()

        # Build request URL
        request_url = 'https://hinetwww11.bosai.go.jp/auth/download/cont/cont_request.php'
        params = {
            'org1': network[:2],
            'org2': network[2:],
            'volc': '0',
            'year': start_time.strftime('%Y'),
            'month': start_time.strftime('%m'),
            'day': start_time.strftime('%d'),
            'hour': start_time.strftime('%H'),
            'min': start_time.strftime('%M'),
            'span': str(duration_minutes),
            'arc': 'ZIP',
            'size': '93680',
            'LANG': 'en',
            'rn': str(int(datetime.now().timestamp()))
        }

        try:
            resp = self._session.post(request_url, params=params, timeout=30)
            logger.debug(f"Request submitted: {resp.status_code}")
            return resp.status_code in (200, 301, 302)
        except Exception as e:
            logger.error(f"Failed to submit request: {e}")
            return False

    def _poll_for_available(self, timeout: int = STATUS_POLL_TIMEOUT) -> Optional[str]:
        """
        Poll status page for available download.

        Returns:
            Request ID if available, None if timeout or error
        """
        client = self._get_client()
        status_url = 'https://hinetwww11.bosai.go.jp/auth/download/cont/cont_status.php'

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = self._session.get(status_url, params={'LANG': 'en'}, timeout=30)

                # Find available downloads with Download buttons
                # Pattern: openDownload('ID') with "Available" status
                available = re.findall(r"openDownload\('(\d+)'\)", resp.text)

                if available:
                    # Return the most recent (first) available ID
                    logger.info(f"Found available download: {available[0]}")
                    return available[0]

                # Check if still preparing
                if 'Preparing' in resp.text:
                    logger.debug("Data still preparing...")
                elif 'Failed' in resp.text and 'Preparing' not in resp.text:
                    # All requests failed, no preparing ones
                    logger.warning("Data preparation failed")
                    return None

            except Exception as e:
                logger.warning(f"Status poll error: {e}")

            time.sleep(STATUS_POLL_INTERVAL)

        logger.warning("Timeout waiting for data preparation")
        return None

    def _download_by_id(self, request_id: str, output_path: Path) -> bool:
        """
        Download data by request ID.

        Args:
            request_id: NIED request ID
            output_path: Path to save ZIP file

        Returns:
            True if download successful
        """
        client = self._get_client()
        download_url = f'https://hinetwww11.bosai.go.jp/auth/download/cont/cont_download.php'

        try:
            resp = self._session.get(
                download_url,
                params={'id': request_id, 'LANG': 'en'},
                timeout=DOWNLOAD_TIMEOUT
            )

            if resp.status_code == 200 and len(resp.content) > 1000:
                # Check if it's a valid ZIP
                if resp.content[:4] == b'PK\x03\x04':
                    with open(output_path, 'wb') as f:
                        f.write(resp.content)
                    logger.info(f"Downloaded {len(resp.content)} bytes to {output_path}")
                    return True
                else:
                    logger.warning(f"Response is not a ZIP file: {resp.content[:100]}")
            else:
                logger.warning(f"Download failed: {resp.status_code}, {len(resp.content)} bytes")

        except Exception as e:
            logger.error(f"Download error: {e}")

        return False

    def _extract_and_convert(
        self,
        zip_path: Path,
        output_dir: Path,
        station_filter: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Extract ZIP and convert WIN32 to SAC format.

        Args:
            zip_path: Path to downloaded ZIP file
            output_dir: Directory for SAC output
            station_filter: Optional list of station codes to extract

        Returns:
            List of SAC file paths
        """
        sac_files = []

        try:
            # Extract ZIP
            extract_dir = output_dir / 'win32'
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
                files = zf.namelist()

            # Find channel table and data files
            ch_file = None
            cnt_files = []
            for f in files:
                if f.endswith('.euc.ch'):
                    ch_file = f
                elif f.endswith('.cnt'):
                    cnt_files.append(f)

            if not ch_file or not cnt_files:
                logger.warning("Missing channel table or data files in ZIP")
                return sac_files

            # Parse channel table to find target channels
            ch_path = extract_dir / ch_file
            channels = self._parse_channel_table(ch_path, station_filter)

            if not channels:
                logger.warning("No matching channels found")
                return sac_files

            # Create win.prm file (required by win2sac_32)
            prm_path = extract_dir / 'win.prm'
            with open(prm_path, 'w') as f:
                f.write(f".\n{ch_file}\n.\n.")

            # Convert each data file
            sac_dir = output_dir / 'sac'
            sac_dir.mkdir(parents=True, exist_ok=True)

            for cnt_file in cnt_files:
                for ch_num, info in channels.items():
                    sac_file = self._convert_channel(
                        extract_dir,
                        cnt_file,
                        ch_num,
                        info,
                        sac_dir
                    )
                    if sac_file:
                        sac_files.append(sac_file)

            logger.info(f"Converted {len(sac_files)} SAC files")

        except Exception as e:
            logger.error(f"Extraction/conversion error: {e}")

        return sac_files

    def _parse_channel_table(
        self,
        ch_path: Path,
        station_filter: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Parse Hi-net channel table.

        Args:
            ch_path: Path to channel table file
            station_filter: Optional list of station codes to include

        Returns:
            Dict mapping channel number to station info
        """
        channels = {}

        try:
            with open(ch_path, 'r', encoding='euc-jp', errors='replace') as f:
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

                        # Apply station filter
                        if station_filter and station not in station_filter:
                            continue

                        # Filter to Kanto region if no specific filter
                        if not station_filter:
                            if not (KANTO_BOUNDS['min_lat'] <= lat <= KANTO_BOUNDS['max_lat'] and
                                    KANTO_BOUNDS['min_lon'] <= lon <= KANTO_BOUNDS['max_lon']):
                                continue

                        # Only vertical component for THD
                        if component == 'U':
                            channels[ch_num] = {
                                'station': station,
                                'component': component,
                                'lat': lat,
                                'lon': lon
                            }

        except Exception as e:
            logger.error(f"Error parsing channel table: {e}")

        return channels

    def _convert_channel(
        self,
        work_dir: Path,
        cnt_file: str,
        ch_num: str,
        info: Dict,
        sac_dir: Path,
    ) -> Optional[Path]:
        """
        Convert a single channel from WIN32 to SAC.

        Args:
            work_dir: Directory containing WIN32 files
            cnt_file: WIN32 data file name
            ch_num: Channel number
            info: Channel info dict
            sac_dir: Output directory for SAC files

        Returns:
            Path to SAC file or None on failure
        """
        station = info['station']
        component = info.get('component', 'U')

        # win2sac_32 creates files as: {station}.{component}.{user_suffix}
        # So pass 'SAC' to get 'N.KI2H.U.SAC'
        sac_suffix = 'SAC'
        expected_name = f"{station}.{component}.{sac_suffix}"
        sac_path = sac_dir / expected_name

        # Skip if already exists
        if sac_path.exists():
            return sac_path

        # Convert Windows path to WSL path
        wsl_work_dir = str(work_dir).replace('C:', '/mnt/c').replace('\\', '/')

        # win2sac_32 creates file in current directory
        cmd = (
            f"cd {wsl_work_dir} && "
            f"{WIN2SAC_PATH} {cnt_file} {ch_num} {sac_suffix}"
        )

        try:
            result = subprocess.run(
                ['wsl', 'bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Check if file was created in work dir
            created_file = work_dir / expected_name
            if created_file.exists():
                # Move to sac_dir
                shutil.move(str(created_file), str(sac_path))
                logger.debug(f"Converted {station} channel {ch_num}")
                return sac_path
            else:
                # Check for any SAC file with this station
                for f in work_dir.iterdir():
                    if f.name.startswith(f"{station}.{component}") and f.suffix in ('.SAC', '.sac'):
                        final_path = sac_dir / f.name
                        shutil.move(str(f), str(final_path))
                        logger.debug(f"Converted {station} channel {ch_num} -> {f.name}")
                        return final_path

                logger.debug(f"Conversion produced no output for {station}")

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout converting {station}")
        except Exception as e:
            logger.warning(f"Conversion error for {station}: {e}")

        return None

    def fetch_continuous(
        self,
        network: str,
        start_time: datetime,
        duration_minutes: int,
        stations: Optional[List[str]] = None,
    ) -> Optional['obspy.Stream']:
        """
        Fetch continuous waveform data from Hi-net.

        Uses status page polling method for reliable download.

        Args:
            network: Hi-net network code (e.g., '0101')
            start_time: Start time for data
            duration_minutes: Duration in minutes (max 5 per request)
            stations: List of station codes (None = Kanto region stations)

        Returns:
            ObsPy Stream object or None on failure
        """
        try:
            from obspy import Stream, read

            # Ensure authenticated
            self._get_client()

            # Chunk into MAX_MINUTES_PER_REQUEST segments
            all_traces = Stream()
            current_start = start_time
            remaining = duration_minutes

            while remaining > 0:
                chunk_minutes = min(remaining, MAX_MINUTES_PER_REQUEST)

                logger.info(f"Fetching {chunk_minutes} minutes from {current_start}")

                # Submit request
                if not self._submit_request(network, current_start, chunk_minutes):
                    logger.warning(f"Failed to submit request for {current_start}")
                    current_start += timedelta(minutes=chunk_minutes)
                    remaining -= chunk_minutes
                    continue

                # Poll for available download
                request_id = self._poll_for_available()
                if not request_id:
                    logger.warning(f"No data available for {current_start}")
                    current_start += timedelta(minutes=chunk_minutes)
                    remaining -= chunk_minutes
                    continue

                # Download ZIP
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    zip_path = tmpdir / f"hinet_{request_id}.zip"

                    if not self._download_by_id(request_id, zip_path):
                        logger.warning(f"Download failed for {current_start}")
                        current_start += timedelta(minutes=chunk_minutes)
                        remaining -= chunk_minutes
                        continue

                    # Extract and convert
                    sac_files = self._extract_and_convert(zip_path, tmpdir, stations)

                    # Read SAC files
                    for sac_file in sac_files:
                        try:
                            st = read(str(sac_file))
                            all_traces += st
                        except Exception as e:
                            logger.warning(f"Error reading {sac_file}: {e}")

                current_start += timedelta(minutes=chunk_minutes)
                remaining -= chunk_minutes

            # Merge traces
            if len(all_traces) > 0:
                all_traces.merge(method=1, fill_value='interpolate')
                logger.info(f"Retrieved {len(all_traces)} traces")
                return all_traces

            return None

        except ImportError:
            logger.error("ObsPy not installed. Run: pip install obspy")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch Hi-net data: {e}")
            return None

    def get_station_catalog(
        self,
        network: str = '0101',
        latitude_range: Tuple[float, float] = (34.5, 37.0),
        longitude_range: Tuple[float, float] = (138.5, 141.5),
    ) -> Dict[str, Tuple[float, float, str]]:
        """
        Get station catalog for a region.

        Args:
            network: Hi-net network code
            latitude_range: (min_lat, max_lat)
            longitude_range: (min_lon, max_lon)

        Returns:
            Dict mapping station code to (lat, lon, name)
        """
        # Known Kanto Hi-net stations (verified from channel table)
        stations = {
            'N.KI2H': (36.88, 140.65, 'Kita-Ibaraki'),
            'N.SSAH': (35.72, 140.50, 'Sanmu'),
            'N.HYMH': (35.26, 139.61, 'Hayama'),
            'N.KMSH': (35.09, 140.10, 'Kamogawa'),
            'N.IWSH': (36.37, 140.14, 'Iwase'),
            'N.TYOH': (36.12, 140.56, 'Toyo'),
            'N.ISGH': (36.11, 139.99, 'Ishige'),
            'N.KYWH': (36.88, 139.45, 'Kawaji'),
            'N.SITH': (35.98, 139.40, 'Saitama'),
            'N.CBAH': (35.60, 140.13, 'Chiba'),
        }

        # Filter by lat/lon
        filtered = {}
        for code, (lat, lon, name) in stations.items():
            if (latitude_range[0] <= lat <= latitude_range[1] and
                longitude_range[0] <= lon <= longitude_range[1]):
                filtered[code] = (lat, lon, name)

        logger.info(f"Found {len(filtered)} stations in Kanto region")
        return filtered

    def fetch_for_thd(
        self,
        station: str,
        network: str = '0101',
        start_time: datetime = None,
        hours: int = 25,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Fetch data formatted for THD analysis.

        Args:
            station: Station code (e.g., 'N.KI2H')
            network: Hi-net network code
            start_time: Start time (default: 2 days ago)
            hours: Duration in hours (default: 25 for 24h THD window)

        Returns:
            Tuple of (data_array, sample_rate) or (None, 0) on failure
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=2)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        stream = self.fetch_continuous(
            network=network,
            start_time=start_time,
            duration_minutes=hours * 60,
            stations=[station],
        )

        if stream is None or len(stream) == 0:
            return None, 0.0

        # Get vertical component
        for tr in stream:
            if tr.stats.channel.endswith('U') or tr.stats.channel.endswith('Z'):
                # Detrend
                tr.detrend('demean')
                tr.detrend('linear')
                return tr.data, tr.stats.sampling_rate

        # Fall back to first trace
        tr = stream[0]
        tr.detrend('demean')
        tr.detrend('linear')

        return tr.data, tr.stats.sampling_rate


# =============================================================================
# INTEGRATION WITH SEISMIC DATA FETCHER
# =============================================================================

def is_hinet_station(network: str, station: str) -> bool:
    """Check if a station should use Hi-net data source."""
    return network.startswith('0') or network == 'HINET' or station.startswith('N.')


def get_hinet_client() -> Optional[HinetClient]:
    """Get a configured Hi-net client from environment variables."""
    username = os.environ.get('HINET_USER')
    password = os.environ.get('HINET_PASSWORD')

    if not username or not password:
        logger.warning("Hi-net credentials not configured")
        return None

    return HinetClient(username=username, password=password)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Hi-net Data Module Test")
    print("=" * 60)

    # Check credentials
    username = os.environ.get('HINET_USER')
    password = os.environ.get('HINET_PASSWORD')

    if not username or not password:
        print("\nHi-net credentials not configured.")
        print("Set environment variables:")
        print("  HINET_USER=your_username")
        print("  HINET_PASSWORD=your_password")
        print("\nSkipping connection test.")
    else:
        print(f"\nCredentials found for user: {username}")

        client = HinetClient(username=username, password=password)

        # Test connection
        print("\nTesting connection...")
        if client.test_connection():
            print("Connection successful!")

            # Get station catalog
            print("\nKanto station catalog:")
            stations = client.get_station_catalog()
            for code, (lat, lon, name) in stations.items():
                print(f"  {code}: {name} ({lat:.2f}N, {lon:.2f}E)")

            # Test data fetch (small request)
            print("\nFetching 3 minutes of test data...")
            stream = client.fetch_continuous(
                network='0101',
                start_time=datetime.now() - timedelta(days=3),
                duration_minutes=3,
            )

            if stream:
                print(f"Retrieved {len(stream)} traces")
                for tr in stream[:5]:
                    print(f"  {tr.stats.station}.{tr.stats.channel}: "
                          f"{tr.stats.npts} samples @ {tr.stats.sampling_rate} Hz")
            else:
                print("No data retrieved")
        else:
            print("Connection failed. Check credentials.")

    print("\n" + "=" * 60)
    print("Test complete")
