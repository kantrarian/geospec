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
from scipy import signal as sp_signal
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fractions import Fraction
import hashlib
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

# INCIDENT 2026-07-31 (D2) — fault_correlation LOCAL band. The 0.01-1.0 Hz DEFAULT_FILTER captures spatially
# COHERENT long-period energy (teleseismic surface waves, microseism, geomagnetic/EM coupling), so any large
# long-period arrival co-elevated segment activity across ALL regions at once -> lambda2/lambda1 collapse ->
# false "decorrelation precursor" (the 07-29 artifact: turkey lambda2/lambda1 0.128 in this band). Local
# fault-segment stress decorrelation lives in the ~1-10 Hz micro-seismic band, which is spatially INCOHERENT
# between distant regions -- a genuine local signal. Re-banding here restores a healthy correlation structure
# (turkey 07-29: lambda2/lambda1 0.128 -> 0.594). Only get_segment_envelopes (fault_correlation) uses this;
# seismic_thd fetches separately and is unaffected.
FAULT_CORR_FILTER = {
    'freqmin': 1.0,    # 1 Hz  -- above the microseism / long-period coherent band
    'freqmax': 10.0,   # 10 Hz -- local micro-seismicity
    'corners': 4,
    'zerophase': True,
}
_FAULT_CORR_BAND_TAG = "1-10Hz"   # cache version tag: re-banded envelopes cache separately from 0.01-1Hz

DEFAULT_DECIMATE_FACTOR = 40  # 40 Hz -> 1 Hz


# =============================================================================
# D2 RE-BAND rev-2 (INCIDENT 2026-07-31) — envelope-order correction + non-forgeable
# provenance. The fault-correlation activity envelope is now computed by ONE core
# (compute_band_envelope_from_array): bandpass at the NATIVE rate -> Hilbert envelope
# WHILE the native carrier still exists -> low-pass + resample the ENVELOPE to 1 Hz.
# The old waveform->decimate-to-1Hz->envelope order destroyed the AM carrier before the
# envelope was taken and is never used on this path. Every field of every cached envelope
# is bound by a canonical EnvelopeCacheIdentity + an output digest, so the provenance
# labels the observability gate reads (pre_envelope_rate_hz / band / version) cannot be
# minted outside this core + verify-on-load cache.
# =============================================================================

# Bumped for rev-2; part of every cache identity + every EnvelopeSeries provenance label.
PROCESSING_VERSION = "d2-reband-v2"
# Correlated-station topology revision that the cache identity is keyed to.
_TOPOLOGY_VERSION = "t1"


class DataUnavailable(Exception):
    """Raised when a segment/stream cannot be honestly turned into a rev-2 envelope
    (rate below the band Nyquist, empty carrier, etc.). Never coerced through the band."""

    def __init__(self, reasons):
        self.reasons = list(reasons) if isinstance(reasons, (list, tuple)) else [str(reasons)]
        super().__init__("; ".join(self.reasons))


@dataclass
class EnvelopeSeries:
    """A 1 Hz activity envelope with the provenance the observability gate depends on."""
    values: np.ndarray            # envelope samples at out_rate (1 Hz)
    start_utc: datetime           # AWARE UTC start of sample 0
    dt_seconds: float             # 1 / out_rate_hz
    coverage: float               # fraction of the carrier covered (0..1)
    gaps: List[Tuple[str, float]] # [(iso_utc_start, seconds), ...]
    source_ids: List[str]         # NSLC[.channel] contributors
    band_tag: str                 # e.g. "1-10Hz"
    processing_version: str       # PROCESSING_VERSION at compute time
    pre_envelope_rate_hz: float   # NATIVE rate the envelope was taken at (anti-alias witness)


@dataclass
class EnvelopeCacheIdentity:
    """Canonical, non-forgeable identity for a cached envelope. Every field participates
    in the cache key (D2R-3a); the payload is only returned on an exact identity match AND
    an output-digest recompute (D2R-3b/3c)."""
    region: str
    segment: str
    source_id: str
    start_utc: datetime
    end_utc: datetime
    native_rate_hz: float
    band_tag: str
    processing_version: str
    topology_version: str
    input_sha256: str


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


def _parse_iso_utc(s):
    """Parse an ISO-8601 UTC datetime, tolerating a trailing 'Z'."""
    if isinstance(s, datetime):
        return s
    txt = str(s)
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    return datetime.fromisoformat(txt)


def _identity_to_dict(identity: "EnvelopeCacheIdentity") -> Dict:
    """Deterministic dict of ALL ten identity fields (datetimes -> ISO)."""
    return {
        "region": identity.region,
        "segment": identity.segment,
        "source_id": identity.source_id,
        "start_utc": _iso(identity.start_utc),
        "end_utc": _iso(identity.end_utc),
        "native_rate_hz": float(identity.native_rate_hz),
        "band_tag": identity.band_tag,
        "processing_version": identity.processing_version,
        "topology_version": identity.topology_version,
        "input_sha256": identity.input_sha256,
    }


def build_envelope_cache_identity(*, region, segment, source_id, start_utc, end_utc,
                                  band_tag, processing_version, topology_version,
                                  native_rate_hz, input_sha256) -> EnvelopeCacheIdentity:
    """Assemble the canonical identity. The SHELL derives native_rate_hz from the actual
    trace and input_sha256 from the exact input sample bytes before calling this."""
    return EnvelopeCacheIdentity(
        region=region, segment=segment, source_id=source_id,
        start_utc=start_utc, end_utc=end_utc,
        native_rate_hz=float(native_rate_hz), band_tag=band_tag,
        processing_version=processing_version, topology_version=topology_version,
        input_sha256=input_sha256,
    )


def compute_band_envelope_from_array(data, rate_hz, *, freqmin=1.0, freqmax=10.0,
                                     out_rate_hz=1.0, min_rate_hz=25.0,
                                     start_utc, source_id) -> EnvelopeSeries:
    """THE rev-2 DSP core. ALL band DSP lives here:
      1. bandpass [freqmin, freqmax] at the NATIVE rate,
      2. Hilbert envelope while the native carrier still exists,
      3. anti-aliased resample of the ENVELOPE to out_rate_hz.
    pre_envelope_rate_hz records the native rate. A trace whose native rate cannot carry
    the band (rate < min_rate_hz, or freqmax >= native Nyquist) raises DataUnavailable and
    is NEVER coerced through the band."""
    data = np.asarray(data, dtype=float).ravel()
    rate_hz = float(rate_hz)
    if not np.isfinite(rate_hz) or rate_hz < float(min_rate_hz):
        raise DataUnavailable([
            f"native sampling rate {rate_hz} Hz below minimum {min_rate_hz} Hz for the "
            f"{freqmin}-{freqmax} Hz band (native Nyquist {rate_hz / 2.0} Hz)"])
    nyq = rate_hz / 2.0
    if freqmax >= nyq:
        raise DataUnavailable([
            f"band max {freqmax} Hz >= native Nyquist {nyq} Hz at rate {rate_hz} Hz"])
    if data.size < 4:
        raise DataUnavailable([f"empty/too-short carrier ({data.size} samples)"])

    # 1) bandpass at the NATIVE rate (zero-phase)
    b, a = sp_signal.butter(4, [freqmin / nyq, freqmax / nyq], btype="band")
    bandpassed = sp_signal.filtfilt(b, a, data)
    # 2) Hilbert envelope at the native rate (before any decimation)
    envelope = np.abs(hilbert(bandpassed))
    # 3) anti-aliased resample of the ENVELOPE to out_rate_hz. codex 0409 #3: form the resampling
    # ratio from the RATIONAL out/native rate (never integer-rounded rates -- a 40.5 Hz native
    # rate must not be coerced to 40) and declare dt from the REALIZED output rate, so the 1 Hz
    # carrier's declared duration equals the real elapsed time.
    ratio = Fraction(float(out_rate_hz) / rate_hz).limit_denominator(100000)
    up, down = ratio.numerator, ratio.denominator
    if up < 1:
        up = 1
    if down < 1:
        down = 1
    env_out = np.asarray(sp_signal.resample_poly(envelope, up, down), dtype=float)
    realized_out_rate_hz = rate_hz * up / down

    return EnvelopeSeries(
        values=env_out,
        start_utc=start_utc,
        dt_seconds=1.0 / realized_out_rate_hz,
        coverage=1.0,
        gaps=[],
        source_ids=[source_id],
        band_tag=_FAULT_CORR_BAND_TAG,
        processing_version=PROCESSING_VERSION,
        pre_envelope_rate_hz=rate_hz,
    )


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

    def _waveform_cache_path(self, region: str, date: datetime, data_type: str) -> Path:
        """Generate the legacy pickle cache file path (raw waveform fragments)."""
        date_str = date.strftime('%Y%m%d')
        return self.cache_dir / region / date_str / f'{data_type}.pkl'

    # ---- D2 rev-2 envelope cache (identity-keyed, verify-on-load) ---------------
    def _cache_path(self, region, start, end, data_type,
                    identity: "EnvelopeCacheIdentity") -> Path:
        """Envelope cache path. ALL TEN identity fields participate in the key: with the
        positional (region, start, end, data_type) held identical, any single identity
        field change yields a different path (D2R-3a)."""
        blob = json.dumps(_identity_to_dict(identity), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()[:32]
        start_str = start.strftime('%Y%m%dT%H%M%SZ') if hasattr(start, 'strftime') else str(start)
        end_str = end.strftime('%Y%m%dT%H%M%SZ') if hasattr(end, 'strftime') else str(end)
        return (Path(self.cache_dir) / region /
                f'{data_type}_{start_str}_{end_str}_{digest}.json')

    def _save_cached_series(self, path, identity: "EnvelopeCacheIdentity",
                            series: "EnvelopeSeries"):
        """Store the envelope with its canonical identity and an output digest over the
        stored values, so a later load can recompute and refuse a tampered payload."""
        values_list = [float(v) for v in np.asarray(series.values, dtype=float).ravel()]
        output_sha = hashlib.sha256(
            json.dumps(values_list, separators=(",", ":")).encode("utf-8")).hexdigest()
        payload = {
            "identity": _identity_to_dict(identity),
            "output_sha256": output_sha,
            "series": {
                "values": values_list,
                "start_utc": _iso(series.start_utc),
                "dt_seconds": float(series.dt_seconds),
                "coverage": float(series.coverage),
                "gaps": [list(g) for g in series.gaps],
                "source_ids": list(series.source_ids),
                "band_tag": series.band_tag,
                "processing_version": series.processing_version,
                "pre_envelope_rate_hz": float(series.pre_envelope_rate_hz),
            },
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def _load_cached_series(self, path, expected_identity: "EnvelopeCacheIdentity"):
        """Return the cached EnvelopeSeries ONLY IF the stored identity equals the expected
        identity AND the stored output digest recomputes over the stored values; otherwise
        None (fail-closed recompute — an old-order relabel or a tampered value is rejected)."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return None
        if payload.get("identity") != _identity_to_dict(expected_identity):
            return None
        s = payload.get("series")
        if not isinstance(s, dict) or "values" not in s:
            return None
        values_list = [float(v) for v in s["values"]]
        recomputed = hashlib.sha256(
            json.dumps(values_list, separators=(",", ":")).encode("utf-8")).hexdigest()
        if recomputed != payload.get("output_sha256"):
            return None
        return EnvelopeSeries(
            values=np.asarray(values_list, dtype=float),
            start_utc=_parse_iso_utc(s["start_utc"]),
            dt_seconds=float(s["dt_seconds"]),
            coverage=float(s["coverage"]),
            gaps=[tuple(g) for g in s.get("gaps", [])],
            source_ids=list(s.get("source_ids", [])),
            band_tag=s["band_tag"],
            processing_version=s["processing_version"],
            pre_envelope_rate_hz=float(s["pre_envelope_rate_hz"]),
        )

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
        cache_path = self._waveform_cache_path(segment.region, start, cache_key)

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
    ) -> Dict[str, "EnvelopeSeries"]:
        """
        rev-2 SHELL. Per stream it may merge/detrend/copy the stream object, but performs
        NO filter/decimate/resample on it — all band DSP lives in
        compute_band_envelope_from_array. It extracts (raw native array, actual native
        rate, NSLC), builds the DERIVED EnvelopeCacheIdentity, consults the verify-on-load
        cache, calls the core exactly once per cache-missed stream, and RETURNS THE CORE'S
        EnvelopeSeries OBJECTS. The legacy process_waveforms/compute_envelope helpers are
        never invoked on the fault-correlation path.

        Returns:
            Dictionary mapping station NSLC -> EnvelopeSeries
        """
        waveforms = self.fetch_segment_waveforms(segment, start, end, use_cache)

        envelopes: Dict[str, "EnvelopeSeries"] = {}
        for nslc, stream in waveforms.items():
            # Extract the raw native array + rate. Stream-level merge/detrend/copy is
            # permitted; band DSP (filter/decimate/resample) on the stream is NOT.
            # codex 0409 #2: TRUTHFUL carrier -- NEVER interpolate over data gaps. Inspect the raw
            # ObsPy gaps first; ANY real gap => skip the station (the core emits coverage=1.0/gaps=[],
            # so an interpolated fill would mint a false full-coverage carrier). Supporting partial
            # coverage later needs segmented DSP + truthful propagated gap metadata.
            try:
                raw_gaps = stream.get_gaps() if hasattr(stream, "get_gaps") else None
                if isinstance(raw_gaps, (list, tuple)) and len(raw_gaps) > 0:
                    logger.info(f"{nslc}: {len(raw_gaps)} raw data gap(s) -> skip (no interpolation)")
                    continue
                work = stream
                try:
                    work = stream.copy()
                    work.merge(method=1)          # NON-interpolating merge (no fill_value)
                    work.detrend('demean')
                    work.detrend('linear')
                except Exception:
                    work = stream
                trace = work[0]
                data = np.asarray(trace.data, dtype=float).ravel()
                native_rate = float(trace.stats.sampling_rate)
            except Exception as exc:
                logger.warning(f"{nslc}: could not extract native trace: {exc}")
                continue

            if data.size == 0:
                continue
            if not np.all(np.isfinite(data)):
                logger.info(f"{nslc}: non-finite samples (masked gap) -> skip (no interpolation)")
                continue

            # DERIVED identity: actual native rate + sha256 of the exact native samples.
            input_sha = hashlib.sha256(data.tobytes()).hexdigest()
            identity = build_envelope_cache_identity(
                region=segment.region, segment=segment.name, source_id=nslc,
                start_utc=start, end_utc=end,
                band_tag=_FAULT_CORR_BAND_TAG, processing_version=PROCESSING_VERSION,
                topology_version=_TOPOLOGY_VERSION,
                native_rate_hz=native_rate, input_sha256=input_sha)

            cache_path = None
            if use_cache:
                cache_path = self._cache_path(segment.region, start, end, "seg_env", identity)
                cached = self._load_cached_series(cache_path, identity)
                if cached is not None:
                    envelopes[nslc] = cached
                    continue

            try:
                series = compute_band_envelope_from_array(
                    data, native_rate, start_utc=start, source_id=nslc)
            except DataUnavailable as exc:
                logger.info(f"{nslc}: {exc}")
                continue

            if use_cache and cache_path is not None:
                try:
                    self._save_cached_series(cache_path, identity, series)
                except Exception as exc:
                    logger.warning(f"{nslc}: envelope cache save failed: {exc}")

            envelopes[nslc] = series

        logger.info(f"Computed rev-2 envelopes for {len(envelopes)} stations")
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
