#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_step4b_providers.py — read-only waveform providers for the D2 step-4b campaign.

Isolated on purpose: `d2_step4b_producer.py` (under cayley's SB-0..8 bar) must not import a
network stack at module load, so the outward I/O lives here and is imported lazily only past the
producer's SB-8 `VERIFIED_DIRECT` launch gate.

Two providers, exactly as frozen in `campaign_plan.json["providers"]`:

* **KOERI** (`istanbul_marmara`, `turkey_kahramanmaras`) — FDSN `dataselect` over the exact
  requested interval (`eida.koeri.boun.edu.tr`).
* **SCEDC** (`socal_coachella`) — anonymous HTTPS day-volumes from the public `s3://scedc-pds`
  bucket over its REST endpoint. This honors the pinned `s3://scedc-pds` endpoint with **no
  credentials and no boto3**; a 24 h non-midnight interval touches up to two UTC day-volumes,
  which are read, merged, and trimmed to the bound interval.

Every fetch returns `(obspy.Stream trimmed to [start, end], sha256-hex of the raw provider bytes)`
or raises `ProviderUnavailable`. Nothing here selects a session, admits a value, or lifts a
freeze; it only retrieves bytes for an already-bound (carrier, day) request interval.
"""
import hashlib
import io
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

KOERI_BASE = "http://eida.koeri.boun.edu.tr"
SCEDC_BASE = "https://scedc-pds.s3.amazonaws.com"
SCEDC_BUCKET = "s3://scedc-pds"                       # the frozen endpoint token
_UA = {"User-Agent": "geospec-d2-step4b/1.0"}


class ProviderUnavailable(Exception):
    """No usable bytes for the requested (nslc, interval) from the bound provider."""


# ---- helpers ---------------------------------------------------------------
def parse_nslc(nslc: str):
    """'CI.BOR..BHZ' -> ('CI', 'BOR', '', 'BHZ'). Exactly four dot-separated fields."""
    parts = nslc.split(".")
    if len(parts) != 4:
        raise ValueError(f"NSLC {nslc!r} is not NET.STA.LOC.CHA")
    net, sta, loc, cha = parts
    return net, sta, loc, cha


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fdsn_time(dt: datetime) -> str:
    # FDSN wants a naive-UTC ISO instant; microseconds preserved.
    return _as_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _http_get(url: str, timeout: int) -> bytes:
    """Anonymous GET. 204/404 (FDSN/S3 no-data) -> ProviderUnavailable; other errors re-raised
    as ProviderUnavailable with the reason attached (fail-closed, never a silent empty result)."""
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                raise ProviderUnavailable(f"204 no content: {url}")
            body = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (204, 404):
            raise ProviderUnavailable(f"{exc.code} {exc.reason}: {url}")
        raise ProviderUnavailable(f"HTTP {exc.code} {exc.reason}: {url}")
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(f"URLError {exc.reason}: {url}")
    if not body:
        raise ProviderUnavailable(f"empty body: {url}")
    return body


def _read_trim(raw: bytes, start: datetime, end: datetime):
    """Parse miniSEED bytes and trim to [start, end] UTC (obspy imported lazily)."""
    import obspy
    from obspy import UTCDateTime
    st = obspy.read(io.BytesIO(raw))
    st.trim(UTCDateTime(_as_utc(start)), UTCDateTime(_as_utc(end)))
    st = st.split()                                  # drop masked gaps -> contiguous traces
    if len(st) == 0:
        raise ProviderUnavailable("no samples inside the requested interval after trim")
    return st


# ---- KOERI (FDSN dataselect) -----------------------------------------------
def koeri_fetch(nslc: str, start: datetime, end: datetime, base: str = KOERI_BASE,
                timeout: int = 240):
    """Fetch KOERI waveforms over the exact [start, end] interval via FDSN dataselect."""
    net, sta, loc, cha = parse_nslc(nslc)
    locq = loc if loc else "--"
    url = (f"{base}/fdsnws/dataselect/1/query?net={net}&sta={sta}&loc={locq}&cha={cha}"
           f"&starttime={_fdsn_time(start)}&endtime={_fdsn_time(end)}&format=miniseed&nodata=404")
    raw = _http_get(url, timeout)
    return _read_trim(raw, start, end), hashlib.sha256(raw).hexdigest()


def koeri_available(net: str, stations, channels, start: datetime, end: datetime,
                    base: str = KOERI_BASE, timeout: int = 120):
    """Read-only availability probe: the set of NET.STA..CHA present in the extent service over
    [start, end]. Used to drive frozen NSLC selection without fetching any waveform bytes."""
    stas = ",".join(stations)
    chas = ",".join(channels)
    url = (f"{base}/fdsnws/availability/1/query?net={net}&sta={stas}&cha={chas}"
           f"&starttime={_fdsn_time(start)}&endtime={_fdsn_time(end)}&format=text&nodata=404")
    present = set()
    try:
        body = _http_get(url, timeout).decode("utf-8", "replace")
    except ProviderUnavailable:
        return present
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) >= 4:
            n, s, l, c = cols[0], cols[1], cols[2], cols[3]
            l = "" if l in ("--", "") else l
            present.add(f"{n}.{s}.{l}.{c}")
    return present


# ---- SCEDC (public s3://scedc-pds day-volumes over HTTPS) ------------------
def _scedc_key(net: str, sta: str, loc: str, cha: str, day: date) -> str:
    """SCEDC PDS continuous-waveform object key, e.g. CI.BOR..BHZ on 2026-03-31 (jday 090) ->
    continuous_waveforms/2026/2026_090/CIBOR__BHZ___2026090.ms
    Layout: {NET:2}{STA:_<5}{CHA:3}{LOC:_<2}_{YYYY}{JJJ}.ms (empty loc -> '__')."""
    jday = day.timetuple().tm_yday
    stem = f"{net}{sta:_<5}{cha}{loc:_<2}_{day.year}{jday:03d}.ms"
    return f"continuous_waveforms/{day.year}/{day.year}_{jday:03d}/{stem}"


def _utc_days_spanned(start: datetime, end: datetime):
    """Inclusive list of UTC calendar days the [start, end) interval touches (1 or 2 for a 24 h
    non-midnight window)."""
    d0 = _as_utc(start).date()
    d1 = _as_utc(end).date()
    days, cur = [], d0
    while cur <= d1:
        days.append(cur)
        cur = cur + timedelta(days=1)
    return days


def scedc_fetch(nslc: str, start: datetime, end: datetime, base: str = SCEDC_BASE,
                timeout: int = 300):
    """Fetch SCEDC waveforms over [start, end] by assembling the touched public day-volumes and
    trimming. A missing day-volume is tolerated (partial coverage); the scoring coverage floor
    decides admissibility. Raises ProviderUnavailable only if NO day-volume yields samples."""
    net, sta, loc, cha = parse_nslc(nslc)
    raws = []
    for day in _utc_days_spanned(start, end):
        url = f"{base}/{_scedc_key(net, sta, loc, cha, day)}"
        try:
            raws.append(_http_get(url, timeout))
        except ProviderUnavailable:
            continue
    if not raws:
        raise ProviderUnavailable(f"no SCEDC day-volume for {nslc} over [{start}, {end}]")
    import obspy
    st = obspy.Stream()
    for raw in raws:
        st += obspy.read(io.BytesIO(raw))
    from obspy import UTCDateTime
    st.merge(method=0)                                # join contiguous day-volume traces
    st.trim(UTCDateTime(_as_utc(start)), UTCDateTime(_as_utc(end)))
    st = st.split()
    if len(st) == 0:
        raise ProviderUnavailable("SCEDC day-volumes carried no samples inside the interval")
    return st, hashlib.sha256(b"".join(raws)).hexdigest()


def scedc_available(net: str, stations, channels, start: datetime, end: datetime,
                    base: str = SCEDC_BASE, timeout: int = 120):
    """Read-only availability probe via S3 object HEAD/GET on the first touched day-volume: the
    set of NET.STA..CHA whose day-volume object exists. No waveform bytes are downloaded here
    beyond the presence check for the first day."""
    day = _utc_days_spanned(start, end)[0]
    present = set()
    for sta in stations:
        for cha in channels:
            key = _scedc_key(net, sta, "", cha, day)
            req = urllib.request.Request(f"{base}/{key}", headers=_UA, method="HEAD")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        present.add(f"{net}.{sta}..{cha}")
            except Exception:
                continue
    return present


# ---- dispatch --------------------------------------------------------------
_FETCH = {"KOERI": koeri_fetch, "SCEDC": scedc_fetch}
_AVAIL = {"KOERI": koeri_available, "SCEDC": scedc_available}


def fetch(provider: str, nslc: str, start: datetime, end: datetime, **kw):
    if provider not in _FETCH:
        raise ValueError(f"unknown provider {provider!r} (frozen set: {sorted(_FETCH)})")
    return _FETCH[provider](nslc, start, end, **kw)
