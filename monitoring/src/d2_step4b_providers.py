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
import os
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


def _http_get(url: str, timeout: int, retries: int = 4) -> bytes:
    """Anonymous GET, resilient to the transient faults expected across a multi-hour campaign
    (connection reset / RemoteDisconnected / IncompleteRead / timeout, and the transient HTTP
    statuses 408/425/429 + 500-599): each is retried with capped exponential backoff. 204/404
    (FDSN/S3 no-data) and every non-transient client status (401/403/etc.) -> ProviderUnavailable
    with NO retry. A persistent transient failure after `retries` attempts -> ProviderUnavailable
    (fail-closed, never a silent empty result)."""
    import http.client
    import socket
    import time
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=_UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if getattr(resp, "status", 200) == 204:
                    raise ProviderUnavailable(f"204 no content: {url}")
                body = resp.read()
            if not body:
                raise ProviderUnavailable(f"empty body: {url}")
            return body
        except urllib.error.HTTPError as exc:
            if exc.code in (204, 404):
                raise ProviderUnavailable(f"{exc.code} {exc.reason}: {url}")
            if not (exc.code in (408, 425, 429) or 500 <= exc.code <= 599):
                # permanent client failure (401/403 and every other non-transient 4xx) ->
                # IMMEDIATE typed failure: one call, no sleep, no request multiplication.
                # codex 113942Z exact set: retry ONLY the transient idempotent-GET statuses.
                raise ProviderUnavailable(f"{exc.code} {exc.reason}: {url}")
            last = f"HTTP {exc.code} {exc.reason}"                 # 408/425/429 + 5xx -> retry
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                TimeoutError, socket.timeout, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"                 # transient network -> retry
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 15))                     # 1,2,4,8 (cap 15) s backoff
    raise ProviderUnavailable(f"network error after {retries} attempts ({last}): {url}")


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


def _stage(stage_dir: str, source_url: str, raw: bytes) -> dict:
    """Persist the exact served bytes to a content-addressed file under stage_dir and return the
    typed raw-object receipt {source, staged_path, size_bytes, sha256} (P2). Called BEFORE any
    parse so a downstream parse failure still leaves the object byte-for-byte on disk."""
    digest = hashlib.sha256(raw).hexdigest()
    staged_path = os.path.join(stage_dir, f"{digest}.ms")
    with open(staged_path, "wb") as fh:
        fh.write(raw)
    return {"source": source_url, "staged_path": staged_path,
            "size_bytes": len(raw), "sha256": digest}


def parse_staged(staged_path: str):
    """Reopen ONE staged raw object and parse IT ALONE (obspy.read on its own bytes). The executor
    uses this for per-object provenance (H3) so a multi-day-volume response yields one provenance
    row per object; keeping the parse here keeps the executor obspy-free + the bar hermetic."""
    import obspy
    with open(staged_path, "rb") as fh:
        return obspy.read(io.BytesIO(fh.read()))


def verify_staged_object(obj: dict) -> bool:
    """Reopen staged_path and re-derive: True iff BOTH size_bytes and sha256 match the file
    exactly; False on any byte mutation or missing file (P2)."""
    try:
        path = obj["staged_path"]
        if not os.path.isfile(path):
            return False
        raw = open(path, "rb").read()
        return (len(raw) == obj.get("size_bytes")
                and hashlib.sha256(raw).hexdigest() == obj.get("sha256"))
    except Exception:
        return False


# ---- KOERI (FDSN dataselect) -----------------------------------------------
def koeri_fetch(nslc: str, start: datetime, end: datetime, *, stage_dir: str,
                base: str = KOERI_BASE, timeout: int = 240):
    """Fetch KOERI waveforms over the exact [start, end] interval via FDSN dataselect. The exact
    served bytes are PERSISTED under stage_dir BEFORE parsing (P2); a parse failure keeps the
    staged object byte-for-byte. Returns {"stream": <trimmed>, "raw_objects": [obj]}."""
    net, sta, loc, cha = parse_nslc(nslc)
    locq = loc if loc else "--"
    url = (f"{base}/fdsnws/dataselect/1/query?net={net}&sta={sta}&loc={locq}&cha={cha}"
           f"&starttime={_fdsn_time(start)}&endtime={_fdsn_time(end)}&format=miniseed&nodata=404")
    raw = _http_get(url, timeout)
    obj = _stage(stage_dir, url, raw)                 # persist BEFORE parse
    stream = _read_trim(raw, start, end)              # may raise -> staged object survives
    return {"stream": stream, "raw_objects": [obj]}


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
        if not line or line.startswith("#") or line[:7].lower() == "network":
            continue
        cols = line.split()
        if len(cols) < 3:
            continue
        n, s = cols[0], cols[1]
        # The FDSN availability text renders an EMPTY location as blank, which a whitespace split
        # collapses; locate the channel as the 3-letter code and the location as the token just
        # before it (empty when absent / '--'), so NET.STA..CHA candidates match correctly.
        cha, loc = None, ""
        for i in range(2, min(len(cols), 5)):
            tok = cols[i]
            if len(tok) == 3 and tok.isalpha():
                cha = tok
                prev = cols[i - 1] if i - 1 >= 2 else None
                if prev is not None and prev not in ("--", ""):
                    loc = prev
                break
        if cha is not None:
            present.add(f"{n}.{s}.{loc}.{cha}")
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
    """HALF-OPEN [start, end) list of UTC calendar days touched (P3). ValueError if end <= start;
    the last enumerated day is (end - 1 microsecond).date(), so an exact-midnight 24 h window
    touches exactly ONE day."""
    s, e = _as_utc(start), _as_utc(end)
    if e <= s:
        raise ValueError(f"end {end} <= start {start} (empty/negative interval)")
    last = (e - timedelta(microseconds=1)).date()
    days, cur = [], s.date()
    while cur <= last:
        days.append(cur)
        cur = cur + timedelta(days=1)
    return days


def scedc_fetch(nslc: str, start: datetime, end: datetime, *, stage_dir: str,
                base: str = SCEDC_BASE, timeout: int = 300):
    """Fetch SCEDC waveforms over [start, end] by assembling the touched public day-volumes. Each
    touched day-volume is a SEPARATELY-hashed raw object staged BEFORE parse (P2), in day order; a
    missing day-volume is tolerated (partial coverage) but every existing touched object is
    retained. The concatenated-bytes digest is never formed. Returns {"stream": <trimmed>,
    "raw_objects": [obj-per-day]}."""
    net, sta, loc, cha = parse_nslc(nslc)
    raw_objects, payloads = [], []
    for day in _utc_days_spanned(start, end):
        url = f"{base}/{_scedc_key(net, sta, loc, cha, day)}"
        try:
            raw = _http_get(url, timeout)
        except ProviderUnavailable:
            continue                                  # missing day-volume tolerated (partial)
        raw_objects.append(_stage(stage_dir, url, raw))   # persist BEFORE parse, day order
        payloads.append(raw)
    if not raw_objects:
        raise ProviderUnavailable(f"no SCEDC day-volume for {nslc} over [{start}, {end}]")
    import obspy
    from obspy import UTCDateTime
    st = obspy.Stream()
    for raw in payloads:
        st += obspy.read(io.BytesIO(raw))
    st.merge(method=0)                                # join contiguous day-volume traces
    st.trim(UTCDateTime(_as_utc(start)), UTCDateTime(_as_utc(end)))
    st = st.split()
    return {"stream": st, "raw_objects": raw_objects}


def scedc_available(net: str, stations, channels, start: datetime, end: datetime,
                    base: str = SCEDC_BASE, timeout: int = 120):
    """Read-only availability: HEAD EVERY touched UTC day-volume for every frozen candidate; the
    NSLC is present iff ANY touched day-volume object exists (P3) — data only in the second
    touched day still registers available. urllib transport (the bar's interception contract);
    no waveform bytes are downloaded beyond the presence checks."""
    days = _utc_days_spanned(start, end)
    present = set()
    for sta in stations:
        for cha in channels:
            for day in days:
                key = _scedc_key(net, sta, "", cha, day)
                req = urllib.request.Request(f"{base}/{key}", headers=_UA, method="HEAD")
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if getattr(resp, "status", None) == 200:
                            present.add(f"{net}.{sta}..{cha}")
                            break                     # any touched day suffices
                except Exception:
                    continue
    return present


# ---- dispatch --------------------------------------------------------------
_FETCH = {"KOERI": koeri_fetch, "SCEDC": scedc_fetch}
_AVAIL = {"KOERI": koeri_available, "SCEDC": scedc_available}


def fetch(provider: str, nslc: str, start: datetime, end: datetime, *, stage_dir: str, **kw):
    if provider not in _FETCH:
        raise ValueError(f"unknown provider {provider!r} (frozen set: {sorted(_FETCH)})")
    return _FETCH[provider](nslc, start, end, stage_dir=stage_dir, **kw)
