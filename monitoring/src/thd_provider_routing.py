"""thd_provider_routing.py -- the explicit network -> provider adapter map for the THD station fetch, and the typed
outcome of each provider attempt (codex review 70ec9745 section 3; cayley 2026-10-05). PROPOSAL on a local branch:
nothing here is active until a reviewed method change carries it.

Why: the fetch used an if/elif ladder whose catch-all sent every other network through four generic FDSN providers
(IRIS, GEOFON, SCEDC, NCEDC). That never asked INGV for IV, sent Hi-net's non-FDSN selector to four FDSN services,
and recorded every refusal as an undifferentiated PROVIDER_ERROR. Here a network either has an explicit route, with
the basis for each adapter, or it is refused by name (NETWORK_NOT_ROUTED); no network is sprayed across providers.

A provider's refusal keeps its exception CLASS and HTTP STATUS (both non-free-text) next to the redacted reason, and
gets one typed outcome:
  NO_DATA             the provider answered "no data" (HTTP 204/404) or returned an empty stream
  AUTH_REQUIRED       HTTP 401/403, or an adapter whose only access path is registered/credentialed
  INVALID_SELECTOR    HTTP 400/422: the provider refused the request's selector itself
  TRANSPORT_ERROR     no usable answer: service discovery failed, timeout, connection, HTTP 429/5xx
  PROVIDER_ERROR      anything else, untyped; the class and status are still kept
  LOCAL_PROCESSING_ERROR  the provider RETURNED data and the local merge/detrend raised (e.g. nonfinite samples):
                      not a provider failure; the pre-merge coverage of each returned trace id is kept
A 204 from dataselect cannot by itself tell "no such channel" from "no data in the window". An inventory listing can
only ADVISE (refine_no_data): it stays NO_DATA with a basis scoped to THAT inventory (codex 998f59fa s2), because a
filtered, incomplete or stale listing cannot establish absence outside its own scope. An authoritative
NO_MATCHING_CHANNEL / absence verdict would need a strict receipt (provider, full N.S.L.C selector, time bounds, level,
retrieval time, complete response and digest, proof it covers the questioned selector and window) and is not produced
by this version.
"""
import re

ROUTING_VERSION = "thd-provider-routing-v1"

FDSN = "FDSN"              # obspy.clients.fdsn.Client(<key>).get_waveforms
NIED_HINET = "NIED_HINET"  # NIED Hi-net: registered-access win32 download; not an FDSN service

TYPED_OUTCOMES = ("DATA_RETURNED", "NO_DATA", "AUTH_REQUIRED", "INVALID_SELECTOR", "TRANSPORT_ERROR",
                  "PROVIDER_ERROR", "OPERATOR_REFUSED", "NETWORK_NOT_ROUTED", "LOCAL_PROCESSING_ERROR")

# basis labels: RETAINED = seen in retained run evidence; LADDER = unchanged from the pre-map if/elif ladder;
# EXPECTATION = data-centre knowledge not yet observed from this host (the later bounded live metadata check verifies).
NETWORK_ROUTES = {
    "IU": {"adapters": ((FDSN, "IRIS"),),
           "basis": "RETAINED: 2026-10-03 DATA_RETURNED at IRIS for IU.COLA, IU.COR, IU.TATO, IU.ANTO, IU.MAJO, IU.TUC "
                    "(served d3_daily_station_attempts); IU.SNZO uses the bound operator, not this route"},
    "II": {"adapters": ((FDSN, "IRIS"),), "basis": "LADDER: IRIS only"},
    "CN": {"adapters": ((FDSN, "IRIS"),), "basis": "LADDER: IRIS only"},
    "UW": {"adapters": ((FDSN, "IRIS"),), "basis": "LADDER: IRIS only"},
    "CI": {"adapters": ((FDSN, "SCEDC"), (FDSN, "IRIS")), "basis": "LADDER: SCEDC then IRIS"},
    "BK": {"adapters": ((FDSN, "NCEDC"), (FDSN, "IRIS")),
           "basis": "LADDER: NCEDC then IRIS; RETAINED: 2026-10-03 BK.BKS DATA_RETURNED at NCEDC"},
    "NC": {"adapters": ((FDSN, "NCEDC"), (FDSN, "IRIS")), "basis": "LADDER: NCEDC then IRIS"},
    "GE": {"adapters": ((FDSN, "GEOFON"), (FDSN, "IRIS")),
           "basis": "LADDER: GEOFON then IRIS, without the ladder's GFZ leg (in obspy 1.5.1 GFZ and GEOFON resolve to "
                    "the same https://geofon.gfz.de, so GFZ only repeated the GEOFON request). RETAINED: GEOFON service "
                    "discovery failed for every "
                    "station on 2026-10-03 (grassmann a7e7d947); no configured THD station is GE"},
    "AK": {"adapters": ((FDSN, "IRIS"),),
           "basis": "EXPECTATION: AK waveforms are archived at EarthScope (IRIS); the generic chain's other legs were "
                    "non-hosting (SCEDC/NCEDC 204) or failing discovery (GEOFON). RETAINED: AK.SSL is absent from the "
                    "2026-09-27 EarthScope station listing that asked for it -- a selector question, not a routing one"},
    "MX": {"adapters": ((FDSN, "IRIS"),),
           "basis": "RETAINED: 2026-10-03 MX.TLIG DATA_RETURNED at IRIS, the first leg of the old generic chain"},
    "G": {"adapters": ((FDSN, "IRIS"),),
          "basis": "EXPECTATION: G.UNM has no retained attempt (NOT_ATTEMPTED on 2026-10-03); IRIS was the old chain's "
                   "first leg and lists G.UNM (earthscope.txt 2026-09-27). IPGP, GEOSCOPE's own centre, is a candidate "
                   "for the live metadata check, not routed here"},
    "IV": {"adapters": ((FDSN, "INGV"),),
           "basis": "EXPECTATION: INGV is IV's FDSN data centre (seismic_data.FDSN_CLIENTS already maps IV->INGV; codex "
                    "cites fdsn.org/datacenters/detail/INGV); the old chain never asked it. RETAINED: IV.CAFE is in the "
                    "2026-09-27 INGV station listing (station level only). The retained probe (grassmann 10-03 inventory "
                    "a6af5e32) returned BHZ 404 and HHZ 200 for its requested selector/epoch; routing alone did not "
                    "establish a working BHZ request. HHZ at 100 Hz would be a separate reviewed channel configuration"},
    "HINET": {"adapters": ((NIED_HINET, "NIED"),),
              "basis": "RETAINED: N.KI2H is not an FDSN station code (IRIS HTTP 422 RequestValidationError, 2026-10-03). "
                       "Hi-net access is NIED registration + win32 download with redistribution terms: owner-handled. "
                       "The adapter makes no request and records AUTH_REQUIRED"},
}

_STATUS_TYPED = {204: "NO_DATA", 404: "NO_DATA", 401: "AUTH_REQUIRED", 403: "AUTH_REQUIRED",
                 400: "INVALID_SELECTOR", 422: "INVALID_SELECTOR",
                 429: "TRANSPORT_ERROR", 500: "TRANSPORT_ERROR", 502: "TRANSPORT_ERROR", 503: "TRANSPORT_ERROR",
                 504: "TRANSPORT_ERROR"}
_CLASS_TYPED = {"FDSNNoServiceException": "TRANSPORT_ERROR", "FDSNTimeoutException": "TRANSPORT_ERROR",
                "TimeoutError": "TRANSPORT_ERROR", "timeout": "TRANSPORT_ERROR", "URLError": "TRANSPORT_ERROR",
                "ConnectionError": "TRANSPORT_ERROR", "ConnectionResetError": "TRANSPORT_ERROR",
                "ConnectionRefusedError": "TRANSPORT_ERROR", "RemoteDisconnected": "TRANSPORT_ERROR",
                "IncompleteRead": "TRANSPORT_ERROR"}
# obspy writes a status as "HTTP Status code: N" (subclasses with a status_code) or "Unknown HTTP code: N" (base class)
_STATUS_IN_TEXT = re.compile(r"(?:HTTP Status code|Unknown HTTP code): (\d{3})\b")
_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def route(network):
    """The ordered adapters for `network`, or None: an unmapped network is refused, never sprayed."""
    entry = NETWORK_ROUTES.get(network)
    return None if entry is None else entry["adapters"]


def _typed(class_name, status):
    if status in _STATUS_TYPED:
        return _STATUS_TYPED[status]
    return _CLASS_TYPED.get(class_name, "PROVIDER_ERROR")


def classify_exception(exc):
    """(typed_outcome, exception_class, http_status) for a provider exception, from the exception object."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        found = _STATUS_IN_TEXT.search(str(exc))
        status = int(found.group(1)) if found else None
    name = type(exc).__name__
    return _typed(name, status), name, status


def classify_retained_reason(reason):
    """The same classification from a RETAINED provider reason, i.e. redact('<Class>: <message>') as the fetch writes
    it, so earlier PROVIDER_ERROR records can be typed without re-running anything. None when it is not that shape."""
    if not isinstance(reason, str) or ": " not in reason:
        return None
    name = reason.split(": ", 1)[0]
    if not _CLASS_NAME.fullmatch(name):
        return None
    found = _STATUS_IN_TEXT.search(reason)
    status = int(found.group(1)) if found else None
    return _typed(name, status), name, status


def parse_fdsn_text(text):
    """Rows of an FDSN station-service text response (level=station or level=channel) as dicts keyed by the header."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#"):
        raise ValueError("FDSN_TEXT_HEADER_MISSING")
    header = [h.strip() for h in lines[0].lstrip("#").split("|")]
    rows = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) != len(header):
            raise ValueError("FDSN_TEXT_ROW_WIDTH")
        rows.append(dict(zip(header, cells)))
    return header, rows


def _overlaps(row, start, end):
    begin, finish = row.get("StartTime") or "", row.get("EndTime") or ""
    return (not finish or finish[:19] >= start[:19]) and (not begin or begin[:19] <= end[:19])


def refine_no_data(network, station, channel, window, inventory_text, asked):
    """ADVISORY context for a NO_DATA attempt from a RETAINED station-service response (codex 998f59fa s2).

    The outcome is ALWAYS "NO_DATA": a listing never promotes an attempt to INVALID_SELECTOR or to an absence verdict
    (a syntactically valid selector with no rows is not a malformed request; an HHZ-filtered or time-restricted listing
    cannot exclude BHZ or another epoch). The basis says what THIS inventory shows, scoped to it. `asked` is the
    (networks, stations) the query requested; for a selector it did not ask about it says nothing. Not wired into the
    daily fetch. Returns ("NO_DATA", basis)."""
    nets, stas = asked
    if network not in nets or station not in stas:
        return "NO_DATA", "INVENTORY_DID_NOT_ASK"
    header, rows = parse_fdsn_text(inventory_text)
    same = [r for r in rows if r.get("Network") == network and r.get("Station") == station]
    if not same:
        return "NO_DATA", "NO_MATCHING_STATION_IN_THIS_INVENTORY"
    if "Channel" not in header:
        return "NO_DATA", "CHANNEL_UNVERIFIED_STATION_LEVEL_ONLY"
    start, end = window
    if not any(r.get("Channel") == channel and _overlaps(r, start, end) for r in same):
        return "NO_DATA", "NO_MATCHING_CHANNEL_IN_THIS_INVENTORY"
    return "NO_DATA", "CHANNEL_IN_THIS_INVENTORY_PROVIDER_RETURNED_NO_DATA"


# ---- candidate additions (cayley 2026-10-08; codex 1515 s4, 1523 item 2) ------------------------------------------------
# EXACT SELECTOR. A location is pinned ONLY from retained evidence: the trace id a station's served value actually used
# (d3_daily_station_attempts trace_id), with that evidence named as the basis. Until a pin is bound the request stays
# the historical wildcard and the attempt says so -- a guessed location could silently select another sensor (MAJO
# has three co-located instruments). Populated from grassmann's retained evidence (3227c6a6): the 8 stations with a
# served VALUE trace id in scored 10-03..10-06; AK.BMR, AK.SSL, G.UNM, HINET.N.KI2H and IV.CAFE had no served value and
# stay wildcard; IU.SNZO is fetched by its bound operator, which fixes its own selector.
_LOCATION = re.compile(r"[0-9A-Z]{0,2}")
STATION_LOCATIONS = {   # "NET.STA" -> {"location", "basis"}; generated from retained evidence, never re-typed
    'BK.BKS': {"location": '00',
               "basis": 'RETAINED: served trace id BK.BKS.00.BHZ on scored 10-03 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'IU.ANTO': {"location": '00',
               "basis": 'RETAINED: served trace id IU.ANTO.00.BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'IU.COLA': {"location": '00',
               "basis": 'RETAINED: served trace id IU.COLA.00.BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'IU.COR': {"location": '00',
               "basis": 'RETAINED: served trace id IU.COR.00.BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'IU.MAJO': {"location": '00',
               "basis": 'RETAINED: served trace id IU.MAJO.00.BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'IU.TATO': {"location": '00',
               "basis": 'RETAINED: served trace id IU.TATO.00.BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'IU.TUC': {"location": '00',
               "basis": 'RETAINED: served trace id IU.TUC.00.BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
    'MX.TLIG': {"location": '',
               "basis": 'RETAINED: served trace id MX.TLIG..BHZ on scored 10-03,10-04,10-05,10-06 (view-2026-10-06 d3_daily_station_attempts; grassmann 3227c6a6 station_pins.json sha256 ef2d6c22401c0d17)'},
}


def selector_for(network, station):
    """(location, selector_basis): a retained pin, or ("*", "WILDCARD_LOCATION_NOT_PINNED")."""
    pin = STATION_LOCATIONS.get("%s.%s" % (network, station))
    if pin is None:
        return "*", "WILDCARD_LOCATION_NOT_PINNED"
    location, basis = pin.get("location"), pin.get("basis")
    if not (isinstance(location, str) and _LOCATION.fullmatch(location) and isinstance(basis, str)
            and basis.startswith("RETAINED: ")):
        raise ValueError("STATION_LOCATION_PIN_INVALID: %s.%s" % (network, station))
    return location, "PINNED " + basis


COVERAGE_VERSION = "thd-coverage-facts-v2"
_COVERAGE_EPS = 1e-6   # seconds: float noise when joining touching intervals, far below any sample period


def _union(intervals):
    """Sorted union of half-open [a, b) intervals in seconds; intervals touching within _COVERAGE_EPS are joined."""
    merged = []
    for a, b in sorted(i for i in intervals if i[1] - i[0] > _COVERAGE_EPS):
        if merged and a <= merged[-1][1] + _COVERAGE_EPS:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def coverage_facts(stream, start, end):
    """Honest missingness of a returned stream BEFORE it is merged, per exact trace id (codex 1614 finding 4). FACTS
    ONLY: the value path (merge(method=1, fill_value='interpolate'), demean, linear detrend) is unchanged; whether a
    partial day may be scored is a separate, reviewed rule.

    Definitions (thd-coverage-facts-v2). The request is half-open [start, end). Sample i of a trace stands for
    [t_i, t_i + delta). A trace's TEMPORAL EXTENT is [starttime, starttime + npts * delta); its SAMPLE SUPPORT is the
    part of that extent whose samples are finite and unmasked. Both are clipped to the request and measured as the
    UNION over the id's traces, so data outside the request, duplicates and overlaps never add or remove coverage.
      covered_seconds / coverage_fraction   sample support (availability), not extent
      extent_seconds                        temporal extent; nonfinite_or_masked_seconds = extent without support
      gaps / gap_seconds                    holes BETWEEN traces inside the request longer than half a sample: what
                                            the merge interpolates (fill)
      missing_head_seconds / _tail_         request before the first / after the last returned sample: NOT filled
                                            by the merge (edge_fill); None when nothing falls in the request
      overlaps / overlap_seconds            traces claiming the same instants: merge method 1 keeps the later trace
                                            (overlap_resolution), which is not gap interpolation
    FULL needs no gap, no nonfinite or masked support, and each end within one sample period of the request;
    NO_SAMPLES_IN_REQUEST when no usable sample falls inside it; otherwise PARTIAL. An empty stream yields {}."""
    import numpy as np
    from obspy import UTCDateTime
    t0, t1 = UTCDateTime(start), UTCDateTime(end)
    requested = float(t1 - t0)
    if not requested > 0:
        raise ValueError("COVERAGE_REQUEST_BOUNDS_NOT_ORDERED")
    by_id = {}
    for trace in stream:
        by_id.setdefault(trace.id, []).append(trace)
    facts = {}
    for trace_id, traces in by_id.items():
        delta = max(float(tr.stats.delta) for tr in traces)
        extents, support = [], []
        for tr in traces:
            rel, step = float(tr.stats.starttime - t0), float(tr.stats.delta)
            values = np.ma.getdata(tr.data)
            extents.append((max(0.0, rel), min(requested, rel + len(values) * step)))
            valid = np.isfinite(values) & ~np.ma.getmaskarray(tr.data)
            edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
            for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                support.append((max(0.0, rel + int(first) * step), min(requested, rel + int(stop) * step)))
        extent = _union(extents)
        covered = sum(b - a for a, b in _union(support))
        extent_seconds = sum(b - a for a, b in extent)
        holes = [b[0] - a[1] for a, b in zip(extent, extent[1:])]
        gap_holes = [h for h in holes if h > 0.5 * delta]
        head = extent[0][0] if extent else None
        tail = requested - extent[-1][1] if extent else None
        clipped = sorted(i for i in extents if i[1] - i[0] > _COVERAGE_EPS)
        overlaps, reach = 0, None
        for a, b in clipped:
            if reach is not None and a < reach - _COVERAGE_EPS:
                overlaps += 1
            reach = b if reach is None else max(reach, b)
        overlap_seconds = sum(b - a for a, b in clipped) - extent_seconds
        unusable = max(0.0, extent_seconds - covered)
        edges_missing = bool(extent) and (head > delta + _COVERAGE_EPS or tail > delta + _COVERAGE_EPS)
        if covered <= _COVERAGE_EPS:
            status = "NO_SAMPLES_IN_REQUEST"
        elif gap_holes or edges_missing or unusable > _COVERAGE_EPS:
            status = "PARTIAL"
        else:
            status = "FULL"

        def r(x):
            return None if x is None else round(x, 6)
        facts[trace_id] = {"coverage_version": COVERAGE_VERSION, "requested_seconds": r(requested),
                           "covered_seconds": r(covered), "coverage_fraction": round(covered / requested, 6),
                           "extent_seconds": r(extent_seconds), "nonfinite_or_masked_seconds": r(unusable),
                           "gaps": len(gap_holes), "gap_seconds": r(sum(gap_holes)),
                           "overlaps": overlaps, "overlap_seconds": r(max(0.0, overlap_seconds)),
                           "missing_head_seconds": r(head), "missing_tail_seconds": r(tail),
                           "sample_period_seconds": r(delta), "status": status,
                           "fill": "MERGE_INTERPOLATE_UNCHANGED" if gap_holes else "NONE",
                           "overlap_resolution": "MERGE_METHOD_1_LATER_TRACE_KEPT_UNCHANGED" if overlaps else "NONE",
                           "edge_fill": "NOT_FILLED" if edges_missing else "NONE",
                           "nonfinite_or_masked_handling": "UNCHANGED_IN_VALUE_PATH" if unusable > _COVERAGE_EPS
                           else "NONE"}
    return facts


_INSTANT = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})([.][0-9]{1,9})?"
                      r"(Z|[+-][0-9]{2}:[0-9]{2})?")


def utc_instant_ns(value):
    """Exact integer UTC nanoseconds for an ISO-8601 instant (codex 1614 finding 3). Up to 9 fractional digits are
    kept and more are refused, never truncated; a Z or +HH:MM / -HH:MM offset is applied; an OFFSET-FREE instant is
    read as UTC, the FDSN StationXML convention. Anything else (a date alone, month 13, hour 24, an offset beyond
    14:00, a non-string) is refused with ValueError."""
    import calendar
    from datetime import datetime
    match = _INSTANT.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("INSTANT_NOT_ISO8601: %r" % (value,))
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    try:
        whole = calendar.timegm(datetime(year, month, day, hour, minute, second).timetuple())
    except ValueError:
        raise ValueError("INSTANT_NOT_A_CALENDAR_TIME: %r" % (value,)) from None
    fraction, zone = match.group(7), match.group(8)
    nanos = int((fraction[1:] + "000000000")[:9]) if fraction else 0
    offset = 0
    if zone and zone != "Z":
        hours, minutes = int(zone[1:3]), int(zone[4:6])
        if hours > 14 or minutes > 59 or (hours == 14 and minutes):
            raise ValueError("INSTANT_OFFSET_OUT_OF_RANGE: %r" % (value,))
        offset = (hours * 3600 + minutes * 60) * (1 if zone[0] == "+" else -1)
    return (whole - offset) * 1_000_000_000 + nanos


def response_epoch_status(epochs, start, end):
    """Whether one RETAINED response epoch covers the whole window. `epochs` are (start, end_or_None, label) ISO
    instants from a retained StationXML (level=response) for the exact selector; this helper reads nothing itself.
    Instants are compared exactly by utc_instant_ns. Window and epochs are HALF-OPEN [start, end): an epoch ending
    exactly at the window start, or starting exactly at its end, does not overlap it; an end of None (or a missing
    start) is unbounded. Unordered bounds are refused (ValueError), never reordered.
      ("SINGLE_EPOCH", label)                     one epoch spans the window
      ("EPOCH_PARTIALLY_COVERS_WINDOW", label)    one epoch overlaps; part of the window has no retained epoch
      ("CROSSES_EPOCH_BOUNDARY", [labels])        sequential epochs with a boundary inside the window. Epoch metadata
                                                  is not response removal: whether the response DIFFERS is not assessed
      ("OVERLAPPING_EPOCH_METADATA", [labels])    two retained epochs claim the same instant in the window: a metadata
                                                  conflict, not a proven response change
      ("NO_COVERING_EPOCH", None)                 no retained epoch overlaps the window (wrong epoch / stale metadata)
    Labels are ordered by epoch start. DIAGNOSTIC ONLY: nothing admits or refuses a value on this status; any such
    policy is a separate, named rule. Wiring it into the daily fetch needs the retained response table bound to the
    method (a named prerequisite)."""
    ws, we = utc_instant_ns(start), utc_instant_ns(end)
    if we <= ws:
        raise ValueError("RESPONSE_WINDOW_BOUNDS_NOT_ORDERED")
    spans = []
    for begin, finish, label in epochs:
        b = None if begin is None else utc_instant_ns(begin)
        f = None if finish is None else utc_instant_ns(finish)
        if b is not None and f is not None and f <= b:
            raise ValueError("RESPONSE_EPOCH_BOUNDS_NOT_ORDERED: %r" % (label,))
        spans.append((b, f, label))
    overlapping = sorted(((b, f, label) for b, f, label in spans if (b is None or b < we) and (f is None or f > ws)),
                         key=lambda span: (span[0] is not None, span[0] or 0))
    if not overlapping:
        return "NO_COVERING_EPOCH", None
    if len(overlapping) == 1:
        b, f, label = overlapping[0]
        whole = (b is None or b <= ws) and (f is None or f >= we)
        return ("SINGLE_EPOCH" if whole else "EPOCH_PARTIALLY_COVERS_WINDOW"), label
    labels, reach = [label for _, _, label in overlapping], overlapping[0][1]
    for b, f, _ in overlapping[1:]:
        if reach is None or b is None or b < reach:
            return "OVERLAPPING_EPOCH_METADATA", labels
        reach = None if f is None else max(reach, f)
    return "CROSSES_EPOCH_BOUNDARY", labels
