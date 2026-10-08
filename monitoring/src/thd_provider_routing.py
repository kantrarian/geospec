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
                  "PROVIDER_ERROR", "OPERATOR_REFUSED", "NETWORK_NOT_ROUTED")

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
# has three co-located instruments). Empty today: the retained trace ids are on the production host (grassmann).
_LOCATION = re.compile(r"[0-9A-Z]{0,2}")
STATION_LOCATIONS = {}   # "NET.STA" -> {"location": "00", "basis": "RETAINED: <evidence>"}


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


def coverage_facts(stream, start, end):
    """Honest missingness of a returned stream BEFORE it is merged, per trace id: the requested span, how much of it
    the traces' samples actually cover, and the gaps/overlaps the merge will fill or resolve. FACTS ONLY -- the value
    path (merge with interpolation, detrend) is unchanged by this version; whether a partial day may be scored is a
    separate, reviewed rule. A day is FULL only with no gap and both ends within one sample of the request."""
    from obspy import UTCDateTime
    t0, t1 = UTCDateTime(start), UTCDateTime(end)
    requested = float(t1 - t0)
    by_id = {}
    for trace in stream:
        by_id.setdefault(trace.id, []).append(trace)
    gaps = stream.get_gaps()  # [net, sta, loc, cha, t_from, t_to, duration, n_samples]; duration < 0 is an overlap
    facts = {}
    for trace_id, traces in by_id.items():
        delta = max(float(tr.stats.delta) for tr in traces)
        mine = [g for g in gaps if "%s.%s.%s.%s" % tuple(g[:4]) == trace_id]
        gap_seconds = sum(float(g[6]) for g in mine if g[6] > 0)
        head = max(0.0, float(min(tr.stats.starttime for tr in traces) - t0))
        tail = max(0.0, float(t1 - max(tr.stats.endtime for tr in traces)) - delta)
        covered = max(0.0, requested - head - tail - gap_seconds)
        full = gap_seconds == 0 and head <= delta and tail <= delta
        facts[trace_id] = {"requested_seconds": round(requested, 3), "covered_seconds": round(covered, 3),
                           "coverage_fraction": round(covered / requested, 6) if requested > 0 else 0.0,
                           "gaps": sum(1 for g in mine if g[6] > 0), "gap_seconds": round(gap_seconds, 3),
                           "overlaps": sum(1 for g in mine if g[6] < 0), "missing_head_seconds": round(head, 3),
                           "missing_tail_seconds": round(tail, 3), "status": "FULL" if full else "PARTIAL",
                           "fill": "NONE" if full else "MERGE_INTERPOLATE_UNCHANGED"}
    return facts


def response_epoch_status(epochs, start, end):
    """Whether one RETAINED response epoch covers the whole window. `epochs` are (start_iso, end_iso_or_None, label)
    from a retained StationXML (level=response) for the exact selector; this helper reads nothing itself.
      ("SINGLE_EPOCH", label)                one epoch spans the window
      ("CROSSES_EPOCH_BOUNDARY", [labels])   the window straddles a response change: not one instrument state
      ("EPOCH_PARTIALLY_COVERS_WINDOW", label)  one epoch overlaps but part of the window has no known response
      ("NO_COVERING_EPOCH", None)            no retained epoch overlaps the window (wrong epoch / stale metadata)
    Wiring this into the daily fetch needs the retained response table bound to the method (a named prerequisite)."""
    s, e = start[:19], end[:19]
    overlapping = [(b, f, label) for b, f, label in epochs if (not f or f[:19] >= s) and (not b or b[:19] <= e)]
    if not overlapping:
        return "NO_COVERING_EPOCH", None
    whole = [label for b, f, label in overlapping if (not b or b[:19] <= s) and (not f or f[:19] >= e)]
    if len(overlapping) == 1:
        return ("SINGLE_EPOCH", whole[0]) if whole else ("EPOCH_PARTIALLY_COVERS_WINDOW", overlapping[0][2])
    return "CROSSES_EPOCH_BOUNDARY", [label for _, _, label in overlapping]
