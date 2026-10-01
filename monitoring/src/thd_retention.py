"""
thd_retention.py -- PROSPECTIVE per-fetch retention record for the seismic_thd acquisition (thd-acquisition-retention-v2).

NOT WIRED. Nothing in the daily runner imports this module; seismic_thd.fetch_continuous_data_for_thd is unchanged.
It defines what a FUTURE THD fetch must retain so a value can later be traced to its exact input, and validators that
refuse -- with a typed code, never a zero -- when the retained facts cannot support a measurement.

Why (codex THD_CONTROLS_RULING_CODEX_20261001 section 4; grassmann 6367f3bb inventory): the current path requests
location '*', merges with linear interpolation and takes st[0]; it retains no raw bytes or hash, no selected
location, no response, no gap or filled-sample count and no actual valid coverage. Past values cannot be replayed
exactly. This module makes those facts retainable going forward; it does not reconstruct any past value.

v2 (codex a2a5cfe0 repairs 2 and 3): coverage is the UNION of distinct sample indices inside the measurement window
on the declared sample grid (duplicates counted separately, never twice); the measurement window must be ordered
and inside the request; every segment must be internally consistent (finite positive rate, integer sample count,
count matching its span); request patterns use a real case-sensitive wildcard grammar; non-finite values are
refused; and admission re-opens the retained objects through a caller-supplied trusted adapter, so a self-hashed
record is never taken as proof that its bytes decode to the segments it names.

Trust boundary, stated plainly. `validate` checks that a RECORD is coherent. `verify_retained_objects` checks that
the retained raw and response objects exist, hash to the record, and that the raw object DECODES (through the
adapter the caller supplies, e.g. an ObsPy reader in production) to exactly the recorded segments. What remains
trusted is that adapter. Neither function says whether the measurement is physically meaningful: tidal-band
sensitivity stays NOT_ASSESSED and is a separate assessment.

Refusal codes (typed; first token of every reason):
  RECORD_MALFORMED          digest mismatch or unreadable record
  MEASUREMENT_WINDOW_INVALID measurement window reversed, empty or outside the request
  NSLC_MALFORMED            a returned NSLC is not four literal parts of the supported grammar
  AMBIGUOUS_LOCATION        more than one NSLC returned (st[0] would be an arbitrary choice) or none
  NSLC_MISMATCH             the returned channel does not match the request pattern, or the response is for another
  SEGMENT_INCONSISTENT      non-finite / non-positive rate, non-integer count, or a count that disagrees with its span
  RATE_INCONSISTENT         segments disagree on the sampling rate
  GRID_MISALIGNED           a segment's samples do not fall on the measurement window's sample grid
  RAW_NOT_RETAINED / RESPONSE_NOT_RETAINED   the record lacks the object hash
  RESPONSE_EPOCH_MISMATCH   no response epoch of the selected channel covers the whole measurement window
  UNITS_NOT_STATED          input units (from the response) or preprocessing output units absent
  INADEQUATE_VALID_COVERAGE distinct in-window samples below the caller-declared fraction
  VALUE_INVALID / NONFINITE_VALUE  the THD value is not a finite real number (bool is not a number here)
  RAW_OBJECT_MISSING / RAW_OBJECT_CORRUPT / RAW_UNDECODABLE / DECODED_IDENTITY_MISMATCH
  RESPONSE_OBJECT_MISSING / RESPONSE_OBJECT_CORRUPT / RESPONSE_REPARSE_MISMATCH
The minimum coverage fraction is a REQUIRED argument with no default (an unstated threshold is a guess).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import numbers
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

RETENTION_SCHEMA = "thd-acquisition-retention-v2"
STATIONXML_NS = "http://www.fdsn.org/xml/station/1"
SENSITIVITY_NOT_ASSESSED = "NOT_ASSESSED"
# A segment whose first sample lies more than this fraction of a sample period off the measurement grid is not on
# that grid. Declared here (not a caller choice) because it is a property of the grid mapping, not of a study.
GRID_TOLERANCE_SAMPLES = 0.01

ACCEPTED = "RETENTION_COMPLETE"
OBJECTS_VERIFIED = "RETAINED_OBJECTS_VERIFIED"
ADMITTED = "VALUE_ADMITTED"
RECORD_MALFORMED = "RECORD_MALFORMED"
MEASUREMENT_WINDOW_INVALID = "MEASUREMENT_WINDOW_INVALID"
NSLC_MALFORMED = "NSLC_MALFORMED"
AMBIGUOUS_LOCATION = "AMBIGUOUS_LOCATION"
NSLC_MISMATCH = "NSLC_MISMATCH"
SEGMENT_INCONSISTENT = "SEGMENT_INCONSISTENT"
RATE_INCONSISTENT = "RATE_INCONSISTENT"
GRID_MISALIGNED = "GRID_MISALIGNED"
RAW_NOT_RETAINED = "RAW_NOT_RETAINED"
RESPONSE_NOT_RETAINED = "RESPONSE_NOT_RETAINED"
RESPONSE_EPOCH_MISMATCH = "RESPONSE_EPOCH_MISMATCH"
UNITS_NOT_STATED = "UNITS_NOT_STATED"
INADEQUATE_VALID_COVERAGE = "INADEQUATE_VALID_COVERAGE"
VALUE_INVALID = "VALUE_INVALID"
NONFINITE_VALUE = "NONFINITE_VALUE"
RAW_OBJECT_MISSING = "RAW_OBJECT_MISSING"
RAW_OBJECT_CORRUPT = "RAW_OBJECT_CORRUPT"
RAW_UNDECODABLE = "RAW_UNDECODABLE"
DECODED_IDENTITY_MISMATCH = "DECODED_IDENTITY_MISMATCH"
RESPONSE_OBJECT_MISSING = "RESPONSE_OBJECT_MISSING"
RESPONSE_OBJECT_CORRUPT = "RESPONSE_OBJECT_CORRUPT"
RESPONSE_REPARSE_MISMATCH = "RESPONSE_REPARSE_MISMATCH"

# Supported NSLC grammar: four dot-separated parts. Literal parts use letters, digits and '-'; a request PATTERN may
# also use '*' and '?' (fnmatch semantics, case-sensitive). Network, station and channel are non-empty; an empty
# location is written '' in returned data and may be requested as '' or '--'.
_LITERAL_PART = re.compile(r"^[A-Za-z0-9-]*$")
_PATTERN_PART = re.compile(r"^[A-Za-z0-9*?-]*$")


class RetentionError(ValueError):
    """A defect in the caller's inputs (not a measurement refusal)."""


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    code: str
    detail: str
    facts: Dict = field(default_factory=dict, compare=False)

    @property
    def reason(self) -> str:
        return "%s: %s" % (self.code, self.detail) if self.detail else self.code


def _refuse(code, detail, **facts):
    return Verdict(False, code, detail, dict(facts))


# ---------------------------------------------------------------------------------------------------------------
# Times and identities
# ---------------------------------------------------------------------------------------------------------------

def _utc(value) -> datetime:
    """An aware UTC datetime from an ISO string ('Z' or offset) or an aware datetime. Naive input is refused."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise RetentionError("unreadable time %r" % (value,)) from exc
    else:
        raise RetentionError("time must be an ISO string or datetime, got %r" % (type(value).__name__,))
    if dt.tzinfo is None:
        raise RetentionError("time %r has no UTC offset" % (value,))
    return dt.astimezone(timezone.utc)


def _station_time(text: Optional[str]) -> Optional[datetime]:
    """A StationXML dateTime (UTC by the FDSN specification; the suffix may be absent) as an aware UTC datetime.
    Fractional seconds beyond microseconds are truncated. None/empty -> None."""
    if not text:
        return None
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1]
    zone = "+00:00"
    for marker in ("+", "-"):
        idx = t.rfind(marker)
        if idx > 10:
            t, zone = t[:idx], t[idx:]
            break
    if "." in t:
        head, frac = t.split(".", 1)
        t = head + "." + (frac + "000000")[:6]
    return _utc(t + zone)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parts(nslc: str) -> Optional[List[str]]:
    parts = nslc.split(".") if isinstance(nslc, str) else []
    return parts if len(parts) == 4 else None


def literal_nslc_ok(nslc: str) -> bool:
    parts = _parts(nslc)
    return bool(parts) and all(_LITERAL_PART.match(p) for p in parts) and all(parts[i] for i in (0, 1, 3))


def pattern_nslc_ok(pattern: str) -> bool:
    parts = _parts(pattern)
    return bool(parts) and all(_PATTERN_PART.match(p) for p in parts) and all(parts[i] for i in (0, 1, 3))


def nslc_matches(pattern: str, nslc: str) -> bool:
    """Case-sensitive fnmatch per part; a requested '--' location matches the empty location."""
    p_parts, n_parts = _parts(pattern), _parts(nslc)
    if not p_parts or not n_parts:
        return False
    if p_parts[2] == "--":
        p_parts = p_parts[:2] + [""] + p_parts[3:]
    return all(fnmatch.fnmatchcase(n, p) for p, n in zip(p_parts, n_parts))


def _nslc(d: dict) -> str:
    return "%s.%s.%s.%s" % (d["network"], d["station"], d.get("location", ""), d["channel"])


# ---------------------------------------------------------------------------------------------------------------
# StationXML (level=response): parsed from the retained document itself
# ---------------------------------------------------------------------------------------------------------------

def response_channels(xml_bytes: bytes) -> List[dict]:
    """Every Channel epoch in an FDSN StationXML 1.x document: NSLC, epoch start/end (end None = open), whether a
    Response element is present, the InstrumentSensitivity value / frequency and its input / output unit names.
    Raises RetentionError on a document that is not StationXML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RetentionError("response document is not XML: %s" % exc) from exc
    ns = {"s": STATIONXML_NS}
    if root.tag != "{%s}FDSNStationXML" % STATIONXML_NS:
        raise RetentionError("not an FDSN StationXML document (root %s)" % root.tag)
    out = []
    for net in root.findall("s:Network", ns):
        for sta in net.findall("s:Station", ns):
            for cha in sta.findall("s:Channel", ns):
                resp = cha.find("s:Response", ns)
                sens = resp.find("s:InstrumentSensitivity", ns) if resp is not None else None

                def text(path, node=sens):
                    el = node.find(path, ns) if node is not None else None
                    return el.text.strip() if el is not None and el.text else None

                start = _station_time(cha.get("startDate"))
                if start is None:
                    raise RetentionError("Channel %s has no startDate" % cha.get("code"))
                end = _station_time(cha.get("endDate"))
                out.append(dict(
                    nslc="%s.%s.%s.%s" % (net.get("code"), sta.get("code"), cha.get("locationCode", ""), cha.get("code")),
                    start=_iso(start),
                    end=None if end is None else _iso(end),
                    has_response=resp is not None,
                    sensitivity_value=float(text("s:Value")) if text("s:Value") else None,
                    sensitivity_frequency_hz=float(text("s:Frequency")) if text("s:Frequency") else None,
                    input_units=text("s:InputUnits/s:Name"),
                    output_units=text("s:OutputUnits/s:Name")))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Segments and coverage
# ---------------------------------------------------------------------------------------------------------------

def _normalize_segment(s: dict) -> dict:
    return dict(nslc=_nslc(s), starttime_utc=_iso(_utc(s["starttime_utc"])), endtime_utc=_iso(_utc(s["endtime_utc"])),
                sampling_rate_hz=s["sampling_rate_hz"], npts=s["npts"])


def _segment_problem(seg: dict) -> Optional[str]:
    rate, npts = seg["sampling_rate_hz"], seg["npts"]
    if isinstance(rate, bool) or not isinstance(rate, numbers.Real) or not math.isfinite(rate) or rate <= 0:
        return "%s: sampling rate %r is not a finite positive number" % (seg["nslc"], rate)
    if isinstance(npts, bool) or not isinstance(npts, numbers.Integral) or npts < 1:
        return "%s: npts %r is not a positive integer" % (seg["nslc"], npts)
    start, end = _utc(seg["starttime_utc"]), _utc(seg["endtime_utc"])
    if end < start:
        return "%s: segment ends before it starts" % seg["nslc"]
    implied = round((end - start).total_seconds() * float(rate)) + 1
    if int(npts) != implied:
        return "%s: npts %d disagrees with its span %s..%s at %g Hz (implies %d)" % (
            seg["nslc"], int(npts), seg["starttime_utc"], seg["endtime_utc"], float(rate), implied)
    return None


def coverage(segments: Sequence[dict], rate: float, window_start: datetime, window_end: datetime) -> dict:
    """Distinct sample indices on the window's grid (index 0 = window_start, step 1/rate), as a union of
    intervals; duplicates and out-of-window samples counted separately. Raises RetentionError(GRID_MISALIGNED)."""
    n_expected = int(round((window_end - window_start).total_seconds() * rate)) + 1
    intervals, inside_total, outside = [], 0, 0
    for seg in segments:
        k0 = (_utc(seg["starttime_utc"]) - window_start).total_seconds() * rate
        first = int(round(k0))
        if abs(k0 - first) > GRID_TOLERANCE_SAMPLES:
            raise RetentionError("%s: %s first sample is %.4f samples off the window grid"
                                 % (GRID_MISALIGNED, seg["nslc"], k0 - first))
        last = first + int(seg["npts"]) - 1
        lo, hi = max(first, 0), min(last, n_expected - 1)
        inside = max(0, hi - lo + 1)
        inside_total += inside
        outside += int(seg["npts"]) - inside
        if inside:
            intervals.append((lo, hi))
    intervals.sort()
    distinct, cur_lo, cur_hi = 0, None, None
    for lo, hi in intervals:
        if cur_hi is None or lo > cur_hi + 1:
            if cur_hi is not None:
                distinct += cur_hi - cur_lo + 1
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        distinct += cur_hi - cur_lo + 1
    return dict(expected=n_expected, distinct=distinct, duplicate=inside_total - distinct, outside_window=outside)


# ---------------------------------------------------------------------------------------------------------------
# Building the record
# ---------------------------------------------------------------------------------------------------------------

def build_record(*, request: dict, provider: dict, segments: Sequence[dict], raw_bytes: bytes,
                 response_xml: Optional[bytes], preprocessing_version: str, output_units: str) -> dict:
    """Assemble the retention record from what the fetch actually saw. Every argument is required (keyword-only,
    no defaults). `segments` are the traces as returned, BEFORE any merge; they are recorded as given and judged by
    `validate`, which is where an inconsistent segment is refused."""
    for key in ("network", "station", "location", "channel", "starttime_utc", "endtime_utc"):
        if key not in request:
            raise RetentionError("request lacks %r" % key)
    pattern = _nslc(request)
    if not pattern_nslc_ok(pattern):
        raise RetentionError("request NSLC %r is outside the supported grammar" % pattern)
    start, end = _utc(request["starttime_utc"]), _utc(request["endtime_utc"])
    if end <= start:
        raise RetentionError("request window is empty or reversed")
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise RetentionError("raw_bytes must be the exact bytes returned")
    record = dict(
        schema=RETENTION_SCHEMA,
        request=dict(nslc=pattern, starttime_utc=_iso(start), endtime_utc=_iso(end)),
        provider=dict(name=provider.get("name"), service_url=provider.get("service_url")),
        returned=[_normalize_segment(s) for s in segments],
        raw=dict(sha256=hashlib.sha256(bytes(raw_bytes)).hexdigest(), n_bytes=len(raw_bytes)),
        response=None if response_xml is None else dict(
            format="StationXML", level="response", sha256=hashlib.sha256(response_xml).hexdigest(),
            n_bytes=len(response_xml), channels=response_channels(response_xml)),
        preprocessing=dict(version=preprocessing_version, output_units=output_units),
        sensitivity_at_tidal_frequencies=SENSITIVITY_NOT_ASSESSED,
    )
    try:
        record["record_sha256"] = record_digest(record)
    except ValueError as exc:   # e.g. a NaN / infinite rate cannot enter a canonical record
        raise RetentionError("record holds a non-finite number: %s" % exc) from exc
    return record


def canonical(record: dict) -> bytes:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_digest(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Validation of the record
# ---------------------------------------------------------------------------------------------------------------

def _declared_fraction(min_valid_fraction) -> float:
    if isinstance(min_valid_fraction, bool) or not isinstance(min_valid_fraction, numbers.Real) \
            or not math.isfinite(min_valid_fraction) or not 0.0 < float(min_valid_fraction) <= 1.0:
        raise RetentionError("min_valid_fraction must be a declared number in (0, 1], got %r" % (min_valid_fraction,))
    return float(min_valid_fraction)


def validate(record: dict, *, min_valid_fraction: float, measurement_start, measurement_end) -> Verdict:
    """Typed verdict on one retained fetch RECORD for the measurement window the value claims to describe.
    Coherence only: see verify_retained_objects for the objects themselves."""
    fraction_floor = _declared_fraction(min_valid_fraction)
    try:
        if record.get("record_sha256") != record_digest(record):
            return _refuse(RECORD_MALFORMED, "record digest does not match its content")
        req_start, req_end = _utc(record["request"]["starttime_utc"]), _utc(record["request"]["endtime_utc"])
        m_start, m_end = _utc(measurement_start), _utc(measurement_end)
        if not (m_start < m_end and req_start <= m_start and m_end <= req_end):
            return _refuse(MEASUREMENT_WINDOW_INVALID, "measurement %s..%s is not an ordered window inside the request "
                           "%s..%s" % (_iso(m_start), _iso(m_end), _iso(req_start), _iso(req_end)))
        returned = record["returned"]
        malformed = sorted({r["nslc"] for r in returned if not literal_nslc_ok(r["nslc"])})
        if malformed:
            return _refuse(NSLC_MALFORMED, "returned %s outside the supported literal grammar" % malformed)
        nslcs = sorted({r["nslc"] for r in returned})
        if len(nslcs) != 1:
            return _refuse(AMBIGUOUS_LOCATION, "returned channels %s (requested %s)"
                           % (nslcs or "none", record["request"]["nslc"]))
        selected = nslcs[0]
        if not nslc_matches(record["request"]["nslc"], selected):
            return _refuse(NSLC_MISMATCH, "returned %s does not match requested %s" % (selected, record["request"]["nslc"]))
        for seg in returned:
            problem = _segment_problem(seg)
            if problem:
                return _refuse(SEGMENT_INCONSISTENT, problem)
        rates = sorted({float(s["sampling_rate_hz"]) for s in returned})
        if len(rates) != 1:
            return _refuse(RATE_INCONSISTENT, "sampling rates %s" % rates)
        raw = record.get("raw") or {}
        if not raw.get("sha256") or isinstance(raw.get("n_bytes"), bool) or not isinstance(raw.get("n_bytes"), int) \
                or raw["n_bytes"] <= 0:
            return _refuse(RAW_NOT_RETAINED, "raw sha256/length absent or empty")
        response = record.get("response")
        if not response or not response.get("sha256"):
            return _refuse(RESPONSE_NOT_RETAINED, "no level=response document for %s" % selected)
        epochs = [c for c in response["channels"] if c["nslc"] == selected]
        if not epochs:
            others = sorted({c["nslc"] for c in response["channels"]})
            return _refuse(NSLC_MISMATCH, "response describes %s, not %s" % (others, selected))
        covering = [c for c in epochs if c["has_response"] and _utc(c["start"]) <= m_start
                    and (c["end"] is None or _utc(c["end"]) >= m_end)]
        if not covering:
            spans = ["%s..%s" % (c["start"], c["end"] or "open") for c in epochs]
            return _refuse(RESPONSE_EPOCH_MISMATCH, "no %s response epoch covers %s..%s (epochs %s)"
                           % (selected, _iso(m_start), _iso(m_end), spans))
        if not covering[0].get("input_units") or not record["preprocessing"].get("output_units"):
            return _refuse(UNITS_NOT_STATED, "input %r, output %r" % (covering[0].get("input_units"),
                                                                       record["preprocessing"].get("output_units")))
        try:
            cov = coverage(returned, rates[0], m_start, m_end)
        except RetentionError as exc:
            if str(exc).startswith(GRID_MISALIGNED):
                return _refuse(GRID_MISALIGNED, str(exc)[len(GRID_MISALIGNED) + 2:])
            raise
        achieved = cov["distinct"] / cov["expected"]
        if achieved < fraction_floor:
            return _refuse(INADEQUATE_VALID_COVERAGE, "%d distinct of %d samples in %s..%s (%.6f < declared %.6f); "
                           "%d duplicate, %d outside the window" % (cov["distinct"], cov["expected"], _iso(m_start),
                                                                    _iso(m_end), achieved, fraction_floor,
                                                                    cov["duplicate"], cov["outside_window"]), **cov)
        return Verdict(True, ACCEPTED, "%s, %d/%d distinct samples (%d duplicate), response epoch %s..%s; tidal-band "
                       "sensitivity %s" % (selected, cov["distinct"], cov["expected"], cov["duplicate"],
                                           covering[0]["start"], covering[0]["end"] or "open",
                                           record["sensitivity_at_tidal_frequencies"]), dict(cov, nslc=selected))
    except (KeyError, TypeError, RetentionError) as exc:
        return _refuse(RECORD_MALFORMED, "%s: %s" % (type(exc).__name__, exc))


# ---------------------------------------------------------------------------------------------------------------
# The retained objects themselves (trusted adapter boundary)
# ---------------------------------------------------------------------------------------------------------------

def _segment_identity(seg: dict) -> Tuple:
    return (seg["nslc"], _iso(_utc(seg["starttime_utc"])), _iso(_utc(seg["endtime_utc"])),
            float(seg["sampling_rate_hz"]), int(seg["npts"]))


def verify_retained_objects(record: dict, *, load_object: Callable[[str], Optional[bytes]],
                            decode_waveform: Callable[[bytes], Sequence[dict]]) -> Verdict:
    """Re-open the retained objects: the raw object must exist, match the recorded sha256 and length, and DECODE
    (through `decode_waveform`, the trusted adapter -- ObsPy in production) to exactly the recorded segments; the
    response object must exist, match its hash, and re-parse to the recorded channel epochs. `load_object(sha256)`
    returns the stored bytes or None."""
    raw_meta = record.get("raw") or {}
    blob = load_object(raw_meta.get("sha256", ""))
    if blob is None:
        return _refuse(RAW_OBJECT_MISSING, "no retained object %s" % raw_meta.get("sha256"))
    if hashlib.sha256(blob).hexdigest() != raw_meta.get("sha256") or len(blob) != raw_meta.get("n_bytes"):
        return _refuse(RAW_OBJECT_CORRUPT, "retained bytes do not match the recorded sha256/length")
    try:
        decoded = [_normalize_segment(s) for s in decode_waveform(blob)]
    except Exception as exc:  # the adapter's own failure is a refusal, not a crash
        return _refuse(RAW_UNDECODABLE, "%s: %s" % (type(exc).__name__, exc))
    try:
        mine = sorted(_segment_identity(s) for s in record["returned"])
        theirs = sorted(_segment_identity(s) for s in decoded)
    except (KeyError, TypeError, ValueError, RetentionError) as exc:
        return _refuse(DECODED_IDENTITY_MISMATCH, "%s: %s" % (type(exc).__name__, exc))
    if mine != theirs:
        return _refuse(DECODED_IDENTITY_MISMATCH, "decoded %d segment(s) %s != recorded %s" % (len(theirs), theirs[:3], mine[:3]))
    response = record.get("response") or {}
    xml = load_object(response.get("sha256", ""))
    if xml is None:
        return _refuse(RESPONSE_OBJECT_MISSING, "no retained response object %s" % response.get("sha256"))
    if hashlib.sha256(xml).hexdigest() != response.get("sha256") or len(xml) != response.get("n_bytes"):
        return _refuse(RESPONSE_OBJECT_CORRUPT, "retained response bytes do not match the recorded sha256/length")
    try:
        reparsed = response_channels(xml)
    except RetentionError as exc:
        return _refuse(RESPONSE_REPARSE_MISMATCH, str(exc))
    if reparsed != response.get("channels"):
        return _refuse(RESPONSE_REPARSE_MISMATCH, "re-parsed channel epochs differ from the record")
    return Verdict(True, OBJECTS_VERIFIED, "%d segment(s) decoded and matched; response re-parsed" % len(mine))


def admissible_value(record: dict, value, *, min_valid_fraction: float, measurement_start, measurement_end,
                     load_object: Callable[[str], Optional[bytes]],
                     decode_waveform: Callable[[bytes], Sequence[dict]]) -> Tuple[Optional[float], Verdict]:
    """The THD value when it is a finite real number, its record is coherent for the measurement window, AND its
    retained objects re-open to that record; otherwise (None, refusal). A refusal is never a 0.0; a genuine 0.0
    is admissible."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None, _refuse(VALUE_INVALID, "value %r is not a real number" % (value,))
    if not math.isfinite(value):
        return None, _refuse(NONFINITE_VALUE, "value %r is not finite" % (value,))
    verdict = validate(record, min_valid_fraction=min_valid_fraction, measurement_start=measurement_start,
                       measurement_end=measurement_end)
    if not verdict.accepted:
        return None, verdict
    objects = verify_retained_objects(record, load_object=load_object, decode_waveform=decode_waveform)
    if not objects.accepted:
        return None, objects
    return float(value), Verdict(True, ADMITTED, "%s; %s" % (verdict.detail, objects.detail), verdict.facts)
