"""
thd_retention.py -- PROSPECTIVE per-fetch retention record for the seismic_thd acquisition (thd-acquisition-retention-v1).

NOT WIRED. Nothing in the daily runner imports this module; seismic_thd.fetch_continuous_data_for_thd is unchanged.
It defines what a FUTURE THD fetch must retain so that a value can later be traced to its exact input, and a
validator that refuses -- with a typed code, never a zero -- when the retained facts cannot support a measurement.

Why (codex THD_CONTROLS_RULING_CODEX_20261001 section 4; grassmann 6367f3bb inventory): the current path requests
location '*', merges with linear interpolation and takes st[0]; it retains no raw bytes or hash, no selected
location, no response, no gap or filled-sample count and no actual valid coverage. Past values therefore cannot be
replayed exactly. This module makes those facts retainable going forward; it does not reconstruct any past value.

The record (all fields required unless marked):
  request        network, station, location (as requested, may be a wildcard), channel, start/end UTC
  provider       data centre name and the service URL that answered (no credentials)
  returned       every segment as returned BEFORE merge: NSLC, start/end UTC, sampling rate, sample count
  raw            sha256 and byte length of the exact bytes returned (the caller stores the bytes, content-addressed)
  response       sha256 / byte length of the level=response StationXML for the SELECTED channel, and the channel
                 epochs, sensitivity and input/output units parsed FROM that document (never caller-asserted)
  samples        native rate, expected samples over the request, returned samples, gaps (start, end, missing
                 samples) and the number of samples a merge would FILL
  preprocessing  a version identifier for the code that turns the bytes into the THD input, and its output units
  sensitivity_at_tidal_frequencies  "NOT_ASSESSED" -- a response document is necessary, not sufficient: whether the
                 instrument resolves ~1e-5 Hz is a separate assessment that must precede any physical interpretation.

Validation (`validate`) refuses, in this order:
  AMBIGUOUS_LOCATION        more than one NSLC returned (st[0] would be an arbitrary choice) or none
  NSLC_MISMATCH             the returned channel is not the requested one, or the response is for another channel
  RATE_INCONSISTENT         segments disagree on the sampling rate
  RAW_NOT_RETAINED          raw hash/length absent or inconsistent
  RESPONSE_NOT_RETAINED     no response document retained
  RESPONSE_EPOCH_MISMATCH   no response epoch of the selected channel covers the whole measurement window
  UNITS_NOT_STATED          input units (from the response) or preprocessing output units absent
  INADEQUATE_VALID_COVERAGE returned (non-filled) samples cover less than the caller-declared fraction
The minimum coverage fraction is a REQUIRED argument: the module registers no default (an unstated threshold is a
guess the caller must declare).
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

RETENTION_SCHEMA = "thd-acquisition-retention-v1"
STATIONXML_NS = "http://www.fdsn.org/xml/station/1"
SENSITIVITY_NOT_ASSESSED = "NOT_ASSESSED"

ACCEPTED = "RETENTION_COMPLETE"
AMBIGUOUS_LOCATION = "AMBIGUOUS_LOCATION"
NSLC_MISMATCH = "NSLC_MISMATCH"
RATE_INCONSISTENT = "RATE_INCONSISTENT"
RAW_NOT_RETAINED = "RAW_NOT_RETAINED"
RESPONSE_NOT_RETAINED = "RESPONSE_NOT_RETAINED"
RESPONSE_EPOCH_MISMATCH = "RESPONSE_EPOCH_MISMATCH"
UNITS_NOT_STATED = "UNITS_NOT_STATED"
INADEQUATE_VALID_COVERAGE = "INADEQUATE_VALID_COVERAGE"
RECORD_MALFORMED = "RECORD_MALFORMED"


class RetentionError(ValueError):
    """A defect in the caller's inputs to build_record (not a measurement refusal)."""


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    code: str
    detail: str

    @property
    def reason(self) -> str:
        return "%s: %s" % (self.code, self.detail) if self.detail else self.code


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
# Building the record
# ---------------------------------------------------------------------------------------------------------------

def _gaps(segments: Sequence[dict], rate: float, start: datetime, end: datetime) -> Tuple[List[dict], int, int]:
    """Gaps between the request start, the returned segments (sorted) and the request end; returns (gaps, samples
    a merge would fill between segments, samples missing at the edges). Overlaps count as zero gap."""
    spans = sorted((_utc(s["starttime_utc"]), _utc(s["endtime_utc"])) for s in segments)
    step = 1.0 / rate
    gaps, filled, edge = [], 0, 0
    cursor = start
    for i, (a, b) in enumerate(spans):
        missing = int(round((a - cursor).total_seconds() / step)) - (1 if i else 0)
        if missing > 0:
            gaps.append(dict(start_utc=_iso(cursor), end_utc=_iso(a), missing_samples=missing,
                             position="leading" if i == 0 else "interior"))
            if i == 0:
                edge += missing
            else:
                filled += missing
        cursor = max(cursor, b)
    trailing = int(round((end - cursor).total_seconds() / step))
    if trailing > 0:
        gaps.append(dict(start_utc=_iso(cursor), end_utc=_iso(end), missing_samples=trailing, position="trailing"))
        edge += trailing
    return gaps, filled, edge


def build_record(*, request: dict, provider: dict, segments: Sequence[dict], raw_bytes: bytes,
                 response_xml: Optional[bytes], preprocessing_version: str, output_units: str) -> dict:
    """Assemble the retention record from what the fetch actually saw. Every argument is required (keyword-only,
    no defaults). `segments` are the traces as returned, BEFORE any merge."""
    for key in ("network", "station", "location", "channel", "starttime_utc", "endtime_utc"):
        if key not in request:
            raise RetentionError("request lacks %r" % key)
    start, end = _utc(request["starttime_utc"]), _utc(request["endtime_utc"])
    if end <= start:
        raise RetentionError("request window is empty or reversed")
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise RetentionError("raw_bytes must be the exact bytes returned")
    rates = sorted({float(s["sampling_rate_hz"]) for s in segments})
    returned = [dict(nslc=_nslc(s), starttime_utc=_iso(_utc(s["starttime_utc"])),
                     endtime_utc=_iso(_utc(s["endtime_utc"])), sampling_rate_hz=float(s["sampling_rate_hz"]),
                     npts=int(s["npts"])) for s in segments]
    rate = rates[0] if len(rates) == 1 else None
    expected = int(round((end - start).total_seconds() * rate)) + 1 if rate else None
    gaps, filled, edge = _gaps(segments, rate, start, end) if rate else ([], None, None)
    returned_samples = sum(r["npts"] for r in returned)
    record = dict(
        schema=RETENTION_SCHEMA,
        request=dict(nslc=_nslc(request), starttime_utc=_iso(start), endtime_utc=_iso(end)),
        provider=dict(name=provider.get("name"), service_url=provider.get("service_url")),
        returned=returned,
        raw=dict(sha256=hashlib.sha256(bytes(raw_bytes)).hexdigest(), n_bytes=len(raw_bytes)),
        response=None if response_xml is None else dict(
            format="StationXML", level="response", sha256=hashlib.sha256(response_xml).hexdigest(),
            n_bytes=len(response_xml), channels=response_channels(response_xml)),
        samples=dict(native_rate_hz=rate, sampling_rates_returned=rates, expected=expected, returned=returned_samples,
                     merge_would_fill=filled, missing_at_edges=edge, gaps=gaps),
        preprocessing=dict(version=preprocessing_version, output_units=output_units),
        sensitivity_at_tidal_frequencies=SENSITIVITY_NOT_ASSESSED,
    )
    record["record_sha256"] = record_digest(record)
    return record


def canonical(record: dict) -> bytes:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_digest(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------------------

def validate(record: dict, *, min_valid_fraction: float, measurement_start, measurement_end) -> Verdict:
    """Typed verdict for one retained fetch. `min_valid_fraction` is the caller's declared threshold (no default);
    the measurement window is the span the THD value claims to describe."""
    if isinstance(min_valid_fraction, bool) or not isinstance(min_valid_fraction, (int, float)) \
            or not math.isfinite(min_valid_fraction) or not 0.0 < float(min_valid_fraction) <= 1.0:
        raise RetentionError("min_valid_fraction must be a declared number in (0, 1], got %r" % (min_valid_fraction,))
    try:
        if record.get("record_sha256") != record_digest(record):
            return Verdict(False, RECORD_MALFORMED, "record digest does not match its content")
        m_start, m_end = _utc(measurement_start), _utc(measurement_end)
        returned = record["returned"]
        nslcs = sorted({r["nslc"] for r in returned})
        if len(nslcs) != 1:
            return Verdict(False, AMBIGUOUS_LOCATION, "returned channels %s (requested %s)"
                           % (nslcs or "none", record["request"]["nslc"]))
        selected = nslcs[0]
        req = record["request"]["nslc"].split(".")
        sel = selected.split(".")
        if any(not any(ch in r for ch in "*?") and r != s for r, s in zip(req, sel)):
            return Verdict(False, NSLC_MISMATCH, "returned %s for requested %s" % (selected, record["request"]["nslc"]))
        samples = record["samples"]
        if samples["native_rate_hz"] is None:
            return Verdict(False, RATE_INCONSISTENT, "sampling rates %s" % samples["sampling_rates_returned"])
        raw = record.get("raw") or {}
        if not raw.get("sha256") or not isinstance(raw.get("n_bytes"), int) or raw["n_bytes"] <= 0:
            return Verdict(False, RAW_NOT_RETAINED, "raw sha256/length absent or empty")
        response = record.get("response")
        if not response or not response.get("sha256"):
            return Verdict(False, RESPONSE_NOT_RETAINED, "no level=response document for %s" % selected)
        epochs = [c for c in response["channels"] if c["nslc"] == selected]
        if not epochs:
            others = sorted({c["nslc"] for c in response["channels"]})
            return Verdict(False, NSLC_MISMATCH, "response describes %s, not %s" % (others, selected))
        covering = [c for c in epochs if c["has_response"] and _utc(c["start"]) <= m_start
                    and (c["end"] is None or _utc(c["end"]) >= m_end)]
        if not covering:
            spans = ["%s..%s" % (c["start"], c["end"] or "open") for c in epochs]
            return Verdict(False, RESPONSE_EPOCH_MISMATCH, "no %s response epoch covers %s..%s (epochs %s)"
                           % (selected, _iso(m_start), _iso(m_end), spans))
        if not covering[0].get("input_units") or not record["preprocessing"].get("output_units"):
            return Verdict(False, UNITS_NOT_STATED, "input %r, output %r" % (covering[0].get("input_units"),
                                                                             record["preprocessing"].get("output_units")))
        fraction = samples["returned"] / samples["expected"] if samples["expected"] else 0.0
        if fraction < float(min_valid_fraction):
            return Verdict(False, INADEQUATE_VALID_COVERAGE, "%d of %d samples returned (%.4f < declared %.4f); "
                           "%d would be filled by merge, %d missing at the edges"
                           % (samples["returned"], samples["expected"], fraction, float(min_valid_fraction),
                              samples["merge_would_fill"], samples["missing_at_edges"]))
        return Verdict(True, ACCEPTED, "%s, %d/%d samples, response epoch %s..%s; tidal-band sensitivity %s"
                       % (selected, samples["returned"], samples["expected"], covering[0]["start"],
                          covering[0]["end"] or "open", record["sensitivity_at_tidal_frequencies"]))
    except (KeyError, TypeError, RetentionError) as exc:
        return Verdict(False, RECORD_MALFORMED, "%s: %s" % (type(exc).__name__, exc))


def admissible_value(record: dict, value: float, *, min_valid_fraction: float, measurement_start,
                     measurement_end) -> Tuple[Optional[float], Verdict]:
    """The THD value when its retained input is admissible, else (None, refusal). A refusal is never a 0.0."""
    verdict = validate(record, min_valid_fraction=min_valid_fraction, measurement_start=measurement_start,
                       measurement_end=measurement_end)
    return (float(value) if verdict.accepted else None), verdict
