#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 PRODUCER machinery (grassmann, s4t lane) -- REV 2, folding
the four codex 1721Z repairs. Mechanical transforms from CLOSED,
CONTENT-ADDRESSED staged-input envelopes to the published window-2
contract shapes. Fixture-verified machinery; REAL runs happen at the
availability cutoff / during accrual under the owner gates.

TRUST BOUNDARY (codex 1721Z item 1 -- the declared choice, routed for
the close): the producer's trust boundary is the STAGED-INPUT ENVELOPE
-- a closed schema whose exact bytes are digested BEFORE any
transformation and whose `source` field content-addresses the upstream
raw bytes (kind/ref/sha256). The `producer_code` slot stays OPEN until
either the acquisition-capture surface is also committed and pinned or
the manifest schema's slot note is revised to this boundary (cayley
authors the amendment); this module never flips the slot by itself.

REPAIRS FOLDED (codex 1721Z):
- item 2: every builder returns (artifact, receipt) where
  receipt.output_sha256 digests EXACTLY the returned artifact; the
  selection artifact is the closed wrapper {carrier, cutoff,
  day_records}; a common selftest KAT enforces the identity over
  every lane.
- item 3: every lane consumes ONE closed envelope digested before
  transformation, binding every operation parameter (axis, matrix,
  measured set, region set, bboxes, snapshot/freeze/cutoff,
  longitude, M3 reference, source identity); receipts bind
  input_envelope_sha256 + an explicit operation record +
  output_sha256 + producer identity; one-field mutation doctors run
  across the complete schemas.
- item 4: ONE registered timestamp grammar -- canonical UTC minute
  strings ending 'Z'. Parsing is timezone-AWARE; naive = UTC by
  declaration; any non-UTC offset refuses (the +14:00 wrong-UTC-day
  trap); interval/order/uniqueness run on NORMALIZED UTC instants;
  only the canonical spelling is stored (alternate spellings of one
  instant collapse -- and collide as duplicates). Bundle-level M3
  check requires byte-equal canonical local/reference indices
  (cayley's runner independently re-checks).

Registered absence is None at the staged boundary everywhere: the
canonical-JSON digest (allow_nan=False) structurally refuses NaN/Inf
in ANY envelope -- PRODUCER_ENVELOPE_NONFINITE.

This module opens no window-2 value and fetches nothing.
"""
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import f2g_sealed_run_instrument_cayley as SRI  # pinned digest formula
import w2_selection as WS

PRODUCER_NAME = "grassmann-w2-producer"
IDENTITY_SCHEMA = "f2g-w2-producer-identity-v1"
RECEIPT_SCHEMA = "f2g-w2-producer-receipt-v2"
CAL_EPOCH = "2026-01-01"          # matches w2_mag1.CAL_EPOCH
W2_CARRIERS = ("istanbul_marmara", "socal_coachella",
               "turkey_kahramanmaras", "cascadia")

ENV_DAYCAPSULE = "f2g-w2-staged-envelope-daycapsule-v1"
ENV_SELECTION = "f2g-w2-staged-envelope-selection-v1"
ENV_MF4 = "f2g-w2-staged-envelope-mf4-v1"
ENV_MAG = "f2g-w2-staged-envelope-mag-v1"
ENVELOPE_KEYS = {
    ENV_DAYCAPSULE: {"schema", "carrier", "day", "station_ids",
                     "matrix", "measured", "source"},
    ENV_SELECTION: {"schema", "carrier", "cutoff", "presence_by_day",
                    "source"},
    ENV_MF4: {"schema", "regions", "bboxes", "risk_by_region",
              "catalog_rows", "snapshot_end", "freeze_day", "cutoff",
              "source"},
    ENV_MAG: {"schema", "observatory", "lon_east", "times",
              "components", "weather", "m3_reference", "cutoff",
              "source"},
}
SOURCE_KEYS = {"kind", "ref", "sha256"}


class ProducerRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _canon_digest(obj):
    try:
        return hashlib.sha256(json.dumps(
            obj, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
    except ValueError:
        raise ProducerRefusal(
            "PRODUCER_ENVELOPE_NONFINITE: NaN/Inf in staged bytes; "
            "registered absence is None")


def producer_identity():
    """The producer's identity record: this file's CRLF-normalized
    source sha (the blob the manifest pins), recomputed at call time."""
    with open(os.path.abspath(__file__), "rb") as f:
        sha = hashlib.sha256(
            f.read().replace(b"\r\n", b"\n")).hexdigest()
    return {"schema": IDENTITY_SCHEMA, "name": PRODUCER_NAME,
            "code_blob_sha256": sha}


def open_envelope(envelope, schema_name):
    """Closure check + content-address: the envelope must carry EXACTLY
    the schema's keys (plus a closed source triple), and its exact
    canonical bytes are digested BEFORE any transformation. Returns the
    envelope digest."""
    if not isinstance(envelope, dict):
        raise ProducerRefusal("PRODUCER_ENVELOPE_NOT_CLOSED: not a dict")
    if envelope.get("schema") != schema_name:
        raise ProducerRefusal(
            f"PRODUCER_ENVELOPE_NOT_CLOSED: schema "
            f"{envelope.get('schema')!r} != {schema_name}")
    want = ENVELOPE_KEYS[schema_name]
    got = set(envelope)
    if got != want:
        raise ProducerRefusal(
            f"PRODUCER_ENVELOPE_NOT_CLOSED: missing="
            f"{sorted(want - got)} unknown={sorted(got - want)}")
    src = envelope["source"]
    if not isinstance(src, dict) or set(src) != SOURCE_KEYS:
        raise ProducerRefusal(
            "PRODUCER_ENVELOPE_NOT_CLOSED: source must carry exactly "
            f"{sorted(SOURCE_KEYS)}")
    return _canon_digest(envelope)


def verify_staged_envelope(envelope, expected_sha256):
    """Content-address check at the allowlist boundary: the staged
    envelope's recomputed digest must equal its pinned digest."""
    schema = envelope.get("schema") if isinstance(envelope, dict) \
        else None
    if schema not in ENVELOPE_KEYS:
        raise ProducerRefusal(
            f"PRODUCER_ENVELOPE_NOT_CLOSED: unknown schema {schema!r}")
    got = open_envelope(envelope, schema)
    if got != expected_sha256:
        raise ProducerRefusal(
            f"PRODUCER_ENVELOPE_MISMATCH: {got[:12]} != "
            f"{str(expected_sha256)[:12]}")
    return got


def _receipt(lane, envelope_sha256, operation, artifact):
    return {"schema": RECEIPT_SCHEMA, "lane": lane,
            "input_envelope_sha256": envelope_sha256,
            "operation": dict(operation),
            "output_sha256": _canon_digest(artifact),
            "producer_identity": producer_identity()}


# ------------------------------------------------------------- day capsules
def build_day_capsule(envelope):
    """One carrier-day capsule from a staged coherence-matrix envelope.
    The envelope carries the ordered axis, the FULL matrix (digested
    before reduction), and the measured set. Edges = upper-triangle
    cells with BOTH endpoints measured and a non-None value (None =
    absent cell, never zero), canonical sorted-pair keys (Phase-A REV-2
    contract). station_index_digest via the imported pinned SRI
    formula, never reimplemented."""
    env_sha = open_envelope(envelope, ENV_DAYCAPSULE)
    carrier = envelope["carrier"]
    if carrier not in W2_CARRIERS:
        raise ProducerRefusal(f"PRODUCER_UNKNOWN_CARRIER: {carrier!r}")
    try:
        date.fromisoformat(str(envelope["day"]))
    except ValueError:
        raise ProducerRefusal(
            f"PRODUCER_DAY_INVALID: {envelope['day']!r}")
    ids = [str(s) for s in envelope["station_ids"]]
    if len(set(ids)) != len(ids):
        raise ProducerRefusal("PRODUCER_STATION_DUPLICATE: axis ids")
    meas = sorted(str(s) for s in envelope["measured"])
    if len(set(meas)) != len(meas):
        raise ProducerRefusal("PRODUCER_STATION_DUPLICATE: measured")
    unknown = [s for s in meas if s not in set(ids)]
    if unknown:
        raise ProducerRefusal(
            f"PRODUCER_STATION_UNKNOWN: {unknown} not on the matrix "
            "axis")
    matrix = envelope["matrix"]
    n = len(ids)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ProducerRefusal(f"PRODUCER_MATRIX_SHAPE: need {n}x{n}")
    mset = set(meas)
    edges = {}
    for i in range(n):
        if ids[i] not in mset:
            continue
        for j in range(i + 1, n):
            if ids[j] not in mset:
                continue
            v = matrix[i][j]
            if v is None:
                continue
            v = float(v)
            if not math.isfinite(v):     # unreachable past the digest;
                continue                 # kept as belt-and-suspenders
            edges["|".join(sorted((ids[i], ids[j])))] = v
    capsule = {"measured": meas,
               "station_index_digest": SRI.station_index_digest(meas),
               "edges": edges}
    receipt = _receipt("DAY_CAPSULE", env_sha,
                       {"carrier": carrier,
                        "day": str(envelope["day"]),
                        "source": dict(envelope["source"])}, capsule)
    return capsule, receipt


RECEIPT_KEYS = {"schema", "lane", "input_envelope_sha256",
                "operation", "output_sha256", "producer_identity"}
DAY_CAPSULE_OP_KEYS = {"carrier", "day", "source"}


def assemble_producer_days(carrier, capsules_by_day, receipts_by_day):
    """{day: capsule} + {day: capsule receipt} -> the adapter's
    per-carrier producer feed. codex 1815Z item 4: every receipt is
    CLOSED and fully verified at aggregation -- schema/lane, producer
    pin, carrier, EXACT day (the cross-day replay class), source
    identity, envelope digest, output digest -- and the capsule and
    receipt day-key sets must be exactly equal (no extras)."""
    if carrier not in W2_CARRIERS:
        raise ProducerRefusal(f"PRODUCER_UNKNOWN_CARRIER: {carrier!r}")
    cap_days = {str(d) for d in capsules_by_day}
    rec_days = {str(d) for d in receipts_by_day}
    if cap_days != rec_days:
        raise ProducerRefusal(
            f"PRODUCER_RECEIPT_MISMATCH: day-key sets differ -- "
            f"capsules_only={sorted(cap_days - rec_days)[:4]} "
            f"receipts_only={sorted(rec_days - cap_days)[:4]}")
    ident = producer_identity()
    out = {}
    env_by_day = {}
    for day in sorted(capsules_by_day):
        day = str(day)
        try:
            date.fromisoformat(day)
        except ValueError:
            raise ProducerRefusal(f"PRODUCER_DAY_INVALID: {day!r}")
        rec = receipts_by_day[day]
        if not isinstance(rec, dict) or set(rec) != RECEIPT_KEYS \
                or rec.get("schema") != RECEIPT_SCHEMA:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_NOT_CLOSED: {day}")
        if rec["lane"] != "DAY_CAPSULE":
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} lane "
                f"{rec['lane']!r} != DAY_CAPSULE")
        if rec["producer_identity"] != ident:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} producer identity "
                "is not this producer's pin")
        op = rec["operation"]
        if not isinstance(op, dict) or set(op) != DAY_CAPSULE_OP_KEYS:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_NOT_CLOSED: {day} operation record")
        if op["day"] != day:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} receipt is for day "
                f"{op['day']!r} (cross-day replay)")
        if op["carrier"] != carrier:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} receipt carrier "
                f"{op['carrier']!r} != {carrier}")
        if not isinstance(op["source"], dict) \
                or set(op["source"]) != SOURCE_KEYS:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_NOT_CLOSED: {day} source identity")
        if rec["output_sha256"] != _canon_digest(capsules_by_day[day]):
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} capsule bytes do "
                "not match their receipt")
        out[day] = capsules_by_day[day]
        env_by_day[day] = rec["input_envelope_sha256"]
    artifact = {carrier: out}
    receipt = _receipt("DAY_CAPSULES_AGGREGATE",
                       _canon_digest(env_by_day),
                       {"carrier": carrier,
                        "envelopes_by_day": env_by_day}, artifact)
    return artifact, receipt


# ------------------------------------------------------- selection records
def build_selection_records(envelope):
    """Staged presence envelope -> the CLOSED selection artifact
    {carrier, cutoff, day_records} (codex item 2: the wrapper IS the
    artifact; consumers use its day_records). The exact lookback frame
    is validated producer-side typed -- a malformed frame is the
    producer's to own, not a selection outcome; the engine still
    re-validates."""
    env_sha = open_envelope(envelope, ENV_SELECTION)
    carrier = envelope["carrier"]
    if carrier not in W2_CARRIERS:
        raise ProducerRefusal(f"PRODUCER_UNKNOWN_CARRIER: {carrier!r}")
    cutoff = str(envelope["cutoff"])
    try:
        cut = date.fromisoformat(cutoff)
    except ValueError:
        raise ProducerRefusal(f"PRODUCER_CUTOFF_INVALID: {cutoff!r}")
    lb = WS.LOOKBACK_DAYS
    expect = [(cut - timedelta(days=lb - 1 - i)).isoformat()
              for i in range(lb)]
    have = sorted(str(d) for d in envelope["presence_by_day"])
    if have != expect:
        missing = sorted(set(expect) - set(have))
        extra = sorted(set(have) - set(expect))
        raise ProducerRefusal(
            f"PRODUCER_FRAME_INCOMPLETE: missing={missing[:4]} "
            f"extra={extra[:4]} (need exactly [{expect[0]}, "
            f"{expect[-1]}])")
    records = {d: sorted(str(s)
                         for s in envelope["presence_by_day"][d])
               for d in expect}
    artifact = {"carrier": carrier, "cutoff": cutoff,
                "day_records": records}
    receipt = _receipt("SELECTION_RECORDS", env_sha,
                       {"carrier": carrier, "cutoff": cutoff,
                        "source": dict(envelope["source"])}, artifact)
    return artifact, receipt


# --------------------------------------------------------------- MF4 feed
def build_mf4_feed(envelope):
    """Staged MF4 envelope -> the calibration-runner feed (1353Z
    contract shape, unchanged). The envelope binds regions, bboxes,
    risk, catalog, snapshot/freeze boundary AND the cutoff in one
    digested object (codex item 3)."""
    env_sha = open_envelope(envelope, ENV_MF4)
    snapshot_end = str(envelope["snapshot_end"])
    freeze_day = str(envelope["freeze_day"])
    cutoff = str(envelope["cutoff"])
    if snapshot_end > freeze_day:
        raise ProducerRefusal(
            f"PRODUCER_MF4_FREEZE_ORDER: snapshot_end {snapshot_end} "
            f"> freeze_day {freeze_day}")
    regs = sorted(str(r) for r in envelope["regions"])
    risk_by_region = envelope["risk_by_region"]
    bboxes = envelope["bboxes"]
    for r in regs:
        if r not in bboxes:
            raise ProducerRefusal(
                f"PRODUCER_MF4_REGION_UNBOUND: {r} has no bbox")
        if r not in risk_by_region:
            raise ProducerRefusal(
                f"PRODUCER_MF4_REGION_UNBOUND: {r} has no risk series")
        for d, v in risk_by_region[r].items():
            if v is None:
                raise ProducerRefusal(
                    f"PRODUCER_MF4_NONFINITE: {r} {d} None")
            if str(d) > cutoff:
                raise ProducerRefusal(
                    f"PRODUCER_MF4_AFTER_CUTOFF: {r} risk row {d} > "
                    f"cutoff {cutoff}")
    snap = []
    for row in envelope["catalog_rows"]:
        for k in ("day", "lat", "lon", "mag"):
            if k not in row:
                raise ProducerRefusal(
                    f"PRODUCER_MF4_CATALOG_ROW_INCOMPLETE: {k}")
        if str(row["day"]) > snapshot_end:
            raise ProducerRefusal(
                f"PRODUCER_MF4_POST_SNAPSHOT_EVENT: {row['day']} > "
                f"snapshot_end {snapshot_end}")
        snap.append({"day": str(row["day"]), "lat": float(row["lat"]),
                     "lon": float(row["lon"]),
                     "mag": float(row["mag"])})
    artifact = {"risk_by_region": {r: {str(d): float(v)
                                       for d, v in risk_by_region[r]
                                       .items()}
                                   for r in regs},
                "catalog_snapshot": snap,
                "snapshot_end": snapshot_end,
                "freeze_day": freeze_day,
                "bboxes": {r: dict(bboxes[r]) for r in regs},
                "regions": regs}
    receipt = _receipt("MF4_FEED", env_sha,
                       {"cutoff": cutoff, "snapshot_end": snapshot_end,
                        "freeze_day": freeze_day, "regions": regs,
                        "source": dict(envelope["source"])}, artifact)
    return artifact, receipt


# --------------------------------------------------------------- MAG feeds
def _canon_times(times, cutoff):
    """The registered timestamp grammar (codex item 4): parse
    timezone-AWARE; naive = UTC by declaration; 'Z' = UTC; any non-UTC
    offset refuses. Minute resolution required. Interval, order and
    uniqueness run on the NORMALIZED UTC instants. Returns the
    canonical spellings: 'YYYY-MM-DDTHH:MMZ'."""
    lo = date.fromisoformat(CAL_EPOCH)
    hi = date.fromisoformat(str(cutoff))
    out = []
    prev = None
    for t in times:
        try:
            d = datetime.fromisoformat(
                str(t).replace("Z", "+00:00"))
        except ValueError:
            raise ProducerRefusal(f"PRODUCER_MAG_TIME_INVALID: {t!r}")
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        elif d.utcoffset() != timezone.utc.utcoffset(None):
            raise ProducerRefusal(
                f"PRODUCER_MAG_TIME_FRAME: {t!r} carries a non-UTC "
                "offset (the canonical frame is UTC)")
        d = d.astimezone(timezone.utc)
        if d.second or d.microsecond:
            raise ProducerRefusal(
                f"PRODUCER_MAG_TIME_RESOLUTION: {t!r} is not a minute "
                "stamp")
        if not (lo <= d.date() <= hi):
            raise ProducerRefusal(
                f"PRODUCER_MAG_TIME_OUT_OF_INTERVAL: {t!r} outside "
                f"[{CAL_EPOCH}, {cutoff}] UTC")
        if prev is not None:
            if d == prev:
                raise ProducerRefusal(
                    f"PRODUCER_MAG_TIME_DUPLICATE: {t!r} (normalized "
                    "instant already present)")
            if d < prev:
                raise ProducerRefusal(
                    f"PRODUCER_MAG_TIME_ORDER: {t!r} precedes "
                    f"{prev.isoformat()}")
        prev = d
        out.append(d.strftime("%Y-%m-%dT%H:%M") + "Z")
    return out


def _normalize_series(name, values, n):
    if len(values) != n:
        raise ProducerRefusal(
            f"PRODUCER_MAG_ALIGNMENT: {name} has {len(values)} rows, "
            f"times has {n}")
    out = []
    for v in values:
        if v is None:
            out.append(None)          # registered absence
            continue
        v = float(v)
        if not math.isfinite(v):      # unreachable past the envelope
            raise ProducerRefusal(    # digest; belt-and-suspenders
                f"PRODUCER_MAG_NONFINITE: {name}")
        out.append(v)
    return out


def build_mag_feed(envelope):
    """Staged per-observatory MAG envelope -> the calibration-runner
    feed (1353Z shape). The envelope binds observatory, longitude,
    times, series, M3 reference AND the cutoff in one digested object;
    only canonical UTC 'Z' spellings are stored in the artifact."""
    env_sha = open_envelope(envelope, ENV_MAG)
    cutoff = str(envelope["cutoff"])
    times = _canon_times([str(t) for t in envelope["times"]], cutoff)
    n = len(times)
    comps = {}
    for c in ("X", "Y"):
        if c not in envelope["components"]:
            raise ProducerRefusal(
                f"PRODUCER_MAG_COMPONENT_MISSING: "
                f"{envelope['observatory']}:{c}")
        comps[c] = _normalize_series(
            f"components:{c}", envelope["components"][c], n)
    wx = {}
    for name in sorted(envelope["weather"]):
        wx[name] = _normalize_series(
            f"weather:{name}", envelope["weather"][name], n)
    ref = envelope["m3_reference"]
    artifact = {"observatory": str(envelope["observatory"]),
                "lon_east": float(envelope["lon_east"]),
                "times": times, "components": comps, "weather": wx,
                "m3_reference": (str(ref) if ref else None)}
    receipt = _receipt("MAG_FEED", env_sha,
                       {"observatory": artifact["observatory"],
                        "cutoff": cutoff,
                        "lon_east": artifact["lon_east"],
                        "m3_reference": artifact["m3_reference"],
                        "source": dict(envelope["source"])}, artifact)
    return artifact, receipt


def build_mag_bundle(envelopes_by_obs):
    """All observatory envelopes -> {obs: feed} for the runner, with
    the bundle-level checks: every envelope's cutoff identical, and
    every M3 pair's canonical time indices BYTE-EQUAL (cayley's runner
    independently re-checks). The bundle receipt binds each
    observatory's envelope digest."""
    feeds = {}
    recs = {}
    cutoffs = set()
    for obs in sorted(envelopes_by_obs):
        feed, rec = build_mag_feed(envelopes_by_obs[obs])
        if feed["observatory"] != str(obs):
            raise ProducerRefusal(
                f"PRODUCER_MAG_BUNDLE_KEY_MISMATCH: {obs!r} maps an "
                f"envelope for {feed['observatory']!r}")
        feeds[obs] = feed
        recs[obs] = rec
        cutoffs.add(rec["operation"]["cutoff"])
    if len(cutoffs) > 1:
        raise ProducerRefusal(
            f"PRODUCER_MAG_CUTOFF_MISMATCH: {sorted(cutoffs)}")
    for obs, feed in feeds.items():
        ref = feed["m3_reference"]
        if not ref:
            continue
        if ref not in feeds:
            raise ProducerRefusal(
                f"PRODUCER_MAG_M3_REFERENCE_ABSENT: {obs} -> {ref}")
        if feed["times"] != feeds[ref]["times"]:
            raise ProducerRefusal(
                f"PRODUCER_MAG_M3_TIME_INDEX_MISMATCH: {obs} vs {ref}")
    receipt = _receipt(
        "MAG_BUNDLE",
        _canon_digest({o: recs[o]["input_envelope_sha256"]
                       for o in sorted(recs)}),
        {"cutoff": sorted(cutoffs)[0] if cutoffs else None,
         "envelopes_by_observatory": {
             o: recs[o]["input_envelope_sha256"]
             for o in sorted(recs)}}, feeds)
    return feeds, receipt


# ---------------------------------------------- staged-envelope records (v1.2)
# The producer_boundary amendment v1 registers the per-lane/per-day
# envelope-record schema; per the codex 1400Z ruling the per-record
# CONTENT doctors are the producer's (this surface). Slot-level shape
# doctors live in the execution verifier (cayley).
ENVELOPE_RECORD_SCHEMA = "f2g-w2-staged-envelope-v1"
ENVELOPE_RECORD_KEYS = {
    "schema", "lane", "carrier", "utc_day", "raw_body_sha256",
    "raw_body_bytes", "source", "endpoint", "request_params",
    "receipt", "capture_time_utc", "cutoff", "operation_params",
    "expected_keys", "output_sha256"}
RECORD_LANES = {"DAY_CAPSULE", "SELECTION_RECORDS", "MF4_FEED",
                "MAG_FEED"}
# codex 1843Z item 3: the ENVELOPE receipt interior is CLOSED and
# carries the transcript link -- the envelope's dynamic seam is a
# projection of the capture transcript, never caller-assembled.
# (Distinct from the lane-receipt RECEIPT_KEYS defined earlier.)
# codex 0349Z item 1: the receipt ALSO carries the closed AUTHORITY
# identity (commit/path/blob/keys digests) so E itself is the
# chronological carrier proving which frozen static authority
# parameterized the request.
ENVELOPE_RECEIPT_KEYS = {"requested_url", "effective_url",
                         "request_start_utc",
                         "response_complete_utc", "http_status",
                         "headers", "transcript_sha256",
                         "authority"}
AUTHORITY_ID_KEYS = {"commit", "path", "blob_sha256", "keys_sha256"}
# codex 1843Z item 2: the contract splits -- STATIC fields are
# registered pre-capture from registered sources; the dynamic seam
# (receipt, capture instant, content digests) comes only from the
# transcript carrier
STATIC_FIELDS = ("endpoint", "request_params", "cutoff",
                 "operation_params", "expected_keys")
STATIC_SOURCE_KEYS = {"kind", "ref"}       # identity only, NO sha
TRANSCRIPT_SCHEMA = "f2g-w2-capture-transcript-v1"
TRANSCRIPT_KEYS = {
    "schema", "lane", "carrier", "utc_day", "static_contract_sha256",
    "requested_url", "effective_url", "request_start_utc",
    "response_complete_utc", "http_status", "headers",
    "raw_body_sha256", "raw_body_bytes", "authority"}


def _validate_authority_id(a, what, refusal_cls=None):
    """The closed authority-identity binding (codex 0349Z item 1):
    exactly {commit, path, blob_sha256, keys_sha256}; commit 40-hex
    lowercase; digests 64-hex lowercase."""
    exc = refusal_cls or ProducerRefusal
    if not isinstance(a, dict) or set(a) != AUTHORITY_ID_KEYS:
        raise exc(f"{what}: authority identity not closed")
    c = a["commit"]
    if not (isinstance(c, str) and len(c) == 40
            and all(x in "0123456789abcdef" for x in c)):
        raise exc(f"{what}: authority commit is not 40-hex")
    for f in ("blob_sha256", "keys_sha256"):
        v = a[f]
        if not (isinstance(v, str) and len(v) == 64
                and all(x in "0123456789abcdef" for x in v)):
            raise exc(f"{what}: authority {f} is not 64-hex")
    if not isinstance(a["path"], str) or not a["path"]:
        raise exc(f"{what}: authority path invalid")
    return a


_CAPTURE_RE = None
_DAY_RE = None


def _canon_utc_instant(t, what):
    """codex 1544Z item 2 + 2235Z item 5: a byte-bound record's
    canonical frame must BE canonical -- exactly ONE registered
    spelling: UTC 'Z' with seconds, fractional part 1-6 digits ONLY
    (the runtime keeps microseconds; precision it would discard is
    refused, so no two distinct spellings collapse to one instant).
    fullmatch, never $-anchored match (a trailing newline must
    refuse). Non-string input refuses."""
    global _CAPTURE_RE
    if _CAPTURE_RE is None:
        import re
        _CAPTURE_RE = re.compile(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z")
    if not isinstance(t, str):
        raise ProducerRefusal(
            f"PRODUCER_RECORD_TIME_FRAME: {what} is not a string")
    if not _CAPTURE_RE.fullmatch(t):
        raise ProducerRefusal(
            f"PRODUCER_RECORD_TIME_FRAME: {what} {t!r} is not the "
            "canonical UTC 'Z' spelling with seconds (fractional "
            "1-6 digits)")
    try:
        return datetime.fromisoformat(
            t.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        raise ProducerRefusal(f"PRODUCER_RECORD_TIME_INVALID: {what} "
                              f"{t!r}")


def _canon_day(d, what):
    """Exact YYYY-MM-DD only (codex: date.fromisoformat admits
    non-canonical spellings)."""
    global _DAY_RE
    if _DAY_RE is None:
        import re
        _DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
    if not isinstance(d, str):
        raise ProducerRefusal(f"PRODUCER_RECORD_DAY_INVALID: {what} "
                              "is not a string")
    if not _DAY_RE.fullmatch(d):
        raise ProducerRefusal(f"PRODUCER_RECORD_DAY_INVALID: {what} "
                              f"{d!r}")
    try:
        date.fromisoformat(d)
    except ValueError:
        raise ProducerRefusal(f"PRODUCER_RECORD_DAY_INVALID: {what} "
                              f"{d!r}")
    return d


def _hex64(v, what):
    if not (isinstance(v, str) and len(v) == 64
            and all(c in "0123456789abcdef" for c in v)):
        raise ProducerRefusal(
            f"PRODUCER_RECORD_BODY_INVALID: {what} is not "
            "lowercase-hex sha256")
    return v


def requested_url_of(endpoint, request_params):
    """The registered URL derivation: endpoint + '?' + urlencoded
    SORTED params (one canonical spelling)."""
    from urllib.parse import urlencode
    if not request_params:
        return str(endpoint)
    return str(endpoint) + "?" + urlencode(
        sorted((str(k), str(v)) for k, v in request_params.items()))


def build_envelope_record(*, lane, carrier, utc_day, raw_body,
                          source, endpoint, request_params,
                          transcript, cutoff, operation_params,
                          expected_keys, artifact):
    """One closed per-lane/per-day envelope record (amendment v1
    schema; codex 1843Z S/T/E design): the DYNAMIC seam (receipt +
    capture instant + content digests) is a PROJECTION of the capture
    transcript -- never caller-assembled. `source` carries identity
    only (kind, ref); its content digest is derived from the exact
    staged bytes."""
    if not isinstance(raw_body, (bytes, bytearray)):
        raise ProducerRefusal(
            "PRODUCER_RECORD_BODY_INVALID: raw_body must be the exact "
            "staged bytes")
    if not isinstance(source, dict) \
            or set(source) != STATIC_SOURCE_KEYS:
        raise ProducerRefusal(
            "PRODUCER_RECORD_NOT_CLOSED: build source carries "
            "identity only (kind, ref); the content digest is derived")
    if not isinstance(transcript, dict) \
            or set(transcript) != TRANSCRIPT_KEYS:
        raise ProducerRefusal(
            "PRODUCER_TRANSCRIPT_INVALID: transcript not closed at "
            "build")
    sha = hashlib.sha256(bytes(raw_body)).hexdigest()
    receipt = {"requested_url": transcript["requested_url"],
               "effective_url": transcript["effective_url"],
               "request_start_utc": transcript["request_start_utc"],
               "response_complete_utc":
                   transcript["response_complete_utc"],
               "http_status": transcript["http_status"],
               "headers": dict(transcript["headers"]),
               "transcript_sha256": _canon_digest(transcript),
               "authority": dict(transcript["authority"])}
    rec = {"schema": ENVELOPE_RECORD_SCHEMA, "lane": str(lane),
           "carrier": str(carrier), "utc_day": str(utc_day),
           "raw_body_sha256": sha,
           "raw_body_bytes": len(raw_body),
           "source": {"kind": str(source["kind"]),
                      "ref": str(source["ref"]), "sha256": sha},
           "endpoint": str(endpoint),
           "request_params": dict(request_params),
           "receipt": receipt,
           "capture_time_utc": transcript["response_complete_utc"],
           "cutoff": str(cutoff),
           "operation_params": dict(operation_params),
           "expected_keys": sorted(str(k) for k in expected_keys),
           "output_sha256": _canon_digest(artifact)}
    verify_envelope_record(rec, raw_body=bytes(raw_body))
    return rec


STATIC_CONTRACT_SCHEMA = "f2g-w2-static-contract-v1"
STATIC_CONTRACT_KEYS = {"schema", "lane", "carrier", "utc_day",
                        "source", "endpoint", "request_params",
                        "cutoff", "operation_params",
                        "expected_keys"}


def verify_transcript(transcript, static_contract, raw_body=None):
    """The T-vs-S leg of the codex join: the transcript is a closed,
    LINKED operation record -- it names its static contract by digest,
    its requested URL must equal the registered derivation, its
    timestamps are canonical and ordered, and (when the bytes are
    supplied) its content address recomputes."""
    t = transcript
    if not isinstance(t, dict) or \
            t.get("schema") != TRANSCRIPT_SCHEMA or \
            set(t) != TRANSCRIPT_KEYS:
        raise ProducerRefusal(
            "PRODUCER_TRANSCRIPT_INVALID: not the closed transcript "
            "schema")
    s = static_contract
    if not isinstance(s, dict) or \
            s.get("schema") != STATIC_CONTRACT_SCHEMA or \
            set(s) != STATIC_CONTRACT_KEYS:
        raise ProducerRefusal(
            "PRODUCER_RECORD_CONTRACT_MISMATCH: static contract not "
            "closed")
    if not isinstance(s["source"], dict) or \
            set(s["source"]) != STATIC_SOURCE_KEYS:
        raise ProducerRefusal(
            "PRODUCER_RECORD_CONTRACT_MISMATCH: static source must "
            "carry identity only (kind, ref) -- content digests are "
            "derived, never registered")
    for k in ("lane", "carrier", "utc_day"):
        if t[k] != s[k]:
            raise ProducerRefusal(
                f"PRODUCER_TRANSCRIPT_INVALID: {k} {t[k]!r} != "
                f"contract {s[k]!r}")
    if t["static_contract_sha256"] != _canon_digest(s):
        raise ProducerRefusal(
            "PRODUCER_TRANSCRIPT_INVALID: transcript is not linked "
            "to this static contract")
    if t["requested_url"] != requested_url_of(
            s["endpoint"], s["request_params"]):
        raise ProducerRefusal(
            "PRODUCER_TRANSCRIPT_INVALID: requested_url diverges "
            "from the registered derivation")
    # codex 2015Z item 2: PARSED-instant ordering, never
    # lexicographic (mixed fractional/non-fractional spellings break
    # string comparison in both directions)
    t_start = _canon_utc_instant(t["request_start_utc"],
                                 "request_start_utc")
    t_done = _canon_utc_instant(t["response_complete_utc"],
                                "response_complete_utc")
    if t_start > t_done:
        raise ProducerRefusal(
            "PRODUCER_TRANSCRIPT_INVALID: reversed instants")
    if t["http_status"] != 200:
        raise ProducerRefusal(
            f"PRODUCER_TRANSCRIPT_INVALID: non-200 transcript "
            f"({t['http_status']}) can never back a staged record")
    _validate_authority_id(t["authority"],
                           "PRODUCER_TRANSCRIPT_INVALID")
    _hex64(t["raw_body_sha256"], "transcript raw_body_sha256")
    if raw_body is not None:
        if hashlib.sha256(raw_body).hexdigest() != \
                t["raw_body_sha256"] or \
                len(raw_body) != t["raw_body_bytes"]:
            raise ProducerRefusal(
                "PRODUCER_TRANSCRIPT_INVALID: staged bytes diverge "
                "from the transcript content address")
    return _canon_digest(t)


def verify_envelope_record(record, *, raw_body=None, artifact=None):
    """Per-record content verification (the producer's doctor
    surface): closed field set (schema extension refuses), registered
    lane, ISO day, canonical-UTC capture instant, content-addressed
    body (recomputed when the bytes are supplied), output digest
    (recomputed when the artifact is supplied), NaN/Inf structurally
    refused via the canonical digest."""
    if not isinstance(record, dict) or \
            record.get("schema") != ENVELOPE_RECORD_SCHEMA or \
            set(record) != ENVELOPE_RECORD_KEYS:
        got = set(record) if isinstance(record, dict) else None
        raise ProducerRefusal(
            f"PRODUCER_RECORD_NOT_CLOSED: "
            f"missing={sorted(ENVELOPE_RECORD_KEYS - got) if got else '?'} "
            f"unknown={sorted(got - ENVELOPE_RECORD_KEYS) if got else '?'}")
    if record["lane"] not in RECORD_LANES:
        raise ProducerRefusal(
            f"PRODUCER_RECORD_LANE_UNREGISTERED: {record['lane']!r}")
    _canon_day(record["utc_day"], "utc_day")
    _canon_utc_instant(record["capture_time_utc"], "capture_time_utc")
    if not isinstance(record["source"], dict) \
            or set(record["source"]) != SOURCE_KEYS:
        raise ProducerRefusal(
            "PRODUCER_RECORD_NOT_CLOSED: source triple")
    rc = record["receipt"]
    if not isinstance(rc, dict) or set(rc) != ENVELOPE_RECEIPT_KEYS:
        raise ProducerRefusal(
            "PRODUCER_RECORD_NOT_CLOSED: receipt interior")
    # codex 2015Z item 2: PARSED-instant ordering, never lexicographic
    rc_start = _canon_utc_instant(rc["request_start_utc"],
                                  "request_start_utc")
    rc_done = _canon_utc_instant(rc["response_complete_utc"],
                                 "response_complete_utc")
    if rc_start > rc_done:
        raise ProducerRefusal(
            "PRODUCER_RECORD_TIME_FRAME: request start after "
            "response completion (reversed instants)")
    if record["capture_time_utc"] != rc["response_complete_utc"]:
        raise ProducerRefusal(
            "PRODUCER_RECORD_TIME_FRAME: capture_time_utc is defined "
            "as the response-complete instant")
    _hex64(rc["transcript_sha256"], "transcript_sha256")
    _validate_authority_id(rc["authority"],
                           "PRODUCER_RECORD_NOT_CLOSED")
    if not isinstance(record["request_params"], dict) \
            or not isinstance(record["operation_params"], dict) \
            or not isinstance(record["expected_keys"], list):
        raise ProducerRefusal(
            "PRODUCER_RECORD_NOT_CLOSED: container shapes")
    _hex64(record["raw_body_sha256"], "raw_body_sha256")
    _hex64(record["output_sha256"], "output_sha256")
    if not isinstance(record["raw_body_bytes"], int) \
            or record["raw_body_bytes"] < 0:
        raise ProducerRefusal(
            "PRODUCER_RECORD_BODY_INVALID: byte-size shape")
    # codex 1843Z item 1 (BLOCKER repair): the source triple's content
    # digest IS the raw-body content address -- divergence refuses at
    # the per-record gate
    if record["source"]["sha256"] != record["raw_body_sha256"]:
        raise ProducerRefusal(
            "PRODUCER_RECORD_BODY_MISMATCH: source.sha256 diverges "
            "from raw_body_sha256")
    if raw_body is not None:
        if hashlib.sha256(raw_body).hexdigest() != \
                record["raw_body_sha256"] \
                or len(raw_body) != record["raw_body_bytes"]:
            raise ProducerRefusal(
                "PRODUCER_RECORD_BODY_MISMATCH: staged bytes diverge "
                "from the content address")
    if artifact is not None and \
            _canon_digest(artifact) != record["output_sha256"]:
        raise ProducerRefusal(
            "PRODUCER_RECORD_OUTPUT_MISMATCH")
    return _canon_digest(record)


def verify_staged_day_set(records_by_day, raw_bodies_by_day,
                          artifacts_by_day, static_contracts_by_day,
                          transcripts_by_day, expected_days, carrier,
                          lane):
    """The day-set gate = the codex 1843Z THREE-CARRIER JOIN, run by
    the VERIFIER (never assembled by the caller): for every registered
    day, S (the pre-capture static contract from registered sources),
    T (the capture transcript), and E (the envelope record) must
    satisfy E.static == S, T valid against S + the reopened bytes,
    and E.dynamic == projection(T). Content recomputation is
    MANDATORY (codex 1544Z); all five string-keyed maps must equal
    the registered day set exactly; cross-day/carrier replays
    refuse."""
    want = {_canon_day(d, "expected_days") for d in expected_days}
    for name, m in (("records", records_by_day),
                    ("raw_bodies", raw_bodies_by_day),
                    ("artifacts", artifacts_by_day),
                    ("static_contracts", static_contracts_by_day),
                    ("transcripts", transcripts_by_day)):
        if not isinstance(m, dict):
            raise ProducerRefusal(
                f"PRODUCER_RECORD_CONTENT_MISSING: {name} is not a "
                "day-keyed map")
        have = {str(d) for d in m}
        if have != want:
            raise ProducerRefusal(
                f"PRODUCER_RECORD_DAY_SET_MISMATCH: {name} "
                f"missing={sorted(want - have)[:4]} "
                f"extra={sorted(have - want)[:4]}")
    out = {}
    for day in sorted(records_by_day):
        rec = records_by_day[day]
        body = raw_bodies_by_day[day]
        art = artifacts_by_day[day]
        s = static_contracts_by_day[day]
        t = transcripts_by_day[day]
        if body is None or art is None or s is None or t is None:
            raise ProducerRefusal(
                f"PRODUCER_RECORD_CONTENT_MISSING: {day} "
                "body/artifact/contract/transcript")
        if not isinstance(body, (bytes, bytearray)):
            raise ProducerRefusal(
                f"PRODUCER_RECORD_CONTENT_MISSING: {day} raw body is "
                "not bytes")
        body = bytes(body)
        # E alone (incl source.sha256 == raw_body_sha256, closed
        # receipt interior, canonical instants)
        verify_envelope_record(rec, raw_body=body, artifact=art)
        # replay checks
        if rec["utc_day"] != str(day):
            raise ProducerRefusal(
                f"PRODUCER_RECORD_REPLAY: record for "
                f"{rec['utc_day']} staged under {day}")
        if rec["carrier"] != str(carrier) or rec["lane"] != str(lane):
            raise ProducerRefusal(
                f"PRODUCER_RECORD_REPLAY: {day} carrier/lane "
                f"{rec['carrier']}/{rec['lane']} != "
                f"{carrier}/{lane}")
        # T against S + the reopened bytes (transcript day/carrier
        # swaps surface HERE as linkage refusals)
        verify_transcript(t, s, raw_body=body)
        if s["lane"] != str(lane) or s["carrier"] != str(carrier) \
                or s["utc_day"] != str(day):
            raise ProducerRefusal(
                f"PRODUCER_RECORD_REPLAY: {day} static contract is "
                "for another lane/carrier/day")
        # E.static == S (source identity only -- the digest is
        # derived, never registered)
        for f in STATIC_FIELDS:
            want_v = s[f]
            got_v = rec[f]
            if f == "expected_keys":
                want_v = sorted(str(k) for k in want_v)
            if got_v != want_v:
                raise ProducerRefusal(
                    f"PRODUCER_RECORD_CONTRACT_MISMATCH: {day} {f}")
        if {k: rec["source"][k] for k in STATIC_SOURCE_KEYS} != \
                dict(s["source"]):
            raise ProducerRefusal(
                f"PRODUCER_RECORD_CONTRACT_MISMATCH: {day} source "
                "identity")
        # E.dynamic == projection(T)
        rc = rec["receipt"]
        proj = {"requested_url": t["requested_url"],
                "effective_url": t["effective_url"],
                "request_start_utc": t["request_start_utc"],
                "response_complete_utc": t["response_complete_utc"],
                "http_status": t["http_status"],
                "headers": dict(t["headers"]),
                "transcript_sha256": _canon_digest(t),
                "authority": dict(t["authority"])}
        if rc != proj:
            diff = sorted(k for k in proj if rc.get(k) != proj[k])
            raise ProducerRefusal(
                f"PRODUCER_RECORD_TRANSCRIPT_MISMATCH: {day} receipt "
                f"diverges from the transcript projection: {diff}")
        if rec["capture_time_utc"] != t["response_complete_utc"] \
                or rec["raw_body_sha256"] != t["raw_body_sha256"] \
                or rec["raw_body_bytes"] != t["raw_body_bytes"]:
            raise ProducerRefusal(
                f"PRODUCER_RECORD_TRANSCRIPT_MISMATCH: {day} "
                "instant/content divergence")
        out[str(day)] = _canon_digest(rec)
    return out


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    import numpy as np
    import w2_accrual_instrument_cayley as ACC
    import w2_b2b as W2B
    import w2_calibration_runner_cayley as CAL

    def refuses(fn, code):
        try:
            fn()
            return False
        except ProducerRefusal as e:
            return str(e).startswith(code)

    src = {"kind": "kat", "ref": "synthetic://fixture", "sha256":
           "cd" * 32}
    all_pairs = []          # (artifact, receipt) for the common KAT

    ident = producer_identity()
    assert ident["name"] == PRODUCER_NAME \
        and len(ident["code_blob_sha256"]) == 64
    assert producer_identity() == ident

    # --- day capsule: hand 4-station matrix, None cells, one
    # unmeasured station; exact expected edges ---
    ids4 = ["CC.B", "UW.A", "UW.C", "UW.D"]
    m = [[1.0, 0.5, None, 0.2],
         [0.5, 1.0, 0.7, None],
         [None, 0.7, 1.0, 0.9],
         [0.2, None, 0.9, 1.0]]

    def dc_env(**over):
        e = {"schema": ENV_DAYCAPSULE, "carrier": "cascadia",
             "day": "2026-10-01", "station_ids": ids4, "matrix": m,
             "measured": ["CC.B", "UW.A", "UW.C"], "source": src}
        e.update(over)
        return e
    cap, cap_rec = build_day_capsule(dc_env())
    all_pairs.append((cap, cap_rec))
    assert cap["measured"] == ["CC.B", "UW.A", "UW.C"]
    assert cap["edges"] == {"CC.B|UW.A": 0.5, "UW.A|UW.C": 0.7}, \
        cap["edges"]
    assert cap["station_index_digest"] == SRI.station_index_digest(
        ["CC.B", "UW.A", "UW.C"])
    assert cap_rec["input_envelope_sha256"] == _canon_digest(dc_env())
    # NaN anywhere in the envelope refuses at the digest
    bad = [row[:] for row in m]
    bad[0][2] = float("nan")
    assert refuses(lambda: build_day_capsule(dc_env(matrix=bad)),
                   "PRODUCER_ENVELOPE_NONFINITE")
    assert refuses(lambda: build_day_capsule(dc_env(measured=["ZZ.Q"])),
                   "PRODUCER_STATION_UNKNOWN")
    assert refuses(lambda: build_day_capsule(dc_env(matrix=m[:3])),
                   "PRODUCER_MATRIX_SHAPE")
    assert refuses(
        lambda: build_day_capsule(dc_env(station_ids=["A", "A", "B",
                                                      "C"])),
        "PRODUCER_STATION_DUPLICATE")
    assert refuses(lambda: build_day_capsule(dc_env(day="not-a-day")),
                   "PRODUCER_DAY_INVALID")
    assert refuses(lambda: build_day_capsule(dc_env(carrier="nope")),
                   "PRODUCER_UNKNOWN_CARRIER")

    # --- adapter e2e: envelopes -> capsules -> aggregate ->
    # build_family_panel -> b2b engine ---
    reg10 = sorted([f"A{i}" for i in range(5)]
                   + [f"B{i}" for i in range(5)])
    cal6 = [f"2026-10-{i:02d}" for i in range(1, 7)]
    big = [[(5.0 if (reg10[i][0] == reg10[j][0]) else 0.1)
            for j in range(10)] for i in range(10)]
    caps, recs = {}, {}
    for d in cal6:
        c, r = build_day_capsule(
            {"schema": ENV_DAYCAPSULE, "carrier": "cascadia",
             "day": d, "station_ids": reg10, "matrix": big,
             "measured": reg10, "source": src})
        caps[d], recs[d] = c, r
    feed, agg_rec = assemble_producer_days("cascadia", caps, recs)
    all_pairs.append((feed, agg_rec))
    assert set(agg_rec["operation"]["envelopes_by_day"]) == set(cal6)
    reg_rec = {"registries": {"cascadia": {
        "selected": reg10, "churn": 1.0, "typing": None}}}
    panel = ACC.build_family_panel(cal6, reg_rec, feed)
    res = W2B.w2_b2b_family(panel, doc_sha256="ab" * 32, n_draws=49)
    assert res["runs_by_carrier"]["cascadia"] == 1 \
        and res["p_value"] == 1.0, res
    # doctored capsule bytes vs receipt refuse at aggregation
    caps2 = json.loads(json.dumps(caps))
    caps2[cal6[0]]["edges"]["A0|A1"] = 9.9
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps2, recs),
        "PRODUCER_RECEIPT_MISMATCH")
    # day-key sets must be exactly equal (missing AND extra receipts)
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps,
                                       {d: recs[d]
                                        for d in cal6[1:]}),
        "PRODUCER_RECEIPT_MISMATCH")
    extra_recs = dict(recs)
    extra_recs["2026-10-09"] = recs[cal6[0]]
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps, extra_recs),
        "PRODUCER_RECEIPT_MISMATCH")
    # codex 1815Z item-4 doctors:
    # (a) the cross-day replay verbatim -- identical capsule bytes
    # under a NEW day key with the day-1 receipt supplied for it
    day1, day2 = cal6[0], "2026-10-09"
    replay_caps = {day1: caps[day1], day2: caps[day1]}
    replay_recs = {day1: recs[day1], day2: recs[day1]}
    assert refuses(
        lambda: assemble_producer_days("cascadia", replay_caps,
                                       replay_recs),
        "PRODUCER_RECEIPT_MISMATCH")
    # (b) cross-carrier receipt
    cc_cap, cc_rec = build_day_capsule(
        {"schema": ENV_DAYCAPSULE, "carrier": "istanbul_marmara",
         "day": "2026-10-01", "station_ids": reg10, "matrix": big,
         "measured": reg10, "source": src})
    assert refuses(
        lambda: assemble_producer_days(
            "cascadia", {"2026-10-01": cc_cap},
            {"2026-10-01": cc_rec}),
        "PRODUCER_RECEIPT_MISMATCH")
    # (c) doctored producer identity (cross-producer)
    fp_recs = json.loads(json.dumps(recs))
    fp_recs[cal6[1]]["producer_identity"]["code_blob_sha256"] = \
        "0" * 64
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps, fp_recs),
        "PRODUCER_RECEIPT_MISMATCH")
    # (d) wrong lane
    wl_recs = json.loads(json.dumps(recs))
    wl_recs[cal6[2]]["lane"] = "MAG_FEED"
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps, wl_recs),
        "PRODUCER_RECEIPT_MISMATCH")
    # (e) non-closed receipt (extra field) + non-closed operation
    nc_recs = json.loads(json.dumps(recs))
    nc_recs[cal6[3]]["surprise"] = 1
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps, nc_recs),
        "PRODUCER_RECEIPT_NOT_CLOSED")
    no_recs = json.loads(json.dumps(recs))
    del no_recs[cal6[4]]["operation"]["day"]
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps, no_recs),
        "PRODUCER_RECEIPT_NOT_CLOSED")

    # --- selection: wrapper artifact + production select e2e ---
    cut = "2026-08-24"
    lb = WS.LOOKBACK_DAYS
    frame = [(date.fromisoformat(cut) - timedelta(days=lb - 1 - i))
             .isoformat() for i in range(lb)]

    def sel_env(**over):
        e = {"schema": ENV_SELECTION, "carrier": "cascadia",
             "cutoff": cut, "presence_by_day":
             {d: list(reg10) for d in frame}, "source": src}
        e.update(over)
        return e
    sel_art, sel_rec = build_selection_records(sel_env())
    all_pairs.append((sel_art, sel_rec))
    assert set(sel_art) == {"carrier", "cutoff", "day_records"}
    sel = WS.select("cascadia", sel_art["day_records"],
                    sel_art["cutoff"])
    assert sel["selected"] == reg10 and sel["typing"] is None
    assert refuses(
        lambda: build_selection_records(
            sel_env(presence_by_day={d: list(reg10)
                                     for d in frame[1:]})),
        "PRODUCER_FRAME_INCOMPLETE")

    # --- MF4 e2e through the REAL repaired runner ---
    repo = tempfile.mkdtemp(prefix="w2_prod_kat_")
    rng = np.random.Generator(np.random.PCG64(23))
    days = [(date(2025, 10, 10) + timedelta(days=i)).isoformat()
            for i in range(120)]
    bbox = {"min_lat": 30, "max_lat": 40, "min_lon": -125,
            "max_lon": -115}
    risk = {r: {d: float(rng.uniform(0, 1)) for d in days}
            for r in ("ra", "rb")}
    catalog = [{"day": (date(2025, 11, 1) + timedelta(days=7 * i))
                .isoformat(), "lat": 35.0, "lon": -120.0, "mag": 4.5}
               for i in range(8)]

    def mf4_env(**over):
        e = {"schema": ENV_MF4, "regions": ["ra", "rb"],
             "bboxes": {"ra": bbox, "rb": bbox},
             "risk_by_region": risk, "catalog_rows": catalog,
             "snapshot_end": "2026-02-08", "freeze_day": "2026-02-10",
             "cutoff": "2026-02-09", "source": src}
        e.update(over)
        return e
    mf4_feed, mf4_rec = build_mf4_feed(mf4_env())
    all_pairs.append((mf4_feed, mf4_rec))
    out = CAL.run_mf4_calibration(repo, mf4_feed, "2026-02-09",
                                  producer_identity())
    v = CAL.verify_receipt(repo, out["receipt"],
                           expected_cutoff="2026-02-09",
                           expected_producer=producer_identity())
    assert v["lane"] == "MF4"
    mf4_feed2, mf4_rec2 = build_mf4_feed(mf4_env())
    assert mf4_rec2["output_sha256"] == mf4_rec["output_sha256"]
    late = catalog + [{"day": "2026-02-09", "lat": 35.0,
                       "lon": -120.0, "mag": 5.0}]
    assert refuses(lambda: build_mf4_feed(mf4_env(catalog_rows=late)),
                   "PRODUCER_MF4_POST_SNAPSHOT_EVENT")
    assert refuses(
        lambda: build_mf4_feed(mf4_env(bboxes={"ra": bbox})),
        "PRODUCER_MF4_REGION_UNBOUND")
    assert refuses(
        lambda: build_mf4_feed(mf4_env(snapshot_end="2026-02-11")),
        "PRODUCER_MF4_FREEZE_ORDER")
    assert refuses(
        lambda: build_mf4_feed(mf4_env(cutoff="2026-01-15")),
        "PRODUCER_MF4_AFTER_CUTOFF")
    bad_risk = json.loads(json.dumps(risk))
    bad_risk["ra"][days[0]] = None
    assert refuses(
        lambda: build_mf4_feed(mf4_env(risk_by_region=bad_risk)),
        "PRODUCER_MF4_NONFINITE")

    # --- MAG: canonical frame + bundle + e2e through the repaired
    # runner ---
    n = 3000
    times = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat()
             for i in range(n)]
    wx = {"symh": rng.normal(size=n).tolist()}

    def mag_env(name, ref, **over):
        e = {"schema": ENV_MAG, "observatory": name,
             "lon_east": -120.0, "times": times,
             "components": {
                 "X": rng.normal(20000, 5, size=n).tolist(),
                 "Y": rng.normal(4000, 5, size=n).tolist()},
             "weather": wx, "m3_reference": ref, "cutoff": cut,
             "source": src}
        e.update(over)
        return e
    envs = {"FRN": mag_env("FRN", "TUC"), "TUC": mag_env("TUC", None)}
    feeds, bundle_rec = build_mag_bundle(envs)
    all_pairs.append((feeds, bundle_rec))
    assert feeds["FRN"]["times"][0] == "2026-01-01T00:00Z"
    out = CAL.run_mag_calibration(repo, feeds, cut,
                                  producer_identity())
    v = CAL.verify_receipt(repo, out["receipt"], expected_cutoff=cut,
                           expected_producer=producer_identity())
    assert v["lane"] == "MAG"
    assert "FRN:TUC:X" in out["results"]["m3"]
    # the +14:00 wrong-UTC-day trap (codex item 4 reproduction)
    t4 = times[:4]
    xs = rng.normal(20000, 5, size=4).tolist()

    def small(name, tt, **over):
        e = mag_env(name, None, times=tt,
                    components={"X": xs, "Y": xs},
                    weather={"symh": [0.0] * 4})
        e.update(over)
        return e
    assert refuses(lambda: build_mag_feed(
        small("FRN", ["2026-01-01T00:00:00+14:00"] + t4[1:])),
        "PRODUCER_MAG_TIME_FRAME")
    # alternate spellings of one instant collapse -> duplicate
    assert refuses(lambda: build_mag_feed(
        small("FRN", ["2026-01-01T00:00", "2026-01-01T00:00:00Z",
                      t4[2], t4[3]])),
        "PRODUCER_MAG_TIME_DUPLICATE")
    # naive/aware mixing is legal ONLY as the same UTC frame
    f_mix, _ = build_mag_feed(
        small("FRN", ["2026-01-01T00:00", "2026-01-01T00:01:00Z",
                      "2026-01-01T00:02:00+00:00", t4[3]]))
    assert f_mix["times"][:3] == ["2026-01-01T00:00Z",
                                  "2026-01-01T00:01Z",
                                  "2026-01-01T00:02Z"]
    assert refuses(lambda: build_mag_feed(
        small("FRN", [t4[1], t4[0], t4[2], t4[3]])),
        "PRODUCER_MAG_TIME_ORDER")
    assert refuses(lambda: build_mag_feed(
        small("FRN", ["2026-09-01T00:00:00"] + t4[1:])),
        "PRODUCER_MAG_TIME_OUT_OF_INTERVAL")
    assert refuses(lambda: build_mag_feed(
        small("FRN", ["2026-01-01T00:00:30"] + t4[1:])),
        "PRODUCER_MAG_TIME_RESOLUTION")
    assert refuses(lambda: build_mag_feed(
        small("FRN", t4, components={"X": xs[:3], "Y": xs})),
        "PRODUCER_MAG_ALIGNMENT")
    assert refuses(lambda: build_mag_feed(
        small("FRN", t4, components={"X": [1.0, float("inf"), 1.0,
                                           1.0], "Y": xs})),
        "PRODUCER_ENVELOPE_NONFINITE")
    assert refuses(lambda: build_mag_feed(
        small("FRN", t4, components={"Y": xs})),
        "PRODUCER_MAG_COMPONENT_MISSING")
    f_none, _ = build_mag_feed(
        small("FRN", t4, components={"X": [1.0, None, 1.0, 1.0],
                                     "Y": xs}))
    assert f_none["components"]["X"][1] is None
    # bundle: shifted-but-equal-length reference clock refuses
    sh = [(datetime(2026, 1, 1) + timedelta(minutes=i + 1))
          .isoformat() for i in range(n)]
    assert refuses(lambda: build_mag_bundle(
        {"FRN": mag_env("FRN", "TUC"),
         "TUC": mag_env("TUC", None, times=sh)}),
        "PRODUCER_MAG_M3_TIME_INDEX_MISMATCH")
    assert refuses(lambda: build_mag_bundle(
        {"FRN": mag_env("FRN", "TUC")}),
        "PRODUCER_MAG_M3_REFERENCE_ABSENT")
    assert refuses(lambda: build_mag_bundle(
        {"FRN": mag_env("FRN", None),
         "TUC": mag_env("TUC", None, cutoff="2026-08-23")}),
        "PRODUCER_MAG_CUTOFF_MISMATCH")

    # --- codex item 2: the COMMON receipt KAT over every lane ---
    for art, rec in all_pairs:
        assert rec["schema"] == RECEIPT_SCHEMA
        assert rec["output_sha256"] == _canon_digest(art), rec["lane"]
        assert rec["producer_identity"]["name"] == PRODUCER_NAME

    # --- codex item 3: one-field mutation doctors across the complete
    # envelope schemas -- every mutation either refuses typed or
    # content-addresses to a DIFFERENT envelope digest ---
    envelopes = [(dc_env(), ENV_DAYCAPSULE, build_day_capsule),
                 (sel_env(), ENV_SELECTION, build_selection_records),
                 (mf4_env(), ENV_MF4, build_mf4_feed),
                 (mag_env("FRN", None), ENV_MAG, build_mag_feed)]
    mutants = {"carrier": "istanbul_marmara", "day": "2026-10-02",
               "cutoff": "2026-08-23", "observatory": "TUC",
               "lon_east": -119.0, "snapshot_end": "2026-02-07",
               "freeze_day": "2026-02-11",
               "regions": ["ra"], "m3_reference": None,
               "source": {"kind": "kat", "ref": "synthetic://other",
                          "sha256": "ee" * 32}}
    for env, schema, builder in envelopes:
        base_sha = open_envelope(env, schema)
        # envelope verification: pin + mismatch
        assert verify_staged_envelope(env, base_sha) == base_sha
        assert refuses(
            lambda: verify_staged_envelope(env, "0" * 64),
            "PRODUCER_ENVELOPE_MISMATCH")
        # closure: dropped key / unknown key
        for key in sorted(ENVELOPE_KEYS[schema] - {"schema"}):
            broken = {k: v for k, v in env.items() if k != key}
            assert refuses(lambda: builder(broken),
                           "PRODUCER_ENVELOPE_NOT_CLOSED"), key
        extra = dict(env)
        extra["surprise"] = 1
        assert refuses(lambda: builder(extra),
                       "PRODUCER_ENVELOPE_NOT_CLOSED")
        # one-field mutations: refusal or a different address
        for key in sorted(ENVELOPE_KEYS[schema] - {"schema",
                                                   "source"}):
            if key not in mutants or mutants[key] == env.get(key):
                continue
            mut = dict(env)
            mut[key] = mutants[key]
            try:
                _, rec = builder(mut)
                assert rec["input_envelope_sha256"] != base_sha, key
            except ProducerRefusal:
                pass
        mut = dict(env)
        mut["source"] = mutants["source"]
        _, rec = builder(mut)
        assert rec["input_envelope_sha256"] != base_sha

    # --- staged-envelope records (amendment v1 + the codex 1843Z
    # S/T/E three-carrier join) ---
    body = b"raw staged bytes \x00\x01"
    src_id = {"kind": "kat", "ref": "synthetic://fixture"}

    def mk_static(day="2026-08-20", **over):
        s = {"schema": STATIC_CONTRACT_SCHEMA, "lane": "DAY_CAPSULE",
             "carrier": "cascadia", "utc_day": day,
             "source": dict(src_id),
             "endpoint": "https://service.example/fdsn",
             "request_params": {"net": "UW,CC,CN", "cha": "HHZ"},
             "cutoff": "2026-08-25",
             "operation_params": {"carrier": "cascadia", "day": day},
             "expected_keys": [day]}
        s.update(over)
        return s

    AUTH_ID = {"commit": "a" * 40,
               "path": ("docs/f2g_window2_execution/"
                        "staged_expected_contracts_v2.json"),
               "blob_sha256": "cd" * 32, "keys_sha256": "ef" * 32}

    def mk_transcript(s, raw=body, **over):
        t = {"schema": TRANSCRIPT_SCHEMA, "lane": s["lane"],
             "carrier": s["carrier"], "utc_day": s["utc_day"],
             "static_contract_sha256": _canon_digest(s),
             "requested_url": requested_url_of(
                 s["endpoint"], s["request_params"]),
             "effective_url": "https://edge.example/fdsn-final",
             "request_start_utc": "2026-08-20T12:01:31Z",
             "response_complete_utc": "2026-08-20T12:01:33Z",
             "http_status": 200,
             "headers": {"content-type": "text/plain"},
             "raw_body_sha256": hashlib.sha256(raw).hexdigest(),
             "raw_body_bytes": len(raw),
             "authority": dict(AUTH_ID)}
        t.update(over)
        return t

    def mk_rec(day="2026-08-20", s=None, t=None, **over):
        s = s or mk_static(day)
        t = t or mk_transcript(s)
        kw = dict(lane=s["lane"], carrier=s["carrier"],
                  utc_day=s["utc_day"], raw_body=body,
                  source=dict(src_id), endpoint=s["endpoint"],
                  request_params=s["request_params"], transcript=t,
                  cutoff=s["cutoff"],
                  operation_params=s["operation_params"],
                  expected_keys=s["expected_keys"],
                  artifact={"edges": {}})
        kw.update(over)
        return build_envelope_record(**kw)
    s20 = mk_static()
    t20 = mk_transcript(s20)
    rec_ok = mk_rec(s=s20, t=t20)
    assert verify_envelope_record(rec_ok, raw_body=body,
                                  artifact={"edges": {}})
    assert rec_ok["source"]["sha256"] == rec_ok["raw_body_sha256"]
    assert rec_ok["receipt"]["transcript_sha256"] == \
        _canon_digest(t20)
    # doctors: extension, missing field, wrong body sha, wrong output
    ext = dict(rec_ok)
    ext["surprise"] = 1
    assert refuses(lambda: verify_envelope_record(ext),
                   "PRODUCER_RECORD_NOT_CLOSED")
    shr = {k: v for k, v in rec_ok.items() if k != "receipt"}
    assert refuses(lambda: verify_envelope_record(shr),
                   "PRODUCER_RECORD_NOT_CLOSED")
    assert refuses(
        lambda: verify_envelope_record(rec_ok, raw_body=body + b"x"),
        "PRODUCER_RECORD_BODY_MISMATCH")
    assert refuses(
        lambda: verify_envelope_record(rec_ok,
                                       artifact={"edges": {"a": 1}}),
        "PRODUCER_RECORD_OUTPUT_MISMATCH")
    # codex 1843Z item-1 BLOCKER lock: source.sha256 divergence
    # refuses at the PER-RECORD gate
    forged_src = json.loads(json.dumps(rec_ok))
    forged_src["source"]["sha256"] = "f" * 64
    assert refuses(lambda: verify_envelope_record(forged_src),
                   "PRODUCER_RECORD_BODY_MISMATCH")
    # canonical instants (1544Z item 2, carried): only Z-with-seconds
    for bad_t in ("2026-08-20", "2026-08-20T00:00:00",
                  "2026-08-20T12:01:33+00:00", "2026-08-20T12:01Z"):
        assert refuses(
            lambda t=bad_t: mk_rec(t=mk_transcript(
                s20, response_complete_utc=t)),
            "PRODUCER_RECORD_TIME_FRAME"), bad_t
    # reversed timestamps refuse
    assert refuses(lambda: mk_rec(t=mk_transcript(
        s20, request_start_utc="2026-08-20T12:01:34Z")),
        "PRODUCER_RECORD_TIME_FRAME")
    # codex 2015Z item-2 locks: PARSED-instant ordering both ways --
    # temporally ORDERED but lexicographically reversed must PASS...
    ok_frac = mk_rec(t=mk_transcript(
        s20, request_start_utc="2026-08-20T12:01:33Z",
        response_complete_utc="2026-08-20T12:01:33.100Z"))
    assert ok_frac["capture_time_utc"] == "2026-08-20T12:01:33.100Z"
    # ...and temporally REVERSED but lexicographically ordered must
    # REFUSE (at the envelope gate AND the transcript gate)
    assert refuses(lambda: mk_rec(t=mk_transcript(
        s20, request_start_utc="2026-08-20T12:01:33.900Z",
        response_complete_utc="2026-08-20T12:01:33Z")),
        "PRODUCER_RECORD_TIME_FRAME")
    assert refuses(lambda: verify_transcript(mk_transcript(
        s20, request_start_utc="2026-08-20T12:01:33.900Z",
        response_complete_utc="2026-08-20T12:01:33Z"), s20),
        "PRODUCER_TRANSCRIPT_INVALID")
    assert verify_transcript(mk_transcript(
        s20, request_start_utc="2026-08-20T12:01:33Z",
        response_complete_utc="2026-08-20T12:01:33.100Z"), s20)
    # 2235Z item-5 locks: fractional precision the runtime would
    # DISCARD refuses (the seventh-digit reversal cannot be spelled);
    # exactly six digits pass; trailing-newline / non-string days
    # refuse (fullmatch, strict types)
    for seven in ("2026-08-20T12:01:33.0000009Z",
                  "2026-08-20T12:01:33.0000001Z"):
        assert refuses(lambda t=seven: mk_rec(t=mk_transcript(
            s20, response_complete_utc=t)),
            "PRODUCER_RECORD_TIME_FRAME"), seven
    assert mk_rec(t=mk_transcript(
        s20, response_complete_utc="2026-08-20T12:01:33.123456Z"))
    assert refuses(lambda: mk_rec(utc_day="2026-08-20\n", s=s20,
                                  t=t20),
                   "PRODUCER_RECORD_DAY_INVALID")
    assert refuses(lambda: verify_envelope_record(
        dict(rec_ok, capture_time_utc=20260820)),
        "PRODUCER_RECORD_TIME_FRAME")
    for bad_d in ("not-a-day", "20260820", "2026-8-20"):
        assert refuses(
            lambda d=bad_d: mk_rec(utc_day=d, s=s20, t=t20),
            "PRODUCER_RECORD_DAY_INVALID"), bad_d
    assert refuses(lambda: mk_rec(lane="NOT_A_LANE", s=s20, t=t20),
                   "PRODUCER_RECORD_LANE_UNREGISTERED")
    assert refuses(
        lambda: mk_rec(request_params={"x": float("nan")}, s=s20,
                       t=t20),
        "PRODUCER_ENVELOPE_NONFINITE")
    fake = dict(rec_ok, raw_body_sha256="g" * 64)
    assert refuses(lambda: verify_envelope_record(fake),
                   "PRODUCER_RECORD_BODY_INVALID")
    fake2 = dict(rec_ok, output_sha256="not-a-digest")
    assert refuses(lambda: verify_envelope_record(fake2),
                   "PRODUCER_RECORD_BODY_INVALID")

    # transcript leg alone: linkage + derivation + status doctors
    assert verify_transcript(t20, s20, raw_body=body)
    assert refuses(lambda: verify_transcript(
        mk_transcript(s20, static_contract_sha256="0" * 64), s20),
        "PRODUCER_TRANSCRIPT_INVALID")        # unlinked transcript
    assert refuses(lambda: verify_transcript(
        mk_transcript(s20, requested_url="https://evil.example/x"),
        s20), "PRODUCER_TRANSCRIPT_INVALID")  # derivation divergence
    assert refuses(lambda: verify_transcript(
        mk_transcript(s20, http_status=503), s20),
        "PRODUCER_TRANSCRIPT_INVALID")
    # static source may NEVER carry a registered digest
    s_sha = mk_static()
    s_sha["source"] = {"kind": "kat", "ref": "x", "sha256": "0" * 64}
    assert refuses(lambda: verify_transcript(t20, s_sha),
                   "PRODUCER_RECORD_CONTRACT_MISMATCH")
    # codex 0349Z item 1: the authority identity binding is closed --
    # non-hex commit, unclosed shape, and an E/T authority divergence
    # all refuse
    assert refuses(lambda: verify_transcript(
        mk_transcript(s20, authority=dict(AUTH_ID, commit="HEAD")),
        s20), "PRODUCER_TRANSCRIPT_INVALID")
    assert refuses(lambda: verify_transcript(
        mk_transcript(s20, authority={"commit": "a" * 40}), s20),
        "PRODUCER_TRANSCRIPT_INVALID")
    div_auth = json.loads(json.dumps(rec_ok))
    div_auth["receipt"]["authority"] = dict(AUTH_ID,
                                            blob_sha256="ab" * 32)
    # the shape stays closed (a valid-looking substitute), so the
    # divergence must surface at the E==projection(T) gate
    assert verify_envelope_record(div_auth)
    assert refuses(lambda: verify_staged_day_set(
        {"2026-08-20": div_auth}, {"2026-08-20": body},
        {"2026-08-20": {"edges": {}}}, {"2026-08-20": s20},
        {"2026-08-20": t20}, ["2026-08-20"], "cascadia",
        "DAY_CAPSULE"), "PRODUCER_RECORD_TRANSCRIPT_MISMATCH")

    # --- the day-set gate: the S/T/E JOIN performed by the verifier
    s21 = mk_static("2026-08-21")
    t21 = mk_transcript(s21)
    r20 = rec_ok
    r21 = mk_rec("2026-08-21", s=s21, t=t21)
    days2 = ["2026-08-20", "2026-08-21"]
    recs2 = {"2026-08-20": r20, "2026-08-21": r21}
    bodies2 = {d: body for d in days2}
    arts2 = {d: {"edges": {}} for d in days2}
    stat2 = {"2026-08-20": s20, "2026-08-21": s21}
    tr2 = {"2026-08-20": t20, "2026-08-21": t21}
    assert set(verify_staged_day_set(
        recs2, bodies2, arts2, stat2, tr2, days2, "cascadia",
        "DAY_CAPSULE")) == set(days2)
    # codex item-1 BLOCKER at the SET gate: forged source.sha copied
    # into a forged static overlay can no longer pass -- the static
    # carrier has no digest field at all, and E refuses first
    bad_recs = dict(recs2, **{"2026-08-20": forged_src})
    assert refuses(lambda: verify_staged_day_set(
        bad_recs, bodies2, arts2, stat2, tr2, days2, "cascadia",
        "DAY_CAPSULE"), "PRODUCER_RECORD_BODY_MISMATCH")
    # codex item-2 BLOCKER: the record-cloned-as-contract bypass is
    # structurally dead -- a transcript derived from E (wrong linkage
    # digest) refuses; a swapped transcript (day/carrier) refuses
    fake_t = dict(t20, static_contract_sha256=_canon_digest(
        mk_static(endpoint="https://other.example/")))
    assert refuses(lambda: verify_staged_day_set(
        recs2, bodies2, arts2, stat2,
        dict(tr2, **{"2026-08-20": fake_t}), days2, "cascadia",
        "DAY_CAPSULE"), "PRODUCER_TRANSCRIPT_INVALID")
    assert refuses(lambda: verify_staged_day_set(
        recs2, bodies2, arts2, stat2,
        {"2026-08-20": t21, "2026-08-21": t20}, days2, "cascadia",
        "DAY_CAPSULE"), "PRODUCER_TRANSCRIPT_INVALID")   # swapped
    # record-vs-transcript divergence (effective URL doctored in E's
    # receipt only)
    div = json.loads(json.dumps(r20))
    div["receipt"]["effective_url"] = "https://elsewhere.example/"
    assert refuses(lambda: verify_staged_day_set(
        dict(recs2, **{"2026-08-20": div}), bodies2, arts2, stat2,
        tr2, days2, "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_TRANSCRIPT_MISMATCH")
    # missing sidecars / wrong body / wrong output / wrong contract
    assert refuses(lambda: verify_staged_day_set(
        recs2, {"2026-08-20": body}, arts2, stat2, tr2, days2,
        "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_DAY_SET_MISMATCH")
    assert refuses(lambda: verify_staged_day_set(
        recs2, dict(bodies2, **{"2026-08-21": None}), arts2, stat2,
        tr2, days2, "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_CONTENT_MISSING")
    assert refuses(lambda: verify_staged_day_set(
        recs2, dict(bodies2, **{"2026-08-20": body + b"x"}), arts2,
        stat2, tr2, days2, "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_BODY_MISMATCH")
    assert refuses(lambda: verify_staged_day_set(
        recs2, bodies2, dict(arts2, **{"2026-08-21": {"edges":
                                                      {"z": 2}}}),
        stat2, tr2, days2, "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_OUTPUT_MISMATCH")
    bad_stat = {"2026-08-20": mk_static(request_params={"net": "XX"}),
                "2026-08-21": s21}
    assert refuses(lambda: verify_staged_day_set(
        recs2, bodies2, arts2, bad_stat, tr2, days2, "cascadia",
        "DAY_CAPSULE"), "PRODUCER_TRANSCRIPT_INVALID")
    # missing/extra day + replays
    assert refuses(lambda: verify_staged_day_set(
        {"2026-08-20": r20}, bodies2, arts2, stat2, tr2, days2,
        "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_DAY_SET_MISMATCH")
    assert refuses(lambda: verify_staged_day_set(
        {"2026-08-20": r20, "2026-08-21": r20}, bodies2, arts2,
        stat2, tr2, days2, "cascadia", "DAY_CAPSULE"),
        "PRODUCER_RECORD_REPLAY")                     # cross-day
    assert refuses(lambda: verify_staged_day_set(
        recs2, bodies2, arts2, stat2, tr2, days2,
        "istanbul_marmara", "DAY_CAPSULE"),
        "PRODUCER_RECORD_REPLAY")                     # cross-carrier

    print("w2_producer selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
