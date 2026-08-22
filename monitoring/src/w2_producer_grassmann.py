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


def assemble_producer_days(carrier, capsules_by_day, receipts_by_day):
    """{day: capsule} + {day: capsule receipt} -> the adapter's
    per-carrier producer feed. Each per-day receipt is re-verified
    against its capsule bytes before aggregation; the aggregate receipt
    binds every per-day envelope digest."""
    if carrier not in W2_CARRIERS:
        raise ProducerRefusal(f"PRODUCER_UNKNOWN_CARRIER: {carrier!r}")
    out = {}
    env_by_day = {}
    for day in sorted(capsules_by_day):
        try:
            date.fromisoformat(str(day))
        except ValueError:
            raise ProducerRefusal(f"PRODUCER_DAY_INVALID: {day!r}")
        rec = receipts_by_day.get(day)
        if rec is None:
            raise ProducerRefusal(f"PRODUCER_RECEIPT_MISSING: {day}")
        if rec["output_sha256"] != _canon_digest(capsules_by_day[day]):
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} capsule bytes do "
                "not match their receipt")
        if rec["operation"].get("carrier") != carrier:
            raise ProducerRefusal(
                f"PRODUCER_RECEIPT_MISMATCH: {day} receipt carrier "
                f"{rec['operation'].get('carrier')!r} != {carrier}")
        out[str(day)] = capsules_by_day[day]
        env_by_day[str(day)] = rec["input_envelope_sha256"]
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
    assert refuses(
        lambda: assemble_producer_days("cascadia", caps,
                                       {d: recs[d]
                                        for d in cal6[1:]}),
        "PRODUCER_RECEIPT_MISSING")

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
                           expected_cutoff="2026-02-09")
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
    v = CAL.verify_receipt(repo, out["receipt"], expected_cutoff=cut)
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

    print("w2_producer selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
