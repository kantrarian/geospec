#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 PRODUCER machinery (grassmann, s4t lane) -- the mechanical
transforms that turn staged raw inputs into the published window-2
contract shapes. Fixture-verified machinery now; REAL runs happen at
the availability cutoff / during accrual under the owner gates, and the
`producer_code` execution-manifest slot binds to this file's bytes then.

The producer role is MECHANICAL-ONLY during accrual (cascadia capsule,
v0.3 s2 stage 2): no policy, no thresholds, no selection judgment --
shape assembly, canonical digests, and typed refusals at the boundary.
Policy lives in the engines behind cayley's adapter/runner seams.

CONTRACT TARGETS (published seams, consumed verbatim -- never
reimplemented):
1. DAY CAPSULES for the adapter (w2_accrual_instrument build_family_
   panel): {carrier: {day: {"measured": [ids], "station_index_digest":
   <pinned SRI formula>, "edges": {"A|B": float}}}}. Edge construction
   per the cascadia capsule = Phase-A REV-2 contract: FINITE
   upper-triangle cells only, measured-station endpoints only,
   canonical sorted pair keys.
2. SELECTION day_records (w2_selection.select production path):
   {iso_day: [present ids]} over EXACTLY the engine's lookback frame
   [cutoff-(L-1), cutoff]; the producer refuses an incomplete frame
   typed BEFORE the engine sees it (the engine still enforces).
3. MF4 FEED (w2_calibration_runner feed contract, 1353Z).
4. MAG FEEDS per observatory (same contract), with the temporal
   boundary enforced AT THE PRODUCER (codex 1358Z item-3 alignment):
   unique, strictly increasing, minute-resolution timestamps inside
   [CAL_EPOCH, cutoff]; non-finite raw values REFUSED -- registered
   absence is None at the boundary, never NaN/Inf (the codex boundary
   normalization rule).

PROVENANCE: every builder returns (artifact, receipt); the receipt
binds the RAW input digests (computed BEFORE assembly), the output
digest, and the producer identity (this file's CRLF-normalized source
sha -- attested by the producer, recorded by the runner).

This module opens no window-2 value: it carries bytes it is handed and
refuses malformed shapes; it fetches nothing.
"""
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import f2g_sealed_run_instrument_cayley as SRI  # pinned digest formula
import w2_selection as WS

PRODUCER_NAME = "grassmann-w2-producer"
IDENTITY_SCHEMA = "f2g-w2-producer-identity-v1"
RECEIPT_SCHEMA = "f2g-w2-producer-receipt-v1"
CAL_EPOCH = "2026-01-01"          # matches w2_mag1.CAL_EPOCH
W2_CARRIERS = ("istanbul_marmara", "socal_coachella",
               "turkey_kahramanmaras", "cascadia")


class ProducerRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _canon_digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def producer_identity():
    """The producer's identity record: this file's CRLF-normalized
    source sha (the blob the manifest pins), recomputed at call time."""
    with open(os.path.abspath(__file__), "rb") as f:
        sha = hashlib.sha256(
            f.read().replace(b"\r\n", b"\n")).hexdigest()
    return {"schema": IDENTITY_SCHEMA, "name": PRODUCER_NAME,
            "code_blob_sha256": sha}


def _receipt(lane, inputs_sha256, artifact):
    return {"schema": RECEIPT_SCHEMA, "lane": lane,
            "inputs_sha256": dict(inputs_sha256),
            "output_sha256": _canon_digest(artifact),
            "producer_identity": producer_identity()}


# ------------------------------------------------------------- day capsules
def day_capsule(station_ids, matrix, measured_ids):
    """One carrier-day capsule from a coherence-matrix artifact.
    `station_ids` = the ordered axis of `matrix` (NxN, rows=cols);
    `measured_ids` = the stations measured that day. Edges = FINITE
    upper-triangle cells whose BOTH endpoints are measured, canonical
    sorted-pair keys (Phase-A REV-2 contract). None/NaN cells are
    ABSENT edges, not zeros. The station_index_digest is computed by
    the imported pinned formula, never reimplemented."""
    ids = [str(s) for s in station_ids]
    if len(set(ids)) != len(ids):
        raise ProducerRefusal("PRODUCER_STATION_DUPLICATE: axis ids")
    meas = sorted(str(s) for s in measured_ids)
    if len(set(meas)) != len(meas):
        raise ProducerRefusal("PRODUCER_STATION_DUPLICATE: measured")
    unknown = [s for s in meas if s not in set(ids)]
    if unknown:
        raise ProducerRefusal(
            f"PRODUCER_STATION_UNKNOWN: {unknown} not on the matrix "
            "axis")
    n = len(ids)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ProducerRefusal(
            f"PRODUCER_MATRIX_SHAPE: need {n}x{n}")
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
            if not math.isfinite(v):
                continue
            edges["|".join(sorted((ids[i], ids[j])))] = v
    return {"measured": meas,
            "station_index_digest": SRI.station_index_digest(meas),
            "edges": edges}


def assemble_producer_days(carrier, capsules_by_day):
    """{day: capsule} -> the adapter's per-carrier producer feed, with
    ISO-day validation. Returns ({carrier: {...}}, receipt)."""
    if carrier not in W2_CARRIERS:
        raise ProducerRefusal(f"PRODUCER_UNKNOWN_CARRIER: {carrier!r}")
    out = {}
    for day in sorted(capsules_by_day):
        try:
            date.fromisoformat(str(day))
        except ValueError:
            raise ProducerRefusal(
                f"PRODUCER_DAY_INVALID: {day!r}")
        out[str(day)] = capsules_by_day[day]
    feed = {carrier: out}
    return feed, _receipt("DAY_CAPSULES",
                          {"capsules": _canon_digest(out)}, feed)


# ------------------------------------------------------- selection records
def build_selection_records(carrier, presence_by_day, cutoff):
    """{day: iterable of present ids} -> the production day_records for
    w2_selection.select, refusing an incomplete/misaligned frame typed
    at the producer boundary. The engine re-validates; this refusal
    exists so a malformed frame is MINE to own, not a selection
    outcome."""
    if carrier not in W2_CARRIERS:
        raise ProducerRefusal(f"PRODUCER_UNKNOWN_CARRIER: {carrier!r}")
    try:
        cut = date.fromisoformat(str(cutoff))
    except ValueError:
        raise ProducerRefusal(f"PRODUCER_CUTOFF_INVALID: {cutoff!r}")
    lb = WS.LOOKBACK_DAYS
    expect = [(cut - timedelta(days=lb - 1 - i)).isoformat()
              for i in range(lb)]
    have = sorted(str(d) for d in presence_by_day)
    if have != expect:
        missing = sorted(set(expect) - set(have))
        extra = sorted(set(have) - set(expect))
        raise ProducerRefusal(
            f"PRODUCER_FRAME_INCOMPLETE: missing={missing[:4]} "
            f"extra={extra[:4]} (need exactly [{expect[0]}, "
            f"{expect[-1]}])")
    records = {d: sorted(str(s) for s in presence_by_day[d])
               for d in expect}
    return records, _receipt(
        "SELECTION_RECORDS",
        {"presence": _canon_digest(
            {d: sorted(str(s) for s in presence_by_day[d])
             for d in have})},
        {"carrier": carrier, "cutoff": str(cutoff),
         "day_records": records})


# --------------------------------------------------------------- MF4 feed
def build_mf4_feed(risk_by_region, catalog_rows, bboxes, regions,
                   snapshot_end, freeze_day):
    """Assemble the MF4 calibration feed (1353Z contract). Refusals:
    region without bbox or risk series; non-finite risk; catalog rows
    missing keys; catalog events dated after snapshot_end (the snapshot
    claims 'as of snapshot_end' -- later events in it are a staleness
    lie at the source); snapshot_end after freeze_day."""
    inputs = {"risk_by_region": _canon_digest(risk_by_region),
              "catalog_rows": _canon_digest(catalog_rows),
              "bboxes": _canon_digest(bboxes)}
    if str(snapshot_end) > str(freeze_day):
        raise ProducerRefusal(
            f"PRODUCER_MF4_FREEZE_ORDER: snapshot_end {snapshot_end} "
            f"> freeze_day {freeze_day}")
    regs = sorted(str(r) for r in regions)
    for r in regs:
        if r not in bboxes:
            raise ProducerRefusal(f"PRODUCER_MF4_REGION_UNBOUND: "
                                  f"{r} has no bbox")
        if r not in risk_by_region:
            raise ProducerRefusal(f"PRODUCER_MF4_REGION_UNBOUND: "
                                  f"{r} has no risk series")
        for d, v in risk_by_region[r].items():
            if v is None or not math.isfinite(float(v)):
                raise ProducerRefusal(
                    f"PRODUCER_MF4_NONFINITE: {r} {d} {v!r}")
    snap = []
    for row in catalog_rows:
        for k in ("day", "lat", "lon", "mag"):
            if k not in row:
                raise ProducerRefusal(
                    f"PRODUCER_MF4_CATALOG_ROW_INCOMPLETE: {k}")
        if str(row["day"]) > str(snapshot_end):
            raise ProducerRefusal(
                f"PRODUCER_MF4_POST_SNAPSHOT_EVENT: {row['day']} > "
                f"snapshot_end {snapshot_end}")
        snap.append({"day": str(row["day"]),
                     "lat": float(row["lat"]),
                     "lon": float(row["lon"]),
                     "mag": float(row["mag"])})
    feed = {"risk_by_region": {r: {str(d): float(v)
                                   for d, v in risk_by_region[r]
                                   .items()}
                               for r in regs},
            "catalog_snapshot": snap,
            "snapshot_end": str(snapshot_end),
            "freeze_day": str(freeze_day),
            "bboxes": {r: dict(bboxes[r]) for r in regs},
            "regions": regs}
    return feed, _receipt("MF4_FEED", inputs, feed)


# --------------------------------------------------------------- MAG feeds
def _validate_times(times, cutoff):
    lo = date.fromisoformat(CAL_EPOCH)
    hi = date.fromisoformat(str(cutoff))
    prev = None
    for t in times:
        try:
            d = datetime.fromisoformat(str(t).replace("Z", ""))
        except ValueError:
            raise ProducerRefusal(f"PRODUCER_MAG_TIME_INVALID: {t!r}")
        if d.second or d.microsecond:
            raise ProducerRefusal(
                f"PRODUCER_MAG_TIME_RESOLUTION: {t!r} is not a "
                "minute stamp")
        if not (lo <= d.date() <= hi):
            raise ProducerRefusal(
                f"PRODUCER_MAG_TIME_OUT_OF_INTERVAL: {t!r} outside "
                f"[{CAL_EPOCH}, {cutoff}]")
        if prev is not None:
            if d == prev:
                raise ProducerRefusal(
                    f"PRODUCER_MAG_TIME_DUPLICATE: {t!r}")
            if d < prev:
                raise ProducerRefusal(
                    f"PRODUCER_MAG_TIME_ORDER: {t!r} after "
                    f"{prev.isoformat()}")
        prev = d


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
        if not math.isfinite(v):
            raise ProducerRefusal(
                f"PRODUCER_MAG_NONFINITE: {name} carries {v!r}; "
                "registered absence is None at the boundary")
        out.append(v)
    return out


def build_mag_feed(observatory, lon_east, times, components, weather,
                   m3_reference, cutoff):
    """Assemble one per-observatory MAG calibration feed (1353Z
    contract) with the temporal boundary enforced at the producer:
    unique, strictly increasing, minute-resolution stamps inside
    [CAL_EPOCH, cutoff]; series aligned to times; non-finite refused
    (absence is None)."""
    times = [str(t) for t in times]
    _validate_times(times, cutoff)
    n = len(times)
    inputs = {"times": _canon_digest(times)}
    comps = {}
    for c in ("X", "Y"):
        if c not in components:
            raise ProducerRefusal(
                f"PRODUCER_MAG_COMPONENT_MISSING: {observatory}:{c}")
        comps[c] = _normalize_series(f"components:{c}",
                                     components[c], n)
        inputs[f"component_{c}"] = _canon_digest(comps[c])
    wx = {}
    for name in sorted(weather):
        wx[name] = _normalize_series(f"weather:{name}",
                                     weather[name], n)
        inputs[f"weather_{name}"] = _canon_digest(wx[name])
    feed = {"observatory": str(observatory),
            "lon_east": float(lon_east),
            "times": times, "components": comps, "weather": wx,
            "m3_reference": (str(m3_reference)
                             if m3_reference else None)}
    return feed, _receipt("MAG_FEED", inputs, feed)


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

    # identity: recompute matches
    ident = producer_identity()
    assert ident["name"] == PRODUCER_NAME \
        and len(ident["code_blob_sha256"]) == 64
    assert producer_identity() == ident

    # --- day capsule: hand 4-station matrix, one NaN cell, one
    # unmeasured station; exact expected edges ---
    ids4 = ["CC.B", "UW.A", "UW.C", "UW.D"]
    m = [[1.0, 0.5, float("nan"), 0.2],
         [0.5, 1.0, 0.7, None],
         [0.0, 0.7, 1.0, 0.9],
         [0.2, 0.0, 0.9, 1.0]]
    cap = day_capsule(ids4, m, ["CC.B", "UW.A", "UW.C"])
    assert cap["measured"] == ["CC.B", "UW.A", "UW.C"]
    # upper triangle among measured: (B,A)=.5, (B,C)=NaN->absent,
    # (A,C)=.7; D unmeasured -> its cells absent
    assert cap["edges"] == {"CC.B|UW.A": 0.5, "UW.A|UW.C": 0.7}, \
        cap["edges"]
    assert cap["station_index_digest"] == SRI.station_index_digest(
        ["CC.B", "UW.A", "UW.C"])
    assert refuses(lambda: day_capsule(ids4, m, ["ZZ.Q"]),
                   "PRODUCER_STATION_UNKNOWN")
    assert refuses(lambda: day_capsule(ids4, m[:3], ids4[:2]),
                   "PRODUCER_MATRIX_SHAPE")
    assert refuses(lambda: day_capsule(["A", "A"], [[1, 1], [1, 1]],
                                       ["A"]),
                   "PRODUCER_STATION_DUPLICATE")

    # --- adapter e2e: capsules -> build_family_panel -> b2b engine ---
    reg10 = sorted([f"A{i}" for i in range(5)]
                   + [f"B{i}" for i in range(5)])
    cal6 = [f"2026-10-{i:02d}" for i in range(1, 7)]
    idx = {s: i for i, s in enumerate(reg10)}
    big = [[(5.0 if (reg10[i][0] == reg10[j][0]) else 0.1)
            for j in range(10)] for i in range(10)]
    caps = {d: day_capsule(reg10, big, reg10) for d in cal6}
    feed, rec = assemble_producer_days("cascadia", caps)
    assert rec["lane"] == "DAY_CAPSULES" \
        and rec["output_sha256"] == _canon_digest(feed)
    reg_rec = {"registries": {"cascadia": {
        "selected": reg10, "churn": 1.0, "typing": None}}}
    panel = ACC.build_family_panel(cal6, reg_rec, feed)
    res = W2B.w2_b2b_family(panel, doc_sha256="ab" * 32, n_draws=49)
    assert res["runs_by_carrier"]["cascadia"] == 1 \
        and res["p_value"] == 1.0, res
    assert refuses(lambda: assemble_producer_days("nope", caps),
                   "PRODUCER_UNKNOWN_CARRIER")
    assert refuses(
        lambda: assemble_producer_days("cascadia", {"not-a-day": {}}),
        "PRODUCER_DAY_INVALID")

    # --- selection e2e: 90-day full-presence frame -> production
    # select ---
    cut = "2026-08-24"
    lb = WS.LOOKBACK_DAYS
    frame = [(date.fromisoformat(cut) - timedelta(days=lb - 1 - i))
             .isoformat() for i in range(lb)]
    pres = {d: list(reg10) for d in frame}
    drecs, rec2 = build_selection_records("cascadia", pres, cut)
    sel = WS.select("cascadia", drecs, cut)
    assert sel["selected"] == reg10 and sel["typing"] is None
    short = {d: pres[d] for d in frame[1:]}
    assert refuses(
        lambda: build_selection_records("cascadia", short, cut),
        "PRODUCER_FRAME_INCOMPLETE")

    # --- MF4 e2e through the real runner ---
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
    mf4_feed, mf4_rec = build_mf4_feed(
        risk, catalog, {"ra": bbox, "rb": bbox}, ["ra", "rb"],
        "2026-02-08", "2026-02-10")
    out = CAL.run_mf4_calibration(repo, mf4_feed, "2026-02-09",
                                  producer_identity())
    v = CAL.verify_receipt(repo, out["receipt"])
    assert v == {"verified_ledgers": 1, "lane": "MF4"}
    # determinism: rebuild -> identical feed digest
    mf4_feed2, mf4_rec2 = build_mf4_feed(
        risk, catalog, {"ra": bbox, "rb": bbox}, ["ra", "rb"],
        "2026-02-08", "2026-02-10")
    assert mf4_rec2["output_sha256"] == mf4_rec["output_sha256"]
    late = catalog + [{"day": "2026-02-09", "lat": 35.0,
                       "lon": -120.0, "mag": 5.0}]
    assert refuses(
        lambda: build_mf4_feed(risk, late, {"ra": bbox, "rb": bbox},
                               ["ra", "rb"], "2026-02-08",
                               "2026-02-10"),
        "PRODUCER_MF4_POST_SNAPSHOT_EVENT")
    assert refuses(
        lambda: build_mf4_feed(risk, catalog, {"ra": bbox},
                               ["ra", "rb"], "2026-02-08",
                               "2026-02-10"),
        "PRODUCER_MF4_REGION_UNBOUND")
    assert refuses(
        lambda: build_mf4_feed(risk, catalog,
                               {"ra": bbox, "rb": bbox}, ["ra", "rb"],
                               "2026-02-11", "2026-02-10"),
        "PRODUCER_MF4_FREEZE_ORDER")
    bad_risk = json.loads(json.dumps(risk))
    bad_risk["ra"][days[0]] = None
    assert refuses(
        lambda: build_mf4_feed(bad_risk, catalog,
                               {"ra": bbox, "rb": bbox}, ["ra", "rb"],
                               "2026-02-08", "2026-02-10"),
        "PRODUCER_MF4_NONFINITE")

    # --- MAG e2e: two observatories through the real runner ---
    n = 3000
    times = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat()
             for i in range(n)]
    wx = {"symh": rng.normal(size=n).tolist()}

    def mk(name, ref):
        f, _ = build_mag_feed(
            name, -120.0, times,
            {"X": rng.normal(20000, 5, size=n).tolist(),
             "Y": rng.normal(4000, 5, size=n).tolist()},
            wx, ref, "2026-08-24")
        return f
    feeds = {"FRN": mk("FRN", "TUC"), "TUC": mk("TUC", None)}
    out = CAL.run_mag_calibration(repo, feeds, "2026-08-24",
                                  producer_identity())
    v = CAL.verify_receipt(repo, out["receipt"])
    assert v["verified_ledgers"] == 6
    assert "FRN:TUC:X" in out["results"]["m3"]
    # temporal doctors
    xs = rng.normal(20000, 5, size=4).tolist()
    base = dict(components={"X": xs, "Y": xs}, weather={"symh":
                [0.0] * 4}, m3_reference=None)
    t4 = times[:4]
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, [t4[0], t4[0], t4[2], t4[3]], base["components"],
        base["weather"], None, "2026-08-24"),
        "PRODUCER_MAG_TIME_DUPLICATE")
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, [t4[1], t4[0], t4[2], t4[3]], base["components"],
        base["weather"], None, "2026-08-24"),
        "PRODUCER_MAG_TIME_ORDER")
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, ["2026-09-01T00:00:00"] + t4[1:],
        base["components"], base["weather"], None, "2026-08-24"),
        "PRODUCER_MAG_TIME_OUT_OF_INTERVAL")
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, ["2026-01-01T00:00:30"] + t4[1:],
        base["components"], base["weather"], None, "2026-08-24"),
        "PRODUCER_MAG_TIME_RESOLUTION")
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, t4, {"X": xs[:3], "Y": xs},
        base["weather"], None, "2026-08-24"),
        "PRODUCER_MAG_ALIGNMENT")
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, t4, {"X": [1.0, float("nan"), 1.0, 1.0],
                            "Y": xs},
        base["weather"], None, "2026-08-24"),
        "PRODUCER_MAG_NONFINITE")
    assert refuses(lambda: build_mag_feed(
        "FRN", -120.0, t4, {"Y": xs}, base["weather"], None,
        "2026-08-24"),
        "PRODUCER_MAG_COMPONENT_MISSING")
    # None absence is carried, not refused
    f_none, _ = build_mag_feed(
        "FRN", -120.0, t4, {"X": [1.0, None, 1.0, 1.0], "Y": xs},
        base["weather"], None, "2026-08-24")
    assert f_none["components"]["X"][1] is None

    print("w2_producer selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
