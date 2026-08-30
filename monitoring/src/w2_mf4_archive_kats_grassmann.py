#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF4 archive + catalog-contract KATs (grassmann) -- the lock list
from the codex 2026-08-29T17:58Z ruling, exercised against the REAL
builder/verifier/acquisition functions (never re-typed copies), in
throwaway sandboxes; the committed capsule and store are never
touched.

Archive locks (ruling 3): A1 missing raw object with still-present
digest; A2 row mutation; A3 string/bool/NaN risk refusals; A4
duplicate region-day; A5 support/census drift (rows-file digest);
A6 golden replay of the REAL committed capsule (read-only).
Training-digest drift is deferred to the v2 finalization KATs after
the catalog acquisition (two-stage, disclosed).

Bbox/catalog locks (ruling 2 + contract): B1 alias-direction
reversal refuses; B2 Tokyo->Tohoku mapping refuses; B3 bbox
expansion to the dashboard REGION_BOUNDS refuses the pin check; B4
boundary event admitted exactly on the bbox edge, refused outside;
B5 bbox-coordinate mutation refuses; C1 top-five lossy projection
refused MF4_CATALOG_LOSSY_VIEW (feeding >5 qualifying events proves
the check would catch a truncated view); C2 query-limit refusal; C3
malformed/null field refusals; C4 temporal-filter refusal; C5
duplicate/cross-region-inconsistent ID refusals.
"""
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import w2_mf4_archive_capsule_gen_grassmann as GEN
import w2_mf4_catalog_acquire_grassmann as ACQ

PASS, FAIL = [], []


def check(name, fn, expect_code=None):
    try:
        fn()
        ok = expect_code is None
        why = "" if ok else f"expected {expect_code}, no refusal"
    except SystemExit as e:
        ok = expect_code is not None and expect_code in str(e)
        why = "" if ok else f"got {e}"
    except Exception as e:                                  # noqa: BLE001
        ok = False
        why = f"unexpected {type(e).__name__}: {e}"
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" -- {why}" if why else ""))


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def sandbox_build(mutate=None):
    """Run the REAL build() against a 3-day, 2-region fixture in a
    sandbox; return (sandbox_root, module) with constants restored by
    the caller via ctx manager semantics."""
    root = tempfile.mkdtemp(prefix="mf4kat_")
    src = os.path.join(root, "monitoring", "data", "ensemble_results")
    os.makedirs(src)
    days = ["2026-01-01", "2026-01-02", "2026-01-03"]
    for i, d in enumerate(days):
        doc = {"date": d, "timestamp": d + "T07:00:00",
               "regions": {
                   "regA": {"combined_risk": 0.1 * (i + 1),
                            "persistence": 0.05},
                   "regB": {"combined_risk": 0.2, "persistence": None}}}
        if mutate:
            doc = mutate(d, doc)
        with open(os.path.join(src, f"ensemble_{d}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(doc, f)
    return root, days


class Sandbox:
    """Temporarily repoint the builder at a sandbox repo/store and a
    2-region set; restore everything on exit."""

    def __init__(self, mutate=None):
        self.mutate = mutate

    def __enter__(self):
        self.root, self.days = sandbox_build(self.mutate)
        self.saved = {k: getattr(GEN, k) for k in
                      ("REPO", "SRC_DIR", "STORE_DIR", "MONITOR_REGIONS",
                       "TYPED_EXCLUSIONS", "ALIAS", "ADMITTED",
                       "CAL_START", "CAL_END", "MATURITY_REL")}
        GEN.REPO = self.root
        GEN.SRC_DIR = os.path.join(self.root, "monitoring", "data",
                                   "ensemble_results")
        GEN.STORE_DIR = os.path.join(self.root, "store")
        GEN.OUT_DIR = os.path.join(self.root, "docs",
                                   "f2g_window2_execution")
        GEN.MONITOR_REGIONS = ["regA", "regB"]
        GEN.TYPED_EXCLUSIONS = {}
        GEN.ALIAS = {}
        GEN.ADMITTED = []          # skip bbox recompute in sandbox
        GEN.CAL_START = dt.date(2026, 1, 1)
        GEN.CAL_END = dt.date(2026, 1, 3)
        mat = {"gate_a_calibration_ledger": {
            "calibration_interval": ["2026-01-01", "2026-01-03"],
            "freeze_day": "2026-01-10", "snapshot_end": "2026-01-09"}}
        mp = os.path.join(self.root, "maturity.json")
        json.dump(mat, open(mp, "w", encoding="utf-8"))
        GEN.MATURITY_REL = "maturity.json"
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(GEN, k, v)
        shutil.rmtree(self.root, ignore_errors=True)
        return False


def _rebind(root):
    """After a deliberate sandbox mutation, re-pin the capsule's rows
    identity and the receipt's capsule digest so the DEEPER verifier
    check under test is the one that fires."""
    cp = os.path.join(root, GEN.CAPSULE_REL)
    rp = os.path.join(root, GEN.RECEIPT_REL)
    rows_raw = open(os.path.join(root, GEN.ROWS_REL), "rb").read()
    cap = json.loads(open(cp, encoding="utf-8").read())
    cap["rows_file"]["sha256"] = _sha(rows_raw)
    cap["rows_file"]["bytes"] = len(rows_raw)
    cap["rows_file"]["rows"] = rows_raw.decode().count("\n") or \
        len(rows_raw.decode().splitlines())
    cap["rows_file"]["rows"] = len(rows_raw.decode().splitlines())
    cap_bytes = (json.dumps(cap, indent=1, sort_keys=True)
                 + "\n").encode("utf-8")
    open(cp, "wb").write(cap_bytes)
    rec = json.loads(open(rp, encoding="utf-8").read())
    rec["capsule"]["sha256"] = _sha(cap_bytes)
    rec["rows_sha256"] = _sha(rows_raw)
    open(rp, "wb").write((json.dumps(rec, indent=1, sort_keys=True)
                          + "\n").encode("utf-8"))


def _mutate_capsule(root, fn, rebind=True):
    cp = os.path.join(root, GEN.CAPSULE_REL)
    cap = json.loads(open(cp, encoding="utf-8").read())
    fn(cap)
    cap_bytes = (json.dumps(cap, indent=1, sort_keys=True)
                 + "\n").encode("utf-8")
    open(cp, "wb").write(cap_bytes)
    if rebind:
        rp = os.path.join(root, GEN.RECEIPT_REL)
        rec = json.loads(open(rp, encoding="utf-8").read())
        rec["capsule"]["sha256"] = _sha(cap_bytes)
        open(rp, "wb").write((json.dumps(rec, indent=1, sort_keys=True)
                              + "\n").encode("utf-8"))


def kat_a1_missing_object():
    with Sandbox() as sb:
        GEN.build()
        cap = json.load(open(os.path.join(sb.root, GEN.CAPSULE_REL),
                             encoding="utf-8"))
        obj = next(iter(cap["raw_source_store"]["inventory"].values()))
        os.remove(os.path.join(GEN.STORE_DIR, obj["object"]))
        GEN.verify_capsule()


def kat_a2_row_mutation():
    with Sandbox() as sb:
        GEN.build()
        rp = os.path.join(sb.root, GEN.ROWS_REL)
        lines = open(rp, encoding="utf-8").readlines()
        r = json.loads(lines[0])
        r["combined_risk"] = 0.999
        lines[0] = json.dumps(r, sort_keys=True) + "\n"
        open(rp, "w", encoding="utf-8", newline="\n").writelines(lines)
        _rebind(sb.root)
        GEN.verify_capsule()


def _risk_mutator(value):
    def mutate(day, doc):
        if day == "2026-01-02":
            doc = copy.deepcopy(doc)
            doc["regions"]["regA"]["combined_risk"] = value
        return doc
    return mutate


def kat_a4_duplicate_region_day():
    with Sandbox() as sb:
        GEN.build()
        rp = os.path.join(sb.root, GEN.ROWS_REL)
        lines = open(rp, encoding="utf-8").readlines()
        lines.append(lines[0])
        open(rp, "w", encoding="utf-8", newline="\n").writelines(lines)
        _rebind(sb.root)
        GEN.verify_capsule()


def kat_a5_census_drift():
    with Sandbox() as sb:
        GEN.build()
        _mutate_capsule(sb.root,
                        lambda c: c["rows_file"].update(sha256="0" * 64))
        GEN.verify_capsule()


def kat_a6_golden_real_capsule():
    res = GEN.verify_capsule()
    assert res["objects_verified"] == 307, res
    assert res["rows_verified"] == 4298, res


def kat_a7_store_portability():
    """Codex 2359Z blocker 3: ONE committed capsule must verify
    through two physical aliases of the same content-addressed store
    without any identity change. Builds in a sandbox, copies the
    store to a second physical root, and verifies the byte-identical
    capsule from both."""
    with Sandbox() as sb:
        GEN.build()
        cap_bytes = open(os.path.join(sb.root, GEN.CAPSULE_REL),
                         "rb").read()
        alias = os.path.join(sb.root, "store_alias_2")
        shutil.copytree(GEN.STORE_DIR, alias)
        r1 = GEN.verify_capsule()
        saved_store = GEN.STORE_DIR
        try:
            GEN.STORE_DIR = alias
            r2 = GEN.verify_capsule()
        finally:
            GEN.STORE_DIR = saved_store
        assert r1 == r2, (r1, r2)
        after = open(os.path.join(sb.root, GEN.CAPSULE_REL),
                     "rb").read()
        assert after == cap_bytes, "capsule identity moved with the store path"
        assert b"local_physical_root" not in cap_bytes, \
            "environment-specific path leaked into capsule identity"


def kat_b1_alias_reversal():
    bad = dict(ACQ.ALIAS)
    bad.pop("socal_saf_coachella")
    bad["socal_coachella"] = "socal_saf_coachella"
    saved = ACQ.ALIAS
    try:
        ACQ.ALIAS = bad
        ACQ.build_bboxes()
    finally:
        ACQ.ALIAS = saved


def kat_b2_tokyo_tohoku():
    saved_adm, saved_alias = ACQ.ADMITTED, dict(ACQ.ALIAS)
    try:
        ACQ.ADMITTED = ACQ.ADMITTED + ["tokyo_kanto"]
        ACQ.ALIAS = dict(saved_alias, tokyo_kanto="japan_tohoku")
        ACQ.build_bboxes()
    finally:
        ACQ.ADMITTED, ACQ.ALIAS = saved_adm, saved_alias


def kat_b3_dashboard_bbox_expansion():
    saved = ACQ.PINNED_BBOXES
    try:
        bad = {k: dict(v) for k, v in saved.items()}
        # the dashboard REGION_BOUNDS anchorage box (wider than segments)
        bad["anchorage"] = {"min_lat": 58.0, "max_lat": 63.0,
                            "min_lon": -154.0, "max_lon": -146.0}
        ACQ.PINNED_BBOXES = bad
        ACQ.build_bboxes()
    finally:
        ACQ.PINNED_BBOXES = saved


def kat_b5_bbox_coordinate_mutation():
    saved = ACQ.PINNED_BBOXES
    try:
        bad = {k: dict(v) for k, v in saved.items()}
        bad["kaikoura"]["min_lat"] = -43.0001
        ACQ.PINNED_BBOXES = bad
        ACQ.build_bboxes()
    finally:
        ACQ.PINNED_BBOXES = saved


def _geojson(events, count=None, ftype="Feature", gtype="Point",
             top="FeatureCollection"):
    feats = [{"type": ftype, "id": e[0],
              "properties": {"mag": e[3], "time": e[4]},
              "geometry": {"type": gtype,
                           "coordinates": [e[2], e[1], 10.0]}}
             for e in events]
    return json.dumps({"type": top,
                       "metadata": {"count": len(feats)
                                    if count is None else count},
                       "features": feats}).encode("utf-8")


BB = {"min_lat": 30.0, "max_lat": 40.0, "min_lon": 100.0,
      "max_lon": 110.0}
T0 = int(dt.datetime(2026, 1, 15,
                     tzinfo=dt.timezone.utc).timestamp() * 1000)
KURL = "kat://expected-url"


def kat_b4_boundary_event():
    # exactly on the bbox edge: admitted (inclusive)
    raw = _geojson([("edge", 40.0, 110.0, 4.5, T0)])
    evs = ACQ.validate_events("katreg", BB, raw, KURL)
    assert len(evs) == 1
    # just outside: refused
    raw2 = _geojson([("out", 40.0001, 110.0, 4.5, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw2, KURL)
        raise AssertionError("outside-bbox event admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_SPATIAL_FILTER" in str(e)


def kat_c1_lossy_view():
    """>5 qualifying events must ALL survive validation -- a top-five
    projection (the dashboard's serialization) loses events and is
    detected as a count mismatch, refused as a lossy view."""
    full = [(f"ev{i}", 35.0 + i * 0.1, 105.0, 4.0 + i * 0.1, T0 + i)
            for i in range(8)]
    evs = ACQ.validate_events("katreg", BB, _geojson(full), KURL)
    assert len(evs) == 8, "full view must retain all 8"
    top5 = ACQ.validate_events("katreg", BB, _geojson(full[:5]), KURL)
    if len(top5) != len(full):
        raise SystemExit("REFUSED MF4_CATALOG_LOSSY_VIEW: top-five "
                         f"projection kept {len(top5)}/{len(full)}")


def kat_c2_query_limit():
    saved = ACQ.LIMIT
    try:
        ACQ.LIMIT = 3
        raw = _geojson([(f"e{i}", 35.0, 105.0, 4.1, T0 + i)
                        for i in range(3)])
        ACQ.validate_events("katreg", BB, raw, KURL)
    finally:
        ACQ.LIMIT = saved


def kat_c3_malformed_fields():
    for bad in [("nullmag", 35.0, 105.0, None, T0),
                ("nulltime", 35.0, 105.0, 4.2, None),
                ("strmag", 35.0, 105.0, "4.2", T0)]:
        try:
            ACQ.validate_events("katreg", BB, _geojson([bad]), KURL)
            raise AssertionError(f"{bad[0]} admitted")
        except SystemExit as e:
            assert "MF4_CATALOG_MALFORMED" in str(e), e
    raw = json.dumps({"type": "FeatureCollection",
                      "metadata": {"count": 1}, "features": [
        {"type": "Feature", "id": "nullcoord",
         "properties": {"mag": 4.2, "time": T0},
         "geometry": {"type": "Point", "coordinates": None}}]}
        ).encode("utf-8")
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("null coords admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e)


def kat_c4_temporal_filter():
    late = int(dt.datetime(2026, 8, 28, 0, 0, 1,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)
    ACQ.validate_events("katreg", BB,
                        _geojson([("late", 35.0, 105.0, 4.2, late)]),
                        KURL)


def kat_c5_duplicate_and_inconsistent_ids():
    dup = _geojson([("same", 35.0, 105.0, 4.2, T0),
                    ("same", 35.1, 105.1, 4.3, T0 + 1)])
    try:
        ACQ.validate_events("katreg", BB, dup, KURL)
        raise AssertionError("duplicate id admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_DUPLICATE_ID" in str(e)
    a = ACQ.validate_events("r1", BB, _geojson([("x", 35.0, 105.0,
                                                 4.2, T0)]), KURL)
    b = ACQ.validate_events("r2", BB, _geojson([("x", 35.5, 105.0,
                                                 4.2, T0)]), KURL)
    try:
        ACQ.canonical_event_table({"r1": a, "r2": b})
        raise AssertionError("inconsistent shared id admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_INCONSISTENT_ID" in str(e)


def _geojson27(events, url=None, api="2.7.0", status=200,
               limit=None, offset=1, generated=1788055936000,
               drop=()):
    """Fixture in the OBSERVED count-absent ComCat API-2.7.0 metadata
    frame (codex 0219Z item 1 variant 2)."""
    feats = [{"type": "Feature", "id": e[0],
              "properties": {"mag": e[3], "time": e[4]},
              "geometry": {"type": "Point",
                           "coordinates": [e[2], e[1], 10.0]}}
             for e in events]
    meta = {"generated": generated, "url": url or KURL,
            "title": "USGS Earthquakes", "status": status,
            "api": api, "limit": limit if limit is not None
            else ACQ.LIMIT, "offset": offset}
    for k in drop:
        meta.pop(k, None)
    return json.dumps({"type": "FeatureCollection", "metadata": meta,
                       "features": feats}).encode("utf-8")


def _c6_series():
    """Codex 0219Z item-1 lock battery: exactly two registered
    metadata variants; every field of the count-absent 2.7.0 frame
    refuses typed on absence or mutation."""
    good = ("g1", 35.0, 105.0, 4.5, T0)

    def ok():
        evs = ACQ.validate_events("katreg", BB, _geojson27([good]),
                                  KURL)
        assert len(evs) == 1
    check("C6a observed count-absent 2.7.0 frame passes", ok)
    check("C6b wrong api version",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], api="3.0.0"), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6c missing api field",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], drop=("api",)), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6d wrong status",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], status=204), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6e missing status",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], drop=("status",)),
              KURL), "MF4_CATALOG_METADATA_FRAME")
    check("C6f url mismatch",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], url="kat://other"),
              KURL), "MF4_CATALOG_METADATA_FRAME")
    check("C6g missing url",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], drop=("url",)), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6h wrong limit",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], limit=100), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6i bool limit",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], limit=True), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6j wrong offset",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], offset=2), KURL),
          "MF4_CATALOG_METADATA_FRAME")
    check("C6k missing generated",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], drop=("generated",)),
              KURL), "MF4_CATALOG_METADATA_FRAME")
    check("C6l non-int generated",
          lambda: ACQ.validate_events(
              "katreg", BB, _geojson27([good], generated="now"),
              KURL), "MF4_CATALOG_METADATA_FRAME")

    def limit_hit():
        saved = ACQ.LIMIT
        try:
            ACQ.LIMIT = 2
            evs = [(f"e{i}", 35.0, 105.0, 4.1, T0 + i)
                   for i in range(2)]
            ACQ.validate_events("katreg", BB, _geojson27(evs), KURL)
        finally:
            ACQ.LIMIT = saved
    check("C6m count-absent truncation guard", limit_hit,
          "MF4_CATALOG_QUERY_LIMIT")

    def count_mismatch_still():
        ACQ.validate_events("katreg", BB, _geojson([good], count=99),
                            KURL)
    check("C6n count-present mismatch still refuses",
          count_mismatch_still, "MF4_CATALOG_COUNT_MISMATCH")


ATTEMPT1_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                            "docs", "f2g_window2_execution",
                            "mf4_catalog_attempt1_refusal")


def kat_a8_attempt1_capsule_integrity():
    """Codex 0219Z item 2: the attempt-1 refusal evidence is
    byte-identical to the identities bound in its capsule manifest,
    and no snapshot was published."""
    man = json.loads(open(os.path.join(
        ATTEMPT1_DIR, "ATTEMPT1_REFUSAL_CAPSULE.json"),
        encoding="utf-8").read())
    assert man["schema"] == \
        "geospec-mf4-catalog-attempt1-refusal-capsule-v1"
    assert man["queries_fired"] == 1
    assert man["queries_not_fired"] == 12
    assert man["snapshot_published"] is False
    assert man["owner_go_status"] == "SPENT"
    for name, ident in man["sealed_files"].items():
        raw = open(os.path.join(ATTEMPT1_DIR, name), "rb").read()
        assert len(raw) == ident["bytes"], name
        assert _sha(raw) == ident["sha256"], name
    raw_resp = open(os.path.join(ATTEMPT1_DIR,
                                 "raw_anchorage.geojson"),
                    "rb").read()
    doc = json.loads(raw_resp.decode("utf-8"))
    assert "count" not in doc["metadata"], \
        "sealed evidence contradicts the count-absent finding"
    assert doc["metadata"]["api"] == "2.7.0"
    assert len(doc["features"]) == man["fired_query"]["features"]
    # attempt 1 published nothing; a snapshot may exist ONLY from a
    # LATER authority -- it must not reuse attempt-1's spent go or
    # its sealed response identity
    snap_rec = os.path.join(os.path.dirname(ATTEMPT1_DIR),
                            "mf4_catalog_snapshot",
                            "acquisition_receipt_v1.json")
    if os.path.exists(snap_rec):
        rec2 = json.loads(open(snap_rec, encoding="utf-8").read())
        go2 = rec2["authorization_content"]["owner_fire_go"][
            "source_framework_commit"]
        assert go2 != man["authority_chain"][
            "owner_go_framework_commit"], \
            "published snapshot reuses attempt-1's SPENT owner go"
        assert rec2["snapshot_sha256"] != _sha(raw_resp), \
            "published snapshot is the attempt-1 sealed response"


def kat_d1_fire_requires_authorization():
    def bomb(url, timeout=None):
        raise AssertionError("HTTP attempted without authorization")
    ACQ.fire(None, opener=bomb)


def kat_d2_partial_failure_staging():
    import tempfile, shutil as _sh
    root = tempfile.mkdtemp(prefix="mf4fire_")
    saved_final = ACQ.FINAL_DIR
    saved_verify = ACQ.verify_fire_authorization
    try:
        ACQ.FINAL_DIR = os.path.join(root, "snap")
        ACQ.verify_fire_authorization = lambda p: (
            {"stub": True}, {"file": "stub", "sha256": "0" * 64,
                             "bytes": 0})
        calls = {"n": 0}

        class FakeResp:
            def __init__(self, raw):
                self.raw = raw
                self.status = 200
                import email.message
                m = email.message.Message()
                m["Content-Type"] = "application/json"
                self.headers = m
                self._url = None
            def geturl(self):
                return self._url
            def read(self):
                return self.raw
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_open(url, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raw = _geojson([("ok1", 61.0, -150.0, 4.5, T0)])
            else:
                raw = _geojson([("", 61.0, -150.0, 4.5, T0)])  # bad id
            r = FakeResp(raw)
            r._url = url
            return r

        # region 1 bbox admits (61,-150); region 2 (campi) will fail on
        # the missing id BEFORE its spatial check.
        try:
            ACQ.fire("stubauth", opener=fake_open)
            raise AssertionError("partial failure did not refuse")
        except SystemExit as e:
            assert "MF4_CATALOG" in str(e), e
        staging = ACQ.FINAL_DIR + ".staging"
        assert os.path.isfile(os.path.join(
            staging, "REFUSAL_MANIFEST.json")), "no refusal manifest"
        assert os.path.isfile(os.path.join(
            staging, "raw_anchorage.geojson")), "region-1 seal missing"
        assert not os.path.exists(ACQ.FINAL_DIR), "success dir exists"
        # second fire refuses staging reuse BEFORE any request
        def bomb(url, timeout=None):
            raise AssertionError("re-query attempted on continuation")
        try:
            ACQ.fire("stubauth", opener=bomb)
            raise AssertionError("second fire did not refuse")
        except SystemExit as e:
            assert "MF4_FIRE_STAGING_EXISTS" in str(e), e
    finally:
        ACQ.FINAL_DIR = saved_final
        ACQ.verify_fire_authorization = saved_verify
        _sh.rmtree(root, ignore_errors=True)


def kat_d3_parser_closures():
    good = ("g1", 35.0, 105.0, 4.5, T0)
    # NaN / Infinity constants in the JSON text
    for const in ("NaN", "Infinity"):
        raw = _geojson([good]).replace(b"4.5", const.encode())
        try:
            ACQ.validate_events("katreg", BB, raw, KURL)
            raise AssertionError(f"{const} admitted")
        except SystemExit as e:
            assert "MF4_CATALOG_NONFINITE_JSON" in str(e), e
    # boolean magnitude
    raw = _geojson([("b", 35.0, 105.0, True, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("bool mag admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e), e
    # magnitude below the registered threshold
    raw = _geojson([("lo", 35.0, 105.0, 3.0, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("mag 3.0 admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MAG_BELOW_THRESHOLD" in str(e), e
    # metadata.count mismatch
    raw = _geojson([good], count=99)
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("count mismatch admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_COUNT_MISMATCH" in str(e), e
    # reverse time order
    raw = _geojson([("t2", 35.0, 105.0, 4.5, T0 + 1000),
                    ("t1", 35.0, 105.0, 4.5, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("reverse order admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_ORDER_VIOLATION" in str(e), e
    # non-Point geometry
    raw = _geojson([good], gtype="LineString")
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("non-Point admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e), e
    # non-FeatureCollection top level
    raw = _geojson([good], top="Whatever")
    try:
        ACQ.validate_events("katreg", BB, raw, KURL)
        raise AssertionError("non-FeatureCollection admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e), e


def kat_d4_canonical_table():
    a = ACQ.validate_events("r1", BB, _geojson([
        ("x1", 35.0, 105.0, 4.5, T0),
        ("x2", 36.0, 106.0, 4.6, T0 + 5)]), KURL)
    b = ACQ.validate_events("r2", BB, _geojson([
        ("x1", 35.0, 105.0, 4.5, T0)]), KURL)    # identical duplicate
    table, membership, dig1 = ACQ.canonical_event_table(
        {"r1": a, "r2": b})
    assert [e["id"] for e in table] == ["x1", "x2"], table
    assert membership["x1"] == ["r1", "r2"], membership
    # response/order permutation cannot change the digest
    table2, _, dig2 = ACQ.canonical_event_table({"r2": b, "r1": a})
    assert dig1 == dig2, "region ordering changed the table digest"


class _FakeGit:
    """Deterministic git stand-in for the authority-chain KAT: clean
    status, matching head/tree, real module blob bytes, then FORGED
    pass/go records -- proving the chain checks fire on forged records
    even when every self-hash is correct."""

    def __init__(self):
        self.module_blob = open(
            os.path.join(_HERE, "w2_mf4_catalog_acquire_grassmann.py"),
            "rb").read().replace(b"\r\n", b"\n")

    def __call__(self, repo, *args):
        a = " ".join(args)
        if a == "status --porcelain":
            return b""
        if a == "rev-parse HEAD":
            return b"deadbeef" * 5 + b"\n"
        if a == "rev-parse HEAD^{tree}":
            return b"treetree" * 5 + b"\n"
        if a == "rev-parse origin/master":
            return b"deadbeef" * 5 + b"\n"
        if a.startswith("rev-parse") and a.endswith("^{tree}"):
            return b"treetree" * 5 + b"\n"
        if a.startswith("show") and ":monitoring/src/" in a:
            return self.module_blob
        if a.startswith("show HEAD:docs/"):
            # correction doc pinning the true identities
            import w2_mf4_catalog_acquire_grassmann as A
            _, cdig = A.query_contract()
            return (hashlib.sha256(self.module_blob).hexdigest()
                    + " " + cdig).encode()
        if a.startswith("merge-base"):
            return args[1].encode() + b"\n"
        if a.startswith("rev-parse"):
            return args[1].encode() + b"\n"
        if "forged-pass" in a:
            return self.pass_record
        if a.startswith("show "):
            return self.go_record
        raise AssertionError(f"unexpected git call: {a}")

    pass_record = b"forged text with no bindings"
    go_record = b"forged text with no bindings"


def kat_d1b_forged_authority_chain():
    import tempfile
    fake = _FakeGit()
    saved = ACQ._git
    try:
        ACQ._git = fake
        _, cdig = ACQ.query_contract()
        auth = {"schema": ACQ.AUTH_SCHEMA,
                "public_head_commit": "deadbeef" * 5,
                "public_head_tree": "treetree" * 5,
                "module_git_blob_sha256":
                    hashlib.sha256(fake.module_blob).hexdigest(),
                "query_contract_sha256": cdig,
                "codex_pass": {"framework_commit": "feedface" * 5,
                               "file": "inbox/forged-pass.md",
                               "patch_sha256": "11" * 32,
                               "base_commit": "22" * 20,
                               "result_tree": "33" * 20},
                "owner_fire_go": {"quote": "forged go",
                                  "utc": "2026-08-29T00:00:00Z",
                                  "scope": ACQ.SCOPE_LITERAL,
                                  "pass_framework_commit":
                                      "feedface" * 5,
                                  "source_framework_commit":
                                      "feedface" * 5,
                                  "source_file": "inbox/forged-go.md"},
                "output_target_must_be_absent": True}
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps(auth).encode())
        os.close(fd)
        try:
            def bomb(url, timeout=None):
                raise AssertionError("HTTP attempted on forged chain")
            ACQ.fire(path, opener=bomb)
            raise AssertionError("forged chain accepted")
        except SystemExit as e:
            assert "MF4_FIRE_AUTH_PASS_UNBOUND" in str(e), e
        finally:
            os.remove(path)
    finally:
        ACQ._git = saved


def _finalization_injection(break_name=None, break_rename=False):
    import tempfile, shutil as _sh
    root = tempfile.mkdtemp(prefix="mf4fin_")
    saved_final = ACQ.FINAL_DIR
    saved_verify = ACQ.verify_fire_authorization
    saved_seal = ACQ._seal
    saved_rename = os.rename
    try:
        ACQ.FINAL_DIR = os.path.join(root, "snap")
        ACQ.verify_fire_authorization = lambda p: (
            {"stub": True}, {"file": "stub", "sha256": "0" * 64,
                             "bytes": 0, "go_source_sha256": "0" * 64})

        class FakeResp:
            def __init__(self, raw, url):
                self.raw, self._url, self.status = raw, url, 200
                import email.message
                m = email.message.Message()
                m["Content-Type"] = "application/json"
                self.headers = m
            def geturl(self):
                return self._url
            def read(self):
                return self.raw
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        centers = {r: ((b["bbox"]["min_lat"] + b["bbox"]["max_lat"]) / 2,
                       (b["bbox"]["min_lon"] + b["bbox"]["max_lon"]) / 2)
                   for r, b in
                   {reg: {"bbox": ACQ.PINNED_BBOXES[reg]}
                    for reg in ACQ.ADMITTED}.items()}
        idx = {"n": 0}

        def fake_open(url, timeout=None):
            region = ACQ.ADMITTED[idx["n"]]
            idx["n"] += 1
            lat, lon = centers[region]
            return FakeResp(_geojson([(f"ev_{region}", lat, lon,
                                       4.5, T0)]), url)

        if break_name is not None:
            def broken_seal(staging, name, data):
                if name == break_name:
                    raise OSError(f"injected seal failure: {name}")
                return saved_seal(staging, name, data)
            ACQ._seal = broken_seal
        if break_rename:
            def broken_rename(a, b):
                raise OSError("injected publish failure")
            os.rename = broken_rename
        try:
            ACQ.fire("stubauth", opener=fake_open)
            raise AssertionError("injected failure did not refuse")
        except SystemExit as e:
            assert "MF4_FIRE_FINALIZATION_EXCEPTION" in str(e), e
        finally:
            if break_rename:
                os.rename = saved_rename
        staging = ACQ.FINAL_DIR + ".staging"
        assert os.path.isfile(os.path.join(
            staging, "REFUSAL_MANIFEST.json")), "no refusal manifest"
        assert not os.path.exists(ACQ.FINAL_DIR), "final target created"
    finally:
        os.rename = saved_rename
        ACQ._seal = saved_seal
        ACQ.FINAL_DIR = saved_final
        ACQ.verify_fire_authorization = saved_verify
        _sh.rmtree(root, ignore_errors=True)


def kat_d6_finalization_injections():
    _finalization_injection(break_name="catalog_snapshot_v1.json")
    _finalization_injection(break_name="acquisition_receipt_v1.json")
    _finalization_injection(break_rename=True)


def _auth_file(source_commit=None):
    """Correction 5: the wrapper's owner_fire_go carries ONLY the
    untrusted source pointer; every semantic go field must come from
    the committed record itself."""
    import tempfile
    fake = _FakeGit()
    _, cdig = ACQ.query_contract()
    auth = {"schema": ACQ.AUTH_SCHEMA,
            "public_head_commit": "deadbeef" * 5,
            "public_head_tree": "treetree" * 5,
            "module_git_blob_sha256":
                hashlib.sha256(fake.module_blob).hexdigest(),
            "query_contract_sha256": cdig,
            "codex_pass": {"framework_commit": "feedface" * 5,
                           "file": "inbox/forged-pass.md"},
            "owner_fire_go": {"source_framework_commit":
                                  source_commit or "beefcafe" * 5,
                              "source_file": "inbox/forged-go.md"},
            "output_target_must_be_absent": True}
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps(auth).encode())
    os.close(fd)
    return fake, path


def _valid_pass_record(fake):
    _, cdig = ACQ.query_contract()
    return json.dumps({
        "verdict": "PRE_HTTP_PASS",
        "base_commit": "22" * 20, "bundle_sha256": "11" * 32,
        "result_tree": "treetree" * 5,
        "module_git_blob_sha256":
            hashlib.sha256(fake.module_blob).hexdigest(),
        "query_contract_sha256": cdig}).encode()


def _valid_go_record():
    return {"verdict": "OWNER_FIRE_GO",
            "quote": "go ahead, fire the 13 queries",
            "utc": "2026-08-30T00:00:00Z",
            "scope": ACQ.SCOPE_LITERAL,
            "pass_framework_commit": "feedface" * 5,
            "public_head_commit": "deadbeef" * 5,
            "public_head_tree": "treetree" * 5}


def kat_d1c_negative_hold_verdict():
    """Codex probe: a reachable record whose text contains PASS but
    whose structured verdict is HOLD must refuse pre-opener."""
    fake, path = _auth_file()
    _, cdig = ACQ.query_contract()
    fake.pass_record = json.dumps({
        "verdict": "HOLD",
        "note": "PRE-HTTP PASS is not issued despite the word PASS",
        "base_commit": "22" * 20, "bundle_sha256": "11" * 32,
        "result_tree": "treetree" * 5,
        "module_git_blob_sha256":
            hashlib.sha256(fake.module_blob).hexdigest(),
        "query_contract_sha256": cdig}).encode()
    saved_git, saved_anc = ACQ._git, ACQ._is_ancestor
    try:
        ACQ._git = fake
        ACQ._is_ancestor = lambda repo, a, b: True
        def bomb(url, timeout=None):
            raise AssertionError("HTTP attempted on HOLD verdict")
        try:
            ACQ.fire(path, opener=bomb)
            raise AssertionError("HOLD verdict accepted")
        except SystemExit as e:
            assert "MF4_FIRE_AUTH_PASS_VERDICT" in str(e), e
    finally:
        ACQ._git, ACQ._is_ancestor = saved_git, saved_anc
        os.remove(path)


def kat_d1d_old_go_unbound():
    """Codex probe: a VALID structured pass + an unrelated old TEXT go
    record must refuse pre-opener (correction 5: any non-JSON go
    source is unparseable authority, regardless of its tokens)."""
    fake, path = _auth_file()
    fake.pass_record = _valid_pass_record(fake)
    fake.go_record = b"an old unrelated note containing forged go only"
    saved_git, saved_anc = ACQ._git, ACQ._is_ancestor
    try:
        ACQ._git = fake
        ACQ._is_ancestor = lambda repo, a, b: True
        def bomb(url, timeout=None):
            raise AssertionError("HTTP attempted on unbound go")
        try:
            ACQ.fire(path, opener=bomb)
            raise AssertionError("unbound old go accepted")
        except SystemExit as e:
            assert "MF4_FIRE_AUTH_GO_UNPARSEABLE" in str(e), e
    finally:
        ACQ._git, ACQ._is_ancestor = saved_git, saved_anc
        os.remove(path)


def kat_d1e_hold_go_refused():
    """Codex 2359Z blocker-1 probe: a committed go source carrying
    EVERY required token/field but a HOLD (non-go) verdict must refuse
    pre-opener -- textual and structured variants."""
    fake, path = _auth_file()
    fake.pass_record = _valid_pass_record(fake)
    saved_git, saved_anc = ACQ._git, ACQ._is_ancestor
    try:
        ACQ._git = fake
        ACQ._is_ancestor = lambda repo, a, b: True
        def bomb(url, timeout=None):
            raise AssertionError("HTTP attempted on HOLD go source")
        # (a) textual HOLD containing every token: not strict JSON
        gr = _valid_go_record()
        fake.go_record = ("HOLD: do not fire\n" + " ".join(
            [gr["pass_framework_commit"], gr["public_head_commit"],
             gr["public_head_tree"], gr["scope"], gr["utc"],
             gr["quote"]])).encode()
        try:
            ACQ.fire(path, opener=bomb)
            raise AssertionError("textual HOLD go accepted")
        except SystemExit as e:
            assert "MF4_FIRE_AUTH_GO_UNPARSEABLE" in str(e), e
        # (b) structured HOLD: strict JSON, every required field
        #     present and exact, verdict != OWNER_FIRE_GO
        hold = dict(_valid_go_record(),
                    verdict="HOLD: do not fire",
                    note="contains the words OWNER_FIRE_GO and PASS")
        fake.go_record = json.dumps(hold).encode()
        try:
            ACQ.fire(path, opener=bomb)
            raise AssertionError("structured HOLD go accepted")
        except SystemExit as e:
            assert "MF4_FIRE_AUTH_GO_VERDICT" in str(e), e
    finally:
        ACQ._git, ACQ._is_ancestor = saved_git, saved_anc
        os.remove(path)


def kat_d1f_same_commit_go_refused():
    """Codex 2359Z blocker-1 probe: a nominally VALID go record
    committed AT the pass commit must refuse (the owner go must be a
    strictly later record); the same record at a later commit
    verifies."""
    saved_git, saved_anc = ACQ._git, ACQ._is_ancestor
    # positive control first: valid pass + valid later go verifies
    fake, path = _auth_file()
    fake.pass_record = _valid_pass_record(fake)
    fake.go_record = json.dumps(_valid_go_record()).encode()
    try:
        ACQ._git = fake
        ACQ._is_ancestor = lambda repo, a, b: True
        auth, ident = ACQ.verify_fire_authorization(path)
        assert ident["go_source_sha256"] == _sha(fake.go_record)
    finally:
        ACQ._git, ACQ._is_ancestor = saved_git, saved_anc
        os.remove(path)
    # same record, but the go source commit IS the pass commit
    fake2, path2 = _auth_file(source_commit="feedface" * 5)
    fake2.pass_record = _valid_pass_record(fake2)
    fake2.go_record = json.dumps(_valid_go_record()).encode()
    try:
        ACQ._git = fake2
        ACQ._is_ancestor = lambda repo, a, b: True
        def bomb(url, timeout=None):
            raise AssertionError("HTTP attempted on same-commit go")
        try:
            ACQ.fire(path2, opener=bomb)
            raise AssertionError("same-commit go accepted")
        except SystemExit as e:
            assert "MF4_FIRE_AUTH_GO_SAME_COMMIT" in str(e), e
    finally:
        ACQ._git, ACQ._is_ancestor = saved_git, saved_anc
        os.remove(path2)


def kat_d7_durability():
    """Positive transaction lock: a full 13-region fake acquisition
    publishes exactly 28 files, every one fsynced; then a raw-seal
    fsync injection refuses typed with no final target."""
    import tempfile, shutil as _sh
    root = tempfile.mkdtemp(prefix="mf4dur_")
    saved_final = ACQ.FINAL_DIR
    saved_verify = ACQ.verify_fire_authorization
    saved_fsync = os.fsync
    counted = {"n": 0}

    def counting_fsync(fd):
        counted["n"] += 1
        return saved_fsync(fd)

    class FakeResp:
        def __init__(self, raw, url):
            self.raw, self._url, self.status = raw, url, 200
            import email.message
            m = email.message.Message()
            m["Content-Type"] = "application/json"
            self.headers = m
        def geturl(self):
            return self._url
        def read(self):
            return self.raw
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    idx = {"n": 0}

    def fake_open(url, timeout=None):
        region = ACQ.ADMITTED[idx["n"]]
        idx["n"] += 1
        b = ACQ.PINNED_BBOXES[region]
        lat = (b["min_lat"] + b["max_lat"]) / 2
        lon = (b["min_lon"] + b["max_lon"]) / 2
        return FakeResp(_geojson([(f"ev_{region}", lat, lon, 4.5, T0)]),
                        url)

    try:
        ACQ.FINAL_DIR = os.path.join(root, "snap")
        ACQ.verify_fire_authorization = lambda p: (
            {"stub": True}, {"file": "stub", "sha256": "0" * 64,
                             "bytes": 0, "go_source_sha256": "0" * 64})
        os.fsync = counting_fsync
        ACQ.fire("stubauth", opener=fake_open)
        files = sorted(os.listdir(ACQ.FINAL_DIR))
        assert len(files) == 28, f"{len(files)} files: {files}"
        assert counted["n"] >= 28, f"only {counted['n']} fsync calls"
        # injection: fsync fails on the FIRST raw seal
        _sh.rmtree(ACQ.FINAL_DIR)
        idx["n"] = 0

        def failing_fsync(fd):
            raise OSError("injected raw fsync failure")

        os.fsync = failing_fsync
        try:
            ACQ.fire("stubauth", opener=fake_open)
            raise AssertionError("raw fsync failure did not refuse")
        except SystemExit as e:
            assert "MF4_CATALOG" in str(e) or "MF4_FIRE" in str(e), e
        assert not os.path.exists(ACQ.FINAL_DIR), "final target created"
    finally:
        os.fsync = saved_fsync
        ACQ.FINAL_DIR = saved_final
        ACQ.verify_fire_authorization = saved_verify
        _sh.rmtree(root, ignore_errors=True)


def _adp():
    import w2_mf4_catalog_adapter_grassmann as ADP
    return ADP


class _patched_chain:
    """Context manager: repoint the adapter/acquire git helpers at a
    _FakeGit chain whose pass/go records are VALID, so receipt-bound
    fixtures exercise the post-anchor seams."""

    def __enter__(self):
        self.fake = _FakeGit()
        self.fake.pass_record = _valid_pass_record(self.fake)
        self.fake.go_record = json.dumps(_valid_go_record()).encode()
        self.saved = (ACQ._git, ACQ._is_ancestor)
        ACQ._git = self.fake
        ACQ._is_ancestor = lambda repo, a, b: True
        return self.fake

    def __exit__(self, *exc):
        ACQ._git, ACQ._is_ancestor = self.saved
        return False


def _snap_receipt_pair(table, policy=None, mutate_snap=None,
                       mutate_rec=None, mutate_auth=None):
    """Fixture pair in the fire's EXACT output format (full closed
    keysets): snapshot bytes + acquisition-receipt bytes, mutually
    bound (digests, contract, authorization identity + content)."""
    ADP = _adp()
    _, contract_sha = ACQ.query_contract()
    module_blob = open(
        os.path.join(_HERE, "w2_mf4_catalog_acquire_grassmann.py"),
        "rb").read().replace(b"\r\n", b"\n")
    auth_ident = {"file": "fire_authorization.json", "sha256": "a1" * 32,
                  "bytes": 512, "go_source_sha256": "b2" * 32}
    auth_content = {"schema": ACQ.AUTH_SCHEMA,
                    "public_head_commit": "deadbeef" * 5,
                    "public_head_tree": "treetree" * 5,
                    "module_git_blob_sha256": _sha(module_blob),
                    "query_contract_sha256": contract_sha,
                    "codex_pass": {"framework_commit": "feedface" * 5,
                                   "file": "inbox/forged-pass.md"},
                    "owner_fire_go": {"source_framework_commit":
                                          "beefcafe" * 5,
                                      "source_file":
                                          "inbox/forged-go.md"},
                    "output_target_must_be_absent": True}
    if mutate_auth:
        mutate_auth(auth_content)
    aci = {"path": ACQ.MODULE_REL,
           "git_blob_sha256": auth_content["module_git_blob_sha256"],
           "git_blob_bytes": len(module_blob),
           "runtime_sha256": auth_content["module_git_blob_sha256"],
           "runtime_bytes": len(module_blob)}
    fsi = {"path": "monitoring/src/fault_segments.py",
           "sha256": "f0" * 32, "bytes": 34410}
    snap = {"schema": ADP.SNAPSHOT_SCHEMA,
            "temporal_role": "CALIBRATION_LATE_REPAIR",
            "temporal_role_policy": policy if policy is not None
                else ACQ.TEMPORAL_ROLE_POLICY,
            "amendment": ("docs/f2g_window2_execution/"
                          "amendment_mf4_late_catalog_repair_"
                          "20260829.md"),
            "lane_status": "AMENDED_AFTER_FREEZE",
            "query_contract": {"provider": ACQ.PROVIDER_URL},
            "canonical_event_table": table,
            "canonical_event_table_sha256":
                _sha(json.dumps(table, sort_keys=True).encode("utf-8")),
            "region_membership": {e["id"]: ["regA"] for e in table
                                  if isinstance(e, dict)
                                  and isinstance(e.get("id"), str)},
            "events_by_region_counts": {"regA": len(table)},
            "query_contract_sha256": contract_sha,
            "authorization": auth_ident,
            "authorization_content": auth_content,
            "acquisition_code_identity": aci,
            "fault_segments_identity": fsi}
    if mutate_snap:
        mutate_snap(snap)
    snap_bytes = (json.dumps(snap, indent=1, sort_keys=True)
                  + "\n").encode("utf-8")
    rec = {"schema": ADP.RECEIPT_SCHEMA,
           "fired_utc": "2026-08-30T00:00:00.000000Z",
           "attempts": {"regA": {"parse_result": "OK"}},
           "snapshot_file": "catalog_snapshot_v1.json",
           "snapshot_sha256": _sha(snap_bytes),
           "canonical_event_table_sha256":
               snap.get("canonical_event_table_sha256"),
           "query_contract_sha256": contract_sha,
           "authorization": auth_ident,
           "authorization_content": snap.get("authorization_content"),
           "acquisition_code_identity": aci,
           "fault_segments_identity": fsi}
    if mutate_rec:
        mutate_rec(rec)
    rec_bytes = (json.dumps(rec, indent=1, sort_keys=True)
                 + "\n").encode("utf-8")
    return snap_bytes, rec_bytes


def _d8_fixture():
    import datetime as _dt
    bbox = {"min_lat": 30.0, "max_lat": 40.0,
            "min_lon": 100.0, "max_lon": 110.0}
    days = [(_dt.date(2025, 10, 18) + _dt.timedelta(n)).isoformat()
            for n in range(24)]
    risk = {"regA": {d: 0.3 + 0.01 * i for i, d in enumerate(days)}}
    ev_ms = int(_dt.datetime(2025, 10, 28, 12,
                             tzinfo=_dt.timezone.utc).timestamp() * 1000)
    table = [{"id": "evA", "time_ms": ev_ms,
              "time_utc": "2025-10-28T12:00:00.000Z",
              "lat": 35.0, "lon": 105.0, "mag": 4.6}]
    return risk, bbox, table, ev_ms


def kat_d8_real_adapter_path():
    """Codex 2359Z blocker 2: the REAL consumer path now consumes only
    receipt-bound bytes -- the frozen w2_mf4.calibrate runs through
    the verified loader; the amended training digest binds policy +
    table + snapshot identity; every live view refuses."""
    ADP = _adp()
    risk, bbox, table, ev_ms = _d8_fixture()
    snap_bytes, rec_bytes = _snap_receipt_pair(table)
    with _patched_chain():
        ledger = ADP.calibrate_with_snapshot(
            risk, snap_bytes, rec_bytes, {"regA": bbox}, ["regA"],
            freeze_day="2025-11-17", snapshot_end="2025-11-16")
    assert "amended_training_digest" in ledger
    b = ledger["amended_training_binding"]
    assert b["engine_training_digest"] == ledger["training_digest"]
    assert b["snapshot_sha256"] == _sha(snap_bytes)
    assert b["authorization_sha256"] == "a1" * 32
    # a table change (valid bound pair) moves the amended digest
    table2 = table + [{"id": "evB", "time_ms": ev_ms + 60000,
                       "time_utc": "2025-10-28T12:01:00.000Z",
                       "lat": 35.0, "lon": 105.0, "mag": 3.9}]
    s2, r2 = _snap_receipt_pair(table2)
    with _patched_chain():
        ledger2 = ADP.calibrate_with_snapshot(
            risk, s2, r2, {"regA": bbox}, ["regA"],
            freeze_day="2025-11-17", snapshot_end="2025-11-16")
    assert (ledger2["amended_training_digest"]
            != ledger["amended_training_digest"])
    # live path: the calibration snapshot refuses as a role violation
    with _patched_chain():
        snap_obj, _ = ADP.load_verified_snapshot(snap_bytes,
                                                 rec_bytes)
    try:
        ADP.live_prediction_events(snap_obj)
        raise AssertionError("live path accepted late snapshot")
    except SystemExit as e:
        assert "MF4_CATALOG_ROLE_VIOLATION" in str(e), e
    # live path: EVERY issue-time view refuses (no registered
    # verifier exists) -- a truthy receipt string is not a receipt
    try:
        ADP.live_prediction_events(
            {"temporal_role": "ISSUE_TIME_VIEW",
             "issue_time_receipt": "not-a-receipt",
             "events": [{"forged": True}]})
        raise AssertionError("forged live view accepted")
    except SystemExit as e:
        assert "MF4_CATALOG_LIVE_UNVERIFIED" in str(e), e
    try:
        ADP.live_prediction_events({"temporal_role": "ISSUE_TIME_VIEW",
                                    "events": []})
        raise AssertionError("receipt-less issue view accepted")
    except SystemExit as e:
        assert "MF4_CATALOG_LIVE_UNVERIFIED" in str(e), e
    try:
        ADP.live_prediction_events({"temporal_role": None})
        raise AssertionError("unbound role accepted")
    except SystemExit as e:
        assert "MF4_CATALOG_ROLE_UNBOUND" in str(e), e


def _d9_series():
    """Codex 2359Z blocker-2 lock battery: unverified calibration
    material must never reach the frozen engine."""
    ADP = _adp()
    risk, bbox, table, _ = _d8_fixture()

    def cal(sb, rb):
        with _patched_chain():
            return ADP.calibrate_with_snapshot(
                risk, sb, rb, {"regA": bbox}, ["regA"],
                freeze_day="2025-11-17", snapshot_end="2025-11-16")

    snap_bytes, rec_bytes = _snap_receipt_pair(table)

    check("D9a fake receipt bytes",
          lambda: cal(snap_bytes, b"not-a-receipt"),
          "MF4_CATALOG_RECEIPT_UNPARSEABLE")
    _, rb = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(schema="forged-schema"))
    check("D9b wrong receipt schema", lambda: cal(snap_bytes, rb),
          "MF4_CATALOG_RECEIPT_SCHEMA")
    tampered = snap_bytes.replace(b'"mag": 4.6', b'"mag": 9.9')
    assert tampered != snap_bytes
    check("D9c snapshot bytes tampered post-receipt",
          lambda: cal(tampered, rec_bytes),
          "MF4_CATALOG_RECEIPT_BINDING")

    def _absent_digest(s):
        del s["canonical_event_table_sha256"]
    sb4, rb4 = _snap_receipt_pair(table, mutate_snap=_absent_digest,
                                  mutate_rec=lambda r: r.update(
                                      canonical_event_table_sha256=None))
    check("D9d absent bound table digest", lambda: cal(sb4, rb4),
          "MF4_CATALOG_TABLE_DIGEST")
    sb5, rb5 = _snap_receipt_pair(
        table, mutate_snap=lambda s: s.update(
            canonical_event_table_sha256="0" * 64))
    check("D9e forged snapshot table digest", lambda: cal(sb5, rb5),
          "MF4_CATALOG_TABLE_DIGEST")
    sb6, rb6 = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(
            canonical_event_table_sha256="0" * 64))
    check("D9f receipt table-digest mismatch", lambda: cal(sb6, rb6),
          "MF4_CATALOG_TABLE_DIGEST")
    sb7, rb7 = _snap_receipt_pair(
        table, policy="a forged permissive policy AMENDED_AFTER_FREEZE")
    check("D9g forged temporal-role policy", lambda: cal(sb7, rb7),
          "MF4_CATALOG_POLICY_UNBOUND")
    sb8, rb8 = _snap_receipt_pair(
        table, mutate_snap=lambda s: s.update(
            query_contract_sha256="0" * 64))
    check("D9h unbound query contract", lambda: cal(sb8, rb8),
          "MF4_CATALOG_CONTRACT_UNBOUND")
    sb9, rb9 = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(
            authorization={"file": "other.json", "sha256": "c3" * 32,
                           "bytes": 1, "go_source_sha256": "d4" * 32}))
    check("D9i authorization identity divergence",
          lambda: cal(sb9, rb9), "MF4_CATALOG_AUTH_UNBOUND")
    check("D9j bare-dict consumption refused",
          lambda: ADP.calibrate_with_snapshot(
              risk, {"temporal_role": "CALIBRATION_LATE_REPAIR",
                     "canonical_event_table": table},
              {"schema": ADP.RECEIPT_SCHEMA},
              {"regA": bbox}, ["regA"],
              freeze_day="2025-11-17", snapshot_end="2025-11-16"),
          "MF4_CATALOG_RECEIPT_UNBOUND")
    sb10, rb10 = _snap_receipt_pair(
        table, mutate_snap=lambda s: s.update(
            temporal_role="ISSUE_TIME_VIEW"))
    check("D9k role forgery through the loader",
          lambda: cal(sb10, rb10), "MF4_CATALOG_ROLE_UNBOUND")


def _d10_series():
    """Codex 0024Z Gate-2 quarantine repair lock battery: a mutually
    self-issued snapshot/receipt pair must refuse; the committed
    chain is the only trust anchor; keysets/paths/UTC/row schema are
    closed."""
    ADP = _adp()
    risk, bbox, table, _ = _d8_fixture()

    def load(sb, rb, patched=True):
        if patched:
            with _patched_chain():
                return ADP.load_verified_snapshot(sb, rb)
        return ADP.load_verified_snapshot(sb, rb)

    sb, rb = _snap_receipt_pair(table)
    check("D10a self-issued pair refuses (no committed chain)",
          lambda: load(sb, rb, patched=False),
          "MF4_CATALOG_TRUST_ANCHOR")
    sb2, rb2 = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(extra_claim="x"))
    check("D10b receipt keyset injection", lambda: load(sb2, rb2),
          "MF4_CATALOG_RECEIPT_KEYSET")
    sb3, rb3 = _snap_receipt_pair(
        table, mutate_snap=lambda s: s.update(extra_claim="y"))
    check("D10c snapshot keyset injection", lambda: load(sb3, rb3),
          "MF4_CATALOG_SNAPSHOT_KEYSET")
    sb4, rb4 = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(
            snapshot_file="other.json"))
    check("D10d wrong named snapshot file", lambda: load(sb4, rb4),
          "MF4_CATALOG_RECEIPT_BINDING")
    sb5, rb5 = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(fired_utc="yesterday"))
    check("D10e non-strict fired UTC", lambda: load(sb5, rb5),
          "MF4_CATALOG_RECEIPT_BINDING")
    sb6, rb6 = _snap_receipt_pair([dict(table[0], injected=1)])
    check("D10f row keyset injection", lambda: load(sb6, rb6),
          "MF4_CATALOG_ROW_SCHEMA")
    sb7, rb7 = _snap_receipt_pair([dict(table[0], mag=True)])
    check("D10g bool row magnitude", lambda: load(sb7, rb7),
          "MF4_CATALOG_ROW_SCHEMA")
    sb8, rb8 = _snap_receipt_pair(
        table, mutate_auth=lambda a: a.update(
            module_git_blob_sha256="0" * 64))
    check("D10h forged authorization module pin",
          lambda: load(sb8, rb8), "MF4_CATALOG_TRUST_ANCHOR")
    sb9, rb9 = _snap_receipt_pair(
        table, mutate_rec=lambda r: r.update(
            authorization_content=dict(r["authorization_content"],
                                       public_head_commit="ab" * 20)))
    check("D10i authorization_content divergence",
          lambda: load(sb9, rb9), "MF4_CATALOG_TRUST_ANCHOR")

    def real_positive():
        base = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                            "docs", "f2g_window2_execution",
                            "mf4_catalog_snapshot")
        sreal = open(os.path.join(base, "catalog_snapshot_v1.json"),
                     "rb").read()
        rreal = open(os.path.join(base,
                                  "acquisition_receipt_v1.json"),
                     "rb").read()
        snap, ident = ADP.load_verified_snapshot(sreal, rreal)
        a = ident["trust_anchor"]
        assert a["pass_framework_commit"].startswith("7601d385"), a
        assert a["go_framework_commit"].startswith("561cfdf4"), a
        assert a["public_head_commit"].startswith("f636c234"), a
        assert len(snap["canonical_event_table"]) == 200
    check("D10j REAL pair verifies through the committed chain",
          real_positive)


def _e_series_c4():
    cases = [
        ("E9 amendment field forgery",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: c.update(amendment="docs/forged.md")),
        ("E10 lane_status forgery",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: c.update(lane_status="ORIGINAL_PREREGISTRATION")),
        ("E11 catalog-binding forgery",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: c["catalog_binding"].update(status="CLOSED")),
        ("E12 training-digest forgery",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: c.update(training_row_digest="0" * 64)),
        ("E13 store locator forgery",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: c["raw_source_store"].update(
             store_root="s4t://forged")),
        ("E14 inventory provenance-path forgery",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: next(iter(
             c["raw_source_store"]["inventory"].values())).update(
                 host_provenance_path="C:/forged.json")),
        ("E15 top-level key injection",
         "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION",
         lambda c: c.update(extra_claim="skill demonstrated")),
    ]
    for name, code, fn in cases:
        def run(fn=fn):
            with Sandbox() as sb:
                GEN.build()
                _mutate_capsule(sb.root, fn, rebind=True)
                GEN.verify_capsule()
        check(name, run, code)

    def run_receipt_schema():
        with Sandbox() as sb:
            GEN.build()
            rp = os.path.join(sb.root, GEN.RECEIPT_REL)
            rec = json.loads(open(rp, encoding="utf-8").read())
            rec["schema"] = "forged-schema"
            open(rp, "wb").write((json.dumps(rec, indent=1,
                                             sort_keys=True)
                                  + "\n").encode())
            GEN.verify_capsule()
    check("E16 receipt schema forgery", run_receipt_schema,
          "MF4_ARCHIVE_RECEIPT_INVALID")

    def run_receipt_refusals():
        with Sandbox() as sb:
            GEN.build()
            rp = os.path.join(sb.root, GEN.RECEIPT_REL)
            rec = json.loads(open(rp, encoding="utf-8").read())
            rec["refusals"] = ["hidden refusal"]
            open(rp, "wb").write((json.dumps(rec, indent=1,
                                             sort_keys=True)
                                  + "\n").encode())
            GEN.verify_capsule()
    check("E17 receipt hidden-refusal", run_receipt_refusals,
          "MF4_ARCHIVE_RECEIPT_INVALID")

    for name, val in (("E18 risk above range", 1.5),
                      ("E19 risk below range", -0.1)):
        def run(val=val):
            with Sandbox(_risk_mutator(val)) as _:
                GEN.build()
        check(name, run, "MF4_ARCHIVE_RISK_RANGE")

    def run_e20():
        with Sandbox() as sb:
            snapdir = os.path.join(sb.root, "docs",
                                   "f2g_window2_execution",
                                   "mf4_catalog_snapshot")
            os.makedirs(snapdir)
            open(os.path.join(snapdir, "catalog_snapshot_v1.json"),
                 "wb").write(
                b'{"canonical_event_table_sha256": "ab12",'
                b' "temporal_role": "CALIBRATION_LATE_REPAIR"}\n')
            open(os.path.join(snapdir,
                              "acquisition_receipt_v1.json"),
                 "wb").write(b'{"fixture": true}\n')
            GEN.build()
            cap = json.load(open(os.path.join(sb.root,
                                              GEN.CAPSULE_REL),
                                 encoding="utf-8"))
            assert cap["catalog_binding"]["status"] == "BOUND_V2", \
                cap["catalog_binding"]["status"]
            with open(os.path.join(snapdir,
                                   "catalog_snapshot_v1.json"),
                      "ab") as f:
                f.write(b"\n")
            GEN.verify_capsule()
    check("E20 v2 catalog-binding snapshot tamper", run_e20,
          "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION")


def kat_d5_role_guard():
    ACQ.calibration_snapshot_role_guard(
        {"temporal_role": "CALIBRATION_LATE_REPAIR"},
        "calibration_labels")
    ACQ.calibration_snapshot_role_guard(
        {"temporal_role": "CALIBRATION_LATE_REPAIR"},
        "calibration_features")
    try:
        ACQ.calibration_snapshot_role_guard(
            {"temporal_role": "CALIBRATION_LATE_REPAIR"},
            "live_prediction_features")
        raise AssertionError("live use admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_ROLE_VIOLATION" in str(e), e
    try:
        ACQ.calibration_snapshot_role_guard(
            {"temporal_role": None}, "calibration_labels")
        raise AssertionError("unbound role admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_ROLE_UNBOUND" in str(e), e


def _e_series():
    cases = [
        ("E1 file-census mutation", "MF4_ARCHIVE_FILE_CENSUS_DRIFT",
         lambda c: c["file_census"].update(usable_files=999), True),
        ("E2 support-census mutation", "MF4_ARCHIVE_SUPPORT_CENSUS_DRIFT",
         lambda c: c["support_census"]["regA"].update(days_supported=0),
         True),
        ("E3 maturity-bound mutation", "MF4_ARCHIVE_MATURITY_DRIFT",
         lambda c: c["maturity_bounds"].update(freeze_day="2099-01-01"),
         True),
        ("E4 receipt->capsule digest", "MF4_ARCHIVE_CAPSULE_DIGEST",
         lambda c: c.update(schema=c["schema"]), False),
        ("E5 missing-cells mutation", "MF4_ARCHIVE_MISSING_CELLS_DRIFT",
         lambda c: c.update(missing_region_cells=[["2026-01-01",
                                                   "regA"]]), True),
        ("E6 region-partition mutation",
         "MF4_ARCHIVE_REGION_PARTITION_DRIFT",
         lambda c: c["region_sets"].update(
             admitted_regions=["regA"]), True),
        ("E7 day-grid mutation", "MF4_ARCHIVE_DAY_GRID_DRIFT",
         lambda c: c["day_index"].update(days_total=999), True),
        ("E8 producer-identity mutation", "MF4_ARCHIVE_PRODUCER_IDENTITY",
         lambda c: c["producer_code_identity"].update(sha256="0" * 64),
         True),
    ]
    for name, code, fn, rebind in cases:
        def run(fn=fn, rebind=rebind, name=name, code=code):
            with Sandbox() as sb:
                GEN.build()
                if name.startswith("E4"):
                    cp = os.path.join(sb.root, GEN.CAPSULE_REL)
                    raw = open(cp, "rb").read()
                    open(cp, "wb").write(raw + b"\n")
                else:
                    _mutate_capsule(sb.root, fn, rebind=rebind)
                GEN.verify_capsule()
        check(name, run, code)


def main():
    check("A1 missing raw object w/ present digest",
          kat_a1_missing_object, "MF4_ARCHIVE_OBJECT_MISSING")
    check("A2 row mutation", kat_a2_row_mutation,
          "MF4_ARCHIVE_ROW_DIVERGENCE")
    with Sandbox(_risk_mutator("0.5")) as _:
        check("A3a string risk", GEN.build, "MF4_ARCHIVE_RISK_MALFORMED")
    with Sandbox(_risk_mutator(True)) as _:
        check("A3b bool risk", GEN.build, "MF4_ARCHIVE_RISK_MALFORMED")
    with Sandbox(_risk_mutator(float("nan"))) as _:
        check("A3c NaN risk", GEN.build, "MF4_ARCHIVE_RISK_MALFORMED")
    check("A4 duplicate region-day row", kat_a4_duplicate_region_day,
          "MF4_ARCHIVE_ROW_DIVERGENCE")
    check("A5 rows-digest/census drift", kat_a5_census_drift,
          "MF4_ARCHIVE_ROWS_DIGEST")
    check("A6 golden replay of committed capsule",
          kat_a6_golden_real_capsule)
    check("A7 store portability (two physical aliases)",
          kat_a7_store_portability)
    check("A8 attempt-1 refusal capsule integrity",
          kat_a8_attempt1_capsule_integrity)
    check("B1 alias-direction reversal", kat_b1_alias_reversal,
          "MF4_BBOX")
    check("B2 tokyo->tohoku mapping", kat_b2_tokyo_tohoku, "MF4_BBOX")
    check("B3 dashboard-bbox expansion", kat_b3_dashboard_bbox_expansion,
          "MF4_BBOX_PIN_MISMATCH")
    check("B4 boundary event in/out", kat_b4_boundary_event)
    check("B5 bbox coordinate mutation", kat_b5_bbox_coordinate_mutation,
          "MF4_BBOX_PIN_MISMATCH")
    check("C1 top-five lossy view refused", kat_c1_lossy_view,
          "MF4_CATALOG_LOSSY_VIEW")
    check("C2 query limit", kat_c2_query_limit,
          "MF4_CATALOG_QUERY_LIMIT")
    check("C3 malformed/null fields", kat_c3_malformed_fields)
    check("C4 temporal filter", kat_c4_temporal_filter,
          "MF4_CATALOG_TEMPORAL_FILTER")
    check("C5 duplicate + inconsistent ids",
          kat_c5_duplicate_and_inconsistent_ids)
    _c6_series()
    check("D1 fire refuses without authorization (pre-HTTP)",
          kat_d1_fire_requires_authorization, "MF4_FIRE_AUTH_MISSING")
    check("D2 partial-failure staging + reuse refusal",
          kat_d2_partial_failure_staging)
    check("D3 closed-parser mutations", kat_d3_parser_closures)
    check("D4 canonical table dedup + order invariance",
          kat_d4_canonical_table)
    check("D1b forged authority chain (self-hashes correct)",
          kat_d1b_forged_authority_chain)
    check("D5 temporal role guard", kat_d5_role_guard)
    check("D6 finalization injections x3",
          kat_d6_finalization_injections)
    check("D1c negative-HOLD verdict refused",
          kat_d1c_negative_hold_verdict)
    check("D1d old unbound go refused", kat_d1d_old_go_unbound)
    check("D1e HOLD go source with every token refused",
          kat_d1e_hold_go_refused)
    check("D1f same-commit go refused (valid go verifies)",
          kat_d1f_same_commit_go_refused)
    check("D7 28-file durability + raw-fsync injection",
          kat_d7_durability)
    check("D8 real adapter path (receipt-bound, frozen calibrate)",
          kat_d8_real_adapter_path)
    _d9_series()
    _d10_series()
    _e_series()
    _e_series_c4()
    print(f"\n{len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
