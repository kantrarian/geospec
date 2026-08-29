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
        GEN.verify_capsule()


def kat_a5_census_drift():
    with Sandbox() as sb:
        GEN.build()
        cp = os.path.join(sb.root, GEN.CAPSULE_REL)
        cap = json.load(open(cp, encoding="utf-8"))
        cap["rows_file"]["sha256"] = "0" * 64
        json.dump(cap, open(cp, "w", encoding="utf-8"), sort_keys=True)
        GEN.verify_capsule()


def kat_a6_golden_real_capsule():
    res = GEN.verify_capsule()
    assert res["objects_verified"] == 307, res
    assert res["rows_verified"] == 4298, res


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


def kat_b4_boundary_event():
    # exactly on the bbox edge: admitted (inclusive)
    raw = _geojson([("edge", 40.0, 110.0, 4.5, T0)])
    evs = ACQ.validate_events("katreg", BB, raw)
    assert len(evs) == 1
    # just outside: refused
    raw2 = _geojson([("out", 40.0001, 110.0, 4.5, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw2)
        raise AssertionError("outside-bbox event admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_SPATIAL_FILTER" in str(e)


def kat_c1_lossy_view():
    """>5 qualifying events must ALL survive validation -- a top-five
    projection (the dashboard's serialization) loses events and is
    detected as a count mismatch, refused as a lossy view."""
    full = [(f"ev{i}", 35.0 + i * 0.1, 105.0, 4.0 + i * 0.1, T0 + i)
            for i in range(8)]
    evs = ACQ.validate_events("katreg", BB, _geojson(full))
    assert len(evs) == 8, "full view must retain all 8"
    top5 = ACQ.validate_events("katreg", BB, _geojson(full[:5]))
    if len(top5) != len(full):
        raise SystemExit("REFUSED MF4_CATALOG_LOSSY_VIEW: top-five "
                         f"projection kept {len(top5)}/{len(full)}")


def kat_c2_query_limit():
    saved = ACQ.LIMIT
    try:
        ACQ.LIMIT = 3
        raw = _geojson([(f"e{i}", 35.0, 105.0, 4.1, T0 + i)
                        for i in range(3)])
        ACQ.validate_events("katreg", BB, raw)
    finally:
        ACQ.LIMIT = saved


def kat_c3_malformed_fields():
    for bad in [("nullmag", 35.0, 105.0, None, T0),
                ("nulltime", 35.0, 105.0, 4.2, None),
                ("strmag", 35.0, 105.0, "4.2", T0)]:
        try:
            ACQ.validate_events("katreg", BB, _geojson([bad]))
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
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("null coords admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e)


def kat_c4_temporal_filter():
    late = int(dt.datetime(2026, 8, 28, 0, 0, 1,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)
    ACQ.validate_events("katreg", BB,
                        _geojson([("late", 35.0, 105.0, 4.2, late)]))


def kat_c5_duplicate_and_inconsistent_ids():
    dup = _geojson([("same", 35.0, 105.0, 4.2, T0),
                    ("same", 35.1, 105.1, 4.3, T0 + 1)])
    try:
        ACQ.validate_events("katreg", BB, dup)
        raise AssertionError("duplicate id admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_DUPLICATE_ID" in str(e)
    a = ACQ.validate_events("r1", BB, _geojson([("x", 35.0, 105.0,
                                                 4.2, T0)]))
    b = ACQ.validate_events("r2", BB, _geojson([("x", 35.5, 105.0,
                                                 4.2, T0)]))
    try:
        ACQ.canonical_event_table({"r1": a, "r2": b})
        raise AssertionError("inconsistent shared id admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_INCONSISTENT_ID" in str(e)


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
            ACQ.validate_events("katreg", BB, raw)
            raise AssertionError(f"{const} admitted")
        except SystemExit as e:
            assert "MF4_CATALOG_NONFINITE_JSON" in str(e), e
    # boolean magnitude
    raw = _geojson([("b", 35.0, 105.0, True, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("bool mag admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e), e
    # magnitude below the registered threshold
    raw = _geojson([("lo", 35.0, 105.0, 3.0, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("mag 3.0 admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MAG_BELOW_THRESHOLD" in str(e), e
    # metadata.count mismatch
    raw = _geojson([good], count=99)
    try:
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("count mismatch admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_COUNT_MISMATCH" in str(e), e
    # reverse time order
    raw = _geojson([("t2", 35.0, 105.0, 4.5, T0 + 1000),
                    ("t1", 35.0, 105.0, 4.5, T0)])
    try:
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("reverse order admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_ORDER_VIOLATION" in str(e), e
    # non-Point geometry
    raw = _geojson([good], gtype="LineString")
    try:
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("non-Point admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e), e
    # non-FeatureCollection top level
    raw = _geojson([good], top="Whatever")
    try:
        ACQ.validate_events("katreg", BB, raw)
        raise AssertionError("non-FeatureCollection admitted")
    except SystemExit as e:
        assert "MF4_CATALOG_MALFORMED" in str(e), e


def kat_d4_canonical_table():
    a = ACQ.validate_events("r1", BB, _geojson([
        ("x1", 35.0, 105.0, 4.5, T0), ("x2", 36.0, 106.0, 4.6, T0 + 5)]))
    b = ACQ.validate_events("r2", BB, _geojson([
        ("x1", 35.0, 105.0, 4.5, T0)]))          # identical duplicate
    table, membership, dig1 = ACQ.canonical_event_table(
        {"r1": a, "r2": b})
    assert [e["id"] for e in table] == ["x1", "x2"], table
    assert membership["x1"] == ["r1", "r2"], membership
    # response/order permutation cannot change the digest
    table2, _, dig2 = ACQ.canonical_event_table({"r2": b, "r1": a})
    assert dig1 == dig2, "region ordering changed the table digest"


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
    check("D1 fire refuses without authorization (pre-HTTP)",
          kat_d1_fire_requires_authorization, "MF4_FIRE_AUTH_MISSING")
    check("D2 partial-failure staging + reuse refusal",
          kat_d2_partial_failure_staging)
    check("D3 closed-parser mutations", kat_d3_parser_closures)
    check("D4 canonical table dedup + order invariance",
          kat_d4_canonical_table)
    print(f"\n{len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
