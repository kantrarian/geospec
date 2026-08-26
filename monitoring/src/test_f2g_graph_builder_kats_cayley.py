"""fault2graph Phase A GRAPH BUILDER KATs (cayley) -- contract `2e0c7a33` A3/A5.

AUTHOR SELF-TESTS, not clearance: codex reviews the builder against V-A and the
contract; these KATs are the executable surface that review drives.

REV 2 (codex builder review `17a550be` -- all five repairs locked):
  B0   REAL-SHAPE (F1): the EXACT committed phase0 bundle -> the authoritative
       d2_campaign_v2_plan builder -> the EXACT committed digest-pinned pool
       -> 110 pool nodes / 35 selected; nesting-flatten, pool-byte, NSLC,
       segment, and carrier mutations each REFUSE before output
  B4b  (F5) two carriers never share a geometric edge; carrier_key on every edge
  B4c  (F5) EPSG:2263 (US survey foot) distances converted to TRUE metres and
       checked against an independent pyproj.Geod ellipsoid recomputation
  B8d  (F4) index-frame counterexample: same pair+algorithm, A-B day0 vs
       A-B-C day1 -> typed NOT_COMPARABLE/STATION_INDEX_MISMATCH
  B9   (F3) ingest verifier flag sequence MUST be [recompute=True, False];
       B9-live drives the REAL producer: produce -> ingest PASSES, then an
       in-bounds self-consistently rehashed doctor -> ingest REFUSES PRE
       (true derivation isolation -- the undoctored artifact passes recompute)
  B10  (F2) heterogeneous MultiGraph round trip: a pair carrying BOTH near and
       coheres_with, plus member_of/adjacent_to/contains, re-imports to
       canonical byte-equality for EVERY node and edge table
  B12  (F2) PyG routes through the pinned city2graph gdf_to_pyg bridge
       (torch-gated: explicit typed line until torch lands for the packet)

REV 3 (codex verify-once `4b365e4e` closures 1 + 3, disclosures ratified):
  B0c  A1 BYTE AUTHORITY: the public build_station_table accepts exact
       plan_bytes + pool_bytes ONLY, hard-pinned to the frozen full-64
       constants with NO public digest override -- row deletion/addition/
       reordering/re-encoding all refuse via the byte pin; exact bytes still
       yield 110/35. Fixture rows go through the clearly PRIVATE helper.
  B12  strengthened to FULL heterogeneous identity: every returned node and
       edge table (station/segment/carrier + coherence/near/member/adjacent/
       contains, with r, support, provenance, distance, unit factor, both CRS
       fields) compares canonically after gdf_to_pyg -> pyg_to_gdf; same-pair
       near+coherence must both survive (torch-gated until the packet env).
  B13/B14  KO.KHMN record requirement: render + phase_a_result declare
       geometry_excluded_station_ids/count/reason -- a map can never silently
       imply every selected station was rendered.

Capability-gated classes print an explicit [CAP] line when a dependency is
absent -- never a silent skip; the full-green run happens under the pinned
f2g-env (py3.12 + city2graph@3892a086 lock).
"""

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import d2_f2g_graph_builder as B  # noqa: E402

FAILS = []
CAPS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"    [{tag}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def cap(name, note):
    print(f"    [CAP ] {name} - {note}")
    CAPS.append(name)


def refuses(fn, code):
    try:
        fn()
        return False, "no refusal"
    except B.F2GRefusal as e:
        return e.reason_code == code, e.reason_code


# ---- fixtures: EXACT v2 schema (F1) --------------------------------------------------
PLAN = {"carriers": ["c_one"], "station_registry": {"c_one": [
    {"segment_name": "seg_a", "station_id": "KO.A01",
     "ordered_nslc_candidates": ["KO.A01..HHZ"]},
    {"segment_name": "seg_a", "station_id": "KO.A02",
     "ordered_nslc_candidates": ["KO.A02..HHZ"]},
    {"segment_name": "seg_b", "station_id": "KO.B01",
     "ordered_nslc_candidates": ["KO.B01..HHZ"]},
]}}


def _pool_row(sid, seg, lon, lat):
    return {"station_id": sid, "network": sid.split(".")[0], "lat": lat,
            "lon": lon, "ordered_nslc": [sid + "..HHZ"], "segment": seg}


POOL_DOC = {"carriers": {"c_one": {"segments": {
    "seg_a": [_pool_row("KO.A01", "seg_a", 29.0, 40.7),
              _pool_row("KO.A02", "seg_a", 29.2, 40.8)],
    "seg_b": [_pool_row("KO.B01", "seg_b", 29.5, 40.9),
              _pool_row("KO.SPARE", "seg_b", 29.9, 41.0)],
}, "segment_polygons": {}}}}
POOL_BYTES = B.canon_bytes(POOL_DOC)
POOL_SHA = B.sha(POOL_BYTES)
TOPO = {"topology_version": "t1",
        "adjacency": [["seg_a", "seg_b", {"along_strike_order": 1,
                                          "shared_fault": True}]],
        "segment_geometry": {}}
IDS3 = ["KO.A01", "KO.A02", "KO.B01"]


def _manifest(ids, alg="a" * 64, carrier="c_one", day="2026-03-02"):
    n = len(ids)
    return {"campaign_id": "f" * 64, "carrier_key": carrier, "day": day,
            "algorithm_config_digest": alg, "station_ids": list(ids),
            "station_index_digest": B.sha(B.canon_bytes(list(ids))),
            "n_overlap": [[600 if i != j else 0 for j in range(n)]
                          for i in range(n)]}


def _snap(day, edges, digest, states=None):
    return B.build_snapshot("f" * 64, "c_one", day, edges, states or {}, digest)


def main():
    st = B._fixture_station_table(PLAN, POOL_BYTES, pool_sha256=POOL_SHA)
    sg = B.build_segment_table(PLAN, TOPO)
    ca = B.build_carrier_table(PLAN, TOPO, {"c_one": {
        "expected_sha256": "1" * 64, "valid_through": "2026-08-23",
        "threshold": 0.21}})
    check("B1 identity tables deterministic (real schema + pinned pool join; "
          "fixture rows via the PRIVATE helper)",
          B.canon_jsonl(st) == B.canon_jsonl(
              B._fixture_station_table(PLAN, POOL_BYTES, pool_sha256=POOL_SHA))
          and len(st) == 4 and len(sg) == 2 and len(ca) == 1
          and ca[0]["topology_version"] == "t1")

    # ---- B0 REAL-SHAPE (F1): committed bundle -> authoritative plan -> pool ----
    try:
        import d2_campaign_v2_plan as V2P
        bundle = open(os.path.join(HERE, "campaign_v2_phase05",
                                   "phase0_bundle.json"), "rb").read()
        plan_real, plan_bytes = V2P.build_v2_campaign_plan(bundle)
        pool_real = open(os.path.join(HERE,
                                      "d2_campaign_v2_candidate_pool.json"),
                         "rb").read()
        rst = B.build_station_table(plan_bytes, pool_real)   # PUBLIC authority
        khmn = next(r for r in rst if r["station_id"] == "KO.KHMN")
        check("B0 real-shape via the PUBLIC bytes-authority seam: exact plan "
              "bytes + exact pool bytes -> 110 pool nodes / 35 selected; "
              "KO.KHMN typed coordinate absence (ratified)",
              len(rst) == 110
              and sum(r["registry_selected"] for r in rst) == 35
              and all((r["lon"] and r["lat"]) or not r["coordinates_available"]
                      for r in rst)
              and khmn["registry_selected"] is True
              and khmn["coordinates_available"] is False
              and khmn["lon"] is None)
        # B0b: cross-join refusals via the PRIVATE helper (join-logic locks)
        flat = dict(plan_real)
        flat["station_registry"] = [r for rows in
                                    plan_real["station_registry"].values()
                                    for r in rows]
        _fx = lambda p, pb: B._fixture_station_table(  # noqa: E731
            p, pb, pool_sha256=B.POOL_SHA256)
        ok_a, code_a = refuses(lambda: _fx(flat, pool_real), "PLAN_SCHEMA")
        mut = json.loads(json.dumps(plan_real))
        c0 = list(mut["station_registry"])[0]
        mut["station_registry"][c0][0]["ordered_nslc_candidates"] = ["XX.NO..HHZ"]
        ok_c, code_c = refuses(lambda: _fx(mut, pool_real), "NSLC_MISMATCH_POOL")
        mut2 = json.loads(json.dumps(plan_real))
        mut2["station_registry"][c0][0]["segment_name"] = "not_a_segment"
        ok_d, code_d = refuses(lambda: _fx(mut2, pool_real),
                               "SEGMENT_MISMATCH_POOL")
        mut3 = json.loads(json.dumps(plan_real))
        cs = list(mut3["station_registry"])
        moved = mut3["station_registry"][cs[0]].pop(0)
        mut3["station_registry"][cs[1]].append(moved)
        ok_e, code_e = refuses(lambda: _fx(mut3, pool_real),
                               "CARRIER_MISMATCH_POOL")
        check("B0b cross-join mutations refuse (nesting/NSLC/segment/carrier)",
              ok_a and ok_c and ok_d and ok_e,
              f"{code_a}/{code_c}/{code_d}/{code_e}")
        # B0c (closure 1): BYTE AUTHORITY on the public seam
        import inspect
        sig = str(inspect.signature(B.build_station_table))
        # HARDENED 2026-08-26 (cayley): the "no digest override" half
        # of B0c used to be `"sha256" not in sig and "expected" not in
        # sig` -- a SUBSTRING scan of the signature string. The
        # conclusion it drew was true (the seam really is
        # (plan_bytes, pool_bytes)), but the proof was weak in the
        # exact way that already burned me once: a parameter named
        # `digest`/`plan_hash`/`override`, or a **kwargs, would carry
        # a caller-supplied digest straight past it. Absence of a NAME
        # is not unreachability of a CAPABILITY. So enumerate the
        # accepted inputs instead of scanning for two spellings.
        _params = inspect.signature(B.build_station_table).parameters
        _kinds = [p.kind for p in _params.values()]
        no_override = (
            tuple(_params) == ("plan_bytes", "pool_bytes")
            and inspect.Parameter.VAR_KEYWORD not in _kinds
            and inspect.Parameter.VAR_POSITIONAL not in _kinds
            and all(p.default is inspect.Parameter.empty
                    for p in _params.values()))
        doc = json.loads(plan_bytes.decode("utf-8"))
        rows0 = doc["station_registry"][list(doc["station_registry"])[0]]
        deleted = json.loads(plan_bytes.decode("utf-8"))
        deleted["station_registry"][list(deleted["station_registry"])[0]].pop(0)
        added = json.loads(plan_bytes.decode("utf-8"))
        added["station_registry"][list(added["station_registry"])[0]].append(
            dict(rows0[0], station_id="KO.FAKE"))
        reordered = json.loads(plan_bytes.decode("utf-8"))
        reordered["station_registry"][
            list(reordered["station_registry"])[0]].reverse()
        recoded = plan_bytes.replace(b",", b", ", 1)     # same JSON, new bytes
        mutants = [B.canon_bytes(deleted)[:-1] + b"\n", B.canon_bytes(added),
                   B.canon_bytes(reordered), recoded]
        oks, codes = [], []
        for mb in mutants:
            ok_m, code_m = refuses(lambda mb=mb: B.build_station_table(
                mb, pool_real), "PLAN_BYTES_AUTHORITY")
            oks.append(ok_m); codes.append(code_m)
        ok_p, code_p = refuses(lambda: B.build_station_table(
            plan_bytes, pool_real + b" "), "POOL_BYTES_AUTHORITY")
        check("B0c byte authority: deletion/addition/reordering/re-encoding "
              "and forged-pool bytes ALL refuse; the public seam exposes NO "
              "digest override",
              all(oks) and ok_p and no_override,
              f"{codes}/{code_p} sig={sig} params={tuple(_params)}")
    except FileNotFoundError as e:
        cap("B0 real-shape", f"committed bundle/pool not present: {e}")

    mo = B.build_member_of_edges(st)
    check("B2 member_of ONLY selected registry; spare has none",
          sorted(e["station_id"] for e in mo) == IDS3
          and all(e["station_id"] != "KO.SPARE" for e in mo))

    adj = B.build_adjacent_to_edges(TOPO, sg)
    adj_none = B.build_adjacent_to_edges({"topology_version": "t1"}, sg)
    cont = B.build_contains_edges(sg)
    check("B3 adjacent_to only from pinned topology (carrier resolved via "
          "segment table); contains from segment table",
          len(adj) == 1 and adj[0]["carrier_key"] == "c_one"
          and adj[0]["shared_fault"] is True and adj_none == []
          and len(cont) == 2)

    near = None
    try:
        ok_geo, code_geo = refuses(
            lambda: B.build_near_edges(st, carrier_crs={"c_one": "EPSG:4326"}),
            "GEOGRAPHIC_CRS")
        ok_nocrs, code_nocrs = refuses(
            lambda: B.build_near_edges(st, carrier_crs={}),
            "NO_CRS_FOR_CARRIER")
        near = B.build_near_edges(st, carrier_crs={"c_one": "EPSG:32635"}, k=2)
        check("B4 near via pinned city2graph knn: geographic CRS refuses, "
              "missing carrier CRS refuses, edges carry carrier+CRS+factor",
              ok_geo and ok_nocrs and near
              and all(e["metric_crs"] == "EPSG:32635"
                      and e["carrier_key"] == "c_one"
                      and e["unit_conversion_factor"] == 1.0
                      and e["distance_m"] > 0
                      and e["builder"].startswith("city2graph.knn_graph")
                      for e in near),
              f"{code_geo}/{code_nocrs}")

        # B4b (F5): two carriers -> no cross-carrier geometric edge, ever
        two = [dict(r) for r in st] + [
            {**st[0], "station_id": "US.NY01", "carrier_key": "c_ny",
             "lon": -73.99, "lat": 40.75, "segment_name": None,
             "pool_segment": "ny", "registry_selected": False},
            {**st[0], "station_id": "US.NY02", "carrier_key": "c_ny",
             "lon": -73.95, "lat": 40.78, "segment_name": None,
             "pool_segment": "ny", "registry_selected": False}]
        near2 = B.build_near_edges(
            two, carrier_crs={"c_one": "EPSG:32635", "c_ny": "EPSG:32618"}, k=2)
        by_carrier_ids = {"c_one": {r["station_id"] for r in st},
                          "c_ny": {"US.NY01", "US.NY02"}}
        check("B4b two carriers never share a geometric edge",
              near2 and all(e["station_a"] in by_carrier_ids[e["carrier_key"]]
                            and e["station_b"] in by_carrier_ids[e["carrier_key"]]
                            for e in near2))

        # B4c (F5): EPSG:2263 US-survey-foot -> TRUE metres vs Geod oracle
        from pyproj import Geod
        ny = [t for t in two if t["carrier_key"] == "c_ny"]
        near_ft = B.build_near_edges(ny, carrier_crs={"c_ny": "EPSG:2263"}, k=1)
        e = near_ft[0]
        a = next(t for t in ny if t["station_id"] == e["station_a"])
        b = next(t for t in ny if t["station_id"] == e["station_b"])
        _az, _baz, dist = Geod(ellps="WGS84").inv(a["lon"], a["lat"],
                                                  b["lon"], b["lat"])
        ok_ft = (abs(e["unit_conversion_factor"] - 0.30480060960121924) < 1e-12
                 and abs(e["distance_m"] - dist) / dist < 0.01)
        check("B4c EPSG:2263 converted to TRUE metres (factor recorded; within "
              "1% of independent Geod ellipsoid distance)", ok_ft,
              f"edge {e['distance_m']}m vs geod {dist:.1f}m factor "
              f"{e['unit_conversion_factor']}")
    except B.CapabilityUnavailable as e:
        cap("B4/B4b/B4c near classes", str(e))

    r = np.array([[1.0, 0.5, np.nan], [0.5, 1.0, -0.3], [np.nan, -0.3, 1.0]])
    man = _manifest(IDS3)
    edges = B.build_coherence_edges(r, man, st)
    dig3 = man["station_index_digest"]
    check("B5 coherence: finite-only, NaN pair ABSENT, sorted, unit=1, "
          "index digest rides every edge",
          [(e["station_a"], e["station_b"]) for e in edges]
          == [("KO.A01", "KO.A02"), ("KO.A02", "KO.B01")]
          and all(e["unit"] == 1 and e["station_index_digest"] == dig3
                  for e in edges))
    man0 = _manifest(IDS3)
    man0["n_overlap"][0][1] = 0
    ok5b, code5b = refuses(lambda: B.build_coherence_edges(r, man0, st),
                           "EDGE_WITHOUT_SUPPORT")
    check("B5b finite r with zero n_overlap REFUSES", ok5b, code5b)

    man_sp = _manifest(["KO.A01", "KO.SPARE"])
    r2 = np.array([[1.0, 0.4], [0.4, 1.0]])
    ok6, code6 = refuses(lambda: B.build_coherence_edges(r2, man_sp, st),
                         "UNSELECTED_STATION_EDGE")
    check("B6 unselected spare in matrix index REFUSES", ok6, code6)

    snap1 = _snap("2026-03-02", edges, dig3,
                  {s: "FETCHED" for s in IDS3})
    bad_edge = [{**edges[0], "carrier_key": "c_two"}]
    ok7, code7 = refuses(lambda: _snap("2026-03-02", bad_edge, dig3),
                         "SNAPSHOT_CARRIER_MIX")
    ok7b, code7b = refuses(lambda: _snap("2026-03-02", edges, "0" * 64),
                           "SNAPSHOT_INDEX_MIX")
    check("B7 snapshot refuses carrier mixing AND index-frame mixing",
          ok7 and ok7b, f"{code7}/{code7b}")

    days = ["2026-03-01", "2026-03-02", "2026-03-03"]
    r_prev = np.array([[1.0, 0.4, 0.1], [0.4, 1.0, np.nan],
                       [0.1, np.nan, 1.0]])
    man_prev = _manifest(IDS3, day="2026-03-01")
    snap0 = _snap("2026-03-01", B.build_coherence_edges(r_prev, man_prev, st),
                  dig3)
    delta = B.build_delta_table(snap0, snap1, days)
    by_pair = {(d["station_a"], d["station_b"]): d for d in delta}
    d_ab = by_pair[("KO.A01", "KO.A02")]
    d_ac = by_pair[("KO.A01", "KO.B01")]
    d_bc = by_pair[("KO.A02", "KO.B01")]
    check("B8 delta typed: comparable exact; absent pairs NOT_COMPARABLE with "
          "reasons, delta_r None never 0",
          d_ab["comparable"] and abs(d_ab["delta_r"] - 0.1) < 1e-12
          and not d_ac["comparable"] and d_ac["reason"]
          == "PAIR_ABSENT_CURRENT_DAY"
          and not d_bc["comparable"] and d_bc["reason"]
          == "PAIR_ABSENT_PREVIOUS_DAY")
    snap3 = _snap("2026-03-03", [], dig3)
    ok8b, code8b = refuses(lambda: B.build_delta_table(snap0, snap3, days),
                           "NOT_PREVIOUS_REGISTERED_DAY")
    check("B8b non-adjacent registered days REFUSE", ok8b, code8b)
    man_alg = _manifest(IDS3, alg="b" * 64)
    snap_alg = _snap("2026-03-02", B.build_coherence_edges(r, man_alg, st), dig3)
    d_alg = {(d["station_a"], d["station_b"]): d
             for d in B.build_delta_table(snap0, snap_alg, days)}
    check("B8c algorithm mismatch -> NOT_COMPARABLE",
          d_alg[("KO.A01", "KO.A02")]["comparable"] is False
          and d_alg[("KO.A01", "KO.A02")]["reason"] == "ALGORITHM_MISMATCH")
    # B8d (F4): the codex counterexample -- A-B day0 vs A-B-C day1
    ids2 = ["KO.A01", "KO.A02"]
    man2 = _manifest(ids2, day="2026-03-01")
    snap_ab = _snap("2026-03-01",
                    B.build_coherence_edges(np.array([[1.0, 0.4], [0.4, 1.0]]),
                                            man2, st),
                    man2["station_index_digest"])
    d_frame = B.build_delta_table(snap_ab, snap1, days)
    check("B8d index-frame counterexample: same pair+algorithm, A-B vs A-B-C "
          "-> EVERY row NOT_COMPARABLE/STATION_INDEX_MISMATCH",
          d_frame and all(row["comparable"] is False
                          and row["reason"] == "STATION_INDEX_MISMATCH"
                          and row["delta_r"] is None for row in d_frame))

    # B9 (F3): flag sequence + LIVE derivation isolation via the real producer
    td = tempfile.mkdtemp()
    mp = os.path.join(td, "m.npy")
    fp = os.path.join(td, "m.manifest.json")
    np.save(mp, r2)
    with open(fp, "wb") as fh:
        fh.write(B.canon_bytes(_manifest(ids2)))
    flags = []
    def _vok(root, m, f, recompute=False):
        flags.append(recompute)
        return True, []
    B.ingest_matrix(td, mp, fp, verifier=_vok)
    check("B9 ingest verifier flag sequence is EXACTLY [recompute=True, "
          "recompute=False]", flags == [True, False], str(flags))
    try:
        import importlib
        bar = open(os.path.join(HERE,
                                "test_f2g_matrix_producer_redkats_cayley.py"),
                   encoding="utf-8").read().replace("\nmain()\n", "\n")
        gg = {"__name__": "fx", "__file__": "bar"}
        exec(compile(bar, "bar", "exec"), gg)
        prod_root = tempfile.mkdtemp()
        gg["_mk_fixture"](prod_root)
        P = importlib.import_module("d2_f2g_matrix_producer")
        P.write_producer_identity(prod_root)
        P.produce_carrier_day_matrix(prod_root, "c_fix", "2026-03-02",
                                     out_dir=os.path.join(prod_root, "out"))
        pmp = os.path.join(prod_root, "out", "c_fix", "2026-03-02.matrix.npy")
        pfp = os.path.join(prod_root, "out", "c_fix",
                           "2026-03-02.manifest.json")
        _rm, _mm = B.ingest_matrix(prod_root, pmp, pfp)   # production default
        arr = np.load(pmp)
        i, j = 0, 1
        arr[i, j] = arr[j, i] = float(np.clip(arr[i, j] + 1e-3, -0.999, 0.999))
        np.save(pmp, arr)
        with open(pmp, "rb") as fh:
            newb = fh.read()
        mdoc = json.loads(open(pfp, "rb").read().decode("utf-8"))
        mdoc["matrix_sha256"] = B.sha(newb)
        mdoc["matrix_size"] = len(newb)
        with open(pfp, "wb") as fh:
            fh.write(B.canon_bytes(mdoc))
        ok9l, code9l = refuses(lambda: B.ingest_matrix(prod_root, pmp, pfp),
                               "MATRIX_VERIFY_FAILED_PRE")
        check("B9-live REAL producer: produced artifact INGESTS clean, then an "
              "in-bounds self-consistently-rehashed doctor REFUSES pre-use "
              "(derivation isolation: recompute is the only catch)",
              ok9l, code9l)
    except (ImportError, ModuleNotFoundError) as e:
        cap("B9-live real-producer derivation", str(e))

    # B10 (F2): heterogeneous MultiGraph round trip -- pair with BOTH relations
    near_syn = near if near else [
        {"type": "near", "carrier_key": "c_one", "station_a": "KO.A01",
         "station_b": "KO.A02", "distance_m": 12345.0,
         "unit_conversion_factor": 1.0, "source_crs": "EPSG:4326",
         "metric_crs": "EPSG:32635", "builder": "fixture"}]
    node_tables = {"station": st, "segment": sg, "carrier": ca}
    edge_tables = {"coheres_with": edges, "near": near_syn, "member_of": mo,
                   "adjacent_to": adj, "contains": cont}
    try:
        G = B.to_networkx(node_tables, edge_tables)
        nt2, et2 = B.from_networkx(G)
        pair_ok = any(e["station_a"] == "KO.A01" and e["station_b"] == "KO.A02"
                      for e in et2.get("near", [])) \
            and any(e["station_a"] == "KO.A01" and e["station_b"] == "KO.A02"
                    for e in et2.get("coheres_with", []))
        tables_ok = all(B.canon_jsonl(nt2[k]) == B.canon_jsonl(node_tables[k])
                        for k in node_tables) \
            and all(B.canon_jsonl(et2[k]) == B.canon_jsonl(edge_tables[k])
                    for k in edge_tables)
        check("B10 hetero MultiGraph round trip: near+coheres_with on the SAME "
              "pair both survive; ALL node+edge tables canonical-byte-equal",
              pair_ok and tables_ok)
    except B.CapabilityUnavailable as e:
        cap("B10 hetero NetworkX round trip", str(e))

    try:
        gdfs = B.to_geodataframes(st, near_syn)
        back = B.from_geodataframe(gdfs["station"])
        check("B11 GeoDataFrame round trip preserves identity columns",
              B.canon_jsonl(sorted(back, key=lambda r_: r_["station_id"]))
              == B.canon_jsonl(st))
    except B.CapabilityUnavailable as e:
        cap("B11 GeoDataFrame identity", str(e))

    try:
        data = B.to_pyg(node_tables, edge_tables)
        back = B.from_pyg(data)
        nb, eb = back if isinstance(back, tuple) else (back, {})
        def _etbl(key):
            for k in eb:
                if (k[1] if isinstance(k, tuple) else k) == key:
                    return B.from_geodataframe(eb[k])
            return None
        pair_both = _etbl("near") is not None and _etbl("coheres_with") is not None
        # codex R1.2-A: FULL heterogeneous identity via bridge structure +
        # structure-bound sidecar ledger -- LIVE GREEN required, never a hold
        node_ok = all(
            B.canon_jsonl(sorted(B.from_geodataframe(nb[kind]),
                                 key=lambda r_: B.canon_bytes(r_)))
            == B.canon_jsonl(sorted(node_tables[kind],
                                    key=lambda r_: B.canon_bytes(r_)))
            for kind in node_tables)
        edge_ok = all(
            (lambda got: got is not None and B.canon_jsonl(
                sorted(got, key=lambda r_: B.canon_bytes(r_)))
             == B.canon_jsonl(sorted(rows, key=lambda r_: B.canon_bytes(r_))))
            (_etbl(name))
            for name, rows in edge_tables.items() if rows)
        adapter_ok = (getattr(data, "f2g_adapter", None) is not None
                      and "bridge_limitation" in data.f2g_adapter
                      and data.f2g_adapter["schema"] == B.SIDECAR_SCHEMA)
        check("B12 LIVE: bridge structure + structure-bound sidecar -- every "
              "node+edge table canonical after the round trip; same-pair "
              "near+coherence both survive; adapter division recorded",
              node_ok and edge_ok and pair_both and adapter_ok)

        def _refused_pyg(mutate):
            d2 = B.to_pyg(node_tables, edge_tables)
            mutate(d2)
            try:
                B.from_pyg(d2)
                return False, "no refusal"
            except B.F2GRefusal as e:
                return True, e.reason_code
        trip_coh = ("station", "coheres_with", "station")
        trip_con = ("carrier", "contains", "segment")
        ok_sw, code_sw = _refused_pyg(lambda d2: (
            setattr(d2[trip_coh], "f2g_sidecar", d2[trip_con].f2g_sidecar),
            setattr(d2[trip_con], "f2g_sidecar", d2[trip_coh].f2g_sidecar)))
        def _reorder(d2):
            sc = dict(d2["station"].f2g_sidecar)
            body = {k: v for k, v in sc.items() if k != "sha256"}
            body["identity_keys"] = list(reversed(body["identity_keys"]))
            body["values"] = {c: list(reversed(v))
                              for c, v in body["values"].items()}
            d2["station"].f2g_sidecar = {**body,
                                         "sha256": B.sha(B.canon_bytes(body))}
        ok_ro, code_ro = _refused_pyg(_reorder)
        def _tamper(d2):
            sc = dict(d2["station"].f2g_sidecar)
            sc["values"] = {**sc["values"],
                            "carrier_key": ["c_evil"] * len(
                                sc["values"]["carrier_key"])}
            d2["station"].f2g_sidecar = sc          # stale digest kept
        ok_tv, code_tv = _refused_pyg(_tamper)
        check("B12b sidecar binding doctors: same-shaped sidecar SWAP, row "
              "REORDER (rehashed), and value TAMPER (stale digest) each "
              "REFUSE typed", ok_sw and ok_ro and ok_tv,
              f"{code_sw}/{code_ro}/{code_tv}")
    except B.CapabilityUnavailable as e:
        cap("B12 PyG bridge FULL-identity round trip (packet-time live bar)",
            str(e))

    try:
        st_x = st + [{**st[0], "station_id": "KO.NOXY", "lon": None,
                      "lat": None, "coordinates_available": False}]
        out = B.render_map(st_x, edges, os.path.join(td, "render.png"),
                           title_suffix="fixture c_one 2026-03-02")
        check("B13 render writes file; mandated label unconditional; "
              "geometry-excluded stations DECLARED in render metadata",
              os.path.getsize(out["path"]) > 0
              and B.RENDER_LABEL.startswith("seismic envelope coherence")
              and out["geometry_excluded_station_ids"] == ["KO.NOXY"]
              and out["geometry_excluded_count"] == 1
              and out["geometry_excluded_reason"] is not None)
    except B.CapabilityUnavailable as e:
        cap("B13 render label + excluded record", str(e))

    # ---- B15: cross_host_consumer_v1 profile (codex ruling 82c31cf3) ----------
    try:
        import obspy as _ob
        import io as _io
        import importlib as _il
        from datetime import datetime as _dt, timezone as _tz
        P = _il.import_module("d2_f2g_matrix_producer")
        import numpy as _np

        def _b15_root():
            rt = tempfile.mkdtemp()
            os.makedirs(os.path.join(rt, "raw_objects"), exist_ok=True)
            objs = []
            for sid in ("KO.X01", "KO.X02", "KO.X03"):
                tr = _ob.Trace(data=(np.sin(2 * np.pi * 2.5 * np.arange(0, 1200, 0.02))
                                     + 0.05 * np.cos(2 * np.pi * 6.0 * np.arange(0, 1200, 0.02))))
                tr.stats.sampling_rate = 50.0
                tr.stats.starttime = _ob.UTCDateTime("2026-03-02T00:00:00")
                tr.stats.network, tr.stats.station, tr.stats.channel = \
                    sid.split(".")[0], sid.split(".")[1], "HHZ"
                buf = _io.BytesIO()
                _ob.Stream([tr]).write(buf, format="MSEED")
                bb = buf.getvalue()
                with open(os.path.join(rt, "raw_objects", B.sha(bb) + ".ms"),
                          "wb") as fh:
                    fh.write(bb)
                objs.append({"sha256": B.sha(bb), "size": len(bb),
                             "relative_path": f"raw_objects/{B.sha(bb)}.ms",
                             "kind": "archive-seismic-miniseed-fragments-v1",
                             "carrier_key": "c_fix", "scored_day": "2026-03-02",
                             "segment_name": "seg_a",
                             "source_id": sid + "..HHZ", "provider": "KOERI",
                             "reuse_disposition": "FETCHED_NEW",
                             "start_utc": "2026-03-02T00:00:00.000000Z",
                             "end_utc": "2026-03-02T00:20:00.000000Z"})
            with open(os.path.join(rt, "input_manifest.json"), "wb") as fh:
                fh.write(B.canon_bytes({"schema": "im-v2-resume",
                                        "producer_commit": "a" * 40,
                                        "implementation_commit": "a" * 40,
                                        "objects": objs}))
            P.write_producer_identity(rt)
            man = P.produce_carrier_day_matrix(
                rt, "c_fix", "2026-03-02", out_dir=os.path.join(rt, "out"))
            mp_ = os.path.join(rt, "out", "c_fix", "2026-03-02.matrix.npy")
            fp_ = os.path.join(rt, "out", "c_fix", "2026-03-02.manifest.json")
            return rt, mp_, fp_, man

        def _doctor(rt, mp_, fp_, offset, cell=(0, 1)):
            m_ = np.load(mp_)
            i_, j_ = cell
            m_[i_, j_] = m_[j_, i_] = m_[i_, j_] + offset
            import io as _io2
            buf = _io2.BytesIO()
            np.save(buf, np.asarray(m_, dtype="<f8", order="C"))
            body = buf.getvalue()
            md = json.loads(open(fp_, "rb").read().decode("utf-8"))
            md["matrix_sha256"], md["matrix_size"] = B.sha(body), len(body)
            with open(mp_, "wb") as fh:
                fh.write(body)
            with open(fp_, "wb") as fh:
                fh.write(B.canon_bytes(md))

        rt15, mp15, fp15, _man15 = _b15_root()
        _r, _m, rc0 = B.ingest_matrix_cross_host(rt15, mp15, fp15)
        check("B15-0 same-host produced artifact: exact mode PASSES, delta 0, "
              "receipt carries profile + both env locks",
              rc0["mode"] == "exact" and rc0["observed_max_abs_delta"] == 0.0
              and rc0["profile"] == "cross_host_consumer_v1"
              and rc0["producer_environment_lock_digest"] is not None
              and rc0["consumer_environment_lock"]["obspy"] is not None)

        _doctor(rt15, mp15, fp15, -5e-10)
        _r, _m, rc1 = B.ingest_matrix_cross_host(rt15, mp15, fp15)
        check("B15a-lo delta 5e-10 (below 1e-9): comparator engages and PASSES "
              "with the delta recorded",
              rc1["mode"] == "cross_host_comparator"
              and 4e-10 < rc1["observed_max_abs_delta"] < 1e-9)
        _doctor(rt15, mp15, fp15, -(2e-9 - 5e-10))   # cumulative ~-2e-9
        try:
            B.ingest_matrix_cross_host(rt15, mp15, fp15)
            check("B15a-hi delta ~2e-9 (above 1e-9) REFUSES", False,
                  "ACCEPTED above-threshold delta")
        except B.F2GRefusal as e:
            check("B15a-hi delta ~2e-9 (above 1e-9) REFUSES",
                  e.reason_code == "CROSS_HOST_DELTA_EXCEEDED", e.reason_code)

        rt15b, mp15b, fp15b, _ = _b15_root()
        m_ = np.load(mp15b)
        m_[0, 1] = m_[1, 0] = np.nan          # finite-mask difference + doctored
        import io as _io3
        buf = _io3.BytesIO(); np.save(buf, np.asarray(m_, dtype="<f8", order="C"))
        body = buf.getvalue()
        md = json.loads(open(fp15b, "rb").read().decode("utf-8"))
        md["matrix_sha256"], md["matrix_size"] = B.sha(body), len(body)
        open(mp15b, "wb").write(body); open(fp15b, "wb").write(B.canon_bytes(md))
        try:
            B.ingest_matrix_cross_host(rt15b, mp15b, fp15b)
            check("B15b finite-mask difference REFUSES", False, "ACCEPTED")
        except B.F2GRefusal as e:
            check("B15b finite-mask difference REFUSES", True)

        rt15c, mp15c, fp15c, _ = _b15_root()
        m_ = np.load(mp15c); m_[0, 1] = m_[1, 0] = m_[0, 1] - 1e-3
        buf = _io3.BytesIO(); np.save(buf, np.asarray(m_, dtype="<f8", order="C"))
        body = buf.getvalue()
        md = json.loads(open(fp15c, "rb").read().decode("utf-8"))
        md["matrix_sha256"], md["matrix_size"] = B.sha(body), len(body)
        md["n_overlap"][0][1] = md["n_overlap"][0][1] + 1   # extra reason rides
        open(mp15c, "wb").write(body); open(fp15c, "wb").write(B.canon_bytes(md))
        try:
            B.ingest_matrix_cross_host(rt15c, mp15c, fp15c)
            check("B15c DERIVATION_MISMATCH plus another reason REFUSES "
                  "(comparator must not engage)", False, "ACCEPTED")
        except B.F2GRefusal as e:
            check("B15c DERIVATION_MISMATCH plus another reason REFUSES "
                  "(comparator must not engage)",
                  e.reason_code == "MATRIX_VERIFY_FAILED_PRE", e.reason_code)

        rt15d, mp15d, fp15d, _ = _b15_root()
        _doctor(rt15d, mp15d, fp15d, -1e-1)   # P17c-order swapped-semantics delta
        try:
            B.ingest_matrix_cross_host(rt15d, mp15d, fp15d)
            check("B15d swapped-semantics-order delta (1e-1) still REFUSES",
                  False, "ACCEPTED")
        except B.F2GRefusal as e:
            check("B15d swapped-semantics-order delta (1e-1) still REFUSES",
                  e.reason_code == "CROSS_HOST_DELTA_EXCEEDED", e.reason_code)

        rt15f, mp15f, fp15f, _ = _b15_root()
        m_ = np.load(mp15f)
        ulp = np.nextafter(m_[0, 1], -np.inf) - m_[0, 1]
        _doctor(rt15f, mp15f, fp15f, ulp)
        ok_ex, rs_ex = P.verify_matrix_artifact(rt15f, mp15f, fp15f,
                                                recompute=True)
        _r, _m, rc_f = B.ingest_matrix_cross_host(rt15f, mp15f, fp15f)
        check("B15f role split: producer EXACT mode rejects a one-ULP "
              "re-derivation difference; the consumer profile accepts it with "
              "the tiny delta RECORDED",
              (not ok_ex) and rs_ex == ["DERIVATION_MISMATCH"]
              and rc_f["mode"] == "cross_host_comparator"
              and 0 < rc_f["observed_max_abs_delta"] < 1e-12)

        import inspect as _ins
        sig15 = str(_ins.signature(B.ingest_matrix_cross_host))
        check("B15e no override: profile signature exposes NO tolerance/"
              "threshold parameter",
              "tol" not in sig15 and "threshold" not in sig15
              and "delta" not in sig15, sig15)
    except ImportError as e:
        cap("B15 cross_host_consumer_v1 profile bars", f"obspy absent: {e}")

    # ---- B16: codex final-verify closures (306d1a5d) --------------------------
    # B16b snapshot-state coverage over the FULL selected registry (core)
    man16 = {"carrier_key": "c_one", "day": "2026-03-02",
             "station_ids": ["KO.A01", "KO.A02"],
             "reason_codes": ["SERIES_UNAVAILABLE:KO.B01"]}
    uni16 = [{"station_id": s} for s in ("KO.A01", "KO.A02", "KO.B01")]
    states16 = B.build_snapshot_states(man16,
                                       ["KO.A01", "KO.A02", "KO.B01",
                                        "KO.C01"], uni16)
    check("B16b snapshot states cover the FULL selected registry with typed "
          "absence (MEASURED / exact manifest reason / NO_BOUND_OBJECT)",
          states16 == {"KO.A01": "MEASURED", "KO.A02": "MEASURED",
                       "KO.B01": "SERIES_UNAVAILABLE:KO.B01",
                       "KO.C01": "NO_BOUND_OBJECT"})
    ok16b, code16b = refuses(
        lambda: B.build_snapshot_states(
            {"carrier_key": "c_one", "day": "2026-03-02",
             "station_ids": ["KO.A01"], "reason_codes": []},
            ["KO.A01", "KO.B01"], uni16),
        "UNEXPLAINED_SELECTED_STATION_ABSENCE")
    check("B16b2 in-universe selected absence with NO reason REFUSES",
          ok16b, code16b)

    # B16a carrier-local render (matplotlib-gated)
    try:
        mixed = st + [{**st[0], "station_id": "US.MX01",
                       "carrier_key": "c_two", "lon": -116.5, "lat": 33.8}]
        ok16a, code16a = refuses(
            lambda: B.render_map(mixed, edges,
                                 os.path.join(td, "mix.png")),
            "RENDER_CARRIER_MIX")
        two_only = [{**st[0], "station_id": "US.T01", "carrier_key": "c_two",
                     "lon": -116.5, "lat": 33.8},
                    {**st[0], "station_id": "US.T02", "carrier_key": "c_two",
                     "lon": -116.4, "lat": 33.9, "coordinates_available": True}]
        out16 = B.render_map(two_only, [], os.path.join(td, "ctwo.png"))
        check("B16a render is carrier-local: mixing REFUSES; a carrier without "
              "the coordinate-less station reports NO exclusions",
              ok16a and out16["geometry_excluded_station_ids"] == [],
              f"{code16a}/{out16['geometry_excluded_station_ids']}")
    except B.CapabilityUnavailable as e:
        cap("B16a carrier-local render", str(e))

    # B16c sidecar insertion-order doctor (geopandas-gated)
    try:
        import geopandas as gpd16
        from shapely.geometry import Point as P16
        rows_a = [{"station_id": "KO.Z01", "alpha": 1, "beta": "x"},
                  {"station_id": "KO.Z02", "alpha": 2, "beta": "y"}]
        rows_b = [dict(reversed(list(r.items()))) for r in rows_a]
        g_a = gpd16.GeoDataFrame(rows_a, geometry=[P16(0, 0), P16(1, 1)],
                                 crs="EPSG:4326")
        g_b = gpd16.GeoDataFrame(rows_b, geometry=[P16(0, 0), P16(1, 1)],
                                 crs="EPSG:4326")
        sc_a = B._make_sidecar("node", "z", ["KO.Z01", "KO.Z02"], g_a)
        sc_b = B._make_sidecar("node", "z", ["KO.Z01", "KO.Z02"], g_b)
        check("B16c sidecar digest is insertion-order independent (canonical "
              "sorted columns)", sc_a["sha256"] == sc_b["sha256"]
              and sc_a["columns"] == sorted(sc_a["columns"]))
    except (ImportError, B.CapabilityUnavailable) as e:
        cap("B16c sidecar insertion-order doctor", str(e))

    res = B.phase_a_result(input_digests={"plan": "x" * 64},
                           code_digests={"builder": "y" * 64},
                           output_digests={}, bar_results={}, status="FIXTURE",
                           geometry_excluded_station_ids=["KO.KHMN"])
    check("B14 phase_a_result carries the four standing non-claims AND the "
          "geometry-excluded record (KO.KHMN ruling)",
          len(res["non_claims"]) == 4
          and any("INCONCLUSIVE" in c for c in res["non_claims"])
          and res["geometry_excluded_station_ids"] == ["KO.KHMN"]
          and res["geometry_excluded_count"] == 1)


main()
print()
if CAPS:
    print(f"CAPABILITY-GATED (explicit, not silent): {CAPS}")
if FAILS:
    print(f"F2G GRAPH-BUILDER KAT FAILURES ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("ALL F2G GRAPH-BUILDER KATs PASS"
      + (" (capability gaps above)" if CAPS else ""))
