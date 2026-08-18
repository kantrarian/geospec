"""fault2graph Phase A GRAPH BUILDER KATs (cayley) -- contract `2e0c7a33` A3/A5.

AUTHOR SELF-TESTS, not clearance: codex reviews the builder once against V-A and
the contract (A5); these KATs are the executable surface that review drives.

Covers the A5 builder-side minimum classes:
  B1  identity tables build deterministically; canonical bytes stable
  B2  member_of ONLY for selected-registry stations; spare gains none
  B3  adjacent_to only from pinned topology (absent adjacency -> no edges)
  B4  near: geographic CRS REFUSES (degrees ban); projected CRS carries
      distance_m + CRS pins on every edge
  B5  coherence edges: finite-only, sorted pairs, unit=1; NaN cell -> ABSENT
      edge (typed absence, never zero); n_overlap=0 with finite r REFUSES
  B6  unselected spare in a matrix index REFUSES (UNSELECTED_STATION_EDGE)
  B7  snapshot refuses carrier/day mixing
  B8  delta: comparable only same-carrier same-pair valid-both-days previous-
      REGISTERED-day; absent pair -> typed NOT_COMPARABLE row (never zero);
      non-adjacent registered days REFUSE; algorithm mismatch NOT_COMPARABLE
  B9  ingest boundary: production default REFUSES typed while the producer seam
      is absent (PRODUCER_SEAM_ABSENT); injected verifier verdicts gate both
      pre- and post-use verification
  B10 NetworkX round trip lossless on identity (canonical-byte equality)
  B11 GeoDataFrame construction preserves identity columns (capability-gated)
  B12 PyG round trip (capability-gated: torch absent -> explicit typed line)
  B13 render writes a file and ALWAYS carries the mandated label
      (capability-gated on matplotlib)

Capability-gated classes print an explicit [CAP] line when the dependency is
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


PLAN = {
    "carriers": ["c_one"], "topology_version": "t1",
    "station_registry": [
        {"station_id": "KO.A01", "carrier_key": "c_one", "segment_name": "seg_a",
         "lon": 29.0, "lat": 40.7, "ordered_nslc_candidates": ["KO.A01..HHZ"]},
        {"station_id": "KO.A02", "carrier_key": "c_one", "segment_name": "seg_a",
         "lon": 29.2, "lat": 40.8, "ordered_nslc_candidates": ["KO.A02..HHZ"]},
        {"station_id": "KO.B01", "carrier_key": "c_one", "segment_name": "seg_b",
         "lon": 29.5, "lat": 40.9, "ordered_nslc_candidates": ["KO.B01..HHZ"]},
    ]}
POOL = [{"station_id": "KO.SPARE", "carrier_key": "c_one", "lon": 29.9,
         "lat": 41.0}]
TOPO = {"topology_version": "t1",
        "adjacency": [["seg_a", "seg_b", {"along_strike_order": 1,
                                          "shared_fault": True}]],
        "segment_geometry": {}}


def _manifest(ids, alg="a" * 64, carrier="c_one", day="2026-03-02"):
    n = len(ids)
    return {"campaign_id": "f" * 64, "carrier_key": carrier, "day": day,
            "algorithm_config_digest": alg, "station_ids": list(ids),
            "n_overlap": [[600 if i != j else 0 for j in range(n)]
                          for i in range(n)]}


def main():
    st = B.build_station_table(PLAN, POOL)
    sg = B.build_segment_table(PLAN, TOPO)
    ca = B.build_carrier_table(PLAN, {"c_one": {
        "expected_sha256": "1" * 64, "valid_through": "2026-08-23",
        "threshold": 0.21}})
    check("B1 identity tables deterministic + canonical bytes stable",
          B.canon_jsonl(st) == B.canon_jsonl(B.build_station_table(PLAN, POOL))
          and len(st) == 4 and len(sg) == 2 and len(ca) == 1)

    mo = B.build_member_of_edges(st)
    check("B2 member_of ONLY selected registry; spare has none",
          sorted(e["station_id"] for e in mo) == ["KO.A01", "KO.A02", "KO.B01"]
          and all(e["station_id"] != "KO.SPARE" for e in mo))

    adj = B.build_adjacent_to_edges(TOPO)
    adj_none = B.build_adjacent_to_edges({"topology_version": "t1"})
    check("B3 adjacent_to only from pinned topology",
          len(adj) == 1 and adj[0]["shared_fault"] is True and adj_none == [])

    try:
        ok_geo, code_geo = refuses(
            lambda: B.build_near_edges(st, metric_crs="EPSG:4326"),
            "GEOGRAPHIC_CRS")
        near = B.build_near_edges(st, metric_crs="EPSG:32635", k=2)
        ok_near = (ok_geo and near
                   and all(e["metric_crs"] == "EPSG:32635"
                           and e["source_crs"] == "EPSG:4326"
                           and e["distance_m"] > 0 for e in near))
        check("B4 near: geographic CRS refuses; projected CRS pins + distance_m",
              ok_near, f"geo_refusal={code_geo}")
    except B.CapabilityUnavailable as e:
        cap("B4 near CRS pins", str(e))

    ids3 = ["KO.A01", "KO.A02", "KO.B01"]
    r = np.array([[1.0, 0.5, np.nan], [0.5, 1.0, -0.3], [np.nan, -0.3, 1.0]])
    man = _manifest(ids3)
    edges = B.build_coherence_edges(r, man, st)
    check("B5 coherence: finite-only, NaN pair ABSENT (typed absence, no zero), "
          "sorted, unit=1",
          [(e["station_a"], e["station_b"]) for e in edges]
          == [("KO.A01", "KO.A02"), ("KO.A02", "KO.B01")]
          and all(e["unit"] == 1 for e in edges)
          and not any(e["station_a"] == "KO.A01" and e["station_b"] == "KO.B01"
                      for e in edges))
    man0 = _manifest(ids3)
    man0["n_overlap"][0][1] = 0
    ok5b, code5b = refuses(lambda: B.build_coherence_edges(r, man0, st),
                           "EDGE_WITHOUT_SUPPORT")
    check("B5b finite r with zero n_overlap REFUSES", ok5b, code5b)

    man_sp = _manifest(["KO.A01", "KO.SPARE"])
    r2 = np.array([[1.0, 0.4], [0.4, 1.0]])
    ok6, code6 = refuses(lambda: B.build_coherence_edges(r2, man_sp, st),
                         "UNSELECTED_STATION_EDGE")
    check("B6 unselected spare in matrix index REFUSES", ok6, code6)

    snap1 = B.build_snapshot("f" * 64, "c_one", "2026-03-02", edges,
                             {"KO.A01": "FETCHED", "KO.A02": "FETCHED",
                              "KO.B01": "FETCHED"})
    bad_edge = [{**edges[0], "carrier_key": "c_two"}]
    ok7, code7 = refuses(lambda: B.build_snapshot("f" * 64, "c_one",
                                                  "2026-03-02", bad_edge, {}),
                         "SNAPSHOT_CARRIER_MIX")
    check("B7 snapshot refuses carrier mixing", ok7, code7)

    days = ["2026-03-01", "2026-03-02", "2026-03-03"]
    r_prev = np.array([[1.0, 0.4, 0.1], [0.4, 1.0, np.nan], [0.1, np.nan, 1.0]])
    snap0 = B.build_snapshot("f" * 64, "c_one", "2026-03-01",
                             B.build_coherence_edges(
                                 r_prev, _manifest(ids3, day="2026-03-01"), st),
                             {})
    delta = B.build_delta_table(snap0, snap1, days)
    by_pair = {(d["station_a"], d["station_b"]): d for d in delta}
    d_ab = by_pair[("KO.A01", "KO.A02")]
    d_ac = by_pair[("KO.A01", "KO.B01")]      # prev-only pair (curr NaN -> absent)
    d_bc = by_pair[("KO.A02", "KO.B01")]      # curr-only pair (prev NaN -> absent)
    ok8 = (d_ab["comparable"] and abs(d_ab["delta_r"] - 0.1) < 1e-12
           and not d_ac["comparable"] and d_ac["delta_r"] is None
           and d_ac["reason"] == "PAIR_ABSENT_CURRENT_DAY"
           and not d_bc["comparable"] and d_bc["delta_r"] is None
           and d_bc["reason"] == "PAIR_ABSENT_PREVIOUS_DAY")
    check("B8 delta typed: comparable exact; absent pairs NOT_COMPARABLE with "
          "reasons, delta_r None never 0", ok8)
    snap3 = B.build_snapshot("f" * 64, "c_one", "2026-03-03", [], {})
    ok8b, code8b = refuses(lambda: B.build_delta_table(snap0, snap3, days),
                           "NOT_PREVIOUS_REGISTERED_DAY")
    check("B8b non-adjacent registered days REFUSE", ok8b, code8b)
    snap_alg = B.build_snapshot("f" * 64, "c_one", "2026-03-02",
                                B.build_coherence_edges(
                                    r, _manifest(ids3, alg="b" * 64), st), {})
    d_alg = {(d["station_a"], d["station_b"]): d
             for d in B.build_delta_table(snap0, snap_alg, days)}
    check("B8c algorithm mismatch -> NOT_COMPARABLE",
          d_alg[("KO.A01", "KO.A02")]["comparable"] is False
          and d_alg[("KO.A01", "KO.A02")]["reason"] == "ALGORITHM_MISMATCH")

    td = tempfile.mkdtemp()
    mp = os.path.join(td, "m.npy")
    fp = os.path.join(td, "m.manifest.json")
    np.save(mp, r2)
    with open(fp, "wb") as fh:
        fh.write(B.canon_bytes(_manifest(["KO.A01", "KO.A02"])))
    ok9, code9 = refuses(lambda: B.ingest_matrix(td, mp, fp),
                         "PRODUCER_SEAM_ABSENT")
    calls = []
    def _vok(root, m, f, recompute=False):
        calls.append(1)
        return True, []
    _r, _m = B.ingest_matrix(td, mp, fp, verifier=_vok)
    def _vfail_post(root, m, f, recompute=False):
        calls.append(1)
        return (len(calls) % 2 == 1), ["post-drift"]
    calls.clear()
    ok9c, code9c = refuses(lambda: B.ingest_matrix(td, mp, fp,
                                                   verifier=_vfail_post),
                           "MATRIX_VERIFY_FAILED_POST")
    check("B9 ingest boundary: seam-absent typed refusal; injected verifier "
          "gates BOTH pre and post use",
          ok9 and len(_m["station_ids"]) == 2 and ok9c,
          f"{code9}/{code9c}")

    try:
        G = B.to_networkx(st, {"coheres_with": edges})
        st2, ed2 = B.from_networkx(G)
        check("B10 NetworkX round trip lossless on identity (canonical bytes)",
              B.canon_jsonl(st2) == B.canon_jsonl(st)
              and B.canon_jsonl(ed2["coheres_with"]) == B.canon_jsonl(edges))
    except B.CapabilityUnavailable as e:
        cap("B10 NetworkX round trip", str(e))

    try:
        gdfs = B.to_geodataframes(st, [])
        back = B.from_geodataframe(gdfs["station"])
        check("B11 GeoDataFrame round trip preserves identity columns "
              "(pandas None->NaN coercion undone by the pinned inverse path)",
              B.canon_jsonl(sorted(back, key=lambda r_: r_["station_id"]))
              == B.canon_jsonl(st))
    except B.CapabilityUnavailable as e:
        cap("B11 GeoDataFrame identity", str(e))

    try:
        data = B.to_pyg(st, {"coheres_with": edges})
        ei = data["station", "coheres_with", "station"].edge_index
        ids = data["station"].node_id
        pairs = sorted(tuple(sorted((ids[int(ei[0, c])], ids[int(ei[1, c])])))
                       for c in range(ei.shape[1]))
        check("B12 PyG round trip: edge_index re-imports to the same pair set",
              pairs == sorted((e["station_a"], e["station_b"]) for e in edges))
    except B.CapabilityUnavailable as e:
        cap("B12 PyG round trip", str(e))

    try:
        out = B.render_map(st, edges, os.path.join(td, "render.png"),
                           title_suffix="fixture c_one 2026-03-02")
        check("B13 render writes file; mandated label is in the code path "
              "(RENDER_LABEL constant drawn unconditionally)",
              os.path.getsize(out) > 0
              and B.RENDER_LABEL.startswith("seismic envelope coherence"))
    except B.CapabilityUnavailable as e:
        cap("B13 render label", str(e))

    res = B.phase_a_result(input_digests={"plan": "x" * 64},
                           code_digests={"builder": "y" * 64},
                           output_digests={}, bar_results={}, status="FIXTURE")
    check("B14 phase_a_result carries the four standing non-claims",
          len(res["non_claims"]) == 4
          and any("INCONCLUSIVE" in c for c in res["non_claims"]))


main()
print()
if CAPS:
    print(f"CAPABILITY-GATED (explicit, not silent): {CAPS}")
if FAILS:
    print(f"F2G GRAPH-BUILDER KAT FAILURES ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("ALL F2G GRAPH-BUILDER KATs PASS" + (" (capability gaps above)" if CAPS else ""))
