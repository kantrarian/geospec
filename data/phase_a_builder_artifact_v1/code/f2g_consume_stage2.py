"""fault2graph Phase A -- builder-consumption STAGE 2 (cayley).

Graph construction over the STAGE-1-VERIFIED matrices (every matrix already
passed cross_host_consumer_v1 ingest under the mirrored numeric lock). Runs in
the A5-pinned graph environment (f2g-env: py3.12, city2graph@3892a086, torch)
-- the stage split is deliberate and disclosed: stage 1 = the recompute
boundary under the numeric lock; stage 2 = arithmetic-free graph construction
(correlation values pass through untouched).

Outputs (contract A3): canonical node/edge tables, per-day G(day) snapshots,
per-carrier typed delta tables, NX + GDF + PyG-sidecar round-trip proofs on the
REAL tables, per-carrier labeled renders with the geometry-excluded record, and
phase_a_result binding every input/code/output digest.

Topology disclosure: the pinned t1 registries define segments and polygons but
NO adjacency relation; adjacent_to is therefore honestly EMPTY (A3 forbids
deriving adjacency from visual proximity). Carrier metric CRS pins: istanbul
EPSG:32635, turkey EPSG:32637, socal EPSG:32611 (UTM zones of the carriers).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "C:/geospec/monitoring/src")
import d2_f2g_graph_builder as B                                   # noqa: E402
import d2_campaign_v2_plan as V2P                                  # noqa: E402

VERIFIED = os.path.join(SCRATCH, "verified_matrices")
RECEIPTS = os.path.join(SCRATCH, "consume_receipts.jsonl")
OUT = os.path.join(SCRATCH, "builder_artifact_v1")
CARRIER_CRS = {"istanbul_marmara": "EPSG:32635",
               "turkey_kahramanmaras": "EPSG:32637",
               "socal_coachella": "EPSG:32611"}
PACKET_SUMMARY_SHA = ("df1e37ec6ca95dab1b4b24cfc5e7e3603f8fffe0"
                      "a8c4f76e4cfd20fa7cadc15c")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rd(p):
    with open(p, "rb") as fh:
        return fh.read()


def w(rel, body):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(body)
    return p


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    checks = {}

    # receipts: all 330 ok, deltas within threshold
    recs = [json.loads(x) for x in
            rd(RECEIPTS).decode("utf-8").splitlines()]
    ok_recs = [r for r in recs if r.get("ok")]
    days_by_carrier = {}
    for r in ok_recs:
        days_by_carrier.setdefault(r["carrier"], []).append(r["day"])
    for c in days_by_carrier:
        days_by_carrier[c] = sorted(set(days_by_carrier[c]))
    max_delta = max(r["observed_max_abs_delta"] for r in ok_recs)
    w("consume_receipts.jsonl", rd(RECEIPTS))   # per-day deltas ride the artifact
    checks["stage1_all_ok"] = (len(ok_recs) == 330
                               and not any(not r.get("ok") for r in recs
                                           if (r["carrier"], r["day"]) in
                                           {(x["carrier"], x["day"])
                                            for x in ok_recs}))
    checks["stage1_max_delta_within"] = max_delta <= B.CROSS_HOST_MAX_ABS_DELTA
    print(f"stage1: {len(ok_recs)} ok, max delta {max_delta:.3e}", flush=True)

    # identity tables via the PUBLIC bytes-authority seam
    bundle = rd("C:/geospec/monitoring/src/campaign_v2_phase05/"
                "phase0_bundle.json")
    _plan, plan_bytes = V2P.build_v2_campaign_plan(bundle)
    pool_bytes = rd("C:/geospec/monitoring/src/"
                    "d2_campaign_v2_candidate_pool.json")
    st = B.build_station_table(plan_bytes, pool_bytes)
    plan = json.loads(plan_bytes.decode("utf-8"))
    pool = json.loads(pool_bytes.decode("utf-8"))
    topology = {"topology_version": "t1",
                "segment_geometry": {
                    seg: geom
                    for c in pool["carriers"].values()
                    for seg, geom in (c.get("segment_polygons") or {}).items()},
                "adjacency": []}       # DISCLOSED: t1 defines no adjacency
    sg = B.build_segment_table(plan, topology)
    reg = json.loads(rd("//192.168.50.1/s4t/geospec_d2_campaign_v2/"
                        "d2_campaign_v2_20260811/registry_candidate.json")
                     .decode("utf-8"))
    caps = {}
    for carrier, ent in reg.items():
        caps[carrier] = {"expected_sha256": ent["expected_sha256"],
                         "valid_through": None, "threshold": None}
    ca = B.build_carrier_table(plan, topology, caps)
    mo = B.build_member_of_edges(st)
    adj = B.build_adjacent_to_edges(topology, sg)
    cont = B.build_contains_edges(sg)
    near = B.build_near_edges(st, carrier_crs=CARRIER_CRS, k=3)
    excluded = sorted(r["station_id"] for r in st
                      if not r.get("coordinates_available", True))
    checks["identity_110_35"] = (len(st) == 110
                                 and sum(r["registry_selected"]
                                         for r in st) == 35)
    checks["adjacency_empty_disclosed"] = adj == []
    w("tables/station.jsonl", B.canon_jsonl(st))
    w("tables/segment.jsonl", B.canon_jsonl(sg))
    w("tables/carrier.jsonl", B.canon_jsonl(ca))
    w("tables/member_of.jsonl", B.canon_jsonl(mo))
    w("tables/adjacent_to.jsonl", B.canon_jsonl(adj))
    w("tables/contains.jsonl", B.canon_jsonl(cont))
    w("tables/near.jsonl", B.canon_jsonl(near))
    print(f"identity tables done ({time.time()-t0:.0f}s)", flush=True)

    # per-day snapshots + per-carrier deltas from VERIFIED matrices
    # (closure 2: typed state over the FULL selected registry per carrier)
    import numpy as np
    root_im = json.loads(rd(os.path.join(SCRATCH, "consume_root",
                                         "input_manifest.json"))
                         .decode("utf-8"))
    uni_by_day = {}
    for o in root_im["objects"]:
        rec = dict(o)
        rec["station_id"] = ".".join(str(o.get("source_id", "")).split(".")[:2])
        uni_by_day.setdefault((o["carrier_key"], o["scored_day"]),
                              []).append(rec)
    selected_by_carrier = {c: [r_["station_id"] for r_ in rows]
                           for c, rows in plan["station_registry"].items()}
    state_hist = {}
    edges_by = {}
    snaps = {}
    ncmp = {"comparable": 0, "not_comparable": 0}
    for carrier, days in sorted(days_by_carrier.items()):
        for day in days:
            man = json.loads(rd(os.path.join(
                VERIFIED, f"{carrier}__{day}.manifest.json")).decode("utf-8"))
            r = np.load(os.path.join(VERIFIED,
                                     f"{carrier}__{day}.matrix.npy"))
            edges = B.build_coherence_edges(r, man, st)
            states = B.build_snapshot_states(
                man, selected_by_carrier[carrier],
                uni_by_day.get((carrier, day), []))
            for v in states.values():
                k2 = v.split(":")[0]
                state_hist[k2] = state_hist.get(k2, 0) + 1
            snap = B.build_snapshot(man["campaign_id"], carrier, day, edges,
                                    states,
                                    man["station_index_digest"])
            snaps[(carrier, day)] = snap
            edges_by[(carrier, day)] = edges
            w(f"snapshots/{carrier}/{day}.json", B.canon_bytes(snap))
        deltas = []
        for prev, curr in zip(days, days[1:]):
            rows = B.build_delta_table(snaps[(carrier, prev)],
                                       snaps[(carrier, curr)], days)
            deltas.extend(rows)
            for row in rows:
                ncmp["comparable" if row["comparable"]
                     else "not_comparable"] += 1
        w(f"deltas/{carrier}.jsonl", B.canon_jsonl(deltas))
    checks["snapshots_330"] = len(snaps) == 330
    print(f"snapshots+deltas done ({time.time()-t0:.0f}s) cmp={ncmp}",
          flush=True)

    # round trips on the REAL tables
    node_tables = {"station": st, "segment": sg, "carrier": ca}
    sample_day = ("istanbul_marmara",
                  days_by_carrier["istanbul_marmara"][0])
    edge_tables = {"coheres_with": edges_by[sample_day], "near": near,
                   "member_of": mo, "adjacent_to": adj, "contains": cont}
    key = lambda r_: B.canon_bytes(r_)                             # noqa: E731
    G = B.to_networkx(node_tables, edge_tables)
    nt2, et2 = B.from_networkx(G)
    checks["nx_round_trip"] = all(
        B.canon_jsonl(sorted(nt2[k], key=key))
        == B.canon_jsonl(sorted(node_tables[k], key=key))
        for k in node_tables) and all(
        B.canon_jsonl(sorted(et2.get(k2, []), key=key))
        == B.canon_jsonl(sorted(v, key=key))
        for k2, v in edge_tables.items() if v)
    gdfs = B.to_geodataframes(st, near)
    back = B.from_geodataframe(gdfs["station"])
    checks["gdf_round_trip"] = B.canon_jsonl(
        sorted(back, key=lambda r_: r_["station_id"])) == B.canon_jsonl(st)
    data = B.to_pyg(node_tables, edge_tables)
    nb, eb = B.from_pyg(data)
    def _etbl(k2):
        for kk in eb:
            if (kk[1] if isinstance(kk, tuple) else kk) == k2:
                return B.from_geodataframe(eb[kk])
        return None
    checks["pyg_sidecar_round_trip"] = all(
        B.canon_jsonl(sorted(B.from_geodataframe(nb[k]), key=key))
        == B.canon_jsonl(sorted(node_tables[k], key=key))
        for k in node_tables) and all(
        (lambda got: got is not None and B.canon_jsonl(sorted(got, key=key))
         == B.canon_jsonl(sorted(v, key=key)))(_etbl(k2))
        for k2, v in edge_tables.items() if v)
    adapter = data.f2g_adapter
    # closure 3: regenerate from the REOPENED canonical tables; sidecar digests
    # must be identical (canonical column order), and the ledger is a bound
    # output of the artifact
    def _reload_jsonl(rel):
        return [json.loads(x) for x in
                rd(os.path.join(OUT, rel)).decode("utf-8").splitlines()]
    nt_re = {"station": _reload_jsonl("tables/station.jsonl"),
             "segment": _reload_jsonl("tables/segment.jsonl"),
             "carrier": _reload_jsonl("tables/carrier.jsonl")}
    et_re = {"coheres_with": edge_tables["coheres_with"],
             "near": _reload_jsonl("tables/near.jsonl"),
             "member_of": _reload_jsonl("tables/member_of.jsonl"),
             "adjacent_to": _reload_jsonl("tables/adjacent_to.jsonl"),
             "contains": _reload_jsonl("tables/contains.jsonl")}
    data_re = B.to_pyg(nt_re, et_re)
    checks["sidecar_digests_reproducible"] = (
        data_re.f2g_adapter["sidecar_sha256s"]
        == adapter["sidecar_sha256s"])
    ledger = {"schema": "f2g-pyg-sidecar-ledger-v1",
              "adapter": adapter,
              "sidecars": {k: data[k].f2g_sidecar
                           for k in ("station", "segment", "carrier")}}
    for name in ("coheres_with", "near", "member_of", "adjacent_to",
                 "contains"):
        for kk in data.edge_types:
            if kk[1] == name:
                ledger["sidecars"][name] = data[kk].f2g_sidecar
    w("pyg_sidecar_ledger.json", B.canon_bytes(ledger))
    print(f"round trips done ({time.time()-t0:.0f}s)", flush=True)

    # renders (one per carrier, latest day)
    renders = {}
    os.makedirs(os.path.join(OUT, "render"), exist_ok=True)
    for carrier, days in sorted(days_by_carrier.items()):
        day = days[-1]
        st_c = [r_ for r_ in st if r_["carrier_key"] == carrier]
        out = B.render_map(st_c, edges_by[(carrier, day)],
                           os.path.join(OUT, f"render/{carrier}_{day}.png"),
                           title_suffix=f"{carrier} {day}")
        renders[carrier] = {"path": f"render/{carrier}_{day}.png",
                            "geometry_excluded_station_ids":
                                out["geometry_excluded_station_ids"],
                            "geometry_excluded_count":
                                out["geometry_excluded_count"]}
    checks["render_exclusions_carrier_local"] = (
        renders["turkey_kahramanmaras"]["geometry_excluded_station_ids"]
        == ["KO.KHMN"]
        and renders["istanbul_marmara"]["geometry_excluded_station_ids"] == []
        and renders["socal_coachella"]["geometry_excluded_station_ids"] == [])
    print(f"renders done ({time.time()-t0:.0f}s)", flush=True)

    # disposition counts from the ROOT manifest universe (the authority --
    # slim result records deliberately do not carry the field)
    disp = {}
    root_im = json.loads(rd(os.path.join(SCRATCH, "consume_root",
                                         "input_manifest.json"))
                         .decode("utf-8"))
    consumed = {(c, d) for c, days in days_by_carrier.items() for d in days}
    for o in root_im["objects"]:
        if (o["carrier_key"], o["scored_day"]) in consumed:
            d = o.get("reuse_disposition", "ABSENT")
            disp[d] = disp.get(d, 0) + 1

    # closure 4: the execution frame is PACKET-BOUND -- both driver sources
    # and the FULL transitive graph-environment lock ride the artifact
    w("code/f2g_consume_stage1.py",
      rd(os.path.join(SCRATCH, "f2g_consume_stage1.py")))
    w("code/f2g_consume_stage2.py", rd(os.path.abspath(__file__)))
    import platform as _pl
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    w("graph_environment_lock.json", B.canon_bytes(
        {"schema": "f2g-graph-environment-lock-v1",
         "python": _pl.python_version(), "platform": _pl.platform(),
         "machine": _pl.machine(),
         "pip_freeze": sorted(freeze.splitlines())}))

    geospec_head = subprocess.run(
        ["git", "-C", "C:/geospec", "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    out_digests = {}
    for dirpath, _dn, files in os.walk(OUT):
        for f in files:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, OUT).replace(os.sep, "/")
            if rel != "phase_a_result.json":
                out_digests[rel] = sha(rd(p))
    result = B.phase_a_result(
        input_digests={
            "packet_summary": PACKET_SUMMARY_SHA,
            "campaign_plan_bytes": sha(plan_bytes),
            "candidate_pool_bytes": sha(pool_bytes),
            "producer_environment_lock":
                ok_recs[0]["producer_environment_lock_digest"],
            "consumer_environment_lock": sha(B.canon_bytes(
                ok_recs[0]["consumer_environment_lock"])),
        },
        code_digests={
            "geospec_commit": geospec_head,
            "d2_f2g_graph_builder.py":
                sha(rd("C:/geospec/monitoring/src/d2_f2g_graph_builder.py")),
            "stage1_driver": "code/f2g_consume_stage1.py (in output_digests)",
            "stage2_driver": "code/f2g_consume_stage2.py (in output_digests)",
            "graph_environment_lock":
                "graph_environment_lock.json (in output_digests)",
        },
        output_digests=out_digests,
        bar_results={"checks": checks,
                     "stage1_days_ok": len(ok_recs),
                     "stage1_max_abs_delta": max_delta,
                     "consumer_profile": "cross_host_consumer_v1",
                     "delta_rows": ncmp,
                     "snapshot_state_histogram": state_hist,
                     "renders": renders,
                     "carrier_metric_crs": CARRIER_CRS,
                     "adjacency_disclosure":
                         "t1 registries define no adjacency relation; "
                         "adjacent_to honestly empty per A3",
                     "stage_split_disclosure":
                         "stage1 = recompute boundary under mirrored numeric "
                         "lock (py3.11.9/numpy2.3.5/scipy1.17.1/obspy1.4.2); "
                         "stage2 = arithmetic-free graph construction in the "
                         "A5-pinned graph env (py3.12, city2graph@3892a086, "
                         "torch 2.13.0+cpu)",
                     "disposition_object_counts": disp},
        status="BUILDER_ARTIFACT_COMPLETE" if all(checks.values())
               else "CHECKS_FAILED",
        geometry_excluded_station_ids=excluded,
        pyg_adapter=adapter)
    w("phase_a_result.json", B.canon_bytes(result))
    print(json.dumps({"checks": checks, "max_delta": max_delta,
                      "disposition_counts": disp,
                      "status": result["status"]}, indent=1), flush=True)
    print(f"STAGE2 DONE ({time.time()-t0:.0f}s) -> {OUT}", flush=True)
    sys.exit(0 if all(checks.values()) else 1)


main()
