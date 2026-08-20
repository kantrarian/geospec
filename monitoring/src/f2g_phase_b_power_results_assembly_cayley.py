#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the Phase-B power-annex results artifacts (cayley) -- FIXTURE-ONLY.

Reads the completed estimation checkpoint (driver v1 run d4edfb2 + v1.1
corrected lemma stage) and emits docs/f2g_phase_b_power_annex_{b1,b2,b3}_
results.json in bar AMENDMENT 1's pinned minimal schema. All binomial bounds
are RECOMPUTED with the v1.1-corrected cp_upper (the v1 tail defect and the
lemma fixture-scope defect are disclosed in-artifact; no Tier-C verdict was
affected). The B-1 induced robust-z report is recomputed live from the
registered generator, not transcribed.
Usage: python f2g_phase_b_power_results_assembly_cayley.py <ckpt> <docs_dir>
"""
import json
import sys
import platform
from collections import defaultdict

import numpy as np

import d2_f2g_phase_b_stats as E
import f2g_phase_b_power_estimation_cayley as D

# FLAT name -> value map (bar AMENDMENT 1 reads digests.values() as a set)
DIGESTS = {
    "frozen_doc_sha256": D.FROZEN_DOC_SHA,
    "engine": "d2_f2g_phase_b_stats.py @ 6034419 (+ G16b certified-flag acceptance)",
    "estimation_run_driver_commit": "d4edfb2",
    "bounds_and_lemma_rev_driver_commit": "951eee9",
    "bar_amendment1_blob_sha256": "06a94e64b8eedb05a5e4874792ca06754c5ac465733d0b9bc7fa0965866429ec",
    "annex_common_rev11_sha256": "baddf2aa259689356d4d942b6282824ad6ad6d7f50075c54c992a997f216f20d",
    "annex_b1_rev11_sha256": "fb2883f5a9be84f2197b79291b116a799a3a2c5cc86afc69cc34fea064a4e14a",
    "annex_b2_rev11_sha256": "dca9dede6b91e35e6dd55ed6104dd3b4c29d2f8dfc8b6ef615187df91b5cb05c",
    "annex_b3_rev11_sha256": "9414bd597134ebbcac8dc2d199a111521a23b0d59af6bc40ab0e803cd3495dc2",
}
DEFECTS = [
    ("cp_upper wrong tail (driver v1): reported one-sided 95% upper bounds "
     "solved P(X<=k)=0.95 instead of 0.05 (0/20 -> 0.003 vs correct 0.139). "
     "All bounds in this artifact are recomputed with the corrected rule; no "
     "Tier-C verdict changes (all candidate counts were 0/20; correct UB "
     "0.139 < 0.80 so every FAILED verdict stands)."),
    ("B-2 sec-L lemma check fixture scope (driver v1): the 3-carrier panel "
     "let noise-carrier partition churn into the family max (p 0.55-0.99). "
     "The lemma's hypothesis is per-carrier; the corrected single-carrier "
     "exact check (stage B2lemma_rev2) returns p == 1.0 in 10/10 including "
     "the 9,999-draw run. v1 rows are retained in the checkpoint as the "
     "defect record."),
]


def agg(rows, stage, family):
    byp = defaultdict(list)
    for r in rows:
        if r.get("stage") == stage and r.get("family") == family:
            byp[json.dumps(r["point"], sort_keys=True)].append(r)
    return byp


def induced_z_report():
    pt = {"delta_lat": 2.4, "k": 50, "n_e": 33}
    panel = D.make_panel("B1", pt, 0)
    loaded = {k: E._load_carrier(c) for k, c in panel["carriers"].items()}
    T, _p, _e, _n, zmap = E._b1_stat(loaded)
    z1, rows1 = zmap["c1"]
    edges1 = loaded["c1"]["edges"]
    inj = set(D.edges_of("c1")[:33])
    im = [i for i, ei in enumerate(rows1) if edges1[ei] in inj]
    nm = [i for i, ei in enumerate(rows1) if edges1[ei] not in inj]
    zin = np.abs(z1[im]); zin = zin[np.isfinite(zin)]
    zn = np.abs(z1[nm]); zn = zn[np.isfinite(zn)]
    zo = np.concatenate([np.abs(zmap[c][0][np.isfinite(zmap[c][0])])
                         for c in ("c2", "c3")])
    return {"point": pt, "rep": 0,
            "injected_abs_z": {"median": float(np.median(zin)),
                               "max": float(zin.max())},
            "c1_noise_abs_z_max": float(zn.max()),
            "other_carriers_noise_abs_z_max": float(zo.max()),
            "family_T_obs": float(T),
            "note": ("saturation-level injection (r -> ~0.99) cannot exceed "
                     "the panel noise maximum at the registered noise scale; "
                     "the bounded correlation domain caps injected |z|")}


def main():
    ckpt, docs = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    gate = next(r for r in rows if r.get("stage") == "gate")
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "machine": platform.machine()}
    for fam, fname in (("B1", "b1"), ("B2", "b2"), ("B3", "b3")):
        s1 = agg(rows, "S1", fam)
        s2 = agg(rows, "S2", fam)
        cvs = [r for r in rows
               if r.get("stage") == "Cverdict" and r.get("family") == fam]
        crows = agg(rows, "C", fam)
        out = {
            "schema": "f2g-phase-b-power-results-v1",
            "family": fam,
            "tier_s1": {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                        "replicates": D.TIER_S_R,
                        "table": [{"point": json.loads(pk),
                                   "pre_loco_recovery": sum(bool(x["pre"])
                                                            for x in v) / len(v),
                                   "n": len(v)}
                                  for pk, v in sorted(s1.items())]},
            "tier_s2": {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                        "replicates": D.TIER_S_R,
                        "table": [{"point": json.loads(pk),
                                   "pre_loco_recovery": sum(bool(x["pre"])
                                                            for x in v) / len(v),
                                   "post_loco_recovery": sum(bool(x["post"])
                                                             for x in v) / len(v),
                                   "n": len(v)}
                                  for pk, v in sorted(s2.items())]},
            "tier_c": {"label": "CERTIFICATION", "n_draws": 9999,
                       "candidates": [
                           {"point": r["point"],
                            "post_loco": {"successes": r["successes"],
                                          "replicates": r["n"]},
                            "lb95": D.cp_lower(r["successes"], r["n"]),
                            "ub95": D.cp_upper(r["successes"], r["n"]),
                            "verdict": r["verdict"],
                            "stopping": f"{r['verdict']}@R={r['n']}"}
                           for r in cvs],
                       "replicate_rows": sum(len(v) for v in crows.values())},
            "certified_points": [],
            "pareto_minimal_certified": [],
            "terminal_type": "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH",
            "equivalence_receipt": {k: v for k, v in gate.items()
                                    if k not in ("key", "stage")},
            "defect_disclosures": DEFECTS,
            "digests": DIGESTS,
            "env": env,
        }
        if fam == "B1":
            out["induced_z_report"] = induced_z_report()
        if fam == "B2":
            out["sec_l_lemma"] = {
                "corrected_single_carrier": [
                    {"rep": r["rep"], "n_draws": r["n_draws"],
                     "p_value": r["p"], "max_switches": r["max_switches"]}
                    for r in rows if r.get("stage") == "B2lemma_rev2"],
                "v1_wrong_scope_rows_retained": [
                    {"rep": r["rep"], "n_draws": r["n_draws"],
                     "p_value": r["p"]}
                    for r in rows if r.get("stage") == "B2lemma"],
                "conclusion": ("per-carrier two-regime class: p == 1.0 "
                               "identically (proof + 10/10 executable check "
                               "incl 9,999-draw exact run)")}
        path = f"{docs}/f2g_phase_b_power_annex_{fname}_results.json"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
