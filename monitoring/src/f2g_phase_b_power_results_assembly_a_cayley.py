#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the AMENDED-family power results artifacts (cayley) -- FIXTURE-ONLY.

Reads the completed amended-lane estimation checkpoint (driver run under the
FINAL ADMISSION PASS authorities) and emits
docs/f2g_phase_b_power_annex_{b1a,b2a,b3a}_results.json in the admitted bar's
pinned minimal schema. Label correction disclosed in-artifact: the driver's
gate receipt carried a stale hardcoded engine label ("b4a7eee"); the module
the gate actually imported and byte-matched was the admitted engine at
493c2b9 (the working tree at launch), and the digests block binds the true
authority.
Usage: python f2g_phase_b_power_results_assembly_a_cayley.py <ckpt> <docs_dir>
"""
import json
import platform
import sys
from collections import defaultdict

import numpy as np

import d2_f2g_phase_b_stats as E
import f2g_phase_b_power_estimation_cayley as D0
import f2g_phase_b_power_estimation_a_cayley as DA

DIGESTS = {
    "frozen_amendment_sha256": DA.AMENDMENT_SHA,
    "engine": "d2_f2g_phase_b_stats.py @ 493c2b9",
    "engine_lf_sha256": "96949d45463f76428d8e0dbab81715d2cc0c48e8164dde5502f87a8153dc5dad",
    "admitted_bar_lf_sha256": "295631fed6f836c18bf40975403db7799422f83dc74ea74cf14b8651fdcad2a6",
    "estimation_driver_commit": "e384e2c (audit deltas inert to estimation)",
    "annex_common_rev13_sha256": "b6352e914bf24b3c54663388daa9e71d15a637491c5fec4462abcd6785bc2e8d",
    "annex_b1a_sha256": "e7e08454ec5f4ba45dd3092797afe26cbb12727b02a6027a962cfb99fb946a9b",
    "annex_b2a_sha256": "1df422a477bd418bd26067b5afdd4211bbb1882f6eff5de5f640fd2d891ceaf6",
    "annex_b3a_sha256": "0c5fc14f6ecc4606b4a2548e788f962fd844ef0997ffb271ecf2e456d497f8d0",
}
DISCLOSURES = [
    ("the evidence capsule's launch-time gate row carries a stale hardcoded "
     "label ('b4a7eee'); it is superseded by the SOURCE-ATTESTED full gate "
     "receipt bound below (codex repair 2): full + ALL 35 LOCO folds proven "
     "exactly equal against admitted engine 493c2b9 post-run."),
    ("the admitted bar's _results_bound checker binds the SUPERSEDED-family "
     "authorities by design and correctly returns False on these amended "
     "artifacts; conformance for the amended lane is established by the "
     "committed strict verifier (codex repair 3), not by that checker."),
]
FULLGATE_PATH = "C:/geospec/docs/f2g_phase_b_power_a_fullgate_receipt.json"
EVIDENCE_PATH = "C:/geospec/docs/f2g_phase_b_power_a_evidence.jsonl"


def agg(rows, stage, family):
    byp = defaultdict(list)
    for r in rows:
        if r.get("stage") == stage and r.get("family") == family:
            byp[json.dumps(r["point"], sort_keys=True)].append(r)
    return byp


def induced_z_b1a():
    pt = {"delta_lat": 2.4, "k": 25, "n_e": 10}
    panel = DA.make_panel("B1A", pt, 0)
    memo = DA.B1AMemo(panel)
    W = memo.edge_window_max("c1", list(range(E.B1A_BLOCKS)))
    inj = set(DA.D0.edges_of("c1")[:10])
    idx = [i for i, e in enumerate(memo.edges["c1"]) if e in inj]
    zin = W[idx]
    zin = zin[np.isfinite(zin)]
    zo = np.concatenate([memo.edge_window_max(c, list(range(E.B1A_BLOCKS)))
                         for c in ("c2", "c3")])
    zo = zo[np.isfinite(zo)]
    return {"point": pt, "rep": 0,
            "injected_edge_window_max": {"median": float(np.median(zin)),
                                         "max": float(zin.max())},
            "noise_carrier_window_max_max": float(zo.max()),
            "note": ("per-edge window-max of mean|z| at the best-recovery "
                     "duration k=25; window averaging lifts signal above the "
                     "noise window ceiling, but the S_11 relocation capture "
                     "caps family power (Tier-C FAILED verdicts)")}


def main():
    ckpt, docs = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "machine": platform.machine()}
    import hashlib
    import subprocess
    fg = json.loads(open(FULLGATE_PATH, encoding="utf-8").read())
    fg_sha = hashlib.sha256(open(FULLGATE_PATH, "rb").read()).hexdigest()
    # codex 22:32Z repair 1: the evidence authority is the COMMITTED GIT BLOB
    # (LF-normalized), never the mutable checkout -- the driver's text-mode
    # appends leave CRLF on disk and a checkout hash matches no committed
    # object.
    ev_blob = subprocess.check_output(
        ["git", "cat-file", "blob",
         "HEAD:docs/f2g_phase_b_power_a_evidence.jsonl"], cwd=r"C:\geospec")
    ev_sha = hashlib.sha256(ev_blob).hexdigest()
    ev_rows = sum(1 for l in ev_blob.decode("utf-8").splitlines()
                  if l.strip())
    receipt = {"full_equal": bool(fg["full_equal"]),
               "fold_equal_all": bool(fg["fold_equal_all"]),
               "folds_checked": int(fg["folds_checked"]),
               "all_equal": bool(fg["all_equal"]),
               "engine_commit_bound": fg["engine_commit"],
               "engine_disk_sha256": fg["engine_disk_sha256"],
               "driver_commit_bound": fg["driver_commit"],
               "driver_disk_sha256": fg["driver_disk_sha256"],
               "receipt_path": "docs/f2g_phase_b_power_a_fullgate_receipt.json",
               "receipt_sha256": fg_sha}
    evidence = {"path": "docs/f2g_phase_b_power_a_evidence.jsonl",
                "git_blob_sha256": ev_sha, "rows": ev_rows,
                "authority": "the committed LF git blob (git cat-file), "
                             "never the checkout representation",
                "note": "immutable checkpoint capsule: every Tier-S/S2/C "
                        "replicate row incl. all Tier-C full+post-LOCO "
                        "outcomes and the launch gate row"}
    for fam, fname in (("B1A", "b1a"), ("B2A", "b2a"), ("B3A", "b3a")):
        s1 = agg(rows, "S1", fam)
        s2 = agg(rows, "S2", fam)
        cvs = [r for r in rows
               if r.get("stage") == "Cverdict" and r.get("family") == fam]
        certified = [
            {"point": r["point"],
             "post_loco": {"successes": r["successes"], "replicates": r["n"]},
             "lb95": r["lb95"], "ub95": r["ub95"]}
            for r in cvs if r["verdict"] == "CERTIFIED"]
        if fam == "B2A" and certified:
            pareto = [min(certified, key=lambda c: c["point"]["m"])]
        else:
            pareto = []  # B1A/B3A: nothing certified
        out = {
            "schema": "f2g-phase-b-power-results-v1",
            "family": fam,
            "tier_s1": {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                        "replicates": D0.TIER_S_R,
                        "table": [{"point": json.loads(pk),
                                   "pre_loco_recovery": sum(bool(x["pre"])
                                                            for x in v) / len(v),
                                   "n": len(v)}
                                  for pk, v in sorted(s1.items())]},
            "tier_s2": {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                        "replicates": D0.TIER_S_R,
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
                            "lb95": r["lb95"], "ub95": r["ub95"],
                            "verdict": r["verdict"],
                            "stopping": f"{r['verdict']}@R={r['n']}"}
                           for r in cvs]},
            "certified_points": certified,
            "pareto_minimal_certified": pareto,
            "terminal_type": ("CERTIFIED" if certified else
                              "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH"),
            "equivalence_receipt": receipt,
            "evidence_capsule": evidence,
            "disclosures": DISCLOSURES,
            "digests": DIGESTS,
            "env": env,
        }
        if fam == "B1A":
            out["induced_z_report"] = induced_z_b1a()
        path = f"{docs}/f2g_phase_b_power_annex_{fname}_results.json"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
