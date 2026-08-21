#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAL results ASSEMBLER (cayley) -- FIXTURE-ONLY, pinned producer of the
exact registered result schema f2g-phase-b-power-cal-results-v1 (codex
0307Z consolidated repair 1: the independent verifier reconstructs from the
committed evidence and compares against artifacts THIS assembler emits; one
schema shared by producer and verifier).

Every claim-bearing field is DERIVED from the evidence capsule rows:
tier tables from S1/S2 rows, Tier-C candidate outcomes + verdict/stopping
from the sequential C rows under the registered R=20/R=40 Clopper-Pearson
stopping semantics, certified_points = exactly the derived-CERTIFIED
candidates, terminal from that set, the complete coordinatewise
Pareto-minimal certified frontier + the registered lexicographic
representative, the equivalence receipt from the gate row, and the panel
reopen sample (rep 0 of every registered grid point + every Stage-C row).
Usage: assembler.py <evidence_jsonl> <evidence_commit> <docs_dir>
(<evidence_commit> = the geospec commit whose blob of the canonical path is
the capsule authority; the artifact binds commit+path+blob sha.)
"""
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict

import numpy as np

import f2g_phase_b_power_estimation_cal_cayley as PD

SCHEMA = "f2g-phase-b-power-cal-results-v1"
CANONICAL_EVIDENCE_PATH = "docs/f2g_phase_b_power_cal_evidence.jsonl"
# authority pins recomputed from the repo's pinned blobs (never the gate row)
PIN_REFS = {
    "frozen_amendment2_sha256": ("337571c",
                                 "docs/f2g_phase_b_prereg_amendment2_DRAFT.md"),
    "engine_lf_sha256": ("24b0d8f", "monitoring/src/d2_f2g_phase_b_stats.py"),
    "admitted_bar_lf_sha256": (
        "89673dc", "monitoring/src/test_f2g_phase_b_stats_redkats_grassmann.py"),
    "driver_lf_sha256": (
        "975638d", "monitoring/src/f2g_phase_b_power_estimation_cal_cayley.py"),
    "annex_common_rev16_sha256": ("feb20bb",
                                  "docs/f2g_phase_b_power_annex_common.md"),
    "annex_b1a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b1a.md"),
    "annex_b2a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b2a.md"),
    "annex_b3a_sha256": ("feb20bb", "docs/f2g_phase_b_power_annex_b3a.md"),
}


def binom_sf_geq(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def cp_lower(k, n, conf=0.95):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf_geq(k, n, mid) > 1 - conf:
            hi = mid
        else:
            lo = mid
    return lo


def cp_upper(k, n, conf=0.95):
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf_geq(k + 1, n, mid) < conf:
            lo = mid
        else:
            hi = mid
    return hi


def derive_candidate(rows_c):
    """Sequential C reps -> (n_stop, successes, lb, ub, verdict, stopping)
    under the registered stopping semantics; refuses non-sequential reps."""
    reps = sorted(r["rep"] for r in rows_c)
    if reps != list(range(len(reps))):
        raise ValueError(f"C reps not sequential: {reps}")
    succ = 0
    for i, r in enumerate(sorted(rows_c, key=lambda x: x["rep"])):
        succ += bool(r["post"])
        n = i + 1
        if n == 20:
            if cp_lower(succ, 20) >= 0.80:
                return n, succ, "CERTIFIED"
            if cp_upper(succ, 20) < 0.80:
                return n, succ, "FAILED"
        if n == 40:
            if cp_lower(succ, 40) >= 0.80:
                return 40, succ, "CERTIFIED"
            if cp_upper(succ, 40) < 0.80:
                return 40, succ, "FAILED"
            return 40, succ, "CANNOT_DETERMINE_POWER_ESTIMATE"
    return len(reps), succ, None  # incomplete -- caller refuses


def pk(point):
    return json.dumps(point, sort_keys=True, separators=(",", ":"))


def assemble(evidence_path, evidence_commit, repo, docs_dir):
    raw = subprocess.check_output(
        ["git", "cat-file", "blob",
         f"{evidence_commit}:{CANONICAL_EVIDENCE_PATH}"], cwd=repo)
    disk = open(evidence_path, "rb").read().replace(b"\r\n", b"\n")
    if hashlib.sha256(disk).hexdigest() != \
            hashlib.sha256(raw).hexdigest():
        raise ValueError("EVIDENCE_NOT_THE_COMMITTED_BLOB")
    ev_sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines()
            if l.strip()]
    gate = next(r for r in rows if r.get("stage") == "gate")
    self_bytes = open(__file__, "rb").read().replace(b"\r\n", b"\n")
    assembler_sha = hashlib.sha256(self_bytes).hexdigest()
    for fam, fname in (("B1A", "b1a"), ("B2A", "b2a"), ("B3A", "b3a")):
        grid = PD.grid_of(fam)
        by_stage = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r.get("family") == fam and r.get("stage") in ("S1", "S2",
                                                             "C"):
                by_stage[r["stage"]][pk(r["point"])].append(r)
        s1_table = []
        for p in grid:
            v = by_stage["S1"].get(pk(p), [])
            reps = sorted(x["rep"] for x in v)
            if reps != list(range(50)):
                raise ValueError(f"{fam} S1 {pk(p)}: reps {len(reps)}!=50")
            s1_table.append({"point": p,
                             "pre_loco_recovery": sum(bool(x["pre"])
                                                      for x in v) / 50,
                             "n": 50})
        rank1 = sorted(s1_table,
                       key=lambda x: (-x["pre_loco_recovery"],
                                      pk(x["point"])))
        s2_points = [x["point"] for x in rank1[:min(8, len(grid))]]
        s2_table = []
        for p in s2_points:
            v = by_stage["S2"].get(pk(p), [])
            reps = sorted(x["rep"] for x in v)
            if reps != list(range(50)):
                raise ValueError(f"{fam} S2 {pk(p)}: reps {len(reps)}!=50")
            s2_table.append({"point": p,
                             "pre_loco_recovery": sum(bool(x["pre"])
                                                      for x in v) / 50,
                             "post_loco_recovery": sum(bool(x["post"])
                                                       for x in v) / 50,
                             "n": 50})
        rank2 = sorted(s2_table,
                       key=lambda x: (-x["post_loco_recovery"],
                                      pk(x["point"])))
        c_points = [x["point"] for x in rank2[:min(3, len(s2_table))]]
        candidates = []
        for p in c_points:
            v = by_stage["C"].get(pk(p), [])
            n, succ, verdict = derive_candidate(v)
            if verdict is None:
                raise ValueError(f"{fam} C {pk(p)}: incomplete stopping")
            candidates.append({"point": p,
                               "post_loco": {"successes": succ,
                                             "replicates": n},
                               "lb95": cp_lower(succ, n),
                               "ub95": cp_upper(succ, n),
                               "verdict": verdict,
                               "stopping": f"{verdict}@R={n}"})
        certified = [c for c in candidates if c["verdict"] == "CERTIFIED"]
        # coordinatewise Pareto-minimal frontier over certified points
        def dominates(a, b):
            ka, kb = a["point"], b["point"]
            keys = sorted(ka)
            return all(ka[k] <= kb[k] for k in keys) and \
                any(ka[k] < kb[k] for k in keys)
        pareto = [c for c in certified
                  if not any(dominates(o, c) for o in certified
                             if o is not c)]
        pareto = sorted(pareto, key=lambda c: pk(c["point"]))
        reopen = [{"family": fam, "point": p, "rep": 0,
                   "panel_sha256": PD.panel_digest(
                       PD.make_panel(fam, p, 0))} for p in grid]
        for p in c_points:
            for r in sorted(by_stage["C"][pk(p)], key=lambda x: x["rep"]):
                reopen.append({"family": fam, "point": p, "rep": r["rep"],
                               "panel_sha256": r["panel_sha256"]})
        out = {
            "schema": SCHEMA,
            "family": fam,
            "calendar_authority_mode": "bound",
            "calendar_authority_sha256": PD.E.CAL_AUTHORITY_SHA256,
            "digests": dict(
                {k: subprocess.check_output(
                    ["git", "cat-file", "blob", f"{c}:{p}"],
                    cwd=repo) and hashlib.sha256(
                        subprocess.check_output(
                            ["git", "cat-file", "blob", f"{c}:{p}"],
                            cwd=repo)).hexdigest()
                 for k, (c, p) in PIN_REFS.items()},
                assembler_lf_sha256=assembler_sha),
            "equivalence_receipt": {
                "full_equal": gate.get("full_equal"),
                "fold_equal_all": gate.get("fold_equal_all"),
                "folds_checked": gate.get("folds_checked"),
                "all_equal": gate.get("all_equal"),
                "engine_commit_bound": gate.get("engine_commit_bound")},
            "evidence_capsule": {"geospec_commit": evidence_commit,
                                 "path": CANONICAL_EVIDENCE_PATH,
                                 "git_blob_sha256": ev_sha,
                                 "rows": len(rows)},
            "tier_s1": {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                        "replicates": 50, "table": s1_table},
            "tier_s2": {"label": "PRELIMINARY_SMOKE", "n_draws": 999,
                        "replicates": 50, "table": s2_table},
            "tier_c": {"label": "CERTIFICATION", "n_draws": 9999,
                       "candidates": candidates},
            "certified_points": certified,
            "pareto_minimal_certified": pareto,
            "pareto_lex_representative": (pareto[0]["point"] if pareto
                                          else None),
            "terminal_type": ("CERTIFIED" if certified else
                              "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH"),
            "panel_reopen_sample": reopen,
            "induced_effect_report": {
                "note": "per-family induced-effect summaries ride the "
                        "registered corner replicate",
                "family": fam},
            "env": {"python": platform.python_version(),
                    "numpy": np.__version__},
        }
        path = f"{docs_dir}/f2g_phase_b_power_annex_{fname}_cal_results.json"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    assemble(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
